"""Portable opt-in wrapper around the experiment's unchanged Jev adapter/runner.

This release wrapper replaces the historical deadline/account launchers. Planning
and freezing are offline. A paid run requires explicit USD, request and runtime
ceilings, and never overwrites/resumes an attempt or retries a failed request.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import freeze_protocol
import rerank
from openrouter_adapter import Budget, OpenRouterAdapter, build_payload, decimal_amount, parse_response, validate_config

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/openrouter.json"
ADDONS = ("openrouter_adapter.py", "openrouter_run.py", "config/openrouter.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def effective():
    value = rerank.read_json(CONFIG)
    validate_config({**value, "max_requests": 0})
    require(value["model"] == "typesafe/jev-1.13"
            and value["expected_response_model"] == "typesafe/jev-1.13-20260917"
            and value["concurrency"] == 4 and value["batch_size"] == 20,
            "This reproduction requires the original pinned model and concurrency")
    return {"model": "jev-openrouter", "model_config": value, "batch_size": 20,
            "warmup_lists": 2, "repeats": 1, "status": "separate_openrouter_provider",
            "implementation_sha256": {name: rerank.sha(ROOT / name) for name in ADDONS}}


def plan(source, split):
    bundle = rerank.Candidates(source, split)
    settings = effective()
    return {**bundle.plan(), "config": settings,
            "pair_calls_including_warmup": len(rerank.job_sequence(bundle, 2, 1)) * 20,
            "paid_calls": False, "exact_local_token_count_available": False}


def tariff_check(session=None):
    """Public read-only metadata check; never reads account credits or sends a key.

    The adapter's per-request reservation relies on precisely this tariff/context
    ceiling. Fail closed if the pinned historical provider is no longer offered.
    """
    import requests
    owned = session is None
    session = session or requests.Session()
    session.trust_env = False
    try:
        response = session.get("https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints",
                               timeout=25, allow_redirects=False)
        try:
            require(response.status_code == 200, "Cannot verify current provider tariff")
            model = response.json()["data"]
        finally:
            response.close()
        endpoints = model["endpoints"]
        require(model["id"] == "typesafe/jev-1.13" and len(endpoints) == 1,
                "Pinned model/provider set changed")
        endpoint = endpoints[0]
        require(endpoint["name"] == "TypeSafe | typesafe/jev-1.13-20260917"
                and endpoint["provider_name"] == "TypeSafe" and endpoint["context_length"] == 32000,
                "Pinned provider/model/context changed")
        pricing = endpoint["pricing"]
        require(Decimal(str(pricing["prompt"])) == Decimal("0.000000042")
                and Decimal(str(pricing["completion"])) == 0
                and Decimal(str(pricing.get("request", 0))) == 0,
                "Tariff changed; the existing reservation is not authorized")
        return {"checked_at": datetime.now(timezone.utc).isoformat(), "model": model}
    finally:
        if owned:
            session.close()


def run(source, split, out, ledger, budget_usd, max_api_requests, max_runtime_seconds,
        *, allow_paid_api=False, protocol=None):
    require(allow_paid_api is True, "Paid run requires --allow-paid-api")
    limit = decimal_amount(budget_usd)
    require(0 < limit <= Decimal("3.90"), "Explicit budget must be positive and at most $3.90")
    require(type(max_api_requests) is int and max_api_requests > 0,
            "An explicit positive --max-api-requests is required")
    require(not isinstance(max_runtime_seconds, bool)
            and isinstance(max_runtime_seconds, (int, float))
            and math.isfinite(max_runtime_seconds) and max_runtime_seconds > 0,
            "An explicit positive finite --max-runtime-seconds is required")
    out, ledger = Path(out), Path(ledger)
    require(not out.exists(), "Never overwrite or automatically repeat/resume an attempt")
    require(not ledger.resolve().is_relative_to(out.resolve()),
            "Use one shared budget ledger outside the run directory for pilot and test")
    bundle, settings = rerank.Candidates(source, split), effective()
    needed = len(rerank.job_sequence(bundle, 2, 1)) * 20
    require(max_api_requests == needed, f"This complete phase requires exactly {needed} request slots")
    if split == "test":
        selected = freeze_protocol.verify_protocol(protocol, bundle, settings)
        require(decimal_amount(selected["adapter"]["budget_ceiling_usd"]) == limit
                and selected["adapter"]["budget_ledger"] == str(ledger.resolve()),
                "Test must use the pilot's shared ledger and unchanged budget")
        require(ledger.is_file(), "Pilot budget ledger is missing; never reset the budget")
    else:
        require(protocol is None, "Protocols authorize test only")
    key = os.environ.get(settings["model_config"]["api_key_env"])
    require(bool(key) and key == key.strip() and "\n" not in key and "\r" not in key,
            "API key must be supplied only through the configured environment variable")
    with_budget = Budget(ledger, limit_usd=str(limit))
    try:
        tariff = tariff_check()
        def factory(callback, remaining):
            extra = out / "additional-source"
            extra.mkdir()
            for name in ADDONS:
                dest = extra / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, dest)
            rerank.write_json(out / "tariff-check.json", tariff)
            return OpenRouterAdapter({**settings["model_config"], "max_requests": remaining},
                                     event_callback=callback, budget=with_budget)
        return rerank.execute(bundle, out, settings, factory, api_limit=max_api_requests,
                              max_runtime_seconds=max_runtime_seconds, protocol=protocol)
    finally:
        try:
            if out.is_dir():
                rerank.write_json(out / "budget-status.json", with_budget.snapshot())
        finally:
            with_budget.close()


def review_pilot(path, source):
    """Verify all responses and complete inputs; never read relevance labels."""
    path = Path(path)
    settings, bundle = effective(), rerank.Candidates(source, "pilot")
    manifest = rerank.read_json(path / "manifest.json")
    identity = manifest["identity"]
    require(identity["split"] == "pilot" and identity["config"] == settings
            and identity["source"] == bundle.plan()
            and identity["code_sha256"] == rerank.source_hashes(), "Pilot identity changed")
    summary = rerank.read_json(path / "summary.json")
    require(summary.get("complete") is True and summary.get("split") == "pilot"
            and summary.get("test_quality_computed") is False, "Pilot is incomplete or used test quality")
    for field, name in (("events_sha256", "events.jsonl"), ("results_sha256", "lists.jsonl")):
        require(summary[field] == rerank.sha(path / name), "Pilot artifacts changed")
    for name in ADDONS:
        require(rerank.sha(path / "additional-source" / name) == settings["implementation_sha256"][name],
                "OpenRouter implementation changed after pilot")
    jobs = {key: (phase, item) for key, phase, item in rerank.job_sequence(bundle, 2, 1)}
    results, events = rerank.rows(path / "lists.jsonl"), rerank.rows(path / "events.jsonl")
    require(len(results) == len(jobs) and {r["job_id"] for r in results} == set(jobs),
            "Pilot has duplicate, missing or extra lists")
    require(not any(e["event"] == "list_error" for e in events)
            and sum(e["event"] == "session_start" for e in events) == 1,
            "Pilot contains errors or restarted sessions; review separately")
    successes = {(e["job_id"], e["attempt"]) for e in events if e["event"] == "list_success"}
    by_list = {row["job_id"]: row for row in results}
    for row in results:
        phase, item = jobs[row["job_id"]]
        require(row["phase"] == phase and row["attempt"] == 1
                and (row["job_id"], 1) in successes
                and (row["dataset"], row["method"], row["query_id"]) ==
                    (item["dataset"], item["method"], item["query_id"])
                and row["ranking"] == rerank.sort_results(item["retrieval"], row["raw_results"]),
                "Pilot ranking differs from full input/results")
    starts = [e for e in events if e["event"] == "api" and e["detail"]["phase"] == "start"]
    finishes = [e for e in events if e["event"] == "api" and e["detail"]["phase"] == "finish"]
    maximum = len(jobs) * 20
    require(len(starts) == len(finishes) == maximum, "Incomplete API journal")
    for sequence in (starts, finishes):
        require(sorted(e["detail"]["request_index"] for e in sequence) == list(range(1, maximum + 1)),
                "Duplicate or missing API request indices")
    by_start = {e["detail"]["request_index"]: e for e in starts}
    for job in jobs:
        require(sorted(e["detail"]["pair_index"] for e in starts if e["job_id"] == job) == list(range(20)),
                "Incomplete pair journal")
    for event in finishes:
        detail = event["detail"]
        start = by_start[detail["request_index"]]
        first = start["detail"]
        require(event["job_id"] == start["job_id"] and detail["pair_index"] == first["pair_index"],
                "API journal pair mismatch")
        pair = jobs[event["job_id"]][1]["pairs"][first["pair_index"]]
        expected = build_payload(pair, settings["model_config"]["model"])
        encoded = json.dumps(expected, ensure_ascii=False, allow_nan=False).encode("utf-8")
        require(first["request"] == expected
                and first["request_sha256"] == hashlib.sha256(encoded).hexdigest() == detail["request_sha256"],
                "Full request changed")
        require(detail["status"] == "ok" and detail["http_status"] == 200, "Unsuccessful API response")
        raw = detail["response"]
        require(json.loads(detail["response_text"]) == raw, "Raw API response changed")
        score, tokens = parse_response(raw, settings["model_config"]["expected_response_model"])
        saved = by_list[event["job_id"]]["raw_results"][first["pair_index"]]
        require(saved["score"] == score and saved["raw"] == raw and saved["input_tokens"] == tokens,
                "Saved score differs from raw API result")
    require(summary["lists"] == len(bundle.lists) and summary["pair_occurrences"] == len(bundle.lists) * 20,
            "Pilot summary count changed")
    return {"config": settings, "environment": identity["environment"],
            "adapter": rerank.read_json(path / "adapter.json"), "pilot_path": str(path.resolve()),
            "pilot_created_at": manifest["created_at"],
            "pilot_artifacts_sha256": {name: rerank.sha(path / name) for name in
                ("manifest.json", "summary.json", "events.jsonl", "lists.jsonl", "adapter.json")},
            "historical_list_errors": 0, "historical_sessions": 1,
            "full_inputs_verified": True, "requests_verified": maximum}


def freeze(pilot, source, base_protocol, out, *, reviewed=False):
    require(reviewed is True, "Review pilot outputs/errors and confirm --reviewed-pilot")
    out = Path(out)
    require(not out.exists(), "Never overwrite a frozen protocol")
    require(Path(pilot).is_dir(), "Pilot directory does not exist")
    base = rerank.read_json(base_protocol)
    prior = base.pop("protocol_sha256")
    require(prior == rerank.fingerprint(base), "Base protocol checksum changed")
    require(base.get("schema") == freeze_protocol.SCHEMA
            and base.get("status") == "frozen_after_reviewed_pilots"
            and base.get("reviewed_pilots") is True and base.get("test_metrics_used_for_tuning") is False
            and freeze_protocol.GPU_MODELS <= base["models"].keys(), "Require reviewed four-GPU pilot protocol")
    require(base["runner_code_sha256"] == rerank.source_hashes()
            and base["inference_code_sha256"] == freeze_protocol.inference_hashes(), "Base protocol code changed")
    require(base["pilot_source"] == rerank.Candidates(source, "pilot").plan()
            and base["test_source"] == rerank.Candidates(source, "test").plan(), "Base candidate source changed")
    require("jev-openrouter" not in base["models"], "Base already contains an OpenRouter pilot")
    with rerank.exclusive_run(Path(pilot)):
        evidence = review_pilot(pilot, source)
    base["models"]["jev-openrouter"] = evidence
    base["created_at"] = datetime.now(timezone.utc).isoformat()
    base["parent_protocol_sha256"] = prior
    base["protocol_sha256"] = rerank.fingerprint(base)
    freeze_protocol.verify_protocol(base, rerank.Candidates(source, "test"), effective())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as target:
        target.write(json.dumps(base, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "freeze"))
    parser.add_argument("--source", type=Path, default=ROOT / "runs/retrieval-20260927")
    parser.add_argument("--split", choices=("pilot", "test"), default="pilot")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--budget-usd")
    parser.add_argument("--max-api-requests", type=int)
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--allow-paid-api", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--base-protocol", type=Path)
    parser.add_argument("--reviewed-pilot", action="store_true")
    args = parser.parse_args()
    if args.action == "plan":
        result = plan(args.source, args.split)
    elif args.action == "freeze":
        if not all((args.pilot, args.base_protocol, args.out)):
            parser.error("freeze requires --pilot, --base-protocol and --out")
        protocol = freeze(args.pilot, args.source, args.base_protocol, args.out, reviewed=args.reviewed_pilot)
        result = {"protocol_sha256": protocol["protocol_sha256"], "allowed_models": sorted(protocol["models"])}
    else:
        if not all((args.out, args.ledger, args.budget_usd, args.max_api_requests, args.max_runtime_seconds)):
            parser.error("run requires --out, --ledger, --budget-usd, --max-api-requests and --max-runtime-seconds")
        result = run(args.source, args.split, args.out, args.ledger, args.budget_usd,
                     args.max_api_requests, args.max_runtime_seconds, allow_paid_api=args.allow_paid_api,
                     protocol=rerank.read_json(args.protocol) if args.protocol else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("complete") is False:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
