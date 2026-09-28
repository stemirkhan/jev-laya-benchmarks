"""Freeze reviewed pilot evidence before test inference. No models or qrels are read."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import rerank

GPU_MODELS = frozenset(("laya-multi", "bge", "qwen-0.6b", "qwen-8b"))
INFERENCE_FILES = ("local_adapters.py", "jev_adapter.py", "audit.py", "prepare.py", "uv.lock")
SCHEMA = "rerank-frozen-protocol-v1"


def inference_hashes():
    return {name: rerank.sha(rerank.ROOT / name) for name in INFERENCE_FILES}


def effective_config(config, model):
    settings = config["models"][model]
    return {"model": model, "model_config": settings, "batch_size": settings["batch_size"],
            "warmup_lists": config["warmup_lists"], "repeats": config["repeats"],
            "status": config["status"]}


def validate_pilot(path, bundle, config):
    """Validate completed evidence; historical failed attempts remain in the journal."""
    path = Path(path)
    manifest = rerank.read_json(path / "manifest.json")
    identity = manifest["identity"]
    settings = identity["config"]
    model = settings["model"]
    if model not in GPU_MODELS | {"jev"}:
        raise ValueError("Unknown pilot model")
    if identity["split"] != "pilot" or identity["source"] != bundle.plan():
        raise ValueError("Pilot candidate source/split differs")
    if settings != effective_config(config, model):
        raise ValueError(f"Pilot settings differ from config: {model}")
    if {name: identity["code_sha256"].get(name) for name in INFERENCE_FILES} != inference_hashes():
        raise ValueError(f"Pilot inference code/instructions/dependencies changed: {model}")
    summary = rerank.read_json(path / "summary.json")
    if summary.get("complete") is not True or summary.get("split") != "pilot":
        raise ValueError("Pilot must be complete and use the pilot split")
    if summary.get("test_quality_computed") is not False:
        raise ValueError("Pilot must not use test quality")
    for field, name in (("results_sha256", "lists.jsonl"), ("events_sha256", "events.jsonl")):
        if summary.get(field) != rerank.sha(path / name):
            raise ValueError(f"Pilot artifact changed: {name}")
    jobs = rerank.job_sequence(bundle, settings["warmup_lists"], settings["repeats"])
    expected = {key: (phase, item) for key, phase, item in jobs}
    results = rerank.rows(path / "lists.jsonl")
    events = rerank.rows(path / "events.jsonl")
    done = {row["job_id"] for row in results}
    starts = {(event["job_id"], event["attempt"]) for event in events if event["event"] == "list_start"}
    successes = {(event["job_id"], event["attempt"]) for event in events if event["event"] == "list_success"}
    if (len(done) != len(results) or not set(expected) <= done
            or {key for key, _ in starts} - done):
        raise ValueError("Pilot has duplicate, missing or interrupted lists")
    for row in results:
        key = row["job_id"]
        if key in expected:
            phase, item = expected[key]
        elif key.startswith("resume_warmup/"):
            # A resumed process records a fresh pilot warmup, not a measurement.
            try:
                _, session, index = key.split("/")
                if int(session) < 1 or not 0 <= int(index) < settings["warmup_lists"]:
                    raise ValueError
                phase, item = "warmup", jobs[int(index)][2]
            except (ValueError, IndexError):
                raise ValueError("Invalid resumed warmup") from None
        else:
            raise ValueError("Unexpected pilot list")
        if (row["phase"] != phase or row["dataset"] != item["dataset"]
                or row["method"] != item["method"] or row["query_id"] != item["query_id"]
                or (key, row["attempt"]) not in starts & successes
                or row["ranking"] != rerank.sort_results(item["retrieval"], row["raw_results"])):
            raise ValueError("Pilot list does not match source or successful journal entry")
    measured = sum(phase == "measure" for phase, _ in expected.values())
    if summary.get("lists") != measured or summary.get("pair_occurrences") != measured * 20:
        raise ValueError("Pilot summary count differs")
    adapter = rerank.read_json(path / "adapter.json")
    return model, {
        "config": settings, "environment": identity["environment"], "adapter": adapter,
        "pilot_path": str(path.resolve()), "pilot_created_at": manifest["created_at"],
        "pilot_artifacts_sha256": {name: rerank.sha(path / name) for name in
                                   ("manifest.json", "summary.json", "events.jsonl", "lists.jsonl", "adapter.json")},
        "historical_list_errors": sum(e["event"] == "list_error" for e in events),
        "historical_sessions": sum(e["event"] == "session_start" for e in events),
    }


def freeze(pilots, source, config, out, *, reviewed=False):
    if not reviewed:
        raise ValueError("Review pilot outputs/errors and confirm --reviewed-pilots first")
    out = Path(out)
    if out.exists():
        raise ValueError("Protocol already exists; never overwrite a frozen protocol")
    pilot_bundle = rerank.Candidates(source, "pilot")
    models = {}
    for pilot in pilots:
        if not Path(pilot).is_dir():
            raise ValueError(f"Pilot directory does not exist: {pilot}")
        # A successful summary alone does not authorize freezing a live writer.
        with rerank.exclusive_run(Path(pilot)):
            model, evidence = validate_pilot(pilot, pilot_bundle, config)
        if model in models:
            raise ValueError("Duplicate pilot model")
        models[model] = evidence
    if not GPU_MODELS <= models.keys():
        raise ValueError("All four successful GPU pilots are required")
    gpu_environments = [models[model]["environment"] for model in sorted(GPU_MODELS)]
    gpu_hardware = [models[model]["adapter"].get("gpu") for model in sorted(GPU_MODELS)]
    if any(value != gpu_environments[0] for value in gpu_environments[1:]):
        raise ValueError("GPU pilot environments differ")
    if not gpu_hardware[0] or any(value != gpu_hardware[0] for value in gpu_hardware[1:]):
        raise ValueError("GPU pilots must use the same recorded GPU")
    protocol = {"schema": SCHEMA, "status": "frozen_after_reviewed_pilots",
                "created_at": datetime.now(timezone.utc).isoformat(), "reviewed_pilots": True,
                "test_metrics_used_for_tuning": False, "models": models,
                "pilot_source": pilot_bundle.plan(),
                "test_source": rerank.Candidates(source, "test").plan(),
                "inference_code_sha256": inference_hashes(),
                "runner_code_sha256": rerank.source_hashes()}
    protocol["protocol_sha256"] = rerank.fingerprint(protocol)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: a later pilot cannot silently replace this protocol.
    with out.open("x") as target:
        target.write(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())
    return protocol


def verify_protocol(protocol, bundle, config):
    if protocol is None:
        raise ValueError("Main test is locked; require --protocol from reviewed pilots")
    payload = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if protocol.get("protocol_sha256") != rerank.fingerprint(payload):
        raise ValueError("Frozen protocol checksum differs")
    if (protocol.get("schema") != SCHEMA or protocol.get("status") != "frozen_after_reviewed_pilots"
            or protocol.get("reviewed_pilots") is not True
            or protocol.get("test_metrics_used_for_tuning") is not False):
        raise ValueError("Invalid frozen pilot protocol")
    if not GPU_MODELS <= protocol["models"].keys():
        raise ValueError("Protocol lacks four GPU pilots")
    model = config["model"]
    if model not in protocol["models"]:
        raise ValueError(f"Model has no reviewed successful pilot in protocol: {model}")
    selected = protocol["models"][model]
    if selected["config"] != config:
        raise ValueError("Settings differ from frozen pilot protocol")
    if (protocol["inference_code_sha256"] != inference_hashes()
            or protocol["runner_code_sha256"] != rerank.source_hashes()):
        raise ValueError("Code/instructions/dependencies differ from frozen protocol")
    if bundle.split != "test" or protocol["test_source"] != bundle.plan():
        raise ValueError("Test source differs from frozen protocol")
    if protocol["pilot_source"] != rerank.Candidates(bundle.source, "pilot").plan():
        raise ValueError("Warmup source differs from frozen protocol")
    if selected["environment"] != rerank.environment():
        raise ValueError("Environment differs from reviewed pilot")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, action="append", required=True,
                        help="One completed pilot directory per model; repeat for all four GPU models")
    parser.add_argument("--source", type=Path, default=rerank.ROOT / "runs/retrieval-20260927")
    parser.add_argument("--config", type=Path, default=rerank.ROOT / "config/rerank-pilot.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reviewed-pilots", action="store_true",
                        help="Explicitly confirm review of all pilot outputs and historical failures")
    args = parser.parse_args()
    result = freeze(args.pilot, args.source, rerank.read_json(args.config), args.out,
                    reviewed=args.reviewed_pilots)
    print(json.dumps({"protocol": str(args.out), "protocol_sha256": result["protocol_sha256"],
                      "allowed_models": sorted(result["models"])}, indent=2))


if __name__ == "__main__":
    main()
