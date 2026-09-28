"""Pairwise reranking of frozen top-20 lists. Planning is offline by default."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def rows(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source]


def append(path, value):
    # An incomplete final line after a crash is deliberately not silently discarded.
    with Path(path).open("a") as dest:
        dest.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        dest.flush()
        os.fsync(dest.fileno())


@contextmanager
def exclusive_run(path):
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This run is already active") from None
        yield


class Candidates:
    """Read only frozen candidates and text. No qrels enter model input."""

    def __init__(self, source, split="pilot"):
        self.source, self.split = Path(source).resolve(), split
        if split not in {"pilot", "test"}:
            raise ValueError("Unknown split")
        summary = read_json(self.source / "summary.json")
        config = read_json(self.source / "config.json")
        if summary["config_sha256"] != fingerprint(config) or config["search"]["top_k"] != 20:
            raise ValueError("Changed retrieval config or top-k")
        self.summary_sha256 = sha(self.source / "summary.json")
        self.files, self.lists, self.pairs = {}, [], {}
        for dataset in config["datasets"]:
            def checked(name):
                rel = f"{dataset}/{name}"
                path = self.source / rel
                if not path.resolve().is_relative_to(self.source):
                    raise ValueError("Path escapes candidate run")
                actual = sha(path)
                if summary["artifacts_sha256"].get(rel) != actual:
                    raise ValueError(f"Changed candidate artifact: {rel}")
                self.files[rel] = actual
                return path

            selected = read_json(checked("queries.json"))[split]
            query_rows = rows(checked("query-texts.jsonl"))
            queries = {r["query_id"]: r["text"] for r in query_rows}
            if len(queries) != len(query_rows) or len(set(selected)) != len(selected):
                raise ValueError("Duplicate query IDs")
            pool = {}
            with gzip.open(checked(f"pairs.{split}.jsonl.gz"), "rt") as src:
                for line in src:
                    pair = json.loads(line)
                    key = (pair["query_id"], pair["doc_id"])
                    if key in pool or key[0] not in selected or pair["query"] != queries[key[0]]:
                        raise ValueError("Duplicate, unexpected or changed pair")
                    if hashlib.sha256(pair["document"].encode()).hexdigest() != pair["text_sha256"]:
                        raise ValueError("Changed full document")
                    pool[key] = pair
                    self.pairs[(dataset, *key)] = pair
            used = set()
            for method in ("bm25", "rrf"):
                data = rows(checked(f"candidates.{split}.{method}.jsonl"))
                if len(data) != len(selected) or {r["query_id"] for r in data} != set(selected):
                    raise ValueError("Missing, duplicate or unexpected query list")
                for row in data:
                    qid, docs = row["query_id"], row["candidates"]
                    if len(docs) != 20 or len({d["doc_id"] for d in docs}) != 20:
                        raise ValueError("Candidate list must contain 20 distinct documents")
                    if [d["rank"] for d in docs] != list(range(1, 21)):
                        raise ValueError("Invalid candidate ranks")
                    pairs = []
                    for d in docs:
                        if not math.isfinite(d["score"]):
                            raise ValueError("Non-finite retrieval score")
                        key = (qid, d["doc_id"])
                        if key not in pool:
                            raise ValueError("Missing pair text")
                        used.add(key)
                        pairs.append(pool[key])
                    self.lists.append({"dataset": dataset, "method": method, "query_id": qid,
                                       "pairs": pairs, "retrieval": docs})
            if used != set(pool):
                raise ValueError("Pair pool differs from candidate lists")

    def plan(self):
        return {"split": self.split, "lists": len(self.lists),
                "pair_occurrences": len(self.lists) * 20, "unique_pairs": len(self.pairs),
                "retrieval_summary_sha256": self.summary_sha256,
                "selected_artifacts_sha256": self.files,
                "qrels_sent_to_models": False, "inference_performed": False}


def sort_results(candidates, results):
    if len(results) != len(candidates):
        raise ValueError("Missing or extra model scores")
    output = []
    for source, result in zip(candidates, results, strict=True):
        score = result["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Invalid model score")
        output.append({"doc_id": source["doc_id"], "score": score, "source_rank": source["rank"]})
    output.sort(key=lambda r: (-r["score"], r["source_rank"], r["doc_id"]))
    for rank, row in enumerate(output, 1):
        row["rank"] = rank
    return output


def source_hashes():
    names = ("rerank.py", "freeze_protocol.py", "local_adapters.py", "jev_adapter.py", "audit.py", "prepare.py", "uv.lock")
    return {name: sha(ROOT / name) for name in names if (ROOT / name).is_file()}


def environment():
    versions = {}
    for name in ("torch", "transformers", "requests", "tokenizers", "numpy", "safetensors", "huggingface-hub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": sys.version, "platform": platform.platform(), "packages": versions}


def job_sequence(bundle, warmups, repeats):
    if warmups < 1 or repeats < 1:
        raise ValueError("Require at least one warmup and measurement")
    # Warmups use pilot data only, even for a future test run.
    warm_bundle = bundle if bundle.split == "pilot" else Candidates(bundle.source, "pilot")
    jobs = [(f"warmup/{i}", "warmup", warm_bundle.lists[i % len(warm_bundle.lists)]) for i in range(warmups)]
    for repeat in range(repeats):
        for item in bundle.lists:
            key = f"measure/{repeat}/{item['dataset']}/{item['method']}/{item['query_id']}"
            jobs.append((key, "measure", item))
    return jobs


def execute(bundle, out, config, factory, *, retry_failed=False, api_limit=None,
            max_runtime_seconds=None, protocol=None):
    """No automatic retries, no score cache within a list, no qrels/quality metrics."""
    session_started = time.monotonic()
    if (max_runtime_seconds is not None
            and (isinstance(max_runtime_seconds, bool) or not isinstance(max_runtime_seconds, (int, float))
                 or not math.isfinite(max_runtime_seconds) or max_runtime_seconds <= 0)):
        raise ValueError("max_runtime_seconds must be finite and positive")
    if bundle.split == "test":
        from freeze_protocol import verify_protocol
        frozen_model = verify_protocol(protocol, bundle, config)
    else:
        if protocol is not None:
            raise ValueError("Frozen protocols authorize only test runs")
        frozen_model = None
    out = Path(out)
    if config.get("model") == "jev":
        from jev_adapter import validate_jev_config
        validate_jev_config({**config["model_config"], "max_requests": api_limit})
    batch_size = config["batch_size"]
    if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 20:
        raise ValueError("batch_size must be in 1..20")
    jobs = job_sequence(bundle, config["warmup_lists"], config["repeats"])
    identity = {"config": config, "split": bundle.split,
                "source": bundle.plan(), "code_sha256": source_hashes(), "environment": environment(),
                "api_request_cap": api_limit}
    if protocol is not None:
        identity["protocol_sha256"] = protocol["protocol_sha256"]
    with exclusive_run(out):
        manifest = out / "manifest.json"
        if manifest.exists():
            if read_json(manifest)["identity"] != identity:
                raise ValueError("Run inputs/config/code/environment changed; use a new run")
        else:
            if any(out.iterdir()) and set(p.name for p in out.iterdir()) != {".lock"}:
                raise ValueError("Nonempty output without manifest")
            write_json(manifest, {"identity": identity, "created_at": datetime.now(timezone.utc).isoformat()})
            (out / "source").mkdir()
            for name in identity["code_sha256"]:
                shutil.copy2(ROOT / name, out / "source" / name)
            if protocol is not None:
                write_json(out / "protocol.json", protocol)
        events_path = out / "events.jsonl"
        events = rows(events_path) if events_path.exists() else []
        results_path = out / "lists.jsonl"
        completed_rows = rows(results_path) if results_path.exists() else []
        done = {r["job_id"] for r in completed_rows}
        planned_ids = {j[0] for j in jobs}
        unexpected = {key for key in done if key not in planned_ids and not key.startswith("resume_warmup/")}
        if len(done) != len(completed_rows) or unexpected:
            raise ValueError("Duplicate or unknown completed job")
        starts = {e["job_id"] for e in events if e["event"] == "list_start"}
        pending = starts - done
        if pending and not retry_failed:
            raise ValueError("Interrupted/failed list retained. Inspect events; --retry-failed reruns full list")
        api_used = sum(e["event"] == "api" and e.get("detail", {}).get("phase") == "start" for e in events)
        if all(key in done for key, _, _ in jobs) and (out / "summary.json").exists():
            summary = read_json(out / "summary.json")
            if summary["results_sha256"] != sha(results_path) or summary["events_sha256"] != sha(events_path):
                raise ValueError("Completed run artifacts changed")
            return summary
        if api_limit is not None and api_used >= api_limit:
            raise ValueError("API request cap reached")

        def stopped_by_runtime():
            if max_runtime_seconds is None or time.monotonic() - session_started < max_runtime_seconds:
                return None
            status = {"complete": False, "reason": "max_runtime_seconds", "split": bundle.split,
                      "completed_lists": sum(key in done for key, phase, _ in jobs if phase == "measure"),
                      "max_runtime_seconds": max_runtime_seconds,
                      "session_seconds": time.monotonic() - session_started}
            append(events_path, {"event": "runtime_stop", **status})
            # Never write summary.json for a partial run. Resume uses the same output directory.
            return status

        stopped = stopped_by_runtime()
        if stopped is not None:
            return stopped
        current = {}
        journal_lock = threading.Lock()

        def event_callback(detail):
            with journal_lock:
                append(events_path, {"event": "api", **current, "detail": detail})

        load_start = time.perf_counter()
        adapter = factory(event_callback, None if api_limit is None else api_limit - api_used)
        try:
            adapter.synchronize()
            metadata = adapter.metadata()
            # Remaining request allowance decreases on resume, not a model setting.
            metadata.pop("max_requests", None)
            if frozen_model is not None and metadata != frozen_model["adapter"]:
                raise ValueError("Adapter/hardware differs from reviewed pilot")
            previous_metadata = out / "adapter.json"
            if previous_metadata.exists() and read_json(previous_metadata) != metadata:
                raise ValueError("Adapter metadata changed; new run required")
            write_json(previous_metadata, metadata)
            append(events_path, {"event": "session_start", "load_seconds": time.perf_counter() - load_start,
                                 "max_runtime_seconds": max_runtime_seconds})
            # A restarted process is cold. Extra warmups are recorded, never treated as measurements.
            session = sum(e["event"] == "session_start" for e in events)
            if session:
                extra = [(f"resume_warmup/{session}/{i}", "warmup", j[2])
                         for i, j in enumerate(jobs[:config["warmup_lists"]])]
            else:
                extra = []
            for job_id, phase, item in [*extra, *jobs]:
                if job_id in done:
                    continue
                stopped = stopped_by_runtime()
                if stopped is not None:
                    return stopped
                current.clear()
                current.update(job_id=job_id)
                attempt = 1 + sum(e.get("job_id") == job_id and e["event"] == "list_start" for e in events)
                append(events_path, {"event": "list_start", "job_id": job_id, "attempt": attempt, "phase": phase})
                adapter.synchronize()
                if hasattr(adapter, "reset_peak_memory"):
                    adapter.reset_peak_memory()
                started = time.perf_counter()
                scores = []
                try:
                    for begin in range(0, 20, batch_size):
                        pairs = item["pairs"][begin:begin + batch_size]
                        batch_start = time.perf_counter()
                        values = adapter.score([(p["query"], p["document"]) for p in pairs])
                        adapter.synchronize()
                        # Preserve raw outputs even if score validation fails below.
                        detail = {"event": "batch", "job_id": job_id, "attempt": attempt, "offset": begin,
                                  "seconds": time.perf_counter() - batch_start, "results": values}
                        # Durable before the next forward pass, including KeyboardInterrupt.
                        append(events_path, detail)
                        if len(values) != len(pairs):
                            raise ValueError("Incomplete model batch")
                        scores.extend(values)
                    ranking = sort_results(item["retrieval"], scores)
                    adapter.synchronize()
                    elapsed = time.perf_counter() - started
                    result = {"job_id": job_id, "phase": phase, "attempt": attempt,
                              "dataset": item["dataset"], "method": item["method"],
                              "query_id": item["query_id"], "elapsed_seconds": elapsed,
                              "ranking": ranking, "raw_results": scores,
                              "memory": adapter.memory_stats() if hasattr(adapter, "memory_stats") else None}
                    append(results_path, result)
                    append(events_path, {"event": "list_success", "job_id": job_id, "attempt": attempt})
                    done.add(job_id)
                except BaseException as exc:
                    append(events_path, {"event": "list_error", "job_id": job_id, "attempt": attempt,
                                         "error_type": type(exc).__name__, "seconds": time.perf_counter() - started})
                    raise
        finally:
            adapter.close()
        all_results = rows(results_path)
        measured = [r for r in all_results if r["phase"] == "measure"]
        expected = {key for key, phase, _ in jobs if phase == "measure"}
        if {r["job_id"] for r in measured} != expected:
            raise ValueError("Incomplete run; no successful summary")
        summary = {"complete": True, "split": bundle.split, "lists": len(measured),
                   "pair_occurrences": len(measured) * 20, "test_quality_computed": False,
                   "latency_scope": "full top-20 rerank only, not retrieval or generation",
                   "timing_includes": "serialization,tokenization,transfer,compute,network,score validation,sort,durable batch/API journal",
                   "results_sha256": sha(results_path), "events_sha256": sha(events_path)}
        write_json(out / "summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--source", type=Path, default=ROOT / "runs/retrieval-20260927")
    parser.add_argument("--split", choices=("pilot", "test"), default="pilot")
    parser.add_argument("--config", type=Path, default=ROOT / "config/rerank-pilot.json")
    parser.add_argument("--model", choices=("jev", "laya-multi", "bge", "qwen-0.6b", "qwen-8b"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--allow-paid-api", action="store_true")
    parser.add_argument("--max-api-requests", type=int)
    parser.add_argument("--protocol", type=Path, help="Frozen reviewed-pilot protocol required for test")
    parser.add_argument("--max-runtime-seconds", type=float,
                        help="Per invocation, including load; stop before the next list, never mid-list")
    args = parser.parse_args()
    bundle = Candidates(args.source, args.split)
    config = read_json(args.config)
    if args.action == "plan":
        plan = bundle.plan()
        plan["warmup_lists"] = config["warmup_lists"]
        plan["repeats"] = config["repeats"]
        plan["pair_calls_including_warmup"] = (plan["lists"] * config["repeats"] + config["warmup_lists"]) * 20
        plan["jev_exact_token_check"] = "pending: no public tokenizer; validate API usage/context errors during authorized pilot"
        from jev_adapter import preflight_jev
        plan["jev_payload_preflight"] = preflight_jev(list(bundle.pairs.values()),
            {**config["models"]["jev"], "max_requests": 0})
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        if args.out:
            write_json(args.out, plan)
        return
    if not args.model or not args.out:
        parser.error("run requires --model and --out")
    if args.split == "test" and args.protocol is None:
        parser.error("Main test is locked. Complete pilot and pass its frozen --protocol first.")
    model_config = config["models"][args.model]
    if args.model == "jev":
        from jev_adapter import validate_jev_config
        validate_jev_config({**model_config, "max_requests": args.max_api_requests})
    effective = {"model": args.model, "model_config": model_config,
                 "batch_size": model_config["batch_size"], "warmup_lists": config["warmup_lists"],
                 "repeats": config["repeats"], "status": config["status"]}
    if args.model == "jev":
        if not args.allow_paid_api or not args.max_api_requests or args.max_api_requests < 1:
            parser.error("Jev requires agreed budget, --allow-paid-api and --max-api-requests")
        def factory(callback, remaining):
            from jev_adapter import create_jev_adapter
            return create_jev_adapter({**model_config, "max_requests": remaining}, event_callback=callback)
    else:
        def factory(callback, remaining):
            from local_adapters import create_local_adapter
            return create_local_adapter(args.model, model_config, project_root=ROOT)
    result = execute(bundle, args.out, effective, factory, retry_failed=args.retry_failed,
                     api_limit=args.max_api_requests if args.model == "jev" else None,
                     max_runtime_seconds=args.max_runtime_seconds,
                     protocol=read_json(args.protocol) if args.protocol else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["complete"]:
        raise SystemExit(75)  # Temporary stop, not a successful completed benchmark.


if __name__ == "__main__":
    main()
