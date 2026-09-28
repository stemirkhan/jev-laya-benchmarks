"""Fixed pilot batch-pipeline throughput; not maximum hardware throughput or latency."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import freeze_protocol
import rerank

REPETITIONS = 3
PAIRS_PER_REPETITION = 480
BATCH_SIZES = {"laya-multi": 4, "bge": 4, "qwen-0.6b": 4, "qwen-8b": 2}
SCOPE = "sequential fixed-batch pilot pipeline, not maximum hardware throughput, interactive latency or quality"


class DeadlineExceeded(TimeoutError):
    pass


def validate_runtime(seconds):
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or not 0 < seconds <= 180):
        raise ValueError("max_runtime_seconds must be positive and at most 180")


@contextmanager
def deadline_alarm(deadline):
    """Interrupt Python work at the deadline; native calls may defer signal delivery."""
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise ValueError("An existing process timer would conflict with the throughput deadline")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DeadlineExceeded("Throughput time budget expired")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise DeadlineExceeded("Throughput time budget expired")

    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, remaining)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def authorized_model(bundle, config, model, protocol):
    """Use the same frozen evidence, without opening test texts or qrels."""
    if model not in BATCH_SIZES:
        raise ValueError("Only the four reviewed local GPU models are supported")
    if not isinstance(protocol, dict):
        raise ValueError("A frozen reviewed-pilot protocol is required")
    payload = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if protocol.get("protocol_sha256") != rerank.fingerprint(payload):
        raise ValueError("Frozen protocol checksum differs")
    if (protocol.get("schema") != freeze_protocol.SCHEMA
            or protocol.get("status") != "frozen_after_reviewed_pilots"
            or protocol.get("reviewed_pilots") is not True
            or protocol.get("test_metrics_used_for_tuning") is not False
            or not freeze_protocol.GPU_MODELS <= protocol.get("models", {}).keys()):
        raise ValueError("A valid protocol with all four reviewed GPU pilots is required")
    effective = freeze_protocol.effective_config(config, model)
    selected = protocol["models"][model]
    if effective != selected["config"]:
        raise ValueError("Settings differ from frozen pilot protocol")
    if (effective["batch_size"] != BATCH_SIZES[model]
            or effective["warmup_lists"] != 2 or effective["repeats"] != 1):
        raise ValueError("Require original batch size, two warmup lists and original pilot config")
    if (protocol["inference_code_sha256"] != freeze_protocol.inference_hashes()
            or protocol["runner_code_sha256"] != rerank.source_hashes()):
        raise ValueError("Code/instructions/dependencies differ from frozen protocol")
    if bundle.split != "pilot" or protocol["pilot_source"] != bundle.plan():
        raise ValueError("Only the exact frozen pilot input is permitted")
    if selected["environment"] != rerank.environment():
        raise ValueError("Environment differs from reviewed pilot")
    if len(bundle.lists) != 24 or sum(len(item["pairs"]) for item in bundle.lists) != PAIRS_PER_REPETITION:
        raise ValueError("Require exactly the 24 frozen pilot lists / 480 pair occurrences")
    return effective, selected


def validate_results(values, count, model, max_length):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("Missing or extra model scores")
    for value in values:
        score, tokens = value["score"], value["input_tokens"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Invalid score")
        if model != "bge" and not 0 <= score <= 1:
            raise ValueError("Probability outside 0..1")
        if type(tokens) is not int or not 0 < tokens <= max_length or "raw" not in value:
            raise ValueError("Invalid token count or missing raw output")


def execute(bundle, out, config, model, protocol, factory, *, max_runtime_seconds=180):
    """One attempt only. Resume/retry would change a timed stream and is forbidden."""
    run_started = time.monotonic()
    validate_runtime(max_runtime_seconds)
    deadline = run_started + max_runtime_seconds
    effective, selected = authorized_model(bundle, config, model, protocol)
    batch_size = effective["batch_size"]
    stream, order = [], []
    for item in bundle.lists:
        for pair, candidate in zip(item["pairs"], item["retrieval"], strict=True):
            stream.append(pair)
            order.append({"dataset": item["dataset"], "method": item["method"],
                          "query_id": item["query_id"], "doc_id": pair["doc_id"],
                          "source_rank": candidate["rank"], "text_sha256": pair["text_sha256"]})
    out = Path(out)
    with rerank.exclusive_run(out):
        if any(path.name != ".lock" for path in out.iterdir()):
            raise ValueError("Existing throughput attempt is preserved; no automatic resume or retry")
        hashes = {**rerank.source_hashes(), "throughput.py": rerank.sha(Path(__file__))}
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "attempt": 1,
                    "scope": SCOPE, "config": effective, "environment": rerank.environment(),
                    "source": bundle.plan(), "code_sha256": hashes,
                    "protocol_sha256": protocol["protocol_sha256"],
                    "input_order": order, "input_order_sha256": rerank.fingerprint(order),
                    "repetitions": REPETITIONS, "pairs_per_repetition": PAIRS_PER_REPETITION,
                    "warmup_pairs": 40, "max_runtime_seconds": max_runtime_seconds,
                    "timing_includes": "pair preparation,serialization,tokenization,transfer,inference,CUDA synchronization,validation,durable batch journal",
                    "timing_excludes": "source preflight,cold model load,warmup,ranking/retrieval/generation",
                    "no_qrels": True, "quality_computed": False, "automatic_retries": 0}
        rerank.write_json(out / "manifest.json", manifest)
        rerank.write_json(out / "protocol.json", protocol)
        (out / "source").mkdir()
        for name in hashes:
            shutil.copy2(Path(__file__) if name == "throughput.py" else rerank.ROOT / name,
                         out / "source" / name)
        events, batches = out / "events.jsonl", out / "batches.jsonl"
        adapter, current = None, {"phase": "load", "attempt": 1}
        stages, cold_seconds = [], None
        measured_pairs = warmup_pairs = 0

        def check_deadline():
            if time.monotonic() >= deadline:
                raise DeadlineExceeded("Throughput time budget expired")

        def stage(phase, repetition, pairs):
            nonlocal measured_pairs, warmup_pairs
            current.clear()
            current.update(phase=phase, repetition=repetition, attempt=1)
            check_deadline()
            adapter.synchronize()
            started = time.perf_counter()
            rerank.append(events, {"event": "stage_start", **current, "pair_count": len(pairs)})
            for offset in range(0, len(pairs), batch_size):
                check_deadline()
                current["offset"] = offset
                rerank.append(events, {"event": "batch_start", **current,
                                       "pair_count": min(batch_size, len(pairs) - offset)})
                prepared = [(pair["query"], pair["document"]) for pair in pairs[offset:offset + batch_size]]
                values = adapter.score(prepared)
                adapter.synchronize()
                # Preserve returned raw outputs before validation, including erroneous outputs.
                try:
                    rerank.append(batches, {**current, "input_order_offsets": list(range(offset, offset + len(prepared))),
                                            "raw_results": values})
                except (TypeError, ValueError):
                    rerank.append(events, {"event": "invalid_raw_serialization", **current,
                                           "raw_results_repr": repr(values)})
                    raise
                validate_results(values, len(prepared), model, effective["model_config"]["max_length"])
                rerank.append(events, {"event": "batch_success", **current, "pair_count": len(prepared)})
                if phase == "measure":
                    measured_pairs += len(prepared)
                else:
                    warmup_pairs += len(prepared)
                check_deadline()
            adapter.synchronize()
            elapsed = time.perf_counter() - started
            result = {"phase": phase, "repetition": repetition, "pairs": len(pairs),
                      "seconds": elapsed, "pairs_per_second": len(pairs) / elapsed}
            rerank.append(events, {"event": "stage_success", **result})
            stages.append(result)

        reason = None
        try:
            with deadline_alarm(deadline):
                check_deadline()
                rerank.append(events, {"event": "load_start", **current})
                started = time.perf_counter()
                adapter = factory()
                adapter.synchronize()
                cold_seconds = time.perf_counter() - started
                metadata = adapter.metadata()
                rerank.write_json(out / "adapter.json", metadata)
                if metadata != selected["adapter"]:
                    raise ValueError("Adapter/hardware differs from reviewed pilot")
                rerank.append(events, {"event": "load_success", "cold_start_seconds": cold_seconds})
                check_deadline()
                stage("warmup", -1, stream[:40])
                for repetition in range(REPETITIONS):
                    stage("measure", repetition, stream)
        except DeadlineExceeded:
            reason = "max_runtime_seconds"
            rerank.append(events, {"event": "runtime_stop", **current,
                                   "measured_pairs": measured_pairs, "warmup_pairs": warmup_pairs})
        except BaseException as exc:
            rerank.append(events, {"event": "error", **current, "error_type": type(exc).__name__,
                                   "measured_pairs": measured_pairs, "warmup_pairs": warmup_pairs})
            rerank.write_json(out / "failure.json", {"complete": False, "error_type": type(exc).__name__,
                                                     **current, "measured_pairs": measured_pairs})
            raise
        finally:
            if adapter is not None:
                adapter.close()
        measurements = [item for item in stages if item["phase"] == "measure"]
        complete = reason is None and len(measurements) == REPETITIONS and measured_pairs == 1440
        seconds = sum(item["seconds"] for item in measurements)
        result = {"complete": complete, "reason": reason, "scope": SCOPE,
                  "model": model, "measured_pairs": measured_pairs, "warmup_pairs": warmup_pairs,
                  "completed_repetitions": len(measurements), "measurements": measurements,
                  "cold_start_seconds": cold_seconds,
                  "warmup_seconds": next((item["seconds"] for item in stages if item["phase"] == "warmup"), None),
                  "total_measured_seconds": seconds if complete else None,
                  "pairs_per_second": measured_pairs / seconds if complete else None,
                  "total_wall_seconds": time.monotonic() - run_started,
                  "batches_sha256": rerank.sha(batches) if batches.exists() else None,
                  "events_sha256": rerank.sha(events), "quality_computed": False}
        rerank.write_json(out / ("summary.json" if complete else "partial.json"), result)
        return result


def supervise(arguments, out, seconds):
    """Keep a hard process deadline even when a native CUDA call defers SIGALRM."""
    validate_runtime(seconds)
    out = Path(out)
    if out.exists() and any(path.name != ".lock" for path in out.iterdir()):
        raise ValueError("Existing throughput attempt is preserved; no automatic resume or retry")
    command = [sys.executable, str(Path(__file__).resolve()), *arguments, "--worker"]
    try:
        return subprocess.run(command, timeout=seconds).returncode
    except subprocess.TimeoutExpired:
        # subprocess.run kills and reaps the worker before raising. No live writer remains.
        with rerank.exclusive_run(out):
            summary = out / "summary.json"
            if summary.exists():
                result = rerank.read_json(summary)
                if (result.get("complete") is True
                        and result["events_sha256"] == rerank.sha(out / "events.jsonl")
                        and result["batches_sha256"] == rerank.sha(out / "batches.jsonl")):
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    return 0
            result = {"complete": False, "reason": "hard_process_deadline", "scope": SCOPE,
                      "max_runtime_seconds": seconds, "pairs_per_second": None,
                      "measured_pairs": 0, "warmup_pairs": 0, "incomplete_event_tail": False,
                      "automatic_retries": 0}
            events = out / "events.jsonl"
            if events.exists():
                with events.open() as source:
                    for line in source:
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            result["incomplete_event_tail"] = True
                            break  # Keep the original interrupted journal, never repair or truncate it.
                        if event["event"] == "batch_success":
                            key = "measured_pairs" if event["phase"] == "measure" else "warmup_pairs"
                            result[key] += event["pair_count"]
            result["artifacts_sha256"] = {name: rerank.sha(out / name) for name in
                                         ("manifest.json", "batches.jsonl", "events.jsonl") if (out / name).exists()}
            rerank.write_json(out / "watchdog.json", result)
            if not (out / "partial.json").exists():
                rerank.write_json(out / "partial.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 75


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=rerank.ROOT / "runs/retrieval-20260927")
    parser.add_argument("--config", type=Path, default=rerank.ROOT / "config/rerank-pilot.json")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(BATCH_SIZES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-runtime-seconds", type=float, default=180)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        raise SystemExit(supervise(sys.argv[1:], args.out, args.max_runtime_seconds))
    config = rerank.read_json(args.config)

    def factory():
        from local_adapters import create_local_adapter
        return create_local_adapter(args.model, config["models"][args.model], project_root=rerank.ROOT)

    result = execute(rerank.Candidates(args.source, "pilot"), args.out, config, args.model,
                     rerank.read_json(args.protocol), factory, max_runtime_seconds=args.max_runtime_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["complete"]:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
