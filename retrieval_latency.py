"""Single-query CPU retrieval latency from frozen artifacts, without qrels.

Run only when explicitly requested:
  .venv/bin/python retrieval_latency.py --out runs/retrieval-latency-cpu-001 --max-seconds 600

The deadline includes loading and warmup; per-query timers exclude them. Results
measure this CPU host's primary retrieval only, not retrieval + remote reranking.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
TEST_COUNTS = {"scifact": 300, "fiqa": 648, "nfcorpus": 323}
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "4", "RAYON_NUM_THREADS": "4", "MKL_DYNAMIC": "FALSE",
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append(path, value):
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quantile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def compare_candidates(actual, frozen):
    actual_ids = [row["doc_id"] for row in actual]
    frozen_ids = [row["doc_id"] for row in frozen]
    frozen_scores = {row["doc_id"]: row["score"] for row in frozen}
    errors = [abs(row["score"] - frozen_scores[row["doc_id"]])
              for row in actual if row["doc_id"] in frozen_scores]
    return {
        "same_order": actual_ids == frozen_ids,
        "same_set": set(actual_ids) == set(frozen_ids),
        "actual_ids": actual_ids, "frozen_ids": frozen_ids,
        "missing_ids": sorted(set(frozen_ids) - set(actual_ids)),
        "added_ids": sorted(set(actual_ids) - set(frozen_ids)),
        "max_abs_score_difference_on_shared_ids": max(errors, default=None),
    }


class SearchEngine:
    def __init__(self, config, index, bm25_ids, vectors, dense_ids, embedder):
        # Heavy libraries are imported only in the worker after setting threads.
        import retrieval as r
        self.r, self.config = r, config
        self.index, self.bm25_ids = index, bm25_ids
        self.vectors, self.dense_ids, self.embedder = vectors, dense_ids, embedder
        cfg = config["bm25"]
        self.token_args = {
            "lower": True, "token_pattern": cfg["token_pattern"], "stopwords": cfg["stopwords"],
            "stemmer": r.Stemmer.Stemmer(cfg["stemmer"]), "return_ids": False,
            "show_progress": False,
        }

    def lexical(self, text):
        tokens = self.r.bm25s.tokenize([text], **self.token_args)[0]
        scores = (self.index.get_scores(tokens) if tokens else
                  self.r.np.zeros(len(self.bm25_ids), dtype=self.r.np.float32))
        return self.r.top_scores(self.bm25_ids, scores, self.config["search"]["depth"])

    def dense(self, vector):
        scores = self.vectors @ vector
        return self.r.top_scores(self.dense_ids, scores, self.config["search"]["depth"])

    def search(self, text, method):
        if method not in ("bm25", "rrf"):
            raise ValueError("Unknown retrieval method")
        started = time.perf_counter()
        top_k = self.config["search"]["top_k"]
        if method == "bm25":
            candidates = self.lexical(text)[:top_k]
            elapsed = time.perf_counter() - started
            return candidates, elapsed, {"bm25_seconds": elapsed}
        # Independent full pipeline: no reuse of the preceding BM25 result or
        # saved query-vectors. Prefix, tokenizer and pooling are the frozen ones.
        vector = self.embedder.encode([self.config["embedding"]["query_prefix"] + text])[0]
        after_embedding = time.perf_counter()
        lexical = self.lexical(text)
        after_bm25 = time.perf_counter()
        dense = self.dense(vector)
        after_dense = time.perf_counter()
        candidates = self.r.rrf([lexical, dense], self.config["rrf"]["k"],
                                self.config["rrf"]["weights"])[:top_k]
        finished = time.perf_counter()
        return candidates, finished - started, {
            "embedding_seconds": after_embedding - started,
            "bm25_seconds": after_bm25 - after_embedding,
            "exact_dense_seconds": after_dense - after_bm25,
            "rrf_seconds": finished - after_dense,
        }


def check_query_selection(selected, texts, expected_count):
    for split, count in (("pilot", 4), ("test", expected_count)):
        ids = selected[split]
        if len(ids) != count or len(set(ids)) != count or any(q not in texts for q in ids):
            raise ValueError(f"Invalid frozen {split} query selection")
    if set(selected["pilot"]) & set(selected["test"]):
        raise ValueError("Pilot overlaps test")


def load_artifacts(source, out):
    import retrieval as r
    config = json.loads((source / "config.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    if r.json_hash(config) != summary["config_sha256"]:
        raise ValueError("Frozen config hash mismatch")
    if (config["datasets"] != list(TEST_COUNTS) or config["embedding"]["threads"] != 4
            or config["embedding"]["provider"] != "CPUExecutionProvider"
            or config["search"] != {"depth": 100, "top_k": 20,
                                    "dense_metric": "cosine", "dense_index": "exact_float32"}
            or config["rrf"] != {"k": 60, "weights": [1.0, 1.0],
                                 "tie_break": "document_id_lexicographic"}):
        raise ValueError("Unexpected frozen retrieval protocol")
    receipts = {"config.json": sha256(source / "config.json"),
                "summary.json": sha256(source / "summary.json")}

    def verified(relative):
        path = source / relative
        actual = sha256(path)
        if actual != summary["artifacts_sha256"][relative]:
            raise ValueError(f"Frozen input hash mismatch: {relative}")
        receipts[relative] = actual
        return path

    embedding_manifest = json.loads((source / "manifests/embedding.json").read_text())
    if (embedding_manifest["repo"], embedding_manifest["revision"]) != (
            config["embedding"]["repo"], config["embedding"]["revision"]):
        raise ValueError("Embedding revision mismatch")
    for name, expected in embedding_manifest["files_sha256"].items():
        actual = sha256(ROOT / "assets/embedding" / name)
        if actual != expected:
            raise ValueError(f"Embedding asset changed: {name}")
        receipts[f"assets/embedding/{name}"] = actual
    embedder = r.Embedder(config["embedding"])
    datasets = {}
    for name, expected_count in TEST_COUNTS.items():
        selected = json.loads(verified(f"{name}/queries.json").read_text())
        text_rows = read_rows(verified(f"{name}/query-texts.jsonl"))
        texts = {row["query_id"]: row["text"] for row in text_rows}
        if len(texts) != len(text_rows):
            raise ValueError("Duplicate query text ID")
        check_query_selection(selected, texts, expected_count)
        bm25_ids = json.loads(verified(f"{name}/bm25-document-ids.json").read_text())
        dense_ids = json.loads(verified(f"{name}/document-vectors.ids.json").read_text())
        progress = json.loads(verified(f"{name}/document-vectors.progress.json").read_text())
        vectors = r.np.load(verified(f"{name}/document-vectors.npy"), allow_pickle=False)
        if (not progress["complete"] or len(set(bm25_ids)) != len(bm25_ids)
                or set(bm25_ids) != set(dense_ids) or len(set(dense_ids)) != len(dense_ids)
                or vectors.dtype != r.np.float32
                or vectors.shape != (len(dense_ids), config["embedding"]["dimensions"])
                or not r.np.isfinite(vectors).all()
                or not r.np.allclose(r.np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
            raise ValueError(f"Invalid stored vectors/IDs: {name}")
        for path in sorted((source / name / "bm25-index").iterdir()):
            if path.is_file():
                verified(str(path.relative_to(source)))
        index = r.bm25s.BM25.load(str(source / name / "bm25-index"), mmap=False, load_corpus=False)
        if index.scores["num_docs"] != len(bm25_ids):
            raise ValueError("BM25 document count mismatch")
        frozen = {}
        for split in ("pilot", "test"):
            frozen[split] = {}
            for method in ("bm25", "rrf"):
                path = verified(f"{name}/candidates.{split}.{method}.jsonl")
                rankings = r.read_rankings(path)
                if set(rankings) != set(selected[split]):
                    raise ValueError("Frozen candidate query mismatch")
                for ranking in rankings.values():
                    r.check_ranking(ranking, set(dense_ids), config["search"]["top_k"])
                frozen[split][method] = rankings
        datasets[name] = {
            "selected": selected, "texts": texts, "frozen": frozen,
            "engine": SearchEngine(config, index, bm25_ids, vectors, dense_ids, embedder),
        }
    write_json(out / "inputs.json", {"source": str(source), "sha256": receipts, "qrels_read": False,
                                   "saved_query_vectors_read": False})
    write_json(out / "config.json", config)
    return datasets


def measure_queries(datasets, out):
    # Warm every index and the encoder on the frozen train/dev pilot once.
    for split in ("pilot", "test"):
        for dataset, bundle in datasets.items():
            for query_id in bundle["selected"][split]:
                for method in ("bm25", "rrf"):
                    identity = {"split": split, "dataset": dataset,
                                "query_id": query_id, "method": method}
                    append(out / "events.jsonl", {"event": "started", **identity})
                    candidates, seconds, components = bundle["engine"].search(bundle["texts"][query_id], method)
                    comparison = compare_candidates(candidates, bundle["frozen"][split][method][query_id])
                    record = {**identity, "seconds": seconds, "components": components,
                              "candidates": candidates, "comparison": comparison}
                    append(out / ("warmup.jsonl" if split == "pilot" else "queries.jsonl"), record)
                    if not comparison["same_order"]:
                        append(out / "differences.jsonl", record)
    append(out / "events.jsonl", {"event": "worker_complete"})


def environment():
    import numpy as np
    cpu = "unknown"
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "python": sys.version, "platform": platform.platform(), "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV},
        "onnx_intra_op_threads": 4, "onnx_inter_op_threads": 1,
        "numpy_build": np.__config__.show(mode="dicts"),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "bm25s", "PyStemmer", "onnxruntime", "tokenizers")},
    }


def worker(source, out):
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        write_json(out / "environment.json", environment())
        datasets = load_artifacts(source, out)
        append(out / "events.jsonl", {"event": "loaded", "seconds": time.perf_counter() - started})
        measure_queries(datasets, out)
        write_json(out / "worker.json", {"wall_seconds": time.perf_counter() - started,
                                         "process_cpu_seconds": time.process_time() - cpu_started})
        return 0
    except Exception as exc:
        append(out / "events.jsonl", {"event": "error", "error": repr(exc), "traceback": traceback.format_exc()})
        return 1


def summarize(out, returncode, reason, wall_seconds):
    manifest = json.loads((out / "manifest.json").read_text())
    records, corrupt_tail = [], False
    path = out / "queries.jsonl"
    if path.exists():
        lines = path.read_text().splitlines()
        for number, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if number != len(lines) - 1:
                    raise
                corrupt_tail = True  # A forced timeout can interrupt one final write.
    expected = {(dataset, qid, method) for dataset, queries in manifest["test_queries"].items()
                for qid in queries for method in ("bm25", "rrf")}
    actual = [(r["dataset"], r["query_id"], r["method"]) for r in records]
    complete = returncode == 0 and not corrupt_tail and set(actual) == expected and len(set(actual)) == len(actual)
    grouped = {}
    for dataset in manifest["test_queries"]:
        grouped[dataset] = {}
        for method in ("bm25", "rrf"):
            rows = [r for r in records if (r["dataset"], r["method"]) == (dataset, method)]
            seconds = [r["seconds"] for r in rows]
            grouped[dataset][method] = {
                "n": len(rows), "expected_n": len(manifest["test_queries"][dataset]),
                "statistics_scope": "completed_queries_only" if not complete else "full_test",
                "p50_ms": 1000 * quantile(seconds, .5) if seconds else None,
                "p95_ms": 1000 * quantile(seconds, .95) if seconds else None,
                "total_seconds": sum(seconds),
                "candidate_order_mismatches": sum(not row["comparison"]["same_order"] for row in rows),
            }
    result = {
        "status": "complete" if complete else "partial", "complete": complete,
        "stop_reason": reason, "worker_returncode": returncode, "wall_seconds": wall_seconds,
        "completed_measurements": len(records), "expected_measurements": len(expected),
        "incomplete_last_record": corrupt_tail, "datasets": grouped,
        "measurement": "CPU primary retrieval on this host only; not end-to-end with remote GPU/API reranking",
        "candidate_differences": "Logged only; frozen candidates and settings never rewritten",
        "quantile_method": "linear interpolation at (n-1)*p",
    }
    write_json(out / "summary.json", result)
    return result


def supervise(command, out, max_seconds):
    started = time.perf_counter()
    cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (out / "worker.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
        reason = "finished"
        try:
            process.wait(timeout=max_seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            reason = "time_limit" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted"
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.returncode != 0 and reason == "finished":
            reason = "worker_error"
    result = summarize(out, process.returncode, reason, time.perf_counter() - started)
    cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result["worker_cpu_user_seconds"] = cpu_after.ru_utime - cpu_before.ru_utime
    result["worker_cpu_system_seconds"] = cpu_after.ru_stime - cpu_before.ru_stime
    result["worker_max_rss_kib"] = cpu_after.ru_maxrss
    write_json(out / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/retrieval-20260927")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be finite and positive")
    os.environ.update(THREAD_ENV)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    source, out = args.source.resolve(), args.out.resolve()
    if args.worker:
        return worker(source, out)
    if out == source or out.is_relative_to(source):
        parser.error("Output must not be inside the frozen retrieval run")
    test_queries = {}
    for name, count in TEST_COUNTS.items():
        selected = json.loads((source / name / "queries.json").read_text())
        if len(selected["test"]) != count or len(set(selected["test"])) != count:
            parser.error(f"Expected {count} unique test queries in {name}")
        test_queries[name] = selected["test"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "source").mkdir()
    for name in ("retrieval_latency.py", "retrieval.py", "audit.py", "prepare.py", "pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, out / "source" / name)
    write_json(out / "manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "source": str(source),
        "max_seconds": args.max_seconds, "test_queries": test_queries,
        "methods": ["bm25", "rrf"], "query_concurrency": 1, "embedding_batch_size": 1,
        "warmup": "all 12 frozen pilot queries, each method once, before test",
        "order": "dataset order, frozen query order, BM25 then independent hybrid",
        "timed": "tokenization, fresh query embedding for hybrid, scoring, ranking, RRF, final top20",
        "outside_query_timer": "imports, verification, model/index/vector load, warmup, comparison and result I/O",
        "no_qrels": True, "no_retries": True, "no_query_vector_cache": True,
        "sources_sha256": {p.name: sha256(p) for p in sorted((out / "source").iterdir())},
    })
    write_json(out / "summary.json", {"status": "partial", "complete": False, "stop_reason": "starting"})
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--source", str(source),
               "--out", str(out), "--max-seconds", str(args.max_seconds)]
    result = supervise(command, out, args.max_seconds)
    print(json.dumps({"out": str(out), "status": result["status"],
                      "completed_measurements": result["completed_measurements"]}))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
