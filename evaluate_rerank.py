"""Offline BEIR/TREC-compatible evaluation of frozen top-20 reranking runs.

No model is imported or called. Qrels are read only here, after inference.
Example: python evaluate_rerank.py --split pilot --run runs/pilot-bge-001
         --out runs/evaluation-pilot
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
METRICS = ("NDCG@10", "Recall@10", "Recall@20", "MRR@10")
BOOTSTRAP_PROTOCOL = {
    "metric": "NDCG@10",
    "comparison": "reranker_minus_retrieval_baseline",
    "method": "paired_percentile_bootstrap",
    "confidence_level": .95,
    "resampling_unit": "query",
    "interval_scope": "pointwise query-resampling uncertainty within this benchmark",
    "resamples": 2000,
    "seed": 20260927,
    "rng": "Python random.Random.choices (MT19937)",
    "query_order": "lexicographic_query_id",
    "percentile_method": "linear interpolation at (n-1)*p (Hyndman-Fan type 7)",
    "multiple_comparisons_adjusted": False,
}
SOURCES = {
    "beir_evaluation": "https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py",
    "trec_ndcg_gain": "https://github.com/usnistgov/trec_eval/blob/master/m_ndcg_cut.c",
    "beir_mrr": "https://github.com/beir-cellar/beir/blob/main/beir/retrieval/custom_metrics.py",
}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode())


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_jsonl_snapshot(path, *, tolerate_partial_tail=False):
    """Read one byte snapshot. A crashed final line is explicitly reported, never repaired."""
    path = Path(path)
    if not path.exists():
        return [], {"path": str(path), "exists": False, "sha256": None, "partial_tail": False}
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    result, partial = [], False
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if tolerate_partial_tail and number == len(lines) and not line.endswith(b"\n"):
                partial = True
                break
            raise ValueError(f"Invalid JSONL: {path}:{number}") from None
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object: {path}:{number}")
        result.append(row)
    return result, {"path": str(path), "exists": True, "sha256": sha_bytes(data),
                    "bytes": len(data), "partial_tail": partial,
                    "parsed_rows": len(result)}


def ranking_metrics(ranking, qrels):
    """Linear gains, all positive qrels in IDCG/recall denominator, unjudged=0.

    The caller supplies an already resolved order (including protocol tie breaks).
    Unlike BEIR's optional ID filter, an identical query/document ID is NOT removed:
    these BEIR datasets have separate ID namespaces, and candidates are frozen.
    This matches EvaluateRetrieval.evaluate(..., ignore_identical_ids=False).
    """
    if len(ranking) != len(set(ranking)):
        raise ValueError("Duplicate documents in ranking")
    gains = [max(0, qrels.get(doc, 0)) for doc in ranking]
    ideal = sorted((gain for gain in qrels.values() if gain > 0), reverse=True)[:10]
    dcg = math.fsum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains[:10]))
    idcg = math.fsum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    relevant = sum(gain > 0 for gain in qrels.values())
    return {"NDCG@10": dcg / idcg if idcg else 0.0,
            "Recall@10": sum(gain > 0 for gain in gains[:10]) / relevant if relevant else 0.0,
            "Recall@20": sum(gain > 0 for gain in gains[:20]) / relevant if relevant else 0.0,
            "MRR@10": next((1.0 / rank for rank, gain in enumerate(gains[:10], 1) if gain > 0), 0.0)}


def mean_metrics(rows):
    rows = list(rows)
    return {name: math.fsum(row[name] for row in rows) / len(rows) for name in METRICS} if rows else None


def percentile(values, p, *, allow_negative=False):
    """Linear interpolation at (n-1)*p (NumPy default / Hyndman-Fan type 7)."""
    values = sorted(values)
    if not values:
        return None
    if not 0 <= p <= 1 or any(not math.isfinite(v) or (v < 0 and not allow_negative) for v in values):
        raise ValueError("Invalid percentile input")
    position = (len(values) - 1) * p
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def paired_bootstrap_ndcg10(model_per_query, baseline_per_query):
    """Resample paired query differences, never individual documents or two runs independently.

    Every replicate samples n query IDs with replacement and averages the n
    model-minus-baseline differences. The fixed seed is reset for each group;
    sorting IDs makes results independent of JSONL/dictionary encounter order.
    A percentile interval is descriptive and has no multiplicity correction.
    """
    if not model_per_query or set(model_per_query) != set(baseline_per_query):
        raise ValueError("Paired bootstrap requires the same nonempty query ID sets")
    differences = []
    for qid in sorted(model_per_query):
        model = model_per_query[qid]["NDCG@10"]
        baseline = baseline_per_query[qid]["NDCG@10"]
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
               for value in (model, baseline)):
            raise ValueError("Paired bootstrap requires finite NDCG values in [0, 1]")
        differences.append(model - baseline)
    n = len(differences)
    rng = random.Random(BOOTSTRAP_PROTOCOL["seed"])
    estimates = [math.fsum(rng.choices(differences, k=n)) / n
                 for _ in range(BOOTSTRAP_PROTOCOL["resamples"])]
    return {**BOOTSTRAP_PROTOCOL, "n_queries": n,
            "constant_query_differences": len(set(differences)) == 1,
            "estimate": math.fsum(differences) / n,
            "ci95": {"lower": percentile(estimates, .025, allow_negative=True),
                     "upper": percentile(estimates, .975, allow_negative=True)}}


def latency_summary(rows):
    values = [r["elapsed_seconds"] for r in rows]
    return {"successful_lists": len(values), "p50_seconds": percentile(values, .5),
            "p95_seconds": percentile(values, .95), "total_successful_seconds": math.fsum(values),
            "includes_failed_attempt_time": False,
            "scope": "successful full top-20 rerank lists; warmup excluded; not retrieval/generation",
            "percentile_method": "linear interpolation at (n-1)*p (Hyndman-Fan type 7)"}


def read_qrels(path):
    data = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")), delimiter="\t")
    if reader.fieldnames != ["query-id", "corpus-id", "score"]:
        raise ValueError(f"Unexpected BEIR qrels header: {path}")
    qrels = {}
    for row in reader:
        if None in row or not row["query-id"] or not row["corpus-id"]:
            raise ValueError(f"Malformed qrel: {path}")
        try:
            gain = int(row["score"])
        except (ValueError, TypeError):
            raise ValueError(f"Qrels require integer relevance grades: {path}") from None
        query = qrels.setdefault(row["query-id"], {})
        if row["corpus-id"] in query:
            raise ValueError(f"Duplicate qrel: {path}/{row['query-id']}/{row['corpus-id']}")
        query[row["corpus-id"]] = gain
    return qrels, sha_bytes(data)


def validate_ranking(rows, *, original=None):
    if len(rows) != 20 or len({row["doc_id"] for row in rows}) != 20:
        raise ValueError("A ranking must contain exactly 20 distinct documents")
    if any(not isinstance(row["doc_id"], str) or not row["doc_id"] for row in rows):
        raise ValueError("Document IDs must be nonempty strings")
    if any(type(row["rank"]) is not int or row["rank"] != i for i, row in enumerate(rows, 1)):
        raise ValueError("Invalid ranking positions")
    if any(type(row["score"]) not in (int, float) or not math.isfinite(row["score"]) for row in rows):
        raise ValueError("Invalid ranking scores")
    if original is None:
        expected = sorted(rows, key=lambda row: (-row["score"], row["doc_id"]))
    else:
        source_ranks = {row["doc_id"]: row["rank"] for row in original}
        if {row["doc_id"] for row in rows} != set(source_ranks):
            raise ValueError("Reranked documents differ from frozen candidates")
        if any(type(row.get("source_rank")) is not int or
               row["source_rank"] != source_ranks[row["doc_id"]] for row in rows):
            raise ValueError("Changed source ranks")
        expected = sorted(rows, key=lambda row: (-row["score"], row["source_rank"], row["doc_id"]))
    if rows != expected:
        raise ValueError("Ranking violates frozen score/tie-break ordering")


@dataclass
class EvaluationSource:
    path: Path
    split: str
    summary_sha256: str
    artifacts_sha256: dict
    rankings: dict
    qrels: dict
    query_ids: dict
    qrels_provenance: dict


def load_source(source, data, split="pilot", *, dataset_manifests=None):
    if split not in {"pilot", "test"}:
        raise ValueError("split must be pilot or test")
    source, data = Path(source).resolve(), Path(data).resolve()
    summary_bytes = (source / "summary.json").read_bytes()
    summary, config = json.loads(summary_bytes), read_json(source / "config.json")
    if summary["config_sha256"] != fingerprint(config) or config["search"]["top_k"] != 20:
        raise ValueError("Changed retrieval configuration or unsupported candidate depth")
    checked_hashes, rankings, selected, qrels_by_dataset, provenance = {}, {}, {}, {}, {}

    def checked(relative):
        path = source / relative
        if not path.resolve().is_relative_to(source):
            raise ValueError("Candidate artifact escapes its run")
        raw = path.read_bytes()
        digest = sha_bytes(raw)
        if summary["artifacts_sha256"].get(relative) != digest:
            raise ValueError(f"Changed candidate artifact: {relative}")
        checked_hashes[relative] = digest
        return raw

    datasets = config["datasets"]
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("Missing or duplicate datasets")
    for dataset in datasets:
        if not isinstance(dataset, str) or Path(dataset).name != dataset or dataset in {".", ".."}:
            raise ValueError("Invalid dataset name")
        queries = json.loads(checked(f"{dataset}/queries.json"))
        qids = queries[split]
        if not qids or len(set(qids)) != len(qids) or any(not isinstance(q, str) or not q for q in qids):
            raise ValueError("Missing, duplicate or invalid query IDs")
        if set(queries["pilot"]) & set(queries["test"]):
            raise ValueError("Pilot/test query overlap")
        qrel_split = "test" if split == "test" else queries["pilot_source_split"]
        if split == "pilot" and qrel_split not in {"train", "dev"}:
            raise ValueError("Pilot qrels must come from train/dev, never test")
        qrel_path = data / dataset / "qrels" / f"{qrel_split}.tsv"
        qrels, digest = read_qrels(qrel_path)
        if not set(qids) <= set(qrels):
            raise ValueError(f"Selected queries missing qrels: {dataset}/{qrel_split}")
        if split == "test" and set(qids) != set(qrels):
            raise ValueError("Test evaluation must cover all test qrels queries")
        verified = False
        if dataset_manifests is not None:
            manifest = read_json(Path(dataset_manifests) / f"dataset-{dataset}.json")
            if manifest["files_sha256"].get(f"qrels/{qrel_split}.tsv") != digest:
                raise ValueError(f"Qrels differ from dataset manifest: {qrel_path}")
            verified = True
        selected[dataset] = qids
        qrels_by_dataset[dataset] = {q: qrels[q] for q in qids}
        provenance[dataset] = {"path": str(qrel_path), "sha256": digest, "split": qrel_split,
                               "dataset_manifest_verified": verified, "selected_queries": len(qids),
                               "queries_with_no_positive_qrels": sum(not any(g > 0 for g in qrels[q].values()) for q in qids)}
        for method in ("bm25", "rrf"):
            raw = checked(f"{dataset}/candidates.{split}.{method}.jsonl")
            rows = [json.loads(line) for line in raw.splitlines()]
            if len(rows) != len(qids) or {row["query_id"] for row in rows} != set(qids):
                raise ValueError("Missing, duplicate or unexpected candidate query list")
            for row in rows:
                validate_ranking(row["candidates"])
                rankings[(dataset, method, row["query_id"])] = row["candidates"]
    return EvaluationSource(source, split, sha_bytes(summary_bytes), checked_hashes,
                            rankings, qrels_by_dataset, selected, provenance)


def source_metrics(source, dataset, method, *, query_ids=None, oracle=False):
    values = {}
    for qid in source.query_ids[dataset] if query_ids is None else query_ids:
        docs = [row["doc_id"] for row in source.rankings[(dataset, method, qid)]]
        qrels = source.qrels[dataset][qid]
        if oracle:
            docs.sort(key=lambda doc: -max(0, qrels.get(doc, 0)))
        values[qid] = ranking_metrics(docs, qrels)
    return values


def baseline_rows(source):
    result = []
    for dataset in source.query_ids:
        for method in ("bm25", "rrf"):
            for oracle in (False, True):
                per_query = source_metrics(source, dataset, method, oracle=oracle)
                result.append({"dataset": dataset, "method": method,
                               "model": "candidate_oracle" if oracle else "retrieval_baseline",
                               "kind": "oracle" if oracle else "baseline", "split": source.split,
                               "complete": True, "queries": len(per_query),
                               "metrics": mean_metrics(per_query.values()), "per_query": per_query,
                               "interpretation": "Upper bound from sorting only these 20 candidates by qrels; not a model" if oracle
                               else "Frozen retrieval order before reranking"})
    return result


def parse_measure_job(job_id):
    parts = job_id.split("/", 4)
    if len(parts) != 5 or parts[0] != "measure" or not parts[1].isdigit():
        raise ValueError(f"Invalid measurement job ID: {job_id}")
    return int(parts[1]), tuple(parts[2:])


def attempt_stats(events, rows, job_ids):
    """Separate successful rows, recorded errors and attempts with no durable result."""
    selected = [e for e in events if e.get("job_id") in job_ids]
    starts = {(e["job_id"], e["attempt"]) for e in selected if e["event"] == "list_start"}
    errors = [e for e in selected if e["event"] == "list_error"]
    error_keys = {(e["job_id"], e["attempt"]) for e in errors}
    durable = {(r["job_id"], r["attempt"]) for r in rows if r["job_id"] in job_ids}
    seconds = [e["seconds"] for e in errors if type(e.get("seconds")) in (int, float)
               and math.isfinite(e["seconds"]) and e["seconds"] >= 0]
    return {"started_attempts": len(starts), "successful_durable_attempts": len(durable),
            "recorded_error_attempts": len(errors),
            "recorded_error_rate": len(errors) / len(starts) if starts else None,
            "error_types": dict(sorted(Counter(e.get("error_type", "unspecified") for e in errors).items())),
            "recorded_error_seconds": math.fsum(seconds),
            "unfinished_attempts": len(starts - error_keys - durable),
            "retried_jobs": sum(count > 1 for count in Counter(job for job, _ in starts).values()),
            "missing_start_records": len(durable - starts),
            "note": "Unfinished means no recorded terminal result; it may be active or interrupted."}


def evaluate_run(source, run):
    run = Path(run).resolve()
    if run.name == "lists.jsonl":
        run = run.parent
    manifest_path = run / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    identity = json.loads(manifest_bytes)["identity"]
    config = identity["config"]
    if identity["split"] != source.split:
        raise ValueError(f"Run split mismatch: {run}")
    if identity["source"]["retrieval_summary_sha256"] != source.summary_sha256:
        raise ValueError(f"Run belongs to another retrieval source: {run}")
    if identity["source"].get("split") != source.split:
        raise ValueError("Run source split mismatch")
    if any(identity["source"]["selected_artifacts_sha256"].get(k) != v
           for k, v in source.artifacts_sha256.items()):
        raise ValueError("Run candidate hashes mismatch")
    repeats, warmups = config["repeats"], config["warmup_lists"]
    if type(repeats) is not int or repeats < 1 or type(warmups) is not int or warmups < 1:
        raise ValueError("Invalid repeat/warmup counts")
    expected = {f"measure/{repeat}/{dataset}/{method}/{qid}"
                for repeat in range(repeats) for dataset, method, qid in source.rankings}
    planned = expected | {f"warmup/{i}" for i in range(warmups)}
    rows, results_snapshot = read_jsonl_snapshot(run / "lists.jsonl", tolerate_partial_tail=True)
    events, events_snapshot = read_jsonl_snapshot(run / "events.jsonl", tolerate_partial_tail=True)
    by_id = {}
    measured = []
    for row in rows:
        job = row["job_id"]
        if job in by_id:
            raise ValueError("Duplicate completed job (do not select best attempt)")
        by_id[job] = row
        if type(row.get("attempt")) is not int or row["attempt"] < 1:
            raise ValueError("Invalid result attempt")
        if type(row.get("elapsed_seconds")) not in (int, float) or not math.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"] < 0:
            raise ValueError("Invalid elapsed time")
        if job not in planned and not job.startswith("resume_warmup/"):
            raise ValueError(f"Unexpected completed job: {job}")
        if job in expected:
            _, key = parse_measure_job(job)
            if row["phase"] != "measure" or key != (row["dataset"], row["method"], row["query_id"]):
                raise ValueError("Measurement metadata/job mismatch")
            validate_ranking(row["ranking"], original=source.rankings[key])
            measured.append(row)
        elif row["phase"] != "warmup":
            raise ValueError("Warmup mislabeled as a measurement")
    event_starts, event_terminals = set(), set()
    for event in events:
        if event.get("event") in {"list_start", "list_success", "list_error"}:
            job, attempt = event["job_id"], event["attempt"]
            if job not in planned and not job.startswith("resume_warmup/"):
                raise ValueError(f"Unexpected event job: {job}")
            if type(attempt) is not int or attempt < 1:
                raise ValueError("Invalid event attempt")
            if event["event"] == "list_start":
                if (job, attempt) in event_starts:
                    raise ValueError("Duplicate attempt start")
                event_starts.add((job, attempt))
            else:
                terminal = (job, attempt, event["event"])
                if terminal in event_terminals:
                    raise ValueError("Duplicate terminal attempt event")
                if (job, attempt) not in event_starts:
                    raise ValueError("Terminal attempt event has no preceding start")
                event_terminals.add(terminal)
                if event["event"] == "list_error" and (
                        type(event.get("seconds")) not in (int, float)
                        or not math.isfinite(event["seconds"]) or event["seconds"] < 0):
                    raise ValueError("Invalid failed-attempt elapsed time")
    partial = results_snapshot["partial_tail"] or events_snapshot["partial_tail"]
    summary_path = run / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else None
    complete = False
    if summary is not None:
        if (summary.get("complete") is not True or summary.get("split") != source.split
                or summary.get("results_sha256") != results_snapshot["sha256"]
                or summary.get("events_sha256") != events_snapshot["sha256"]
                or summary.get("lists") != len(expected) or not planned <= by_id.keys() or partial):
            raise ValueError("Completion summary does not match durable complete results")
        complete = True
    if any((r["job_id"], r["attempt"]) not in event_starts for r in rows):
        if complete:
            raise ValueError("Complete run has results without attempt starts")
    measured_stats = attempt_stats(events, rows, expected)
    warmup_ids = {r["job_id"] for r in rows if r["phase"] == "warmup"}
    warmup_ids |= {e["job_id"] for e in events if e.get("job_id", "").startswith(("warmup/", "resume_warmup/"))}
    state = "complete" if complete else "incomplete"
    model_config = config.get("model_config", {})
    model_info = {k: model_config[k] for k in ("model", "repo", "revision", "dtype", "max_length", "device", "batch_size") if k in model_config}
    model_name = model_config.get("repo") or model_config.get("model") or config["model"]
    info = {"run": str(run), "model": config["model"], "model_name": model_name, "model_info": model_info, "status": state,
            "complete": complete, "split": source.split, "manifest_sha256": sha_bytes(manifest_bytes),
            "expected_measurements": len(expected), "completed_measurements": len(measured),
            "completion_fraction": len(measured) / len(expected), "repeats": repeats,
            "quality_repeat": 0, "summary_present": summary is not None,
            "snapshots": {"lists": results_snapshot, "events": events_snapshot},
            "errors": measured_stats, "warmup_errors": attempt_stats(events, rows, warmup_ids),
            "latency": latency_summary(measured)}
    groups = []
    for dataset in source.query_ids:
        for method in ("bm25", "rrf"):
            selected = [r for r in measured if (r["dataset"], r["method"]) == (dataset, method)]
            first = [r for r in selected if parse_measure_job(r["job_id"])[0] == 0]
            per_query = {r["query_id"]: ranking_metrics([d["doc_id"] for d in r["ranking"]],
                         source.qrels[dataset][r["query_id"]]) for r in first}
            job_ids = {job for job in expected if parse_measure_job(job)[1][:2] == (dataset, method)}
            qids = source.query_ids[dataset]
            quality_complete = len(per_query) == len(qids)
            values = mean_metrics(per_query.values())
            matched_baseline = source_metrics(source, dataset, method, query_ids=per_query)
            paired_bootstrap = paired_bootstrap_ndcg10(per_query, matched_baseline) if complete and quality_complete else None
            first_orders = {r["query_id"]: [d["doc_id"] for d in r["ranking"]] for r in first}
            changed = sum(parse_measure_job(r["job_id"])[0] != 0 and r["query_id"] in first_orders and
                          [d["doc_id"] for d in r["ranking"]] != first_orders[r["query_id"]] for r in selected)
            groups.append({"dataset": dataset, "method": method, "model": config["model"],
                           "model_name": model_name,
                           "kind": "reranker", "run": str(run), "split": source.split,
                           "complete": complete and quality_complete,
                           "run_complete": complete, "quality_query_coverage_complete": quality_complete,
                           "queries": len(qids), "quality_queries_completed": len(per_query),
                           "quality_repeat": 0, "expected_measurements": len(job_ids),
                           "completed_measurements": len(selected),
                           "missing_quality_query_ids": [q for q in qids if q not in per_query],
                           "metrics": values if complete and quality_complete else None,
                           "paired_bootstrap_ndcg10_vs_baseline": paired_bootstrap,
                           "completed_only_diagnostic_metrics": values,
                           "matched_baseline_diagnostic_metrics": mean_metrics(matched_baseline.values()),
                           "matched_oracle_diagnostic_metrics": mean_metrics(source_metrics(source, dataset, method, query_ids=per_query, oracle=True).values()),
                           "per_query_completed": per_query,
                           "ranking_repeat_zero": first_orders,
                           "later_repeat_rankings_differing_from_repeat_zero": changed,
                           "latency": latency_summary(selected), "errors": attempt_stats(events, rows, job_ids)})
    return info, groups


def cross_check_pytrec(source, rerankers):
    """Optional external check; no dependency installation, network or score reties.

    Strictly descending synthetic scores preserve the frozen ordering. MRR@10 is
    checked using trec_eval recip_rank on rankings truncated to 10 documents.
    """
    try:
        import pytrec_eval
    except ImportError:
        return {"status": "not_available", "checked_rankings": 0,
                "note": "Install pytrec-eval-terrier in a separate environment to enable the optional check."}
    cases = []
    for key, ranking in source.rankings.items():
        dataset, method, qid = key
        docs = [d["doc_id"] for d in ranking]
        cases.append((dataset, qid, docs))
        cases.append((dataset, qid, sorted(docs, key=lambda d: -max(0, source.qrels[dataset][qid].get(d, 0)))))
    # Use the already validated snapshot, never re-read a possibly active run.
    for group in rerankers:
        for qid, docs in group["ranking_repeat_zero"].items():
            cases.append((group["dataset"], qid, docs))
    max_difference = 0.0
    for dataset, qid, docs in cases:
        qrels = {qid: source.qrels[dataset][qid]}
        scored = {qid: {doc: float(len(docs) - rank) for rank, doc in enumerate(docs)}}
        ref = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.10,20"}).evaluate(scored)[qid]
        short = {qid: {doc: float(10 - rank) for rank, doc in enumerate(docs[:10])}}
        rr = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"}).evaluate(short)[qid]["recip_rank"]
        got = ranking_metrics(docs, qrels[qid])
        for name, want in zip(METRICS, (ref["ndcg_cut_10"], ref["recall_10"], ref["recall_20"], rr), strict=True):
            delta = abs(got[name] - want)
            max_difference = max(max_difference, delta)
            if delta > 1e-12:
                raise ValueError(f"pytrec_eval mismatch: {dataset}/{qid}/{name}: {got[name]} != {want}")
    versions = {}
    for package in ("pytrec-eval", "pytrec-eval-terrier"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"status": "passed", "checked_rankings": len(cases), "max_absolute_difference": max_difference,
            "versions": versions, "score_policy": "strictly descending rank-surrogate scores preserve protocol tie breaks"}


def make_report(source, runs=(), *, cross_check=False):
    baseline = baseline_rows(source)
    result = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "evaluator_sha256": sha_bytes(Path(__file__).read_bytes()),
              "split": source.split, "source": str(source.path),
              "retrieval_summary_sha256": source.summary_sha256,
              "candidate_artifacts_sha256": source.artifacts_sha256,
              "qrels": source.qrels_provenance, "qrels_sent_to_models": False,
              "protocol": {"gain": "max(qrel, 0), linear, not 2**rel-1", "discount": "log2(rank+1)",
                           "relevant": "qrel > 0", "unjudged_gain": 0, "identical_ids_removed": False,
                           "metric_compatibility": "BEIR with ignore_identical_ids=False; no rounding before aggregation",
                           "aggregation": "macro mean over selected queries, separately for each dataset/method/model/run",
                           "quality_repeat": "repeat 0, predeclared; later repeats used only for latency/stability",
                           "paired_bootstrap_ndcg10_vs_baseline": {**BOOTSTRAP_PROTOCOL,
                               "completeness_policy": "only complete runs with every selected query; otherwise null",
                               "grouping": "separate dataset/retrieval method/model/run; paired to that method's baseline"},
                           "incomplete_policy": "headline metrics are null until a valid completion summary; completed-only diagnostics are separately labelled",
                           "oracle": "Qrels-sort within frozen top-20, ideal DCG from all positive qrels; candidate Recall@20 cannot change under reranking",
                           "sources": SOURCES, "sources_checked_date": "2026-09-27"},
              "baseline_and_oracle": baseline, "runs": [], "rerankers": []}
    paths = [Path(path).resolve().parent if Path(path).name == "lists.jsonl" else Path(path).resolve() for path in runs]
    if len(set(paths)) != len(paths):
        raise ValueError("The same run was supplied more than once")
    for path in paths:
        info, groups = evaluate_run(source, path)
        result["runs"].append(info)
        result["rerankers"].extend(groups)
    result["pytrec_eval_cross_check"] = cross_check_pytrec(source, result["rerankers"]) if cross_check else {"status": "not_requested"}
    return result


def render_markdown(report):
    def metric(value):
        return "-" if value is None else f"{value:.5f}"

    lines = [f"# Качество реранжирования: {report['split']}", "",
             "NDCG@10 использует линейные gains qrel и log2(rank+1). Recall@10, Recall@20 и MRR@10: релевантность > 0.",
             "Среднее по запросам отдельно для каждого набора. Качество - только повтор 0; прогрев исключен.",
             "Oracle сортирует только общие 20 кандидатов по разметке и не является моделью. Его Recall@20 - полнота пула кандидатов.", "",
             "| Набор | Поиск | Модель | NDCG@10 | Recall@10 | Recall@20 | MRR@10 | Запросы | Статус |",
             "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in report["baseline_and_oracle"] + report["rerankers"]:
        values = row["metrics"] or {name: None for name in METRICS}
        done = row.get("quality_queries_completed", row["queries"])
        lines.append(f"| {row['dataset']} | {row['method']} | {row.get('model_name', row['model'])} | " +
                     " | ".join(metric(values[name]) for name in METRICS) +
                     f" | {done}/{row['queries']} | {'готово' if row['complete'] else 'НЕ ЗАВЕРШЕН'} |")
    lines += ["", "У незавершенных прогонов итоговые метрики и доверительные интервалы не публикуются. Диагностика по готовым запросам, парный baseline и пропущенные ID сохранены в JSON.", "",
              "## Изменение NDCG@10 относительно исходного поиска", "",
              "Разница = реранкер минус исходный порядок того же поиска. Положительное значение означает рост NDCG@10.",
              "Парный bootstrap: запросы выбираются с возвращением, 2000 выборок исходного размера, seed=20260927. ID упорядочены лексикографически; seed сбрасывается для каждой группы.",
              "95% percentile CI: квантили 2.5% и 97.5% с линейной интерполяцией. Поправка на множественные сравнения не применяется.", "",
              "Интервалы отражают пересэмплирование запросов данного набора. При малом числе запросов они могут быть грубыми или вырожденными.", "",
              "| Набор | Поиск | Модель | Разница NDCG@10 | 95% CI | Запросы |",
              "|---|---|---|---:|---|---:|"]
    for row in report["rerankers"]:
        bootstrap = row["paired_bootstrap_ndcg10_vs_baseline"]
        estimate = metric(bootstrap["estimate"]) if bootstrap else "-"
        interval = f"[{metric(bootstrap['ci95']['lower'])}, {metric(bootstrap['ci95']['upper'])}]" if bootstrap else "-"
        count = str(bootstrap["n_queries"]) if bootstrap else "-"
        lines.append(f"| {row['dataset']} | {row['method']} | {row.get('model_name', row['model'])} | {estimate} | {interval} | {count} |")
    lines += ["", "## Задержка и ошибки", "",
              "P50/P95 в секундах на успешный полный список из 20 документов; все измерительные повторы. Не включают прогрев, ошибки, первичный поиск и генерацию.",
              "Квантили: линейная интерполяция (n-1)*p. Отдельно сохранено время завершившихся ошибкой попыток.", "",
              "| Набор | Поиск | Модель | Готовые списки | P50, с | P95, с | Ошибки / попытки | Незавершенные попытки |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in report["rerankers"]:
        latency, errors = row["latency"], row["errors"]
        lines.append(f"| {row['dataset']} | {row['method']} | {row.get('model_name', row['model'])} | "
                     f"{row['completed_measurements']}/{row['expected_measurements']} | "
                     f"{metric(latency['p50_seconds'])} | {metric(latency['p95_seconds'])} | "
                     f"{errors['recorded_error_attempts']}/{errors['started_attempts']} | {errors['unfinished_attempts']} |")
    lines += ["", "## Происхождение", ""]
    for dataset, info in report["qrels"].items():
        lines.append(f"- {dataset}: qrels `{info['split']}`, {info['selected_queries']} запросов, SHA-256 `{info['sha256']}`.")
    for run in report["runs"]:
        lines.append(f"- {run['model_name']}: `{run['run']}` - {run['status']}, {run['completed_measurements']}/{run['expected_measurements']} списков; ошибки прогрева: {run['warmup_errors']['recorded_error_attempts']}.")
    lines += [f"- Независимая сверка pytrec_eval: {report['pytrec_eval_cross_check']['status']}.", "",
              "Соглашения проверены по [BEIR](" + SOURCES["beir_evaluation"] + "), [NIST trec_eval](" + SOURCES["trec_ndcg_gain"] + ") и [BEIR MRR](" + SOURCES["beir_mrr"] + ").", ""]
    return "\n".join(lines)


def write_report(report, out, *, overwrite=False):
    out = Path(out)
    targets = [out / name for name in ("report.json", "report.md")]
    if not overwrite and any(path.exists() for path in targets):
        raise ValueError("Report already exists; choose a fresh --out or explicitly pass --overwrite")
    if any((out / name).exists() for name in ("manifest.json", "candidates.test.bm25.jsonl", "summary.json")):
        raise ValueError("Use a separate evaluation directory, not a source/run artifact directory")
    out.mkdir(parents=True, exist_ok=True)
    contents = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", render_markdown(report))
    for path, content in zip(targets, contents, strict=True):
        temporary = path.with_name(path.name + ".tmp")
        # Refuse a stale/foreign temporary target rather than overwrite it.
        with temporary.open("x") as stream:
            stream.write(content)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/retrieval-20260927")
    parser.add_argument("--data", type=Path, default=ROOT / "data/beir")
    parser.add_argument("--split", choices=("pilot", "test"), default="pilot")
    parser.add_argument("--run", type=Path, action="append", default=[], help="Run directory or its lists.jsonl; repeat for multiple runs")
    parser.add_argument("--dataset-manifests", type=Path, help="Optional dataset-*.json hashes; defaults to assets/manifests when present")
    parser.add_argument("--cross-check", action="store_true", help="Cross-check with pytrec_eval if installed; does not install anything")
    parser.add_argument("--out", type=Path, required=True, help="Directory for report.json and report.md")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing evaluation reports")
    args = parser.parse_args()
    manifests = args.dataset_manifests
    if manifests is None and args.data.resolve() == (ROOT / "data/beir").resolve() and (ROOT / "assets/manifests").is_dir():
        manifests = ROOT / "assets/manifests"
    source = load_source(args.source, args.data, args.split, dataset_manifests=manifests)
    report = make_report(source, args.run, cross_check=args.cross_check)
    write_report(report, args.out, overwrite=args.overwrite)
    print(json.dumps({"report": str((args.out / "report.json").resolve()), "split": source.split,
                      "runs": [{k: row[k] for k in ("run", "model", "complete", "completed_measurements", "expected_measurements")} for row in report["runs"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
