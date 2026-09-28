#!/usr/bin/env python3
"""Offline audit of the public projections, with optional original BEIR qrels.

Requires only Python's standard library. Does not call a model or download data.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

METRICS = ("NDCG@10", "Recall@10", "Recall@20", "MRR@10")
DATASETS = ("scifact", "fiqa", "nfcorpus")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), f"number mismatch: {actual} != {expected}")


def read(path):
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    if ".jsonl" in path.name:
        return [json.loads(line) for line in data.splitlines() if line]
    return json.loads(data)


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def metrics(ids, qrels):
    gains = [max(qrels.get(doc_id, 0), 0) for doc_id in ids]
    ideal = sorted((grade for grade in qrels.values() if grade > 0), reverse=True)[:10]
    dcg = math.fsum(grade / math.log2(rank + 2) for rank, grade in enumerate(gains[:10]))
    idcg = math.fsum(grade / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    relevant = sum(grade > 0 for grade in qrels.values())
    return {"NDCG@10": dcg / idcg if idcg else 0.0,
            "Recall@10": sum(grade > 0 for grade in gains[:10]) / relevant if relevant else 0.0,
            "Recall@20": sum(grade > 0 for grade in gains[:20]) / relevant if relevant else 0.0,
            "MRR@10": next((1 / rank for rank, grade in enumerate(gains[:10], 1) if grade > 0), 0.0)}


def verify(directory, data_dir=None, charts=None):
    manifest = read(directory / "manifest.json")
    for name, expected in manifest["public_artifacts"].items():
        payload = (directory / name).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == expected["sha256"], f"checksum mismatch: {name}")
        require(len(payload) == expected["bytes"], f"size mismatch: {name}")
    quality = read(directory / "quality.json.gz")
    candidates = {}
    for dataset in DATASETS:
        for split in ("pilot", "test"):
            for method in ("bm25", "rrf"):
                rows = read(directory / f"candidates/{dataset}.{split}.{method}.jsonl.gz")
                require(len({row["query_id"] for row in rows}) == len(rows), "duplicate candidate query")
                for row in rows:
                    items = row["candidates"]
                    require(len(items) == len({item["doc_id"] for item in items}) == 20, "candidate count")
                    require([item["rank"] for item in items] == list(range(1, 21)), "candidate ranks")
                    require(items == sorted(items, key=lambda item: (-item["score"], item["doc_id"])), "candidate tie break")
                    candidates[dataset, split, method, row["query_id"]] = items
    lists, measured_groups = {}, defaultdict(list)
    memory = {row["model"]: row for row in read(directory / "memory.json")["models"]}
    for run in quality["runs"]:
        model = run["model"]
        rows = read(directory / f"main/{model}.jsonl.gz")
        require(len(rows) == 2544, "main run must contain 2542 measured and 2 warmup lists")
        require(len({row["job_id"] for row in rows}) == len(rows), "duplicate job")
        measured = []
        for row in rows:
            require(row["attempt"] == 1, "unexpected attempt")
            require(row["phase"] in ("measure", "warmup"), "unexpected phase")
            split = "test" if row["phase"] == "measure" else "pilot"
            original = candidates[row["dataset"], split, row["method"], row["query_id"]]
            source_ranks = {item["doc_id"]: item["rank"] for item in original}
            ranking = row["ranking"]
            require(len(ranking) == len({item["doc_id"] for item in ranking}) == 20, "ranking count")
            require({item["doc_id"] for item in ranking} == set(source_ranks), "candidate membership")
            require([item["rank"] for item in ranking] == list(range(1, 21)), "ranks")
            require(all(item["source_rank"] == source_ranks[item["doc_id"]] for item in ranking), "source ranks")
            require(all(math.isfinite(item["score"]) for item in ranking), "nonfinite score")
            require(ranking == sorted(ranking, key=lambda item: (-item["score"], item["source_rank"], item["doc_id"])), "score order or tie break")
            if row["phase"] == "measure":
                measured.append(row)
                key = (model, row["dataset"], row["method"])
                measured_groups[key].append(row)
                lists[(*key, row["query_id"])] = row
        require(len(measured) == run["completed_measurements"] == 2542, "measurement count")
        require(run["errors"]["recorded_error_attempts"] == run["errors"]["retried_jobs"] == run["errors"]["unfinished_attempts"] == 0, "historical run error summary")
        check_latency(measured, run["latency"])
        if model in memory:
            for field in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
                require(max(row["memory"][field] for row in measured) == memory[model][field], "CUDA memory peak")

    qrels = {}
    if data_dir:
        for dataset in DATASETS:
            path = data_dir / dataset / "qrels/test.tsv"
            require(hashlib.sha256(path.read_bytes()).hexdigest() == quality["qrels"][dataset]["sha256"], f"qrels checksum: {dataset}")
            grades = defaultdict(dict)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    require(row["corpus-id"] not in grades[row["query-id"]], "duplicate qrel")
                    grades[row["query-id"]][row["corpus-id"]] = int(row["score"])
            qrels[dataset] = grades
    baselines = {(row["dataset"], row["method"]): row["per_query"] for row in quality["baseline_and_oracle"] if row["kind"] == "baseline"}
    metric_rankings = 0
    for row in quality["baseline_and_oracle"] + quality["rerankers"]:
        per_query = row.get("per_query", row.get("per_query_completed"))
        require(len(per_query) == row["queries"], "quality query count")
        for metric in METRICS:
            close(math.fsum(value[metric] for value in per_query.values()) / len(per_query), row["metrics"][metric])
        if row["kind"] not in ("baseline", "oracle"):
            group = measured_groups[row["model"], row["dataset"], row["method"]]
            require(set(per_query) == {item["query_id"] for item in group}, "quality query coverage")
            check_latency(group, row["latency"])
            bootstrap = row["paired_bootstrap_ndcg10_vs_baseline"]
            baseline = baselines[row["dataset"], row["method"]]
            differences = [per_query[query]["NDCG@10"] - baseline[query]["NDCG@10"] for query in sorted(per_query)]
            rng = random.Random(bootstrap["seed"])
            estimates = [math.fsum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(bootstrap["resamples"])]
            close(math.fsum(differences) / len(differences), bootstrap["estimate"])
            close(percentile(estimates, 0.025), bootstrap["ci95"]["lower"])
            close(percentile(estimates, 0.975), bootstrap["ci95"]["upper"])
        if data_dir:
            for query, expected in per_query.items():
                grades = qrels[row["dataset"]][query]
                if row["kind"] in ("baseline", "oracle"):
                    ranking = candidates[row["dataset"], "test", row["method"], query]
                    if row["kind"] == "oracle":
                        ranking = sorted(ranking, key=lambda item: -max(grades.get(item["doc_id"], 0), 0))
                else:
                    ranking = lists[row["model"], row["dataset"], row["method"], query]["ranking"]
                actual = metrics([item["doc_id"] for item in ranking], grades)
                for metric in METRICS:
                    close(actual[metric], expected[metric])
                metric_rankings += 1

    for row in read(directory / "throughput.json")["models"]:
        require(sum(item["pairs"] for item in row["measurements"]) == row["measured_pairs"] == 1440, "throughput pairs")
        close(math.fsum(item["seconds"] for item in row["measurements"]), row["seconds"])
        close(row["measured_pairs"] / row["seconds"], row["pairs_per_second"])
        for item in row["measurements"]:
            close(item["pairs"] / item["seconds"], item["pairs_per_second"])
    cost = read(directory / "api-cost.json")
    require(sum(Decimal(row["all"]["cost_usd"]) for row in cost["runs"].values()) == Decimal(cost["combined_cost_usd"]), "combined API cost")
    require(sum(row["all"]["requests"] for row in cost["runs"].values()) == cost["combined_paid_requests"], "combined API requests")
    for row in cost["runs"].values():
        for field in ("requests", "lists", "input_tokens", "output_tokens"):
            require(row["measure"][field] + row["warmup"][field] == row["all"][field], "API usage count")
        require(Decimal(row["measure"]["cost_usd"]) + Decimal(row["warmup"]["cost_usd"]) == Decimal(row["all"]["cost_usd"]), "API phase cost")
    cpu = read(directory / "cpu-retrieval.json")
    cpu_groups = defaultdict(list)
    for row in read(directory / "cpu-retrieval.jsonl.gz"):
        cpu_groups[row["dataset"], row["method"]].append(row["seconds"])
    for (dataset, method), seconds in cpu_groups.items():
        expected = cpu["datasets"][dataset][method]
        require(len(seconds) == expected["n"], "CPU measurement count")
        close(percentile(seconds, 0.5) * 1000, expected["p50_ms"])
        close(percentile(seconds, 0.95) * 1000, expected["p95_ms"])
        close(math.fsum(seconds), expected["total_seconds"])
    if charts:
        chart = read(charts)
        for row in chart["quality"]:
            close(next(item["metrics"]["NDCG@10"] for item in quality["baseline_and_oracle"] if item["kind"] == "baseline" and item["dataset"] == row["dataset"] and item["method"] == row["method"]), row["baseline_ndcg10"])
            for model, value in row["models"].items():
                close(next(item["metrics"]["NDCG@10"] for item in quality["rerankers"] if item["model"] == model and item["dataset"] == row["dataset"] and item["method"] == row["method"]), value["ndcg10"])
        for row in chart["latency"]:
            run = next(item for item in quality["runs"] if item["model"] == row["model"])
            close(run["latency"]["p50_seconds"] * 1000, row["p50_ms"])
            close(run["latency"]["p95_seconds"] * 1000, row["p95_ms"])
        for row in chart["memory"]:
            close(memory[row["model"]]["peak_cuda_allocated_bytes"] / 2**30, row["allocated_gib"])
            close(memory[row["model"]]["peak_cuda_reserved_bytes"] / 2**30, row["reserved_gib"])
        for row in chart["throughput"]:
            close(next(item["pairs_per_second"] for item in read(directory / "throughput.json")["models"] if item["alias"] == row["model"]), row["pairs_per_second"])
    print(f"Verified {len(manifest['public_artifacts'])} artifact hashes, 12720 reranked lists (including 10 warmup), 30 bootstrap intervals, latency, memory, throughput and API cost totals.")
    print(f"Recomputed {metric_rankings} metric rankings against checksum-matched BEIR qrels." if data_dir else "Qrels not provided: independently recomputing relevance metrics was skipped; saved per-query aggregates were verified.")
    if charts:
        print("Chart quality, latency, memory and throughput values match the public projections.")


def check_latency(rows, expected):
    seconds = [row["elapsed_seconds"] for row in rows]
    require(len(seconds) == expected["successful_lists"], "latency count")
    close(percentile(seconds, 0.5), expected["p50_seconds"])
    close(percentile(seconds, 0.95), expected["p95_seconds"])
    close(math.fsum(seconds), expected["total_successful_seconds"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--data-dir", type=Path, help="BEIR directory containing <dataset>/qrels/test.tsv")
    parser.add_argument("--charts", type=Path, help="optional benchmark-charts.data.json to cross-check")
    args = parser.parse_args()
    verify(args.results, args.data_dir, args.charts)
