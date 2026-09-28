#!/usr/bin/env python3
"""Project reviewed numerical fields from the private historical archive.

No network access or model calls. Explicit field allowlists, never recursive copies
of manifests/events. Run only against the original local archive, then audit the
output before publishing. Public files are projections, not original raw logs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

DATASETS = ("scifact", "fiqa", "nfcorpus")
GPU_MODELS = ("laya-multi", "bge", "qwen-0.6b", "qwen-8b")
MODELS = (*GPU_MODELS, "jev-openrouter")
METRICS = ("NDCG@10", "Recall@10", "Recall@20", "MRR@10")
GPU_RUN = "runs/vast-20260927T182111Z"
JEV_RUN = "runs/jev-openrouter-20260927"
THROUGHPUT_RUN = "runs/vast-throughput-20260927T195100Z"
LATENCY_FIELDS = "successful_lists p50_seconds p95_seconds total_successful_seconds includes_failed_attempt_time scope percentile_method"
ERROR_FIELDS = "started_attempts successful_durable_attempts recorded_error_attempts recorded_error_rate recorded_error_seconds unfinished_attempts retried_jobs missing_start_records"
MEMORY_FIELDS = "peak_cuda_allocated_bytes peak_cuda_reserved_bytes"
COST_FIELDS = "lists requests input_tokens output_tokens cost_usd cost_per_1000_pair_requests_usd cost_per_1000_top20_lists_usd p50_ms p95_ms sum_list_seconds mean_list_ms"
MODEL_FIELDS = "repo revision dtype max_length device batch_size model"
ADAPTER_FIELDS = "alias attn_implementation code_revision device dtype fallback instruction max_len_override max_length offline precision_mode repo resident_weights_dtype revision score_type torch transformers truncation adapter concurrency context_tokens endpoint expected_response_model input_price_per_million_usd model output_price_per_million_usd primitive question question_id redirects retries timeout_s"
BOOTSTRAP_FIELDS = "metric comparison method confidence_level resampling_unit interval_scope resamples seed rng query_order percentile_method multiple_comparisons_adjusted n_queries constant_query_differences estimate"


def pick(value, fields):
    """Missing optional keys stay absent, and unlisted keys never escape."""
    if isinstance(fields, str):
        fields = fields.split()
    return {key: value[key] for key in fields if key in value}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def export(source, output):
    output.mkdir(parents=True, exist_ok=True)
    sources, artifacts = {}, {}

    def record(path, label):
        sources[label] = sha256(path)
        return path

    def write(name, value, jsonl=False):
        payload = (("\n".join(encode(row) for row in value) if jsonl else encode(value)) + "\n").encode()
        if name.endswith(".gz"):
            # Empty original filename, zero timestamp: byte-identical exports.
            payload = gzip.compress(payload, compresslevel=9, mtime=0)
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts[name] = {"sha256": sha256(path), "bytes": len(payload)}

    report = read_json(record(source / JEV_RUN / "evaluation-main-all-five/report.json", "evaluation"))
    quality = pick(report, "schema_version split qrels_sent_to_models")
    quality["projection"] = "reviewed public numerical fields; originals retained privately"
    quality["protocol"] = pick(report["protocol"], "gain discount relevant unjudged_gain identical_ids_removed metric_compatibility aggregation quality_repeat incomplete_policy oracle")
    quality["protocol"]["paired_bootstrap_ndcg10_vs_baseline"] = pick(report["protocol"]["paired_bootstrap_ndcg10_vs_baseline"], BOOTSTRAP_FIELDS)
    quality["pytrec_eval_cross_check"] = pick(report["pytrec_eval_cross_check"], "status checked_rankings max_absolute_difference score_policy")
    quality["pytrec_eval_cross_check"]["versions"] = pick(report["pytrec_eval_cross_check"]["versions"], "pytrec-eval-terrier")
    quality["qrels"] = {dataset: pick(report["qrels"][dataset], "sha256 split selected_queries queries_with_no_positive_qrels dataset_manifest_verified") for dataset in DATASETS}
    quality["baseline_and_oracle"] = []
    for row in report["baseline_and_oracle"]:
        public = pick(row, "complete dataset kind method model queries split")
        public["metrics"] = pick(row["metrics"], METRICS)
        public["per_query"] = {query: pick(metrics, METRICS) for query, metrics in row["per_query"].items()}
        quality["baseline_and_oracle"].append(public)
    quality["rerankers"] = []
    for row in report["rerankers"]:
        public = pick(row, "complete completed_measurements dataset expected_measurements kind method model model_name quality_queries_completed quality_query_coverage_complete quality_repeat queries run_complete split later_repeat_rankings_differing_from_repeat_zero")
        public["metrics"] = pick(row["metrics"], METRICS)
        public["latency"] = pick(row["latency"], LATENCY_FIELDS)
        public["errors"] = pick(row["errors"], ERROR_FIELDS)
        public["per_query_completed"] = {query: pick(metrics, METRICS) for query, metrics in row["per_query_completed"].items()}
        public["paired_bootstrap_ndcg10_vs_baseline"] = pick(row["paired_bootstrap_ndcg10_vs_baseline"], BOOTSTRAP_FIELDS)
        public["paired_bootstrap_ndcg10_vs_baseline"]["ci95"] = pick(row["paired_bootstrap_ndcg10_vs_baseline"]["ci95"], "lower upper")
        quality["rerankers"].append(public)
    quality["runs"] = []
    model_settings, memory = {}, []
    for run in report["runs"]:
        public = pick(run, "complete completed_measurements completion_fraction expected_measurements model model_name quality_repeat repeats split status summary_present")
        public["latency"] = pick(run["latency"], LATENCY_FIELDS)
        public["errors"] = pick(run["errors"], ERROR_FIELDS)
        public["warmup_errors"] = pick(run["warmup_errors"], ERROR_FIELDS)
        quality["runs"].append(public)
        model_settings[run["model"]] = pick(run["model_info"], MODEL_FIELDS)

    for model in MODELS:
        relative = (f"{JEV_RUN}/final-snapshot/runs/test-jev-openrouter-001" if model == "jev-openrouter" else f"{GPU_RUN}/test-{model}-001")
        run_dir = source / relative
        adapter = read_json(record(run_dir / "adapter.json", f"main/{model}/adapter"))
        model_settings[model]["adapter"] = pick(adapter, ADAPTER_FIELDS)
        if "runtime" in adapter:
            model_settings[model]["adapter"]["runtime"] = pick(adapter["runtime"], "cuda cudnn cudnn_allow_tf32 cudnn_benchmark deterministic_algorithms matmul_allow_tf32")
        if "gpu" in adapter:
            model_settings[model]["gpu"] = pick(adapter["gpu"], "name total_memory major minor multiprocessor_count")
        rows = []
        for row in read_jsonl(record(run_dir / "lists.jsonl", f"main/{model}/lists")):
            public = pick(row, "attempt dataset elapsed_seconds job_id method phase query_id")
            public["ranking"] = [pick(item, "doc_id score source_rank rank") for item in row["ranking"]]
            public["memory"] = pick(row["memory"], MEMORY_FIELDS) if row["memory"] else None
            # Raw response bodies, even from an API, are deliberately excluded.
            rows.append(public)
        assert len(rows) == 2544 and sum(row["phase"] == "measure" for row in rows) == 2542
        assert all(row["attempt"] == 1 and len(row["ranking"]) == 20 for row in rows)
        write(f"main/{model}.jsonl.gz", rows, jsonl=True)
        if model in GPU_MODELS:
            measured = [row for row in rows if row["phase"] == "measure"]
            memory.append({"model": model, **{key: max(row["memory"][key] for row in measured) for key in MEMORY_FIELDS.split()}})
    write("quality.json.gz", quality)
    write("model-settings.json", model_settings)
    write("memory.json", {"scope": "maximum per-list CUDA peaks across measured main-test lists; warmup excluded", "unit": "bytes", "models": memory})

    for dataset in DATASETS:
        for split in ("pilot", "test"):
            for method in ("bm25", "rrf"):
                name = f"{dataset}.{split}.{method}.jsonl.gz"
                source_path = source / f"runs/retrieval-20260927/{dataset}/candidates.{split}.{method}.jsonl"
                rows = [{"query_id": row["query_id"], "candidates": [pick(item, "doc_id rank score") for item in row["candidates"]]} for row in read_jsonl(record(source_path, f"candidates/{dataset}/{split}/{method}"))]
                write(f"candidates/{name}", rows, jsonl=True)

    throughput = read_json(record(source / THROUGHPUT_RUN / "throughput-report.json", "throughput/report"))
    public = {"scope": "separate RTX 5090, unchanged model settings, pilot candidates; no new quality test", "quality_retested": throughput["quality_retested"], "gpu": pick(throughput["gpu"], "name total_memory major minor multiprocessor_count"), "packages": pick(throughput["environment"]["packages"], "torch transformers requests tokenizers numpy safetensors huggingface-hub"), "models": []}
    for row in throughput["models"]:
        item = pick(row, "alias batch_size cold_start_seconds measured_pairs model pairs_per_second seconds warmup_pairs warmup_seconds")
        item["measurements"] = [pick(value, "phase repetition pairs seconds pairs_per_second") for value in row["measurements"]]
        public["models"].append(item)
    write("throughput.json", public)

    cost = read_json(record(source / JEV_RUN / "cost-latency-report.json", "api-cost/report"))
    public = pick(cost, "combined_cost_usd combined_paid_requests latency_percentile_method latency_scope provider response_model separate_api_throughput_benchmark_performed")
    public["scope"] = "OpenRouter API response usage costs only; account balances and billing records excluded"
    public["runs"] = {}
    for split in ("pilot", "test"):
        row = cost["runs"][split]
        item = pick(row, "max_http_request_seconds max_input_tokens max_observed_inflight_requests split")
        for phase in ("all", "measure", "warmup"):
            item[phase] = pick(row[phase], COST_FIELDS)
        item["by_dataset_method_measured"] = {group: pick(value, COST_FIELDS) for group, value in row["by_dataset_method_measured"].items()}
        public["runs"][split] = item
    write("api-cost.json", public)

    cpu_dir = source / "runs/retrieval-latency-cpu-20260927-001"
    cpu = read_json(record(cpu_dir / "summary.json", "cpu-retrieval/summary"))
    public = pick(cpu, "candidate_differences complete completed_measurements expected_measurements incomplete_last_record measurement quantile_method status wall_seconds worker_cpu_system_seconds worker_cpu_user_seconds worker_max_rss_kib worker_returncode")
    public["datasets"] = {dataset: {method: pick(cpu["datasets"][dataset][method], "n expected_n statistics_scope p50_ms p95_ms total_seconds candidate_order_mismatches") for method in ("bm25", "rrf")} for dataset in DATASETS}
    write("cpu-retrieval.json", public)
    rows = [pick(row, "dataset method query_id seconds split") for row in read_jsonl(record(cpu_dir / "queries.jsonl", "cpu-retrieval/queries"))]
    write("cpu-retrieval.jsonl.gz", rows, jsonl=True)

    manifest = {"schema_version": 1, "projection": "explicit reviewed field allowlists; original raw artifacts are not redistributed", "source_artifacts_sha256": sources, "public_artifacts": artifacts}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Exported {len(artifacts)} public artifacts; no model or network calls.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="private historical benchmark archive root")
    parser.add_argument("--output", type=Path, required=True, help="new, empty public-results directory")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be absent or empty; never overwrite an existing export")
    export(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
