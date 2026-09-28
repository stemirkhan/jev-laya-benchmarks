"""Frozen BM25 / exact dense / RRF candidates. No reranker or paid API calls."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("RAYON_NUM_THREADS", "4")

import bm25s
import numpy as np
import onnxruntime as ort
import Stemmer
from tokenizers import Tokenizer

from audit import Profile, chunks, document_text, read_dataset, summary
from prepare import ROOT, digest, write_json


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def select_queries(queries, qrels, dataset: str, n: int, seed: str) -> dict:
    test = sorted({q for q, _, _ in qrels["test"]})
    test_texts = {normalized(queries[q]["text"]) for q in test}
    source = "dev" if "dev" in qrels else "train"
    pool = sorted({q for q, _, _ in qrels[source]})
    seen, eligible, excluded = set(), [], []
    for qid in pool:
        text = normalized(queries[qid]["text"])
        reason = ("overlaps_test" if qid in test or text in test_texts else
                  "empty_query" if not text else "duplicate_pilot_text" if text in seen else None)
        if reason:
            excluded.append({"query_id": qid, "reason": reason})
        else:
            eligible.append(qid)
            seen.add(text)
    ranked = sorted(eligible, key=lambda q: (hashlib.sha256(f"{seed}/{dataset}/{q}".encode()).hexdigest(), q))
    if len(ranked) < n:
        raise ValueError(f"Not enough disjoint pilot queries for {dataset}")
    return {"test": test, "pilot": sorted(ranked[:n]), "pilot_source_split": source,
            "pilot_eligible_queries": len(eligible), "excluded_from_pilot": excluded,
            "selection_rule": "smallest SHA256(seed/dataset/query_id), no relevance-score filtering"}


def top_scores(ids: list[str], scores: np.ndarray, k: int) -> list[dict]:
    scores = np.asarray(scores)
    if scores.shape != (len(ids),) or not np.isfinite(scores).all() or len(ids) != len(set(ids)):
        raise ValueError("Invalid scores or duplicate document IDs")
    # Full sort includes all cutoff ties, unlike a bare argpartition top-k.
    order = np.lexsort((np.asarray(ids), -scores))[:k]
    return [{"doc_id": ids[int(i)], "score": float(scores[i]), "rank": rank}
            for rank, i in enumerate(order, 1)]


def rrf(lists: list[list[dict]], k: int = 60, weights=(1.0, 1.0)) -> list[dict]:
    if len(lists) != len(weights) or k < 0:
        raise ValueError("Invalid RRF configuration")
    scores, ranks = Counter(), {}
    for source, (rows, weight) in enumerate(zip(lists, weights)):
        if len({r["doc_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate document in one source ranking")
        for rank, row in enumerate(rows, 1):
            did = row["doc_id"]
            scores[did] += weight / (k + rank)
            ranks.setdefault(did, {})[str(source)] = rank
    ranked = top_scores(list(scores), np.array(list(scores.values()), dtype=np.float64), len(scores))
    for row in ranked:
        row["source_ranks"] = ranks[row["doc_id"]]
    return ranked


def write_rows(path: Path, rows) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def read_rankings(path: Path) -> dict:
    result = {}
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            qid = row["query_id"]
            if qid in result:
                raise ValueError("Duplicate query in rankings")
            result[qid] = row["candidates"]
    return result


def check_ranking(rows, docs, expected_count):
    if len(rows) != expected_count or len({r["doc_id"] for r in rows}) != len(rows):
        raise ValueError("Wrong candidate count or duplicate ID")
    if [r["rank"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Incorrect ranks")
    if any(r["doc_id"] not in docs or not np.isfinite(r["score"]) for r in rows):
        raise ValueError("Invalid candidate")
    if rows != sorted(rows, key=lambda row: (-row["score"], row["doc_id"])):
        raise ValueError("Ranking violates score/ID ordering")


def candidate_hashes(out: Path, datasets: list[str]) -> dict[str, str]:
    """Only finalized dataset artifacts; the append-only stage log is not immutable."""
    return {str(p.relative_to(out)): digest(p) for name in datasets for p in sorted((out / name).rglob("*"))
            if p.is_file() and p.suffix in (".json", ".jsonl", ".npy", ".gz")}


def verify_assets(config) -> None:
    for name in config["datasets"]:
        manifest = json.loads((ROOT / f"assets/manifests/dataset-{name}.json").read_text())
        for path, expected in manifest["files_sha256"].items():
            if digest(ROOT / "data/beir" / name / path) != expected:
                raise ValueError(f"Dataset changed: {name}/{path}")
    manifest = json.loads((ROOT / "assets/manifests/embedding.json").read_text())
    embedding = config["embedding"]
    if (manifest["repo"], manifest["revision"]) != (embedding["repo"], embedding["revision"]):
        raise ValueError("Wrong embedding revision")
    for name, expected in manifest["files_sha256"].items():
        if digest(ROOT / "assets/embedding" / name) != expected:
            raise ValueError(f"Embedding asset changed: {name}")
    for alias in ("laya-multi", "bge", "qwen-0.6b", "qwen-8b"):
        manifest = json.loads((ROOT / f"assets/manifests/model-{alias}.json").read_text())
        for name, expected in manifest["files_sha256"].items():
            if digest(ROOT / "assets/models" / alias / name) != expected:
                raise ValueError(f"Reranker tokenizer/config changed: {alias}/{name}")


def init_run(config: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", config)
    shutil.copytree(ROOT / "assets/manifests", out / "manifests")
    for name in ("retrieval.py", "audit.py", "prepare.py", "prepare_embedding.py", "pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, out / name)
    shutil.copytree(ROOT / "tests", out / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    write_json(out / "environment.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "bm25s", "PyStemmer", "onnxruntime", "tokenizers")},
        "config_sha256": json_hash(config), "test_quality_used_for_tuning": False,
    })
    for name in config["datasets"]:
        folder = out / name
        folder.mkdir()
        docs, queries, qrels, _ = read_dataset(name)
        selected = select_queries(queries, qrels, name, config["pilot_queries_per_dataset"], config["pilot_seed"])
        write_json(folder / "queries.json", selected)
        write_rows(folder / "query-texts.jsonl", [{"query_id": q, "text": queries[q]["text"]}
                    for q in selected["pilot"] + selected["test"]])
        write_rows(folder / "document-text-hashes.jsonl", [{"doc_id": d,
                    "text_sha256": hashlib.sha256(document_text(docs[d]).encode()).hexdigest()} for d in sorted(docs)])
        del docs, queries, qrels
    print(f"Run created: {out}", flush=True)


def bm25_stage(config, out):
    cfg, depth = config["bm25"], config["search"]["depth"]
    stemmer = Stemmer.Stemmer(cfg["stemmer"])
    token_args = {"lower": True, "token_pattern": cfg["token_pattern"], "stopwords": cfg["stopwords"],
                  "stemmer": stemmer, "return_ids": False, "show_progress": False}
    for name in config["datasets"]:
        folder = out / name
        if (folder / "bm25.complete.json").exists():
            continue
        started = time.perf_counter()
        docs, queries, _, _ = read_dataset(name)
        ids = sorted(docs)
        tokens = bm25s.tokenize([document_text(docs[d]) for d in ids], **token_args)
        index = bm25s.BM25(k1=cfg["k1"], b=cfg["b"], method=cfg["method"], backend="numpy")
        index.index(tokens, show_progress=False)
        index.save(str(folder / "bm25-index"))
        write_json(folder / "bm25-document-ids.json", ids)
        selected = json.loads((folder / "queries.json").read_text())
        for split in ("pilot", "test"):
            rows = []
            for qid in selected[split]:
                query_tokens = bm25s.tokenize([queries[qid]["text"]], **token_args)[0]
                scores = index.get_scores(query_tokens) if query_tokens else np.zeros(len(ids), dtype=np.float32)
                rows.append({"query_id": qid, "candidates": top_scores(ids, scores, depth)})
            write_rows(folder / f"bm25.{split}.top100.jsonl", rows)
        write_json(folder / "bm25.complete.json", {"documents": len(ids), "seconds_preparation": time.perf_counter() - started})
        print(f"BM25 ready: {name}, {len(ids)} documents", flush=True)
        del docs, queries, tokens, index
        gc.collect()


class Embedder:
    def __init__(self, config):
        self.config = config
        self.tok = Tokenizer.from_file(str(ROOT / "assets/embedding/tokenizer.json"))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.pad = self.tok.token_to_id("[PAD]")
        if self.pad is None:
            raise ValueError("Missing PAD token")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = config["threads"]
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(ROOT / "assets/embedding" / config["model_file"]),
                                           sess_options=opts, providers=[config["provider"]])
        if [o.name for o in self.session.get_outputs()] != ["last_hidden_state"]:
            raise ValueError("Unexpected official ONNX output")

    def lengths(self, texts):
        lengths = []
        for batch in chunks(texts, 256):
            lengths.extend(len(e.ids) for e in self.tok.encode_batch(batch, add_special_tokens=True))
        if max(lengths, default=0) > self.config["max_length"]:
            raise ValueError(f"Embedding input exceeds {self.config['max_length']}; refusing truncation")
        return lengths

    def encode(self, texts):
        encoded = self.tok.encode_batch(texts, add_special_tokens=True)
        size = max(len(e.ids) for e in encoded)
        if size > self.config["max_length"]:
            raise ValueError("Embedding input too long")
        ids = np.full((len(texts), size), self.pad, dtype=np.int64)
        mask, types = np.zeros_like(ids), np.zeros_like(ids)
        for i, e in enumerate(encoded):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.ids)] = 1
            types[i, :len(e.ids)] = e.type_ids
        hidden = self.session.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": types})[0]
        pooled = (hidden * mask[:, :, None]).sum(axis=1, dtype=np.float32) / mask.sum(axis=1, keepdims=True)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        if not np.isfinite(pooled).all() or np.any(norm <= 0):
            raise ValueError("Non-finite or zero embedding")
        return (pooled / norm).astype(np.float32)


def batch_end(lengths: list[int], start: int, config: dict) -> int:
    end = start + 1  # A long single input is allowed; it is never clipped to fit a batch.
    while end < len(lengths) and end - start < config["max_batch_size"]:
        if max(lengths[start:end + 1]) * (end + 1 - start) > config["padded_token_budget"]:
            break
        end += 1
    return end


def encode_store(model, records: dict[str, str], folder: Path, kind: str):
    ids = sorted(records)
    lengths = model.lengths([records[i] for i in ids])
    order = sorted(range(len(ids)), key=lambda i: (lengths[i], ids[i]))
    ids, lengths = [ids[i] for i in order], [lengths[i] for i in order]
    fingerprint = json_hash({"inputs": [(i, hashlib.sha256(records[i].encode()).hexdigest()) for i in ids],
                             "embedding": model.config})
    state_path, vector_path = folder / f"{kind}.progress.json", folder / f"{kind}.npy"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state and state["input_fingerprint"] != fingerprint:
        raise ValueError("Cannot resume embeddings with changed inputs/settings")
    if state and state["completed"] == len(ids):
        print(f"embeddings cached: {folder.name}/{kind}", flush=True)
        return
    start = state["completed"] if state else 0
    if vector_path.exists() and not state:
        raise ValueError("Uncheckpointed vectors exist; preserve them and use a new run")
    matrix = np.lib.format.open_memmap(vector_path, mode="r+" if state else "w+", dtype=np.float32,
                                      shape=(len(ids), model.config["dimensions"]))
    write_json(folder / f"{kind}.ids.json", ids)
    write_json(folder / f"{kind}.lengths.json", lengths)
    times = state.get("seconds_encoding", 0) if state else 0
    write_json(state_path, {"completed": start, "total": len(ids), "input_fingerprint": fingerprint,
                           "seconds_encoding": times, "max_input_tokens": max(lengths), "complete": False})
    segment = time.perf_counter()
    last_print = segment
    while start < len(ids):
        end = batch_end(lengths, start, model.config)
        matrix[start:end] = model.encode([records[i] for i in ids[start:end]])
        matrix.flush()
        write_json(state_path, {"completed": end, "total": len(ids), "input_fingerprint": fingerprint,
                               "seconds_encoding": times + time.perf_counter() - segment,
                               "max_input_tokens": max(lengths), "complete": end == len(ids)})
        if time.perf_counter() - last_print > 25 or end == len(ids):
            print(f"embed {folder.name}/{kind}: {end}/{len(ids)}, max tokens in batch={max(lengths[start:end])}", flush=True)
            last_print = time.perf_counter()
        start = end
    del matrix


def embedding_stage(config, out):
    model = Embedder(config["embedding"])
    for name in config["datasets"]:
        folder = out / name
        docs, queries, _, _ = read_dataset(name)
        selected = json.loads((folder / "queries.json").read_text())
        encode_store(model, {i: config["embedding"]["document_prefix"] + document_text(d) for i, d in docs.items()}, folder, "document-vectors")
        encode_store(model, {i: config["embedding"]["query_prefix"] + queries[i]["text"] for i in selected["pilot"] + selected["test"]}, folder, "query-vectors")
        del docs, queries
        gc.collect()


def dense_stage(config, out):
    for name in config["datasets"]:
        folder = out / name
        if (folder / "dense.complete.json").exists():
            continue
        for kind in ("document-vectors", "query-vectors"):
            if not json.loads((folder / f"{kind}.progress.json").read_text())["complete"]:
                raise ValueError("Embedding calculation is incomplete")
        docs = np.load(folder / "document-vectors.npy", mmap_mode="r")
        queries = np.load(folder / "query-vectors.npy", mmap_mode="r")
        dids = json.loads((folder / "document-vectors.ids.json").read_text())
        qids = json.loads((folder / "query-vectors.ids.json").read_text())
        selected = json.loads((folder / "queries.json").read_text())
        for matrix, ids in ((docs, dids), (queries, qids)):
            if matrix.shape != (len(ids), config["embedding"]["dimensions"]) or len(set(ids)) != len(ids):
                raise ValueError("Embedding shape/IDs mismatch")
            if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5):
                raise ValueError("Invalid or unnormalized embeddings")
        by_query = {}
        for start in range(0, len(qids), 32):
            scores = queries[start:start + 32] @ docs.T
            for index, row in enumerate(scores):
                by_query[qids[start + index]] = top_scores(dids, row, config["search"]["depth"])
        for split in ("pilot", "test"):
            write_rows(folder / f"dense.{split}.top100.jsonl", [{"query_id": qid, "candidates": by_query[qid]} for qid in selected[split]])
        write_json(folder / "dense.complete.json", {"documents": len(dids), "queries": len(qids), "metric": "cosine", "index": "exact_float32"})
        print(f"dense ready: {name}", flush=True)


def fusion_stage(config, out):
    for name in config["datasets"]:
        folder = out / name
        for split in ("pilot", "test"):
            lexical = read_rankings(folder / f"bm25.{split}.top100.jsonl")
            dense = read_rankings(folder / f"dense.{split}.top100.jsonl")
            if lexical.keys() != dense.keys():
                raise ValueError("Query mismatch")
            hybrid = {qid: rrf([lexical[qid], dense[qid]], config["rrf"]["k"], config["rrf"]["weights"]) for qid in lexical}
            write_rows(folder / f"rrf.{split}.union.jsonl", [{"query_id": q, "candidates": rows} for q, rows in hybrid.items()])
            for method, rankings in (("bm25", lexical), ("rrf", hybrid)):
                write_rows(folder / f"candidates.{split}.{method}.jsonl", [{"query_id": q,
                    "candidates": rows[:config["search"]["top_k"]]} for q, rows in rankings.items()])
        print(f"RRF ready: {name}", flush=True)


def validate_stage(config, out):
    overall = {"mode": "candidate_preparation_no_reranking", "datasets": {}, "config_sha256": json_hash(config)}
    for name in config["datasets"]:
        folder = out / name
        docs, queries, qrels, _ = read_dataset(name)
        selected = json.loads((folder / "queries.json").read_text())
        if selected != select_queries(queries, qrels, name, config["pilot_queries_per_dataset"], config["pilot_seed"]):
            raise ValueError("Changed query selection")
        result = {"documents": len(docs), "test_queries": len(selected["test"]), "pilot_queries": len(selected["pilot"]),
                  "pilot_exclusions": selected["excluded_from_pilot"], "splits": {}}
        for split in ("pilot", "test"):
            expected_queries, pairs = set(selected[split]), set()
            rankings = {m: read_rankings(folder / f"candidates.{split}.{m}.jsonl") for m in ("bm25", "rrf")}
            lexical = read_rankings(folder / f"bm25.{split}.top100.jsonl")
            dense = read_rankings(folder / f"dense.{split}.top100.jsonl")
            union = read_rankings(folder / f"rrf.{split}.union.jsonl")
            if any(set(source) != expected_queries for source in (lexical, dense, union)):
                raise ValueError("Source ranking query mismatch")
            for qid in selected[split]:
                for source in (lexical, dense):
                    check_ranking(source[qid], docs, config["search"]["depth"])
                expected_rrf = rrf([lexical[qid], dense[qid]], config["rrf"]["k"], config["rrf"]["weights"])
                if union[qid] != expected_rrf:
                    raise ValueError("Saved RRF disagrees with original rankings")
                for method, source in (("bm25", lexical[qid]), ("rrf", expected_rrf)):
                    if rankings[method].get(qid) != source[:config["search"]["top_k"]]:
                        raise ValueError("Candidate list differs from source top-k")
            for method, by_query in rankings.items():
                if set(by_query) != expected_queries:
                    raise ValueError("Missing or extra query")
                for qid, rows in by_query.items():
                    check_ranking(rows, docs, config["search"]["top_k"])
                    for row in rows:
                        pairs.add((qid, row["doc_id"]))
            ordered = sorted(pairs)
            lengths = {}
            for alias in ("laya-multi", "bge", "qwen-0.6b", "qwen-8b"):
                profile = Profile(alias)
                values = []
                for batch in chunks(ordered):
                    values.extend(profile.full_lengths([(queries[q]["text"], document_text(docs[d])) for q, d in batch]))
                limit = 8192 if alias == "laya-multi" else max(profile.limits)
                lengths[alias] = summary(values, [limit])
                if lengths[alias]["over_limits"][str(limit)]["n"]:
                    raise ValueError(f"Reranker input too long: {name}/{split}/{alias}")
                del profile
                gc.collect()
            overlap = [len({r["doc_id"] for r in rankings["bm25"][q]} & {r["doc_id"] for r in rankings["rrf"][q]}) for q in selected[split]]
            result["splits"][split] = {"queries": len(selected[split]), "pairs_per_method": len(selected[split]) * config["search"]["top_k"],
                                        "unique_pairs_both_methods": len(pairs), "top20_overlap": summary(overlap),
                                        "bm25_no_positive_score_queries": [q for q in selected[split] if lexical[q][0]["score"] <= 0],
                                        "full_input_lengths": lengths}
            # One immutable-text pool for all rerankers, with stable query/document IDs.
            with gzip.open(folder / f"pairs.{split}.jsonl.gz", "wt") as target:
                for qid, did in ordered:
                    text = document_text(docs[did])
                    target.write(json.dumps({"query_id": qid, "doc_id": did, "query": queries[qid]["text"],
                                             "document": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}, ensure_ascii=False) + "\n")
        result["embedding_lengths"] = {kind: summary(json.loads((folder / f"{kind}.lengths.json").read_text()), [8192])
                                      for kind in ("document-vectors", "query-vectors")}
        overall["datasets"][name] = result
        print(f"validated: {name}", flush=True)
    overall["jev_input_length"] = "pending; Jev tokenizer/API not called"
    overall["artifacts_sha256"] = candidate_hashes(out, config["datasets"])
    write_json(out / "summary.json", overall)
    write_json(ROOT / "runs/latest-retrieval.json", {"run": out.name, "complete": True})


def probe(config):
    """Technical CPU probe on documents selected by length, without test relevance scores."""
    model = Embedder(config["embedding"])
    docs, _, _, _ = read_dataset("scifact")
    ids = sorted(docs)
    lengths = model.lengths([document_text(docs[i]) for i in ids])
    ordered = sorted(range(len(ids)), key=lambda i: lengths[i])
    chosen = [ordered[int(p * (len(ordered) - 1))] for p in np.linspace(.05, .95, 32)]
    examples = [document_text(docs[ids[i]]) for i in chosen]
    model.encode([examples[0]])
    start = time.perf_counter()
    for batch in chunks(examples, 4):
        model.encode(batch)
    seconds = time.perf_counter() - start
    # Ensure the long-input path actually runs, not just the tokenizer.
    start_long = time.perf_counter()
    model.encode([document_text(docs[ids[ordered[-1]]])])
    result = {"documents": len(examples), "seconds": seconds, "documents_per_second": len(examples) / seconds,
              "longest_scifact_tokens": lengths[ordered[-1]], "longest_document_seconds": time.perf_counter() - start_long,
              "not_a_reranker_latency_benchmark": True}
    write_json(ROOT / "runs/embedding-cpu-probe.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["probe", "all", "bm25", "embed", "dense", "fuse", "validate"])
    parser.add_argument("--run")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/retrieval.json").read_text())
    verify_assets(config)
    if args.stage == "probe":
        probe(config)
        return
    run = args.run or ("retrieval-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if Path(run).name != run or run in (".", ".."):
        raise ValueError("run must be a directory name")
    out = ROOT / "runs" / run
    if out.exists():
        if json.loads((out / "config.json").read_text()) != config:
            raise ValueError("Configuration changed: preserve old run and create a new one")
        environment = json.loads((out / "environment.json").read_text())
        for package, version in environment["packages"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"Environment changed since run creation: {package}")
        for manifest in (out / "manifests").glob("*.json"):
            current = ROOT / "assets/manifests" / manifest.name
            if not current.exists() or current.read_bytes() != manifest.read_bytes():
                raise ValueError(f"Asset manifest changed since run creation: {manifest.name}")
        if (out / "summary.json").exists():
            saved = json.loads((out / "summary.json").read_text())
            for rel, expected in saved["artifacts_sha256"].items():
                if digest(out / rel) != expected:
                    raise ValueError(f"Completed artifact changed: {rel}")
            print(f"Run already complete and hashes verified: {out}", flush=True)
            return
    else:
        init_run(config, out)
    stages = {"bm25": bm25_stage, "embed": embedding_stage, "dense": dense_stage,
              "fuse": fusion_stage, "validate": validate_stage}
    for stage, function in stages.items():
        if args.stage in ("all", stage):
            # Preserve the exact script used for each stage, including resumed runs.
            code_hash = digest(Path(__file__))
            source_folder = out / "stage-sources"
            source_folder.mkdir(exist_ok=True)
            shutil.copy2(Path(__file__), source_folder / f"{code_hash}.py")
            attempt = {"stage": stage, "started_at": datetime.now(timezone.utc).isoformat(), "code_sha256": code_hash}
            try:
                function(config, out)
            except BaseException as exc:
                attempt["status"] = "failed"
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                raise
            else:
                attempt["status"] = "completed"
            finally:
                attempt["finished_at"] = datetime.now(timezone.utc).isoformat()
                with (out / "attempts.jsonl").open("a") as log:
                    log.write(json.dumps(attempt, ensure_ascii=False) + "\n")
    print(f"Stage {args.stage} finished: {out}", flush=True)


if __name__ == "__main__":
    main()
