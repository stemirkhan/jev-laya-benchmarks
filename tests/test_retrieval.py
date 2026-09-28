import json
import math
from pathlib import Path
import gzip

import bm25s
import numpy as np
import pytest

import retrieval as r


def test_bm25_matches_lucene_formula():
    corpus = [["cat", "cat", "dog"], ["cat"], ["fish"]]
    index = bm25s.BM25(k1=1.2, b=.75, method="lucene")
    index.index(corpus, show_progress=False)
    actual = index.get_scores(["cat"])
    idf = math.log(1 + (3 - 2 + .5) / (2 + .5))
    expected = [idf * tf / (tf + 1.2 * (1 - .75 + .75 * dl / (5 / 3))) if tf else 0
                for tf, dl in [(2, 3), (1, 1), (0, 1)]]
    # bm25s Lucene omits the constant (k1 + 1), which does not affect the order.
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_top_scores_resolves_cutoff_ties_by_id():
    result = r.top_scores(["d4", "d1", "d3", "d2"], np.array([0., 1., 1., 1.]), 2)
    assert [v["doc_id"] for v in result] == ["d1", "d2"]
    with pytest.raises(ValueError):
        r.top_scores(["a"], np.array([np.nan]), 1)


def test_rrf_one_based_weighted_and_unique():
    lexical = [{"doc_id": "b"}, {"doc_id": "a"}]
    dense = [{"doc_id": "a"}, {"doc_id": "c"}]
    result = r.rrf([lexical, dense])
    assert [v["doc_id"] for v in result] == ["a", "b", "c"]
    assert result[0]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert result[0]["source_ranks"] == {"0": 2, "1": 1}
    assert len(result) == 3
    with pytest.raises(ValueError):
        r.rrf([[{"doc_id": "a"}, {"doc_id": "a"}], dense])


def test_rrf_ties_are_deterministic():
    left = [{"doc_id": "b"}, {"doc_id": "a"}]
    right = [{"doc_id": "a"}, {"doc_id": "b"}]
    assert [v["doc_id"] for v in r.rrf([left, right])] == ["a", "b"]


def test_ranking_check_rejects_bad_order_unknown_and_missing_documents():
    valid = [{"doc_id": "a", "score": 1., "rank": 1}, {"doc_id": "b", "score": 1., "rank": 2}]
    r.check_ranking(valid, {"a", "b"}, 2)
    wrong = [{"doc_id": "b", "score": 1., "rank": 1}, {"doc_id": "a", "score": 1., "rank": 2}]
    with pytest.raises(ValueError, match="ordering"):
        r.check_ranking(wrong, {"a", "b"}, 2)
    with pytest.raises(ValueError, match="Invalid candidate"):
        r.check_ranking(valid, {"a"}, 2)
    with pytest.raises(ValueError, match="count"):
        r.check_ranking(valid, {"a", "b"}, 3)


def test_candidate_hashes_include_id_maps_but_not_mutable_logs(tmp_path):
    (tmp_path / "tiny").mkdir()
    (tmp_path / "tiny/vectors.ids.json").write_text('["id"]')
    (tmp_path / "tiny/candidates.test.bm25.jsonl").write_text('{}\n')
    (tmp_path / "attempts.jsonl").write_text('{}\n')
    before = r.candidate_hashes(tmp_path, ["tiny"])
    assert "tiny/vectors.ids.json" in before
    assert "attempts.jsonl" not in before
    (tmp_path / "attempts.jsonl").write_text('{}\n{}\n')
    assert r.candidate_hashes(tmp_path, ["tiny"]) == before


def test_pilot_preserves_test_and_excludes_text_leaks():
    queries = {"t": {"text": "The Test"}, "copy": {"text": "the   test"}, "p1": {"text": "one"}, "p2": {"text": "two"}}
    qrels = {"test": [("t", "d", 1)], "dev": [(q, "d", 1) for q in ("copy", "p1", "p2")]}
    result = r.select_queries(queries, qrels, "dataset", 2, "seed")
    assert result["test"] == ["t"] and result["pilot"] == ["p1", "p2"]
    assert result["excluded_from_pilot"] == [{"query_id": "copy", "reason": "overlaps_test"}]
    assert result == r.select_queries(dict(reversed(list(queries.items()))), qrels, "dataset", 2, "seed")


def test_dynamic_batches_never_drop_a_long_document():
    cfg = {"max_batch_size": 4, "padded_token_budget": 100}
    lengths = [10, 20, 30, 1000]
    assert r.batch_end(lengths, 0, cfg) == 3
    assert r.batch_end(lengths, 3, cfg) == 4


def test_fusion_artifacts_preserve_both_source_orders(tmp_path):
    folder = tmp_path / "example"
    folder.mkdir()
    lexical = [{"doc_id": "b", "score": 3., "rank": 1}, {"doc_id": "a", "score": 1., "rank": 2}]
    dense = [{"doc_id": "a", "score": .8, "rank": 1}, {"doc_id": "c", "score": .6, "rank": 2}]
    for split in ("pilot", "test"):
        for method, rows in (("bm25", lexical), ("dense", dense)):
            r.write_rows(folder / f"{method}.{split}.top100.jsonl", [{"query_id": "q", "candidates": rows}])
    cfg = {"datasets": ["example"], "rrf": {"k": 60, "weights": [1., 1.]}, "search": {"top_k": 2}}
    r.fusion_stage(cfg, tmp_path)
    assert r.read_rankings(folder / "bm25.test.top100.jsonl")["q"] == lexical
    assert [row["doc_id"] for row in r.read_rankings(folder / "candidates.test.rrf.jsonl")["q"]] == ["a", "b"]


def test_embedding_full_length_padding_and_unit_norm():
    if not (r.ROOT / "assets/embedding/model.onnx").exists():
        pytest.skip("prepare_embedding.py required")
    cfg = json.loads((r.ROOT / "config/retrieval.json").read_text())["embedding"]
    model = r.Embedder(cfg)
    short, long = "A cat sleeps.", "A long document. " * 200
    assert model.lengths([long])[0] > 512
    alone = model.encode([short])[0]
    together = model.encode([short, long])
    assert together.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(together, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(alone, together[0], atol=2e-6)
    with pytest.raises(ValueError, match="exceeds"):
        model.lengths(["word " * 9000])


def test_resumed_encoding_checks_fingerprint_and_skips_completed(tmp_path):
    class FakeModel:
        config = {"max_batch_size": 2, "padded_token_budget": 20, "dimensions": 2}
        calls = 0

        def lengths(self, texts):
            return list(map(len, texts))

        def encode(self, texts):
            self.calls += 1
            return np.array([[1., 0.]] * len(texts), dtype=np.float32)

    model = FakeModel()
    records = {"a": "short", "b": "a longer text"}
    r.encode_store(model, records, tmp_path, "vectors")
    calls = model.calls
    r.encode_store(model, records, tmp_path, "vectors")
    assert model.calls == calls
    with pytest.raises(ValueError, match="changed inputs"):
        r.encode_store(model, {**records, "a": "changed"}, tmp_path, "vectors")


def test_dense_fusion_validation_preserves_complete_texts(tmp_path, monkeypatch):
    cfg = {"datasets": ["tiny"], "pilot_queries_per_dataset": 1, "pilot_seed": "test",
           "embedding": {"dimensions": 2}, "search": {"depth": 2, "top_k": 1},
           "rrf": {"k": 60, "weights": [1., 1.]}}
    out = tmp_path / "runs/example"
    folder = out / "tiny"
    folder.mkdir(parents=True)
    docs = {"a": {"title": "Title", "text": " complete text with spaces \n"},
            "b": {"title": "", "text": "Other full document"}}
    queries = {"q": {"text": "test query"}, "p": {"text": "pilot query"}}
    qrels = {"test": [("q", "a", 1)], "train": [("p", "b", 1)]}
    selection = r.select_queries(queries, qrels, "tiny", 1, "test")
    r.write_json(folder / "queries.json", selection)
    for kind, ids in (("document-vectors", ["a", "b"]), ("query-vectors", ["q", "p"])):
        np.save(folder / f"{kind}.npy", np.eye(2, dtype=np.float32))
        r.write_json(folder / f"{kind}.ids.json", ids)
        r.write_json(folder / f"{kind}.lengths.json", [10, 20])
        r.write_json(folder / f"{kind}.progress.json", {"complete": True})
    for split, qid in (("test", "q"), ("pilot", "p")):
        rows = [{"doc_id": "a", "score": 1., "rank": 1}, {"doc_id": "b", "score": 0., "rank": 2}]
        r.write_rows(folder / f"bm25.{split}.top100.jsonl", [{"query_id": qid, "candidates": rows}])

    class FakeProfile:
        limits = [8192]
        def __init__(self, alias):
            pass
        def full_lengths(self, pairs):
            return [len(q) + len(d) + 34 for q, d in pairs]

    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "Profile", FakeProfile)
    monkeypatch.setattr(r, "read_dataset", lambda name: (docs, queries, qrels, {}))
    r.dense_stage(cfg, out)
    r.fusion_stage(cfg, out)
    r.validate_stage(cfg, out)
    with gzip.open(folder / "pairs.test.jsonl.gz", "rt") as source:
        pair = json.loads(source.readline())
    assert pair["document"] == "Title\n complete text with spaces \n"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["datasets"]["tiny"]["splits"]["test"]["queries"] == 1
    assert summary["artifacts_sha256"]["tiny/pairs.test.jsonl.gz"] == r.digest(folder / "pairs.test.jsonl.gz")
