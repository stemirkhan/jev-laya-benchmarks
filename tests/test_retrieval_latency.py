import json
import subprocess

import numpy as np
import pytest

import retrieval as r
import retrieval_latency as latency


def rows(*ids):
    return [{"doc_id": did, "rank": i, "score": float(len(ids) - i)} for i, did in enumerate(ids, 1)]


def test_bm25_and_hybrid_run_independently_with_fresh_query_embedding(monkeypatch):
    class Embedder:
        calls = []
        def encode(self, texts):
            self.calls.append(texts)
            return np.array([[1., 0.]], dtype=np.float32)

    config = {"bm25": {"token_pattern": r"(?u)\b\w\w+\b", "stopwords": "en", "stemmer": "english"},
              "search": {"top_k": 2, "depth": 3}, "embedding": {"query_prefix": "prefix "},
              "rrf": {"k": 60, "weights": [1., 1.]}}
    index = r.bm25s.BM25(method="lucene", k1=1.2, b=.75)
    index.index([["cat"], ["dog"], ["cat", "dog"]], show_progress=False)
    embedder = Embedder()
    engine = latency.SearchEngine(config, index, ["a", "b", "c"],
                                 np.array([[0., 1.], [1., 0.], [.6, .8]], dtype=np.float32),
                                 ["a", "b", "c"], embedder)
    calls = []
    original = engine.lexical
    def lexical(text):
        calls.append(text)
        return original(text)
    monkeypatch.setattr(engine, "lexical", lexical)
    bm25, _, _ = engine.search("cat", "bm25")
    hybrid, seconds, components = engine.search("cat", "rrf")
    assert calls == ["cat", "cat"]
    assert embedder.calls == [["prefix cat"]]
    expected = r.rrf([original("cat"), engine.dense(np.array([1., 0.], dtype=np.float32))])[:2]
    assert hybrid == expected
    assert bm25 == original("cat")[:2]
    assert sum(components.values()) == pytest.approx(seconds)
    assert set(components) == {"embedding_seconds", "bm25_seconds", "exact_dense_seconds", "rrf_seconds"}


def test_missing_words_keep_zero_score_id_tie_break():
    config = {"bm25": {"token_pattern": r"(?u)\b\w\w+\b", "stopwords": "en", "stemmer": "english"},
              "search": {"top_k": 2, "depth": 2}}
    index = r.bm25s.BM25()
    index.index([["cat"], ["dog"]], show_progress=False)
    engine = latency.SearchEngine(config, index, ["z", "a"], None, None, None)
    assert [x["doc_id"] for x in engine.search("the", "bm25")[0]] == ["a", "z"]


def test_candidate_difference_keeps_actual_and_frozen_order():
    frozen, actual = rows("a", "b"), rows("b", "c")
    comparison = latency.compare_candidates(actual, frozen)
    assert comparison["same_order"] is False
    assert comparison["missing_ids"] == ["a"]
    assert comparison["added_ids"] == ["c"]
    assert frozen == rows("a", "b")
    assert latency.compare_candidates(rows("b", "a"), frozen)["same_set"] is True


def test_pilot_runs_before_test_and_differences_do_not_retry_or_change_candidates(tmp_path):
    calls = []
    class Engine:
        def search(self, text, method):
            calls.append((text, method))
            return rows("wrong", "a"), .01, {"total": .01}
    frozen = {split: {method: {qid: rows("a", "b")} for method in ("bm25", "rrf")}
              for split, qid in (("pilot", "p"), ("test", "t"))}
    bundles = {name: {"selected": {"pilot": ["p"], "test": ["t"]}, "engine": Engine(),
                      "texts": {"p": name + " pilot", "t": name + " test"}, "frozen": frozen}
               for name in ("first", "second")}
    latency.measure_queries(bundles, tmp_path)
    assert calls == [("first pilot", "bm25"), ("first pilot", "rrf"),
                     ("second pilot", "bm25"), ("second pilot", "rrf"),
                     ("first test", "bm25"), ("first test", "rrf"),
                     ("second test", "bm25"), ("second test", "rrf")]
    assert len(latency.read_rows(tmp_path / "queries.jsonl")) == 4
    assert len(latency.read_rows(tmp_path / "warmup.jsonl")) == 4
    assert len(latency.read_rows(tmp_path / "differences.jsonl")) == 8
    assert frozen["test"]["bm25"]["t"] == rows("a", "b")


def test_selection_requires_all_frozen_queries_and_disjoint_pilot():
    selected = {"pilot": ["a", "b", "c", "d"], "test": ["t", "u"]}
    texts = dict.fromkeys(selected["pilot"] + selected["test"], "text")
    latency.check_query_selection(selected, texts, 2)
    with pytest.raises(ValueError, match="test"):
        latency.check_query_selection(selected, texts, 3)
    with pytest.raises(ValueError, match="overlaps"):
        latency.check_query_selection({**selected, "test": ["a", "u"]}, texts, 2)


def make_results(out, methods=("bm25",)):
    latency.write_json(out / "manifest.json", {"test_queries": {"tiny": ["q"]}})
    for method in methods:
        latency.append(out / "queries.jsonl", {
            "dataset": "tiny", "query_id": "q", "method": method, "seconds": .25,
            "comparison": {"same_order": False},
        })


def test_partial_does_not_claim_full_statistics_and_preserves_complete_rows(tmp_path):
    make_results(tmp_path)
    with (tmp_path / "queries.jsonl").open("a") as source:
        source.write('{"query_id":')
    result = latency.summarize(tmp_path, -15, "time_limit", 10.)
    assert result["status"] == "partial"
    assert result["incomplete_last_record"] is True
    assert result["completed_measurements"] == 1
    assert result["datasets"]["tiny"]["bm25"]["statistics_scope"] == "completed_queries_only"
    assert result["datasets"]["tiny"]["rrf"]["p95_ms"] is None


def test_full_measurement_can_report_candidate_differences_without_relabeling_quality(tmp_path):
    make_results(tmp_path, ("bm25", "rrf"))
    result = latency.summarize(tmp_path, 0, "finished", 1.)
    assert result["complete"] is True
    assert result["datasets"]["tiny"]["rrf"]["candidate_order_mismatches"] == 1
    assert result["datasets"]["tiny"]["bm25"]["p95_ms"] == 250
    assert "not end-to-end" in result["measurement"]


def test_wall_deadline_terminates_worker_and_summarizes_partial(monkeypatch, tmp_path):
    make_results(tmp_path)
    class Process:
        returncode = None
        calls = []
        def wait(self, timeout=None):
            self.calls.append(("wait", timeout))
            if len(self.calls) == 1:
                raise subprocess.TimeoutExpired("synthetic", timeout)
            self.returncode = -15
        def terminate(self):
            self.calls.append(("terminate", None))
        def kill(self):
            pytest.fail("graceful termination should finish")
    process = Process()
    monkeypatch.setattr(latency.subprocess, "Popen", lambda *a, **kw: process)
    result = latency.supervise(["synthetic"], tmp_path, 10.)
    assert process.calls == [("wait", 10.), ("terminate", None), ("wait", 2)]
    assert result["status"] == "partial"
    assert result["stop_reason"] == "time_limit"


def test_percentiles_interpolate_instead_of_selecting_nearest_rank():
    assert latency.quantile([1, 2, 3, 4], .5) == 2.5
    assert latency.quantile([1, 2, 3, 4], .95) == pytest.approx(3.85)
