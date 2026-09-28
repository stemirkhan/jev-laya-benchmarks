"""Synthetic fixtures only; no model inference or network calls."""
import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest

import evaluate_rerank as ev
import rerank


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def fixture_source(tmp_path, pilot_split="train"):
    source, data = tmp_path / "retrieval", tmp_path / "data"
    folder = source / "demo"
    config = {"datasets": ["demo"], "search": {"top_k": 20}}
    write_json(source / "config.json", config)
    write_json(folder / "queries.json", {"pilot": ["q", "r"], "test": ["t"], "pilot_source_split": pilot_split})
    write_rows(folder / "query-texts.jsonl", [{"query_id": q, "text": f"query {q}"} for q in ("q", "r", "t")])
    for split, qids in (("pilot", ["q", "r"]), ("test", ["t"])):
        candidates = [{"doc_id": str(i), "rank": i + 1, "score": float(20 - i)} for i in range(20)]
        for method in ("bm25", "rrf"):
            write_rows(folder / f"candidates.{split}.{method}.jsonl",
                       [{"query_id": q, "candidates": candidates} for q in qids])
        with gzip.open(folder / f"pairs.{split}.jsonl.gz", "wt") as output:
            for q in qids:
                for i in range(20):
                    text = f"document {i}"
                    output.write(json.dumps({"query_id": q, "doc_id": str(i), "query": f"query {q}",
                                             "document": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}) + "\n")
    write_json(source / "summary.json", {"config_sha256": ev.fingerprint(config),
               "artifacts_sha256": {str(path.relative_to(source)): ev.sha_bytes(path.read_bytes()) for path in folder.iterdir()}})
    qrel_dir = data / "demo/qrels"
    qrel_dir.mkdir(parents=True)
    for split in ("train", "dev"):
        # An unretrieved relevant doc makes candidate Recall@20 < 1.
        (qrel_dir / f"{split}.tsv").write_text("query-id\tcorpus-id\tscore\nq\t19\t2\nq\tunretrieved\t1\nr\t0\t1\n")
    (qrel_dir / "test.tsv").write_text("query-id\tcorpus-id\tscore\nt\t1\t1\n")
    return source, data


class FakeAdapter:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def synchronize(self):
        pass

    def metadata(self):
        return {"model": "synthetic-fixture-not-real-inference"}

    def close(self):
        pass

    def score(self, pairs):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic failure")
        return [{"score": float(i)} for i, _ in enumerate(pairs)]


def execute_fixture(source, out, *, repeats=1, fail_at=None, retry=False):
    config = {"model": "fake", "model_config": {"repo": "synthetic/model", "dtype": "fp32"},
              "batch_size": 20, "warmup_lists": 1, "repeats": repeats}
    return rerank.execute(rerank.Candidates(source), out, config,
                          lambda *args: FakeAdapter(fail_at), retry_failed=retry)


def test_ndcg_uses_linear_not_exponential_gains():
    got = ev.ranking_metrics(["b", "a"], {"a": 2, "b": 1})
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    exponential = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert got["NDCG@10"] == pytest.approx(expected, abs=1e-15)
    assert got["NDCG@10"] != pytest.approx(exponential)
    assert got["Recall@10"] == 1
    assert got["Recall@20"] == 1
    assert got["MRR@10"] == 1


def test_ndcg_denominator_uses_all_qrels_not_only_candidates():
    got = ev.ranking_metrics(["low"], {"low": 1, "outside": 3})
    assert got["NDCG@10"] == pytest.approx(1 / (3 + 1 / math.log2(3)))
    assert got["Recall@10"] == .5
    assert got["Recall@20"] == .5


def test_cutoffs_and_unjudged_documents():
    docs = [f"d{i}" for i in range(30)]
    got = ev.ranking_metrics(docs, {"d10": 2, "d19": 1, "d20": 1})
    assert got == {"NDCG@10": 0, "Recall@10": 0, "Recall@20": 2 / 3, "MRR@10": 0}
    assert ev.ranking_metrics(docs, {"d9": 1})["MRR@10"] == .1
    assert ev.ranking_metrics(docs, {"d9": 1, "d10": 1})["Recall@10"] == .5


def test_zero_and_negative_relevance_and_identical_ids():
    assert ev.ranking_metrics(["a", "b"], {"a": 0, "b": -1}) == dict.fromkeys(ev.METRICS, 0)
    assert ev.ranking_metrics(["q"], {"q": 1}) == dict.fromkeys(ev.METRICS, 1)
    assert ev.ranking_metrics([], {"a": 1}) == dict.fromkeys(ev.METRICS, 0)


def test_duplicate_document_is_not_counted_twice():
    with pytest.raises(ValueError, match="Duplicate"):
        ev.ranking_metrics(["a", "a"], {"a": 1})


def test_macro_not_relevance_weighted_average():
    rows = [ev.ranking_metrics(["a"], {"a": 1}), ev.ranking_metrics([], {f"d{i}": 1 for i in range(100)})]
    assert ev.mean_metrics(rows) == dict.fromkeys(ev.METRICS, .5)
    assert ev.mean_metrics([]) is None


def test_percentiles():
    assert ev.percentile([1, 2, 3, 4], .5) == 2.5
    assert ev.percentile([1, 2, 3, 4], .95) == pytest.approx(3.85)
    assert ev.percentile([4], .95) == 4
    assert ev.percentile([], .95) is None
    with pytest.raises(ValueError):
        ev.percentile([1, float("nan")], .5)


@pytest.mark.parametrize("split", ["train", "dev"])
def test_pilot_reads_only_its_train_or_dev_qrels(tmp_path, split):
    source, data = fixture_source(tmp_path, split)
    # A malformed test file must never be read for pilot evaluation.
    (data / "demo/qrels/test.tsv").write_text("intentionally not qrels")
    loaded = ev.load_source(source, data)
    assert loaded.qrels_provenance["demo"]["split"] == split
    assert set(loaded.qrels["demo"]) == {"q", "r"}


def test_test_reads_only_test_qrels(tmp_path):
    source, data = fixture_source(tmp_path)
    (data / "demo/qrels/train.tsv").write_text("intentionally not qrels")
    loaded = ev.load_source(source, data, "test")
    assert loaded.qrels_provenance["demo"]["split"] == "test"
    assert set(loaded.qrels["demo"]) == {"t"}


def test_pilot_cannot_use_test_qrels(tmp_path):
    source, data = fixture_source(tmp_path, "test")
    with pytest.raises(ValueError, match="train/dev"):
        ev.load_source(source, data)


def test_test_cannot_omit_qrels_query(tmp_path):
    source, data = fixture_source(tmp_path)
    with (data / "demo/qrels/test.tsv").open("a") as output:
        output.write("extra\t1\t1\n")
    with pytest.raises(ValueError, match="all test"):
        ev.load_source(source, data, "test")


def test_changed_candidate_and_qrels_hashes_rejected(tmp_path):
    source, data = fixture_source(tmp_path)
    manifests = tmp_path / "manifests"
    write_json(manifests / "dataset-demo.json", {"files_sha256": {"qrels/train.tsv": "wrong"}})
    with pytest.raises(ValueError, match="Qrels differ"):
        ev.load_source(source, data, dataset_manifests=manifests)
    with (source / "demo/candidates.pilot.bm25.jsonl").open("a") as output:
        output.write("\n")
    with pytest.raises(ValueError, match="Changed candidate"):
        ev.load_source(source, data)


def test_duplicate_qrels_rejected(tmp_path):
    path = tmp_path / "qrels.tsv"
    path.write_text("query-id\tcorpus-id\tscore\nq\td\t1\nq\td\t1\n")
    with pytest.raises(ValueError, match="Duplicate qrel"):
        ev.read_qrels(path)


def test_baseline_oracle_recall_is_candidate_recall(tmp_path):
    source, data = fixture_source(tmp_path)
    report = ev.make_report(ev.load_source(source, data))
    baseline, oracle = report["baseline_and_oracle"][:2]
    assert baseline["metrics"]["Recall@20"] == oracle["metrics"]["Recall@20"] == .75
    assert baseline["metrics"]["Recall@10"] == .5
    assert oracle["metrics"]["Recall@10"] == .75
    assert baseline["metrics"]["NDCG@10"] < oracle["metrics"]["NDCG@10"] < 1
    assert "не является моделью" in ev.render_markdown(report)


def test_complete_actual_runner_format_and_repeats(tmp_path):
    source, data = fixture_source(tmp_path)
    out = tmp_path / "run"
    execute_fixture(source, out, repeats=2)
    loaded = ev.load_source(source, data)
    info, groups = ev.evaluate_run(loaded, out / "lists.jsonl")
    assert info["complete"] is True
    assert info["completed_measurements"] == info["expected_measurements"] == 8
    assert info["latency"]["successful_lists"] == 8
    assert info["errors"]["started_attempts"] == 8
    for group in groups:
        assert group["complete"] and group["quality_queries_completed"] == 2
        assert group["latency"]["successful_lists"] == 4
        assert group["metrics"]["Recall@20"] == .75
        assert group["metrics"]["Recall@10"] == .25
        bootstrap = group["paired_bootstrap_ndcg10_vs_baseline"]
        assert bootstrap["n_queries"] == 2  # Four lists / two repeats still have two queries.
        assert bootstrap["estimate"] == pytest.approx(group["metrics"]["NDCG@10"] - group["matched_baseline_diagnostic_metrics"]["NDCG@10"])
        assert group["later_repeat_rankings_differing_from_repeat_zero"] == 0
    assert info["warmup_errors"]["started_attempts"] == 1


def test_failed_run_has_no_headline_metrics_and_no_missing_as_zero(tmp_path):
    source, data = fixture_source(tmp_path)
    out = tmp_path / "run"
    # warmup, q BM25 succeed, r BM25 fails; both RRF jobs are unattempted.
    with pytest.raises(RuntimeError):
        execute_fixture(source, out, fail_at=3)
    report = ev.make_report(ev.load_source(source, data), [out])
    info = report["runs"][0]
    assert not info["complete"] and info["completed_measurements"] == 1
    assert info["errors"]["recorded_error_attempts"] == 1
    assert info["errors"]["unfinished_attempts"] == 0
    bm25, rrf = report["rerankers"]
    assert bm25["metrics"] is None and rrf["metrics"] is None
    assert bm25["paired_bootstrap_ndcg10_vs_baseline"] is None
    assert rrf["paired_bootstrap_ndcg10_vs_baseline"] is None
    assert bm25["quality_queries_completed"] == 1
    assert bm25["completed_only_diagnostic_metrics"]["Recall@20"] == .5
    assert bm25["matched_baseline_diagnostic_metrics"]["Recall@20"] == .5
    assert bm25["missing_quality_query_ids"] == ["r"]
    assert rrf["completed_only_diagnostic_metrics"] is None
    assert rrf["latency"]["p95_seconds"] is None
    assert "НЕ ЗАВЕРШЕН" in ev.render_markdown(report)


def test_retried_failure_stays_in_error_accounting(tmp_path):
    source, data = fixture_source(tmp_path)
    out = tmp_path / "run"
    with pytest.raises(RuntimeError):
        execute_fixture(source, out, fail_at=3)
    execute_fixture(source, out, retry=True)
    info, groups = ev.evaluate_run(ev.load_source(source, data), out)
    assert info["complete"]
    assert info["errors"]["recorded_error_attempts"] == 1
    assert info["errors"]["started_attempts"] == 5
    assert info["errors"]["retried_jobs"] == 1
    assert info["warmup_errors"]["started_attempts"] == 2
    assert groups[0]["errors"]["recorded_error_attempts"] == 1


def test_no_manifest_completion_even_when_all_lists_written(tmp_path):
    source, data = fixture_source(tmp_path)
    out = tmp_path / "run"
    execute_fixture(source, out)
    (out / "summary.json").unlink()
    info, groups = ev.evaluate_run(ev.load_source(source, data), out)
    assert not info["complete"] and info["completion_fraction"] == 1
    assert groups[0]["quality_query_coverage_complete"]
    assert groups[0]["metrics"] is None
    assert groups[0]["paired_bootstrap_ndcg10_vs_baseline"] is None
    assert groups[0]["completed_only_diagnostic_metrics"] is not None


def test_partial_tail_is_explicit_and_never_completed(tmp_path):
    source, data = fixture_source(tmp_path)
    out = tmp_path / "run"
    with pytest.raises(RuntimeError):
        execute_fixture(source, out, fail_at=3)
    with (out / "lists.jsonl").open("a") as output:
        output.write('{"unfinished":')
    info, groups = ev.evaluate_run(ev.load_source(source, data), out)
    assert not info["complete"]
    assert info["snapshots"]["lists"]["partial_tail"]
    assert groups[0]["completed_measurements"] == 1


def test_bad_json_in_middle_is_not_silently_skipped(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event":"ok"}\n{broken}\n{"event":"ok"}\n')
    with pytest.raises(ValueError, match="Invalid JSONL"):
        ev.read_jsonl_snapshot(path, tolerate_partial_tail=True)


def test_interrupted_start_not_an_error_or_success(tmp_path):
    source, data = fixture_source(tmp_path)
    out = tmp_path / "run"
    with pytest.raises(RuntimeError):
        execute_fixture(source, out, fail_at=3)
    rows, _ = ev.read_jsonl_snapshot(out / "events.jsonl")
    write_rows(out / "events.jsonl", [e for e in rows if e["event"] != "list_error"])
    info, _ = ev.evaluate_run(ev.load_source(source, data), out)
    assert info["errors"]["unfinished_attempts"] == 1
    assert info["errors"]["recorded_error_attempts"] == 0


def test_summary_hash_and_split_are_enforced(tmp_path):
    source, data = fixture_source(tmp_path)
    out = tmp_path / "run"
    execute_fixture(source, out)
    with pytest.raises(ValueError, match="split mismatch"):
        ev.evaluate_run(ev.load_source(source, data, "test"), out)
    with (out / "events.jsonl").open("a") as output:
        output.write('{"event":"additional"}\n')
    with pytest.raises(ValueError, match="Completion summary"):
        ev.evaluate_run(ev.load_source(source, data), out)


def test_duplicate_results_rejected_not_cherry_picked(tmp_path):
    source, data = fixture_source(tmp_path)
    out = tmp_path / "run"
    execute_fixture(source, out)
    rows, _ = ev.read_jsonl_snapshot(out / "lists.jsonl")
    write_rows(out / "lists.jsonl", rows + [rows[1]])
    with pytest.raises(ValueError, match="Duplicate completed"):
        ev.evaluate_run(ev.load_source(source, data), out)


def test_rerank_ties_keep_source_order():
    original = [{"doc_id": f"d{i}", "rank": i + 1, "score": 20 - i} for i in range(20)]
    ranking = [{**r, "score": .5, "source_rank": r["rank"]} for r in original]
    ev.validate_ranking(ranking, original=original)
    ranking[0], ranking[1] = ranking[1], ranking[0]
    ranking[0]["rank"], ranking[1]["rank"] = 1, 2
    with pytest.raises(ValueError, match="tie-break"):
        ev.validate_ranking(ranking, original=original)


@pytest.mark.parametrize("mutation", ["missing", "new_doc", "wrong_source_rank", "nonfinite"])
def test_changed_ranking_is_rejected(mutation):
    original = [{"doc_id": f"d{i}", "rank": i + 1, "score": 20 - i} for i in range(20)]
    ranking = [{**r, "source_rank": r["rank"]} for r in original]
    if mutation == "missing":
        ranking.pop()
    elif mutation == "new_doc":
        ranking[0]["doc_id"] = "outside"
    elif mutation == "wrong_source_rank":
        ranking[0]["source_rank"] = 9
    else:
        ranking[0]["score"] = float("nan")
    with pytest.raises(ValueError):
        ev.validate_ranking(ranking, original=original)


def test_optional_pytrec_eval_reference():
    trec = pytest.importorskip("pytrec_eval")
    qrels = {"q": {"a": 3, "b": 1, "outside": 2, "zero": 0}}
    docs = ["unjudged", "b", "zero", "a"]
    scores = {"q": {doc: float(20 - i) for i, doc in enumerate(docs)}}
    want = trec.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.10,20", "recip_rank"}).evaluate(scores)["q"]
    got = ev.ranking_metrics(docs, qrels["q"])
    assert got["NDCG@10"] == pytest.approx(want["ndcg_cut_10"], abs=1e-12)
    assert got["Recall@10"] == pytest.approx(want["recall_10"], abs=1e-12)
    assert got["Recall@20"] == pytest.approx(want["recall_20"], abs=1e-12)
    assert got["MRR@10"] == pytest.approx(want["recip_rank"], abs=1e-12)


def test_optional_full_report_pytrec_cross_check(tmp_path):
    pytest.importorskip("pytrec_eval")
    source, data = fixture_source(tmp_path)
    out = tmp_path / "run"
    execute_fixture(source, out)
    report = ev.make_report(ev.load_source(source, data), [out], cross_check=True)
    assert report["pytrec_eval_cross_check"]["status"] == "passed"
    assert report["pytrec_eval_cross_check"]["checked_rankings"] == 12


def test_existing_reports_and_raw_run_directories_are_protected(tmp_path):
    source, data = fixture_source(tmp_path)
    report = ev.make_report(ev.load_source(source, data))
    out = tmp_path / "evaluation"
    ev.write_report(report, out)
    before = (out / "report.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        ev.write_report(report, out)
    assert (out / "report.json").read_bytes() == before
    ev.write_report(report, out, overwrite=True)
    with pytest.raises(ValueError, match="separate evaluation"):
        ev.write_report(report, source, overwrite=True)


def test_duplicate_or_orphan_terminal_events_rejected(tmp_path):
    source, data = fixture_source(tmp_path)
    out = tmp_path / "run"
    with pytest.raises(RuntimeError):
        execute_fixture(source, out, fail_at=3)
    events, _ = ev.read_jsonl_snapshot(out / "events.jsonl")
    error = next(e for e in events if e["event"] == "list_error")
    write_rows(out / "events.jsonl", events + [error])
    with pytest.raises(ValueError, match="Duplicate terminal"):
        ev.evaluate_run(ev.load_source(source, data), out)
    write_rows(out / "events.jsonl", [error] + events)
    with pytest.raises(ValueError, match="no preceding start"):
        ev.evaluate_run(ev.load_source(source, data), out)


def ndcg_rows(values):
    return {qid: {"NDCG@10": value} for qid, value in values.items()}


def test_paired_bootstrap_identical_models_have_zero_interval():
    # Varying query difficulty must not add variance to a paired zero difference.
    rows = ndcg_rows({"hard": 0, "medium": .5, "easy": 1})
    result = ev.paired_bootstrap_ndcg10(rows, rows)
    assert result["estimate"] == 0
    assert result["ci95"] == {"lower": 0, "upper": 0}
    assert result["resamples"] == 2000 and result["seed"] == 20260927
    assert result["confidence_level"] == .95 and result["resampling_unit"] == "query"
    assert result["multiple_comparisons_adjusted"] is False
    assert result["constant_query_differences"] is True


def test_paired_bootstrap_constant_negative_effect():
    model = ndcg_rows({"a": 0, "b": .25, "c": .5})
    baseline = ndcg_rows({"a": .25, "b": .5, "c": .75})
    result = ev.paired_bootstrap_ndcg10(model, baseline)
    assert result["estimate"] == -.25
    assert result["ci95"] == {"lower": -.25, "upper": -.25}


def test_paired_bootstrap_two_query_distribution():
    # Paired differences are -.5 and +.5. The exact n=2 bootstrap mean takes
    # -.5, 0, +.5 with probabilities 1/4, 1/2, 1/4; both tail quantiles are known.
    model = ndcg_rows({"loss": 0, "win": 1})
    baseline = ndcg_rows({"loss": .5, "win": .5})
    result = ev.paired_bootstrap_ndcg10(model, baseline)
    assert result["estimate"] == 0
    assert result["ci95"] == {"lower": -.5, "upper": .5}


def test_bootstrap_is_reproducible_and_query_order_independent():
    model = ndcg_rows({"z": .1, "a": .8, "m": .5, "b": .9})
    baseline = ndcg_rows({"m": .4, "b": .5, "z": .3, "a": .4})
    first = ev.paired_bootstrap_ndcg10(model, baseline)
    assert ev.paired_bootstrap_ndcg10(model, baseline) == first
    assert ev.paired_bootstrap_ndcg10(dict(reversed(list(model.items()))), baseline) == first
    assert ev.paired_bootstrap_ndcg10(model, dict(reversed(list(baseline.items())))) == first


def test_bootstrap_single_query_is_explicitly_degenerate():
    result = ev.paired_bootstrap_ndcg10(ndcg_rows({"q": .75}), ndcg_rows({"q": .5}))
    assert result["n_queries"] == 1
    assert result["constant_query_differences"] is True
    assert result["estimate"] == .25
    assert result["ci95"] == {"lower": .25, "upper": .25}


@pytest.mark.parametrize("model,baseline", [({}, {}), ({"a": .5}, {"b": .5}), ({"a": .5}, {"a": .5, "b": 0})])
def test_bootstrap_rejects_empty_or_unpaired_query_sets(model, baseline):
    with pytest.raises(ValueError, match="same nonempty query ID sets"):
        ev.paired_bootstrap_ndcg10(ndcg_rows(model), ndcg_rows(baseline))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.5", -.1, 1.1])
def test_bootstrap_rejects_invalid_ndcg(value):
    with pytest.raises(ValueError, match="finite NDCG"):
        ev.paired_bootstrap_ndcg10(ndcg_rows({"q": value}), ndcg_rows({"q": .5}))


def test_report_includes_recall10_bootstrap_and_multiplicity_note(tmp_path):
    source, data = fixture_source(tmp_path)
    out = tmp_path / "run"
    execute_fixture(source, out)
    report = ev.make_report(ev.load_source(source, data), [out])
    assert report["protocol"]["paired_bootstrap_ndcg10_vs_baseline"]["resamples"] == 2000
    assert report["protocol"]["paired_bootstrap_ndcg10_vs_baseline"]["multiple_comparisons_adjusted"] is False
    markdown = ev.render_markdown(report)
    assert "Recall@10" in markdown and "Recall@20" in markdown
    assert "95% CI" in markdown and "2000" in markdown and "seed=20260927" in markdown
    assert "Поправка на множественные сравнения не применяется" in markdown
    for group in report["rerankers"]:
        assert group["paired_bootstrap_ndcg10_vs_baseline"]["n_queries"] == group["queries"]
