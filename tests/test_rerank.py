import gzip
import hashlib
import json
from pathlib import Path

import pytest

import rerank


def fixture_candidates(path):
    folder = path / "demo"
    folder.mkdir(parents=True)
    config = {"datasets": ["demo"], "search": {"top_k": 20}}
    rerank.write_json(path / "config.json", config)
    rerank.write_json(folder / "queries.json", {"pilot": ["q"], "test": ["t"]})
    (folder / "query-texts.jsonl").write_text("".join(
        json.dumps({"query_id": qid, "text": "test query"}) + "\n" for qid in ("q", "t")))
    for split, qid in (("pilot", "q"), ("test", "t")):
        with gzip.open(folder / f"pairs.{split}.jsonl.gz", "wt") as f:
            for i in range(20):
                document = f"Full text {i}\nwith preserved whitespace. "
                f.write(json.dumps({"query_id": qid, "doc_id": str(i), "query": "test query", "document": document,
                                    "text_sha256": hashlib.sha256(document.encode()).hexdigest()}) + "\n")
        docs = [{"doc_id": str(i), "rank": i + 1, "score": 20 - i} for i in range(20)]
        for method in ("bm25", "rrf"):
            (folder / f"candidates.{split}.{method}.jsonl").write_text(json.dumps({"query_id": qid, "candidates": docs}) + "\n")
    rerank.write_json(path / "summary.json", {"config_sha256": rerank.fingerprint(config),
                      "artifacts_sha256": {str(f.relative_to(path)): rerank.sha(f) for f in folder.iterdir()}})
    return rerank.Candidates(path)


class FakeAdapter:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.closed = False
        self.fail_at = fail_at

    def score(self, pairs):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("simulated failure")
        return [{"score": float(i), "raw": {"fixture": True}} for i, _ in enumerate(pairs)]

    def synchronize(self):
        pass

    def metadata(self):
        return {"model": "fake-not-a-measurement"}

    def close(self):
        self.closed = True


def config():
    return {"model": "fake", "batch_size": 20, "warmup_lists": 1, "repeats": 1}


def test_candidates_no_qrels_and_exact_text(tmp_path):
    bundle = fixture_candidates(tmp_path)
    assert bundle.plan()["lists"] == 2
    assert bundle.plan()["unique_pairs"] == 20
    assert bundle.plan()["pair_occurrences"] == 40
    assert bundle.pairs[("demo", "q", "0")]["document"].endswith(" ")


def test_changed_artifact_rejected(tmp_path):
    fixture_candidates(tmp_path)
    with (tmp_path / "demo/candidates.pilot.bm25.jsonl").open("a") as out:
        out.write("\n")
    with pytest.raises(ValueError, match="Changed candidate"):
        rerank.Candidates(tmp_path)


def test_ties_keep_search_rank():
    docs = [{"doc_id": "z", "rank": 1}, {"doc_id": "a", "rank": 2}, {"doc_id": "c", "rank": 3}]
    result = rerank.sort_results(docs, [{"score": .5}, {"score": .5}, {"score": .9}])
    assert [r["doc_id"] for r in result] == ["c", "z", "a"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.5"])
def test_bad_score_rejected(value):
    with pytest.raises(ValueError):
        rerank.sort_results([{"doc_id": "a", "rank": 1}], [{"score": value}])


def test_missing_scores_rejected():
    with pytest.raises(ValueError):
        rerank.sort_results([{"doc_id": "a", "rank": 1}], [])


def test_complete_run_and_no_repeated_inference(tmp_path):
    bundle = fixture_candidates(tmp_path / "source")
    fake = FakeAdapter()
    out = tmp_path / "run"
    summary = rerank.execute(bundle, out, config(), lambda *a: fake)
    assert summary["complete"] and summary["lists"] == 2
    assert fake.calls == 3 and fake.closed
    assert len(rerank.rows(out / "lists.jsonl")) == 3
    rerank.execute(bundle, out, config(), lambda *a: pytest.fail("Must not initialize completed run"))
    with pytest.raises(ValueError, match="changed"):
        rerank.execute(bundle, out, {**config(), "batch_size": 4}, lambda *a: fake)


def test_failed_list_kept_requires_explicit_retry(tmp_path):
    bundle = fixture_candidates(tmp_path / "source")
    out = tmp_path / "run"
    fake = FakeAdapter(fail_at=2)
    with pytest.raises(RuntimeError):
        rerank.execute(bundle, out, config(), lambda *a: fake)
    assert fake.closed
    assert not (out / "summary.json").exists()
    assert len(rerank.rows(out / "lists.jsonl")) == 1
    with pytest.raises(ValueError, match="Interrupted/failed"):
        rerank.execute(bundle, out, config(), lambda *a: fake)
    recovered = FakeAdapter()
    rerank.execute(bundle, out, config(), lambda *a: recovered, retry_failed=True)
    assert recovered.calls == 3  # fresh warmup + two complete measured lists
    assert any(e["event"] == "list_error" for e in rerank.rows(out / "events.jsonl"))
    rerank.execute(bundle, out, config(), lambda *a: pytest.fail("already complete"))


def test_api_request_count_survives_restart(tmp_path):
    bundle = fixture_candidates(tmp_path / "source")
    out = tmp_path / "run"
    remaining_limits = []
    def failed_factory(callback, remaining):
        remaining_limits.append(remaining)
        class Fail(FakeAdapter):
            def score(self, pairs):
                callback({"phase": "start", "request_index": 1})
                callback({"phase": "finish", "request_index": 1, "ok": False})
                raise RuntimeError("HTTP failure")
        return Fail()
    with pytest.raises(RuntimeError):
        rerank.execute(bundle, out, config(), failed_factory, api_limit=10)
    def success_factory(callback, remaining):
        remaining_limits.append(remaining)
        return FakeAdapter()
    rerank.execute(bundle, out, config(), success_factory, retry_failed=True, api_limit=10)
    assert remaining_limits == [10, 9]


def test_run_lock_prevents_parallel_writer(tmp_path):
    with rerank.exclusive_run(tmp_path):
        with pytest.raises(ValueError, match="already active"):
            with rerank.exclusive_run(tmp_path):
                pass


def test_incomplete_jsonl_not_silently_dropped(tmp_path):
    bundle = fixture_candidates(tmp_path / "source")
    out = tmp_path / "run"
    rerank.execute(bundle, out, config(), lambda *a: FakeAdapter())
    with (out / "events.jsonl").open("a") as f:
        f.write('{"unfinished":')
    with pytest.raises(json.JSONDecodeError):
        rerank.execute(bundle, out, config(), lambda *a: FakeAdapter())


def test_keyboard_interrupt_keeps_successful_batch(tmp_path):
    bundle = fixture_candidates(tmp_path / "source")
    out = tmp_path / "run"
    class Interrupt(FakeAdapter):
        def score(self, pairs):
            if self.calls == 1:
                raise KeyboardInterrupt()
            return super().score(pairs)
    fake = Interrupt()
    with pytest.raises(KeyboardInterrupt):
        rerank.execute(bundle, out, {**config(), "batch_size": 4}, lambda *a: fake)
    assert fake.closed
    events = rerank.rows(out / "events.jsonl")
    assert len([e for e in events if e["event"] == "batch"]) == 1
    assert events[-1]["error_type"] == "KeyboardInterrupt"


def test_inline_credential_rejected_before_any_file_is_written(tmp_path):
    bundle = fixture_candidates(tmp_path / "source")
    out = tmp_path / "run"
    bad = {**config(), "model": "jev", "model_config": {"api_key": "fake-must-not-persist"}}
    with pytest.raises(ValueError, match="inline credential"):
        rerank.execute(bundle, out, bad, lambda *a: FakeAdapter(), api_limit=20)
    assert not out.exists()


def test_runtime_stops_between_lists_and_resumes_without_retry(tmp_path, monkeypatch):
    bundle = fixture_candidates(tmp_path / "source")
    now = [0.0]
    monkeypatch.setattr(rerank.time, "monotonic", lambda: now[0])
    class Timed(FakeAdapter):
        def score(self, pairs):
            now[0] += 1
            return super().score(pairs)
    first = Timed()
    out = tmp_path / "run"
    partial = rerank.execute(bundle, out, config(), lambda *a: first, max_runtime_seconds=1.5)
    assert not partial["complete"] and partial["completed_lists"] == 1
    assert first.closed and first.calls == 2
    assert not (out / "summary.json").exists()
    assert rerank.rows(out / "events.jsonl")[-1]["event"] == "runtime_stop"
    second = Timed()
    complete = rerank.execute(bundle, out, config(), lambda *a: second, max_runtime_seconds=3)
    assert complete["complete"] and second.calls == 2  # fresh warmup + remaining list
    assert len([r for r in rerank.rows(out / "lists.jsonl") if r["phase"] == "measure"]) == 2
    assert any(e["event"] == "runtime_stop" for e in rerank.rows(out / "events.jsonl"))


def test_runtime_includes_model_load_and_closes_adapter(tmp_path, monkeypatch):
    bundle = fixture_candidates(tmp_path / "source")
    now = [0.0]
    monkeypatch.setattr(rerank.time, "monotonic", lambda: now[0])
    fake = FakeAdapter()
    def load(*args):
        now[0] += 10
        return fake
    result = rerank.execute(bundle, tmp_path / "run", config(), load, max_runtime_seconds=1)
    assert not result["complete"] and fake.calls == 0 and fake.closed
    assert not (tmp_path / "run/lists.jsonl").exists()


def test_runtime_stop_before_load_resumes_without_spurious_warmup(tmp_path, monkeypatch):
    bundle = fixture_candidates(tmp_path / "source")
    clock = iter([0.0, 2.0, 2.0])
    monkeypatch.setattr(rerank.time, "monotonic", lambda: next(clock, 2.0))
    out = tmp_path / "run"
    result = rerank.execute(bundle, out, config(), lambda *a: pytest.fail("deadline passed"),
                            max_runtime_seconds=1)
    assert not result["complete"]
    fake = FakeAdapter()
    rerank.execute(bundle, out, config(), lambda *a: fake, max_runtime_seconds=1)
    assert fake.calls == 3
    assert not any(row["job_id"].startswith("resume_warmup/") for row in rerank.rows(out / "lists.jsonl"))


def test_cli_runtime_stop_has_nonzero_exit_status(tmp_path, monkeypatch, capsys):
    fixture_candidates(tmp_path / "source")
    settings = {"models": {"bge": {"batch_size": 20}}, "status": "pilot", "warmup_lists": 1, "repeats": 1}
    rerank.write_json(tmp_path / "config.json", settings)
    monkeypatch.setattr(rerank, "execute", lambda *args, **kwargs: {"complete": False, "reason": "max_runtime_seconds"})
    monkeypatch.setattr(rerank.sys, "argv", ["rerank.py", "run", "--source", str(tmp_path / "source"),
                        "--config", str(tmp_path / "config.json"), "--model", "bge", "--out", str(tmp_path / "run"),
                        "--max-runtime-seconds", "1"])
    with pytest.raises(SystemExit) as stopped:
        rerank.main()
    assert stopped.value.code == 75
    assert json.loads(capsys.readouterr().out)["complete"] is False


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf"), True])
def test_invalid_runtime_budget_rejected_without_loading(tmp_path, limit):
    bundle = fixture_candidates(tmp_path / "source")
    with pytest.raises(ValueError, match="max_runtime_seconds"):
        rerank.execute(bundle, tmp_path / "run", config(), lambda *a: pytest.fail("no load"),
                       max_runtime_seconds=limit)
    assert not (tmp_path / "run").exists()


def test_direct_test_execute_requires_frozen_protocol(tmp_path):
    fixture_candidates(tmp_path / "source")
    bundle = rerank.Candidates(tmp_path / "source", "test")
    with pytest.raises(ValueError, match="Main test is locked"):
        rerank.execute(bundle, tmp_path / "run", config(), lambda *a: pytest.fail("no load"))
    assert not (tmp_path / "run").exists()
