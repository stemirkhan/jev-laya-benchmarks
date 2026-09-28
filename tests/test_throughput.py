import copy
import gzip
import hashlib
import json
import signal
import subprocess

import pytest

import freeze_protocol
import rerank
import throughput


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "candidates"
    folder = source / "demo"
    folder.mkdir(parents=True)
    qids = [f"q{i}" for i in range(12)]
    retrieval = {"datasets": ["demo"], "search": {"top_k": 20}}
    rerank.write_json(source / "config.json", retrieval)
    rerank.write_json(folder / "queries.json", {"pilot": qids})
    (folder / "query-texts.jsonl").write_text("".join(
        json.dumps({"query_id": qid, "text": f"query {qid}"}) + "\n" for qid in qids))
    with gzip.open(folder / "pairs.pilot.jsonl.gz", "wt") as stream:
        for qid in qids:
            for i in range(20):
                document = f"Complete document {i}\nwith trailing whitespace. "
                stream.write(json.dumps({"query_id": qid, "doc_id": str(i), "query": f"query {qid}",
                    "document": document, "text_sha256": hashlib.sha256(document.encode()).hexdigest()}) + "\n")
    docs = [{"doc_id": str(i), "rank": i + 1, "score": 20 - i} for i in range(20)]
    for method in ("bm25", "rrf"):
        (folder / f"candidates.pilot.{method}.jsonl").write_text("".join(
            json.dumps({"query_id": qid, "candidates": docs}) + "\n" for qid in qids))
    rerank.write_json(source / "summary.json", {"config_sha256": rerank.fingerprint(retrieval),
        "artifacts_sha256": {str(path.relative_to(source)): rerank.sha(path) for path in folder.iterdir()}})
    bundle = rerank.Candidates(source, "pilot")
    config = rerank.read_json(rerank.ROOT / "config/rerank-pilot.json")
    protocol = {"schema": freeze_protocol.SCHEMA, "status": "frozen_after_reviewed_pilots",
                "reviewed_pilots": True, "test_metrics_used_for_tuning": False,
                "pilot_source": bundle.plan(), "inference_code_sha256": freeze_protocol.inference_hashes(),
                "runner_code_sha256": rerank.source_hashes(),
                "models": {model: {"config": freeze_protocol.effective_config(config, model),
                    "environment": rerank.environment(), "adapter": {"model": model, "fixture": True}}
                    for model in throughput.BATCH_SIZES}}
    protocol["protocol_sha256"] = rerank.fingerprint(protocol)
    return bundle, config, protocol


class FakeAdapter:
    def __init__(self, metadata, *, fail_at=None, invalid_at=None, deadline_at=None):
        self.info = metadata
        self.calls = []
        self.closed = False
        self.syncs = 0
        self.fail_at = fail_at
        self.invalid_at = invalid_at
        self.deadline_at = deadline_at

    def metadata(self):
        return copy.deepcopy(self.info)

    def synchronize(self):
        self.syncs += 1

    def score(self, pairs):
        self.calls.append(list(pairs))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("fixture inference failure")
        if len(self.calls) == self.deadline_at:
            raise throughput.DeadlineExceeded()
        return [{"score": "invalid" if len(self.calls) == self.invalid_at else 0.25,
                 "input_tokens": 10, "raw": {"fixture": True, "source_document": doc}}
                for query, doc in pairs]

    def close(self):
        self.closed = True


@pytest.mark.parametrize("model", ["bge", "qwen-8b"])
def test_three_full_pilot_streams_after_warmup_preserve_order_and_rates(tmp_path, setup, model):
    bundle, config, protocol = setup
    before = rerank.source_hashes()
    adapter = FakeAdapter(protocol["models"][model]["adapter"])
    out = tmp_path / "throughput"
    summary = throughput.execute(bundle, out, config, model, protocol, lambda: adapter)
    assert summary["complete"] and summary["measured_pairs"] == 1440 and summary["warmup_pairs"] == 40
    assert summary["completed_repetitions"] == 3 and len(summary["measurements"]) == 3
    assert summary["pairs_per_second"] == 1440 / summary["total_measured_seconds"]
    assert summary["total_measured_seconds"] == sum(row["seconds"] for row in summary["measurements"])
    assert all(row["pairs"] == 480 for row in summary["measurements"])
    stream = [(pair["query"], pair["document"]) for item in bundle.lists for pair in item["pairs"]]
    assert [pair for batch in adapter.calls for pair in batch] == stream[:40] + stream * 3
    assert all(len(batch) == throughput.BATCH_SIZES[model] for batch in adapter.calls)
    assert adapter.closed and adapter.syncs >= len(adapter.calls) + 9
    batches = rerank.rows(out / "batches.jsonl")
    assert sum(len(row["raw_results"]) for row in batches) == 1480
    assert all(row["attempt"] == 1 for row in batches)
    assert len(rerank.read_json(out / "manifest.json")["input_order"]) == 480
    assert summary["events_sha256"] == rerank.sha(out / "events.jsonl")
    assert summary["batches_sha256"] == rerank.sha(out / "batches.jsonl")
    assert (out / "source/throughput.py").exists() and before == rerank.source_hashes()
    assert not (out / "partial.json").exists()
    with pytest.raises(ValueError, match="no automatic resume or retry"):
        throughput.execute(bundle, out, config, model, protocol, lambda: pytest.fail("no second attempt"))


def test_deadline_keeps_successful_batches_and_started_attempt(tmp_path, setup):
    bundle, config, protocol = setup
    # Ten warmup batches, one measured success, then an interrupted measured attempt.
    adapter = FakeAdapter(protocol["models"]["bge"]["adapter"], deadline_at=12)
    out = tmp_path / "partial"
    result = throughput.execute(bundle, out, config, "bge", protocol, lambda: adapter)
    assert not result["complete"] and result["reason"] == "max_runtime_seconds"
    assert result["measured_pairs"] == 4 and result["warmup_pairs"] == 40 and adapter.closed
    assert result["pairs_per_second"] is None and not (out / "summary.json").exists()
    assert (out / "partial.json").exists()
    assert len(rerank.rows(out / "batches.jsonl")) == 11
    events = rerank.rows(out / "events.jsonl")
    assert events[-2]["event"] == "batch_start" and events[-1]["event"] == "runtime_stop"
    with pytest.raises(ValueError, match="Existing throughput attempt"):
        throughput.execute(bundle, out, config, "bge", protocol, lambda: pytest.fail("no retry"))


def test_deadline_includes_cold_load_without_running_scores(tmp_path, setup, monkeypatch):
    bundle, config, protocol = setup
    now = [0.0]
    monkeypatch.setattr(throughput.time, "monotonic", lambda: now[0])
    adapter = FakeAdapter(protocol["models"]["bge"]["adapter"])
    def load():
        now[0] = 3.0
        return adapter
    result = throughput.execute(bundle, tmp_path / "run", config, "bge", protocol, load, max_runtime_seconds=2)
    assert not result["complete"] and result["measured_pairs"] == result["warmup_pairs"] == 0
    assert not adapter.calls and adapter.closed


@pytest.mark.parametrize("invalid", [False, True])
def test_failure_and_invalid_raw_scores_retained_without_retry(tmp_path, setup, invalid):
    bundle, config, protocol = setup
    adapter = FakeAdapter(protocol["models"]["bge"]["adapter"],
                          invalid_at=11 if invalid else None, fail_at=None if invalid else 11)
    out = tmp_path / "failed"
    with pytest.raises((RuntimeError, ValueError)):
        throughput.execute(bundle, out, config, "bge", protocol, lambda: adapter)
    assert adapter.closed and len(adapter.calls) == 11
    assert not (out / "summary.json").exists() and (out / "failure.json").exists()
    assert rerank.rows(out / "events.jsonl")[-1]["event"] == "error"
    batches = rerank.rows(out / "batches.jsonl")
    assert len(batches) == (11 if invalid else 10)
    if invalid:
        assert batches[-1]["raw_results"][0]["score"] == "invalid"


@pytest.mark.parametrize("limit", [0, -1, 181, float("inf"), float("nan"), True])
def test_budget_must_be_at_most_180_before_output_or_model_load(tmp_path, setup, limit):
    bundle, config, protocol = setup
    out = tmp_path / "bad-budget"
    with pytest.raises(ValueError, match="at most 180"):
        throughput.execute(bundle, out, config, "bge", protocol, lambda: pytest.fail("no load"), max_runtime_seconds=limit)
    assert not out.exists()


def resign(protocol):
    protocol["protocol_sha256"] = rerank.fingerprint({key: value for key, value in protocol.items() if key != "protocol_sha256"})


def test_frozen_gate_rejects_changes_without_loading(tmp_path, setup, monkeypatch):
    bundle, config, protocol = setup
    changed = copy.deepcopy(config)
    changed["models"]["bge"]["dtype"] = "fp32"
    with pytest.raises(ValueError, match="Settings differ"):
        throughput.execute(bundle, tmp_path / "changed-config", changed, "bge", protocol, lambda: pytest.fail("no load"))
    with pytest.raises(ValueError, match="frozen reviewed-pilot protocol"):
        throughput.execute(bundle, tmp_path / "no-protocol", config, "bge", None, lambda: pytest.fail("no load"))
    changed_protocol = copy.deepcopy(protocol)
    changed_protocol["models"]["bge"]["config"]["model_config"]["dtype"] = "fp32"
    with pytest.raises(ValueError, match="checksum differs"):
        throughput.authorized_model(bundle, config, "bge", changed_protocol)
    with monkeypatch.context() as patch:
        patch.setattr(rerank, "environment", lambda: {"changed": True})
        with pytest.raises(ValueError, match="Environment differs"):
            throughput.authorized_model(bundle, config, "bge", protocol)
    with monkeypatch.context() as patch:
        patch.setattr(rerank, "source_hashes", lambda: {"changed": "code"})
        with pytest.raises(ValueError, match="Code/instructions/dependencies differ"):
            throughput.authorized_model(bundle, config, "bge", protocol)
    changed_protocol = copy.deepcopy(protocol)
    changed_protocol["pilot_source"]["retrieval_summary_sha256"] = "changed"
    resign(changed_protocol)
    with pytest.raises(ValueError, match="exact frozen pilot"):
        throughput.authorized_model(bundle, config, "bge", changed_protocol)
    assert not (tmp_path / "changed-config").exists()


def test_hardware_mismatch_is_journaled_before_any_scores(tmp_path, setup):
    bundle, config, protocol = setup
    fake = FakeAdapter({"different": "hardware"})
    out = tmp_path / "hardware"
    with pytest.raises(ValueError, match="Adapter/hardware differs"):
        throughput.execute(bundle, out, config, "bge", protocol, lambda: fake)
    assert fake.closed and not fake.calls
    assert rerank.read_json(out / "adapter.json") == {"different": "hardware"}
    assert (out / "failure.json").exists()


def test_alarm_installs_and_restores_timeout_handler_without_waiting():
    before = signal.getsignal(signal.SIGALRM)
    with pytest.raises(throughput.DeadlineExceeded):
        with throughput.deadline_alarm(throughput.time.monotonic() + 10):
            handler = signal.getsignal(signal.SIGALRM)
            handler(signal.SIGALRM, None)
    assert signal.getsignal(signal.SIGALRM) == before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_hard_watchdog_retains_partial_raw_tail_and_never_retries(tmp_path, monkeypatch, capsys):
    out = tmp_path / "killed-worker"
    calls = []
    def timed_out(command, *, timeout):
        calls.append((command, timeout))
        out.mkdir()
        rerank.append(out / "events.jsonl", {"event": "batch_success", "phase": "warmup", "pair_count": 4})
        rerank.append(out / "events.jsonl", {"event": "batch_success", "phase": "measure", "pair_count": 4})
        with (out / "events.jsonl").open("a") as dest:
            dest.write('{"event":')
        (out / "batches.jsonl").write_text('{"unfinished_raw":')
        raise subprocess.TimeoutExpired(command, timeout)
    monkeypatch.setattr(throughput.subprocess, "run", timed_out)
    assert throughput.supervise(["--model", "bge"], out, 180) == 75
    assert len(calls) == 1 and calls[0][1] == 180 and calls[0][0][-1] == "--worker"
    result = rerank.read_json(out / "partial.json")
    assert result["reason"] == "hard_process_deadline" and result["incomplete_event_tail"]
    assert result["measured_pairs"] == result["warmup_pairs"] == 4
    assert (out / "events.jsonl").read_text().endswith('{"event":')
    assert (out / "batches.jsonl").read_text() == '{"unfinished_raw":'
    assert json.loads(capsys.readouterr().out)["complete"] is False
    with pytest.raises(ValueError, match="no automatic resume or retry"):
        throughput.supervise([], out, 180)
    assert len(calls) == 1


def test_supervisor_forwards_successful_exit_without_extra_attempt(tmp_path, monkeypatch):
    calls = []
    def finished(command, *, timeout):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(throughput.subprocess, "run", finished)
    assert throughput.supervise(["--model", "bge"], tmp_path / "output", 10) == 0
    assert len(calls) == 1
