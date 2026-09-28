import copy
import json
from decimal import Decimal

import pytest

import openrouter_adapter as oa
import openrouter_run as run
import rerank
import freeze_protocol
from test_rerank import fixture_candidates
from test_freeze_protocol import pilots


MODEL = {"id": "typesafe/jev-1.13", "endpoints": [{
    "name": "TypeSafe | typesafe/jev-1.13-20260917", "provider_name": "TypeSafe",
    "context_length": 32000, "pricing": {"prompt": "0.000000042", "completion": "0"}}]}
RAW = {"id": "offline-fixture", "model": "typesafe/jev-1.13-20260917", "provider": "TypeSafe",
       "answers": {"relevance": {"type": "noul", "noul": .5}},
       "usage": {"input_tokens": 10, "output_tokens": 0, "cost": .00000042}}


class Response:
    status_code = 200
    headers = {}
    def __init__(self, body):
        self.body, self.text, self.closed = body, json.dumps(body), False
    def json(self):
        return self.body
    def close(self):
        self.closed = True


class Session:
    def __init__(self, model):
        self.response = Response({"data": model})
        self.calls = []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-only-key")
    monkeypatch.setattr(run, "tariff_check", lambda: {"fixture": True})
    monkeypatch.setattr(oa.OpenRouterAdapter, "_http_post", lambda *a: Response(copy.deepcopy(RAW)))


def test_public_tariff_check_never_sends_account_credentials():
    session = Session(copy.deepcopy(MODEL))
    result = run.tariff_check(session)
    assert result["model"] == MODEL and session.response.closed and session.trust_env is False
    assert len(session.calls) == 1
    assert "headers" not in session.calls[0][1] and session.calls[0][1]["allow_redirects"] is False
    assert session.calls[0][0].endswith("/endpoints")


@pytest.mark.parametrize("change", ["price", "context", "provider", "model", "duplicate", "redirect"])
def test_changed_tariff_fails_closed(change):
    model = copy.deepcopy(MODEL)
    endpoint = model["endpoints"][0]
    if change == "price":
        endpoint["pricing"]["prompt"] = "1"
    elif change == "context":
        endpoint["context_length"] = 100000
    elif change == "provider":
        endpoint["provider_name"] = "another-provider"
    elif change == "model":
        model["id"] = "another-model"
    elif change == "duplicate":
        model["endpoints"].append(copy.deepcopy(endpoint))
    session = Session(model)
    if change == "redirect":
        session.response.status_code = 302
    with pytest.raises(ValueError):
        run.tariff_check(session)
    assert session.response.closed


def test_plan_is_offline_and_does_not_write(tmp_path, monkeypatch):
    source = tmp_path / "candidates"
    fixture_candidates(source)
    monkeypatch.setattr(run, "tariff_check", lambda: pytest.fail("plan is offline"))
    result = run.plan(source, "test")
    assert result["pair_calls_including_warmup"] == 80 and result["paid_calls"] is False
    assert sorted(p.name for p in tmp_path.iterdir()) == ["candidates"]


@pytest.mark.parametrize("field,value", [
    ("allow_paid_api", False), ("budget_usd", "4"), ("budget_usd", "0"),
    ("max_api_requests", 0), ("max_runtime_seconds", 0), ("max_runtime_seconds", float("nan")),
])
def test_explicit_paid_and_bounded_guards_before_any_network(tmp_path, monkeypatch, field, value):
    kwargs = dict(allow_paid_api=True, budget_usd="1", max_api_requests=80, max_runtime_seconds=60)
    kwargs[field] = value
    monkeypatch.setattr(run, "tariff_check", lambda: pytest.fail("must reject before network"))
    with pytest.raises(ValueError):
        run.run(tmp_path / "missing", "pilot", tmp_path / "out", tmp_path / "ledger", **kwargs)
    assert not list(tmp_path.iterdir())


def test_test_requires_protocol_before_network(tmp_path, monkeypatch):
    source = tmp_path / "candidates"
    fixture_candidates(source)
    monkeypatch.setattr(run, "tariff_check", lambda: pytest.fail("must reject before network"))
    with pytest.raises(ValueError, match="locked"):
        run.run(source, "test", tmp_path / "out", tmp_path / "ledger", "1", 80, 60, allow_paid_api=True)
    assert not (tmp_path / "ledger").exists()


def test_wrong_request_cap_rejected_before_network(tmp_path, monkeypatch):
    source = tmp_path / "candidates"
    fixture_candidates(source)
    monkeypatch.setattr(run, "tariff_check", lambda: pytest.fail("must reject before network"))
    with pytest.raises(ValueError, match="exactly 80"):
        run.run(source, "pilot", tmp_path / "out", tmp_path / "ledger", "1", 81, 60, allow_paid_api=True)


def test_complete_pilot_freeze_and_test_with_unchanged_computation(tmp_path, pilots, offline):
    source, config, gpu_pilots = pilots
    base = tmp_path / "base.json"
    freeze_protocol.freeze(gpu_pilots, source, config, base, reviewed=True)
    pilot, ledger = tmp_path / "openrouter-pilot", tmp_path / "ledger.jsonl"
    result = run.run(source, "pilot", pilot, ledger, "1", 80, 60, allow_paid_api=True)
    assert result["complete"] and result["pair_occurrences"] == 40
    evidence = run.review_pilot(pilot, source)
    assert evidence["requests_verified"] == 80 and evidence["full_inputs_verified"]
    protocol_path = tmp_path / "openrouter-frozen.json"
    with pytest.raises(ValueError, match="Review"):
        run.freeze(pilot, source, base, protocol_path)
    protocol = run.freeze(pilot, source, base, protocol_path, reviewed=True)
    result = run.run(source, "test", tmp_path / "main", ledger, "1", 80, 60,
                     allow_paid_api=True, protocol=protocol)
    assert result["complete"] and result["split"] == "test"
    budget = oa.Budget(ledger, "1")
    assert budget.snapshot()["requests_reserved"] == 160
    assert Decimal(budget.snapshot()["spent_or_conservatively_charged_usd"]) == Decimal(".00000042") * 160
    budget.close()
    with pytest.raises(ValueError, match="overwrite"):
        run.run(source, "pilot", pilot, ledger, "1", 80, 60, allow_paid_api=True)
    with pytest.raises(ValueError, match="overwrite"):
        run.freeze(pilot, source, base, protocol_path, reviewed=True)
    with pytest.raises(ValueError, match="shared ledger"):
        run.run(source, "test", tmp_path / "other-main", tmp_path / "new-ledger", "1", 80, 60,
                allow_paid_api=True, protocol=protocol)
    assert not (tmp_path / "new-ledger").exists()


def test_review_rejects_tampered_full_input(tmp_path, offline):
    source = tmp_path / "candidates"
    fixture_candidates(source)
    pilot = tmp_path / "pilot"
    run.run(source, "pilot", pilot, tmp_path / "ledger", "1", 80, 60, allow_paid_api=True)
    events = rerank.rows(pilot / "events.jsonl")
    event = next(e for e in events if e["event"] == "api" and e["detail"]["phase"] == "start")
    event["detail"]["request"]["state"]["document"] = "changed"
    (pilot / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    summary = rerank.read_json(pilot / "summary.json")
    summary["events_sha256"] = rerank.sha(pilot / "events.jsonl")
    rerank.write_json(pilot / "summary.json", summary)
    with pytest.raises(ValueError, match="Full request"):
        run.review_pilot(pilot, source)
