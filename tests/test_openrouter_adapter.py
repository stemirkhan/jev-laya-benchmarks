import copy
from decimal import Decimal
import json
import os
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import openrouter_adapter as oa


CONFIG = {"model": "typesafe/jev-1.13", "expected_response_model": "typesafe/jev-1.13-20260917",
          "max_requests": 20, "concurrency": 4, "batch_size": 20}
RAW = {"id": "gen-dec-test", "model": "typesafe/jev-1.13-20260917", "provider": "TypeSafe",
       "answers": {"relevance": {"type": "noul", "noul": 0.7}},
       "usage": {"input_tokens": 100, "output_tokens": 0, "cost": 0.0000042}}


class Response:
    def __init__(self, raw=None, *, text=None, code=200):
        self.status_code = code
        self.text = text if text is not None else json.dumps(RAW if raw is None else raw)
        self.headers = {"content-type": "application/json"}
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def budget(tmp_path):
    guard = oa.Budget(tmp_path / "budget.jsonl", "3.50")
    yield guard
    guard.close()


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-private-secret")


def test_round_trip_budget(budget):
    ticket = budget.reserve("a")
    assert budget.snapshot()["reserved_usd"] == str(oa.RESERVATION_USD)
    budget.settle(ticket, "0.001")
    assert budget.snapshot()["spent_or_conservatively_charged_usd"] == "0.001"
    assert budget.snapshot()["reserved_usd"] == "0"


def test_budget_reopen(tmp_path):
    path = tmp_path / "ledger"
    first = oa.Budget(path, "3.50")
    first.settle(first.reserve("pilot"), "0.001")
    first.close()
    second = oa.Budget(path, "3.50")
    assert second.snapshot()["spent_or_conservatively_charged_usd"] == "0.001"
    assert second.reserve("main") == 2
    second.settle(2, "0.0005")
    second.close()


def test_budget_exclusive_lock(tmp_path):
    path = tmp_path / "ledger"
    first = oa.Budget(path, "3.50")
    try:
        with pytest.raises(BlockingIOError):
            oa.Budget(path, "3.50")
    finally:
        first.close()


@pytest.mark.parametrize("limit", ["4", "-1", "NaN", "Infinity", True, 0])
def test_invalid_budget(tmp_path, limit):
    with pytest.raises(oa.BudgetError):
        oa.Budget(tmp_path / "ledger", limit)


@pytest.mark.parametrize("cost", [None, True, "NaN", "-0.1", "Infinity", {}])
def test_unknown_charge_reserves_and_stops(budget, cost):
    ticket = budget.reserve("one")
    with pytest.raises(oa.BudgetError):
        budget.settle(ticket, cost)
    assert budget.spent == oa.RESERVATION_USD and budget.tripped
    with pytest.raises(oa.BudgetError):
        budget.reserve("next")


def test_excess_charge_is_recorded_and_stops(budget):
    ticket = budget.reserve("one")
    with pytest.raises(oa.BudgetError):
        budget.settle(ticket, "0.003")
    assert budget.spent == Decimal("0.003") and budget.tripped


def test_dollar_ceiling_atomic_concurrent_admission(tmp_path):
    guard = oa.Budget(tmp_path / "ledger", oa.RESERVATION_USD * 3)
    def reserve(_):
        try:
            return guard.reserve("same")
        except oa.BudgetError:
            return None
    with ThreadPoolExecutor(max_workers=20) as executor:
        result = list(executor.map(reserve, range(20)))
    assert len([ticket for ticket in result if ticket is not None]) == 3
    assert sum(guard.pending.values()) <= guard.limit
    guard.close()


def test_crash_pending_blocks_restart(tmp_path):
    path = tmp_path / "ledger"
    first = oa.Budget(path, "3.50")
    first.reserve("interrupted")
    first.close()
    with pytest.raises(oa.BudgetError, match="Unresolved"):
        oa.Budget(path, "3.50")


def test_cannot_reset_terms(tmp_path):
    path = tmp_path / "ledger"
    oa.Budget(path, "3.50").close()
    with pytest.raises(oa.BudgetError, match="terms"):
        oa.Budget(path, "3.60")


def test_logging_fails_before_network(budget, monkeypatch):
    called = []
    adapter = oa.OpenRouterAdapter(CONFIG, budget=budget, event_callback=lambda row: None,
                                  transport=lambda *args: called.append(True))
    monkeypatch.setattr(oa.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk full")))
    try:
        with pytest.raises(oa.OpenRouterError):
            adapter.score([("q", "d")])
        assert called == [] and budget.tripped
    finally:
        adapter.close()


def test_callback_fails_before_network(budget):
    called = []
    def fail(event):
        raise RuntimeError("secret should not propagate")
    adapter = oa.OpenRouterAdapter(CONFIG, budget=budget, event_callback=fail,
                                  transport=lambda *args: called.append(True))
    try:
        with pytest.raises(oa.OpenRouterError, match="not sent"):
            adapter.score([("q", "d")])
        assert called == [] and budget.tripped and budget.spent == 0
    finally:
        adapter.close()


def test_success_records_full_payload_and_order(budget):
    events, responses, sent = [], [], []
    def transport(body, headers, timeout):
        payload = json.loads(body)
        assert payload["model"] == CONFIG["model"]
        assert headers["Authorization"] == "Bearer test-private-secret"
        assert budget.pending  # durably reserved before HTTP
        sent.append(payload)
        raw = copy.deepcopy(RAW)
        raw["answers"]["relevance"]["noul"] = int(payload["state"]["query"])/10
        response = Response(raw)
        responses.append(response)
        return response
    adapter = oa.OpenRouterAdapter(CONFIG, budget=budget, event_callback=events.append, transport=transport)
    try:
        result = adapter.score([(str(i), "Документ" * 100) for i in range(10)])
        assert [row["score"] for row in result] == [i/10 for i in range(10)]
        assert len(events) == 20 and len(sent) == 10
        assert all(r.closed for r in responses)
        assert budget.spent == Decimal("0.0000420")
        assert "test-private-secret" not in json.dumps(events)
    finally:
        adapter.close()


@pytest.mark.parametrize("change", [
    {"model": "other"}, {"provider": "other"}, {"answers": {}},
    {"answers": {"relevance": {"type": "noul", "noul": True}}},
    {"answers": {"relevance": {"type": "noul", "noul": 2}}},
    {"usage": {"input_tokens": 32001, "output_tokens": 0, "cost": 0.000001}},
    {"usage": {"input_tokens": True, "output_tokens": 0, "cost": 0.000001}},
    {"usage": {"input_tokens": 10, "output_tokens": 0, "cost": "0.000001"}},
    {"usage": {"input_tokens": 10, "output_tokens": 0}},
])
def test_response_fail_closed(budget, change):
    events, calls = [], []
    def transport(*args):
        calls.append(True)
        return Response({**copy.deepcopy(RAW), **change})
    adapter = oa.OpenRouterAdapter({**CONFIG, "concurrency": 1}, budget=budget, event_callback=events.append, transport=transport)
    try:
        with pytest.raises(oa.OpenRouterError):
            adapter.score([("q", "d")] * 20)
        assert len(calls) == 1 and budget.tripped and not budget.pending
        assert events[-1]["status"] == "error"
    finally:
        adapter.close()


@pytest.mark.parametrize("text", ["{", '{"model":"x","model":"y"}', '{"cost":NaN}'])
def test_malformed_response_unknown_charge(budget, text):
    adapter = oa.OpenRouterAdapter(CONFIG, budget=budget, event_callback=lambda row: None,
                                  transport=lambda *args: Response(text=text))
    try:
        with pytest.raises(oa.OpenRouterError):
            adapter.score([("q", "d")])
        assert budget.tripped and budget.spent == oa.RESERVATION_USD
    finally:
        adapter.close()


def test_transport_exception_does_not_leak_secret(budget):
    events = []
    def transport(*args):
        raise RuntimeError("test-private-secret")
    adapter = oa.OpenRouterAdapter(CONFIG, budget=budget, event_callback=events.append, transport=transport)
    try:
        with pytest.raises(oa.OpenRouterError) as error:
            adapter.score([("q", "d")])
        assert "test-private-secret" not in str(error.value) + json.dumps(events)
        assert budget.tripped and budget.spent == oa.RESERVATION_USD
    finally:
        adapter.close()


def test_response_secret_redaction(budget):
    raw = {**RAW, "note": "test-private-secret", "authorization": "anything"}
    events = []
    adapter = oa.OpenRouterAdapter(CONFIG, budget=budget, event_callback=events.append,
                                  transport=lambda *args: Response(raw))
    try:
        values = adapter.score([("q", "d")])
        assert "test-private-secret" not in json.dumps(events) + json.dumps(values)
        assert values[0]["raw"]["authorization"] == "[REDACTED]"
    finally:
        adapter.close()


def test_request_limit_checked_before_any_send(budget):
    calls = []
    adapter = oa.OpenRouterAdapter({**CONFIG, "max_requests": 1}, budget=budget, event_callback=lambda row: None,
                                  transport=lambda *args: calls.append(True))
    try:
        with pytest.raises(oa.OpenRouterError):
            adapter.score([("q", "d")] * 2)
        assert not calls and not budget.pending
    finally:
        adapter.close()


def test_transport_disables_redirect_retry_proxy(budget, monkeypatch):
    import requests
    seen = {}
    class Session:
        trust_env = True
        def mount(self, prefix, adapter):
            seen["retry"] = adapter.max_retries.total
        def post(self, url, **kwargs):
            seen.update(url=url, trust_env=self.trust_env, **kwargs)
            return Response()
        def close(self):
            pass
    monkeypatch.setattr(requests, "Session", Session)
    adapter = oa.OpenRouterAdapter(CONFIG, budget=budget, event_callback=lambda row: None)
    try:
        adapter.score([("q", "d")])
        assert seen["url"] == oa.ENDPOINT and seen["allow_redirects"] is False
        assert seen["trust_env"] is False and seen["retry"] == 0
    finally:
        adapter.close()


def test_finish_callback_failure_stops_further_calls(budget):
    calls = []
    def callback(event):
        if event["phase"] == "finish":
            raise OSError("disk full")
    def transport(*args):
        calls.append(True)
        return Response()
    adapter = oa.OpenRouterAdapter({**CONFIG, "concurrency": 1}, budget=budget, event_callback=callback, transport=transport)
    try:
        with pytest.raises(oa.OpenRouterError):
            adapter.score([("q", "d")] * 20)
        assert len(calls) == 1 and budget.tripped
        assert budget.spent == Decimal("0.0000042")
    finally:
        adapter.close()
