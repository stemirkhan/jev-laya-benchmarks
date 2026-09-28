"""No-network checks. Fake credentials and an injected fake HTTP transport only."""
import json
import threading
import time

import pytest

from audit import QUESTION
from jev_adapter import (BASE_URL, JevAdapter, JevError, build_jev_payload,
                         parse_jev_response, preflight_jev, validate_jev_config)


FAKE_KEY = "fake-test-key-not-valid"


def config(**kw):
    return {"model": "jev-1.13.0", "base_url": BASE_URL, "api_key_env": "JEV_UNIT_TEST_KEY",
            "timeout_s": 3, "concurrency": 1, "max_requests": 20, **kw}


def answer(score=.75, **kw):
    return {"model": "jev-1.13.0", "answers": {"relevance": {"type": "noul", "noul": score}},
            "usage": {"input_tokens": 42, "output_tokens": 3}, **kw}


class Response:
    def __init__(self, body=None, *, status=200, text=None):
        self.status_code = status
        self.text = text if text is not None else json.dumps(answer() if body is None else body)
        self.headers = {"x-request-id": "fake-request", "authorization": FAKE_KEY}
        self.closed = False

    def close(self):
        self.closed = True


def adapter(monkeypatch, *, settings=None, transport=None, callback=None):
    monkeypatch.setenv("JEV_UNIT_TEST_KEY", FAKE_KEY)
    events = []
    result = JevAdapter(settings or config(), event_callback=callback or events.append,
                        transport=transport or (lambda *args: Response()))
    return result, events


def test_offline_payload_preserves_all_text_and_does_not_read_env(monkeypatch):
    monkeypatch.delenv("JEV_UNIT_TEST_KEY", raising=False)
    pair = {"query": "  вопрос\n", "document": "Длинный текст\n" * 4000}
    payload = build_jev_payload(pair)
    assert payload["state"] == pair
    assert payload["questions"] == {"relevance": {"type": "noul", "instructions": QUESTION}}
    report = preflight_jev([pair], config())
    assert report["input_tokens"] is None
    assert report["context_acceptance_verified"] is False
    assert report["max_state_chars"] > 40000


def test_worker_survives_between_warmup_and_measurement(monkeypatch):
    workers = []
    def transport(*args):
        workers.append(threading.get_ident())
        return Response()
    client, _ = adapter(monkeypatch, transport=transport)
    try:
        client.score([("q", "warmup")])
        client.score([("q", "measured")])
        assert workers[0] == workers[1]
        assert len(client._executor._threads) == 1
    finally:
        client.close()


@pytest.mark.parametrize("overrides", [{"base_url": "https://evil.test"}, {"model": "jev-latest"},
    {"api_key": FAKE_KEY}, {"max_requests": True}, {"max_requests": -1},
    {"timeout_s": float("nan")}, {"concurrency": 0}, {"api_key_env": "not a name"}])
def test_config_rejects_unsafe_or_ambiguous_values(overrides):
    with pytest.raises(ValueError):
        validate_jev_config(config(**overrides))


def test_live_construction_requires_env_and_never_discloses_it(monkeypatch):
    monkeypatch.delenv("JEV_UNIT_TEST_KEY", raising=False)
    with pytest.raises(JevError, match="environment variable"):
        JevAdapter(config(), event_callback=lambda event: None)


def test_success_preserves_raw_usage_and_logs_before_transport(monkeypatch):
    events, response = [], Response(answer(extra={"kept": True}))
    def transport(body, headers, timeout):
        assert events[-1]["phase"] == "start"
        assert json.loads(body)["state"] == {"query": "q", "document": "d"}
        assert headers["Authorization"] == "Bearer " + FAKE_KEY
        assert timeout == 3
        return response
    obj, _ = adapter(monkeypatch, transport=transport, callback=events.append)
    assert obj.score([("q", "d")]) == [{"score": .75, "raw": answer(extra={"kept": True}), "input_tokens": 42}]
    assert [e["phase"] for e in events] == ["start", "finish"]
    assert events[-1]["usage"]["input_tokens"] == 42
    assert events[-1]["duration_s"] >= 0
    assert response.closed
    assert FAKE_KEY not in json.dumps(events + [obj.metadata()])


@pytest.mark.parametrize("score", [True, False, "0.8", None, -0.01, 1.01, 10**400, float("inf"), float("nan")])
def test_rejects_invalid_scores(score):
    with pytest.raises(JevError, match="finite number"):
        parse_jev_response(answer(score), "jev-1.13.0")


@pytest.mark.parametrize("body", [answer(model="jev-1.14.0"), answer(model=None),
    answer(usage={"input_tokens": True}), answer(usage={"output_tokens": -1}),
    answer(answers={"other": {"type": "noul", "noul": .5}}),
    answer(answers={"relevance": {"type": "score", "noul": .5}})])
def test_rejects_wrong_model_question_type_or_usage(body):
    with pytest.raises(JevError):
        parse_jev_response(body, "jev-1.13.0")


def test_missing_usage_is_unknown_not_zero():
    body = answer()
    del body["usage"]
    assert parse_jev_response(body, "jev-1.13.0") == (.75, None)


@pytest.mark.parametrize("response", [Response(text="not JSON"), Response(text='{"duplicate":0,"duplicate":1}'),
    Response(text='{"a": NaN}'), Response(answer(True)), Response({"error": "overloaded"}, status=529),
    Response({"error": "redirect"}, status=302)])
def test_every_failed_response_is_logged_once_and_never_retried(monkeypatch, response):
    calls = []
    def transport(*args):
        calls.append(1)
        return response
    obj, events = adapter(monkeypatch, transport=transport)
    with pytest.raises(JevError):
        obj.score([("q", "first"), ("q", "not-sent")])
    assert len(calls) == 1
    assert len(events) == 2
    assert events[-1]["status"] == "error"
    assert events[-1]["response_text"] == response.text
    assert response.closed


def test_secrets_are_redacted_in_responses_logs_and_errors(monkeypatch):
    response = Response(answer(extra={"echo": FAKE_KEY, "api_key": "other-key"}))
    obj, events = adapter(monkeypatch, transport=lambda *args: response)
    result = obj.score_batch([{"query": "q", "document": "d"}])
    assert FAKE_KEY not in json.dumps(result + events + [obj.metadata()])
    assert result[0]["raw"]["extra"]["api_key"] == "[REDACTED]"
    def broken(*args):
        raise ConnectionError("Authorization: Bearer " + FAKE_KEY)
    obj, events = adapter(monkeypatch, transport=broken)
    with pytest.raises(JevError) as exc:
        obj.score([("q", "d")])
    assert FAKE_KEY not in str(exc.value) + json.dumps(events)
    assert events[-1]["response"] is None


def test_budget_is_total_and_batch_checked_before_requests(monkeypatch):
    obj, events = adapter(monkeypatch, settings=config(max_requests=2))
    obj.score([("q", "d")])
    with pytest.raises(JevError, match="remaining"):
        obj.score([("q", "d"), ("q", "d")])
    assert len(events) == 2
    obj.score([("q", "d")])
    with pytest.raises(JevError, match="remaining"):
        obj.score([("q", "d")])
    assert [e["request_index"] for e in events if e["phase"] == "start"] == [1, 2]


def test_concurrency_preserves_pair_order_and_records_all_inflight_failures(monkeypatch):
    def transport(body, *args):
        doc = json.loads(body)["state"]["document"]
        time.sleep(.01 if doc == "0" else .001)
        return Response(answer(int(doc) / 10))
    obj, events = adapter(monkeypatch, settings=config(concurrency=3), transport=transport)
    assert [x["score"] for x in obj.score([("q", str(i)) for i in range(5)])] == [0, .1, .2, .3, .4]
    assert len(events) == 10
    barrier, sent = threading.Barrier(2), []
    def fails(body, *args):
        doc = json.loads(body)["state"]["document"]
        sent.append(doc)
        barrier.wait(timeout=2)
        if doc == "1":
            time.sleep(.03)
        return Response(answer(.5) if doc == "1" else answer(True))
    obj, events = adapter(monkeypatch, settings=config(concurrency=2), transport=fails)
    with pytest.raises(JevError):
        obj.score([("q", str(i)) for i in range(5)])
    assert sorted(sent) == ["0", "1"]
    assert len([e for e in events if e["phase"] == "finish"]) == 2


def test_failed_start_journal_prevents_http_and_masks_callback_error(monkeypatch):
    sent = []
    def callback(event):
        raise OSError("disk error " + FAKE_KEY)
    obj, _ = adapter(monkeypatch, callback=callback, transport=lambda *args: sent.append(1))
    with pytest.raises(JevError, match="persistence") as exc:
        obj.score([("q", "d")])
    assert not sent
    assert FAKE_KEY not in str(exc.value)


def test_close_and_bad_input_do_not_make_requests(monkeypatch):
    obj, events = adapter(monkeypatch)
    with pytest.raises(ValueError):
        obj.score([("q", 123)])
    assert events == []
    obj.close()
    with pytest.raises(JevError, match="closed"):
        obj.score([("q", "d")])


def test_http_transport_is_pinned_no_redirects_no_retries_or_ambient_auth(monkeypatch):
    import requests
    sessions = []
    class Session:
        def __init__(self):
            self.trust_env = True
            self.closed = False
            sessions.append(self)
        def mount(self, prefix, adapter):
            assert prefix == "https://"
            assert adapter.max_retries.total == 0
        def post(self, url, **kw):
            assert url == BASE_URL + "/systemone"
            assert self.trust_env is False
            assert kw["allow_redirects"] is False
            assert kw["timeout"] == 3
            assert json.loads(kw["data"])["state"]["document"] == "d"
            return Response()
        def close(self):
            self.closed = True
    monkeypatch.setattr(requests, "Session", Session)
    monkeypatch.setenv("JEV_UNIT_TEST_KEY", FAKE_KEY)
    obj = JevAdapter(config(), event_callback=lambda event: None)
    assert obj.score([("q", "d")])[0]["score"] == .75
    obj.close()
    assert len(sessions) == 1 and sessions[0].closed
