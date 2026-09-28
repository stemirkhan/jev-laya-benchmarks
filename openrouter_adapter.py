"""Pinned Jev Decisions adapter with a durable, shared, fail-closed USD guard.

No network is performed on import. This is additive to the native Jev adapter.
The reservation relies on the externally verified provider tariff/context ceiling;
it is not a provider-enforced account spending limit.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time

from audit import QUESTION

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
RESERVATION_USD = Decimal("0.002752512")
SOURCES = ["https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request"]


class OpenRouterError(RuntimeError):
    pass


class BudgetError(OpenRouterError):
    pass


def decimal_amount(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise BudgetError("USD amount must be a finite nonnegative number")
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        raise BudgetError("Invalid USD amount") from None
    if not amount.is_finite() or amount < 0:
        raise BudgetError("USD amount must be finite and nonnegative")
    return amount


class Budget:
    """Exclusive cross-process ledger. Unresolved requests block all later sends.

    Reservations are written and fsynced BEFORE admission. An unknown cost keeps
    the entire reservation charged and permanently trips this ledger. Known
    charges remain recorded even when a model response cannot be used.
    """

    def __init__(self, path, limit_usd, reserve_usd=RESERVATION_USD):
        self.path = Path(path)
        self.limit = decimal_amount(limit_usd)
        self.reserve_amount = decimal_amount(reserve_usd)
        if self.limit <= 0 or self.limit > Decimal("3.90") or self.reserve_amount != RESERVATION_USD:
            raise BudgetError("Require ceiling at most $3.90 and the verified fixed reservation")
        self._lock = threading.Lock()
        self.spent = Decimal("0")
        self.pending = {}
        self.next_id = 1
        self.tripped = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._file.seek(0)
            rows = [json.loads(line) for line in self._file]
            if not rows:
                self._append({"event": "init", "schema": "openrouter-usd-budget-v1",
                              "limit_usd": str(self.limit), "reserve_usd": str(self.reserve_amount)})
            else:
                first = rows.pop(0)
                if (first.get("schema") != "openrouter-usd-budget-v1" or first.get("event") != "init"
                        or decimal_amount(first["limit_usd"]) != self.limit
                        or decimal_amount(first["reserve_usd"]) != self.reserve_amount):
                    raise BudgetError("Budget terms differ; cannot reset or increase an existing ledger")
                for row in rows:
                    self._apply(row)
            if self.pending:
                self.tripped = True
                raise BudgetError("Unresolved API reservations; reconcile before any further request")
            if self.tripped:
                raise BudgetError("Budget ledger is tripped; no further requests allowed")
        except BaseException:
            self._file.close()
            raise

    def _append(self, event):
        try:
            self._file.write(json.dumps({**event, "at": datetime.now(timezone.utc).isoformat()}, allow_nan=False) + "\n")
            self._file.flush()
            os.fsync(self._file.fileno())
        except BaseException:
            self.tripped = True
            raise BudgetError("Budget journal could not be persisted; no further requests") from None

    def _apply(self, row):
        action, ticket = row.get("event"), row.get("ticket")
        if action == "reserve":
            if self.tripped or type(ticket) is not int or ticket != self.next_id or ticket in self.pending:
                raise BudgetError("Invalid reservation sequence")
            if self.spent + sum(self.pending.values()) + self.reserve_amount > self.limit:
                raise BudgetError("Journal reservation exceeds the budget")
            self.pending[ticket] = self.reserve_amount
            self.next_id += 1
        elif action == "settle":
            if ticket not in self.pending:
                raise BudgetError("Unknown or duplicate settlement")
            reserved = self.pending.pop(ticket)
            cost = decimal_amount(row["cost_usd"])
            self.spent += cost
            self.tripped |= row.get("unknown_charge") is True or cost > reserved
        elif action == "trip":
            self.tripped = True
        else:
            raise BudgetError("Unknown budget ledger event")

    def reserve(self, request_sha256):
        with self._lock:
            if self.tripped:
                raise BudgetError("Budget guard stopped further requests")
            if self.spent + sum(self.pending.values()) + self.reserve_amount > self.limit:
                self.tripped = True
                self._append({"event": "trip", "reason": "insufficient_unreserved_budget"})
                raise BudgetError("USD ceiling reached before sending a request")
            event = {"event": "reserve", "ticket": self.next_id, "request_sha256": request_sha256}
            self._append(event)
            self._apply(event)
            return event["ticket"]

    def settle(self, ticket, cost_usd):
        with self._lock:
            if ticket not in self.pending:
                raise BudgetError("Unknown or already settled reservation")
            try:
                amount = decimal_amount(cost_usd)
                unknown = False
            except BudgetError:
                amount, unknown = self.pending[ticket], True
            event = {"event": "settle", "ticket": ticket, "cost_usd": str(amount),
                     "unknown_charge": unknown}
            self._append(event)
            self._apply(event)
            if unknown or amount > self.reserve_amount:
                raise BudgetError("Missing/invalid charge or charge above reservation; no further requests")

    def trip(self, reason):
        with self._lock:
            self.tripped = True
            self._append({"event": "trip", "reason": reason})

    def snapshot(self):
        with self._lock:
            return {"limit_usd": str(self.limit), "reserve_usd": str(self.reserve_amount),
                    "spent_or_conservatively_charged_usd": str(self.spent),
                    "reserved_usd": str(sum(self.pending.values())), "tripped": self.tripped,
                    "requests_reserved": self.next_id - 1}

    def close(self):
        with self._lock:
            self._file.close()


def validate_config(config):
    allowed = {"model", "expected_response_model", "api_key_env", "timeout_s", "concurrency", "max_requests", "batch_size"}
    if set(config) - allowed:
        raise ValueError("Unknown OpenRouter config key; credentials belong only in the environment")
    pattern = r"typesafe/jev-1\.13(?:-20260917)?"
    for name in ("model", "expected_response_model"):
        if not isinstance(config.get(name), str) or not re.fullmatch(pattern, config[name]):
            raise ValueError("Require verified pinned Jev model and exact expected response model")
    env = config.get("api_key_env", "OPENROUTER_API_KEY")
    if not isinstance(env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env):
        raise ValueError("Invalid API key environment variable")
    timeout, concurrency, maximum = config.get("timeout_s", 60), config.get("concurrency", 4), config.get("max_requests")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Require a finite positive timeout")
    if type(concurrency) is not int or not 1 <= concurrency <= 4 or type(maximum) is not int or maximum < 0:
        raise ValueError("Require concurrency 1..4 and explicit nonnegative request ceiling")
    return {"model": config["model"], "expected_response_model": config["expected_response_model"],
            "api_key_env": env, "timeout_s": float(timeout), "concurrency": concurrency, "max_requests": maximum}


def build_payload(pair, model):
    if isinstance(pair, dict):
        query, document = pair.get("query"), pair.get("document")
    elif isinstance(pair, (list, tuple)) and len(pair) == 2:
        query, document = pair
    else:
        raise ValueError("Expected a query/document pair")
    if not isinstance(query, str) or not isinstance(document, str):
        raise ValueError("Query and document must be full strings")
    return {"model": model, "state": {"query": query, "document": document},
            "questions": {"relevance": {"type": "noul", "instructions": QUESTION}}}


def _pairs_no_duplicates(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Duplicate JSON key")
        output[key] = value
    return output


def _nonfinite(value):
    raise ValueError("Non-finite JSON value")


def parse_response(raw, expected_model):
    if not isinstance(raw, dict) or raw.get("model") != expected_model or raw.get("provider") != "TypeSafe":
        raise OpenRouterError("Unexpected response model/provider")
    answers = raw.get("answers")
    if not isinstance(answers, dict) or set(answers) != {"relevance"}:
        raise OpenRouterError("Expected exactly one relevance answer")
    answer = answers["relevance"]
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise OpenRouterError("Expected a noul relevance answer")
    score = answer.get("noul")
    if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise OpenRouterError("Invalid relevance score")
    usage = raw.get("usage")
    if not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0 for key in ("input_tokens", "output_tokens")):
        raise OpenRouterError("Missing or invalid token usage")
    if isinstance(usage.get("cost"), bool) or not isinstance(usage.get("cost"), (int, float)):
        raise OpenRouterError("Missing or invalid numeric API charge")
    decimal_amount(usage.get("cost"))
    if usage["input_tokens"] > 32000:
        raise OpenRouterError("Observed input usage exceeds the verified 32k context bound")
    return float(score), usage["input_tokens"]


class OpenRouterAdapter:
    def __init__(self, config, *, event_callback, budget, transport=None):
        self.config = validate_config(config)
        if not callable(event_callback) or not isinstance(budget, Budget):
            raise ValueError("Durable callback and shared Budget required")
        key = os.environ.get(self.config["api_key_env"])
        if not key or key != key.strip() or "\r" in key or "\n" in key:
            raise OpenRouterError("Missing/invalid API key environment variable")
        self._key, self._callback, self.budget = key, event_callback, budget
        self._transport = transport or self._http_post
        self._event_lock, self._counter_lock, self._score_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self._local, self._sessions, self._requests = threading.local(), [], 0
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=self.config["concurrency"])

    def metadata(self):
        return {"adapter": "openrouter_decisions_http", **self.config, "endpoint": ENDPOINT,
                "question": QUESTION, "primitive": "noul", "question_id": "relevance",
                "retries": 0, "redirects": False, "truncation": False, "context_tokens": 32000,
                "budget_ceiling_usd": str(self.budget.limit), "reservation_usd": str(self.budget.reserve_amount),
                "budget_ledger": str(self.budget.path.resolve()), "sources": SOURCES,
                "input_price_per_million_usd": "0.042", "output_price_per_million_usd": "0"}

    def synchronize(self):
        pass

    def _safe(self, value):
        if isinstance(value, str):
            return value.replace(self._key, "[REDACTED]")
        if isinstance(value, dict):
            return {self._safe(str(k)): "[REDACTED]" if str(k).lower() in {"authorization", "api_key", "apikey", "access_token"} else self._safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._safe(v) for v in value]
        return value

    def _emit(self, event):
        with self._event_lock:
            self._callback(self._safe(event))

    def _http_post(self, payload, headers, timeout_s):
        import requests
        if not hasattr(self._local, "session"):
            session = requests.Session()
            session.trust_env = False
            session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0, pool_maxsize=1))
            self._local.session = session
            with self._counter_lock:
                self._sessions.append(session)
        return self._local.session.post(ENDPOINT, data=payload, headers=headers, timeout=timeout_s, allow_redirects=False)

    def _one(self, pair_index, pair):
        payload = build_payload(pair, self.config["model"])
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        with self._counter_lock:
            if self._requests >= self.config["max_requests"]:
                raise OpenRouterError("API request ceiling reached")
            self._requests += 1
            index = self._requests
        identity = {"adapter": "jev-openrouter", "request_index": index, "pair_index": pair_index,
                    "request_sha256": hashlib.sha256(body).hexdigest()}
        ticket = self.budget.reserve(identity["request_sha256"])
        try:
            self._emit({**identity, "phase": "start", "started_at": datetime.now(timezone.utc).isoformat(),
                        "budget_ticket": ticket, "endpoint": ENDPOINT, "request": payload})
        except BaseException:
            self.budget.trip("api_start_journal_failure")
            self.budget.settle(ticket, 0)
            raise OpenRouterError("Request journal failure; request not sent") from None
        started = time.perf_counter()
        result = {**identity, "phase": "finish", "status": "error", "http_status": None,
                  "response_text": None, "response": None, "usage": None, "budget_ticket": ticket}
        response, cost = None, None
        try:
            response = self._transport(body, {"Authorization": "Bearer " + self._key, "Content-Type": "application/json"}, self.config["timeout_s"])
            result["http_status"], result["response_text"] = response.status_code, response.text
            result["response_headers"] = {name: response.headers[name] for name in ("x-request-id", "request-id", "retry-after", "content-type") if name in response.headers}
            try:
                raw = json.loads(response.text, object_pairs_hook=_pairs_no_duplicates, parse_constant=_nonfinite)
            except (ValueError, TypeError, RecursionError):
                raise OpenRouterError("Response is not strict JSON") from None
            result["response"] = raw
            if isinstance(raw, dict) and isinstance(raw.get("usage"), dict):
                result["usage"] = raw["usage"]
                cost = raw["usage"].get("cost")
                if isinstance(cost, bool) or not isinstance(cost, (int, float)):
                    cost = None
            if response.status_code != 200:
                raise OpenRouterError(f"OpenRouter HTTP {response.status_code}; no automatic retry")
            score, tokens = parse_response(raw, self.config["expected_response_model"])
            result.update(status="ok", score=score, input_tokens=tokens)
            return {"score": score, "raw": self._safe(raw), "input_tokens": tokens}
        except Exception as exc:
            error = str(exc) if isinstance(exc, OpenRouterError) else "OpenRouter transport failed; no automatic retry"
            result.update(error_type=type(exc).__name__, error=self._safe(error))
            # Stop concurrent submissions before a failing future is consumed.
            self.budget.trip("request_or_response_failure")
            raise OpenRouterError(self._safe(error)) from None
        finally:
            result["duration_s"] = time.perf_counter() - started
            result["finished_at"] = datetime.now(timezone.utc).isoformat()
            try:
                try:
                    self.budget.settle(ticket, cost)
                except BudgetError:
                    result.update(status="error", error_type="BudgetError", error="Charge settlement failed; no further requests")
                    raise
            finally:
                try:
                    if response is not None:
                        response.close()
                finally:
                    try:
                        self._emit(result)
                    except BaseException:
                        self.budget.trip("api_finish_journal_failure")
                        raise OpenRouterError("Response journal failure; no further requests") from None

    def score(self, pairs):
        pairs = list(pairs)
        for pair in pairs:
            build_payload(pair, self.config["model"])
        with self._score_lock:
            if self._closed:
                raise OpenRouterError("Adapter is closed")
            if len(pairs) > self.config["max_requests"] - self._requests:
                raise OpenRouterError("Batch exceeds remaining API request ceiling; nothing sent")
            output, next_index, first_error = [None] * len(pairs), 0, None
            pending = {}
            while next_index < min(len(pairs), self.config["concurrency"]):
                pending[self._executor.submit(self._one, next_index, pairs[next_index])] = next_index
                next_index += 1
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    index = pending.pop(future)
                    try:
                        output[index] = future.result()
                    except Exception as exc:
                        first_error = first_error or exc
                while first_error is None and next_index < len(pairs) and len(pending) < self.config["concurrency"]:
                    pending[self._executor.submit(self._one, next_index, pairs[next_index])] = next_index
                    next_index += 1
            if first_error is not None:
                raise OpenRouterError(str(first_error) if isinstance(first_error, OpenRouterError) else "Adapter failure; no further requests") from None
            return output

    def score_batch(self, pairs):
        return self.score(pairs)

    def close(self):
        with self._score_lock:
            self._executor.shutdown(wait=True)
            for session in self._sessions:
                session.close()
            self._sessions.clear()
            self._closed = True
