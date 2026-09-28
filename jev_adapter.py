"""TypeSafe direct HTTP adapter. No SDK retries, clipping or implicit live calls.

Sources checked 2026-09-27:
https://docs.typesafe.ai/api
https://docs.typesafe.ai/models
https://docs.typesafe.ai/sdk/python/api/types/responses

Exact Jev token counting remains pending: the public reference does not expose
a tokenizer or a token-count endpoint. Character counts are NOT token counts.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import threading
import time

from audit import QUESTION

BASE_URL = "https://api.typesafe.ai/v1"
ENDPOINT = BASE_URL + "/systemone"
VERIFIED_MODEL = "jev-1.13.0"
SOURCES = ["https://docs.typesafe.ai/api", "https://docs.typesafe.ai/models",
           "https://docs.typesafe.ai/sdk/python/api/types/responses"]


class JevError(RuntimeError):
    """Safe error; complete sanitized attempt details are in the event journal."""


def validate_jev_config(config: dict) -> dict:
    """Whitelist public configuration; never accept an inline credential."""
    if any(key in config for key in ("api_key", "headers", "authorization", "token")):
        raise ValueError("Use api_key_env, not an inline credential")
    allowed = {"model", "base_url", "api_key_env", "timeout_s", "concurrency", "max_requests", "batch_size"}
    if set(config) - allowed:
        raise ValueError("Unknown Jev configuration key; credentials belong only in the environment")
    if config.get("base_url", BASE_URL) != BASE_URL:
        raise ValueError("Jev base_url must be the fixed official TypeSafe endpoint")
    model = config.get("model", VERIFIED_MODEL)
    if not isinstance(model, str) or not re.fullmatch(r"jev-\d+\.\d+\.\d+", model):
        raise ValueError("Jev requires a pinned versioned model ID, not an alias")
    env = config.get("api_key_env", "TYPESAFE_API_KEY")
    if not isinstance(env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env):
        raise ValueError("Invalid api_key_env name")
    timeout = config.get("timeout_s", 60.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_s must be positive and finite")
    concurrency = config.get("concurrency", 1)
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    cap = config.get("max_requests")
    if type(cap) is not int or cap < 0:
        raise ValueError("max_requests must be an explicit nonnegative integer")
    return {"model": model, "base_url": BASE_URL, "api_key_env": env,
            "timeout_s": float(timeout), "concurrency": concurrency, "max_requests": cap}


def pair_texts(pair) -> tuple[str, str]:
    if isinstance(pair, dict):
        query, document = pair.get("query"), pair.get("document")
    elif isinstance(pair, (tuple, list)) and len(pair) == 2:
        query, document = pair
    else:
        raise ValueError("A pair must contain query and document strings")
    if not isinstance(query, str) or not isinstance(document, str):
        raise ValueError("A pair must contain query and document strings")
    return query, document


def build_jev_payload(pair, model: str = VERIFIED_MODEL) -> dict:
    """Pure offline payload builder; keeps the complete source text unchanged."""
    if not isinstance(model, str) or not re.fullmatch(r"jev-\d+\.\d+\.\d+", model):
        raise ValueError("Jev requires a pinned versioned model ID")
    query, document = pair_texts(pair)
    return {"model": model, "state": {"query": query, "document": document},
            "questions": {"relevance": {"type": "noul", "instructions": QUESTION}}}


def encoded_payload(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


def preflight_jev(pairs, config: dict) -> dict:
    """Validate schema offline, without claiming unverified token acceptance."""
    settings = validate_jev_config(config)
    lengths = []
    for pair in pairs:
        payload = build_jev_payload(pair, settings["model"])
        lengths.append({"state_chars": len(json.dumps(payload["state"], ensure_ascii=False)),
                        "payload_bytes": len(encoded_payload(payload))})
    return {"pairs": len(lengths), "max_state_chars": max((x["state_chars"] for x in lengths), default=0),
            "max_payload_bytes": max((x["payload_bytes"] for x in lengths), default=0),
            "input_tokens": None, "token_count_status": "pending_official_tokenizer_or_api_usage",
            "context_acceptance_verified": False, "truncation": False,
            "documented_context": {"request": "64k tokens", "state_plus_longest_question": "32k tokens"},
            "documented_state_char_limit": None, "sources": SOURCES, "checked_on": "2026-09-27"}


def _no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("Duplicate JSON key")
        out[key] = value
    return out


def _reject_nonfinite(value):
    raise ValueError("Non-finite JSON value")


def parse_jev_response(raw: dict, model: str) -> tuple[float, int | None]:
    if not isinstance(raw, dict):
        raise JevError("Jev response must be a JSON object")
    if raw.get("model") != model:
        raise JevError("Jev response model is missing or differs from the pinned model")
    answers = raw.get("answers")
    if not isinstance(answers, dict) or set(answers) != {"relevance"}:
        raise JevError("Jev response must contain exactly the relevance answer")
    answer = answers["relevance"]
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise JevError("Jev relevance answer must have type noul")
    score = answer.get("noul")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1 or not math.isfinite(score):
        raise JevError("Jev noul must be a finite number between 0 and 1")
    usage = raw.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise JevError("Jev usage must be an object when present")
    usage = usage or {}
    for field in ("input_tokens", "output_tokens"):
        count = usage.get(field)
        if count is not None and (type(count) is not int or count < 0):
            raise JevError("Jev token usage must be a nonnegative integer or null")
    return float(score), usage.get("input_tokens")


class JevAdapter:
    def __init__(self, config: dict, *, event_callback, transport=None):
        self.config = validate_jev_config(config)
        if not callable(event_callback):
            raise ValueError("A durable event_callback is required for live Jev requests")
        key = os.environ.get(self.config["api_key_env"])
        if not key or not key.strip():
            raise JevError("The configured Jev API-key environment variable is empty")
        if key != key.strip() or "\r" in key or "\n" in key:
            raise JevError("The Jev API key has invalid whitespace")
        self._key, self._callback = key, event_callback
        self._transport = transport or self._http_post
        self._event_lock, self._counter_lock, self._score_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self._local, self._sessions, self._requests = threading.local(), [], 0
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=self.config["concurrency"])

    def metadata(self) -> dict:
        return {"adapter": "typesafe_direct_http", **self.config, "endpoint": ENDPOINT,
                "question": QUESTION, "primitive": "noul", "question_id": "relevance",
                "retries": 0, "redirects": False, "truncation": False,
                "token_count_status": "pending_official_tokenizer_or_api_usage",
                "documented_context": {"request": "64k tokens", "state_plus_longest_question": "32k tokens"},
                "sources": SOURCES, "checked_on": "2026-09-27"}

    def synchronize(self):
        """score waits for every in-flight HTTP request before returning."""

    def _safe(self, value):
        if isinstance(value, str):
            return value.replace(self._key, "[REDACTED]")
        if isinstance(value, dict):
            return {self._safe(str(k)): ("[REDACTED]" if str(k).lower() in
                    {"authorization", "api_key", "apikey", "access_token"} else self._safe(v))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [self._safe(v) for v in value]
        return value

    def _emit(self, event):
        with self._event_lock:
            self._callback(self._safe(event))

    def _http_post(self, payload: bytes, headers: dict, timeout_s: float):
        import requests
        if not hasattr(self._local, "session"):
            session = requests.Session()
            # No ambient .netrc credentials/proxies; no redirect can forward auth.
            session.trust_env = False
            session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0, pool_maxsize=1))
            self._local.session = session
            with self._counter_lock:
                self._sessions.append(session)
        return self._local.session.post(ENDPOINT, data=payload, headers=headers,
                                        timeout=timeout_s, allow_redirects=False)

    def _one(self, pair_index: int, pair) -> dict:
        payload = build_jev_payload(pair, self.config["model"])
        body = encoded_payload(payload)
        with self._counter_lock:
            if self._requests >= self.config["max_requests"]:
                raise JevError("Jev request ceiling reached; no automatic retry")
            self._requests += 1
            number = self._requests
        identity = {"adapter": "jev", "request_index": number, "pair_index": pair_index,
                    "request_sha256": hashlib.sha256(body).hexdigest()}
        # Persist intention BEFORE network I/O. If logging fails, do not send.
        self._emit({**identity, "phase": "start", "started_at": datetime.now(timezone.utc).isoformat(),
                    "endpoint": ENDPOINT, "request": payload})
        started = time.perf_counter()
        result = {**identity, "phase": "finish", "status": "error", "http_status": None,
                  "response_text": None, "response": None, "usage": None}
        response = None
        try:
            response = self._transport(body, {"Authorization": "Bearer " + self._key,
                                             "Content-Type": "application/json"}, self.config["timeout_s"])
            result["http_status"] = response.status_code
            result["response_text"] = response.text
            result["response_headers"] = {name: response.headers[name] for name in
                ("x-request-id", "request-id", "retry-after", "content-type") if name in response.headers}
            try:
                raw = json.loads(response.text, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_nonfinite)
            except (ValueError, TypeError, RecursionError):
                raise JevError("Jev response is not valid strict JSON") from None
            result["response"] = raw
            if isinstance(raw, dict):
                result["usage"] = raw.get("usage")
            if response.status_code != 200:
                raise JevError(f"Jev HTTP {response.status_code}; attempt saved, no automatic retry")
            score, tokens = parse_jev_response(raw, self.config["model"])
            result.update(status="ok", score=score, input_tokens=tokens)
            return {"score": score, "raw": self._safe(raw), "input_tokens": tokens}
        except Exception as exc:
            # Exception strings from HTTP libraries may embed a request or key.
            error = str(exc) if isinstance(exc, JevError) else "Jev transport failed; attempt saved, no automatic retry"
            result.update(error_type=type(exc).__name__, error=self._safe(error))
            raise JevError(self._safe(error)) from None
        finally:
            result["duration_s"] = time.perf_counter() - started
            result["finished_at"] = datetime.now(timezone.utc).isoformat()
            try:
                if response is not None:
                    response.close()
            finally:
                self._emit(result)

    def score(self, pairs) -> list[dict]:
        """Bounded concurrent pair scoring. Preserve input order, fail without retries.

        On a failure, no more pairs are submitted; requests already in flight are
        awaited and logged. Their results remain in the durable event journal.
        """
        pairs = list(pairs)
        for pair in pairs:
            build_jev_payload(pair, self.config["model"])
        with self._score_lock:
            if self._closed:
                raise JevError("Jev adapter is closed")
            if len(pairs) > self.config["max_requests"] - self._requests:
                raise JevError("Batch exceeds the remaining Jev request ceiling; nothing sent")
            if not pairs:
                return []
            output, next_index, first_error = [None] * len(pairs), 0, None
            # Keep workers and their HTTP sessions alive across warmup/measured lists.
            executor = self._executor
            pending = {}
            while next_index < min(len(pairs), self.config["concurrency"]):
                pending[executor.submit(self._one, next_index, pairs[next_index])] = next_index
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
                    pending[executor.submit(self._one, next_index, pairs[next_index])] = next_index
                    next_index += 1
            if first_error is not None:
                # Logging callbacks can fail; never include their arbitrary text.
                if isinstance(first_error, JevError):
                    raise first_error
                raise JevError("Jev event persistence failed; no further requests sent") from None
            return output

    def score_batch(self, pairs) -> list[dict]:
        return self.score(pairs)

    def close(self):
        with self._score_lock:
            self._executor.shutdown(wait=True)
            for session in self._sessions:
                session.close()
            self._sessions.clear()
            self._closed = True


def create_jev_adapter(config: dict, *, event_callback) -> JevAdapter:
    """Construct only after the runner has checked explicit API authorization."""
    return JevAdapter(config, event_callback=event_callback)
