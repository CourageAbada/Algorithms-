"""Bounded retry with backoff for READ-ONLY historical requests.

Outcomes are kept distinct (a retry never hides which one happened):

* legitimate empty result      -> the call returns an empty array; nothing to retry, handled by the caller
* history unavailable          -> HistoryUnavailableError (terminal said "no data"); ``unavailable_retries`` confirmations
* request timeout              -> RequestTimeoutError (slow/cold load; the adapter already ran a health check)
* terminal error               -> TerminalRequestError (e.g. -1 'Call failed'); retried only for ``retry_terminal_codes``
* actual connection loss       -> ConnectionLostError/CallTimeoutError: NEVER retried here (reconnect is the caller's job)

Every retry is reported through ``on_event`` (and logged) with attempt number, reason and wait. Retries are bounded.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from fxscalp.brokers.errors import HistoryUnavailableError, RequestTimeoutError, TerminalRequestError

log = logging.getLogger("fxscalp.retry")
T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3                       # retries after the first attempt (so at most 4 attempts)
    backoff_base_s: float = 5.0
    backoff_factor: float = 3.0
    backoff_max_s: float = 60.0
    unavailable_retries: int = 1               # confirmation retries for "no data"
    retry_terminal_codes: tuple[int, ...] = (-1,)   # -1 = RES_E_FAIL ('Terminal: Call failed'); others are not retried

    def delay(self, attempt: int) -> float:
        """Wait before retry number ``attempt`` (1-based)."""
        return min(self.backoff_max_s, self.backoff_base_s * (self.backoff_factor ** (attempt - 1)))


NO_RETRY = RetryPolicy(max_retries=0, unavailable_retries=0)


def call_with_retry(fn: Callable[[], T], policy: RetryPolicy = RetryPolicy(), *, label: str = "request",
                    sleep: Callable[[float], None] = time.sleep,
                    on_event: Callable[[dict[str, Any]], None] | None = None) -> T:
    attempt = 0          # retries performed so far (timeouts + terminal errors)
    unavailable = 0
    err: Exception
    while True:
        try:
            return fn()
        except RequestTimeoutError as exc:
            reason, err = "request_timeout", exc
        except TerminalRequestError as exc:
            if exc.code not in policy.retry_terminal_codes:
                raise
            reason, err = f"terminal_error_{exc.code}", exc
        except HistoryUnavailableError as exc:
            if unavailable >= policy.unavailable_retries:
                raise
            unavailable += 1
            reason, err = "history_unavailable_confirmation", exc
            ev = {"event": "retry", "label": label, "attempt": unavailable, "reason": reason, "error": str(exc),
                  "wait_s": policy.delay(1), "ts_utc": datetime.now(timezone.utc).isoformat()}
            _emit(ev, on_event)
            sleep(policy.delay(1))
            continue
        if attempt >= policy.max_retries:
            _emit({"event": "retry_exhausted", "label": label, "attempts": attempt + 1, "reason": reason,
                   "error": str(err), "ts_utc": datetime.now(timezone.utc).isoformat()}, on_event)
            raise err
        attempt += 1
        wait = policy.delay(attempt)
        _emit({"event": "retry", "label": label, "attempt": attempt, "reason": reason, "error": str(err),
               "wait_s": wait, "ts_utc": datetime.now(timezone.utc).isoformat()}, on_event)
        sleep(wait)


def _emit(ev: dict[str, Any], on_event: Callable[[dict[str, Any]], None] | None) -> None:
    log.warning("history %s: %s (%s) %s", ev["event"], ev["label"], ev["reason"], ev.get("error", ""),
                extra={"ctx": {k: v for k, v in ev.items() if k != "error"}})
    if on_event:
        on_event(ev)
