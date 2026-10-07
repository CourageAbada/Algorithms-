"""Single owner thread for ALL MetaTrader5 calls.

Concurrency model (documented assumptions)
------------------------------------------
* The MetaTrader5 package talks to the terminal over a single IPC channel and its thread-safety is
  UNDOCUMENTED (unverified). We therefore never call it concurrently: every call, including
  ``initialize`` and ``shutdown``, is executed by ONE dedicated daemon thread, FIFO, one at a time.
* Any number of caller threads may submit work through ``call()``; each blocks until its result,
  error or deadline. Submitting from the owner thread itself is a programming error (deadlock) and
  raises immediately.
* Whether MT5 requires initialize/use/shutdown to happen on the *same* thread is unknown; running
  them all on the owner thread is the conservative choice.
* A blocking MT5 call cannot be cancelled. On a deadline overrun the owner is marked WEDGED: queued
  and new calls fail fast with CallTimeoutError/ConnectionLostError. ``replace()`` abandons the stuck
  thread (daemon) and starts a fresh owner; a stuck thread may still hold IPC state, so if this
  happens repeatedly the process should be restarted.
* Per-call latency is recorded for the performance baseline and health reporting.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Callable

from fxscalp.brokers.errors import CallTimeoutError, ConnectionLostError, RequestTimeoutError

_STOP = object()


class MT5Owner:
    def __init__(self, name: str = "mt5-owner", default_timeout_s: float = 30.0, history: int = 5000):
        self._name = name
        self._default_timeout = default_timeout_s
        self._q: "queue.Queue[Any]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._wedged = False
        self._lock = threading.Lock()
        self._lat: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=history))
        self._epoch = 0

    # ---- lifecycle -------------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive() and not self._wedged:
                return
            self._wedged = False
            self._epoch += 1
            self._q = queue.Queue()
            t = threading.Thread(target=self._run, args=(self._q,), name=f"{self._name}-{self._epoch}", daemon=True)
            self._thread = t
            t.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        with self._lock:
            t, q = self._thread, self._q
            self._thread = None
        if t and t.is_alive():
            q.put(_STOP)
            t.join(timeout_s)

    def replace(self) -> None:
        """Abandon a wedged owner and start a new one (see module docstring for caveats)."""
        with self._lock:
            self._thread = None
        self.start()

    @property
    def wedged(self) -> bool:
        return self._wedged

    def is_owner_thread(self) -> bool:
        t = self._thread
        return t is not None and threading.current_thread() is t

    # ---- execution -------------------------------------------------------------------------
    def _run(self, q: "queue.Queue[Any]") -> None:
        while True:
            item = q.get()
            if item is _STOP:
                return
            fn, args, kwargs, fut, label = item
            if not fut.set_running_or_notify_cancel():
                continue
            t0 = time.perf_counter()
            try:
                res = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - propagate every failure to the caller
                fut.set_exception(exc)
            else:
                fut.set_result(res)
            finally:
                self._lat[label].append(time.perf_counter() - t0)

    def call(self, fn: Callable[..., Any], *args: Any, label: str | None = None,
             timeout_s: float | None = None, soft: bool = False, **kwargs: Any) -> Any:
        """Run ``fn`` on the owner thread.

        ``soft=False`` (default): a deadline overrun marks the owner WEDGED and raises CallTimeoutError.
        ``soft=True`` (history requests): a deadline overrun raises RequestTimeoutError and does NOT wedge the
        owner. The blocking call is still running on the owner thread (it cannot be cancelled); any later call,
        e.g. the caller's health check, queues behind it and therefore also tells whether it ever returns.
        """
        if self.is_owner_thread():
            raise RuntimeError("MT5Owner.call() invoked from the owner thread (would deadlock)")
        if self._wedged:
            raise ConnectionLostError("MT5 owner thread is wedged by an earlier timed-out call; reconnect required")
        if not self._thread or not self._thread.is_alive():
            raise ConnectionLostError("MT5 owner thread is not running; call connect() first")
        label = label or getattr(fn, "__name__", "call")
        fut: Future[Any] = Future()
        self._q.put((fn, args, kwargs, fut, label))
        timeout = self._default_timeout if timeout_s is None else timeout_s
        try:
            return fut.result(timeout)
        except FutureTimeout:
            if soft:
                raise RequestTimeoutError(
                    f"MT5 request {label!r} did not return within {timeout:.1f}s (cold history load or slow "
                    "terminal); the connection is not assumed lost: a health check decides") from None
            self._wedged = True
            raise CallTimeoutError(
                f"MT5 call {label!r} did not return within {timeout:.1f}s; the terminal may be hung. "
                "The session is marked lost; restart the terminal and reconnect.") from None

    # ---- stats -----------------------------------------------------------------------------
    def latency_stats(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for label, d in self._lat.items():
            v = sorted(d)
            if not v:
                continue
            n = len(v)
            out[label] = {"count": float(n), "mean_ms": 1000 * sum(v) / n, "p50_ms": 1000 * v[n // 2],
                          "p95_ms": 1000 * v[min(n - 1, int(0.95 * n))], "max_ms": 1000 * v[-1]}
        return out
