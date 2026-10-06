"""MT5BrokerAdapter: READ-ONLY (Phase 1).

Lifecycle:  initialize -> verify (account class, identity, terminal link) -> use -> health -> shutdown.
One persistent session, all calls on one owner thread (see owner.py). No call to any trading function
is possible: the adapter only holds a ReadOnlyMT5Api, and the trading methods raise TradingDisabledError.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping

import numpy as np

from fxscalp.brokers.accounts import PHASE1_ALLOWED_ACCOUNT_CLASSES, AccountClass, AccountIdentity, \
    assert_account_allowed, mask_login
from fxscalp.brokers.base import (AccountInfo, AdapterHealth, BrokerAdapter, OrderRequest, OrderResult,
                                  SymbolInfo, TerminalInfo, Tick)
from fxscalp.brokers.errors import (AccountMismatchError, AccountNotAllowedError, CallTimeoutError, ConnectionLostError, DataFormatError,
                                    MT5AuthError, MT5InitializationError, NoTickDataError, StaleTickError,
                                    SymbolNotFoundError, TradingDisabledError)
from fxscalp.brokers.mt5 import convert
from fxscalp.brokers.mt5.api import ReadOnlyMT5Api, load_mt5_module
from fxscalp.brokers.mt5.owner import MT5Owner

log = logging.getLogger("fxscalp.mt5")

_IPC_ERROR_RANGE = range(-10005, -9999)   # RES_E_INTERNAL_FAIL .. _TIMEOUT (verified constants)
_RES_AUTH_FAILED = -6                      # RES_E_AUTH_FAILED (verified constant)
_TIMEFRAMES = ("M1", "M2", "M3", "M4", "M5", "M6", "M10", "M12", "M15", "M20", "M30",
               "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D1", "W1", "MN1")


class AdapterState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"      # initialised AND verified (demo gate passed)
    LOST = "LOST"                # connection/owner failure; reconnect required
    REJECTED = "REJECTED"        # account not allowed; session closed
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class Mt5Credentials:
    """Credentials from the environment only. repr() never shows the password."""

    login: int | None = None
    password: str | None = field(default=None, repr=False)
    server: str | None = None
    path: str | None = None
    timeout_ms: int = 60_000

    def __repr__(self) -> str:
        return (f"Mt5Credentials(login={mask_login(self.login)}, password={'***' if self.password else None}, "
                f"server={self.server!r}, path={'set' if self.path else None})")

    @staticmethod
    def from_env(env: Mapping[str, str]) -> "Mt5Credentials":
        login = (env.get("MT5_LOGIN") or "").strip()
        try:
            login_i = int(login) if login and login != "00000000" else None
        except ValueError as exc:
            raise MT5AuthError("MT5_LOGIN must be an integer account number") from exc
        pw = env.get("MT5_PASSWORD") or None
        if pw == "change-me":
            pw = None
        server = (env.get("MT5_SERVER") or "").strip() or None
        path = (env.get("MT5_TERMINAL_PATH") or "").strip() or None
        return Mt5Credentials(login_i, pw, server, path)


class MT5BrokerAdapter(BrokerAdapter):
    def __init__(self, credentials: Mt5Credentials | None = None, *,
                 module_loader: Callable[[], Any] = load_mt5_module,
                 allowed_classes: frozenset[AccountClass] = PHASE1_ALLOWED_ACCOUNT_CLASSES,
                 call_timeout_s: float = 30.0, min_call_interval_s: float = 0.0):
        self._creds = credentials or Mt5Credentials()
        self._loader = module_loader
        self._allowed = allowed_classes
        self._owner = MT5Owner(default_timeout_s=call_timeout_s)
        self._api: ReadOnlyMT5Api | None = None
        self._state = AdapterState.DISCONNECTED
        self._identity: AccountIdentity | None = None
        self._selected: set[str] = set()
        self._min_interval = min_call_interval_s
        self._last_call = 0.0
        self._state_lock = threading.Lock()

    # ---- state ---------------------------------------------------------------------------
    @property
    def state(self) -> AdapterState:
        return self._state

    @property
    def identity(self) -> AccountIdentity | None:
        return self._identity

    def latency_stats(self) -> dict[str, dict[str, float]]:
        return self._owner.latency_stats()

    def _set_state(self, new: AdapterState, why: str = "") -> None:
        with self._state_lock:
            old, self._state = self._state, new
        if old != new:
            log.info("mt5 state %s -> %s %s", old.value, new.value, why, extra={"ctx": {"event": "state", "from": old.value, "to": new.value}})

    # ---- low-level call wrapper --------------------------------------------------------------
    def _call(self, fname: str, *args: Any, timeout_s: float | None = None, require_connected: bool = True,
              **kwargs: Any) -> Any:
        if require_connected and self._state != AdapterState.CONNECTED:
            raise ConnectionLostError(f"MT5 adapter state is {self._state.value}; call connect() first"
                                      if self._state in (AdapterState.DISCONNECTED, AdapterState.CLOSED)
                                      else f"MT5 adapter state is {self._state.value}; reconnect required")
        assert self._api is not None
        fn = self._api.function(fname)
        if self._min_interval:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
        try:
            res = self._owner.call(fn, *args, label=fname, timeout_s=timeout_s, **kwargs)
        except CallTimeoutError:
            self._set_state(AdapterState.LOST, f"timeout in {fname}")
            raise
        except ConnectionLostError:
            self._set_state(AdapterState.LOST, f"owner failure in {fname}")
            raise
        finally:
            self._last_call = time.monotonic()
        return res

    def _last_error(self) -> tuple[int, str]:
        try:
            r = self._owner.call(self._api.function("last_error"), label="last_error", timeout_s=5)  # type: ignore[union-attr]
            return int(r[0]), str(r[1])
        except Exception:  # noqa: BLE001 - diagnostics must never mask the original failure
            return 0, "last_error unavailable"

    def _raise_for_none(self, what: str) -> None:
        code, msg = self._last_error()
        if code in _IPC_ERROR_RANGE:
            self._set_state(AdapterState.LOST, f"IPC error {code} in {what}")
            raise ConnectionLostError(f"{what}: terminal IPC failure ({code}, {msg}); restart the terminal and reconnect")
        raise DataFormatError(f"{what} returned None (last_error=({code}, {msg}))")

    # ---- lifecycle ---------------------------------------------------------------------------
    def connect(self) -> AccountIdentity:
        if self._state == AdapterState.CONNECTED and self._identity:
            return self._identity
        self._set_state(AdapterState.CONNECTING)
        module = self._loader()                       # MT5UnavailableError if unusable
        self._api = ReadOnlyMT5Api(module)
        self._owner.start()
        kwargs: dict[str, Any] = {"timeout": self._creds.timeout_ms}
        if self._creds.login is not None:
            kwargs["login"] = self._creds.login
            if self._creds.password:
                kwargs["password"] = self._creds.password
            if self._creds.server:
                kwargs["server"] = self._creds.server
        args = (self._creds.path,) if self._creds.path else ()
        log.info("mt5 initialize requested", extra={"ctx": {"event": "initialize", "login": mask_login(self._creds.login),
                                                            "server": self._creds.server, "path_set": bool(self._creds.path)}})
        try:
            ok = self._call("initialize", *args, require_connected=False, timeout_s=self._creds.timeout_ms / 1000 + 15, **kwargs)
        except (CallTimeoutError, ConnectionLostError):
            self._owner.stop()
            raise
        if not ok:
            code, msg = self._last_error()
            self._owner.stop()
            self._set_state(AdapterState.DISCONNECTED)
            if code == _RES_AUTH_FAILED:
                raise MT5AuthError(f"MT5 login failed ({code}, {msg}): check MT5_LOGIN, MT5_PASSWORD and MT5_SERVER "
                                   "for the DEMO account, then retry")
            raise MT5InitializationError(
                f"MT5 initialize() failed ({code}, {msg}): ensure the MetaTrader 5 terminal is installed, started "
                "and logged in, and MT5_TERMINAL_PATH (if set) points at terminal64.exe")
        try:
            acct = self._call("account_info", require_connected=False)
            if acct is None:
                self._raise_for_none("account_info")
            info = convert.to_account_info(acct)
            ident = info.identity
            # identity cross-checks
            if self._creds.login is not None and mask_login(self._creds.login) != ident.login_masked:
                raise AccountMismatchError(
                    f"Terminal is logged into account {ident.login_masked} but MT5_LOGIN is "
                    f"{mask_login(self._creds.login)}; refusing to collect data from an unexpected account")
            if self._creds.server and ident.server and self._creds.server.lower() != ident.server.lower():
                raise AccountMismatchError(
                    f"Terminal server {ident.server!r} differs from MT5_SERVER {self._creds.server!r}")
            log.info("mt5 account classified", extra={"ctx": {"event": "account", "login": ident.login_masked,
                                                              "server": ident.server, "class": ident.account_class.value,
                                                              "trade_mode": ident.trade_mode_raw}})
            assert_account_allowed(ident, self._allowed)
            term = self._call("terminal_info", require_connected=False)
            tinfo = convert.to_terminal_info(term)
            if not tinfo.connected:
                raise MT5InitializationError("Terminal is running but not connected to the trade server; check the "
                                             "terminal's connection indicator and internet access")
        except Exception as exc:
            try:
                self._call("shutdown", require_connected=False, timeout_s=10)
            except Exception:  # noqa: BLE001
                pass
            self._owner.stop()
            self._set_state(AdapterState.REJECTED if isinstance(exc, AccountNotAllowedError) else AdapterState.DISCONNECTED)
            raise
        self._identity = ident
        self._set_state(AdapterState.CONNECTED)
        return ident

    def disconnect(self) -> None:
        if self._api is not None and self._owner._thread and self._owner._thread.is_alive() and not self._owner.wedged:
            try:
                self._call("shutdown", require_connected=False, timeout_s=10)
            except Exception as exc:  # noqa: BLE001
                log.warning("mt5 shutdown raised %s", type(exc).__name__)
        self._owner.stop()
        self._selected.clear()
        self._set_state(AdapterState.CLOSED)

    def reconnect(self, max_attempts: int = 3, backoff_s: float = 2.0) -> AccountIdentity:
        last: Exception | None = None
        for i in range(max_attempts):
            try:
                try:
                    self.disconnect()
                except Exception:  # noqa: BLE001
                    pass
                self._owner = MT5Owner(default_timeout_s=self._owner._default_timeout)
                self._state = AdapterState.DISCONNECTED
                return self.connect()
            except AccountNotAllowedError:  # never retry a safety rejection
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
                log.warning("mt5 reconnect attempt %d/%d failed: %s", i + 1, max_attempts, type(exc).__name__)
                time.sleep(backoff_s * (2 ** i))
        assert last is not None
        raise last

    def health_check(self) -> AdapterHealth:
        if self._state != AdapterState.CONNECTED:
            return AdapterHealth(False, self._state.value, "not connected")
        try:
            ti = convert.to_terminal_info(self._call("terminal_info", timeout_s=10))
            return AdapterHealth(ti.connected, self._state.value,
                                 "ok" if ti.connected else "terminal not connected to trade server",
                                 terminal_connected=ti.connected, ping_last_us=ti.ping_last_us)
        except Exception as exc:  # noqa: BLE001 - health checks report, never raise
            return AdapterHealth(False, self._state.value, f"{type(exc).__name__}: {exc}")

    # ---- read API ----------------------------------------------------------------------------
    def get_terminal_info(self) -> TerminalInfo:
        ti = self._call("terminal_info")
        if ti is None:
            self._raise_for_none("terminal_info")
        ver = None
        try:
            ver = self._call("version")
        except Exception:  # noqa: BLE001
            pass
        return convert.to_terminal_info(ti, ver)

    def get_account_info(self) -> AccountInfo:
        a = self._call("account_info")
        if a is None:
            self._raise_for_none("account_info")
        return convert.to_account_info(a)

    def discover_symbols(self, group: str | None = None) -> list[SymbolInfo]:
        res = self._call("symbols_get", group) if group else self._call("symbols_get")
        if res is None:
            self._raise_for_none("symbols_get")
        out = [convert.to_symbol_info(s) for s in res]
        log.info("mt5 symbols discovered", extra={"ctx": {"event": "symbols", "count": len(out), "group": group}})
        return out

    def _ensure_selected(self, symbol: str) -> None:
        # symbol_select only adds the symbol to Market Watch (data visibility); it does not trade.
        if symbol in self._selected:
            return
        ok = self._call("symbol_select", symbol, True)
        if not ok:
            raise SymbolNotFoundError(f"Symbol {symbol!r} could not be selected in Market Watch (does it exist on "
                                      "this server? check discovery output)")
        self._selected.add(symbol)

    def get_symbol_info(self, broker_symbol: str) -> SymbolInfo:
        res = self._call("symbol_info", broker_symbol)
        if res is None:
            code, msg = self._last_error()
            if code in _IPC_ERROR_RANGE:
                self._raise_for_none("symbol_info")
            raise SymbolNotFoundError(f"Symbol {broker_symbol!r} not found on this server (last_error=({code}, {msg}))")
        return convert.to_symbol_info(res)

    def get_latest_tick(self, broker_symbol: str, *, max_age_s: float | None = None, offset_s: int = 0,
                        now_utc_s: float | None = None) -> Tick:
        """Latest tick. If ``max_age_s`` is given, the age is computed with the supplied source-time
        offset (source - UTC, from calibration) and StaleTickError is raised when exceeded."""
        self._ensure_selected(broker_symbol)
        res = self._call("symbol_info_tick", broker_symbol)
        if res is None:
            code, msg = self._last_error()
            if code in _IPC_ERROR_RANGE:
                self._raise_for_none("symbol_info_tick")
            raise NoTickDataError(f"No latest tick for {broker_symbol!r} (last_error=({code}, {msg}))")
        tick = convert.to_tick(res)
        if tick.time_msc_raw == 0 and tick.time_raw == 0:
            raise NoTickDataError(f"Latest tick for {broker_symbol!r} has no timestamp (symbol has not quoted yet)")
        if max_age_s is not None:
            now = time.time() if now_utc_s is None else now_utc_s
            age = now - (tick.time_msc_raw / 1000.0 - offset_s)
            if age > max_age_s:
                raise StaleTickError(f"Latest {broker_symbol} tick is {age:.0f}s old (limit {max_age_s}s): market "
                                     "closed, feed down, or local clock/time offset wrong")
            if age < -max_age_s:
                raise StaleTickError(f"Latest {broker_symbol} tick is {-age:.0f}s in the FUTURE: the supplied time "
                                     "offset or the local clock is wrong")
        return tick

    def get_ticks_range(self, broker_symbol: str, start_epoch_s: int, end_epoch_s: int) -> np.ndarray:
        if end_epoch_s < start_epoch_s:
            raise ValueError("end before start")
        self._ensure_selected(broker_symbol)
        t0 = datetime.fromtimestamp(start_epoch_s, tz=timezone.utc)   # tz-aware: epoch preserved exactly
        t1 = datetime.fromtimestamp(end_epoch_s, tz=timezone.utc)
        flags = self._api.COPY_TICKS_ALL  # type: ignore[union-attr]
        t_start = time.perf_counter()
        res = self._call("copy_ticks_range", broker_symbol, t0, t1, flags)
        elapsed = time.perf_counter() - t_start
        if res is None:
            code, msg = self._last_error()
            if code in _IPC_ERROR_RANGE:
                self._raise_for_none("copy_ticks_range")
            raise NoTickDataError(f"copy_ticks_range({broker_symbol}) returned no data for [{start_epoch_s}, "
                                  f"{end_epoch_s}] (last_error=({code}, {msg}))")
        arr = convert.ticks_to_array(res)
        log.info("mt5 ticks fetched", extra={"ctx": {"event": "ticks", "symbol": broker_symbol, "rows": int(len(arr)),
                                                     "start": start_epoch_s, "end": end_epoch_s,
                                                     "latency_ms": round(elapsed * 1000, 2)}})
        return arr

    def get_ticks_from(self, broker_symbol: str, start_epoch_s: int, count: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("count must be positive")
        self._ensure_selected(broker_symbol)
        res = self._call("copy_ticks_from", broker_symbol, datetime.fromtimestamp(start_epoch_s, tz=timezone.utc),
                         int(count), self._api.COPY_TICKS_ALL)  # type: ignore[union-attr]
        if res is None:
            code, msg = self._last_error()
            if code in _IPC_ERROR_RANGE:
                self._raise_for_none("copy_ticks_from")
            raise NoTickDataError(f"copy_ticks_from({broker_symbol}) returned no data (last_error=({code}, {msg}))")
        return convert.ticks_to_array(res)

    def get_rates_range(self, broker_symbol: str, timeframe: str, start_epoch_s: int, end_epoch_s: int) -> np.ndarray:
        if timeframe not in _TIMEFRAMES:
            raise ValueError(f"unsupported timeframe {timeframe!r}; use one of {_TIMEFRAMES}")
        self._ensure_selected(broker_symbol)
        tf = getattr(self._api, f"TIMEFRAME_{timeframe}")
        res = self._call("copy_rates_range", broker_symbol, tf,
                         datetime.fromtimestamp(start_epoch_s, tz=timezone.utc),
                         datetime.fromtimestamp(end_epoch_s, tz=timezone.utc))
        if res is None:
            code, msg = self._last_error()
            if code in _IPC_ERROR_RANGE:
                self._raise_for_none("copy_rates_range")
            raise NoTickDataError(f"copy_rates_range({broker_symbol},{timeframe}) returned no data (last_error=({code}, {msg}))")
        return convert.rates_to_array(res)

    def get_market_book(self, broker_symbol: str) -> list[dict[str, Any]] | None:
        """Depth of market (read-only subscription). None if unavailable for this symbol/broker."""
        self._ensure_selected(broker_symbol)
        if not self._call("market_book_add", broker_symbol):
            return None
        try:
            book = self._call("market_book_get", broker_symbol)
            return None if book is None else [convert.struct_to_dict(b) for b in book]
        finally:
            try:
                self._call("market_book_release", broker_symbol)
            except Exception:  # noqa: BLE001
                pass

    # ---- trading surface: disabled ---------------------------------------------------------
    @staticmethod
    def _disabled(what: str) -> TradingDisabledError:
        return TradingDisabledError(f"{what} is disabled: Phase 1 is a read-only data foundation. "
                                    "No orders, positions or order history are accessed.")

    def get_positions(self) -> list[dict]:
        raise self._disabled("get_positions")

    def get_orders(self) -> list[dict]:
        raise self._disabled("get_orders")

    def submit_order(self, request: OrderRequest) -> OrderResult:
        raise self._disabled("submit_order")

    def modify_order(self, broker_order_id: str, sl: float | None, tp: float | None) -> OrderResult:
        raise self._disabled("modify_order")

    def cancel_order(self, broker_order_id: str) -> OrderResult:
        raise self._disabled("cancel_order")

    def close_position(self, broker_position_id: str) -> OrderResult:
        raise self._disabled("close_position")


class HealthMonitor:
    """Periodic health_check in a background thread. Records status; does NOT auto-reconnect by default."""

    def __init__(self, adapter: MT5BrokerAdapter, interval_s: float = 10.0, failures_to_degrade: int = 2,
                 on_degraded: Callable[[AdapterHealth], None] | None = None):
        self._adapter, self._interval, self._n = adapter, interval_s, failures_to_degrade
        self._on_degraded = on_degraded
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last: AdapterHealth | None = None
        self.consecutive_failures = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mt5-health", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self._interval + 5)

    def check_once(self) -> AdapterHealth:
        h = self._adapter.health_check()
        self.last = h
        self.consecutive_failures = 0 if h.ok else self.consecutive_failures + 1
        if not h.ok:
            log.warning("mt5 health check failed: %s", h.detail, extra={"ctx": {"event": "health", "ok": False}})
            if self.consecutive_failures >= self._n and self._on_degraded:
                self._on_degraded(h)
        return h

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.check_once()
