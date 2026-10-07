"""FakeMT5Module / FakeMT5Adapter: a deterministic stand-in for the MetaTrader5 package (Linux CI).

The fake has the same function surface and field names as the real package so the REAL
MT5BrokerAdapter runs unchanged on top of it. It is a TEST DOUBLE: none of its behaviour is evidence
about how a real terminal behaves (that is what scripts/verify_mt5.py on Windows is for).

Ground truth is known to tests: the fake generates ticks in UTC and presents them in a configurable
source-time domain (fixed offset, or a DST-following rule), so time calibration can be validated.

Trading functions exist only as tripwires: they record the call and raise, so tests can assert that
no order function was ever invoked.
"""

from __future__ import annotations

import fnmatch
import threading
import time as _time
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from fxscalp.brokers.mt5.adapter import MT5BrokerAdapter, Mt5Credentials
from fxscalp.market_data.timebase import ServerTimeRule

_TICK = namedtuple("Tick", "time bid ask last volume time_msc flags volume_real")
_RAW_DT = np.dtype([("time", "<i8"), ("bid", "<f8"), ("ask", "<f8"), ("last", "<f8"), ("volume", "<u8"),
                    ("time_msc", "<i8"), ("flags", "<u4"), ("volume_real", "<f8")])
_ACCOUNT = namedtuple("AccountInfo", "login trade_mode leverage limit_orders margin_so_mode trade_allowed "
                      "trade_expert margin_mode currency_digits fifo_close balance credit profit equity margin "
                      "margin_free margin_level margin_so_call margin_so_so margin_initial margin_maintenance "
                      "assets liabilities commission_blocked name server currency company")
_TERMINAL = namedtuple("TerminalInfo", "community_account community_connection connected dlls_allowed "
                       "trade_allowed tradeapi_disabled email_enabled ftp_enabled notifications_enabled mqid "
                       "build maxbars codepage ping_last community_balance retransmission company name language "
                       "path data_path commondata_path")
_SYMBOL_FIELDS = ("custom select visible session_deals session_buy_orders session_sell_orders volume volumehigh "
                  "volumelow time digits spread spread_float ticks_bookdepth trade_calc_mode trade_mode "
                  "trade_stops_level trade_freeze_level trade_exemode swap_mode swap_rollover3days "
                  "start_time expiration_time trade_tick_value trade_tick_value_profit trade_tick_value_loss "
                  "trade_tick_size trade_contract_size volume_min volume_max volume_step volume_limit swap_long "
                  "swap_short margin_initial margin_maintenance filling_mode expiration_mode order_mode "
                  "currency_base currency_profit currency_margin bank description exchange formula isin name "
                  "page path point")
_SYMBOL = namedtuple("SymbolInfo", _SYMBOL_FIELDS)
_BOOK = namedtuple("BookInfo", "type price volume volume_dbl")

_TF_MIN = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


@dataclass
class FakeScenario:
    # session / account
    init_ok: bool = True
    init_error: tuple[int, str] = (-10003, "IPC initialize failed")
    auth_ok: bool = True
    trade_mode: int = 0                   # 0 demo, 1 contest, 2 real, other = unknown
    login: int = 12345678
    server: str = "FakeBroker-Demo"
    company: str = "Fake Broker Ltd"
    currency: str = "USD"
    terminal_connected: bool = True
    # symbols
    symbol_names: tuple[str, ...] = ("XAUUSDm", "EURUSDm", "GBPUSDm", "BTCUSDm")
    extra_symbols: dict[str, dict[str, Any]] = field(default_factory=dict)
    # time
    rule: ServerTimeRule = field(default_factory=lambda: ServerTimeRule.fixed(3 * 3600))
    now_utc_s: Callable[[], float] = _time.time
    latest_tick_age_s: float = 0.3
    # data
    ticks_per_day: int = 20000
    seed: int = 7
    weekend_closure: bool = True
    max_rows_per_request: int | None = None
    inclusive_end: bool = True
    no_ticks_symbols: tuple[str, ...] = ()
    drop_volume_real: bool = False
    # injected anomalies (probabilities per tick, deterministic by seed)
    neg_spread_prob: float = 0.0
    dup_prob: float = 0.0
    reversal_prob: float = 0.0
    nan_prob: float = 0.0
    zero_ask_prob: float = 0.0
    corrupt_return_type: bool = False
    # failure injection
    fail_after_calls: int | None = None     # after N calls every function fails with an IPC error
    hang_function: str | None = None
    hang_s: float = 0.0
    dom_available: bool = False
    history_start_s: float | None = None    # ticks before this UTC epoch do not exist (history depth)
    #: cold-history simulation: the first ``slow_first_n`` calls of ``slow_function`` sleep ``slow_s`` seconds
    slow_function: str | None = None
    slow_s: float = 0.0
    slow_first_n: int = 0
    #: the first ``terminal_error_first_n`` calls of ``terminal_error_function`` return None with a non-IPC error
    terminal_error_function: str | None = None
    terminal_error_first_n: int = 0
    terminal_error: tuple[int, str] = (-1, "Terminal: Call failed")
    #: the first ``none_first_n`` calls of copy_ticks_range return None with last_error=(1, "Success") (unavailable)
    none_first_n: int = 0


class FakeMT5Module:
    # constants (subset; values identical to the verified package constants)
    COPY_TICKS_ALL, COPY_TICKS_INFO, COPY_TICKS_TRADE = -1, 1, 2
    TICK_FLAG_BID, TICK_FLAG_ASK, TICK_FLAG_LAST = 0x02, 0x04, 0x08
    ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_CONTEST, ACCOUNT_TRADE_MODE_REAL = 0, 1, 2
    RES_S_OK, RES_E_FAIL, RES_E_NOT_FOUND, RES_E_AUTH_FAILED = 1, -1, -4, -6
    TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_M30 = 1, 5, 15, 30
    TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1 = 1 | 0x4000, 4 | 0x4000, 24 | 0x4000

    def __init__(self, scenario: FakeScenario | None = None):
        self.sc = scenario or FakeScenario()
        self.calls: list[str] = []
        self.trading_calls: list[str] = []
        self.threads: set[int] = set()          # ids of threads that executed any MT5 function
        self._err: tuple[int, str] = (1, "Success")
        self._initialized = False
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        self._n_calls: dict[str, int] = {}

    # ---------------- plumbing ----------------
    def _enter(self, name: str) -> bool:
        """Returns False if the call should fail (connection loss injection)."""
        self.calls.append(name)
        self.threads.add(threading.get_ident())
        self._n_calls[name] = k = self._n_calls.get(name, 0) + 1
        if self.sc.hang_function == name:
            _time.sleep(self.sc.hang_s)
        if self.sc.slow_function == name and k <= self.sc.slow_first_n:
            _time.sleep(self.sc.slow_s)
        if self.sc.terminal_error_function == name and k <= self.sc.terminal_error_first_n:
            self._err = self.sc.terminal_error
            return False
        if name == "copy_ticks_range" and k <= self.sc.none_first_n:
            self._err = (1, "Success")
            return False
        if self.sc.fail_after_calls is not None and len(self.calls) > self.sc.fail_after_calls:
            self._err = (-10001, "IPC send failed")
            return False
        return True

    def _specs(self, name: str) -> dict[str, Any]:
        if name in self.sc.extra_symbols:
            return self.sc.extra_symbols[name]
        u = name.upper()
        if "XAU" in u or "GOLD" in u:
            return dict(base="XAU", prof="USD", digits=2, point=0.01, tsize=0.01, tval=1.0, csize=100.0, price=2000.0,
                        spread_pts=25, vmin=0.01, vmax=100.0, vstep=0.01, calc=2)
        if "EUR" in u:
            return dict(base="EUR", prof="USD", digits=5, point=1e-5, tsize=1e-5, tval=1.0, csize=100000.0,
                        price=1.08, spread_pts=12, vmin=0.01, vmax=200.0, vstep=0.01, calc=0)
        if "GBP" in u:
            return dict(base="GBP", prof="USD", digits=5, point=1e-5, tsize=1e-5, tval=1.0, csize=100000.0,
                        price=1.27, spread_pts=15, vmin=0.01, vmax=200.0, vstep=0.01, calc=0)
        return dict(base="BTC", prof="USD", digits=2, point=0.01, tsize=0.01, tval=0.01, csize=1.0, price=60000.0,
                    spread_pts=3000, vmin=0.01, vmax=10.0, vstep=0.01, calc=2)

    def _symbol(self, name: str) -> Any:
        s = self._specs(name)
        vals = dict(custom=False, select=True, visible=True, session_deals=0, session_buy_orders=0,
                    session_sell_orders=0, volume=0, volumehigh=0, volumelow=0, time=0, digits=s["digits"],
                    spread=s["spread_pts"], spread_float=True, ticks_bookdepth=0, trade_calc_mode=s["calc"],
                    trade_mode=4, trade_stops_level=10, trade_freeze_level=0, trade_exemode=2, swap_mode=1,
                    swap_rollover3days=3, start_time=0, expiration_time=0, trade_tick_value=s["tval"],
                    trade_tick_value_profit=s["tval"], trade_tick_value_loss=s["tval"], trade_tick_size=s["tsize"],
                    trade_contract_size=s["csize"], volume_min=s["vmin"], volume_max=s["vmax"],
                    volume_step=s["vstep"], volume_limit=0.0, swap_long=-5.0, swap_short=2.0, margin_initial=0.0,
                    margin_maintenance=0.0, filling_mode=3, expiration_mode=15, order_mode=127,
                    currency_base=s["base"], currency_profit=s["prof"], currency_margin=s["base"], bank="",
                    description=f"{name} (fake)", exchange="", formula="", isin="", name=name, page="",
                    path=f"Fake\\{name}", point=s["point"])
        return _SYMBOL(**vals)

    # ---------------- lifecycle ----------------
    def initialize(self, *args: Any, **kwargs: Any) -> bool:
        if not self._enter("initialize"):
            return False
        if not self.sc.init_ok:
            self._err = self.sc.init_error
            return False
        if not self.sc.auth_ok:
            self._err = (-6, "Authorization failed")
            return False
        self._initialized = True
        self._err = (1, "Success")
        return True

    def shutdown(self) -> None:
        self.calls.append("shutdown")
        self.threads.add(threading.get_ident())
        self._initialized = False

    def version(self) -> tuple[int, int, str]:
        return (500, 5602, "24 Sep 2026")

    def last_error(self) -> tuple[int, str]:
        self.calls.append("last_error")
        return self._err

    # ---------------- state ----------------
    def terminal_info(self) -> Any:
        if not self._enter("terminal_info"):
            return None
        return _TERMINAL(False, False, self.sc.terminal_connected, False, True, False, False, False, False, 0, 5602,
                         100000, 1251, 42000, 0.0, 0, self.sc.company, "FakeTerminal", "English",
                         "C:\\FakeMT5", "C:\\FakeMT5\\data", "C:\\Common")

    def account_info(self) -> Any:
        if not self._enter("account_info"):
            return None
        return _ACCOUNT(self.sc.login, self.sc.trade_mode, 500, 200, 0, True, True, 2, 2, False, 10000.0, 0.0, 0.0,
                        10000.0, 0.0, 10000.0, 0.0, 50.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Fake Trader",
                        self.sc.server, self.sc.currency, self.sc.company)

    # ---------------- symbols ----------------
    def symbols_total(self) -> int:
        self._enter("symbols_total")
        return len(self.sc.symbol_names)

    def symbols_get(self, group: str | None = None) -> Any:
        if not self._enter("symbols_get"):
            return None
        names = [n for n in self.sc.symbol_names if group is None or fnmatch.fnmatch(n, group)]
        return tuple(self._symbol(n) for n in names)

    def symbol_info(self, symbol: str) -> Any:
        if not self._enter("symbol_info"):
            return None
        if symbol not in self.sc.symbol_names:
            self._err = (-4, "Terminal: Not found")
            return None
        return self._symbol(symbol)

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        if not self._enter("symbol_select"):
            return False
        return symbol in self.sc.symbol_names

    # ---------------- ticks ----------------
    def _utc_hour_ticks(self, symbol: str, hour_idx: int) -> np.ndarray:
        key = (symbol, hour_idx)
        if key in self._cache:
            return self._cache[key]
        sc, sp = self.sc, self._specs(symbol)
        h0 = hour_idx * 3600
        closed = False
        if sc.weekend_closure:
            # weekly closure anchored to New York local time (Fri 17:00 -> Sun 17:00), like spot FX
            ny = pd.Timestamp(h0, unit="s", tz="UTC").tz_convert("America/New_York")
            wd, hr = ny.weekday(), ny.hour
            closed = (wd == 4 and hr >= 17) or wd == 5 or (wd == 6 and hr < 17)
        if closed or (sc.history_start_s is not None and h0 + 3600 <= sc.history_start_s):
            arr = np.empty(0, dtype=_RAW_DT)
        else:
            rng = np.random.default_rng([sc.seed, hour_idx, sum(map(ord, symbol))])
            n = int(rng.poisson(sc.ticks_per_day / 24.0))
            ms = np.sort(rng.integers(0, 3_600_000, n))
            utc_ms = h0 * 1000 + ms
            t_s = utc_ms / 1000.0
            mid = sp["price"] * (1 + 0.003 * np.sin(t_s / 7200.0)) + rng.normal(0, sp["price"] * 2e-5, n)
            widen = 4.0 if (h0 // 3600) % 24 in (21, 22) else 1.0
            spread = sp["spread_pts"] * sp["point"] * widen * (1 + rng.exponential(0.15, n))
            bid = np.round(mid - spread / 2, sp["digits"])
            ask = np.round(bid + np.maximum(spread, sp["point"]), sp["digits"])
            src_ms = utc_ms + sc.rule.offset_at_utc_ms(utc_ms) * 1000
            arr = np.empty(n, dtype=_RAW_DT)
            arr["time"], arr["time_msc"] = src_ms // 1000, src_ms
            arr["bid"], arr["ask"], arr["last"] = bid, ask, 0.0
            arr["volume"], arr["volume_real"] = 0, 0.0
            arr["flags"] = rng.choice([6, 2, 4], n, p=[0.8, 0.1, 0.1])
            self._inject(arr, rng)
        self._cache[key] = arr
        return arr

    def _inject(self, arr: np.ndarray, rng: np.random.Generator) -> None:
        n, sc = len(arr), self.sc
        if n < 3:
            return
        if sc.neg_spread_prob:
            m = rng.random(n) < sc.neg_spread_prob
            arr["ask"][m] = arr["bid"][m] - 0.05
        if sc.zero_ask_prob:
            arr["ask"][rng.random(n) < sc.zero_ask_prob] = 0.0
        if sc.nan_prob:
            arr["bid"][rng.random(n) < sc.nan_prob] = np.nan
        if sc.dup_prob:
            m = np.flatnonzero(rng.random(n) < sc.dup_prob)
            m = m[m > 0]
            arr[m] = arr[m - 1]
        if sc.reversal_prob:
            m = np.flatnonzero(rng.random(n) < sc.reversal_prob)
            m = m[m > 0]
            arr["time_msc"][m] = arr["time_msc"][m - 1] - 5000
            arr["time"][m] = arr["time_msc"][m] // 1000

    def _range(self, symbol: str, s_ms: int, e_ms: int) -> np.ndarray:
        # source -> candidate UTC hours (generous window), then filter by source time
        off = int(self.sc.rule.offset_at_utc_ms(np.array([s_ms]))[0]) * 1000
        h_from, h_to = (s_ms - off) // 3_600_000 - 2, (e_ms - off) // 3_600_000 + 2
        parts = [self._utc_hour_ticks(symbol, h) for h in range(int(h_from), int(h_to) + 1)]
        arr = np.concatenate(parts) if parts else np.empty(0, dtype=_RAW_DT)
        hi = e_ms if self.sc.inclusive_end else e_ms - 1
        arr = arr[(arr["time_msc"] >= s_ms) & (arr["time_msc"] <= hi)]
        if self.sc.max_rows_per_request is not None:
            arr = arr[: self.sc.max_rows_per_request]
        if self.sc.drop_volume_real:
            arr = arr[[n for n in _RAW_DT.names if n != "volume_real"]]
        return arr

    def copy_ticks_range(self, symbol: str, date_from: Any, date_to: Any, flags: int) -> Any:
        if not self._enter("copy_ticks_range"):
            return None
        if symbol in self.sc.no_ticks_symbols or symbol not in self.sc.symbol_names:
            self._err = (-4, "Terminal: Not found")
            return None
        if self.sc.corrupt_return_type:
            return [(1, 2, 3)]
        s = int(date_from.timestamp() * 1000) if isinstance(date_from, datetime) else int(date_from) * 1000
        e = int(date_to.timestamp() * 1000) if isinstance(date_to, datetime) else int(date_to) * 1000
        return self._range(symbol, s, e)

    def copy_ticks_from(self, symbol: str, date_from: Any, count: int, flags: int) -> Any:
        if not self._enter("copy_ticks_from"):
            return None
        s = int(date_from.timestamp() * 1000) if isinstance(date_from, datetime) else int(date_from) * 1000
        if self.sc.history_start_s is not None:
            # like the real terminal: the first tick AFTER the date, even if that is the start of all history
            hs = int(self.sc.history_start_s * 1000)
            hs += int(self.sc.rule.offset_at_utc_ms(np.array([hs]))[0]) * 1000
            s = max(s, hs)
        return self._range(symbol, s, s + 7 * 24 * 3600 * 1000)[:count]

    def symbol_info_tick(self, symbol: str) -> Any:
        if not self._enter("symbol_info_tick"):
            return None
        if symbol not in self.sc.symbol_names or symbol in self.sc.no_ticks_symbols:
            self._err = (-4, "Terminal: Not found")
            return None
        sp = self._specs(symbol)
        now_utc = self.sc.now_utc_s()
        utc_ms = int((now_utc - self.sc.latest_tick_age_s) * 1000)
        off = int(self.sc.rule.offset_at_utc_ms(np.array([utc_ms]))[0])
        src_ms = utc_ms + off * 1000
        bid = round(sp["price"], sp["digits"])
        ask = round(bid + sp["spread_pts"] * sp["point"], sp["digits"])
        return _TICK(src_ms // 1000, bid, ask, 0.0, 0, src_ms, 6, 0.0)

    def copy_rates_range(self, symbol: str, timeframe: int, date_from: Any, date_to: Any) -> Any:
        if not self._enter("copy_rates_range"):
            return None
        mins = timeframe & 0x3FFF if timeframe < 0x4000 else (timeframe & 0x3FFF) * 60
        s = int(date_from.timestamp()) if isinstance(date_from, datetime) else int(date_from)
        e = int(date_to.timestamp()) if isinstance(date_to, datetime) else int(date_to)
        ticks = self._range(symbol, s * 1000, e * 1000)
        dt = np.dtype([("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
                       ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")])
        if len(ticks) == 0:
            return np.empty(0, dtype=dt)
        step = mins * 60
        bucket = (ticks["time"] // step) * step
        uniq, idx = np.unique(bucket, return_index=True)
        out = np.empty(len(uniq), dtype=dt)
        mid = (ticks["bid"] + ticks["ask"]) / 2
        bounds = list(idx) + [len(ticks)]
        for i, u in enumerate(uniq):
            seg = mid[bounds[i]:bounds[i + 1]]
            out[i] = (u, seg[0], seg.max(), seg.min(), seg[-1], len(seg), 0, 0)
        return out

    # ---------------- DOM ----------------
    def market_book_add(self, symbol: str) -> bool:
        self._enter("market_book_add")
        return self.sc.dom_available

    def market_book_get(self, symbol: str) -> Any:
        self._enter("market_book_get")
        return (_BOOK(1, 2000.1, 10, 10.0), _BOOK(2, 1999.9, 10, 10.0)) if self.sc.dom_available else None

    def market_book_release(self, symbol: str) -> bool:
        self._enter("market_book_release")
        return True

    # ---------------- trading tripwires (must never be called) ----------------
    def _trip(self, name: str) -> Any:
        self.trading_calls.append(name)
        raise AssertionError(f"FAKE MT5 TRIPWIRE: trading function {name!r} was called")

    def order_send(self, *a: Any, **k: Any) -> Any:
        return self._trip("order_send")

    def order_check(self, *a: Any, **k: Any) -> Any:
        return self._trip("order_check")

    def positions_get(self, *a: Any, **k: Any) -> Any:
        return self._trip("positions_get")

    def orders_get(self, *a: Any, **k: Any) -> Any:
        return self._trip("orders_get")


class FakeMT5Adapter(MT5BrokerAdapter):
    """The real MT5BrokerAdapter wired to a FakeMT5Module. ``adapter.fake`` exposes the module for assertions."""

    def __init__(self, scenario: FakeScenario | None = None, credentials: Mt5Credentials | None = None, **kw: Any):
        self.fake = FakeMT5Module(scenario)
        super().__init__(credentials, module_loader=lambda: self.fake, **kw)
