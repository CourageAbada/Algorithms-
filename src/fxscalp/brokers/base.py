"""Broker-agnostic interfaces and neutral domain types.

Business logic imports only from here. Phase 1 implements the READ-ONLY interface; the trading
methods of ``BrokerAdapter`` exist for the later architecture but must raise TradingDisabledError.

Timestamps: broker times are carried as RAW source-domain integers (``*_raw``). Whether those are
UTC or server time is determined empirically (see market_data/timebase.py); nothing here assumes it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from fxscalp.brokers.accounts import AccountIdentity


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class TerminalInfo:
    """Terminal state. ``raw`` holds every field the terminal returned, unmodified."""

    connected: bool
    trade_allowed: bool | None
    ping_last_us: int | None
    build: int | None
    company: str
    name: str
    version: tuple[int, ...] | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AccountInfo:
    identity: AccountIdentity
    balance: float
    equity: float
    margin: float
    margin_free: float
    raw_fields_present: tuple[str, ...] = ()


@dataclass(frozen=True)
class SymbolInfo:
    """Broker symbol metadata.

    ``raw`` contains every field exactly as returned (no unit conversion). The named attributes are
    a typed projection of the same values. Units are NOT assumed: see convert.check_symbol_consistency.
    """

    canonical: str | None
    broker_symbol: str
    digits: int | None
    point: float | None
    tick_size: float | None            # trade_tick_size
    tick_value: float | None           # trade_tick_value
    tick_value_profit: float | None
    tick_value_loss: float | None
    contract_size: float | None        # trade_contract_size
    volume_min: float | None
    volume_max: float | None
    volume_step: float | None
    volume_limit: float | None
    stops_level: int | None            # trade_stops_level (points)
    freeze_level: int | None           # trade_freeze_level (points)
    spread: int | None                 # current spread in points (as reported)
    spread_float: bool | None          # floating (True) vs fixed (False) spread mode
    trade_mode: int | None
    trade_exemode: int | None
    trade_calc_mode: int | None
    filling_mode: int | None           # bitmask; bit meanings verified on the Windows run
    order_mode: int | None
    expiration_mode: int | None
    swap_mode: int | None
    swap_long: float | None
    swap_short: float | None
    swap_rollover3days: int | None
    currency_base: str
    currency_profit: str
    currency_margin: str
    description: str
    path: str
    visible: bool | None
    select: bool | None
    ticks_bookdepth: int | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Tick:
    """One tick exactly as received (single-tick API). Historical ticks use RAW_TICK_DTYPE arrays."""

    time_raw: int          # seconds, source domain
    time_msc_raw: int      # milliseconds, source domain
    bid: float
    ask: float
    last: float
    volume: int
    volume_real: float
    flags: int


#: Raw tick record layout used for historical acquisition (matches MT5 copy_ticks_* fields).
RAW_TICK_DTYPE = np.dtype([
    ("time", "<i8"), ("bid", "<f8"), ("ask", "<f8"), ("last", "<f8"),
    ("volume", "<u8"), ("time_msc", "<i8"), ("flags", "<u4"), ("volume_real", "<f8"),
])


@dataclass(frozen=True)
class AdapterHealth:
    ok: bool
    state: str
    detail: str
    terminal_connected: bool | None = None
    ping_last_us: int | None = None
    latest_tick_age_s: float | None = None


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Side
    volume: float
    sl: float | None
    tp: float | None
    max_deviation_points: int
    client_tag: str


@dataclass(frozen=True)
class OrderResult:
    accepted: bool
    broker_order_id: str | None
    fill_price: float | None
    filled_volume: float
    retcode: int | None
    comment: str


class ReadOnlyBrokerAdapter(ABC):
    """Everything Phase 1 needs. Times in/out of range queries are RAW source-domain epoch values."""

    @abstractmethod
    def connect(self) -> AccountIdentity: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def health_check(self) -> AdapterHealth: ...

    @abstractmethod
    def get_terminal_info(self) -> TerminalInfo: ...

    @abstractmethod
    def get_account_info(self) -> AccountInfo: ...

    @abstractmethod
    def discover_symbols(self, group: str | None = None) -> list[SymbolInfo]: ...

    @abstractmethod
    def get_symbol_info(self, broker_symbol: str) -> SymbolInfo: ...

    @abstractmethod
    def get_latest_tick(self, broker_symbol: str) -> Tick: ...

    @abstractmethod
    def get_ticks_range(self, broker_symbol: str, start_epoch_s: int, end_epoch_s: int) -> np.ndarray:
        """Ticks in [start, end] as a RAW_TICK_DTYPE array, in the order returned by the broker."""

    @abstractmethod
    def get_ticks_from(self, broker_symbol: str, start_epoch_s: int, count: int) -> np.ndarray:
        """Up to ``count`` ticks starting at ``start`` (RAW_TICK_DTYPE). Used for cheap weekly-open probes."""

    @abstractmethod
    def get_rates_range(self, broker_symbol: str, timeframe: str, start_epoch_s: int,
                        end_epoch_s: int) -> np.ndarray: ...


class BrokerAdapter(ReadOnlyBrokerAdapter):
    """Full interface (later phases). Phase 1 implementations must raise TradingDisabledError."""

    @abstractmethod
    def get_positions(self) -> list[dict]: ...

    @abstractmethod
    def get_orders(self) -> list[dict]: ...

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> OrderResult: ...

    @abstractmethod
    def modify_order(self, broker_order_id: str, sl: float | None, tp: float | None) -> OrderResult: ...

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> OrderResult: ...

    @abstractmethod
    def close_position(self, broker_position_id: str) -> OrderResult: ...

