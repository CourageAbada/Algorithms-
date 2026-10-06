"""Broker-agnostic interface. Business logic must only import from this module.

Phase 0: interface and neutral domain types only. MT5BrokerAdapter arrives in Phase 1.
Monetary/size fields use float here; Phase 1 reviews Decimal for order-facing values.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class SymbolInfo:
    """Broker-resolved instrument metadata. Risk maths must use these, never constants."""

    canonical: str
    broker_symbol: str
    digits: int
    tick_size: float
    tick_value: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level_points: int
    freeze_level_points: int
    currency_profit: str
    currency_margin: str
    trade_allowed: bool


@dataclass(frozen=True)
class Tick:
    time_utc: datetime  # tz-aware UTC, converted from broker server time at the boundary
    time_msc: int
    bid: float
    ask: float
    last: float
    volume: float
    flags: int


@dataclass(frozen=True)
class AccountInfo:
    broker: str
    is_demo: bool
    currency: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    leverage: int


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
    submitted_at_utc: datetime
    acknowledged_at_utc: datetime | None


class BrokerAdapter(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def health_check(self) -> bool: ...

    @abstractmethod
    def get_symbol_info(self, canonical: str) -> SymbolInfo: ...

    @abstractmethod
    def get_ticks(self, canonical: str, start_utc: datetime, end_utc: datetime) -> list[Tick]: ...

    @abstractmethod
    def get_account(self) -> AccountInfo: ...

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
