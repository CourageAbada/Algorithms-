"""HistoricalDataProvider abstraction (decision D-3).

Phase 0.5: interface only. Start with broker MT5 tick history; a higher-quality
external tick vendor can later be added as another implementation without
touching features, models, risk or backtests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from fxscalp.brokers.base import Tick


@dataclass(frozen=True)
class DatasetProvenance:
    """Where a tick set came from. Models are only valid for the feed they were trained on."""

    provider: str          # e.g. "mt5:<broker-id>", "vendor:<name>"
    symbol_canonical: str
    broker_symbol: str | None
    server: str | None
    time_basis: str        # "utc_verified" | "server_time_calibrated" | "unverified"
    utc_offset_seconds: int | None  # offset applied to raw timestamps, with how it was derived stored elsewhere


class HistoricalDataProvider(ABC):
    @abstractmethod
    def provenance(self, canonical: str) -> DatasetProvenance: ...

    @abstractmethod
    def get_ticks(self, canonical: str, start_utc: datetime, end_utc: datetime) -> list[Tick]:
        """Return ticks with tz-aware UTC timestamps, chronologically ordered, raw (no cleaning)."""
