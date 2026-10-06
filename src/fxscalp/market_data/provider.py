"""HistoricalDataProvider abstraction (decision D-3) and its MT5 implementation.

Phase 1: broker MT5 tick history. A higher-quality external tick vendor can later be added as another
implementation without touching features, models, risk or backtests. Every provider reports provenance
(feed identity + time basis); models are only valid for the feed they were trained on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pyarrow as pa

from fxscalp.market_data.acquire import AcquisitionSummary, TickAcquirer
from fxscalp.market_data.store import ChunkKey
from fxscalp.market_data.timebase import TimeBasis


@dataclass(frozen=True)
class DatasetProvenance:
    provider: str                  # e.g. "mt5:<broker>", "vendor:<name>"
    broker: str
    server: str | None
    canonical: str
    broker_symbol: str | None
    time_basis: TimeBasis
    dst_determined: bool


class HistoricalDataProvider(ABC):
    @abstractmethod
    def provenance(self) -> DatasetProvenance: ...

    @abstractmethod
    def acquire(self, start: date, end: date, *, force: bool = False) -> AcquisitionSummary: ...

    @abstractmethod
    def read_ticks(self, start: date, end: date, columns: list[str] | None = None) -> pa.Table:
        """RAW ticks (verified chunks only), chronologically by chunk, columns as stored."""


class MT5HistoricalDataProvider(HistoricalDataProvider):
    def __init__(self, acquirer: TickAcquirer):
        self._acq = acquirer

    def provenance(self) -> DatasetProvenance:
        a = self._acq
        return DatasetProvenance(f"mt5:{a.broker}", a.broker, a.server, a.canonical, a.sym, a.tb.spec.basis,
                                 a.tb.spec.dst_determined)

    def acquire(self, start: date, end: date, *, force: bool = False) -> AcquisitionSummary:
        return self._acq.acquire_range(start, end, force=force)

    def read_ticks(self, start: date, end: date, columns: list[str] | None = None) -> pa.Table:
        from datetime import timedelta
        a, parts, d = self._acq, [], start
        while d <= end:
            parts.append(a.store.read_raw_chunk(ChunkKey(a.broker, a.server, a.canonical, d), columns))
            d += timedelta(days=1)
        return pa.concat_tables(parts) if parts else pa.table({})


