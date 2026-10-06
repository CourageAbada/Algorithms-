"""Collect latest-tick samples from an adapter for time calibration (read-only)."""

from __future__ import annotations

import time
from typing import Callable

from fxscalp.brokers.base import ReadOnlyBrokerAdapter
from fxscalp.market_data.timebase import OffsetSample


def collect_offset_samples(adapter: ReadOnlyBrokerAdapter, broker_symbol: str, n: int = 30, interval_s: float = 1.0, *,
                           now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                           ) -> list[OffsetSample]:
    """Sample the latest tick ``n`` times. ``local_utc_s`` is the midpoint of the call (receipt time)."""
    out: list[OffsetSample] = []
    for i in range(n):
        t0 = now()
        tick = adapter.get_latest_tick(broker_symbol)
        t1 = now()
        out.append(OffsetSample(local_utc_s=(t0 + t1) / 2, tick_time_s=tick.time_raw, tick_time_msc=tick.time_msc_raw,
                                call_latency_s=t1 - t0))
        if i + 1 < n:
            sleep(interval_s)
    return out
