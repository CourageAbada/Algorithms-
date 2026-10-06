"""Tick dataset schemas and the raw -> parquet conversion.

RAW dataset  : exactly what the broker returned (+ explicit time columns required in every dataset).
DERIVED      : mid, spread, spread_points, relative_spread (separate dataset; never overwrites raw).
QUALITY      : flags + quarantine marker per tick (separate dataset; raw rows are never deleted).
All three are keyed by ``seq`` (0-based row order in the raw chunk as received from the broker).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pyarrow as pa

from fxscalp.market_data.timebase import NormalizedTime

RAW_SCHEMA_VERSION = "tick_raw/1"
DERIVED_SCHEMA_VERSION = "tick_derived/1"
QUALITY_SCHEMA_VERSION = "tick_quality/1"

RAW_SCHEMA = pa.schema([
    pa.field("seq", pa.int64()),
    pa.field("source_time", pa.int64()),                 # raw tick.time (s), source domain
    pa.field("source_time_msc", pa.int64()),             # raw tick.time_msc (ms), source domain
    pa.field("normalized_utc_time", pa.timestamp("ms", tz="UTC")),
    pa.field("ingestion_time_utc", pa.timestamp("ms", tz="UTC")),
    pa.field("time_basis", pa.string()),
    pa.field("bid", pa.float64()),
    pa.field("ask", pa.float64()),
    pa.field("last", pa.float64()),
    pa.field("volume", pa.uint64()),
    pa.field("volume_real", pa.float64()),
    pa.field("flags", pa.uint32()),
])
#: Columns that are the broker's own data; their hash identifies the raw content independent of normalisation.
RAW_BROKER_COLUMNS = ("source_time", "source_time_msc", "bid", "ask", "last", "volume", "volume_real", "flags")

DERIVED_SCHEMA = pa.schema([
    pa.field("seq", pa.int64()), pa.field("mid", pa.float64()), pa.field("spread", pa.float64()),
    pa.field("spread_points", pa.float64()), pa.field("relative_spread", pa.float64()),
])
QUALITY_SCHEMA = pa.schema([
    pa.field("seq", pa.int64()), pa.field("quality_flags", pa.uint32()), pa.field("quarantined", pa.bool_()),
])


def raw_table(ticks: np.ndarray, normalized: NormalizedTime, ingestion_utc: datetime, *, seq_start: int = 0) -> pa.Table:
    """Build the RAW table from a RAW_TICK_DTYPE array. ``ticks`` order is preserved exactly."""
    n = len(ticks)
    ing_ms = int(ingestion_utc.astimezone(timezone.utc).timestamp() * 1000)
    cols = {
        "seq": pa.array(np.arange(seq_start, seq_start + n, dtype="int64")),
        "source_time": pa.array(ticks["time"].astype("int64")),
        "source_time_msc": pa.array(ticks["time_msc"].astype("int64")),
        "normalized_utc_time": pa.array(normalized.utc_ms.astype("int64"), type=pa.timestamp("ms", tz="UTC")),
        "ingestion_time_utc": pa.array(np.full(n, ing_ms, dtype="int64"), type=pa.timestamp("ms", tz="UTC")),
        "time_basis": pa.array([normalized.basis.value] * n, type=pa.string()),
        "bid": pa.array(ticks["bid"].astype("float64")),
        "ask": pa.array(ticks["ask"].astype("float64")),
        "last": pa.array(ticks["last"].astype("float64")),
        "volume": pa.array(ticks["volume"].astype("uint64")),
        "volume_real": pa.array(ticks["volume_real"].astype("float64")),
        "flags": pa.array(ticks["flags"].astype("uint32")),
    }
    return pa.table(cols, schema=RAW_SCHEMA)


def derived_table(raw: pa.Table, point: float | None) -> pa.Table:
    """mid / spread / spread_points / relative_spread from RAW bid/ask. Bad inputs yield NaN, never exceptions."""
    bid = raw["bid"].to_numpy(zero_copy_only=False)
    ask = raw["ask"].to_numpy(zero_copy_only=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        mid = (bid + ask) / 2.0
        spread = ask - bid
        sp_pts = spread / point if point and point > 0 else np.full(len(bid), np.nan)
        rel = spread / mid
    return pa.table({"seq": raw["seq"], "mid": pa.array(mid), "spread": pa.array(spread),
                     "spread_points": pa.array(sp_pts), "relative_spread": pa.array(rel)}, schema=DERIVED_SCHEMA)
