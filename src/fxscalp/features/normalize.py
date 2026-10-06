"""Tick normalisation: RAW (immutable) -> NORMALIZED. Raw columns are never modified; normalised ticks carry ``seq``
so every row links back to the raw dataset row it came from.

Units: ``spread`` is in price units, ``spread_points`` = spread / point where ``point`` comes from the stored symbol
metadata. Digits/point/tick size are NEVER assumed: when ``point`` is unknown ``spread_points`` is NaN and the report
says so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

TICK_NORM_SCHEMA_VERSION = "tick_norm/1"
NORM_COLUMNS = ("seq", "timestamp_utc_ms", "bid", "ask", "mid", "spread", "spread_points", "relative_spread",
                "last", "volume", "volume_real", "flags", "quality_flags")


@dataclass(frozen=True)
class NormalizationReport:
    n_input: int
    n_output: int
    n_quarantined_excluded: int
    n_reordered: int
    n_same_timestamp: int
    point: float | None
    point_known: bool
    exclude_quarantined: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def from_arrow(raw: pa.Table, quality: pa.Table | None) -> dict[str, np.ndarray]:
    """Column arrays from a Phase 1 raw chunk table (+ its tick_quality table, joined on seq)."""
    seq = raw["seq"].to_numpy(zero_copy_only=False)
    ts = raw["normalized_utc_time"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    out = {"seq": seq, "timestamp_utc_ms": ts}
    for c in ("bid", "ask", "last", "volume", "volume_real", "flags"):
        out[c] = raw[c].to_numpy(zero_copy_only=False)
    if quality is not None:
        if not np.array_equal(quality["seq"].to_numpy(zero_copy_only=False), seq):
            raise ValueError("quality table seq does not match raw table seq")
        out["quality_flags"] = quality["quality_flags"].to_numpy(zero_copy_only=False)
        out["quarantined"] = quality["quarantined"].to_numpy(zero_copy_only=False)
    return out


def normalize_ticks(cols: dict[str, np.ndarray], *, point: float | None, exclude_quarantined: bool = True,
                    ) -> tuple[pd.DataFrame, NormalizationReport]:
    n = len(cols["bid"])
    quarantined = np.asarray(cols.get("quarantined", np.zeros(n, bool)), dtype=bool)
    qflags = np.asarray(cols.get("quality_flags", np.zeros(n, "uint32")), dtype="uint32")
    keep = ~quarantined if exclude_quarantined else np.ones(n, bool)
    df = pd.DataFrame({
        "seq": np.asarray(cols["seq"], dtype="int64")[keep],
        "timestamp_utc_ms": np.asarray(cols["timestamp_utc_ms"], dtype="int64")[keep],
        "bid": np.asarray(cols["bid"], dtype="float64")[keep], "ask": np.asarray(cols["ask"], dtype="float64")[keep],
        "last": np.asarray(cols["last"], dtype="float64")[keep], "volume": np.asarray(cols["volume"], dtype="uint64")[keep],
        "volume_real": np.asarray(cols["volume_real"], dtype="float64")[keep],
        "flags": np.asarray(cols["flags"], dtype="uint32")[keep], "quality_flags": qflags[keep]})
    n_reordered = 0
    ts = df["timestamp_utc_ms"].to_numpy()
    if len(ts) > 1 and (np.diff(ts) < 0).any():
        n_reordered = int((np.diff(ts) < 0).sum())
        df = df.sort_values(["timestamp_utc_ms", "seq"], kind="stable").reset_index(drop=True)   # explicit, counted
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    known = point is not None and point > 0
    df["spread_points"] = df["spread"] / point if known else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        df["relative_spread"] = df["spread"] / df["mid"]
    ts = df["timestamp_utc_ms"].to_numpy()
    rep = NormalizationReport(n, len(df), int(n - keep.sum()), n_reordered,
                              int((np.diff(ts) == 0).sum()) if len(ts) > 1 else 0, point, bool(known), exclude_quarantined)
    return df[list(NORM_COLUMNS)], rep


def split_segments(ts_ms: np.ndarray, gap_s: float) -> list[tuple[int, int]]:
    """Index ranges [lo, hi) separated by gaps longer than ``gap_s`` (closures/outages). Computation never spans a
    long gap, so a window can not silently bridge a weekend or an outage."""
    ts = np.asarray(ts_ms, dtype="int64")
    if len(ts) == 0:
        return []
    cut = np.flatnonzero(np.diff(ts) > gap_s * 1000) + 1
    edges = np.concatenate(([0], cut, [len(ts)]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def from_raw_array(arr: np.ndarray, *, point: float | None, exclude_quarantined: bool = True,
                   utc_offset_s: int = 0) -> tuple[pd.DataFrame, NormalizationReport]:
    """Normalise a RAW_TICK_DTYPE array (tests / benchmarks / non-Phase-1 sources). Runs the Phase 1 quality rules so
    out-of-order, crossed and invalid ticks are tagged and quarantined exactly as in the stored datasets."""
    from fxscalp.market_data import quality as Qm
    q = Qm.assess(arr["time"], arr["time_msc"], arr["bid"], arr["ask"], arr["last"], arr["volume"], arr["flags"])
    cols = {"seq": np.arange(len(arr), dtype="int64"), "timestamp_utc_ms": arr["time_msc"].astype("int64") - utc_offset_s * 1000,
            "bid": arr["bid"], "ask": arr["ask"], "last": arr["last"], "volume": arr["volume"],
            "volume_real": arr["volume_real"], "flags": arr["flags"], "quality_flags": q.flags, "quarantined": q.quarantined}
    return normalize_ticks(cols, point=point, exclude_quarantined=exclude_quarantined)
