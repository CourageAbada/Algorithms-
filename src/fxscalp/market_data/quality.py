"""Tick data-quality rules. Records are TAGGED, never deleted; severe tags mark a tick quarantined.

Thresholds here are data-quality heuristics (what is *suspicious*), not trading parameters. They are
configuration, recorded in every quality manifest, and are to be re-derived from the data profile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntFlag
from typing import Any

import numpy as np
import pyarrow as pa

from fxscalp.market_data.schema import QUALITY_SCHEMA

KNOWN_TICK_FLAG_BITS = 0x02 | 0x04 | 0x08 | 0x10 | 0x20 | 0x40  # verified constants (BID..SELL)


class Q(IntFlag):
    MISSING_BID = 1
    MISSING_ASK = 2
    NEGATIVE_SPREAD = 4
    ZERO_SPREAD = 8
    EXTREME_SPREAD = 16
    DUP_TIMESTAMP = 32
    DUP_TICK = 64
    TIME_REVERSAL = 128
    LARGE_GAP = 256
    SEC_MSEC_MISMATCH = 512
    NONFINITE = 1024
    TIME_AMBIGUOUS = 2048
    TIME_NONEXISTENT = 4096
    UNKNOWN_FLAG_BITS = 8192       # informational: undocumented/broker-specific bits (e.g. 0x400); NOT corruption
    FUTURE_TICK = 16384            # normalized UTC time later than the ingestion/receive time (+ tolerance)
    TIME_BASIS_UNCERTAIN = 32768   # instant lies outside the validity period of the applicable time calibration


QUARANTINE_MASK = int(Q.MISSING_BID | Q.MISSING_ASK | Q.NEGATIVE_SPREAD | Q.NONFINITE | Q.TIME_REVERSAL
                      | Q.TIME_NONEXISTENT | Q.FUTURE_TICK)
# Deliberately NOT quarantined: DUP_TIMESTAMP (same-millisecond ticks with different content are legitimate and are
# kept in broker order), DUP_TICK, UNKNOWN_FLAG_BITS, TIME_BASIS_UNCERTAIN (kept and tagged; consumers decide).


@dataclass(frozen=True)
class QualityConfig:
    extreme_spread_multiple: float = 20.0   # spread > k * batch median positive spread
    large_gap_s: float = 300.0              # gap between consecutive ticks worth tagging
    long_gap_s: float = 6 * 3600.0          # gaps listed individually in the report
    max_listed_gaps: int = 50
    future_tolerance_s: float = 2.0         # future-tick tolerance; the measured receive latency is ADDED by the caller


@dataclass(frozen=True)
class PrevTick:
    time_msc: int
    bid: float
    ask: float
    last: float
    volume: int
    flags: int


@dataclass
class QualityResult:
    flags: np.ndarray            # uint32
    quarantined: np.ndarray      # bool
    summary: dict[str, Any]
    unknown_bits: np.ndarray | None = None   # uint32: raw flags & ~KNOWN_TICK_FLAG_BITS

    def to_table(self, seq: np.ndarray) -> pa.Table:
        ub = self.unknown_bits if self.unknown_bits is not None else np.zeros(len(self.flags), dtype="uint32")
        return pa.table({"seq": pa.array(seq.astype("int64")), "quality_flags": pa.array(self.flags),
                         "quarantined": pa.array(self.quarantined), "unknown_flag_bits": pa.array(ub)},
                        schema=QUALITY_SCHEMA)


def assess(source_time: np.ndarray, source_time_msc: np.ndarray, bid: np.ndarray, ask: np.ndarray,
           last: np.ndarray, volume: np.ndarray, flags_raw: np.ndarray, *, ambiguous: np.ndarray | None = None,
           nonexistent: np.ndarray | None = None, cfg: QualityConfig = QualityConfig(),
           prev: PrevTick | None = None, uncertain: np.ndarray | None = None,
           normalized_utc_ms: np.ndarray | None = None, ingestion_utc_ms: int | None = None,
           receive_latency_s: float = 0.0) -> QualityResult:
    n = len(bid)
    f = np.zeros(n, dtype="uint32")
    if n == 0:
        return QualityResult(f, np.zeros(0, bool), {"n": 0, "config": asdict(cfg)}, np.zeros(0, dtype="uint32"))
    finite = np.isfinite(bid) & np.isfinite(ask) & np.isfinite(last)
    f[~finite] |= int(Q.NONFINITE)
    with np.errstate(invalid="ignore"):
        f[~(bid > 0)] |= int(Q.MISSING_BID)     # NaN, 0 or negative
        f[~(ask > 0)] |= int(Q.MISSING_ASK)
        spread = ask - bid
        both = (bid > 0) & (ask > 0)
        f[both & (spread < 0)] |= int(Q.NEGATIVE_SPREAD)
        f[both & (spread == 0)] |= int(Q.ZERO_SPREAD)
        pos = spread[both & (spread > 0)]
        med = float(np.median(pos)) if pos.size else float("nan")
        if np.isfinite(med) and med > 0:
            f[both & (spread > cfg.extreme_spread_multiple * med)] |= int(Q.EXTREME_SPREAD)

    # time ordering / duplicates (use `prev` for the first element when provided)
    t = source_time_msc.astype("int64")
    pt = np.concatenate(([prev.time_msc] if prev else [t[0]], t[:-1]))
    has_prev = np.ones(n, bool)
    if not prev:
        has_prev[0] = False
    dt = t - pt
    f[has_prev & (dt == 0)] |= int(Q.DUP_TIMESTAMP)
    f[has_prev & (dt < 0)] |= int(Q.TIME_REVERSAL)
    f[has_prev & (dt > cfg.large_gap_s * 1000)] |= int(Q.LARGE_GAP)
    same = np.zeros(n, bool)
    cols = (bid, ask, last, volume.astype("float64"), flags_raw.astype("float64"))
    prevs = [np.concatenate(([getattr(prev, nm)] if prev else [c[0]], c[:-1]))
             for nm, c in zip(("bid", "ask", "last", "volume", "flags"), cols)]
    same = (dt == 0)
    for c, pc in zip(cols, prevs):
        same &= (c == pc)
    f[has_prev & same] |= int(Q.DUP_TICK)
    f[np.abs(t // 1000 - source_time.astype("int64")) > 1] |= int(Q.SEC_MSEC_MISMATCH)
    unknown = (flags_raw.astype("uint32") & ~np.uint32(KNOWN_TICK_FLAG_BITS)).astype("uint32")
    f[unknown != 0] |= int(Q.UNKNOWN_FLAG_BITS)
    future_tol_ms = int((cfg.future_tolerance_s + max(0.0, receive_latency_s)) * 1000)
    if normalized_utc_ms is not None and ingestion_utc_ms is not None:
        f[np.asarray(normalized_utc_ms, dtype="int64") > ingestion_utc_ms + future_tol_ms] |= int(Q.FUTURE_TICK)
    if uncertain is not None:
        f[uncertain] |= int(Q.TIME_BASIS_UNCERTAIN)
    if ambiguous is not None:
        f[ambiguous] |= int(Q.TIME_AMBIGUOUS)
    if nonexistent is not None:
        f[nonexistent] |= int(Q.TIME_NONEXISTENT)
    quarantined = (f & np.uint32(QUARANTINE_MASK)) != 0

    # summary
    counts = {q.name: int(((f & int(q)) != 0).sum()) for q in Q}
    gap_idx = np.flatnonzero(has_prev & (dt > cfg.long_gap_s * 1000))
    gaps = [{"after_source_time_msc": int(pt[i]), "before_source_time_msc": int(t[i]), "gap_s": float(dt[i] / 1000)}
            for i in gap_idx[: cfg.max_listed_gaps]]
    flag_hist: dict[str, int] = {}
    uniq, cnt = np.unique(flags_raw, return_counts=True)
    for u, c in zip(uniq, cnt):
        flag_hist[f"0x{int(u):x}"] = int(c)
    uflag_hist: dict[str, int] = {}
    uu, uc = np.unique(unknown[unknown != 0], return_counts=True)
    for u, c in zip(uu, uc):
        uflag_hist[f"0x{int(u):x}"] = int(c)
    dup_ts = (f & int(Q.DUP_TIMESTAMP)) != 0
    dup_tick = (f & int(Q.DUP_TICK)) != 0
    summary: dict[str, Any] = {
        "n": n, "config": asdict(cfg), "counts": counts, "quarantined": int(quarantined.sum()),
        # same-millisecond ticks are NOT automatically duplicates: only identical content is DUP_TICK
        "duplicate_timestamp_total": int(dup_ts.sum()),
        "duplicate_timestamp_distinct_content": int((dup_ts & ~dup_tick).sum()),
        "duplicate_tick_identical": int(dup_tick.sum()),
        "unknown_flag_bits_histogram": uflag_hist, "future_tolerance_s_effective": future_tol_ms / 1000.0,
        "clean": int((f == 0).sum()), "tick_flag_histogram": flag_hist,
        "long_gaps": gaps, "n_long_gaps": int(gap_idx.size),
        "first_tick_context": "previous-chunk tick supplied" if prev else
                              "first tick not checked for gap/duplicate/reversal (no previous tick)",
        "median_positive_spread": None if not np.isfinite(med) else med,
    }
    return QualityResult(f, quarantined, summary, unknown)
