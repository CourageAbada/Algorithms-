"""Frozen Phase 2B policies: raw-dataset identity, eligibility, segmentation and the observed feed-regime boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

import numpy as np

from fxscalp.market_data.quality import QUARANTINE_MASK, Q, QualityConfig

RAW_V1_NAME = "XAUUSD_RAW_V1"
RAW_V1_DATASET_ID = "tickraw-fe5b9137d477c7c866d5"
QUALITY_POLICY_VERSION = "quality_policy/1"
ELIGIBILITY_POLICY_VERSION = "eligibility/1"
SEGMENT_POLICY_VERSION = "segments/1"

#: Ticks carrying ANY of these tags are excluded from the Phase 2B view (the raw data is never modified).
EXCLUDE_TAGS: tuple[Q, ...] = (Q.TIME_BASIS_UNCERTAIN, Q.FUTURE_TICK, Q.TIME_REVERSAL, Q.MISSING_BID, Q.MISSING_ASK,
                               Q.NEGATIVE_SPREAD, Q.NONFINITE, Q.TIME_NONEXISTENT, Q.TIME_AMBIGUOUS)
EXCLUDE_MASK = 0
for _q in EXCLUDE_TAGS:
    EXCLUDE_MASK |= int(_q)
#: Tags that are deliberately KEPT (retained so later experiments can test explicit policies).
RETAIN_TAGS: tuple[Q, ...] = (Q.DUP_TIMESTAMP, Q.DUP_TICK, Q.EXTREME_SPREAD, Q.ZERO_SPREAD, Q.LARGE_GAP, Q.UNKNOWN_FLAG_BITS,
                              Q.SEC_MSEC_MISMATCH)

#: A gap between consecutive ELIGIBLE ticks longer than this starts a new segment. 300 s is the Phase-1 quality
#: LARGE_GAP threshold (fixed before any data was seen). Rationale: the largest ordinary inter-tick gap in 85.6 M ticks
#: is 27 s; every one of the 6 intra-day gaps above 300 s (456 s ... 7,361 s) is unexplained feed silence in the middle of
#: a trading session. Any threshold in [230 s, 450 s] yields the identical segmentation of this dataset, so the choice is
#: not tuned; it was not selected using any model or label result.
SEGMENT_GAP_S = 300

#: First tick whose raw flags carry the undocumented bit 0x400 (and every tick after it does): the observed FEED regime
#: boundary. It is the weekly open of source day 2026-09-07. It is a statement about the FEED, not about the market.
FEED_REGIME_BOUNDARY_UTC_MS = 1788732124012      # 2026-09-06T22:02:04.012Z (verified against the raw chunk)
FEED_REGIME_BOUNDARY_NAME = "FEED_REGIME_BOUNDARY"


def eligibility_mask(quality_flags: np.ndarray) -> np.ndarray:
    return (np.asarray(quality_flags, dtype="uint32") & np.uint32(EXCLUDE_MASK)) == 0


def exclusion_reasons(quality_flags: np.ndarray) -> dict[str, int]:
    qf = np.asarray(quality_flags, dtype="uint32")
    return {q.name: int(((qf & int(q)) != 0).sum()) for q in EXCLUDE_TAGS}


def segment_starts(utc_ms: np.ndarray, gap_s: float = SEGMENT_GAP_S, prev_last_utc_ms: int | None = None) -> np.ndarray:
    """Boolean array: True where a new segment starts (gap to the previous ELIGIBLE tick exceeds ``gap_s``)."""
    t = np.asarray(utc_ms, dtype="int64")
    start = np.zeros(len(t), dtype=bool)
    if len(t) == 0:
        return start
    start[0] = prev_last_utc_ms is None or (t[0] - prev_last_utc_ms) > gap_s * 1000
    start[1:] = np.diff(t) > gap_s * 1000
    return start


def feed_regime(utc_ms: np.ndarray) -> np.ndarray:
    """0 = before FEED_REGIME_BOUNDARY, 1 = at/after it (int8)."""
    return (np.asarray(utc_ms, dtype="int64") >= FEED_REGIME_BOUNDARY_UTC_MS).astype("int8")


def quality_policy_fingerprint() -> str:
    body = {"version": QUALITY_POLICY_VERSION, "tags": {q.name: int(q) for q in Q}, "quarantine_mask": int(QUARANTINE_MASK),
            "config_defaults": asdict(QualityConfig())}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def policy_manifest() -> dict[str, Any]:
    return {
        "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION, "quality_policy_version": QUALITY_POLICY_VERSION,
        "quality_policy_sha256": quality_policy_fingerprint(),
        "exclude_tags": sorted(q.name for q in EXCLUDE_TAGS), "exclude_mask": EXCLUDE_MASK,
        "retained_tags": sorted(q.name for q in RETAIN_TAGS),
        "raw_data_modified": False,
        "segment_policy": {"version": SEGMENT_POLICY_VERSION, "gap_s": SEGMENT_GAP_S, "basis": "gap between consecutive ELIGIBLE ticks (normalized UTC)",
                           "applies_to": ["rolling features", "returns/targets", "labels", "events", "trade simulation", "sequences"],
                           "no_bridging": True, "segment_id": "UTC ms of the first eligible tick of the segment",
                           "rationale": "300 s = Phase-1 LARGE_GAP threshold fixed before data was seen; largest ordinary gap 27 s; any threshold in [230,450] s gives the identical segmentation; not tuned on any model/label result"},
        "feed_regime": {"name": FEED_REGIME_BOUNDARY_NAME, "boundary_utc_ms": FEED_REGIME_BOUNDARY_UTC_MS,
                        "boundary_utc": "2026-09-06T22:02:04.012Z", "source_day_label": "2026-09-07",
                        "evidence": "raw tick flag bit 0x400 absent on every earlier tick and present on every tick from this instant",
                        "claim": "observed FEED change only; makes NO claim that the economic market regime changed",
                        "forbidden_as_features": ["flags", "unknown_flag_bits", "quality_flags"]},
    }
