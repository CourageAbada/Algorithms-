"""Deterministic Phase 2B eligibility view over an immutable raw dataset (raw data is never modified).

Per source day it writes ``<root>/derived/eligibility/<same hierarchy>/eligibility.parquet`` with, per raw tick (row = raw ``seq``):
  eligible            bool    no EXCLUDE tag (TIME_BASIS_UNCERTAIN, FUTURE_TICK, TIME_REVERSAL, MISSING_BID/ASK, NEGATIVE_SPREAD, ...)
  exclusion_flags     uint32  the exclusion tags that apply (quality_flags & EXCLUDE_MASK)
  segment_id          int64   UTC ms of the first eligible tick of the tick's segment (-1 when not eligible)
  feed_regime         int8    0 before / 1 at-or-after FEED_REGIME_BOUNDARY
Retained quality tags (DUP_TIMESTAMP, EXTREME_SPREAD, ZERO_SPREAD, ...) stay in tick_quality for later policy experiments.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from fxscalp.core.provenance import canonical_json, sha256_bytes, sha256_file
from fxscalp.market_data.store import ChunkKey, TickStore, _atomic_write_parquet
from fxscalp.research import policy

ELIGIBILITY_SCHEMA = pa.schema([pa.field("seq", pa.int64()), pa.field("eligible", pa.bool_()),
                                pa.field("exclusion_flags", pa.uint32()), pa.field("segment_id", pa.int64()),
                                pa.field("feed_regime", pa.int8())])


def day_view(utc_ms: np.ndarray, quality_flags: np.ndarray, prev_last_eligible_ms: int | None) -> dict[str, np.ndarray]:
    """Pure per-day computation (also used by the tests)."""
    utc = np.asarray(utc_ms, dtype="int64")
    qf = np.asarray(quality_flags, dtype="uint32")
    elig = policy.eligibility_mask(qf)
    seg = np.full(len(utc), -1, dtype="int64")
    ue = utc[elig]
    if len(ue):
        starts = policy.segment_starts(ue, policy.SEGMENT_GAP_S, prev_last_eligible_ms)
        ids = np.where(starts, ue, 0)
        ids = np.maximum.accumulate(ids)
        seg[elig] = ids                      # 0 only if the first eligible tick continues the previous day's segment (carried by caller)
    return {"eligible": elig, "exclusion_flags": (qf & np.uint32(policy.EXCLUDE_MASK)).astype("uint32"), "segment_id": seg,
            "feed_regime": policy.feed_regime(utc), "segment_start": policy.segment_starts(ue, policy.SEGMENT_GAP_S, prev_last_eligible_ms) if len(ue) else np.zeros(0, bool)}


def eligibility_dir(root: Path, key: ChunkKey) -> Path:
    return root / "derived" / "eligibility" / key.rel_dir()


def build_eligibility_view(store: TickStore, manifest: dict[str, Any]) -> dict[str, Any]:
    prev_last: int | None = None
    carry_seg: int | None = None
    days: list[dict[str, Any]] = []
    tot = {"ticks": 0, "eligible": 0, "excluded": 0, "segments": 0}
    reasons = {q.name: 0 for q in policy.EXCLUDE_TAGS}
    for c in manifest["chunks"]:
        day = date.fromisoformat(c["source_day"])
        key = ChunkKey(manifest["broker"], manifest["server"], manifest["instrument"], day)
        n = int(c["record_count"])
        rec: dict[str, Any] = {"day": c["source_day"], "ticks": n}
        if n == 0:
            days.append({**rec, "eligible": 0, "excluded": 0, "segments": 0, "sha256": None})
            continue
        raw = store.read_raw_chunk(key, ["normalized_utc_time"], verify=False)
        qual = store.read_derived("tick_quality", key)
        utc = raw["normalized_utc_time"].cast("int64").to_numpy()
        qf = qual["quality_flags"].to_numpy()
        v = day_view(utc, qf, prev_last)
        seg = v["segment_id"].copy()
        if carry_seg is not None and v["eligible"].any() and not v["segment_start"][0]:
            seg[v["eligible"] & (seg == 0)] = carry_seg                      # the segment continues from the previous day
        ue = utc[v["eligible"]]
        n_el = int(v["eligible"].sum())
        for q in policy.EXCLUDE_TAGS:
            reasons[q.name] += int(((qf & np.uint32(int(q))) != 0).sum())
        tbl = pa.table({"seq": pa.array(np.arange(n, dtype="int64")), "eligible": pa.array(v["eligible"]),
                        "exclusion_flags": pa.array(v["exclusion_flags"]), "segment_id": pa.array(seg),
                        "feed_regime": pa.array(v["feed_regime"])}, schema=ELIGIBILITY_SCHEMA)
        d = eligibility_dir(store.root, key)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "eligibility.parquet"
        import os
        import stat
        if p.exists():
            os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
        _atomic_write_parquet(tbl, p)
        n_seg = int(v["segment_start"].sum())
        days.append({**rec, "eligible": n_el, "excluded": n - n_el, "segments": n_seg, "sha256": sha256_file(p)})
        tot["ticks"] += n
        tot["eligible"] += n_el
        tot["excluded"] += n - n_el
        tot["segments"] += n_seg
        if n_el:
            prev_last = int(ue[-1])
            carry_seg = int(seg[v["eligible"]][-1])
    view = {"manifest_type": "eligibility_view", "policy": policy.ELIGIBILITY_POLICY_VERSION,
            "exclude_mask": policy.EXCLUDE_MASK, "exclude_tags": sorted(q.name for q in policy.EXCLUDE_TAGS),
            "segment_gap_s": policy.SEGMENT_GAP_S, "raw_dataset_id": manifest["dataset_id"], "totals": tot,
            "exclusion_tag_counts_non_exclusive": reasons, "days": days}
    view["eligibility_view_id"] = "elig-" + sha256_bytes(canonical_json(
        {"raw": manifest["dataset_id"], "policy": policy.ELIGIBILITY_POLICY_VERSION, "mask": policy.EXCLUDE_MASK,
         "gap": policy.SEGMENT_GAP_S, "days": [(d["day"], d["eligible"], d["sha256"]) for d in days]}).encode())[:20]
    out = store.root / "derived" / "eligibility" / manifest["dataset_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(view, indent=2, sort_keys=True), encoding="utf-8")
    return view
