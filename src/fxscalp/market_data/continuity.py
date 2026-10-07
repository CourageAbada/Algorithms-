"""Cross-day (chunk boundary) continuity validation of a raw tick dataset.

For every pair of ADJACENT TRADING chunks (empty closed days are skipped) compare the previous chunk's final tick with the
next chunk's first tick: raw and normalised ordering, gap classification against the expected market structure, duplicate
tick / duplicate timestamp across the boundary, and time-basis transitions. Normal daily maintenance breaks and weekend
closures are classified, never reported as missing data; anything that matches no expected pattern is UNEXPECTED.

Classification thresholds (documented conventions of this feed, observed in the profile, not tuned on any model):
  daily_break          55-70 min  (observed: 3,661 s, daily 20:59-22:00 UTC in summer time)
  weekly_closure       47-74 h    (observed: 49.05 h; 73.0 h over the Easter weekend)
  holiday_schedule     any gap > 70 min that STARTS on a known US market holiday (early close; e.g. 12,600 s on 2026-05-25)
  UNEXPECTED           everything else
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import numpy as np

from fxscalp.market_data.quality import Q
from fxscalp.market_data.store import ChunkKey, TickStore

DAILY_BREAK_S = (55 * 60, 70 * 60)
WEEKLY_CLOSURE_S = (47 * 3600, 74 * 3600)
#: US market holidays inside the acquired range (documented schedule knowledge, used only to EXPLAIN early closes).
KNOWN_HOLIDAYS = frozenset({"2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07"})


@dataclass(frozen=True)
class EdgeTick:
    day: str
    source_time_msc: int
    utc_ms: int
    bid: float
    ask: float
    last: float
    volume: int
    flags: int
    time_basis_id: str
    uncertain: bool


@dataclass
class BoundaryResult:
    prev_day: str
    next_day: str
    gap_s: float
    raw_gap_s: float
    kind: str                 # daily_break | weekly_closure | holiday_schedule | UNEXPECTED | ORDER_VIOLATION
    normalized_ordered: bool
    raw_ordered: bool
    duplicate_tick: bool
    duplicate_timestamp: bool
    time_basis_transition: bool
    offset_shift_s: float     # (raw gap - normalised gap): non-zero means the normalisation rule changed across the boundary
    problems: list[str] = field(default_factory=list)


def classify_gap(gap_s: float, prev_utc_ms: int) -> str:
    start_day = datetime.fromtimestamp(prev_utc_ms / 1000, tz=timezone.utc).date().isoformat()
    if DAILY_BREAK_S[0] <= gap_s <= DAILY_BREAK_S[1]:
        return "daily_break"
    if WEEKLY_CLOSURE_S[0] <= gap_s <= WEEKLY_CLOSURE_S[1]:
        return "weekly_closure"
    if gap_s > DAILY_BREAK_S[1] and start_day in KNOWN_HOLIDAYS:
        return "holiday_schedule"
    return "UNEXPECTED"


def check_boundary(prev: EdgeTick, nxt: EdgeTick) -> BoundaryResult:
    gap = (nxt.utc_ms - prev.utc_ms) / 1000.0
    raw_gap = (nxt.source_time_msc - prev.source_time_msc) / 1000.0
    norm_ok, raw_ok = nxt.utc_ms >= prev.utc_ms, nxt.source_time_msc >= prev.source_time_msc
    dup_ts = nxt.source_time_msc == prev.source_time_msc
    dup_tick = dup_ts and (nxt.bid, nxt.ask, nxt.last, nxt.volume, nxt.flags) == (prev.bid, prev.ask, prev.last, prev.volume, prev.flags)
    transition = nxt.time_basis_id != prev.time_basis_id or nxt.uncertain != prev.uncertain
    problems: list[str] = []
    if not norm_ok:
        problems.append("normalised timestamps go backwards across the boundary")
    if not raw_ok:
        problems.append("raw timestamps go backwards across the boundary")
    if dup_tick:
        problems.append("identical tick repeated across the boundary")
    if abs(raw_gap - gap) > 1e-6:
        problems.append(f"normalisation shifts by {raw_gap - gap:.0f}s across the boundary (rule changed)")
    kind = "ORDER_VIOLATION" if not (norm_ok and raw_ok) else classify_gap(gap, prev.utc_ms)
    if kind == "UNEXPECTED":
        problems.append(f"unexplained cross-boundary gap of {gap:.0f}s")
    return BoundaryResult(prev.day, nxt.day, gap, raw_gap, kind, norm_ok, raw_ok, dup_tick, dup_ts, transition, raw_gap - gap, problems)


def _edge(store: TickStore, key: ChunkKey, first: bool) -> EdgeTick:
    raw = store.read_raw_chunk(key, ["source_time_msc", "normalized_utc_time", "bid", "ask", "last", "volume", "flags",
                                     "time_basis_id"], verify=False)
    qual = store.read_derived("tick_quality", key)
    i = 0 if first else raw.num_rows - 1
    flags = int(qual["quality_flags"][i].as_py())
    return EdgeTick(key.source_day.isoformat(), int(raw["source_time_msc"][i].as_py()), int(raw["normalized_utc_time"][i].value),
                    float(raw["bid"][i].as_py()), float(raw["ask"][i].as_py()), float(raw["last"][i].as_py()),
                    int(raw["volume"][i].as_py()), int(raw["flags"][i].as_py()), str(raw["time_basis_id"][i].as_py()),
                    bool(flags & int(Q.TIME_BASIS_UNCERTAIN)))


def check_continuity(store: TickStore, manifest: dict[str, Any]) -> dict[str, Any]:
    trading = [c["source_day"] for c in manifest["chunks"] if c["record_count"] > 0]
    results: list[BoundaryResult] = []
    for a, b in zip(trading, trading[1:]):
        ka = ChunkKey(manifest["broker"], manifest["server"], manifest["instrument"], date.fromisoformat(a))
        kb = ChunkKey(manifest["broker"], manifest["server"], manifest["instrument"], date.fromisoformat(b))
        results.append(check_boundary(_edge(store, ka, first=False), _edge(store, kb, first=True)))
    kinds: dict[str, int] = {}
    for r in results:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    by = lambda k: np.array([r.gap_s for r in results if r.kind == k])  # noqa: E731
    stats = {k: ({"n": int(len(by(k))), "min_s": float(by(k).min()), "median_s": float(np.median(by(k))), "max_s": float(by(k).max())}
                 if len(by(k)) else {"n": 0}) for k in ("daily_break", "weekly_closure", "holiday_schedule")}
    serious = [asdict(r) for r in results if r.problems and (not r.normalized_ordered or not r.raw_ordered or r.duplicate_tick
                                                           or r.kind == "UNEXPECTED")]
    transitions = [asdict(r) for r in results if r.time_basis_transition]
    return {"n_trading_days": len(trading), "n_boundaries": len(results), "kinds": kinds, "gap_stats": stats,
            "ordering_violations": sum(not (r.normalized_ordered and r.raw_ordered) for r in results),
            "duplicate_ticks_across_boundary": sum(r.duplicate_tick for r in results),
            "duplicate_timestamps_across_boundary": sum(r.duplicate_timestamp for r in results),
            "offset_shifts_across_boundary": sum(abs(r.offset_shift_s) > 1e-6 for r in results),
            "time_basis_transitions": transitions,
            "holiday_schedule_boundaries": [asdict(r) for r in results if r.kind == "holiday_schedule"],
            "serious_problems": serious, "ok": not serious}
