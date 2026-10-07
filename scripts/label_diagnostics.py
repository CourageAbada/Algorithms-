"""Label DIAGNOSTICS on the DEVELOPMENT period only (never the final holdout); no model, no PnL, no selection.

  python -m scripts.label_diagnostics --out research/phase2b/diagnostics/label_diagnostics.json

For every candidate label spec (3 families x 3 horizons) generate labels per eligible segment and report only descriptive
properties: class balance, NO_TRADE share, barrier-hit frequencies, event duration, horizon coverage, and what is lost to
warm-up / segment boundaries / purge; broken down by month, session and feed regime. Cost sensitivity (scenarios C1, C2) is
computed on every 4th development day. Degenerate labels are DOCUMENTED, never tuned away.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fxscalp.features.bars import aggregate_bars
from fxscalp.features.dataset import find_dataset_manifest
from fxscalp.market_data.store import ChunkKey, TickStore
from fxscalp.research import folds, policy
from fxscalp.research.costs import SCENARIOS
from fxscalp.research.eligibility import eligibility_dir
from fxscalp.research.labels import (STATUS_BOUNDARY, STATUS_OK, STATUS_WARMUP, LabelSpec, candidate_specs, generate_labels,
                                     grid_from_bars)
from fxscalp.sessions.calendar import classify_sessions

SESSIONS = ["ASIA", "LONDON", "NEW_YORK", "LONDON_NEW_YORK_OVERLAP", "OFF_HOURS"]
COLS = ["LONG", "SHORT", "NO_TRADE", "ambiguous", "WARMUP", "BOUNDARY"]


def _acc(d: dict, key: tuple, vec: np.ndarray) -> None:
    d[key] = d.get(key, np.zeros(len(vec), dtype="int64")) + vec


def _count(out: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    st, lab = out["status"][mask], out["label"][mask]
    ok = st == STATUS_OK
    return np.array([(ok & (lab == 1)).sum(), (ok & (lab == -1)).sum(), (ok & (lab == 0)).sum(), (ok & out["ambiguous"][mask]).sum(),
                     (st == STATUS_WARMUP).sum(), (st == STATUS_BOUNDARY).sum()], dtype="int64")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Descriptive label diagnostics (development period only).")
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset-id", default=policy.RAW_V1_DATASET_ID)
    p.add_argument("--out", default="research/phase2b/diagnostics/label_diagnostics.json")
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--sensitivity-every", type=int, default=4)
    a = p.parse_args(argv)
    store = TickStore(Path(a.data_root))
    man = find_dataset_manifest(store, a.dataset_id)
    specs = candidate_specs()
    plan: list[tuple[LabelSpec, str]] = []
    hold = folds.FinalHoldout()
    day_list = [c for c in man["chunks"] if c["record_count"] > 0 and folds.FIRST_ELIGIBLE_DAY <= date.fromisoformat(c["source_day"]) <= folds.DEV_LAST_DAY]
    if a.max_days:
        day_list = day_list[:a.max_days]
    cnt: dict[tuple, np.ndarray] = {}
    tb_x: dict[tuple, np.ndarray] = {}
    that: dict[tuple, np.ndarray] = {}
    barrier_sum: dict[tuple, list[float]] = {}
    purge_tot = {f.name: {} for f in folds.default_folds()}
    seg_lengths: list[int] = []
    n_rows = 0
    t0 = time.perf_counter()
    for di, c in enumerate(day_list):
        day = date.fromisoformat(c["source_day"])
        key = ChunkKey(man["broker"], man["server"], man["instrument"], day)
        raw = store.read_raw_chunk(key, ["normalized_utc_time", "bid", "ask"], verify=False)
        el = pq.read_table(eligibility_dir(store.root, key) / "eligibility.parquet")
        eligible = el["eligible"].to_numpy()
        seg_id = el["segment_id"].to_numpy()
        utc = raw["normalized_utc_time"].cast("int64").to_numpy()[eligible]
        folds.assert_development_only(utc, hold, what=f"day {day}")
        bid, ask = raw["bid"].to_numpy()[eligible], raw["ask"].to_numpy()[eligible]
        sid = seg_id[eligible]
        cuts = np.concatenate(([0], np.flatnonzero(np.diff(sid) != 0) + 1, [len(sid)]))
        sensitive = (di % a.sensitivity_every == 0)
        for s0, s1 in zip(cuts[:-1], cuts[1:]):
            if s1 - s0 < 10:
                continue
            bars = aggregate_bars(utc[s0:s1], bid[s0:s1], ask[s0:s1], np.zeros(s1 - s0), 1, data_continues_after=True)
            grid = grid_from_bars(bars, 0.01)
            if grid is None:
                continue
            seg_lengths.append(grid.n)
            obs = grid.end_ms
            n_rows += grid.n
            month = pd.to_datetime(obs, unit="ms", utc=True).strftime("%Y-%m").to_numpy()
            minute = obs // 60_000
            um, inv = np.unique(minute, return_inverse=True)
            sess = classify_sessions(um * 60_000)[inv]
            regime = policy.feed_regime(obs)
            for sp in specs:
                for scn in (["C0_spread_only"] + (["C1_moderate", "C2_pessimistic"] if sensitive else [])):
                    out = generate_labels(grid, sp, scn)
                    k = (sp.name, scn)
                    _acc(cnt, k + ("all", "all"), _count(out, np.ones(grid.n, bool)))
                    for m in np.unique(month):
                        _acc(cnt, k + ("month", m), _count(out, month == m))
                    for s in SESSIONS:
                        mk = sess == s
                        if mk.any():
                            _acc(cnt, k + ("session", s), _count(out, mk))
                    for r in (0, 1):
                        mk = regime == r
                        if mk.any():
                            _acc(cnt, k + ("feed_regime", "post" if r else "pre"), _count(out, mk))
                    if sp.family == "tb":
                        ok = out["status"] == STATUS_OK
                        for nm in ("long_outcome", "short_outcome"):
                            vals = out[nm][ok]
                            _acc(tb_x, k + (nm,), np.array([(vals == 1).sum(), (vals == -1).sum(), (vals == 0).sum()], dtype="int64"))
                        th = out["t_hit"][ok & (out["label"] != 0)]
                        _acc(that, k, np.bincount(th[th >= 0], minlength=sp.horizon_s + 2)[:sp.horizon_s + 2])
                        barrier_sum.setdefault(k, []).append(float(np.nanmean(out["barrier"][ok]) / 0.01) if ok.any() else float("nan"))
            # purge accounting for the LONGEST span (latency 1 s + 300 s horizon): independent of label values
            span = 301
            valid = np.arange(grid.n) + span <= grid.n - 1
            for f in folds.default_folds():
                pc = folds.purge_counts(f, obs[valid], obs[valid] + span * 1000)
                for kk, vv in pc.items():
                    purge_tot[f.name][kk] = purge_tot[f.name].get(kk, 0) + vv
        if (di + 1) % 10 == 0:
            print(f"{di + 1}/{len(day_list)} days {time.perf_counter() - t0:.0f}s", flush=True)

    def table(k: tuple, grp: str) -> dict[str, Any]:
        rows = {}
        for key2, v in cnt.items():
            if key2[:2] == k and key2[2] == grp:
                tot = max(int(v[:3].sum()), 1)
                rows[key2[3]] = {"n_labelled": int(v[:3].sum()), "LONG": int(v[0]), "SHORT": int(v[1]), "NO_TRADE": int(v[2]),
                                 "pct_LONG": round(100 * v[0] / tot, 2), "pct_SHORT": round(100 * v[1] / tot, 2),
                                 "pct_NO_TRADE": round(100 * v[2] / tot, 2), "ambiguous": int(v[3]),
                                 "lost_warmup": int(v[4]), "lost_boundary": int(v[5])}
        return rows

    res: dict[str, Any] = {"schema": "label_diagnostics/1", "period": {"first_day": folds.FIRST_ELIGIBLE_DAY.isoformat(),
                                                                       "last_day": folds.DEV_LAST_DAY.isoformat(),
                                                                       "final_holdout_inspected": False},
                           "n_days": len(day_list), "n_segments": len(seg_lengths), "n_grid_rows": int(n_rows),
                           "segment_length_s": {"min": int(min(seg_lengths)), "median": float(np.median(seg_lengths)), "max": int(max(seg_lengths))},
                           "sensitivity_days": "every %dth development day" % a.sensitivity_every, "labels": {}, "purge_by_fold": purge_tot,
                           "elapsed_s": round(time.perf_counter() - t0, 1)}
    for sp in specs:
        for scn in SCENARIOS:
            k = (sp.name, scn)
            if (*k, "all", "all") not in cnt:
                continue
            allr = table(k, "all")["all"]
            rec = {"spec": sp.to_dict(), "scenario": scn, "overall": allr,
                   "coverage_fraction_labelled": round(allr["n_labelled"] / max(allr["n_labelled"] + allr["lost_warmup"] + allr["lost_boundary"], 1), 5),
                   "by_month": table(k, "month"), "by_session": table(k, "session"), "by_feed_regime": table(k, "feed_regime")}
            if sp.family == "tb":
                for nm in ("long_outcome", "short_outcome"):
                    v = tb_x[k + (nm,)]
                    t = max(int(v.sum()), 1)
                    rec[nm] = {"profit_first_pct": round(100 * v[0] / t, 2), "stop_first_pct": round(100 * v[1] / t, 2), "timeout_pct": round(100 * v[2] / t, 2)}
                h = that[k]
                cum = np.cumsum(h)
                rec["event_duration_s"] = {"median": int(np.searchsorted(cum, 0.5 * cum[-1])) if cum[-1] else None,
                                           "p90": int(np.searchsorted(cum, 0.9 * cum[-1])) if cum[-1] else None, "horizon_s": sp.horizon_s}
                rec["mean_barrier_points"] = round(float(np.nanmean(barrier_sum[k])), 1)
            res["labels"][f"{sp.name}|{scn}"] = rec
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"done in {res['elapsed_s']}s: {len(day_list)} days, {len(seg_lengths)} segments, {n_rows:,} grid rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
