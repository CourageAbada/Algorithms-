"""Descriptive profile of the real acquired tick dataset, streaming ONE source day at a time (bounded memory).

  python -m scripts.profile_real_dataset --start 2026-04-01 --end 2026-10-06

Reads the immutable raw/derived/quality chunks (never modifies them) and writes
  <data-root>/profiles/<instrument>_daily_profile.csv   one row per source day
  <data-root>/profiles/<instrument>_profile.json        aggregates, coverage, unusual days
Descriptive only: no labels, no thresholds, no regimes. Session labels use the project's convention windows
(fxscalp.sessions.calendar), i.e. a labelling convention, not a claim about this broker's liquidity.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxscalp.market_data.quality import Q
from fxscalp.market_data.store import ChunkKey, TickStore
from fxscalp.sessions.calendar import classify_sessions

BIN = 0.1                  # spread histogram resolution (points)
NBINS = 30_000             # up to 3000 points; larger values are clipped into the last bin (true max tracked separately)
SESSIONS = ["ASIA", "LONDON", "NEW_YORK", "LONDON_NEW_YORK_OVERLAP", "OFF_HOURS", "MARKET_CLOSED"]
QNAMES = [q.name for q in Q]


def hist_add(h: np.ndarray, values: np.ndarray) -> None:
    v = values[np.isfinite(values) & (values >= 0)]
    if v.size:
        h += np.bincount(np.minimum((v / BIN).astype("int64"), NBINS - 1), minlength=NBINS)


def hist_q(h: np.ndarray, qs: tuple[float, ...]) -> dict[str, float | None]:
    n = int(h.sum())
    if n == 0:
        return {f"p{q:g}": None for q in qs}
    c = np.cumsum(h)
    return {f"p{q:g}": float((np.searchsorted(c, q / 100.0 * n, side="left") + 0.5) * BIN) for q in qs}


def hist_summary(h: np.ndarray) -> dict[str, Any]:
    n = int(h.sum())
    if n == 0:
        return {"n": 0}
    idx = np.flatnonzero(h)
    mean = float((np.arange(NBINS) + 0.5) @ h * BIN / n)
    return {"n": n, "min_approx": float(idx[0] * BIN), "mean": round(mean, 3),
            **{k: (None if v is None else round(v, 2)) for k, v in hist_q(h, (5, 25, 50, 75, 95, 99, 99.9)).items()},
            "max_approx": float((idx[-1] + 1) * BIN)}


def pct(a: np.ndarray, qs: tuple[float, ...]) -> dict[str, float | None]:
    if a.size == 0:
        return {f"p{q:g}": None for q in qs}
    return {f"p{q:g}": float(np.percentile(a, q)) for q in qs}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Streaming descriptive profile of the real tick dataset.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--instrument", default="XAU_USD")
    p.add_argument("--broker", default="Raw Trading Ltd")
    p.add_argument("--server", default="ICMarketsSC-Demo")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args(argv)
    root = Path(a.data_root)
    store = TickStore(root)
    out_dir = Path(a.out_dir) if a.out_dir else root / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)

    h_all = np.zeros(NBINS, dtype="int64")
    h_sess = {s: np.zeros(NBINS, dtype="int64") for s in SESSIONS}
    h_hour = {h: np.zeros(NBINS, dtype="int64") for h in range(24)}
    h_special = {k: np.zeros(NBINS, dtype="int64") for k in ("rollover_pre_20_30_21_00", "rollover_post_22_00_22_30",
                                                           "week_open_first_60min", "week_close_last_60min")}
    rate_hist = np.zeros(400, dtype="int64")                      # ticks per second (within trading windows)
    ticks_hour = np.zeros(24, dtype="int64")
    ticks_sess = {s: 0 for s in SESSIONS}
    days_with_sess = {s: 0 for s in SESSIONS}
    qtot = {n: 0 for n in QNAMES}
    unknown_total: dict[str, int] = {}
    raw_flag_total: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    top_spread: list[tuple[float, str]] = []                      # min-heap of the widest individual spread events
    gaps_global: list[dict[str, Any]] = []
    boundary_gaps: list[dict[str, Any]] = []
    prev_last_ms: int | None = None
    prev_day_has_ticks = False
    first_tick_ms = last_tick_ms = None
    total = uncertain_total = nonuniq = 0
    t_start = time.perf_counter()

    d = start
    while d <= end:
        key = ChunkKey(a.broker, a.server, a.instrument, d)
        man = store.chunk_manifest(key)
        row: dict[str, Any] = {"day": d.isoformat(), "weekday": d.strftime("%a")}
        if man is None:
            row.update({"status": "MISSING_CHUNK", "n_ticks": 0})
            rows.append(row)
            d += timedelta(days=1)
            continue
        n = int(man["record_count"])
        row.update({"status": "ok" if n else "closed_no_ticks", "n_ticks": n})
        if n == 0:
            rows.append(row)
            d += timedelta(days=1)
            continue
        raw = store.read_raw_chunk(key, ["source_time_msc", "normalized_utc_time", "bid", "ask", "flags"])
        qual = store.read_derived("tick_quality", key)
        der = store.read_derived("tick_basic", key)
        t_src = raw["source_time_msc"].to_numpy()
        utc = raw["normalized_utc_time"].cast("int64").to_numpy()
        flags = raw["flags"].to_numpy()
        qf = qual["quality_flags"].to_numpy()
        ub = qual["unknown_flag_bits"].to_numpy()
        quar = qual["quarantined"].to_numpy()
        sp = der["spread_points"].to_numpy()
        mid = der["mid"].to_numpy()
        total += n
        first_ms, last_ms = int(utc[0]), int(utc[-1])
        first_tick_ms = first_ms if first_tick_ms is None else min(first_tick_ms, first_ms)
        last_tick_ms = last_ms if last_tick_ms is None else max(last_tick_ms, last_ms)
        dur = (last_ms - first_ms) / 1000.0
        # ---- ticks / spread / flags
        valid = np.isfinite(sp) & (sp >= 0)
        spv = sp[valid]
        row.update({"first_utc": datetime.fromtimestamp(first_ms / 1000, tz=timezone.utc).isoformat(),
                    "last_utc": datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc).isoformat(),
                    "duration_s": round(dur, 1), "tick_rate_mean_hz": round(n / dur, 3) if dur > 0 else None})
        for k, v in zip(("spread_min", "spread_median", "spread_p95", "spread_p99", "spread_p99_9", "spread_max"),
                        (spv.min(), np.median(spv), *np.percentile(spv, [95, 99, 99.9]), spv.max())):
            row[k] = round(float(v), 3)
        cnt = {nm: int(((qf & int(Q[nm])) != 0).sum()) for nm in QNAMES}
        for nm, v in cnt.items():
            qtot[nm] += v
        row.update({"dup_timestamp": cnt["DUP_TIMESTAMP"], "dup_tick": cnt["DUP_TICK"], "negative_spread": cnt["NEGATIVE_SPREAD"],
                    "zero_spread": cnt["ZERO_SPREAD"], "missing_bid": cnt["MISSING_BID"], "missing_ask": cnt["MISSING_ASK"],
                    "reversals": cnt["TIME_REVERSAL"], "future_ticks": cnt["FUTURE_TICK"],
                    "time_basis_uncertain": cnt["TIME_BASIS_UNCERTAIN"], "extreme_spread": cnt["EXTREME_SPREAD"],
                    "quarantined": int(quar.sum())})
        uncertain_total += cnt["TIME_BASIS_UNCERTAIN"]
        uv, uc = np.unique(ub[ub != 0], return_counts=True)
        udist = {f"0x{int(x):x}": int(c) for x, c in zip(uv, uc)}
        for k, v in udist.items():
            unknown_total[k] = unknown_total.get(k, 0) + v
        fv, fc = np.unique(flags, return_counts=True)
        for x, c in zip(fv, fc):
            raw_flag_total[f"0x{int(x):x}"] = raw_flag_total.get(f"0x{int(x):x}", 0) + int(c)
        row["unknown_flag_dist"] = json.dumps(udist, sort_keys=True)
        # ---- gaps inside the day (the daily break and weekend closure fall on source-day boundaries)
        dt = np.diff(t_src)
        row["largest_gap_s"] = round(float(dt.max()) / 1000.0, 3) if dt.size else 0.0
        for i in np.flatnonzero(dt > 60_000)[:20]:
            gaps_global.append({"after_utc": datetime.fromtimestamp(utc[i] / 1000, tz=timezone.utc).isoformat(),
                                "gap_s": float(dt[i] / 1000.0)})
        # ---- tick rate per second (seconds between first and last tick, quiet seconds included)
        sec = ((utc - first_ms) // 1000).astype("int64")
        per_s = np.bincount(sec)
        rate_hist += np.bincount(np.minimum(per_s, 399), minlength=400)
        row["tick_rate_p50_per_s"] = float(np.percentile(per_s, 50))
        row["tick_rate_p99_per_s"] = float(np.percentile(per_s, 99))
        row["tick_rate_max_per_s"] = int(per_s.max())
        # ---- volatility (descriptive): range and 1-minute mid return dispersion
        minute = utc // 60_000
        last_idx = np.flatnonzero(np.diff(minute, append=minute[-1] + 1))
        m_close = mid[last_idx]
        r = np.diff(np.log(m_close)) * 1e4 if m_close.size > 2 else np.array([])
        row["range_bps"] = round(float((mid.max() - mid.min()) / mid.mean() * 1e4), 2)
        row["ret_1m_std_bps"] = round(float(r.std()), 3) if r.size else None
        # ---- sessions / hours (labels are conventions)
        um, inv = np.unique(minute, return_inverse=True)
        lab = classify_sessions(um * 60_000)[inv]
        for s in SESSIONS:
            m = lab == s
            c = int(m.sum())
            ticks_sess[s] += c
            if c:
                days_with_sess[s] += 1
                hist_add(h_sess[s], sp[m])
            row[f"ticks_{s.lower()}"] = c
        hod = ((utc // 3_600_000) % 24).astype(int)
        ticks_hour += np.bincount(hod, minlength=24)
        for h in np.unique(hod):
            hist_add(h_hour[int(h)], sp[hod == h])
        hist_add(h_all, sp)
        # ---- widest individual spread events
        for i in np.argpartition(sp, -5)[-5:] if sp.size >= 5 else range(sp.size):
            if np.isfinite(sp[i]):
                heapq.heappush(top_spread, (float(sp[i]), datetime.fromtimestamp(utc[i] / 1000, tz=timezone.utc).isoformat()))
                if len(top_spread) > 25:
                    heapq.heappop(top_spread)
        # ---- structure: daily break / week open / week close (crossing days)
        sod = (utc % 86_400_000) / 1000.0
        pre = (sod >= 20.5 * 3600) & (sod < 21 * 3600)
        post = (sod >= 22 * 3600) & (sod < 22.5 * 3600)
        hist_add(h_special["rollover_pre_20_30_21_00"], sp[pre])
        hist_add(h_special["rollover_post_22_00_22_30"], sp[post])
        if prev_last_ms is not None:
            g = (first_ms - prev_last_ms) / 1000.0
            if 600 <= g < 24 * 3600:
                boundary_gaps.append({"kind": "daily_break", "from_utc": datetime.fromtimestamp(prev_last_ms / 1000, tz=timezone.utc).isoformat(),
                                      "to_utc": datetime.fromtimestamp(first_ms / 1000, tz=timezone.utc).isoformat(), "gap_s": g})
            elif g >= 24 * 3600:
                boundary_gaps.append({"kind": "weekly_closure", "from_utc": datetime.fromtimestamp(prev_last_ms / 1000, tz=timezone.utc).isoformat(),
                                      "to_utc": datetime.fromtimestamp(first_ms / 1000, tz=timezone.utc).isoformat(), "gap_s": g})
                m = utc < first_ms + 3600_000
                hist_add(h_special["week_open_first_60min"], sp[m])
                row["week_open"] = True
                row["week_open_ticks_60min"] = int(m.sum())
        elif not prev_day_has_ticks:
            pass
        # a day followed by >24h of nothing is a week close; decided when the next non-empty day arrives (below)
        row["_last_ms"] = last_ms
        row["_sp_last60"] = None
        tail = utc > last_ms - 3600_000
        row["_tail"] = (sp[tail].copy())
        prev_last_ms = last_ms
        prev_day_has_ticks = True
        rows.append(row)
        d += timedelta(days=1)

    # week-close windows: a non-empty day whose next calendar day is closed (Fri -> Sat) and whose next non-empty day is >24h away
    nonempty = [r for r in rows if r.get("n_ticks", 0) > 0]
    for cur, nxt in zip(nonempty, nonempty[1:]):
        gap = (datetime.fromisoformat(nxt["first_utc"]) - datetime.fromisoformat(cur["last_utc"])).total_seconds()
        if gap >= 24 * 3600:
            cur["week_close"] = True
            hist_add(h_special["week_close_last_60min"], cur["_tail"])
            cur["week_close_gap_to_next_open_h"] = round(gap / 3600, 1)
    for r in rows:
        r.pop("_last_ms", None), r.pop("_sp_last60", None), r.pop("_tail", None)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{a.instrument}_daily_profile.csv", index=False)
    td = df[df["n_ticks"] > 0].copy()
    weekday_td = td
    # ---- aggregates
    expected_weekdays = [x for x in pd.date_range(start, end) if x.weekday() < 5]
    missing_weekdays = [x.date().isoformat() for x in expected_weekdays
                        if df.loc[df["day"] == x.date().isoformat(), "n_ticks"].fillna(0).sum() == 0]
    med_ticks = float(td["n_ticks"].median()) if len(td) else 0.0
    low_days = td[td["n_ticks"] < 0.4 * med_ticks][["day", "weekday", "n_ticks"]].to_dict("records")

    def top(col: str, k: int = 5, asc: bool = False, frame: pd.DataFrame = td) -> list[dict[str, Any]]:
        f = frame.dropna(subset=[col]).sort_values(col, ascending=asc).head(k)
        return f[list(dict.fromkeys(["day", "weekday", "n_ticks", col]))].to_dict("records")

    agg: dict[str, Any] = {
        "schema": "xauusd_real_profile/1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "range_requested": [start.isoformat(), end.isoformat()], "n_source_days": int(len(df)),
        "n_trading_days_with_ticks": int(len(td)), "n_closed_days_no_ticks": int((df["n_ticks"] == 0).sum()),
        "n_missing_chunks": int((df["status"] == "MISSING_CHUNK").sum()),
        "total_raw_ticks": int(total),
        "first_raw_tick_utc": None if first_tick_ms is None else datetime.fromtimestamp(first_tick_ms / 1000, tz=timezone.utc).isoformat(),
        "last_raw_tick_utc": None if last_tick_ms is None else datetime.fromtimestamp(last_tick_ms / 1000, tz=timezone.utc).isoformat(),
        "ticks_per_trading_day": {"min": int(td["n_ticks"].min()), "p25": float(td["n_ticks"].quantile(.25)),
                                  "median": med_ticks, "p75": float(td["n_ticks"].quantile(.75)), "max": int(td["n_ticks"].max()),
                                  "mean": float(td["n_ticks"].mean())},
        "ticks_by_utc_hour": {f"{h:02d}": int(c) for h, c in enumerate(ticks_hour)},
        "ticks_by_session": ticks_sess, "days_with_ticks_in_session": days_with_sess,
        "spread_points_overall": hist_summary(h_all),
        "spread_points_by_session": {s: hist_summary(h_sess[s]) for s in SESSIONS},
        "spread_points_by_utc_hour": {f"{h:02d}": hist_summary(h_hour[h]) for h in range(24)},
        "spread_points_special_windows": {k: hist_summary(v) for k, v in h_special.items()},
        "spread_points_by_day_summary": {"median_of_daily_median": float(td["spread_median"].median()),
                                         "min_daily_median": float(td["spread_median"].min()),
                                         "max_daily_median": float(td["spread_median"].max()),
                                         "max_daily_p99": float(td["spread_p99"].max()), "max_daily_max": float(td["spread_max"].max())},
        "tick_rate": {"mean_hz_per_day": pct(td["tick_rate_mean_hz"].to_numpy(), (5, 25, 50, 75, 95)),
                      "per_second_counts_overall": {"p50": int(np.searchsorted(np.cumsum(rate_hist), .5 * rate_hist.sum())),
                                                     "p90": int(np.searchsorted(np.cumsum(rate_hist), .9 * rate_hist.sum())),
                                                     "p99": int(np.searchsorted(np.cumsum(rate_hist), .99 * rate_hist.sum())),
                                                     "max_bin": int(np.flatnonzero(rate_hist)[-1]), "seconds_observed": int(rate_hist.sum())}},
        "duplicate_timestamp_rate": qtot["DUP_TIMESTAMP"] / max(total, 1), "duplicate_tick_rate": qtot["DUP_TICK"] / max(total, 1),
        "quality_tag_counts": qtot, "quarantined_total": int(td["quarantined"].sum()),
        "unknown_flag_bit_distribution": unknown_total, "raw_flag_distribution": raw_flag_total,
        "time_basis": {"ticks_total": int(total), "ticks_verified": int(total - uncertain_total),
                       "ticks_time_basis_uncertain": int(uncertain_total),
                       "pct_verified": round(100.0 * (total - uncertain_total) / max(total, 1), 4),
                       "pct_uncertain": round(100.0 * uncertain_total / max(total, 1), 4)},
        "missing_periods": {"weekdays_with_no_ticks": missing_weekdays, "low_activity_days_lt_40pct_of_median": low_days,
                            "intra_day_gaps_gt_60s_count": len(gaps_global),
                            "largest_intra_day_gaps": sorted(gaps_global, key=lambda g: -g["gap_s"])[:15],
                            "structural_gaps": {"daily_break_count": sum(g["kind"] == "daily_break" for g in boundary_gaps),
                                                "daily_break_gap_s": pct(np.array([g["gap_s"] for g in boundary_gaps if g["kind"] == "daily_break"]), (0, 50, 100)),
                                                "weekly_closure_count": sum(g["kind"] == "weekly_closure" for g in boundary_gaps),
                                                "weekly_closure_gap_h": pct(np.array([g["gap_s"] / 3600 for g in boundary_gaps if g["kind"] == "weekly_closure"]), (0, 50, 100)),
                                                "unusual_breaks": [g for g in boundary_gaps if (g["kind"] == "daily_break" and not (1800 <= g["gap_s"] <= 5400))
                                                                   or (g["kind"] == "weekly_closure" and not (40 * 3600 <= g["gap_s"] <= 60 * 3600))][:20]}},
        "unusual_days": {"highest_range_bps": top("range_bps"), "highest_ret_1m_std_bps": top("ret_1m_std_bps"),
                         "widest_spread_p99": top("spread_p99"), "widest_spread_max": top("spread_max"),
                         "highest_mean_tick_rate": top("tick_rate_mean_hz"),
                         "lowest_mean_tick_rate": top("tick_rate_mean_hz", asc=True, frame=td[td["n_ticks"] >= 0.4 * med_ticks]),
                         "lowest_tick_count_nonempty": top("n_ticks", asc=True), "largest_gap_s": top("largest_gap_s"),
                         "widest_individual_spread_events_points": [{"spread_points": s, "utc": t} for s, t in sorted(top_spread, reverse=True)[:15]]},
        "week_open_days": td.loc[td.get("week_open", pd.Series(False, index=td.index)).fillna(False).astype(bool), "day"].tolist() if "week_open" in td else [],
        "week_close_days": td.loc[td.get("week_close", pd.Series(False, index=td.index)).fillna(False).astype(bool), "day"].tolist() if "week_close" in td else [],
        "profile_seconds": round(time.perf_counter() - t_start, 1),
    }
    (out_dir / f"{a.instrument}_profile.json").write_text(json.dumps(agg, indent=2, default=str), encoding="utf-8")
    print(f"profiled {len(td)} trading days / {len(df)} source days, {total:,} ticks in {agg['profile_seconds']}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
