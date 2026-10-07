"""Bounded Phase 2A feature-pipeline feasibility run on REPRESENTATIVE single days (never the whole history at once).

  python -m scripts.feature_feasibility --profile-csv data/profiles/XAU_USD_daily_profile.csv

Each selected day is a separate one-day dataset: manifest -> normalise -> 119 features -> validation -> prefix-invariance on
the real ticks (normalised UTC axis) -> peak RSS / time. Descriptive/engineering check only: no labels, no regime
thresholds, no model. Days are chosen from the descriptive profile by simple extremes (not by any future return).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from fxscalp.brokers.base import RAW_TICK_DTYPE
from fxscalp.features.dataset import build_from_store
from fxscalp.features.leakage import check_prefix_invariance
from fxscalp.features.normalize import from_raw_array
from fxscalp.features.pipeline import PipelineConfig, build_feature_frame
from fxscalp.market_data.acquire import build_dataset_manifest
from fxscalp.market_data.store import ChunkKey, TickStore
from fxscalp.market_data.timebase import TimeBase
from fxscalp.workflows import timebase_file


def pick_days(df: pd.DataFrame) -> dict[str, str]:
    td = df[df["n_ticks"] > 0].copy()
    med = td["n_ticks"].median()
    used: set[str] = set()
    out: dict[str, str] = {}

    def choose(label: str, frame: pd.DataFrame, col: str, how: str) -> None:
        f = frame[~frame["day"].isin(used)].dropna(subset=[col])
        if f.empty:
            return
        if how == "max":
            row = f.loc[f[col].idxmax()]
        elif how == "min":
            row = f.loc[f[col].idxmin()]
        else:
            row = f.loc[(f[col] - med).abs().idxmin()]
        out[label] = row["day"]
        used.add(row["day"])

    mid_week = td[td["weekday"].isin(["Tue", "Wed", "Thu"])]
    choose("normal_activity", mid_week, "n_ticks", "median")
    choose("high_volatility", td, "range_bps", "max")
    choose("wide_spread", td, "spread_p99", "max")
    choose("widest_single_event_rollover_window", td, "spread_max", "max")
    choose("low_activity", td[td["n_ticks"] >= 20_000], "n_ticks", "min")
    if "week_open" in td:
        wo = td[td["week_open"].fillna(False).astype(bool)]
        choose("week_open", wo, "n_ticks", "median")
    if "week_close" in td:
        wc = td[td["week_close"].fillna(False).astype(bool)]
        choose("week_close", wc, "n_ticks", "median")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Per-day Phase 2A feasibility on representative days.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--profile-csv", required=True)
    p.add_argument("--instrument", default="XAU_USD")
    p.add_argument("--broker", default="Raw Trading Ltd")
    p.add_argument("--server", default="ICMarketsSC-Demo")
    p.add_argument("--days", default=None, help="comma list of days to use instead of the automatic selection")
    p.add_argument("--json-output", default=None)
    a = p.parse_args(argv)
    root = Path(a.data_root)
    store = TickStore(root)
    df = pd.read_csv(a.profile_csv)
    days = ({f"manual_{d}": d for d in a.days.split(",")} if a.days else pick_days(df))
    tb = TimeBase.load(timebase_file(root, a.broker, a.server))
    proc = psutil.Process()
    out_root = root / "derived"
    results: list[dict[str, Any]] = []
    for label, day_s in days.items():
        day = date.fromisoformat(day_s)
        key = ChunkKey(a.broker, a.server, a.instrument, day)
        cm = store.chunk_manifest(key)
        man = build_dataset_manifest(store, tb.spec, broker=a.broker, server=a.server, canonical=a.instrument,
                                     broker_symbol=cm["broker_symbol"], symbol_info_sha=cm.get("symbol_info_sha256"),
                                     account_fingerprint=cm.get("account_fingerprint"), start_day=day, end_day=day)
        gc.collect()
        peak = [proc.memory_info().rss]
        stop = threading.Event()

        def sample() -> None:
            while not stop.wait(0.05):
                peak[0] = max(peak[0], proc.memory_info().rss)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        t0 = time.perf_counter()
        res = build_from_store(store, man, out_root, PipelineConfig(instrument=a.instrument), force=True)
        build_s = time.perf_counter() - t0
        stop.set()
        th.join()
        m = res.manifest
        # prefix invariance on the real ticks of this day (normalised UTC axis, full normalise+features chain)
        t = store.read_raw_chunk(key)
        arr = np.empty(t.num_rows, dtype=RAW_TICK_DTYPE)
        utc_ms = t["normalized_utc_time"].cast("int64").to_numpy()
        arr["time_msc"], arr["time"] = utc_ms, utc_ms // 1000
        for c in ("bid", "ask", "last", "volume", "volume_real", "flags"):
            arr[c] = t[c].to_numpy()
        tdf = pd.DataFrame({n: arr[n] for n in arr.dtype.names})
        tdf.insert(0, "seq", np.arange(len(arr)))

        def compute(frame: pd.DataFrame) -> pd.DataFrame:
            a2 = np.empty(len(frame), dtype=RAW_TICK_DTYPE)
            for n in RAW_TICK_DTYPE.names:
                a2[n] = frame[n].to_numpy()
            ticks, _ = from_raw_array(a2, point=0.01)
            return build_feature_frame(ticks, PipelineConfig(instrument=a.instrument), point=0.01).df

        n = len(tdf)
        t1 = time.perf_counter()
        rep = check_prefix_invariance(compute, tdf, [int(n * f) + 7 for f in (0.2, 0.5, 0.8, 0.97)], ts_col="seq",
                                      key_col="timestamp_utc_ms")
        res_row = {"label": label, "day": day_s, "ticks": int(man["record_count"]), "rows": m["rows"], "valid_rows": m["valid_rows"],
                   "n_features": len(m["feature_columns"]), "segments": len(m["segments"]),
                   "validation_ok": bool(m["quality"]["ok"]), "validation_errors": m["quality"]["errors"][:5],
                   "validation_warnings": m["quality"]["warnings"][:8],
                   "build_seconds": round(build_s, 1), "build_peak_rss_mb": round(peak[0] / 2**20, 1),
                   "prefix_invariant": bool(rep.ok), "prefix_cutoffs": rep.cutoffs_checked, "prefix_rows_compared": rep.rows_compared,
                   "prefix_mismatches": rep.mismatches[:3], "prefix_seconds": round(time.perf_counter() - t1, 1),
                   "rss_after_mb": round(proc.memory_info().rss / 2**20, 1), "ml_dataset_id": m["ml_dataset_id"]}
        results.append(res_row)
        print(f"{label:38} {day_s} ticks={res_row['ticks']:>9,} feats={res_row['n_features']} valid={res_row['validation_ok']} "
              f"prefix={res_row['prefix_invariant']} peak={res_row['build_peak_rss_mb']}MB {res_row['build_seconds']}s", flush=True)
        del res, t, arr, tdf
        gc.collect()
    summary = {"selected_days": days, "results": results,
               "all_119_features": all(r["n_features"] == 119 for r in results),
               "all_validation_ok": all(r["validation_ok"] for r in results),
               "all_prefix_invariant": all(r["prefix_invariant"] for r in results),
               "max_build_peak_rss_mb": max(r["build_peak_rss_mb"] for r in results) if results else None}
    if a.json_output:
        Path(a.json_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))
    return 0 if summary["all_119_features"] and summary["all_validation_ok"] and summary["all_prefix_invariant"] else 1


if __name__ == "__main__":
    sys.exit(main())
