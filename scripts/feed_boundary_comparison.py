"""Descriptive pre/post comparison around FEED_REGIME_BOUNDARY (2026-09-06T22:02:04.012Z, source day 2026-09-07).

Groups (chronological, development-visible days only):
  PRE          2026-04-06 .. 2026-09-04   (all eligible pre-boundary days)
  PRE_MATCHED  2026-08-10 .. 2026-09-04   (the 4 weeks just before the boundary)
  POST_DEV     2026-09-07 .. 2026-09-11   (the only post-boundary days inside the development window)
Days from the final holdout are NOT read. (Feed-level properties of ALL post-boundary days were already published in
docs/XAUUSD_REAL_DATASET_PROFILE.md, before the holdout protocol existed.)
This says something about the FEED. It does not show that the economic market regime changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fxscalp.features.dataset import build_from_store, find_dataset_manifest
from fxscalp.features.pipeline import PipelineConfig
from fxscalp.market_data.acquire import build_dataset_manifest
from fxscalp.market_data.store import ChunkKey, TickStore
from fxscalp.market_data.timebase import TimeBase
from fxscalp.research import folds, policy
from fxscalp.workflows import timebase_file

GROUPS = {"PRE": ("2026-04-06", "2026-09-04"), "PRE_MATCHED": ("2026-08-10", "2026-09-04"), "POST_DEV": ("2026-09-07", "2026-09-11")}
FEATURE_DAYS = {"PRE": ["2026-08-12", "2026-08-19", "2026-08-26", "2026-09-02"], "POST_DEV": ["2026-09-08", "2026-09-09", "2026-09-10"]}


def in_group(day: str, g: str) -> bool:
    a, b = GROUPS[g]
    return a <= day <= b


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pre/post feed-regime descriptive comparison (development days only).")
    p.add_argument("--data-root", default="data")
    p.add_argument("--profile-csv", default="data/profiles/XAU_USD_daily_profile.csv")
    p.add_argument("--out", default="research/phase2b/diagnostics/feed_regime_comparison.json")
    a = p.parse_args(argv)
    root = Path(a.data_root)
    store = TickStore(root)
    man = find_dataset_manifest(store, policy.RAW_V1_DATASET_ID)
    df = pd.read_csv(a.profile_csv)
    df = df[(df["n_ticks"] > 0) & (df["day"] >= folds.FIRST_ELIGIBLE_DAY.isoformat()) & (df["day"] <= folds.DEV_LAST_DAY.isoformat())].copy()
    assert df["day"].max() < folds.HOLDOUT_FIRST_DAY.isoformat()
    out: dict[str, Any] = {"schema": "feed_regime_comparison/1", "boundary": policy.policy_manifest()["feed_regime"],
                           "groups": GROUPS, "final_holdout_inspected": False, "note": "describes the feed; no claim about the market", "groups_stats": {}}
    sess_cols = [c for c in df.columns if c.startswith("ticks_") and c != "ticks_market_closed"]
    # ---- quote-change behaviour per day (eligible ticks), then per group
    qc_rows = []
    for _, r in df.iterrows():
        key = ChunkKey(man["broker"], man["server"], man["instrument"], date.fromisoformat(r["day"]))
        raw = store.read_raw_chunk(key, ["bid", "ask"], verify=False)
        el = pq.read_table(root / "derived" / "eligibility" / key.rel_dir() / "eligibility.parquet")["eligible"].to_numpy()
        bid, ask = raw["bid"].to_numpy()[el], raw["ask"].to_numpy()[el]
        db, da = np.diff(bid) != 0, np.diff(ask) != 0
        dmid = np.diff((bid + ask) / 2.0) / 0.01
        n = len(db)
        qc_rows.append({"day": r["day"], "p_bid_changes": db.mean(), "p_ask_changes": da.mean(), "p_both_change": (db & da).mean(),
                        "p_neither_changes": (~db & ~da).mean(), "p_only_one_side_changes": (db ^ da).mean(),
                        "p_mid_changes": (dmid != 0).mean(), "mean_abs_mid_change_points_when_nonzero": float(np.abs(dmid[dmid != 0]).mean()),
                        "ticks_per_mid_change": float(n / max((dmid != 0).sum(), 1))})
    qc = pd.DataFrame(qc_rows).merge(df, on="day")
    for g in GROUPS:
        sub = qc[qc["day"].apply(lambda d: in_group(d, g))]
        if sub.empty:
            continue
        tot = float(sub["n_ticks"].sum())
        flags: dict[str, int] = {}
        for s in sub["unknown_flag_dist"]:
            for k, v in json.loads(s).items():
                flags[k] = flags.get(k, 0) + int(v)
        sess = {c.replace("ticks_", "").upper(): round(100 * float(sub[c].sum()) / tot, 2) for c in sess_cols}
        out["groups_stats"][g] = {
            "days": int(len(sub)), "ticks_per_day": {"median": float(sub["n_ticks"].median()), "mean": float(sub["n_ticks"].mean())},
            "tick_rate_hz_median_of_days": round(float(sub["tick_rate_mean_hz"].median()), 3),
            "spread_points": {"median_of_daily_median": float(sub["spread_median"].median()), "median_of_daily_p95": float(sub["spread_p95"].median()),
                              "median_of_daily_p99": float(sub["spread_p99"].median()), "max_daily_max": float(sub["spread_max"].max())},
            "duplicate_timestamp_rate": round(float(sub["dup_timestamp"].sum()) / tot, 6),
            "zero_spread_ticks": int(sub["zero_spread"].sum()), "extreme_spread_ticks": int(sub["extreme_spread"].sum()),
            "unknown_flag_bits_share": {k: round(v / tot, 4) for k, v in sorted(flags.items())},
            "session_share_pct": sess,
            "quote_change": {c: round(float(sub[c].mean()), 4) for c in ("p_bid_changes", "p_ask_changes", "p_both_change", "p_neither_changes",
                                                                         "p_only_one_side_changes", "p_mid_changes",
                                                                         "mean_abs_mid_change_points_when_nonzero", "ticks_per_mid_change")},
            "range_bps_median": round(float(sub["range_bps"].median()), 1), "ret_1m_std_bps_median": round(float(sub["ret_1m_std_bps"].median()), 3)}
    # ---- feature missingness on matched sample days (Phase 2B pipeline config: 300 s segments, eligibility mask)
    tb = TimeBase.load(timebase_file(root, man["broker"], man["server"]))
    miss: dict[str, dict[str, list[float]]] = {}
    cfg = PipelineConfig(segment_gap_s=policy.SEGMENT_GAP_S)
    for grp, days in FEATURE_DAYS.items():
        for d in days:
            key = ChunkKey(man["broker"], man["server"], man["instrument"], date.fromisoformat(d))
            cm = store.chunk_manifest(key)
            dm = build_dataset_manifest(store, tb.spec, broker=man["broker"], server=man["server"], canonical=man["instrument"],
                                        broker_symbol=cm["broker_symbol"], symbol_info_sha=cm.get("symbol_info_sha256"),
                                        account_fingerprint=cm.get("account_fingerprint"), start_day=date.fromisoformat(d), end_day=date.fromisoformat(d))
            res = build_from_store(store, dm, root / "derived", cfg, write_bars=False, force=True, exclusion_mask=policy.EXCLUDE_MASK)
            f = pd.read_parquet(res.out_dir / "features.parquet")
            cols = res.manifest["feature_columns"]
            valid = f[f["row_valid"] == 1]
            for c in cols:
                miss.setdefault(c, {}).setdefault(grp + "_valid", []).append(float(valid[c].isna().mean() * 100))
                miss[c].setdefault(grp + "_all", []).append(float(f[c].isna().mean() * 100))
            out.setdefault("feature_days", {})[d] = {"group": grp, "rows": len(f), "valid_rows": int(f["row_valid"].sum()),
                                                      "segments": len(res.manifest["segments"]), "validation_ok": res.manifest["quality"]["ok"],
                                                      "warnings": res.manifest["quality"]["warnings"][:4]}
    summ = {}
    for c, d in miss.items():
        pre, post = float(np.mean(d["PRE_valid"])), float(np.mean(d["POST_DEV_valid"]))
        summ[c] = {"pre_nan_pct_valid_rows": round(pre, 3), "post_nan_pct_valid_rows": round(post, 3), "delta_pp": round(post - pre, 3)}
    shifted = {c: v for c, v in summ.items() if abs(v["delta_pp"]) > 1.0}
    out["feature_missingness"] = {"days_pre": FEATURE_DAYS["PRE"], "days_post_dev": FEATURE_DAYS["POST_DEV"], "n_features": len(summ),
                                  "features_with_nan_above_1pct_pre": sum(v["pre_nan_pct_valid_rows"] > 1 for v in summ.values()),
                                  "features_with_nan_above_1pct_post": sum(v["post_nan_pct_valid_rows"] > 1 for v in summ.values()),
                                  "features_with_|delta|>1pp": shifted, "max_abs_delta_pp": max(abs(v["delta_pp"]) for v in summ.values()),
                                  "per_feature": summ}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"wrote {a.out}; groups: {list(out['groups_stats'])}; max |NaN delta| {out['feature_missingness']['max_abs_delta_pp']} pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
