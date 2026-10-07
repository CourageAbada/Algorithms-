"""Generate the Phase 2B-A frozen artifacts: eligibility view, XAUUSD_RAW_V1 freeze manifest, machine-readable specification.

  python -m scripts.build_phase2b_artifacts            # writes research/phase2b/*.json (commit them) and data/derived/eligibility

Deterministic: re-running on the same data and code reproduces identical files (except ``created_at_utc``/``code_commit`` inside the
freeze manifest, which are excluded from its ``freeze_id``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from fxscalp.features.dataset import find_dataset_manifest
from fxscalp.market_data.store import ChunkKey, TickStore
from fxscalp.research import folds, policy, spec
from fxscalp.research.eligibility import build_eligibility_view, eligibility_dir
from fxscalp.research.feature_policy import build_feature_manifest
from fxscalp.research.freeze import build_freeze_manifest


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the frozen Phase 2B-A artifacts.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default=str(spec.SPEC_DIR))
    p.add_argument("--created-at", default=None, help="override creation timestamp (tests)")
    a = p.parse_args(argv)
    store = TickStore(Path(a.data_root))
    man = find_dataset_manifest(store, policy.RAW_V1_DATASET_ID)
    view = build_eligibility_view(store, man)
    fmf = build_feature_manifest()
    freeze = build_freeze_manifest(store, man, feature_set_ref=fmf["feature_set_version"], code_commit=git("rev-parse", "HEAD"),
                                   created_at_utc=a.created_at or datetime.now(timezone.utc).isoformat())

    # ---- segment boundaries inside trading sessions (unexplained feed silence) with their gap sizes
    boundaries: list[dict[str, Any]] = []
    for d in view["days"]:
        if d["segments"] <= 1:
            continue
        key = ChunkKey(man["broker"], man["server"], man["instrument"], date.fromisoformat(d["day"]))
        el = pq.read_table(eligibility_dir(store.root, key) / "eligibility.parquet")
        raw = store.read_raw_chunk(key, ["normalized_utc_time"], verify=False)
        utc = raw["normalized_utc_time"].cast("int64").to_numpy()[el["eligible"].to_numpy()]
        sid = el["segment_id"].to_numpy()[el["eligible"].to_numpy()]
        for i in np.flatnonzero(np.diff(sid) != 0) + 1:
            boundaries.append({"day": d["day"], "gap_start_utc": datetime.fromtimestamp(utc[i - 1] / 1000, tz=timezone.utc).isoformat(),
                               "gap_end_utc": datetime.fromtimestamp(utc[i] / 1000, tz=timezone.utc).isoformat(),
                               "gap_s": round(float(utc[i] - utc[i - 1]) / 1000.0, 3)})
    by_day = {d["day"]: d for d in view["days"]}

    def span(first: date, last: date) -> dict[str, int]:
        ds = [v for k, v in by_day.items() if first <= date.fromisoformat(k) <= last and v["ticks"] > 0]
        return {"trading_days": len(ds), "eligible_ticks": int(sum(v["eligible"] for v in ds)), "segments": int(sum(v["segments"] for v in ds))}

    fold_counts = {}
    for f in folds.default_folds():
        fold_counts[f.name] = {"train": span(date.fromisoformat(f.train_first_day), date.fromisoformat(f.train_last_day)),
                               "validation": span(date.fromisoformat(f.val_first_day), date.fromisoformat(f.val_last_day))}
    summary = {
        "eligibility_view_id": view["eligibility_view_id"], "raw_dataset_id": view["raw_dataset_id"],
        "policy": policy.policy_manifest()["exclude_tags"], "totals": view["totals"],
        "ticks_total": view["totals"]["ticks"] + 0, "ticks_eligible": view["totals"]["eligible"], "ticks_excluded": view["totals"]["excluded"],
        "pct_eligible": round(100.0 * view["totals"]["eligible"] / view["totals"]["ticks"], 4),
        "exclusion_tag_counts_non_exclusive": view["exclusion_tag_counts_non_exclusive"],
        "excluded_days": [{"day": d["day"], "ticks": d["ticks"], "excluded": d["excluded"]} for d in view["days"] if d["excluded"]],
        "segments": {"total": view["totals"]["segments"], "gap_s": policy.SEGMENT_GAP_S,
                     "intra_day_boundaries": boundaries},
        "development": span(folds.FIRST_ELIGIBLE_DAY, folds.DEV_LAST_DAY), "final_holdout_counts_only": span(folds.HOLDOUT_FIRST_DAY, folds.HOLDOUT_LAST_DAY),
        "fold_counts": fold_counts,
        "retained_tag_counts_note": "DUP_TIMESTAMP, EXTREME_SPREAD, ZERO_SPREAD etc. are retained and counted in tick_quality, see the dataset profile",
    }
    idx = spec.write_spec(Path(a.out_dir), {"XAUUSD_RAW_V1.freeze.json": freeze, "eligibility_summary.json": summary})
    print(f"freeze_id {freeze['freeze_id']} code_commit {freeze['code_commit'][:7]}")
    print(f"eligibility_view_id {view['eligibility_view_id']}: eligible {summary['ticks_eligible']:,} of {summary['ticks_total']:,} "
          f"({summary['pct_eligible']}%), excluded {summary['ticks_excluded']:,}, segments {summary['segments']['total']}")
    print(f"spec_hash {idx['spec_hash']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
