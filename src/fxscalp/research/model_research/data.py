"""Model-research data matrix: features + tb_H60 labels (C0/C1/C2) on the 30 s decision grid, DEVELOPMENT days only.

Per source day (days are separated by gaps far above the 300 s segment rule, so per-day processing equals global processing):
eligible ticks (Phase 2B exclusion mask) -> Phase 2A feature pipeline with ``segment_gap_s=300`` -> 1-second bars -> tb_H60 labels
per segment and cost scenario -> rows kept where ``row_valid == 1`` and the bar end is on the 30 s grid. The final holdout is never
read: ``build_matrix`` asserts every timestamp is in the development window.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxscalp.core.provenance import canonical_json, sha256_bytes, sha256_file
from fxscalp.features.normalize import from_arrow, normalize_ticks
from fxscalp.features.pipeline import PipelineConfig, build_feature_frame
from fxscalp.market_data.store import ChunkKey, TickStore
from fxscalp.research import folds, policy
from fxscalp.research.labels import LabelSpec, generate_labels, grid_from_bars
from fxscalp.research.model_research import protocol as P

MATRIX_VERSION = "model_matrix/1"
LABEL = LabelSpec("tb", 60)
POINT = 0.01
LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}          # SHORT, NO_TRADE, LONG


def matrix_config() -> dict[str, Any]:
    return {"version": MATRIX_VERSION, "label": LABEL.to_dict(), "scenarios": list(P.SCENARIOS), "grid_step_s": P.GRID_STEP_S,
            "pipeline": PipelineConfig(segment_gap_s=policy.SEGMENT_GAP_S).to_dict(), "exclude_mask": policy.EXCLUDE_MASK,
            "raw_dataset": policy.RAW_V1_DATASET_ID, "dtype": "float32"}


def build_day_matrix(store: TickStore, key: ChunkKey, cfg: PipelineConfig | None = None) -> pd.DataFrame:
    cfg = cfg or PipelineConfig(segment_gap_s=policy.SEGMENT_GAP_S)
    raw = store.read_raw_chunk(key, verify=False)
    qual = store.read_derived("tick_quality", key)
    cols = from_arrow(raw, qual)
    cols["quarantined"] = (np.asarray(cols["quality_flags"], dtype="uint32") & np.uint32(policy.EXCLUDE_MASK)) != 0
    cols["seq"] = np.arange(len(cols["bid"]), dtype="int64")
    ticks, _ = normalize_ticks(cols, point=POINT, exclude_quarantined=True)
    ff = build_feature_frame(ticks, cfg, point=POINT, collect_bars=True)
    df = ff.df
    bars = ff.bars["1s"]
    parts: list[pd.DataFrame] = []
    for sid, seg in bars.groupby("segment_id", sort=True):
        g = grid_from_bars(seg.reset_index(drop=True), POINT)
        if g is None:
            continue
        d = {"end_ms": g.end_ms}
        for s in P.SCENARIOS:
            out = generate_labels(g, LABEL, s)
            d[f"y_{s}"] = np.vectorize(LABEL_TO_CLASS.get)(out["label"]).astype("int8")
            d[f"st_{s}"] = out["status"]
            d[f"le_{s}"] = out["label_end_ms"]
        parts.append(pd.DataFrame(d))
    lab = pd.concat(parts, ignore_index=True)
    feat_cols = ff.feature_columns
    m = df.merge(lab, left_on="timestamp_utc_ms", right_on="end_ms", how="inner")
    m = m[(m["row_valid"] == 1) & ((m["timestamp_utc_ms"] // 1000) % P.GRID_STEP_S == 0)]
    cols_out: dict[str, np.ndarray] = {"ts": m["timestamp_utc_ms"].to_numpy("int64"), "segment_id": m["segment_id"].to_numpy("int64")}
    for c in feat_cols:
        cols_out[c] = m[c].to_numpy("float32")
    for s in P.SCENARIOS:
        cols_out[f"y_{s}"], cols_out[f"st_{s}"], cols_out[f"le_{s}"] = m[f"y_{s}"].to_numpy(), m[f"st_{s}"].to_numpy(), m[f"le_{s}"].to_numpy("int64")
    out = pd.DataFrame(cols_out)
    folds.assert_development_only(out["ts"].to_numpy(), what=f"matrix day {key.source_day}")
    return out.reset_index(drop=True)


def build_matrix(store: TickStore, manifest: dict[str, Any], out_dir: Path, *, progress: bool = True) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    days = [c for c in manifest["chunks"] if c["record_count"] > 0
            and folds.FIRST_ELIGIBLE_DAY <= date.fromisoformat(c["source_day"]) <= folds.DEV_LAST_DAY]
    entries = []
    for i, c in enumerate(days):
        key = ChunkKey(manifest["broker"], manifest["server"], manifest["instrument"], date.fromisoformat(c["source_day"]))
        p = out_dir / f"{c['source_day']}.parquet"
        if not p.exists():
            df = build_day_matrix(store, key)
            df.to_parquet(p, index=False)
            n = len(df)
        else:
            n = len(pd.read_parquet(p, columns=["ts"]))
        entries.append({"day": c["source_day"], "rows": n, "sha256": sha256_file(p)})
        if progress and (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(days)} days", flush=True)
    cfg = matrix_config()
    man = {"config": cfg, "days": entries, "rows": int(sum(e["rows"] for e in entries)),
           "matrix_id": "matrix-" + sha256_bytes(canonical_json({"cfg": cfg, "days": [(e["day"], e["sha256"]) for e in entries]}).encode())[:20]}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


def load_matrix(out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    man = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    df = pd.concat([pd.read_parquet(out_dir / f"{e['day']}.parquet") for e in man["days"]], ignore_index=True)
    folds.assert_development_only(df["ts"].to_numpy(), what="loaded matrix")
    return df.sort_values("ts", kind="stable").reset_index(drop=True), man
