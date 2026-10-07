"""Phase 1 raw dataset -> normalised ticks -> bars -> features -> ML-READY dataset, with provenance at every stage.

Nothing here touches raw data except to read it (checksums verified against the dataset manifest first). Derived stages
are written under ``<out_root>/features/<instrument>/<ml_dataset_id>/`` and each one records its parents' hashes, schema
version, feature-set version, code version and configuration. No labels are attached: the dataset is features only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fxscalp.core.provenance import canonical_json, sha256_bytes, sha256_file, slug, software_versions
from fxscalp.features.bars import BAR_SCHEMA_VERSION
from fxscalp.features.normalize import TICK_NORM_SCHEMA_VERSION, NormalizationReport, from_arrow, normalize_ticks
from fxscalp.features.pipeline import ML_SCHEMA_VERSION, PIPELINE_VERSION, FeatureFrame, PipelineConfig, build_feature_frame
from fxscalp.features.validation import FeatureQualityReport, render_report_markdown, validate_feature_frame
from fxscalp.market_data.store import ChunkKey, TickStore
from fxscalp.market_data.timebase import TimeBasis


class DatasetError(RuntimeError):
    pass


def find_dataset_manifest(store: TickStore, dataset_id: str) -> dict[str, Any]:
    hits = list((store.root / "datasets").glob(f"*/*/*/{dataset_id}.json"))
    if not hits:
        raise DatasetError(f"dataset {dataset_id!r} not found under {store.root / 'datasets'}")
    return json.loads(hits[0].read_text(encoding="utf-8"))


def load_symbol_meta(store: TickStore, manifest: dict[str, Any]) -> dict[str, Any]:
    d = store.root / "metadata" / "symbol_info" / slug(manifest["broker"]) / slug(manifest["server"]) / manifest["instrument"]
    sha = manifest.get("symbol_info_sha256") or ""
    p = d / f"{sha[:16]}.json"
    if not p.exists():
        raise DatasetError(f"symbol metadata snapshot {p} referenced by the dataset manifest is missing")
    return json.loads(p.read_text(encoding="utf-8"))


def load_normalized_ticks(store: TickStore, manifest: dict[str, Any], *, exclude_quarantined: bool = True,
                          verify: bool = True, max_days: int | None = None) -> tuple[pd.DataFrame, NormalizationReport, dict[str, Any]]:
    """Read + verify every chunk of a Phase 1 dataset and normalise. Hash chain: manifest -> chunk manifest -> file."""
    meta = load_symbol_meta(store, manifest)
    point = (meta.get("typed") or {}).get("point")
    if manifest["time_basis"] == TimeBasis.UNVERIFIED.value:
        raise DatasetError("dataset time_basis is 'unverified': features must not be built from it (calibrate time first)")
    parts: list[dict[str, np.ndarray]] = []
    chunks = manifest["chunks"][:max_days] if max_days else manifest["chunks"]
    for c in chunks:
        key = ChunkKey(manifest["broker"], manifest["server"], manifest["instrument"], date.fromisoformat(c["source_day"]))
        cm = store.chunk_manifest(key)
        if cm is None or cm["raw_content_sha256"] != c["raw_content_sha256"] or cm["file_sha256"] != c["file_sha256"]:
            raise DatasetError(f"chunk {c['source_day']} does not match the dataset manifest (hash chain broken)")
        if c["record_count"] == 0:
            continue
        raw = store.read_raw_chunk(key, verify=verify)
        qual = store.read_derived("tick_quality", key)
        parts.append(from_arrow(raw, qual))
    if not parts:
        raise DatasetError("dataset contains no ticks")
    cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    cols["seq"] = np.arange(len(cols["bid"]), dtype="int64")      # global row id: concatenation order of the verified chunks
    df, rep = normalize_ticks(cols, point=point, exclude_quarantined=exclude_quarantined)
    return df, rep, meta


def content_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for c in df.columns:
        a = df[c].to_numpy()
        h.update(str(c).encode())
        h.update((np.ascontiguousarray(a).astype("<f8").tobytes() if a.dtype.kind in "fiu" else "|".join(map(str, a)).encode()))
    return h.hexdigest()


def _write(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd", compression_level=3)
    return {"file": path.name, "sha256": sha256_file(path), "rows": len(df), "columns": list(df.columns)}


@dataclass
class BuildResult:
    out_dir: Path
    manifest: dict[str, Any]
    frame: FeatureFrame
    quality: FeatureQualityReport


def ml_dataset_id(raw_dataset_id: str, feature_set_version: str, cfg: PipelineConfig, exclude_quarantined: bool) -> str:
    ident = {"raw": raw_dataset_id, "fsv": feature_set_version, "cfg": cfg.config_hash(), "pipeline": PIPELINE_VERSION,
             "schema": ML_SCHEMA_VERSION, "excl_q": exclude_quarantined}
    return "mlds-" + sha256_bytes(canonical_json(ident).encode())[:20]


def build_from_store(store: TickStore, manifest: dict[str, Any], out_root: Path, cfg: PipelineConfig | None = None, *,
                     exclude_quarantined: bool = True, write_bars: bool = True, max_days: int | None = None,
                     force: bool = False) -> BuildResult:
    cfg = cfg or PipelineConfig(instrument=manifest["instrument"])
    ticks, nrep, meta = load_normalized_ticks(store, manifest, exclude_quarantined=exclude_quarantined, max_days=max_days)
    point = (meta.get("typed") or {}).get("point")
    ff = build_feature_frame(ticks, cfg, point=point, collect_bars=write_bars)
    qrep = validate_feature_frame(ff)
    mid = ml_dataset_id(manifest["dataset_id"], ff.registry.version, cfg, exclude_quarantined)
    out = out_root / "features" / cfg.instrument / mid
    feat_hash = content_hash(ff.df)
    if (out / "manifest.json").exists() and not force:
        old = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if old.get("features_content_sha256") == feat_hash:
            return BuildResult(out, old, ff, qrep)
        raise DatasetError(f"{out} exists with different content; pass force=True to rebuild this derived dataset")
    out.mkdir(parents=True, exist_ok=True)
    stages: list[dict[str, Any]] = []
    stages.append({"name": "tick_norm", "schema_version": TICK_NORM_SCHEMA_VERSION, **_write(ticks, out / "tick_norm.parquet"),
                   "parents": [{"kind": "tick_raw_dataset", "dataset_id": manifest["dataset_id"],
                                "manifest_checksum_sha256": manifest["manifest_checksum_sha256"]}],
                   "normalization": nrep.to_json()})
    if write_bars:
        for tf, b in ff.bars.items():
            stages.append({"name": f"bars_{tf}", "schema_version": BAR_SCHEMA_VERSION, **_write(b, out / f"bars_{tf}.parquet"),
                           "parents": [{"kind": "tick_norm", "sha256": stages[0]["sha256"]}]})
    stages.append({"name": "features", "schema_version": ML_SCHEMA_VERSION, **_write(ff.df, out / "features.parquet"),
                   "feature_set_version": ff.registry.version,
                   "parents": [{"kind": "tick_norm", "sha256": stages[0]["sha256"]}]})
    (out / "feature_quality_report.json").write_text(json.dumps(qrep.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")
    (out / "feature_quality_report.md").write_text(render_report_markdown(qrep), encoding="utf-8")
    man = {
        "manifest_type": "ml_dataset", "schema_version": ML_SCHEMA_VERSION, "ml_dataset_id": mid, "instrument": cfg.instrument,
        "labels": None, "note": "features only; no labels, no model, no strategy",
        "raw_dataset": {"dataset_id": manifest["dataset_id"], "manifest_checksum_sha256": manifest["manifest_checksum_sha256"],
                        "record_count": manifest["record_count"], "time_basis": manifest["time_basis"],
                        "timebase_spec_sha256": manifest.get("timebase_spec_sha256"),
                        "symbol_info_sha256": manifest.get("symbol_info_sha256"), "broker": manifest["broker"],
                        "server": manifest["server"], "broker_symbol": manifest["broker_symbol"]},
        "feature_set_name": ff.registry.name, "feature_set_version": ff.registry.version,
        "registry_groups": [{"name": n, "definition_version": v, "code_fingerprint": f} for n, v, f in ff.registry.groups],
        "pipeline_config": cfg.to_dict(), "config_hash": cfg.config_hash(), "pipeline_version": PIPELINE_VERSION,
        "code": software_versions(), "stages": stages, "feature_columns": ff.feature_columns,
        "metadata_columns": ["timestamp_utc_ms", "segment_id", "row_valid", "session", "spread_regime", "volatility_regime"],
        "rows": len(ff.df), "valid_rows": int(ff.df["row_valid"].sum()) if len(ff.df) else 0, "segments": ff.segments,
        "features_content_sha256": feat_hash,
        "regime_calibration": {"spread": cfg.regimes.spread.calibration_status, "volatility": cfg.regimes.volatility.calibration_status},
        "quality": {"ok": qrep.ok, "errors": qrep.errors, "warnings": qrep.warnings},
        "units_note": "spread_points uses the symbol 'point' from the stored metadata snapshot; none is assumed",
    }
    man["manifest_checksum_sha256"] = sha256_bytes(canonical_json(man).encode())
    (out / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return BuildResult(out, man, ff, qrep)


def read_ml_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    man = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    f = next(s for s in man["stages"] if s["name"] == "features")
    if sha256_file(path / f["file"]) != f["sha256"]:
        raise DatasetError("features.parquet checksum does not match its manifest")
    return pd.read_parquet(path / f["file"]), man
