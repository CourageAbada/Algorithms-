"""XAUUSD_RAW_V1 freeze manifest: the immutable research source, referenced by checksums (no data is duplicated)."""

from __future__ import annotations

from datetime import date
from typing import Any

from fxscalp.core.provenance import canonical_json, sha256_bytes
from fxscalp.market_data.store import ChunkKey, TickStore
from fxscalp.market_data.timebase import TimeBaseSpec, build_calibration_record
from fxscalp.research import policy

FREEZE_VERSION = "freeze/1"


def build_freeze_manifest(store: TickStore, manifest: dict[str, Any], *, feature_set_ref: str, code_commit: str,
                          created_at_utc: str) -> dict[str, Any]:
    spec = TimeBaseSpec.from_json(manifest["timebase_spec"])
    rec = build_calibration_record(spec)
    chunks, first_src = [], None
    last_src = first_utc = last_utc = None
    n_trading = 0
    for c in manifest["chunks"]:
        key = ChunkKey(manifest["broker"], manifest["server"], manifest["instrument"], date.fromisoformat(c["source_day"]))
        cm = store.chunk_manifest(key)
        if cm is None or cm["raw_content_sha256"] != c["raw_content_sha256"] or cm["file_sha256"] != c["file_sha256"]:
            raise ValueError(f"chunk {c['source_day']} does not match the dataset manifest; refusing to freeze")
        entry = {"source_day": c["source_day"], "record_count": c["record_count"], "raw_content_sha256": c["raw_content_sha256"],
                 "file_sha256": c["file_sha256"]}
        for kind in ("tick_basic", "tick_quality"):
            dm = store.derived_dir(kind, key) / "manifest.json"
            import json
            entry[f"{kind}_file_sha256"] = json.loads(dm.read_text(encoding="utf-8"))["file_sha256"]
        chunks.append(entry)
        if cm["record_count"]:
            n_trading += 1
            if first_src is None:
                first_src, first_utc = cm["first_source_time_msc"], cm["first_normalized_utc_ms"]
            last_src, last_utc = cm["last_source_time_msc"], cm["last_normalized_utc_ms"]
    body = {
        "freeze_name": policy.RAW_V1_NAME, "freeze_version": FREEZE_VERSION, "dataset_id": manifest["dataset_id"],
        "dataset_manifest_checksum_sha256": manifest["manifest_checksum_sha256"],
        "broker": manifest["broker"], "server": manifest["server"], "broker_symbol": manifest["broker_symbol"],
        "instrument": manifest["instrument"], "feed": "MT5 DEMO tick feed (quote-only: last and volume are always 0)",
        "first_source_time_msc": first_src, "last_source_time_msc": last_src,
        "first_normalized_utc_ms": first_utc, "last_normalized_utc_ms": last_utc,
        "tick_count": manifest["record_count"], "source_day_count": len(manifest["chunks"]), "trading_day_count": n_trading,
        "schema_version": manifest["schema_version"], "quality_policy_version": policy.QUALITY_POLICY_VERSION,
        "quality_policy_sha256": policy.quality_policy_fingerprint(),
        "time_calibration": {"record_id": rec["record_id"], "basis_id": spec.basis_id(), "offset_s": rec["offset_s"],
                             "effective_from_utc": rec["effective_from_utc"], "effective_to_utc": rec["effective_to_utc"],
                             "dst_status": rec["dst_status"], "confidence": rec["confidence"]},
        "feature_set_reference": feature_set_ref,
        "symbol_info_sha256": manifest.get("symbol_info_sha256"),
        "chunks": chunks,
        "storage": "immutable chunks are REFERENCED by checksum under data/raw|derived; nothing is copied",
    }
    freeze_id = "freeze-" + sha256_bytes(canonical_json(body).encode())[:20]
    return {**body, "freeze_id": freeze_id, "created_at_utc": created_at_utc, "code_commit": code_commit}
