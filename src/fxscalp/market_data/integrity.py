"""Dataset integrity verification as a pure, testable function (used by scripts.verify_dataset_integrity)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import pyarrow.parquet as pq

from fxscalp.core.provenance import canonical_json, sha256_bytes, sha256_file
from fxscalp.market_data.acquire import day_window_s
from fxscalp.market_data.schema import DERIVED_SCHEMA, QUALITY_SCHEMA, RAW_SCHEMA
from fxscalp.market_data.store import ChunkKey, TickStore, compute_dataset_id


def verify_dataset_integrity(store: TickStore, man: dict[str, Any]) -> dict[str, Any]:
    root = store.root
    problems: list[str] = []
    info: dict[str, Any] = {"dataset_id": man["dataset_id"], "chunks_in_manifest": len(man["chunks"]),
                            "record_count_manifest": man["record_count"]}

    # ---- manifest-level
    body = {k: v for k, v in man.items() if k not in ("dataset_id", "manifest_checksum_sha256")}
    did = compute_dataset_id(man)
    if did != man["dataset_id"]:
        problems.append(f"dataset id not reproducible: recomputed {did}")
    doc = {**body, "dataset_id": man["dataset_id"]}
    if sha256_bytes(canonical_json(doc).encode()) != man["manifest_checksum_sha256"]:
        problems.append("dataset manifest checksum mismatch")
    days = [c["source_day"] for c in man["chunks"]]
    if len(set(days)) != len(days):
        problems.append("duplicate source days in the dataset manifest")
    d0, d1 = date.fromisoformat(days[0]), date.fromisoformat(days[-1])
    expected = [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]
    if days != expected:
        problems.append(f"manifest days are not contiguous/ordered: missing {sorted(set(expected) - set(days))[:10]}")

    # ---- per chunk
    total_rows = 0
    prev_last_ms: int | None = None
    seen_raw_hash: dict[str, str] = {}
    bytes_raw = bytes_basic = bytes_qual = 0
    n_nonempty = 0
    prev_window_end: int | None = None
    for c in man["chunks"]:
        day = date.fromisoformat(c["source_day"])
        key = ChunkKey(man["broker"], man["server"], man["instrument"], day)
        m = store.chunk_manifest(key)
        tag = c["source_day"]
        if m is None:
            problems.append(f"{tag}: raw manifest missing")
            continue
        ok, why = store.verify_chunk(key)
        if not ok:
            problems.append(f"{tag}: verify_chunk failed ({why})")
            continue
        if m["source_day"] != tag or m["raw_content_sha256"] != c["raw_content_sha256"] or m["file_sha256"] != c["file_sha256"]:
            problems.append(f"{tag}: chunk manifest does not match the dataset manifest")
        if m["status"] != "verified":
            problems.append(f"{tag}: status {m['status']}")
        s, e = day_window_s(day)
        if m.get("requested_window_source_s") != [s, e]:
            problems.append(f"{tag}: requested window {m.get('requested_window_source_s')} != {[s, e]}")
        mw = m.get("requested_window_source_s") or [s, e]
        if prev_window_end is not None and mw[0] != prev_window_end:
            problems.append(f"{tag}: declared window does not start where the previous day's ended (overlap/gap)")
        prev_window_end = mw[1]
        rawf = store.raw_dir(key) / "ticks.parquet"
        pf = pq.ParquetFile(rawf)
        if pf.schema_arrow != RAW_SCHEMA:
            problems.append(f"{tag}: raw parquet schema differs from RAW_SCHEMA")
        n = int(m["record_count"])
        total_rows += n
        bytes_raw += rawf.stat().st_size
        if n:
            n_nonempty += 1
            h = m["raw_content_sha256"]
            if h in seen_raw_hash:
                problems.append(f"{tag}: raw content identical to {seen_raw_hash[h]} (duplicate chunk content)")
            seen_raw_hash[h] = tag
            t = pq.read_table(rawf, columns=["source_time_msc", "seq"])
            ts = t["source_time_msc"].to_numpy()
            if ts[0] < s * 1000 or ts[-1] >= e * 1000:
                problems.append(f"{tag}: ticks outside the day's source window (collision with a neighbour)")
            if (ts[1:] < ts[:-1]).any() and m.get("clipped_rows_outside_window") == 0:
                pass  # broker order is preserved verbatim; reversals are tagged by quality, not an integrity failure
            if t["seq"].to_numpy()[0] != 0 or t["seq"].to_numpy()[-1] != n - 1:
                problems.append(f"{tag}: seq is not 0..n-1")
            if prev_last_ms is not None and ts[0] <= prev_last_ms:
                problems.append(f"{tag}: first tick not after the previous day's last tick")
            prev_last_ms = int(ts[-1])
        for kind, schema in (("tick_basic", DERIVED_SCHEMA), ("tick_quality", QUALITY_SCHEMA)):
            dm_path = store.derived_dir(kind, key) / "manifest.json"
            pqf = store.derived_dir(kind, key) / f"{kind}.parquet"
            if not dm_path.exists() or not pqf.exists():
                problems.append(f"{tag}: {kind} artifact missing")
                continue
            dm = json.loads(dm_path.read_text(encoding="utf-8"))
            if sha256_file(pqf) != dm["file_sha256"]:
                problems.append(f"{tag}: {kind} checksum mismatch")
            dp = pq.ParquetFile(pqf)
            if dp.schema_arrow != schema or dp.metadata.num_rows != n or dm["record_count"] != n:
                problems.append(f"{tag}: {kind} schema/row-count mismatch")
            if dm["parents"][0]["raw_content_sha256"] != m["raw_content_sha256"]:
                problems.append(f"{tag}: {kind} parent hash does not match the raw chunk")
            if kind == "tick_basic":
                bytes_basic += pqf.stat().st_size
            else:
                bytes_qual += pqf.stat().st_size
    if total_rows != man["record_count"]:
        problems.append(f"sum of chunk rows {total_rows} != dataset record_count {man['record_count']}")

    # ---- strays: temp files, chunk directories outside the manifest range
    strays = [str(x.relative_to(root)) for sub in ("raw", "derived") for x in (root / sub).rglob("*.tmp")]
    if strays:
        problems.append(f"leftover temp files: {strays[:5]}")
    inside = set(days)
    outside = []
    base = root / "raw" / "ticks"
    for mp in base.rglob("manifest.json"):
        try:
            sd = json.loads(mp.read_text(encoding="utf-8")).get("source_day")
        except (OSError, ValueError):
            problems.append(f"unreadable manifest {mp}")
            continue
        if sd not in inside and json.loads(mp.read_text(encoding="utf-8")).get("instrument") == man["instrument"]:
            outside.append(sd)
    info.update({"chunks_outside_dataset_range": sorted(outside), "nonempty_chunks": n_nonempty,
                 "total_rows_verified": total_rows, "parquet_bytes": {"raw": bytes_raw, "tick_basic": bytes_basic,
                                                                    "tick_quality": bytes_qual,
                                                                    "total": bytes_raw + bytes_basic + bytes_qual},
                 "problems": problems, "ok": not problems})
    return info
