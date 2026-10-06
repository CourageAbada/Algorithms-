"""Immutable raw tick store (Parquet) with per-chunk manifests and dataset manifests.

Layout (hive-style; one directory per SOURCE-TIME day = the acquisition chunk):

  <root>/raw/ticks/broker=<slug>/server=<slug>/instrument=<CANONICAL>/year=YYYY/month=MM/day=DD/
        ticks.parquet          RAW dataset (broker fields + explicit time columns). Read-only after write.
        manifest.json          chunk manifest (identity, counts, checksums, time basis, provenance)
  <root>/derived/<kind>/<same hierarchy>/<kind>.parquet + manifest.json       kind = tick_basic | tick_quality
  <root>/datasets/<broker>/<server>/<instrument>/<dataset_id>.json            dataset manifest

A chunk is COMPLETE only when its manifest exists and the parquet file's checksum matches it. Writes are
atomic (temp file + fsync + rename). Existing chunks are never overwritten: re-writing identical content
is a no-op; different content raises ChunkExistsError (a new acquisition needs a new dataset root/version).
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fxscalp.core.provenance import canonical_json, sha256_bytes, sha256_file, slug, software_versions
from fxscalp.market_data.schema import RAW_BROKER_COLUMNS, RAW_SCHEMA, RAW_SCHEMA_VERSION

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
ROW_GROUP_SIZE = 262_144


class ChunkExistsError(RuntimeError):
    pass


class ChunkCorruptError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChunkKey:
    broker: str
    server: str
    instrument: str          # canonical, e.g. XAU_USD
    source_day: date         # day in the SOURCE time domain

    def rel_dir(self) -> Path:
        d = self.source_day
        return (Path(f"broker={slug(self.broker)}") / f"server={slug(self.server)}" / f"instrument={self.instrument}"
                / f"year={d.year:04d}" / f"month={d.month:02d}" / f"day={d.day:02d}")


def raw_content_sha256(table: pa.Table) -> str:
    """Hash of the broker's own data (independent of Parquet encoding, normalisation and ingestion time)."""
    import hashlib
    h = hashlib.sha256()
    for name in RAW_BROKER_COLUMNS:
        a = table[name].to_numpy(zero_copy_only=False)
        h.update(name.encode())
        h.update(np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<"), copy=False).tobytes())
    return h.hexdigest()


def _atomic_write_parquet(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression=PARQUET_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL,
                   row_group_size=ROW_GROUP_SIZE, use_dictionary=["time_basis"] if "time_basis" in table.column_names else False,
                   write_statistics=True)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)   # immutable by convention + permission


def _atomic_write_json(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


class TickStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    # ---- paths --------------------------------------------------------------------------------
    def raw_dir(self, key: ChunkKey) -> Path:
        return self.root / "raw" / "ticks" / key.rel_dir()

    def derived_dir(self, kind: str, key: ChunkKey) -> Path:
        return self.root / "derived" / kind / key.rel_dir()

    # ---- raw chunks ---------------------------------------------------------------------------
    def chunk_manifest(self, key: ChunkKey) -> dict[str, Any] | None:
        p = self.raw_dir(key) / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else None

    def verify_chunk(self, key: ChunkKey) -> tuple[bool, str]:
        """(ok, reason). Recomputes the file checksum and compares row count with parquet metadata."""
        d = self.raw_dir(key)
        man = self.chunk_manifest(key)
        pqf = d / "ticks.parquet"
        if man is None:
            return False, "manifest missing"
        if not pqf.exists():
            return False, "parquet missing"
        if sha256_file(pqf) != man["file_sha256"]:
            return False, "file checksum mismatch"
        if pq.ParquetFile(pqf).metadata.num_rows != man["record_count"]:
            return False, "row count mismatch"
        return True, "ok"

    def write_raw_chunk(self, key: ChunkKey, table: pa.Table, manifest_fields: dict[str, Any]) -> dict[str, Any]:
        if table.schema != RAW_SCHEMA:
            raise ValueError("raw table schema does not match RAW_SCHEMA")
        d = self.raw_dir(key)
        content = raw_content_sha256(table)
        existing = self.chunk_manifest(key)
        if existing is not None:
            ok, why = self.verify_chunk(key)
            if ok and existing["raw_content_sha256"] == content:
                return existing                                    # idempotent no-op
            raise ChunkExistsError(f"chunk {key} already exists ({'different content' if ok else why}); raw data is "
                                   "immutable: use a new storage root/version to re-acquire")
        d.mkdir(parents=True, exist_ok=True)
        pqf = d / "ticks.parquet"
        if pqf.exists():                                           # orphan from an interrupted write
            os.chmod(pqf, stat.S_IRUSR | stat.S_IWUSR)
            pqf.unlink()
        _atomic_write_parquet(table, pqf)
        n = table.num_rows
        manifest = {
            "manifest_type": "tick_raw_chunk", "schema_version": RAW_SCHEMA_VERSION,
            "broker": key.broker, "server": key.server, "instrument": key.instrument,
            "source_day": key.source_day.isoformat(), "record_count": n,
            "first_source_time_msc": int(table["source_time_msc"][0].as_py()) if n else None,
            "last_source_time_msc": int(table["source_time_msc"][n - 1].as_py()) if n else None,
            "first_normalized_utc_ms": int(table["normalized_utc_time"][0].value) if n else None,
            "last_normalized_utc_ms": int(table["normalized_utc_time"][n - 1].value) if n else None,
            "file_sha256": sha256_file(pqf), "file_bytes": pqf.stat().st_size, "raw_content_sha256": content,
            "parquet": {"compression": PARQUET_COMPRESSION, "level": PARQUET_COMPRESSION_LEVEL,
                        "row_group_size": ROW_GROUP_SIZE},
            "software": software_versions(), "status": "verified",
            **manifest_fields,
        }
        _atomic_write_json(manifest, d / "manifest.json")
        return manifest

    def read_raw_chunk(self, key: ChunkKey, columns: Sequence[str] | None = None, *, verify: bool = True) -> pa.Table:
        if verify:
            ok, why = self.verify_chunk(key)
            if not ok:
                raise ChunkCorruptError(f"chunk {key} failed verification: {why}")
        return pq.read_table(self.raw_dir(key) / "ticks.parquet", columns=list(columns) if columns else None)

    def list_chunks(self, broker: str, server: str, instrument: str) -> list[ChunkKey]:
        base = self.root / "raw" / "ticks" / f"broker={slug(broker)}" / f"server={slug(server)}" / f"instrument={instrument}"
        out: list[ChunkKey] = []
        if not base.exists():
            return out
        for man in sorted(base.glob("year=*/month=*/day=*/manifest.json")):
            m = json.loads(man.read_text())
            out.append(ChunkKey(m["broker"], m["server"], m["instrument"], date.fromisoformat(m["source_day"])))
        return out

    # ---- derived (separate from raw) ------------------------------------------------------------
    def write_derived(self, kind: str, key: ChunkKey, table: pa.Table, parent_manifest: dict[str, Any],
                      extra: dict[str, Any]) -> dict[str, Any]:
        d = self.derived_dir(kind, key)
        d.mkdir(parents=True, exist_ok=True)
        pqf = d / f"{kind}.parquet"
        if pqf.exists():
            os.chmod(pqf, stat.S_IRUSR | stat.S_IWUSR)
            pqf.unlink()
        _atomic_write_parquet(table, pqf)
        man = {"manifest_type": f"{kind}_chunk", "broker": key.broker, "server": key.server,
               "instrument": key.instrument, "source_day": key.source_day.isoformat(), "record_count": table.num_rows,
               "file_sha256": sha256_file(pqf),
               "parents": [{"kind": "tick_raw_chunk", "raw_content_sha256": parent_manifest["raw_content_sha256"],
                            "file_sha256": parent_manifest["file_sha256"]}],
               "software": software_versions(), **extra}
        mp = d / "manifest.json"
        if mp.exists():
            os.chmod(mp, stat.S_IRUSR | stat.S_IWUSR)
        _atomic_write_json(man, mp)
        return man

    def read_derived(self, kind: str, key: ChunkKey) -> pa.Table:
        return pq.read_table(self.derived_dir(kind, key) / f"{kind}.parquet")

    # ---- dataset manifests ----------------------------------------------------------------------
    def dataset_dir(self, broker: str, server: str, instrument: str) -> Path:
        return self.root / "datasets" / slug(broker) / slug(server) / instrument

    def write_dataset_manifest(self, body: dict[str, Any]) -> dict[str, Any]:
        """Dataset ID = hash of identity fields + chunk content hashes; checksum covers the whole body."""
        identity = {k: body[k] for k in ("schema_version", "broker", "server", "instrument", "broker_symbol",
                                         "time_basis", "timebase_spec_sha256", "chunks")}
        did = "tickraw-" + sha256_bytes(canonical_json({**identity, "chunks": [
            (c["source_day"], c["raw_content_sha256"]) for c in body["chunks"]]}).encode())[:20]
        doc = {**body, "dataset_id": did}
        doc["manifest_checksum_sha256"] = sha256_bytes(canonical_json(doc).encode())
        d = self.dataset_dir(body["broker"], body["server"], body["instrument"])
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{did}.json"
        if path.exists():
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        _atomic_write_json(doc, path)
        return doc

    def iter_dataset_manifests(self, broker: str, server: str, instrument: str) -> Iterator[dict[str, Any]]:
        d = self.dataset_dir(broker, server, instrument)
        for p in sorted(d.glob("tickraw-*.json")) if d.exists() else []:
            yield json.loads(p.read_text())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
