"""Hashing, software versions and dataset identity helpers (start of the provenance chain)."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fxscalp

_TRACKED = ("MetaTrader5", "numpy", "pandas", "pyarrow", "tzdata")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON (sorted keys, no whitespace) for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def git_commit(repo: Path | None = None) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo or Path(__file__).resolve().parents[3],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def software_versions() -> dict[str, Any]:
    pkgs: dict[str, str | None] = {}
    for name in _TRACKED:
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    return {"fxscalp": fxscalp.__version__, "git_commit": git_commit(), "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(), "packages": pkgs}


def slug(text: str) -> str:
    """Filesystem-safe lowercase slug."""
    out = "".join(c.lower() if c.isalnum() or c in "-_." else "_" for c in text.strip())
    return out.strip("._") or "unknown"


@dataclass(frozen=True)
class DatasetRef:
    """Identity of a dataset: enough to say 'MODEL X was trained from DATASET Y'."""

    dataset_id: str
    manifest_sha256: str


@dataclass(frozen=True)
class LineageRecord:
    """How a derived dataset/feature set/model was produced from parents (Phase 1 only defines it)."""

    output_kind: str                      # "tick_derived" | "tick_quality" | "features" | "model" ...
    parents: tuple[DatasetRef, ...]
    transformation: str
    transformation_version: str
    parameters: dict[str, Any] = field(default_factory=dict)
    code_commit: str | None = None
