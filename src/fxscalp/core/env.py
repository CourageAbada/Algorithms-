"""Minimal .env loader (KEY=VALUE lines). Never overrides variables already set in the environment."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | str = ".env", environ: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ if environ is None else environ
    loaded: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return loaded
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.split(" #", 1)[0].strip().strip('"').strip("'")
        if k and k not in env:
            env[k] = v
            loaded[k] = v
    return loaded
