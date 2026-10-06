"""YAML configuration loading. Secrets never live in YAML; they come from the environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_INSTRUMENT_KEYS = ("canonical", "broker_symbol_candidates", "risk", "spread")


class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def load_instrument(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    missing = [k for k in REQUIRED_INSTRUMENT_KEYS if k not in cfg]
    if missing:
        raise ConfigError(f"{path}: missing keys {missing}")
    return cfg
