"""Feature registry and FEATURE_SET_VERSION.

The registry is produced by running every group once on a tiny dummy context, so metadata and computation come from the
same ``emit`` calls. The feature-set version is a hash of: group definition versions, normalised group + shared
source (so editing a feature definition changes the version), the feature configuration, the regime configuration and
the schema versions. A model must record the exact version it was trained on.

Source normalisation (CRLF->LF, trailing whitespace stripped) keeps the hash identical between Linux and Windows
checkouts. ``configs/features/xauusd_core.registry.yaml`` is the committed snapshot; a test fails when it drifts.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import yaml

from fxscalp.core.provenance import canonical_json, sha256_bytes
from fxscalp.features import bars as bars_mod
from fxscalp.features import groups as groups_mod
from fxscalp.features import ops as ops_mod
from fxscalp.features import session_features as session_mod
from fxscalp.features.groups import GROUPS, Emitter, FeatureGroup
from fxscalp.features.segment import build_context
from fxscalp.features.spec import FeatureConfig, FeatureSpec
from fxscalp.regimes import spread_regime, volatility_regime
from fxscalp.regimes.config import RegimeConfig

FEATURE_SET_NAME = "xauusd_core"
FEATURE_SET_SCHEMA = 1            # bump if the registry/dataset layout (not a feature) changes


def _norm(src: str) -> str:
    return "\n".join(line.rstrip() for line in src.replace("\r\n", "\n").split("\n"))


def _src(obj: Any) -> str:
    return _norm(inspect.getsource(obj))


@lru_cache(maxsize=None)
def _shared_source_hash() -> str:
    return sha256_bytes("\n#--\n".join(_src(m) for m in (ops_mod, bars_mod, session_mod)).encode() +
                        _src(groups_mod.FeatureContext).encode() + _src(groups_mod.Emitter).encode())


def group_fingerprint(g: FeatureGroup) -> str:
    extra = ""
    if g.name == "regimes":
        extra = _src(spread_regime) + _src(volatility_regime)
    return sha256_bytes((_src(g.fn) + extra).encode() + _shared_source_hash().encode())[:16]


def _dummy_context(cfg: FeatureConfig, regime_cfg: RegimeConfig):
    n = 40
    ts = 1_760_000_000_000 + np.arange(n, dtype="int64") * 500
    bid = 2000.0 + np.sin(np.arange(n)) * 0.05
    return build_context(ts, bid, bid + 0.25, np.zeros(n), grid_s=1, point=0.01, cfg=cfg, regime_cfg=regime_cfg,
                         data_continues_after=True)


@dataclass(frozen=True)
class Registry:
    name: str
    version: str
    specs: tuple[FeatureSpec, ...]
    groups: tuple[tuple[str, int, str], ...]           # (name, definition_version, fingerprint)
    feature_config: dict[str, Any]
    regime_config: dict[str, Any]

    @property
    def enabled(self) -> tuple[FeatureSpec, ...]:
        return tuple(s for s in self.specs if s.enabled)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.enabled]

    @property
    def max_lookback_s(self) -> int:
        return max((s.lookback_s for s in self.enabled), default=0)

    def spec(self, name: str) -> FeatureSpec:
        for s in self.specs:
            if s.name == name:
                return s
        raise KeyError(name)

    def to_yaml_dict(self) -> dict[str, Any]:
        return {"feature_set_name": self.name, "feature_set_version": self.version, "feature_set_schema": FEATURE_SET_SCHEMA,
                "groups": [{"name": n, "definition_version": v, "code_fingerprint": f} for n, v, f in self.groups],
                "feature_config": self.feature_config, "regime_config": self.regime_config,
                "features": [s.to_dict() for s in self.specs]}


@lru_cache(maxsize=16)
def _build(cfg: FeatureConfig, regime_cfg: RegimeConfig) -> Registry:
    ctx = _dummy_context(cfg, regime_cfg)
    prior: dict[str, np.ndarray] = {}
    specs: list[FeatureSpec] = []
    ginfo = []
    for g in GROUPS:
        em = Emitter(g.name, g.definition_version, cfg.disabled_features)
        g.fn(ctx, cfg, prior, em)
        for spec, arr in em.items:
            if spec.name in prior:
                raise ValueError(f"duplicate feature name {spec.name!r}")
            prior[spec.name] = arr
            specs.append(spec)
        ginfo.append((g.name, g.definition_version, group_fingerprint(g)))
    ident = {"name": FEATURE_SET_NAME, "schema": FEATURE_SET_SCHEMA, "groups": ginfo, "cfg": cfg.to_dict(),
             "regimes": regime_cfg.to_dict(), "features": [s.to_dict() for s in specs]}
    version = f"{FEATURE_SET_NAME}-fs{FEATURE_SET_SCHEMA}-{sha256_bytes(canonical_json(ident).encode())[:12]}"
    return Registry(FEATURE_SET_NAME, version, tuple(specs), tuple(ginfo), cfg.to_dict(), regime_cfg.to_dict())


def build_registry(cfg: FeatureConfig | None = None, regime_cfg: RegimeConfig | None = None) -> Registry:
    return _build(cfg or FeatureConfig(), regime_cfg or RegimeConfig())


def registry_yaml(reg: Registry) -> str:
    return yaml.safe_dump(reg.to_yaml_dict(), sort_keys=False, default_flow_style=False, width=140)
