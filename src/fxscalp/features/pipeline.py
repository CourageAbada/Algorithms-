"""In-memory feature pipeline:  normalised ticks -> segments -> bars -> features -> regimes -> feature frame.

The pure function ``build_feature_frame`` is what tests (prefix invariance, leakage) call; ``dataset.run_from_store``
wraps it with Phase 1 Parquet I/O and provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from fxscalp.core.provenance import canonical_json, sha256_bytes
from fxscalp.features.groups import Emitter, FeatureContext
from fxscalp.features.normalize import split_segments
from fxscalp.features.registry import Registry, build_registry
from fxscalp.features.segment import build_context
from fxscalp.features.spec import FeatureConfig
from fxscalp.regimes import spread_regime, volatility_regime
from fxscalp.regimes.config import RegimeConfig
from fxscalp.sessions.calendar import DEFAULT_RULES, SessionRules

SESSION_LABELS = {0: "ASIA", 1: "LONDON", 2: "NEW_YORK", 3: "LONDON_NEW_YORK_OVERLAP", 4: "OFF_HOURS", 5: "MARKET_CLOSED"}
ML_SCHEMA_VERSION = "ml_dataset/1"
PIPELINE_VERSION = "features_pipeline/1"


@dataclass(frozen=True)
class PipelineConfig:
    instrument: str = "XAU_USD"
    grid_s: int = 1
    segment_gap_s: int = 1800
    features: FeatureConfig = field(default_factory=FeatureConfig)
    regimes: RegimeConfig = field(default_factory=RegimeConfig)
    rules: SessionRules = DEFAULT_RULES

    def to_dict(self) -> dict[str, Any]:
        return {"instrument": self.instrument, "grid_s": self.grid_s, "segment_gap_s": self.segment_gap_s,
                "features": self.features.to_dict(), "regimes": self.regimes.to_dict(), "rules": asdict(self.rules),
                "pipeline_version": PIPELINE_VERSION, "ml_schema_version": ML_SCHEMA_VERSION}

    def config_hash(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict()).encode())


@dataclass
class FeatureFrame:
    df: pd.DataFrame
    registry: Registry
    segments: list[dict[str, Any]]
    config: PipelineConfig
    bars: dict[str, pd.DataFrame] = field(default_factory=dict)       # filled when collect_bars=True

    @property
    def feature_columns(self) -> list[str]:
        return [c for c in self.registry.names if c in self.df.columns]


ExtraFn = Callable[[FeatureContext, dict[str, np.ndarray]], dict[str, np.ndarray]]


def build_feature_frame(ticks: pd.DataFrame, cfg: PipelineConfig | None = None, *, point: float | None,
                        extra: ExtraFn | None = None, collect_bars: bool = False) -> FeatureFrame:
    """``ticks``: output of normalize_ticks (time-ordered). ``extra`` lets tests inject deliberately bad features."""
    cfg = cfg or PipelineConfig()
    reg = build_registry(cfg.features, cfg.regimes)
    from fxscalp.features.groups import GROUPS
    warm_rows = int(np.ceil(reg.max_lookback_s / cfg.grid_s))
    ts_all = ticks["timestamp_utc_ms"].to_numpy()
    segs = split_segments(ts_all, cfg.segment_gap_s)
    frames, info = [], []
    bars_acc: dict[str, list[pd.DataFrame]] = {}
    for sid, (lo, hi) in enumerate(segs):
        sl = ticks.iloc[lo:hi]
        ctx = build_context(sl["timestamp_utc_ms"].to_numpy(), sl["bid"].to_numpy(), sl["ask"].to_numpy(),
                            sl["volume"].to_numpy(), grid_s=cfg.grid_s, point=point, cfg=cfg.features,
                            regime_cfg=cfg.regimes, rules=cfg.rules, data_continues_after=sid < len(segs) - 1)
        if ctx is None:
            info.append({"segment_id": sid, "ticks": hi - lo, "rows": 0, "note": "too short for a complete grid bar"})
            continue
        prior: dict[str, np.ndarray] = {}
        out: dict[str, np.ndarray] = {}
        for g in GROUPS:
            em = Emitter(g.name, g.definition_version, cfg.features.disabled_features)
            g.fn(ctx, cfg.features, prior, em)
            for spec, arr in em.items:
                prior[spec.name] = arr
                if spec.enabled:
                    out[spec.name] = arr
        if extra is not None:
            out.update(extra(ctx, prior))
        n = ctx.n
        df = pd.DataFrame(out)
        df.insert(0, "timestamp_utc_ms", ctx.end_ms)
        df.insert(1, "segment_id", sid)
        df.insert(2, "row_valid", (np.arange(n) >= warm_rows).astype("int8"))
        sc = np.nan_to_num(prior["session_code"], nan=4).astype(int)
        df["session"] = [SESSION_LABELS[c] for c in sc]
        df["spread_regime"] = [spread_regime.LABELS[int(c)] for c in prior["spread_regime_code"]]
        df["volatility_regime"] = [volatility_regime.LABELS[int(c)] for c in prior["volatility_regime_code"]]
        frames.append(df)
        if collect_bars:
            bars_acc.setdefault(f"{cfg.grid_s}s", []).append(ctx.base.assign(segment_id=sid))
            for tf, b in ctx.mtf.items():
                bars_acc.setdefault(f"{tf}s", []).append(b.assign(segment_id=sid))
        info.append({"segment_id": sid, "ticks": hi - lo, "rows": n, "start_utc_ms": int(ctx.start_ms[0]),
                     "end_utc_ms": int(ctx.end_ms[-1]), "valid_rows": int(max(0, n - warm_rows))})
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp_utc_ms"])
    bars = {k: pd.concat(v, ignore_index=True) for k, v in bars_acc.items()}
    return FeatureFrame(df, reg, info, cfg, bars)
