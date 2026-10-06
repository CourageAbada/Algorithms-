"""Build the per-segment FeatureContext (base grid bars + multi-timeframe bars) from normalised ticks."""

from __future__ import annotations

import numpy as np

from fxscalp.features.bars import aggregate_bars
from fxscalp.features.groups import FeatureContext
from fxscalp.features.spec import FeatureConfig
from fxscalp.regimes.config import RegimeConfig
from fxscalp.sessions.calendar import DEFAULT_RULES, SessionRules


def build_context(ts_ms: np.ndarray, bid: np.ndarray, ask: np.ndarray, volume: np.ndarray, *, grid_s: int,
                  point: float | None, cfg: FeatureConfig, regime_cfg: RegimeConfig | None = None,
                  rules: SessionRules = DEFAULT_RULES, data_continues_after: bool = False) -> FeatureContext | None:
    base = aggregate_bars(ts_ms, bid, ask, volume, grid_s, data_continues_after=data_continues_after, micro=True)
    if base.empty:
        return None
    base = base[base["complete"]].reset_index(drop=True)          # decision rows = complete grid bars only
    if base.empty:
        return None
    tfs = sorted(set(cfg.mtf_timeframes_s) | set(cfg.atr_timeframes_s))
    mtf = {tf: aggregate_bars(ts_ms, bid, ask, volume, tf, data_continues_after=data_continues_after) for tf in tfs}
    return FeatureContext(cfg=cfg, grid_s=grid_s, point=point, base=base, mtf=mtf, rules=rules,
                          regime_cfg=regime_cfg or RegimeConfig())
