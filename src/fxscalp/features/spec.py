"""Typed feature metadata and feature-engine configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    family: str
    description: str
    source: str                 # "ticks" | "bars_<tf>s" | "session" | "regime_engine"
    timeframe: str              # grid timeframe label or the bar timeframe used
    lookback_s: int             # seconds of history needed for a fully valid value (warm-up)
    units: str
    expected_range: tuple[float | None, float | None] | None
    nan_policy: str
    leakage_risk: str
    reason: str
    enabled: bool = True
    version: int = 1            # == the owning group's definition_version

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["expected_range"] = list(self.expected_range) if self.expected_range else None
        return d


@dataclass(frozen=True)
class FeatureConfig:
    """All tunable feature parameters. Part of the feature-set version: changing any value changes the version."""

    micro_windows_s: tuple[int, ...] = (5, 15, 60)
    vol_windows_s: tuple[int, ...] = (15, 60, 300)
    ret_windows_s: tuple[int, ...] = (5, 15, 60, 300)
    abs_ret_windows_s: tuple[int, ...] = (15, 60, 300)
    range_windows_s: tuple[int, ...] = (30, 60, 300)
    breakout_windows_s: tuple[int, ...] = (60, 300)
    breakout_guard_s: int = 5            # prior-high window excludes the most recent guard seconds
    failed_breakout_window_s: int = 300
    failed_breakout_lookback_s: int = 30
    spread_mean_windows_s: tuple[int, ...] = (15, 60, 300)
    spread_median_windows_s: tuple[int, ...] = (60, 300)
    spread_change_windows_s: tuple[int, ...] = (5, 60)
    spread_hist_s: int = 1800            # percentile / z-score history for spread
    vol_hist_s: int = 3600               # percentile / z-score history for volatility and range
    mean_windows_s: tuple[int, ...] = (60, 300)
    ema_spans_s: tuple[int, ...] = (30, 120)
    ema_slope_step_s: int = 5
    mtf_timeframes_s: tuple[int, ...] = (5, 15, 30, 60, 300)
    atr_timeframes_s: tuple[int, ...] = (15, 60, 300)
    atr_period: int = 14
    min_median_frac: float = 0.2
    disabled_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}
