"""Regime engine configuration.

IMPORTANT: every default below is a RELATIVE parameter (a percentile or a ratio against the instrument's own recent
history), not an absolute XAU/USD spread or volatility threshold, and all of them are UNCALIBRATED placeholders chosen
only so the engine can be exercised on synthetic data. They are recorded in every dataset manifest together with
``calibration_status``. Real calibration (per-session reference quantiles from real broker data) happens in Phase 2B via
``calibrate_reference_quantiles``; nothing derived from synthetic data may be treated as a market fact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReferenceQuantiles:
    """Per-session reference levels measured on a (real) reference dataset. Replaces local-percentile logic."""

    source_dataset_id: str
    quantiles: tuple[float, ...]                       # which quantiles were measured, e.g. (0.95, 0.995)
    by_session_code: dict[int, tuple[float, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"source_dataset_id": self.source_dataset_id, "quantiles": list(self.quantiles),
                "by_session_code": {str(k): list(v) for k, v in sorted(self.by_session_code.items())}}


@dataclass(frozen=True)
class SpreadRegimeConfig:
    percentile_key: str = "spread_percentile_1800s"
    median_key: str = "spread_median_points_300s"
    value_key: str = "spread_points"
    session_key: str = "session_code"
    p_elevated: float = 0.90
    p_extreme: float = 0.99
    ratio_elevated: float = 1.5          # current / recent (5 min) median spread
    ratio_extreme: float = 3.0
    calibration: ReferenceQuantiles | None = None      # (q_elevated, q_extreme) per session when calibrated
    calibration_status: str = "UNCALIBRATED_PLACEHOLDER_DEFAULTS"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["calibration"] = self.calibration.to_dict() if self.calibration else None
        return d


@dataclass(frozen=True)
class VolatilityRegimeConfig:
    value_key: str = "realized_vol_bps_60s"
    percentile_key: str = "vol_percentile_60s_1h"
    zscore_key: str = "vol_zscore_60s_1h"
    session_key: str = "session_code"
    p_low: float = 0.20
    p_high: float = 0.80
    p_extreme: float = 0.98
    z_extreme: float = 2.0
    calibration: ReferenceQuantiles | None = None      # (q_low, q_high, q_extreme) per session when calibrated
    calibration_status: str = "UNCALIBRATED_PLACEHOLDER_DEFAULTS"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["calibration"] = self.calibration.to_dict() if self.calibration else None
        return d


@dataclass(frozen=True)
class RegimeConfig:
    spread: SpreadRegimeConfig = field(default_factory=SpreadRegimeConfig)
    volatility: VolatilityRegimeConfig = field(default_factory=VolatilityRegimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return {"spread": self.spread.to_dict(), "volatility": self.volatility.to_dict()}
