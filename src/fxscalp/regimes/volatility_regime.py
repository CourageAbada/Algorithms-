"""Volatility regime engine: LOW / NORMAL / HIGH / EXTREME / UNKNOWN.

Uncalibrated mode: percentile of short-term realised volatility within its own trailing history (so the classes are
quantile-balanced BY CONSTRUCTION and carry no absolute meaning), plus a z-score condition for EXTREME. Calibrated mode:
per-session reference levels from a real reference dataset. See spread_regime.py for the causality notes.
"""

from __future__ import annotations

import numpy as np

from fxscalp.regimes.config import ReferenceQuantiles, VolatilityRegimeConfig

UNKNOWN, LOW, NORMAL, HIGH, EXTREME = 0, 1, 2, 3, 4
LABELS = {UNKNOWN: "UNKNOWN", LOW: "LOW", NORMAL: "NORMAL", HIGH: "HIGH", EXTREME: "EXTREME"}


def classify_volatility(cols: dict[str, np.ndarray], cfg: VolatilityRegimeConfig | None = None) -> np.ndarray:
    cfg = cfg or VolatilityRegimeConfig()
    v = np.asarray(cols[cfg.value_key], dtype="float64")
    out = np.full(len(v), UNKNOWN, dtype="float64")
    if cfg.calibration is not None:
        sess = np.asarray(cols[cfg.session_key], dtype="float64")
        for code, (q_lo, q_hi, q_ex) in cfg.calibration.by_session_code.items():
            m = (sess == code) & ~np.isnan(v)
            out[m] = NORMAL
            out[m & (v <= q_lo)] = LOW
            out[m & (v >= q_hi)] = HIGH
            out[m & (v >= q_ex)] = EXTREME
        return out
    pct = np.asarray(cols[cfg.percentile_key], dtype="float64")
    z = np.asarray(cols[cfg.zscore_key], dtype="float64")
    ok = ~(np.isnan(v) | np.isnan(pct))
    out[ok] = NORMAL
    out[ok & (pct <= cfg.p_low)] = LOW
    out[ok & (pct >= cfg.p_high)] = HIGH
    out[ok & (pct >= cfg.p_extreme) & ~np.isnan(z) & (z >= cfg.z_extreme)] = EXTREME
    return out


def calibrate_reference_quantiles(vol_bps: np.ndarray, session_code: np.ndarray, source_dataset_id: str,
                                  quantiles: tuple[float, float, float] = (0.20, 0.80, 0.98),
                                  min_obs: int = 500) -> ReferenceQuantiles:
    v = np.asarray(vol_bps, dtype="float64")
    s = np.asarray(session_code, dtype="float64")
    by: dict[int, tuple[float, ...]] = {}
    for code in np.unique(s[~np.isnan(s)]):
        x = v[(s == code) & np.isfinite(v)]
        if len(x) >= min_obs:
            by[int(code)] = tuple(float(np.quantile(x, q)) for q in quantiles)
    return ReferenceQuantiles(source_dataset_id, tuple(quantiles), by)
