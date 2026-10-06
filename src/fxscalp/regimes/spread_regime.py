"""Spread regime engine: NORMAL / ELEVATED / EXTREME / UNKNOWN. No permanent absolute thresholds.

Uncalibrated mode (tonight): a row is ELEVATED/EXTREME only when BOTH its percentile within the trailing history AND its
ratio to the recent median spread exceed the (relative) cut-offs, so a persistently wide but stable spread is not
flagged by the percentile alone and a tiny jitter is not flagged by the ratio alone.
Calibrated mode (after real data): per-session reference levels measured on a reference dataset decide the class.
Causality: inputs are trailing-window features; the calibration is a fixed object supplied from outside the data being
classified (it must come from a different/earlier dataset to avoid look-ahead; recorded in the manifest).
"""

from __future__ import annotations

import numpy as np

from fxscalp.regimes.config import ReferenceQuantiles, SpreadRegimeConfig

UNKNOWN, NORMAL, ELEVATED, EXTREME = 0, 1, 2, 3
LABELS = {UNKNOWN: "UNKNOWN", NORMAL: "NORMAL", ELEVATED: "ELEVATED", EXTREME: "EXTREME"}


def classify_spread(cols: dict[str, np.ndarray], cfg: SpreadRegimeConfig | None = None) -> np.ndarray:
    cfg = cfg or SpreadRegimeConfig()
    v = np.asarray(cols[cfg.value_key], dtype="float64")
    out = np.full(len(v), UNKNOWN, dtype="float64")
    if cfg.calibration is not None:
        sess = np.asarray(cols[cfg.session_key], dtype="float64")
        for code, (q_el, q_ex) in cfg.calibration.by_session_code.items():
            m = (sess == code) & ~np.isnan(v)
            out[m] = NORMAL
            out[m & (v >= q_el)] = ELEVATED
            out[m & (v >= q_ex)] = EXTREME
        return out
    pct = np.asarray(cols[cfg.percentile_key], dtype="float64")
    med = np.asarray(cols[cfg.median_key], dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(med > 0, v / med, np.nan)
    ok = ~(np.isnan(v) | np.isnan(pct) | np.isnan(ratio))
    out[ok] = NORMAL
    out[ok & (pct >= cfg.p_elevated) & (ratio >= cfg.ratio_elevated)] = ELEVATED
    out[ok & (pct >= cfg.p_extreme) & (ratio >= cfg.ratio_extreme)] = EXTREME
    return out


def calibrate_reference_quantiles(spread_points: np.ndarray, session_code: np.ndarray, source_dataset_id: str,
                                  quantiles: tuple[float, float] = (0.95, 0.995), min_obs: int = 500) -> ReferenceQuantiles:
    """Measure per-session reference levels on a reference dataset (to run on REAL data in Phase 2B).
    Sessions with fewer than ``min_obs`` finite observations are omitted (-> UNKNOWN when classifying)."""
    v = np.asarray(spread_points, dtype="float64")
    s = np.asarray(session_code, dtype="float64")
    by: dict[int, tuple[float, ...]] = {}
    for code in np.unique(s[~np.isnan(s)]):
        x = v[(s == code) & np.isfinite(v)]
        if len(x) >= min_obs:
            by[int(code)] = tuple(float(np.quantile(x, q)) for q in quantiles)
    return ReferenceQuantiles(source_dataset_id, tuple(quantiles), by)
