"""Pure statistical helpers named in the v2 draft (effective sample size). Draft helper: executes nothing on market data by itself."""

from __future__ import annotations

import numpy as np


def effective_sample_size(series: np.ndarray, max_lag: int = 120) -> float:
    """n_eff = n / (1 + 2 * sum_{k=1..K} (1 - k/(K+1)) * rho_k) with a Bartlett window (K = max_lag), floored at 1 and capped at n."""
    x = np.asarray(series, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float(n)
    x = x - x.mean()
    v = float((x * x).mean())
    if v == 0:
        return float(n)
    K = min(max_lag, n - 2)
    acc = 0.0
    for k in range(1, K + 1):
        rho = float((x[:-k] * x[k:]).mean() / v)
        acc += (1 - k / (K + 1)) * rho
    return float(min(n, max(1.0, n / max(1.0 + 2.0 * acc, 1e-9))))
