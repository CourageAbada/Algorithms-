"""Causal array primitives. THE ONLY place feature code may aggregate over time.

Every function returns a value at row i that depends only on rows <= i of its inputs (the property enforced by the
prefix-invariance tests). Feature groups must be built exclusively from these primitives plus element-wise arithmetic;
`features/leakage.py::scan_source` rejects direct statistical calls (`.mean()`, `.std()`, `np.mean`, ...) and any
negative shift / centred window in group code.

Windows are expressed in ROWS of a uniform grid. ``warm`` marks the first k-1 rows NaN so that a partially filled
window is never presented as a full one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _s(x: np.ndarray) -> pd.Series:
    return pd.Series(np.asarray(x, dtype="float64"))


def warm(x: np.ndarray, k: int) -> np.ndarray:
    """NaN the first k-1 rows (window not yet full)."""
    out = np.array(x, dtype="float64", copy=True)
    if k > 1:
        out[: k - 1] = np.nan
    return out


def lag(x: np.ndarray, k: int) -> np.ndarray:
    """Value k rows ago (k >= 0). Never negative: a negative shift would read the future."""
    if k < 0:
        raise ValueError("lag(k) requires k >= 0 (negative shifts read future rows)")
    x = np.asarray(x, dtype="float64")
    out = np.full(x.shape, np.nan)
    if k == 0:
        return x.copy()
    if k < len(x):
        out[k:] = x[:-k]
    return out


def rsum(x: np.ndarray, k: int) -> np.ndarray:
    return warm(_s(x).rolling(k, min_periods=1).sum().to_numpy(), k)


def rmean(x: np.ndarray, k: int) -> np.ndarray:
    return warm(_s(x).rolling(k, min_periods=1).mean().to_numpy(), k)


def rmax(x: np.ndarray, k: int) -> np.ndarray:
    return warm(_s(x).rolling(k, min_periods=1).max().to_numpy(), k)


def rmin(x: np.ndarray, k: int) -> np.ndarray:
    return warm(_s(x).rolling(k, min_periods=1).min().to_numpy(), k)


def rstd(x: np.ndarray, k: int) -> np.ndarray:
    return warm(_s(x).rolling(k, min_periods=2).std().to_numpy(), k)


def rmedian(x: np.ndarray, k: int, min_frac: float = 0.2) -> np.ndarray:
    """Rolling median of the non-NaN values; needs at least ceil(min_frac*k) of them."""
    mp = max(1, int(np.ceil(min_frac * k)))
    return warm(_s(x).rolling(k, min_periods=mp).median().to_numpy(), k)


def rrank(x: np.ndarray, k: int) -> np.ndarray:
    """Percentile rank (0..1, average method) of x[i] within the trailing window of k rows ending at i."""
    return warm(_s(x).rolling(k, min_periods=k).rank(pct=True).to_numpy(), k)


def ewma(x: np.ndarray, span: int) -> np.ndarray:
    """Exponentially weighted mean (adjust=False): recursive, causal. First span-1 rows NaN (warm-up bias)."""
    return warm(_s(x).ewm(span=span, adjust=False).mean().to_numpy(), span)


def ffill_asof(x: np.ndarray) -> np.ndarray:
    """EXPLICIT as-of state: carry the last observation forward. Only for state variables (last price/spread),
    always paired with a staleness feature. Never used to fill missing bar data."""
    return _s(x).ffill().to_numpy()


def cummax_by(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype="float64")).groupby(np.asarray(group)).cummax().to_numpy()


def cummin_by(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype="float64")).groupby(np.asarray(group)).cummin().to_numpy()


def cummean_by(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    g = np.asarray(group)
    s = pd.Series(np.asarray(x, dtype="float64"))
    return (s.groupby(g).cumsum() / pd.Series(np.ones(len(s))).groupby(g).cumsum()).to_numpy()


def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a/b with b == 0 or NaN -> NaN (never inf)."""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b != 0, a / b, np.nan)
    return out


def bps(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return 1e4 * ratio(num, den)


def log_ret_bps(x: np.ndarray, k: int) -> np.ndarray:
    """1e4 * ln(x_t / x_{t-k}); NaN where either is non-positive/NaN."""
    x = np.asarray(x, dtype="float64")
    p = lag(x, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((x > 0) & (p > 0), 1e4 * np.log(x / p), np.nan)
