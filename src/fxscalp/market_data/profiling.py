"""Exact histogram utilities for the streaming dataset profile (pure, unit-tested)."""

from __future__ import annotations

from typing import Any

import numpy as np

BIN = 0.1                  # spread histogram resolution (points)
NBINS = 30_000             # up to 3000 points; larger values are clipped into the last bin (true max tracked separately)


def hist_add(h: np.ndarray, values: np.ndarray) -> None:
    v = values[np.isfinite(values) & (values >= 0)]
    if v.size:
        h += np.bincount(np.minimum((v / BIN).astype("int64"), NBINS - 1), minlength=NBINS)


def hist_q(h: np.ndarray, qs: tuple[float, ...]) -> dict[str, float | None]:
    n = int(h.sum())
    if n == 0:
        return {f"p{q:g}": None for q in qs}
    c = np.cumsum(h)
    return {f"p{q:g}": float((np.searchsorted(c, q / 100.0 * n, side="left") + 0.5) * BIN) for q in qs}


def hist_summary(h: np.ndarray) -> dict[str, Any]:
    n = int(h.sum())
    if n == 0:
        return {"n": 0}
    idx = np.flatnonzero(h)
    mean = float((np.arange(NBINS) + 0.5) @ h * BIN / n)
    return {"n": n, "min_approx": float(idx[0] * BIN), "mean": round(mean, 3),
            **{k: (None if v is None else round(v, 2)) for k, v in hist_q(h, (5, 25, 50, 75, 95, 99, 99.9)).items()},
            "max_approx": float((idx[-1] + 1) * BIN)}


def pct(a: np.ndarray, qs: tuple[float, ...]) -> dict[str, float | None]:
    if a.size == 0:
        return {f"p{q:g}": None for q in qs}
    return {f"p{q:g}": float(np.percentile(a, q)) for q in qs}
