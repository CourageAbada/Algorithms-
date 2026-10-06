"""Causal feature scaling: parameters are fit ONLY on rows strictly before ``fit_end_ms`` (the training boundary).
Fitting on the whole dataset would leak the future distribution into the past; ``fit_end_ms`` is therefore mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CausalScaler:
    fit_end_ms: int                                   # exclusive: rows with timestamp < fit_end_ms are used
    ts_col: str = "timestamp_utc_ms"
    params: dict[str, tuple[float, float]] = field(default_factory=dict)    # col -> (median, scale)
    n_fit_rows: int = 0

    def fit(self, frame: pd.DataFrame, cols: list[str]) -> "CausalScaler":
        train = frame[frame[self.ts_col] < self.fit_end_ms]
        self.n_fit_rows = len(train)
        self.params = {}
        for c in cols:
            x = train[c].to_numpy(dtype="float64")
            x = x[np.isfinite(x)]
            if len(x) < 2:
                self.params[c] = (float("nan"), float("nan"))
                continue
            med = float(np.median(x))
            iqr = float(np.quantile(x, 0.75) - np.quantile(x, 0.25))
            self.params[c] = (med, iqr if iqr > 0 else float(np.std(x)) or 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for c, (med, sc) in self.params.items():
            out[c] = (out[c] - med) / sc
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"fit_end_ms": self.fit_end_ms, "n_fit_rows": self.n_fit_rows, "params": {k: list(v) for k, v in self.params.items()}}


def check_scaler_fit_is_causal(make_scaler: Any, frame: pd.DataFrame, cols: list[str], fit_end_ms: int,
                               ts_col: str = "timestamp_utc_ms") -> bool:
    """True iff the parameters fit on the FULL frame equal those fit on the frame truncated at ``fit_end_ms``.
    A scaler that ignores ``fit_end_ms`` (whole-dataset fitting) fails this check."""
    a = make_scaler(fit_end_ms).fit(frame, cols).params
    b = make_scaler(fit_end_ms).fit(frame[frame[ts_col] < fit_end_ms], cols).params
    return all(np.allclose(np.array(a[c]), np.array(b[c]), equal_nan=True, rtol=0, atol=0) for c in cols)
