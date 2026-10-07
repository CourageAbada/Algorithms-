"""Training-window-only preprocessing (non-stationarity policy).

Nothing learned from a distribution may be fit on the whole six months: scalers, winsorisation limits, volatility/spread/regime
thresholds, outlier detectors, projections and feature selection are all fit on a fold's TRAINING block and then only
APPLIED to validation/test. ``FoldPreprocessor`` enforces that (``LeakageError`` if a fit sees a non-training timestamp) and
``train_only_*`` helpers give the same guarantee for the other fitted quantities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fxscalp.features.scaler import CausalScaler
from fxscalp.research.folds import EMBARGO_S, FoldSpec


class LeakageError(RuntimeError):
    """A statistic was (about to be) fit on data outside the fold's training block."""


def assert_training_only(ts_ms: np.ndarray, fold: FoldSpec, label_end_ms: np.ndarray | None = None,
                         embargo_s: int = EMBARGO_S) -> None:
    t = np.asarray(ts_ms, dtype="int64")
    if t.size and (int(t.min()) < fold.train_start_ms or int(t.max()) >= fold.train_end_ms):
        raise LeakageError(f"fit data spans [{t.min()}, {t.max()}] outside the {fold.name} training block "
                           f"[{fold.train_start_ms}, {fold.train_end_ms})")
    if label_end_ms is not None and np.asarray(label_end_ms).size and int(np.max(label_end_ms)) > fold.val_start_ms - embargo_s * 1000:
        raise LeakageError("fit samples have labels that extend into the embargo/validation period (not purged)")


@dataclass
class FoldPreprocessor:
    fold: FoldSpec
    columns: list[str]
    ts_col: str = "timestamp_utc_ms"
    scaler: CausalScaler | None = None
    fitted_on: dict[str, Any] = field(default_factory=dict)

    def fit(self, train_frame: pd.DataFrame) -> "FoldPreprocessor":
        assert_training_only(train_frame[self.ts_col].to_numpy(), self.fold)
        self.scaler = CausalScaler(fit_end_ms=self.fold.train_end_ms, ts_col=self.ts_col).fit(train_frame, self.columns)
        self.fitted_on = {"fold": self.fold.name, "n_rows": self.scaler.n_fit_rows,
                          "first_ms": int(train_frame[self.ts_col].min()), "last_ms": int(train_frame[self.ts_col].max())}
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise LeakageError("transform before fit: statistics must come from the training block")
        return self.scaler.transform(frame)

    # deliberately NO fit_transform: fitting and applying in one call invites fitting on the evaluation frame


def train_only_quantile(values: np.ndarray, ts_ms: np.ndarray, fold: FoldSpec, q: float | list[float]) -> np.ndarray:
    """Distribution threshold (winsorisation limit, volatility/spread/regime cut-off) from the training block only."""
    assert_training_only(ts_ms, fold)
    v = np.asarray(values, dtype="float64")
    return np.nanquantile(v[np.isfinite(v)], q)


def train_only_baseline(values: np.ndarray, ts_ms: np.ndarray, fold: FoldSpec) -> dict[str, float]:
    """Median / MAD baseline (e.g. spread baseline) from the training block only."""
    assert_training_only(ts_ms, fold)
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    med = float(np.median(v))
    return {"median": med, "mad": float(np.median(np.abs(v - med))), "n": int(len(v))}
