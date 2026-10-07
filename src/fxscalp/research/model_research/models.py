"""Models, preprocessing, calibration and regime fitting - every fitted object sees TRAINING rows only.

Pipelines: multinomial logistic regression (median/IQR scaling, winsorisation and imputation fit on the training block) and
LightGBM gradient boosting (native missing values). Probability calibration is single-parameter temperature scaling fit on a
chronologically later slice of the TRAINING block. Regime codes (REGIME config) are recomputed from reference quantiles fit
on the fold's training rows (``fit_regime_config``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.optimize import minimize_scalar

from fxscalp.regimes.config import RegimeConfig, SpreadRegimeConfig, VolatilityRegimeConfig
from fxscalp.regimes.spread_regime import calibrate_reference_quantiles as spread_cal
from fxscalp.regimes.spread_regime import classify_spread
from fxscalp.regimes.volatility_regime import calibrate_reference_quantiles as vol_cal
from fxscalp.regimes.volatility_regime import classify_volatility
from fxscalp.research.folds import FoldSpec
from fxscalp.research.model_research import protocol as P
from fxscalp.research.preprocessing import assert_training_only

REGIME_INPUTS = ("spread_points", "session_code", "realized_vol_bps_60s")
SESSION_AGGREGATES = ("minutes_since_session_open", "minutes_until_session_close", "session_high_so_far", "session_low_so_far",
                      "session_range_so_far_bps", "dist_session_high_bps", "dist_session_low_bps", "dist_session_mean_bps")
GATE = "session_so_far_complete"


# ------------------------------------------------------------------------------------------ regimes (train-only)
def fit_regime_config(train_cols: dict[str, np.ndarray], ts_ms: np.ndarray, fold: FoldSpec) -> RegimeConfig:
    assert_training_only(ts_ms, fold)
    sp = spread_cal(train_cols["spread_points"], train_cols["session_code"], f"{fold.name}-training-block", (0.95, 0.995))
    vo = vol_cal(train_cols["realized_vol_bps_60s"], train_cols["session_code"], f"{fold.name}-training-block", (0.20, 0.80, 0.98))
    tag = f"CALIBRATED_TRAIN_ONLY:{fold.name}"
    return RegimeConfig(SpreadRegimeConfig(calibration=sp, calibration_status=tag), VolatilityRegimeConfig(calibration=vo, calibration_status=tag))


def apply_regime(cols: dict[str, np.ndarray], cfg: RegimeConfig) -> tuple[np.ndarray, np.ndarray]:
    return classify_spread(cols, cfg.spread), classify_volatility(cols, cfg.volatility)


def feature_matrix(frame: Any, config: str, allowed: list[str], regime_cfg: RegimeConfig | None = None) -> tuple[np.ndarray, list[str]]:
    """Feature matrix (float32) for a feature configuration. CONDITIONAL features are masked unless the gate holds."""
    names = list(allowed)
    cols = [frame[c].to_numpy("float32") for c in names]
    if config == "CONDITIONAL":
        gate = frame[GATE].to_numpy("float32") == 1.0
        for c in SESSION_AGGREGATES:
            v = frame[c].to_numpy("float32").copy()
            v[~gate] = np.nan
            names.append(c)
            cols.append(v)
    elif config == "REGIME":
        if regime_cfg is None:
            raise ValueError("REGIME needs a regime configuration fitted on the training block")
        base = {k: frame[k].to_numpy("float64") for k in REGIME_INPUTS}
        s, v = apply_regime(base, regime_cfg)
        names += ["spread_regime_code", "volatility_regime_code"]
        cols += [s.astype("float32"), v.astype("float32")]
    elif config != "CORE":
        raise ValueError(f"unknown feature configuration {config!r}")
    return np.column_stack(cols).astype("float32", copy=False), names


# ------------------------------------------------------------------------------------------ calibration
def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    lp = np.log(np.clip(proba, 1e-15, 1.0))

    def nll(logT: float) -> float:
        z = lp / np.exp(logT)
        z = z - z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=1))
        return float(-(z[np.arange(len(y)), y] - lse).mean())

    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded", options={"xatol": 1e-4})
    return float(np.exp(r.x))


def apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(proba, 1e-15, 1.0)) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------ pipelines
@dataclass
class ModelPipeline:
    family: str
    hp: dict[str, Any]
    feature_names: list[str]
    seed: int = P.SEED
    n_jobs: int = 8
    prep: dict[str, np.ndarray] = field(default_factory=dict)
    estimator: Any = None
    temperature: float | None = None

    # ---- preprocessing (fit on the TRAINING block only)
    def fit_preprocessing(self, X_train: np.ndarray) -> "ModelPipeline":
        if self.family == "logistic":
            lo = np.nanquantile(X_train, 0.001, axis=0)
            hi = np.nanquantile(X_train, 0.999, axis=0)
            Xw = np.clip(X_train, lo, hi)
            med = np.nanmedian(Xw, axis=0)
            iqr = np.nanquantile(Xw, 0.75, axis=0) - np.nanquantile(Xw, 0.25, axis=0)
            sd = np.nanstd(Xw, axis=0)
            scale = np.where(iqr > 0, iqr, np.where(sd > 0, sd, 1.0))
            self.prep = {"lo": lo, "hi": hi, "median": med, "scale": scale}
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.family == "logistic":
            Z = (np.clip(X, self.prep["lo"], self.prep["hi"]) - self.prep["median"]) / self.prep["scale"]
            return np.nan_to_num(Z, nan=0.0).astype("float32")
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModelPipeline":
        Z = self.transform(X)
        if self.family == "logistic":
            from sklearn.linear_model import LogisticRegression
            self.estimator = LogisticRegression(C=self.hp["C"], class_weight=self.hp.get("class_weight"), random_state=self.seed,
                                                **P.LOGISTIC_FIXED).fit(Z, y)
        elif self.family == "lightgbm":
            import lightgbm as lgb
            fx = dict(P.LGBM_FIXED)
            self.estimator = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=self.hp["n_estimators"],
                                                num_leaves=self.hp["num_leaves"], random_state=self.seed, n_jobs=self.n_jobs,
                                                deterministic=True, force_row_wise=True, verbose=-1, **fx).fit(Z, y)
        else:
            raise ValueError(self.family)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(self.transform(X))

    def predict_calibrated(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return apply_temperature(p, self.temperature) if self.temperature else p

    def importance(self) -> dict[str, float]:
        if self.family == "logistic":
            imp = np.abs(self.estimator.coef_).mean(axis=0)            # features are scaled: comparable magnitudes
        else:
            imp = self.estimator.booster_.feature_importance(importance_type="gain")
            imp = imp / max(imp.sum(), 1e-12)
        return {n: float(v) for n, v in zip(self.feature_names, imp)}

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return str(path)

    @staticmethod
    def load(path: Path) -> "ModelPipeline":
        return joblib.load(path)
