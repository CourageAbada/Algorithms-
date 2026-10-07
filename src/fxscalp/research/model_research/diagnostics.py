"""Feature diagnostics for serious candidates (development folds only): importance, permutation importance, stability, drift,
missingness, feed-boundary behaviour, month-identifiability and the pre-registered proxy flag."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fxscalp.research import folds as F
from fxscalp.research import policy
from fxscalp.research.model_research import metrics as M
from fxscalp.research.model_research import protocol as P
from fxscalp.research.model_research.experiment import Runner
from fxscalp.research.model_research.models import ModelPipeline, feature_matrix, fit_regime_config

SESSION_COLS = ("session_code", "is_asia", "is_london", "is_new_york", "is_london_new_york_overlap")
GROUP_ALIASES = {"regime": "regimes"}        # protocol ablation name -> feature group name in the manifest


def feature_groups(manifest: dict[str, Any]) -> dict[str, str]:
    return {r["name"]: r["group"] for r in manifest["features"]}


def fold_validation_matrix(runner: Runner, scenario: str, feature_config: str, fold: F.FoldSpec, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild X_val, y_val, ts_val for a stored fold model (regime thresholds are re-fit on that fold's TRAINING rows)."""
    rows = runner.fold_rows(scenario, fold)
    tr, va = runner.df.iloc[rows["train"]], runner.df.iloc[rows["val"]]
    regime_cfg = None
    if feature_config == "REGIME":
        regime_cfg = fit_regime_config({k: tr[k].to_numpy("float64") for k in ("spread_points", "session_code", "realized_vol_bps_60s")}, tr["ts"].to_numpy(), fold)
    Xva, all_names = feature_matrix(va, feature_config, runner.allowed, regime_cfg)
    keep = [all_names.index(c) for c in names]
    return Xva[:, keep], runner.y(scenario)[rows["val"]], va["ts"].to_numpy()


def permutation_importance(pipe: ModelPipeline, X: np.ndarray, y: np.ndarray, groups: dict[str, list[int]], features: list[int], *, seed: int,
                           n_rows: int = 40000, repeats: int = 3) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y), size=min(n_rows, len(y)), replace=False))
    Xs, ys = X[idx], y[idx]
    base_p = pipe.predict_calibrated(Xs)
    base = (M.from_confusion(M.confusion(ys, base_p.argmax(axis=1)))["macro_f1"], M.log_loss(ys, base_p))

    def perturb(cols: list[int]) -> tuple[float, float]:
        d_f1, d_ll = [], []
        for _ in range(repeats):
            Xp = Xs.copy()
            perm = rng.permutation(len(ys))
            Xp[:, cols] = Xs[perm][:, cols]                     # joint permutation keeps within-group structure
            p = pipe.predict_calibrated(Xp)
            d_f1.append(base[0] - M.from_confusion(M.confusion(ys, p.argmax(axis=1)))["macro_f1"])
            d_ll.append(M.log_loss(ys, p) - base[1])
        return float(np.mean(d_f1)), float(np.mean(d_ll))

    out_g = {g: dict(zip(("d_macro_f1", "d_log_loss"), perturb(c))) for g, c in groups.items()}
    out_f = {pipe.feature_names[j]: dict(zip(("d_macro_f1", "d_log_loss"), perturb([j]))) for j in features}
    return {"base_macro_f1": base[0], "base_log_loss": base[1], "groups": out_g, "features": out_f}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else float("nan")


def psi(train: np.ndarray, val: np.ndarray, bins: int = 10) -> float:
    t, v = train[np.isfinite(train)], val[np.isfinite(val)]
    if len(t) < 100 or len(v) < 100:
        return float("nan")
    edges = np.unique(np.quantile(t, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pt = np.histogram(t, edges)[0] / len(t)
    pv = np.histogram(v, edges)[0] / len(v)
    pt, pv = np.clip(pt, 1e-4, None), np.clip(pv, 1e-4, None)
    return float(np.sum((pv - pt) * np.log(pv / pt)))


def drift_and_missingness(runner: Runner, scenario: str, feature_config: str, names: list[str]) -> dict[str, Any]:
    per_fold, miss = [], []
    for fold in runner.folds:
        rows = runner.fold_rows(scenario, fold)
        tr, va = runner.df.iloc[rows["train"]], runner.df.iloc[rows["val"]]
        cols = [c for c in names if c in tr.columns]
        psis = {c: psi(tr[c].to_numpy("float64"), va[c].to_numpy("float64")) for c in cols}
        per_fold.append(psis)
        miss.append({c: (float(tr[c].isna().mean() * 100), float(va[c].isna().mean() * 100)) for c in cols})
    df = pd.DataFrame(per_fold)
    mean_psi = df.mean().sort_values(ascending=False)
    post = runner.df[runner.df["ts"] >= policy.FEED_REGIME_BOUNDARY_UTC_MS]
    pre_ref = runner.df[(runner.df["ts"] < policy.FEED_REGIME_BOUNDARY_UTC_MS) & (runner.df["ts"] >= F.day_start_utc_ms(F.date(2026, 8, 10)))]
    feed_psi = {c: psi(pre_ref[c].to_numpy("float64"), post[c].to_numpy("float64")) for c in names if c in post.columns}
    fp = pd.Series(feed_psi).sort_values(ascending=False)
    return {"mean_psi_top15": {k: round(float(v), 4) for k, v in mean_psi.head(15).items()},
            "features_mean_psi_gt_0.25": int((mean_psi > 0.25).sum()), "features_mean_psi_gt_0.10": int((mean_psi > 0.10).sum()),
            "psi_by_fold_max": [round(float(np.nanmax(list(p.values()))), 3) for p in per_fold],
            "missing_pct_train_val_top10": {c: [round(float(np.mean([m[c][0] for m in miss])), 2), round(float(np.mean([m[c][1] for m in miss])), 2)]
                                            for c in sorted(names, key=lambda c: -np.mean([m[c][1] for m in miss if c in m]))[:10]},
            "feed_boundary_psi_pre_vs_post_top10": {k: round(float(v), 4) for k, v in fp.head(10).items()},
            "feed_boundary_features_psi_gt_0.25": int((fp > 0.25).sum()),
            "feed_boundary_reference": "pre = 2026-08-10..2026-09-04 development rows; post = 2026-09-07..2026-09-11 development rows"}


def month_identifiability(df: pd.DataFrame, allowed: list[str], seed: int = P.SEED) -> dict[str, Any]:
    """Can the CORE features identify the calendar month? (day-grouped 5-fold CV; high accuracy = features carry date/regime fingerprints)."""
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    month = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.month.to_numpy() - 4          # 0..5 (Apr..Sep)
    day = (df["ts"].to_numpy() // 86_400_000)
    X = df[allowed].to_numpy("float32")
    accs, imps = [], np.zeros(len(allowed))
    for tr, te in GroupKFold(n_splits=5).split(X, month, day):
        m = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.1, subsample=0.5, subsample_freq=1, colsample_bytree=0.7,
                               random_state=seed, n_jobs=8, verbose=-1, deterministic=True, force_row_wise=True)
        m.fit(X[tr], month[tr])
        accs.append(float((m.predict(X[te]) == month[te]).mean()))
        imps += m.booster_.feature_importance("gain")
    top = np.argsort(-imps)[:10]
    maj = float(np.bincount(month).max() / len(month))
    return {"cv_accuracy_mean": float(np.mean(accs)), "cv_accuracy_per_fold": accs, "majority_rate": maj, "chance_uniform": 1 / 6,
            "top_month_identifying_features": [allowed[i] for i in top]}
