"""Deterministic, never-overwriting experiment runner for Phase 2B model research (development folds only).

Experiment id = hash of: dataset freeze, feature-manifest hash, label spec (+ cost scenario), fold specification, model family,
hyperparameters, seed, preprocessing / calibration configuration, feature configuration (+ explicit column subset for
controls/ablations), decision grid, protocol hash, matrix id. Each experiment stores configuration, per-fold metrics, predictions,
feature importance, training duration, memory, library versions and the Git commit under ``<root>/<id>/`` and is never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from fxscalp.research import folds as F
from fxscalp.research import policy, spec
from fxscalp.research.labels import LabelSpec
from fxscalp.research.model_research import metrics as M
from fxscalp.research.model_research import protocol as P
from fxscalp.research.model_research.frozen import git_state
from fxscalp.research.model_research.models import (ModelPipeline, apply_temperature, feature_matrix, fit_regime_config,
                                                    fit_temperature)

warnings.filterwarnings("ignore", message="'penalty' was deprecated", category=FutureWarning)   # sklearn 1.8+: penalty=l2 is the default

BASELINE_FAMILIES = ("majority", "stratified_random", "momentum", "mean_reversion")


class ExperimentExists(RuntimeError):
    """An experiment with this id already exists; experiments are never overwritten."""


@dataclass(frozen=True)
class ExpConfig:
    kind: str                       # model | baseline | control | ablation
    feature_config: str             # CORE | CONDITIONAL | REGIME | "-" (baselines)
    family: str
    hp: tuple[tuple[str, Any], ...]
    scenario: str
    seed: int = P.SEED
    columns: tuple[str, ...] | None = None      # explicit subset of the configuration's columns (controls / ablations)
    tag: str = ""

    def hp_dict(self) -> dict[str, Any]:
        return dict(self.hp)


def hp_tuple(hp: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((k, v) for k, v in hp.items() if k != "family"))


def canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, default=str, separators=(",", ":"))


def context(matrix_id: str, frozen: dict[str, Any]) -> dict[str, Any]:
    idx = json.loads((spec.SPEC_DIR / "spec_index.json").read_text(encoding="utf-8"))
    return {"freeze_id": frozen["raw_dataset_freeze"], "spec_hash": frozen["spec_hash"],
            "feature_manifest_sha": idx["files"]["feature_set_v1.json"], "folds_sha": idx["files"]["folds_v1.json"],
            "label": LabelSpec("tb", 60).to_dict(), "protocol_hash": frozen["protocol_hash"], "matrix_id": matrix_id,
            "preprocessing": P.PREPROCESSING, "calibration": P.CALIBRATION, "grid_step_s": P.GRID_STEP_S,
            "models_fixed": {"lightgbm": P.LGBM_FIXED, "logistic": P.LOGISTIC_FIXED}}


def experiment_id(cfg: ExpConfig, ctx: dict[str, Any]) -> str:
    return "exp-" + hashlib.sha256(canon({"cfg": asdict(cfg), "ctx": ctx}).encode()).hexdigest()[:16]


def env_info() -> dict[str, Any]:
    import lightgbm
    import scipy
    import sklearn
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "scipy": scipy.__version__}


def peak_rss_mb() -> float:
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2**20, 1)


class ExperimentStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def dir(self, eid: str) -> Path:
        return self.root / eid

    def exists(self, eid: str) -> bool:
        return (self.dir(eid) / "result.json").exists()

    def load(self, eid: str) -> dict[str, Any]:
        return json.loads((self.dir(eid) / "result.json").read_text(encoding="utf-8"))

    def predictions(self, eid: str) -> pd.DataFrame:
        return pd.read_parquet(self.dir(eid) / "predictions.parquet")

    def save(self, eid: str, result: dict[str, Any], preds: pd.DataFrame) -> None:
        d = self.dir(eid)
        if (d / "result.json").exists():
            raise ExperimentExists(eid)
        d.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(d / "predictions.parquet", index=False)
        (d / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")   # written LAST = commit marker

    def fail(self, eid: str, cfg: ExpConfig, exc: BaseException) -> None:
        d = self.dir(eid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "failed.json").write_text(json.dumps({"config": asdict(cfg), "error": repr(exc), "trace": traceback.format_exc()}, indent=1, default=str),
                                       encoding="utf-8")


# ------------------------------------------------------------------------------------------------- fold data
class Runner:
    def __init__(self, df: pd.DataFrame, matrix_id: str, frozen: dict[str, Any], allowed: list[str], store: ExperimentStore,
                 n_jobs: int = 8):
        F.assert_development_only(df["ts"].to_numpy(), what="experiment matrix")
        self.df, self.matrix_id, self.frozen, self.allowed, self.store, self.n_jobs = df, matrix_id, frozen, allowed, store, n_jobs
        self.ctx = context(matrix_id, frozen)
        self.folds = F.default_folds()
        self.git, self.env = git_state(), env_info()

    # ---- row sets (purge + embargo + segments are already in labels / folds)
    def fold_rows(self, scenario: str, fold: F.FoldSpec) -> dict[str, np.ndarray]:
        df = self.df
        ts, le, st = df["ts"].to_numpy(), df[f"le_{scenario}"].to_numpy(), df[f"st_{scenario}"].to_numpy()
        ok = st == 0
        tr = ok & F.train_mask(fold, ts, le)
        va = ok & F.val_mask(fold, ts, le)
        tr_idx = np.flatnonzero(tr)
        t_c = np.quantile(ts[tr_idx], 0.8)                               # chronological calibration split inside the TRAINING block
        fit = tr & (ts < t_c) & (le <= t_c - F.EMBARGO_S * 1000)
        cal = tr & (ts >= t_c)
        assert ts[tr].max() < fold.val_start_ms and (le[tr].max() <= fold.val_start_ms - F.EMBARGO_S * 1000)
        assert fold.val_start_ms <= ts[va].min() and ts[va].max() < fold.val_end_ms
        return {"train": tr_idx, "fit": np.flatnonzero(fit), "cal": np.flatnonzero(cal), "val": np.flatnonzero(va)}

    def y(self, scenario: str) -> np.ndarray:
        return self.df[f"y_{scenario}"].to_numpy().astype("int64")

    # ---- model experiments for one (scenario, feature config): all folds, many hyperparameter sets
    def run_models(self, scenario: str, feature_config: str, hps: list[dict[str, Any]], *, kind: str = "model", columns: tuple[str, ...] | None = None,
                   tag: str = "", log: Any = print) -> list[str]:
        cfgs = [ExpConfig(kind, feature_config, h["family"], hp_tuple(h), scenario, P.SEED, columns, tag) for h in hps]
        ids = [experiment_id(c, self.ctx) for c in cfgs]
        todo = [(c, i) for c, i in zip(cfgs, ids) if not self.store.exists(i)]
        if not todo:
            return ids
        y_all = self.y(scenario)
        per: dict[str, list[dict[str, Any]]] = {i: [] for _, i in todo}
        preds: dict[str, list[pd.DataFrame]] = {i: [] for _, i in todo}
        t_block = time.perf_counter()
        for fold in self.folds:
            rows = self.fold_rows(scenario, fold)
            tr_df, va_df = self.df.iloc[rows["train"]], self.df.iloc[rows["val"]]
            regime_cfg = None
            if feature_config == "REGIME":
                regime_cfg = fit_regime_config({k: tr_df[k].to_numpy("float64") for k in ("spread_points", "session_code", "realized_vol_bps_60s")},
                                               tr_df["ts"].to_numpy(), fold)
            Xtr, names = feature_matrix(tr_df, feature_config, self.allowed, regime_cfg)
            Xva, _ = feature_matrix(va_df, feature_config, self.allowed, regime_cfg)
            if columns is not None:
                keep = [names.index(c) for c in columns]
                Xtr, Xva, names = Xtr[:, keep], Xva[:, keep], [names[k] for k in keep]
            pos = {g: k for k, g in enumerate(rows["train"])}
            fit_p = np.array([pos[g] for g in rows["fit"]])
            cal_p = np.array([pos[g] for g in rows["cal"]])
            y_tr, y_va = y_all[rows["train"]], y_all[rows["val"]]
            for cfg, eid in todo:
                t0 = time.perf_counter()
                pipe = ModelPipeline(cfg.family, cfg.hp_dict(), names, cfg.seed, self.n_jobs).fit_preprocessing(Xtr)
                pipe.fit(Xtr[fit_p], y_tr[fit_p])
                p_cal_raw = pipe.predict_proba(Xtr[cal_p])
                pipe.temperature = fit_temperature(p_cal_raw, y_tr[cal_p])
                p_unc = pipe.predict_proba(Xva)
                p_cal = apply_temperature(p_unc, pipe.temperature)
                dur = time.perf_counter() - t0
                mpath = pipe.save(self.store.dir(eid) / f"model_{fold.name}.joblib")
                rec = fold_record(fold, rows, y_va, va_df["ts"].to_numpy(), p_unc, p_cal)
                rec.update({"temperature": pipe.temperature, "duration_s": round(dur, 2), "peak_rss_mb": peak_rss_mb(), "model_file": Path(mpath).name,
                            "importance": pipe.importance(), "feature_names": names,
                            "regime_config": regime_cfg.to_dict() if regime_cfg else None})
                per[eid].append(rec)
                preds[eid].append(pred_frame(fold.name, va_df["ts"].to_numpy(), y_va, p_unc, p_cal))
            log(f"  [{scenario} {feature_config}{' '+tag if tag else ''}] {fold.name} done ({len(todo)} experiments) {time.perf_counter() - t_block:.0f}s")
        for cfg, eid in todo:
            self.store.save(eid, assemble(cfg, eid, self.ctx, per[eid], self.git, self.env, protocol_hash=self.frozen["protocol_hash"]),
                            pd.concat(preds[eid], ignore_index=True))
        return ids

    # ---- baselines
    def run_baselines(self, scenario: str) -> list[str]:
        out = []
        y_all = self.y(scenario)
        for fam in BASELINE_FAMILIES:
            cfg = ExpConfig("baseline", "-", fam, (), scenario, P.SEED, None, "")
            eid = experiment_id(cfg, self.ctx)
            out.append(eid)
            if self.store.exists(eid):
                continue
            per, preds = [], []
            for k, fold in enumerate(self.folds):
                rows = self.fold_rows(scenario, fold)
                y_tr, y_va = y_all[rows["train"]], y_all[rows["val"]]
                va_df = self.df.iloc[rows["val"]]
                prior = np.bincount(y_tr, minlength=3) / len(y_tr)
                n = len(y_va)
                if fam == "majority":
                    yhat = np.full(n, int(prior.argmax()))
                elif fam == "stratified_random":
                    yhat = np.random.default_rng(P.SEED + k).choice(3, size=n, p=prior)
                else:
                    r = va_df["ret_bps_60s"].to_numpy("float64")
                    yhat = np.where(r > 0, 2, np.where(r < 0, 0, 1))
                    if fam == "mean_reversion":
                        yhat = 2 - yhat
                p = np.tile(prior, (n, 1))
                rec = fold_record(fold, rows, y_va, va_df["ts"].to_numpy(), p, p, yhat_override=yhat,
                                  with_prob=fam in ("majority", "stratified_random"))
                rec.update({"duration_s": 0.0, "peak_rss_mb": peak_rss_mb(), "prior": prior.tolist()})
                per.append(rec)
                preds.append(pred_frame(fold.name, va_df["ts"].to_numpy(), y_va, p, p, yhat))
            self.store.save(eid, assemble(cfg, eid, self.ctx, per, self.git, self.env, protocol_hash=self.frozen["protocol_hash"]),
                            pd.concat(preds, ignore_index=True))
        return out


def pred_frame(fold: str, ts: np.ndarray, y: np.ndarray, p_unc: np.ndarray, p_cal: np.ndarray, yhat: np.ndarray | None = None) -> pd.DataFrame:
    d = {"fold": fold, "ts": ts, "y": y.astype("int8")}
    for k in range(3):
        d[f"u{k}"] = p_unc[:, k].astype("float32")
        d[f"c{k}"] = p_cal[:, k].astype("float32")
    d["pred"] = (p_cal.argmax(axis=1) if yhat is None else yhat).astype("int8")
    return pd.DataFrame(d)


def fold_record(fold: F.FoldSpec, rows: dict[str, np.ndarray], y_va: np.ndarray, ts_va: np.ndarray, p_unc: np.ndarray, p_cal: np.ndarray, *,
                yhat_override: np.ndarray | None = None, with_prob: bool = True) -> dict[str, Any]:
    yh_u = p_unc.argmax(axis=1) if yhat_override is None else yhat_override
    yh_c = p_cal.argmax(axis=1) if yhat_override is None else yhat_override
    nonov = (ts_va // 1000) % P.NONOVERLAP_STEP_S == 0
    post = ts_va >= policy.FEED_REGIME_BOUNDARY_UTC_MS
    rec = {"fold": fold.name, "n_train": int(len(rows["train"])), "n_fit": int(len(rows["fit"])), "n_cal": int(len(rows["cal"])),
           "n_val": int(len(y_va)),
           "uncalibrated": M.full_metrics(y_va, yh_u, p_unc if with_prob else None),
           "calibrated": M.full_metrics(y_va, yh_c, p_cal if with_prob else None),
           "nonoverlap_60s": M.full_metrics(y_va[nonov], yh_c[nonov], p_cal[nonov] if with_prob else None),
           "selectivity": M.selectivity(y_va, p_cal, tuple(P.DECISION_POLICY["selectivity_research_preregistered"]["max_prob_thresholds"])) if with_prob else None}
    if post.any():
        rec["feed_split"] = {"pre": M.full_metrics(y_va[~post], yh_c[~post], p_cal[~post] if with_prob else None),
                             "post": M.full_metrics(y_va[post], yh_c[post], p_cal[post] if with_prob else None)}
    return rec


def assemble(cfg: ExpConfig, eid: str, ctx: dict[str, Any], per_fold: list[dict[str, Any]], git: dict[str, Any], env: dict[str, Any], *,
             protocol_hash: str) -> dict[str, Any]:
    def agg(key: str, sub: str = "calibrated") -> dict[str, float]:
        v = np.array([r[sub][key] for r in per_fold if key in r[sub]], dtype=float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "per_fold": v.tolist()} if len(v) else {}
    keys = ("macro_f1", "balanced_accuracy", "accuracy", "log_loss", "brier", "ece_top")
    return {"experiment_id": eid, "config": asdict(cfg), "context": ctx, "protocol_hash": protocol_hash, "git": git, "env": env,
            "created_utc": pd.Timestamp.now("UTC").isoformat(), "folds": per_fold,
            "aggregate_calibrated": {k: agg(k) for k in keys}, "aggregate_uncalibrated": {k: agg(k, "uncalibrated") for k in keys},
            "total_duration_s": round(sum(r["duration_s"] for r in per_fold), 2),
            "peak_rss_mb": max(r["peak_rss_mb"] for r in per_fold)}
