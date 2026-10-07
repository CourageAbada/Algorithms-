"""Candidate selection and verdict: implements the PRE-REGISTERED rule in protocol.SELECTION_RULE / VERDICT_RULE exactly.

Reads stored development-fold experiment results only. Predictive classification metrics only - no PnL of any kind.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fxscalp.research.model_research import metrics as M
from fxscalp.research.model_research import protocol as P
from fxscalp.research.model_research.experiment import BASELINE_FAMILIES, ExperimentStore

CONFIG_RANK = {"CORE": 0, "CONDITIONAL": 1, "REGIME": 2}
FAMILY_RANK = {"logistic": 0, "lightgbm": 1}
TIE = 0.005


def load_results(store: ExperimentStore) -> list[dict[str, Any]]:
    out = []
    for d in sorted(store.root.iterdir()) if store.root.exists() else []:
        if (d / "result.json").exists():
            out.append(store.load(d.name))
    return out


def cand_key(r: dict[str, Any]) -> tuple:
    c = r["config"]
    return (c["feature_config"], c["family"], tuple(map(tuple, c["hp"])), tuple(c["columns"]) if c.get("columns") else None, c.get("tag", ""))


def split(results: list[dict[str, Any]]) -> tuple[dict, dict]:
    """-> (candidates[key][scenario] = result, baselines[family][scenario] = result) for kind == model / baseline."""
    cands: dict[tuple, dict[str, Any]] = {}
    base: dict[str, dict[str, Any]] = {}
    for r in results:
        c = r["config"]
        if c["kind"] == "baseline":
            base.setdefault(c["family"], {})[c["scenario"]] = r
        elif c["kind"] == "model":
            cands.setdefault(cand_key(r), {})[c["scenario"]] = r
    return cands, base


def fold_matrix(r: dict[str, Any], key: str, sub: str = "calibrated") -> np.ndarray:
    return np.array([f[sub][key] for f in r["folds"]], dtype=float)


def baseline_refs(base: dict) -> dict[str, dict[str, np.ndarray]]:
    """per scenario: best naive macro-F1 per fold and the training-prior log loss per fold."""
    refs = {}
    for s in P.SCENARIOS:
        f1 = np.max([fold_matrix(base[b][s], "macro_f1") for b in BASELINE_FAMILIES], axis=0)
        refs[s] = {"best_baseline_macro_f1": f1, "prior_log_loss": fold_matrix(base["majority"][s], "log_loss"),
                   "best_baseline_name": [BASELINE_FAMILIES[i] for i in np.argmax([fold_matrix(base[b][s], "macro_f1") for b in BASELINE_FAMILIES], axis=0)]}
    return refs


def evaluate(cands: dict, base: dict) -> pd.DataFrame:
    refs = baseline_refs(base)
    rows = []
    for key, by_s in cands.items():
        if set(by_s) != set(P.SCENARIOS):
            continue
        f1 = np.array([fold_matrix(by_s[s], "macro_f1") for s in P.SCENARIOS])             # (3, 5)
        ll = np.array([fold_matrix(by_s[s], "log_loss") for s in P.SCENARIOS])
        ece = np.array([fold_matrix(by_s[s], "ece_top") for s in P.SCENARIOS])
        bacc = np.array([fold_matrix(by_s[s], "balanced_accuracy") for s in P.SCENARIOS])
        rec = np.array([[f["calibrated"]["recall"] for f in by_s[s]["folds"]] for s in P.SCENARIOS])   # (3, 5, 3)
        bf1 = np.array([refs[s]["best_baseline_macro_f1"] for s in P.SCENARIOS])
        pll = np.array([refs[s]["prior_log_loss"] for s in P.SCENARIOS])
        g1_folds = int((f1.mean(axis=0) > bf1.mean(axis=0)).sum())
        g2_folds = int((ll.mean(axis=0) < pll.mean(axis=0)).sum())
        min_recall = float(rec.mean(axis=(0, 1)).min())
        M_, D_, E_ = float(f1.mean()), float(f1.std(axis=1, ddof=1).mean()), float(ece.mean())
        rows.append({"key": key, "feature_config": key[0], "family": key[1], "hp": dict(key[2]), "M_macro_f1": M_, "D_fold_sd": D_, "E_ece": E_,
                     "S": M_ - D_ - E_, "mean_balanced_acc": float(bacc.mean()), "mean_log_loss": float(ll.mean()),
                     "baseline_macro_f1": float(bf1.mean()), "prior_log_loss": float(pll.mean()),
                     "G1_folds_beating_baseline": g1_folds, "G2_folds_with_skill": g2_folds, "G3_min_class_recall": min_recall,
                     "G1": g1_folds >= 4, "G2": g2_folds >= 4, "G3": min_recall >= 0.05,
                     "recall_by_class": rec.mean(axis=(0, 1)).tolist(), "per_scenario_macro_f1": f1.mean(axis=1).tolist(),
                     "per_fold_macro_f1": f1.mean(axis=0).tolist()})
    df = pd.DataFrame(rows)
    df["gates_123"] = df["G1"] & df["G2"] & df["G3"]
    return df.sort_values("S", ascending=False).reset_index(drop=True)


def simplicity(row: pd.Series) -> tuple:
    hp = row["hp"]
    return (FAMILY_RANK[row["family"]], CONFIG_RANK[row["feature_config"]], hp.get("n_estimators", 0), hp.get("num_leaves", 0),
            -hp.get("C", 0.0))


def tie_break(df: pd.DataFrame) -> pd.Series | None:
    """Among the candidates within TIE of the best S, pick the simplest (pre-registered order)."""
    if df.empty:
        return None
    best = df["S"].max()
    close = df[df["S"] >= best - TIE]
    return min((r for _, r in close.iterrows()), key=simplicity)


def bootstrap_vs_baseline(store: ExperimentStore, cand: dict[str, dict], base: dict, *, B: int = P.BOOTSTRAP_B, seed: int = P.SEED,
                          prior_correct: bool = False) -> dict[str, Any]:
    """Paired block bootstrap per scenario: candidate vs the best baseline (by pooled macro-F1) and vs the training prior (log loss)."""
    out = {}
    for s in P.SCENARIOS:
        pc = store.predictions(cand[s]["experiment_id"])
        priors = [f["prior"] for f in base["majority"][s]["folds"]] if prior_correct else None
        blocks_c = _fold_blocks(pc, use_cal=True, priors=priors)
        # best baseline by pooled macro-F1 point estimate
        best_name, best_blocks, best_f1 = None, None, -1.0
        for b in BASELINE_FAMILIES:
            bb = _fold_blocks(store.predictions(base[b][s]["experiment_id"]), use_cal=False)
            f1 = M.pooled_point(bb)["macro_f1"]
            if f1 > best_f1:
                best_name, best_blocks, best_f1 = b, bb, f1
        prior_blocks = _fold_blocks(store.predictions(base["majority"][s]["experiment_id"]), use_cal=False)
        vs_best = M.paired_block_bootstrap(blocks_c, best_blocks, B=B, seed=seed)
        vs_prior = M.paired_block_bootstrap(blocks_c, prior_blocks, B=B, seed=seed)
        per_fold_beats = sum(M.pooled_point([bc])["macro_f1"] > M.pooled_point([bb])["macro_f1"] for bc, bb in zip(blocks_c, best_blocks))
        out[s] = {"best_baseline": best_name, "pooled_candidate": M.pooled_point(blocks_c), "pooled_best_baseline": M.pooled_point(best_blocks),
                  "pooled_prior": M.pooled_point(prior_blocks), "macro_f1_vs_best_baseline": vs_best["macro_f1"],
                  "balanced_accuracy_vs_best_baseline": vs_best["balanced_accuracy"], "log_loss_vs_prior": vs_prior["log_loss"],
                  "folds_beating_best_baseline": int(per_fold_beats)}
    return out


def prior_corrected(p_unc: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Elkan prior-shift correction for a class-weighted model: p_true ∝ p_weighted * prior (the weighted model implies a uniform prior)."""
    q = np.asarray(p_unc, dtype="float64") * np.asarray(prior, dtype="float64")
    return q / q.sum(axis=1, keepdims=True)


def _fold_blocks(pred: pd.DataFrame, use_cal: bool, priors: list[list[float]] | None = None) -> list[np.ndarray]:
    out = []
    for i, f in enumerate(sorted(pred["fold"].unique())):
        d = pred[pred["fold"] == f].sort_values("ts", kind="stable")
        p = d[[("c" if use_cal else "u") + str(k) for k in range(3)]].to_numpy("float64")
        if priors is not None:      # EXPLORATORY: keep the decisions, score the probabilities after prior-shift correction
            p = prior_corrected(d[[f"u{k}" for k in range(3)]].to_numpy("float64"), np.array(priors[i]))
        yh = d["pred"].to_numpy("int64")
        out.append(M.block_counts(d["ts"].to_numpy(), d["y"].to_numpy("int64"), yh, p, P.BLOCK_S))
    return out


def verdict(boot: dict[str, Any], proxy_flag: bool) -> dict[str, Any]:
    floors = P.VERDICT_RULE["effect_floors"]
    crit: dict[str, Any] = {}
    detected = True
    pos_scen = 0
    for s, b in boot.items():
        f1d, lld = b["macro_f1_vs_best_baseline"]["diff"], b["log_loss_vs_prior"]["diff"]
        imp_ll_lo, imp_ll_pt = -lld["hi95"], -lld["mean"]
        c = {"d_macro_f1": f1d, "d_macro_f1_ok": bool(f1d["lo95"] > 0 and b["pooled_candidate"]["macro_f1"] - b["pooled_best_baseline"]["macro_f1"] >= floors["delta_macro_f1"]),
             "logloss_improvement_vs_prior": {"point": b["pooled_prior"]["log_loss"] - b["pooled_candidate"]["log_loss"], "lo95": imp_ll_lo},
             "d_logloss_ok": bool(imp_ll_lo > 0 and b["pooled_prior"]["log_loss"] - b["pooled_candidate"]["log_loss"] >= floors["delta_log_loss_nats"]),
             "folds_beating_best_baseline": b["folds_beating_best_baseline"], "folds_ok": b["folds_beating_best_baseline"] >= 4}
        crit[s] = c
        detected &= c["d_macro_f1_ok"] and c["d_logloss_ok"] and c["folds_ok"]
        pos_scen += int(b["pooled_candidate"]["macro_f1"] > b["pooled_best_baseline"]["macro_f1"] and b["pooled_candidate"]["log_loss"] < b["pooled_prior"]["log_loss"])
    detected &= not proxy_flag
    if detected:
        v = "PREDICTIVE SIGNAL DETECTED"
    elif pos_scen >= 2:
        v = "WEAK/UNSTABLE SIGNAL"
    else:
        v = "NO RELIABLE SIGNAL DETECTED"
    return {"verdict": v, "criteria_by_scenario": crit, "proxy_flag": proxy_flag, "scenarios_with_positive_point_estimates": pos_scen,
            "holdout_evaluation_justified": v == "PREDICTIVE SIGNAL DETECTED"}
