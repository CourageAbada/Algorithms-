"""Phase 2B model-research PROTOCOL v1 - written and committed BEFORE any model is trained.

Everything that could be tuned after seeing results is fixed here: the decision grid, feature configurations, model families,
the (small) hyperparameter search spaces, calibration, metrics, the block-bootstrap, the selection rule, the leakage/proxy
rule, the ablation list and the verdict rule. ``protocol_manifest()`` is hashed; every experiment records that hash. A change
requires a new protocol version (and, if it touches the frozen research spec, a new spec version).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any

PROTOCOL_VERSION = "model_research_protocol/1"
EXPECTED_SPEC_HASH = "57fb49f5eb10c5f0e8fd3a3c82c1c1c39603ee915be3cf510286c35475331845"
EXPECTED_FREEZE_ID = "freeze-48779c44e3f2999274a2"
PRIMARY_LABEL = "tb_H60"
SCENARIOS = ("C0_spread_only", "C1_moderate", "C2_pessimistic")
SEED = 7
CLASS_NAMES = ("SHORT", "NO_TRADE", "LONG")          # class index 0, 1, 2  (label -1, 0, +1)
GRID_STEP_S = 30                                      # decision grid: bar ends with (ts // 1000) % 30 == 0
BLOCK_S = 600                                         # bootstrap block = 10 x the 60 s label horizon
BOOTSTRAP_B = 1000
NONOVERLAP_STEP_S = 60                                # robustness: non-overlapping 60 s labels

FEATURE_CONFIGS = {
    "CORE": "the 108 ALLOWED features",
    "CONDITIONAL": "108 ALLOWED + 8 session running aggregates, each masked to NaN unless session_so_far_complete == 1",
    "REGIME": "108 ALLOWED + spread_regime_code + volatility_regime_code recomputed per fold from reference quantiles fitted "
              "on that fold's TRAINING rows only (calibrate_reference_quantiles; spread (0.95, 0.995), volatility (0.20, 0.80, 0.98), "
              "per session code)",
}

LOGISTIC_SPACE = {"C": [0.01, 0.1, 1.0], "class_weight": [None, "balanced"]}
LGBM_SPACE = {"num_leaves": [15, 31], "n_estimators": [150, 400]}
LGBM_FIXED = {"learning_rate": 0.05, "min_child_samples": 500, "subsample": 0.7, "subsample_freq": 1, "colsample_bytree": 0.7,
              "reg_lambda": 10.0, "class_weight": "balanced", "max_bin": 255}
LOGISTIC_FIXED = {"penalty": "l2", "solver": "lbfgs", "max_iter": 300, "tol": 1e-3}

PREPROCESSING = {
    "logistic": {"impute": "training-block median (NaN -> 0 after scaling)", "scale": "median/IQR (CausalScaler fit on the training block)",
                 "winsorize": "clip to training-block 0.1% / 99.9% quantiles per feature"},
    "lightgbm": {"impute": "none (native missing handling)", "scale": "none", "winsorize": "none"},
}
CALIBRATION = {"method": "temperature scaling (single parameter T on log-probabilities, NLL-minimised)",
               "fit_data": "the LAST 20% (by time) of each fold's TRAINING rows; the model is fit on the first 80% minus an 1,800 s embargo "
                           "and a 60 s+latency purge; the SAME 80%-model is reported uncalibrated and calibrated",
               "validation_leakage": "none: validation rows never enter model, preprocessing, regime thresholds or calibrator"}
DECISION_POLICY = {"rule": "argmax of calibrated class probabilities", "thresholds": "none optimised; no PnL objective",
                   "selectivity_research_preregistered": {"max_prob_thresholds": [0.40, 0.50, 0.60],
                                                           "report": "coverage and macro-F1 / accuracy among accepted rows (descriptive calibration/selectivity study only)"}}
BASELINES = {
    "majority": "predict the training-fold majority class; probabilities = training class priors",
    "stratified_random": f"sample classes from the training priors with seed {SEED}; probabilities = training priors",
    "momentum": "LONG if ret_bps_60s > 0, SHORT if < 0, NO_TRADE if == 0 (causal trailing 60 s return)",
    "mean_reversion": "the opposite of momentum",
    "prior_probabilities": "log-loss/Brier reference only: training-fold class priors",
}
METRICS = ["balanced_accuracy", "macro_f1", "per_class_precision_recall_f1", "confusion_matrix", "accuracy (secondary)", "log_loss",
           "multiclass_brier", "top_label_ece (15 equal-mass bins)", "reliability tables", "predicted-class frequency", "class distribution"]

CONTROLS = {"session_only": "session_code, is_asia, is_london, is_new_york, is_london_new_york_overlap",
            "spread_only": "all features of group 'spread'",
            "purpose": "detect calendar/feed/cost proxies: same pipeline, LightGBM with the leading hyperparameters"}
ABLATIONS = ["spread", "session", "mtf", "regime", "momentum", "mean_reversion", "structure", "micro"]   # feature groups removed one at a time
PROBES = {"month_identifiability": "predict the calendar month from CORE features with day-grouped 5-fold CV (descriptive)",
          "importance": "built-in + permutation importance (macro-F1 and log-loss deltas, 3 repeats, <=40,000 validation rows) for the leading logistic "
                        "and leading LightGBM candidates; stability = Spearman rank correlation between folds",
          "drift": "population stability index of every feature, training block vs validation block (10 training-quantile bins)",
          "feed_boundary": "fold F5 validation split into pre (2026-08-24..09-04) and post (2026-09-07..09-11) rows"}

SELECTION_RULE = {
    "unit": "candidate = (feature_config, model_family, hyperparameters); each trained separately for C0, C1 and C2 labels",
    "gates (all must hold, evaluated on calibrated probabilities)": {
        "G1_beats_baselines": "mean-over-scenarios fold macro-F1 > the best naive baseline's, in >= 4 of 5 folds",
        "G2_probabilistic_skill": "fold log-loss < training-prior log-loss in >= 4 of 5 folds",
        "G3_all_classes_predicted": "mean validation recall of every class >= 0.05",
        "G4_no_proxy_flag": "no leakage/proxy flag from the rule below"},
    "score": "S = M - D - E; M = mean over scenarios and folds of macro-F1; D = mean over scenarios of the SD over folds of macro-F1; "
             "E = mean over scenarios and folds of top-label ECE (calibrated)",
    "choice": "highest S among gate-passers; if candidates are within 0.005 of the best S, prefer the simpler: logistic < LightGBM, "
              "CORE < CONDITIONAL < REGIME, fewer trees, fewer leaves, larger C-regularisation",
    "no_gate_passer": "report NO CANDIDATE; the verdict becomes NO RELIABLE SIGNAL DETECTED",
    "proxy_flag": "set if (a) a control model (session_only or spread_only) reaches >= 80% of the candidate's mean macro-F1 lift over the "
                  "best baseline in the same scenario, or (b) the session + spread feature groups carry >= 50% of the candidate's total "
                  "group-permutation log-loss increase",
    "never_used": ["final holdout", "PnL", "Sharpe", "profit factor", "any threshold optimised for returns"]}

VERDICT_RULE = {
    "paired_block_bootstrap": f"blocks of {BLOCK_S} s resampled within each fold, B={BOOTSTRAP_B}, seed {SEED}, pooled predictions; paired with the best baseline",
    "effect_floors": {"delta_macro_f1": 0.02, "delta_log_loss_nats": 0.005},
    "PREDICTIVE SIGNAL DETECTED": "for the selected candidate in ALL THREE scenarios: lower 95% CI of delta macro-F1 vs the best baseline > 0 and "
                                  "point >= 0.02; lower 95% CI of the log-loss improvement over the prior > 0 and point >= 0.005; the candidate beats the "
                                  "best baseline in >= 4 of 5 folds per scenario; no proxy flag",
    "WEAK/UNSTABLE SIGNAL": "the pooled point estimates beat the best baseline (macro-F1 and log loss) in >= 2 of 3 scenarios but at least one "
                            "'DETECTED' criterion fails",
    "NO RELIABLE SIGNAL DETECTED": "otherwise (including no gate-passing candidate)",
    "holdout_evaluation_justified": "only for PREDICTIVE SIGNAL DETECTED; for WEAK/UNSTABLE the pre-registered answer is NO (any evaluation would "
                                    "need an explicit decision by the owner knowing it is not supported); for NO RELIABLE SIGNAL the answer is NO",
    "economic_claims": "none: predictive classification performance only; costs C1/C2 are hypothetical; profitability is UNVERIFIED"}


def hp_grid() -> list[dict[str, Any]]:
    out = []
    for c, cw in itertools.product(LOGISTIC_SPACE["C"], LOGISTIC_SPACE["class_weight"]):
        out.append({"family": "logistic", "C": c, "class_weight": cw})
    for nl, ne in itertools.product(LGBM_SPACE["num_leaves"], LGBM_SPACE["n_estimators"]):
        out.append({"family": "lightgbm", "num_leaves": nl, "n_estimators": ne})
    return out


def protocol_manifest() -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION, "written_before_training": True, "frozen_spec_hash": EXPECTED_SPEC_HASH,
        "frozen_freeze_id": EXPECTED_FREEZE_ID, "primary_label": PRIMARY_LABEL, "cost_scenarios": list(SCENARIOS),
        "final_holdout": "LOCKED: not read, not evaluated, not inspected in this phase",
        "class_index": {n: i for i, n in enumerate(CLASS_NAMES)}, "seed": SEED,
        "sampling": {"decision_grid_s": GRID_STEP_S, "rule": "bar ends with (ts//1000) % 30 == 0, row_valid == 1, label status OK",
                     "note": "1-s rows are strongly autocorrelated; a 30 s grid keeps ~1/30 of rows with 50% overlap between neighbouring 60 s labels. "
                             "Dependence is handled by block bootstrap and a non-overlapping 60 s robustness evaluation."},
        "folds": "the five frozen expanding folds F1..F5 (folds_v1.json), purge/embargo as frozen",
        "feature_configs": FEATURE_CONFIGS, "models": {"logistic": {"search": LOGISTIC_SPACE, "fixed": LOGISTIC_FIXED},
                                                       "lightgbm": {"search": LGBM_SPACE, "fixed": LGBM_FIXED,
                                                                    "determinism": "deterministic=True, force_row_wise=True, fixed seed"}},
        "third_family": "NOT used: no justification beyond logistic + gradient boosting for a first causal-signal test",
        "n_candidates_per_scenario": len(hp_grid()) * len(FEATURE_CONFIGS), "hyperparameter_grid": hp_grid(),
        "preprocessing": PREPROCESSING, "calibration": CALIBRATION, "decision_policy": DECISION_POLICY, "baselines": BASELINES,
        "metrics": METRICS, "controls": CONTROLS, "ablations": ABLATIONS, "probes": PROBES,
        "selection_rule": SELECTION_RULE, "verdict_rule": VERDICT_RULE,
        "search_policy": "this grid is final; it is never expanded after seeing results (a larger search needs a new protocol version)",
        "block_bootstrap": {"block_s": BLOCK_S, "B": BOOTSTRAP_B, "basis": "10 x the label horizon (60 s); the autocorrelation of the label series "
                                                                           "is reported as a check, not used to choose the block"},
    }


def protocol_hash() -> str:
    return hashlib.sha256(json.dumps(protocol_manifest(), sort_keys=True, default=str).encode()).hexdigest()
