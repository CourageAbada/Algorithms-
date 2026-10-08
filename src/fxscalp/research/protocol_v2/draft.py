"""Phase 2B PROTOCOL v2 - DRAFT (NOT ACTIVE). Nothing here trains, scores or evaluates anything.

v2 is a REVISED protocol informed by what Protocol v1 revealed on the development data. It is NOT an independent confirmatory
study. The configuration files are drafts for review; ``assert_active`` refuses to hand them to any runner until an explicit
activation file (``protocol_v2_ACTIVATION.json``, created only after the owner's authorization) exists. The frozen research
specification, the five folds, purge/embargo and the LOCKED final holdout are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL_V2_VERSION = "model_research_protocol/2-draft"
DRAFT_DIR = Path(__file__).resolve().parents[4] / "research" / "phase2b" / "protocol_v2_draft"
ACTIVATION_FILE = DRAFT_DIR / "protocol_v2_ACTIVATION.json"
FILES = ("protocol_v2_draft.json", "calibration_v2.json", "tasks_v2.json", "evaluation_v2.json", "fingerprint_tests_v2.json", "cost_v2.json")

#: Month-fingerprint features from the v1 month-identifiability probe (top-10 by gain). Chosen from that probe, disclosed as v1-informed.
FINGERPRINT_FEATURES = ["relative_spread_bps", "spread_points", "directional_persistence_60s", "realized_vol_bps_300s", "atr14_bps_300s",
                        "tick_count_60s", "realized_vol_bps_60s", "tick_interval_mean_ms_60s", "spread_zscore_1800s", "spread_mean_points_15s"]


class ProtocolNotActive(RuntimeError):
    """Protocol v2 is a draft; no runner may use it before explicit activation."""


def disclosure() -> dict[str, Any]:
    return {
        "statement": "Protocol v2 is a REVISED protocol informed by the findings of Protocol v1 on the same development data. It is NOT an independent "
                     "confirmatory study. The development period (2026-04-06..2026-09-11) has already been examined extensively; any v2 result on it "
                     "is exploratory-to-confirmatory at best and carries inflated false-positive risk, which the stricter gates below only partly offset.",
        "v1_informed_choices": [
            "separating direction from opportunity (v1: no directional information, 50.3% direction accuracy)",
            "binary unweighted proper-scoring models with a separate decision rule (v1: class weighting broke gate G2)",
            "fixed barrier distance 150 points and cost multiple 16 chosen to match the v1 mean H=60 barrier (148 points) so event rates are comparable",
            "the ten fingerprint features removed in the NEUTRAL configuration come from the v1 month-identifiability probe",
            "fewer/smaller hyper-parameter sets and min_child_samples=1000 (v1: weak, diffuse signal)"],
        "what_stays_independent": "the final holdout 2026-09-14..2026-10-06 (still LOCKED). Even it was covered by the descriptive whole-dataset profile "
                                  "published before the holdout protocol existed; no labels, features, predictions or outcomes inside it have been inspected."}


def master() -> dict[str, Any]:
    return {
        "version": PROTOCOL_V2_VERSION, "status": "DRAFT_NOT_ACTIVE", "active": False,
        "activation_required": "protocol_v2_ACTIVATION.json (absent) - created only after explicit owner authorization naming this draft's hash",
        "preserves": {"v1_immutable_manifest": "research/phase2b/v1_immutable_manifest.json",
                      "v1_formal_verdict": "NO RELIABLE SIGNAL DETECTED (unchanged; not reclassified)",
                      "v1_exploratory_marking": "the prior-corrected LightGBM analysis in v1 is POST-HOC / EXPLORATORY and never used for v2 design choices "
                                                "(calibration method and gates below were fixed by theory and inner out-of-fold evidence, not by that result)"},
        "frozen_and_unchanged": {"spec_hash": "57fb49f5eb10c5f0e8fd3a3c82c1c1c39603ee915be3cf510286c35475331845", "raw_dataset_freeze": "freeze-48779c44e3f2999274a2",
                                 "eligibility_policy": "eligibility/1", "segment_gap_s": 300, "folds": ["F1", "F2", "F3", "F4", "F5"], "embargo_s": 1800,
                                 "decision_grid_s": 30, "feature_set": "xauusd_core-fs1-ac928c2b1771",
                                 "cost_scenarios": ["C0_spread_only", "C1_moderate", "C2_pessimistic"]},
        "holdout": {"first_day": "2026-09-14", "last_day": "2026-10-06", "status": "LOCKED", "v2_access": "none",
                    "policy": "No diagnostics, labels, features, predictions or metrics on it. v2 training/evaluation never unlocks it automatically; "
                              "a separate written authorization after v2 is complete and its candidate is frozen is required."},
        "disclosure": disclosure(),
        "objective": "Determine whether XAUUSD contains a reproducible short-horizon DIRECTIONAL prediction signal, separately from volatility/opportunity prediction.",
        "tasks": ["A (direction)", "B (opportunity)"], "forbidden": ["PnL", "Sharpe", "profit factor", "backtests", "execution", "holdout access",
                                                                    "choosing any label/barrier by apparent profitability"],
        "stages": [
            {"id": "S0", "name": "label construction diagnostics (no models)", "outputs": "event rates, volatility dependence of each label, ambiguity shares, coverage"},
            {"id": "S1", "name": "Task A screening", "experiments": 21, "metric": "calibration-invariant (AUC, balanced accuracy)"},
            {"id": "S2", "name": "Task B screening (3 barrier variants, C0)", "experiments": 63, "metric": "AUC, average precision"},
            {"id": "S3", "name": "nested-OOF calibration + confirmatory metrics + gates for the screened candidates", "experiments": "1 candidate per task/variant"},
            {"id": "S4", "name": "fingerprint tests FP1-FP6, controls, ablations, cost sensitivity C1/C2 for the selected candidates"},
            {"id": "S5", "name": "verdicts and report; freeze MODEL_CANDIDATE_V2 manifest (or NONE)"}],
        "stop_rules": ["any failure condition in evaluation_v2.json", "protocol hash mismatch", "any read of a timestamp >= 2026-09-14 (HoldoutViolation)"],
    }


def calibration() -> dict[str, Any]:
    return {
        "principle": "Separate PROBABILITY ESTIMATION (proper scoring, unweighted training) from the DECISION RULE (threshold / prior-adjusted argmax). "
                     "Class weighting is NOT used, so the v1 failure mode (class-weight-induced probability distortion) cannot occur by construction.",
        "mathematics": {
            "why_v1_g2_failed": "A model trained with class weights w_k = 1/(K pi_k) estimates p_w(k|x) proportional to p(k|x)/pi_k, i.e. an implied uniform prior. Its log loss "
                                "against the true labels exceeds the training-prior log loss even if p(.|x) is informative, and a one-parameter temperature cannot move the prior.",
            "prior_correction_elkan": "p(k|x) proportional to p_w(k|x) * pi_k. Exact ONLY if the weighted model's class-conditional likelihood ratios are correct and the "
                                      "implied prior is exactly uniform; it fixes a prior shift, not general distortion (over/under-confidence, tree-ensemble saturation).",
            "multiclass_calibrators": "temperature (1 parameter), vector scaling (K scales + K biases), matrix scaling (K^2+K), Dirichlet calibration (on log-probabilities, "
                                      "equivalent to matrix scaling with L2). More parameters fix more distortion but need more calibration data and overfit more easily.",
            "binary_calibrators": "Platt scaling (a,b on the logit), temperature (a only), isotonic regression (monotone, non-parametric; needs >~1000 positives, step artefacts).",
            "decision_rule": "Binary tasks: predict class 1 when calibrated p > tau with tau = 0.5 (and tau = training base rate as the balanced operating point); both reported. "
                             "This yields balanced decisions without touching the probabilities.",
            "comparison_basis": "Chosen by statistical validity and data requirements (K=2 outcomes, ~100k+ calibration rows per fold), NOT by v1's exploratory best result."},
        "training_loss": {"class_weight": None, "sample_weight": None, "objective": "binary log loss (proper scoring rule)"},
        "candidate_calibrators": ["none", "temperature", "platt", "isotonic"],
        "calibrator_selection": "per outer fold, by inner out-of-fold log loss only; ties within 0.0005 nats -> simpler (none < temperature < platt < isotonic)",
        "nested_scheme": {
            "outer": "the five frozen walk-forward folds (train block -> validation block)",
            "inner": "within each outer TRAINING block: 4 time-ordered expanding blocks; inner block j (j=2..4) is predicted by a model fit on blocks 1..j-1 "
                     "with the frozen purge and 1,800 s embargo between them (out-of-fold predictions for blocks 2-4)",
            "fit": "calibrators are fit on the pooled inner out-of-fold predictions only",
            "final": "the base model is refit on the whole outer training block (purge/embargo as frozen) and predicts the outer validation block; the selected calibrator "
                     "is applied. The outer validation block is never used to fit the base model, the preprocessing, the regime/feature normalisation or the calibrator.",
            "known_approximation": "inner models see less data than the final model, so inner out-of-fold probabilities are slightly less confident than final-model ones; "
                                   "this is reported, and residual miscalibration is measured on the outer validation fold (diagnostic gates below)."},
        "scoring_gates_diagnostic": {
            "ece_max": 0.02, "calibration_slope_range": [0.9, 1.1], "calibration_intercept_abs_max": 0.10,
            "definition": "ECE: 15 equal-mass bins on pooled outer validation predictions; slope/intercept: logistic regression of the outcome on the logit of the "
                          "calibrated probability, fit on the pooled outer validation block ONLY to DIAGNOSE (never to correct and re-score)",
            "failure_action": "a failed calibration gate downgrades the log-loss/Brier evidence for that candidate to 'uncalibrated, indicative only'; discrimination metrics (AUC, balanced accuracy) are unaffected"},
        "compare_in_report": ["uncalibrated", "selected calibrator (nested OOF)"],
    }


def tasks() -> dict[str, Any]:
    return {
        "common": {"decision_grid_s": 30, "primary_horizon_s": 60, "robustness_horizons_s": [15, 300],
                   "horizon_rule": "robustness horizons are evaluated ONLY for the already-selected candidate; no re-selection across horizons",
                   "latency": {"primary_s": 1, "sensitivity_s": 0, "note": "decision at the bar end t; fill/entry quote at t+L; L is a hypothetical, UNCALIBRATED latency (1 s grid)"},
                   "segments": "labels never cross a segment boundary; rows whose window leaves the segment are dropped and counted",
                   "executable_quotes": "entry/exit use the observed historical bid and ask; the mid is used ONLY for the cost-free direction target U and is labelled as such"},
        "task_A_direction": {
            "question": "Given only information at t, does the mid price at t+L+H end above or below its value at t+L? (direction, independent of volatility/opportunity)",
            "target_U": "U = 1 if mid(t+L+H) > mid(t+L); U = 0 if < ; rows with equality are dropped (share reported)",
            "why_mid": "a cost-free direction target removes barrier width, volatility scaling and NO_TRADE from the definition; costs enter only in A-exec and the magnitude strata",
            "A_exec_secondary": {"definition": "executable opportunity rows: the v1 fixed-horizon label under scenario s is LONG or SHORT (net of observed spread + hypothetical extra cost); "
                                               "target = which side; reported WITH its coverage (share of all rows) and the selection caveat that the conditioning event is a "
                                               "future outcome", "gate": False},
            "magnitude_strata": {"basis": "realized |mid move| in multiples of the entry cost (entry spread points + extra_rt_points of the scenario)", "bounds": [0, 1, 2, 4, "inf"],
                                 "purpose": "show where directional accuracy lives; accuracy on rows chosen by realized magnitude is an UPPER BOUND diagnostic, not a gate"},
            "class_balance": "report the training/validation base rate P(U=1) per fold; metrics are balanced-accuracy and AUC based; log loss is compared with the training base rate",
            "primary_metrics": ["balanced_accuracy", "auc", "log_loss_skill_vs_training_base_rate", "brier_skill"],
            "unconditional_coverage": "the directional gate is evaluated on ALL rows (coverage 100%), with no abstention and no selection by realized outcomes",
            "abstention_policy": {"in_gate": False, "study": "abstain when |p-0.5| < delta, delta in {0.02, 0.05} fixed a priori; report coverage, accuracy and balanced accuracy on accepted rows, "
                                                            "and compare with (i) random abstention at equal coverage, (ii) abstention driven by a volatility-only score at equal coverage",
                                  "selection_accounting": "a selectively favourable subset only counts if it beats BOTH controls with a block-bootstrap CI; the unconditional metric is always reported next to it"},
            "ambiguous_paths": {"endpoint_target": "no path ambiguity", "barrier_direction_secondary": "among Task-B event rows, which side touched first; same-second ties dropped "
                                                                                                           "and counted; sensitivity: ties assigned to the stop side (conservative)"},
            "baselines": ["training base rate", "momentum: sign(ret_bps_60s)", "mean reversion: -sign(ret_bps_60s)", "coin flip (seed 7)"]},
        "task_B_opportunity": {
            "question": "Does an executable barrier event (either side) occur within H, regardless of direction?",
            "target_E": "E = 1 if the LONG or the SHORT profit barrier is hit strictly before its own stop within H (executable bid/ask path, scenario cost widening the profit barrier); "
                        "same-second profit+stop -> stop first (as v1); both sides hit -> E = 1 (direction ambiguous, counted)",
            "barrier_variants": {
                "V_vol_scaled": {"rule": "B_i = max(1.0 * sigma_i * mid * sqrt(H), 3 * spread_i); sigma_i = trailing 600 s std of 1-s log mid returns (identical to v1)"},
                "F_fixed": {"rule": "B = 150 points (USD 1.50) for every row", "anchor": "approximately the v1 mean H=60 barrier (148 points); chosen so event rates are comparable, not for performance"},
                "N_cost_normalized": {"rule": "B_i = 16 * (entry_spread_points + extra_rt_points) points", "anchor": "16 x the median C0 spread (9 points) is ~ 144 points; comparable scale"}},
            "controlled_comparison": {
                "question": "Is the v1 target mainly a volatility/timeout classification problem?",
                "tests": [
                    "VT1 volatility-only baseline: LightGBM on the 'volatility' feature group only; ratio R = lift(volatility-only) / lift(full model) for each variant (lift = AUC - 0.5)",
                    "VT2 within-volatility-stratum skill: AUC of the full model inside train-defined terciles of trailing sigma (fit on the outer training block)",
                    "VT3 label-volatility dependence: AUC of trailing sigma alone for predicting E in each variant (no model, descriptive, Stage S0)"],
                "interpretation_rule": "a variant is called VOLATILITY-DOMINATED when R >= 0.8 AND skill does not survive in >= 2 of 3 volatility terciles (AUC >= 0.52). "
                                       "No variant is selected or discarded by trading profitability; variant results are reported side by side (Holm-corrected)."},
            "primary_metrics": ["auc", "average_precision", "log_loss_skill_vs_training_base_rate", "brier_skill", "ece"],
            "reported": ["event probability (base rate) per fold/month/session", "calibration table", "precision and recall at tau = training base rate and 0.5",
                         "performance by train-defined volatility tercile, spread tercile and session"],
            "interpretation_constraint": "Task B skill is NEVER interpreted as directional trading skill.",
            "baselines": ["training base rate", "volatility-only model", "spread-only model", "session-only model"]},
        "feature_configs": {
            "RAW": "the 108 ALLOWED features",
            "NEUTRAL": "RAW minus the 10 month-fingerprint features (FINGERPRINT_FEATURES)",
            "TRAILING_NORM": "each RAW feature centered/scaled by its own median/MAD over the previous 5 trading days (strictly causal, per-feature, computed from past rows only; "
                             "the first 5 days of the development window provide the initial window and are not scored)"},
        "fingerprint_features": FINGERPRINT_FEATURES,
    }


def evaluation() -> dict[str, Any]:
    models = {"logistic": {"C": [0.01, 0.1, 1.0]}, "lightgbm": {"num_leaves": [7, 15], "n_estimators": [100, 300],
                                                                  "fixed": {"learning_rate": 0.05, "min_child_samples": 1000, "subsample": 0.7,
                                                                            "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 10.0,
                                                                            "class_weight": None, "deterministic": True, "seed": 7}}}
    n_models = len(models["logistic"]["C"]) + len(models["lightgbm"]["num_leaves"]) * len(models["lightgbm"]["n_estimators"])
    return {
        "folds": "frozen F1..F5, purge/embargo/segments/eligibility exactly as frozen; chronological only; no random splits",
        "models": models, "n_model_configs": n_models, "n_feature_configs": 3, "configs_per_label_set": n_models * 3, "hyperparameter_budget_cap_per_label_set": 24,
        "search_policy": "this grid is final; never expanded after seeing results (a larger search needs protocol v3)",
        "baselines": "see tasks_v2.json; all fit on the training block only",
        "uncertainty": {
            "block_bootstrap": {"block_s": [600, 1800], "B": 2000, "resampling": "within fold, paired with the baseline", "requirement": "conclusions must hold under BOTH block lengths"},
            "day_cluster_bootstrap": {"B": 2000, "resampling": "calendar days within fold", "requirement": "reported; the gate uses the more conservative of the three intervals"},
            "effective_sample_size": {"formula": "n_eff = n / (1 + 2 * sum_{k=1..K} (1 - k/(K+1)) * rho_k)", "series": "per-row score contribution (log-loss difference vs the baseline)",
                                      "K_lags": 120, "unit": "30 s grid steps (1 h)", "min_n_eff_per_fold": 500,
                                      "underpowered_rule": "a fold with n_eff < 500 is flagged UNDERPOWERED and cannot contribute a 'fold beats baseline' vote"}},
        "multiplicity": {"alpha_one_sided_family": 0.025, "task_A": "one gate, 95% two-sided CI", "task_B": "3 barrier variants -> Bonferroni-equivalent 98.33% two-sided CI for each",
                         "note": "stricter than v1 because the development data has been reused"},
        "stage_metrics": {"screening": "calibration-invariant: A = mean fold AUC; B = mean fold AUC (average precision reported); log-loss metrics are NOT used to screen",
                          "confirmation": "after nested-OOF calibration on the selected candidate: balanced accuracy, AUC, log-loss skill, Brier skill, ECE, slope"},
        "selection_rule": {
            "screening_gate": "mean AUC > 0.5 and AUC > 0.5 in >= 4 of 5 folds (n_eff-adequate folds only)",
            "score": "S = mean_fold(AUC) - 0.5 - SD_fold(AUC)",
            "choice": "highest S; if candidates are within 0.002 of the best S prefer: NEUTRAL or TRAILING_NORM over RAW, then logistic over LightGBM, then fewer trees/leaves",
            "per": "Task A: one choice. Task B: one choice per barrier variant (reported side by side)",
            "never_used": ["final holdout", "PnL", "Sharpe", "profit factor", "threshold tuned for returns"]},
        "gates": {
            "directional_skill_gate_task_A": {
                "D1_discrimination": "pooled balanced accuracy >= 0.505 AND lower 95% CI bound of (balanced accuracy - 0.5) > 0, under both block lengths and the day-cluster bootstrap",
                "D2_probabilistic": "after calibration: log-loss skill vs training base rate > 0 with lower CI bound > 0 and point improvement >= 0.0003 nats",
                "D3_folds": "balanced accuracy > 0.5 in >= 4 of 5 folds (n_eff-adequate folds)",
                "D4_robust_to_fingerprints": "the same candidate family in NEUTRAL AND TRAILING_NORM retains >= 50% of the RAW lift with lower CI bound > 0",
                "D5_months": "skill positive in >= 3 of the 4 validation months (Jun, Jul, Aug, Sep) and no month with the CI upper bound below 0",
                "D6_coverage": "evaluated at 100% coverage; any abstention result is supplementary",
                "D7_placebo": "the within-day label-permutation placebo (FP6) shows no skill (CI contains 0)",
                "outcomes": {"DIRECTIONAL SIGNAL DETECTED": "all of D1-D7", "WEAK/UNSTABLE": "D1 point estimate positive in the pooled data and >= 4 of D1-D7 hold",
                             "NO RELIABLE DIRECTIONAL SIGNAL": "otherwise"}},
            "opportunity_skill_gate_task_B": {
                "O1_discrimination": "AUC >= 0.55 with lower CI bound > 0.5 (98.33% interval, both block lengths + day-cluster)",
                "O2_probabilistic": "after calibration: log-loss skill vs training base rate with lower CI bound > 0 and point >= 0.005 nats",
                "O3_calibration": "ECE <= 0.02 and slope in [0.9, 1.1] (diagnostic gates of calibration_v2.json)",
                "O4_beyond_volatility": "AUC >= 0.52 in >= 2 of 3 train-defined volatility terciles (VT2) AND not VOLATILITY-DOMINATED (VT1)",
                "O5_folds": "AUC > 0.5 in >= 4 of 5 folds (n_eff-adequate folds)",
                "O6_fingerprints": "survives NEUTRAL and TRAILING_NORM with >= 50% of the RAW lift",
                "outcomes": {"OPPORTUNITY SKILL DETECTED": "O1-O6", "WEAK/UNSTABLE": "O1 holds and >= 4 of O1-O6 hold", "NONE": "otherwise"},
                "interpretation": "even when detected, this is event (volatility/opportunity) prediction, not directional skill"}},
        "failure_conditions": [
            "calibrator, normaliser, regime threshold or feature selection fit on any outer-validation row (LeakageError)",
            "any timestamp >= 2026-09-14 touched (HoldoutViolation)",
            "frozen spec hash / protocol hash mismatch or modified v1 artifacts (seal check fails)",
            "label causality or segment-protection test fails",
            "experiment non-determinism (same id, different predictions)",
            "a fold with fewer than 5,000 validation rows after dropping ties/ambiguous rows",
            "unconverged fits beyond 10% of experiments (reported; results from unconverged fits flagged)",
            "more than 24 configurations per label set attempted"],
        "controls_and_ablations": {"controls": ["volatility_only", "spread_only", "session_only", "coin_flip", "momentum", "mean_reversion"],
                                   "ablations_selected_candidate": ["spread", "session", "mtf", "momentum", "mean_reversion", "structure", "micro", "volatility"],
                                   "note": "v1 omitted the volatility group from its ablation list; v2 includes it"},
        "verdict_scope": "predictive classification only; no economic claim",
    }


def fingerprints() -> dict[str, Any]:
    return {
        "purpose": "Does directional or opportunity performance survive month-to-month shifts, the September feed boundary, April-May vs June-onward spread differences "
                   "and removal of obvious time/provenance proxies? (development data only)",
        "tests": {
            "FP1_month_stratified": {"how": "pooled validation predictions grouped by calendar month (Jun, Jul, Aug, Sep)", "pass": "skill positive in >= 3 of 4 months; no month with CI upper bound < 0"},
            "FP2_feed_boundary": {"how": "fold F5 validation split into pre (2026-08-24..09-04) and post (2026-09-07..09-11) rows", "pass": "post-boundary skill point estimate >= 0 and "
                                                                                                                                     "not significantly negative; underpowered (5 days) so significance is NOT required"},
            "FP3_regime_transfer": {
                "a_chronological": "train 2026-04-06..05-29 only -> validate 2026-06-01..08-21 (April-May regime -> June-onward regime), fixed selected hyper-parameters",
                "b_regime_restricted": "train 2026-06-01..08-21 only -> validate 2026-08-24..09-11 (no April-May data in training)",
                "c_reversed_diagnostic": "train 2026-06-01..08-21 -> test 2026-04-06..05-29. NON-CHRONOLOGICAL diagnostic (future predicts past): reported, excluded from every gate and from selection",
                "pass": "(a) and (b) skill positive and >= 50% of the main result"},
            "FP4_proxy_removal": {"how": "NEUTRAL and TRAILING_NORM feature configurations", "pass": "retain >= 50% of the RAW lift with lower CI bound > 0 (this is gate D4/O6)"},
            "FP5_adversarial_validation": {"how": "classifier distinguishing training from validation rows per outer fold (RAW features); AUC and top shift features",
                                           "use": "descriptive; AUC > 0.9 flags severe covariate shift; overlap of top shift features with the candidate's top permutation features is reported"},
            "FP6_placebo": {"how": "permute the TARGET within calendar day (preserves day-level base rate and volatility), refit the selected pipeline, 5 folds, 3 seeds",
                            "pass": "placebo skill CI contains 0 (no skill from period fingerprints + label drift)"}},
        "month_identifiability_check": "repeat the v1 probe on NEUTRAL and TRAILING_NORM features; target: accuracy well below the RAW 0.664 (reported, not a gate)",
    }


def cost() -> dict[str, Any]:
    return {
        "scenarios": "C0_spread_only / C1_moderate / C2_pessimistic exactly as frozen in cost_model_v0.json",
        "status": {"observed_spread": "OBSERVED (historical bid/ask)", "commission_round_turn_points": "UNKNOWN/UNCALIBRATED", "slippage_points_per_side": "UNKNOWN/UNCALIBRATED",
                   "latency_s": "UNKNOWN/UNCALIBRATED", "C1_C2_values": "HYPOTHETICAL sensitivity points, not broker facts"},
        "use": "Task A primary target is cost-free (mid direction); costs enter A-exec, magnitude strata and every Task B barrier. All Task B/A-exec results are shown under C0, C1 and C2.",
        "no_profitability_claim": True,
        "broker_details_needed_to_establish_realistic_commissions": [
            "exact account type and server of the account used for the live run (the DEMO feed's account type is not proof of the live account's pricing)",
            "commission schedule: amount, currency, per lot per side or per round turn, and any volume tiers/rebates",
            "XAUUSD contract specification on that account: contract size (100 oz?), tick size/value, minimum/step lot",
            "whether spreads are raw (commission charged) or marked-up (commission-free) and how live spreads compare with the DEMO spreads in this dataset",
            "swap/financing rates (long/short) and rollover time (relevant only beyond intraday)",
            "order execution mode (market/instant/request), slippage/requote policy, maximum deviation, last-look or price improvement behaviour",
            "measured round-trip latency from the intended execution host to the trade server (VPS location) and typical fill times",
            "news-time behaviour: spread widening, trading restrictions, freeze/stops levels, minimum stop distance",
            "an actual fill sample (many small live or demo orders) to estimate slippage - NOT available in this project and NOT to be generated without explicit authorization"],
    }


def build_all() -> dict[str, dict[str, Any]]:
    return {"protocol_v2_draft.json": master(), "calibration_v2.json": calibration(), "tasks_v2.json": tasks(), "evaluation_v2.json": evaluation(),
            "fingerprint_tests_v2.json": fingerprints(), "cost_v2.json": cost()}


def canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, indent=2, ensure_ascii=True, default=str) + "\n"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_draft(directory: Path = DRAFT_DIR) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    idx: dict[str, Any] = {"status": "DRAFT_NOT_ACTIVE", "files": {}}
    for name, body in build_all().items():
        text = canon(body)
        (directory / name).write_text(text, encoding="utf-8", newline="\n")
        idx["files"][name] = sha(text)
    idx["draft_hash"] = sha(canon(idx["files"]))
    (directory / "draft_index.json").write_text(canon(idx), encoding="utf-8", newline="\n")
    return idx


def assert_active(directory: Path = DRAFT_DIR) -> None:
    if not (directory / "protocol_v2_ACTIVATION.json").exists():
        raise ProtocolNotActive("Protocol v2 is a DRAFT: no activation file; no runner may use it without explicit owner authorization")


def validate_draft(directory: Path = DRAFT_DIR) -> list[str]:
    """Static validation. Returns a list of problems (empty = valid). Executes nothing."""
    bad: list[str] = []
    idx_path = directory / "draft_index.json"
    if not idx_path.exists():
        return ["draft_index.json missing"]
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    docs: dict[str, Any] = {}
    for name in FILES:
        p = directory / name
        if not p.exists():
            bad.append(f"{name} missing")
            continue
        text = p.read_text(encoding="utf-8")
        if sha(text) != idx["files"].get(name):
            bad.append(f"{name}: hash differs from draft_index (edited after generation)")
        docs[name] = json.loads(text)
    if bad:
        return bad
    if sha(canon(idx["files"])) != idx["draft_hash"]:
        bad.append("draft_hash inconsistent")
    for name, body in build_all().items():
        if canon(body) != canon(docs[name]):
            bad.append(f"{name}: differs from the in-code definition")
    m, cal, tk, ev, fp, co = (docs[n] for n in FILES)
    if m["status"] != "DRAFT_NOT_ACTIVE" or m["active"] is not False or idx["status"] != "DRAFT_NOT_ACTIVE":
        bad.append("draft must be marked DRAFT_NOT_ACTIVE")
    if (directory / "protocol_v2_ACTIVATION.json").exists():
        bad.append("an activation file exists but this validation is for the DRAFT (activation requires owner authorization and a new hash)")
    # disclosure
    d = m["disclosure"]["statement"]
    if "informed by" not in d or "NOT an independent confirmatory study" not in d:
        bad.append("disclosure must state v2 is informed by v1 and is not an independent confirmatory study")
    # v1 preserved / frozen items unchanged / holdout locked
    fz = m["frozen_and_unchanged"]
    if fz["spec_hash"] != "57fb49f5eb10c5f0e8fd3a3c82c1c1c39603ee915be3cf510286c35475331845" or fz["embargo_s"] != 1800 or fz["segment_gap_s"] != 300 or fz["decision_grid_s"] != 30:
        bad.append("frozen specification parameters changed")
    if m["holdout"]["status"] != "LOCKED" or m["holdout"]["v2_access"] != "none" or m["holdout"]["first_day"] != "2026-09-14":
        bad.append("holdout must remain LOCKED with no v2 access")
    if "NO RELIABLE SIGNAL DETECTED" not in m["preserves"]["v1_formal_verdict"]:
        bad.append("v1 formal verdict must be preserved")
    # holdout never inside any declared window
    txt = json.dumps(docs)
    for forbidden in ("2026-09-14..", "2026-10-0"):
        for n, body in docs.items():
            if n != "protocol_v2_draft.json" and forbidden in json.dumps(body) and "NON-CHRONOLOGICAL" not in json.dumps(body):
                bad.append(f"{n} references the holdout window ({forbidden})")
    # calibration validity
    if cal["training_loss"]["class_weight"] is not None or cal["training_loss"]["sample_weight"] is not None:
        bad.append("v2 models must be trained unweighted (proper scoring)")
    if "outer validation block is never used to fit" not in cal["nested_scheme"]["final"] or cal["nested_scheme"]["inner"].count("4 time-ordered") != 1:
        bad.append("calibration must be nested / out-of-fold with the outer validation excluded")
    if not set(cal["candidate_calibrators"]) <= {"none", "temperature", "platt", "isotonic"} or len(cal["candidate_calibrators"]) != 4:
        bad.append("calibrator list changed")
    g = cal["scoring_gates_diagnostic"]
    if not (0 < g["ece_max"] <= 0.05 and g["calibration_slope_range"][0] < 1 < g["calibration_slope_range"][1]):
        bad.append("calibration gates implausible")
    # budget
    if ev["configs_per_label_set"] > ev["hyperparameter_budget_cap_per_label_set"] or ev["configs_per_label_set"] != 21:
        bad.append("hyper-parameter budget violated or changed")
    if ev["models"]["lightgbm"]["fixed"]["class_weight"] is not None:
        bad.append("LightGBM must be unweighted")
    # gates well-formed
    gd, go = ev["gates"]["directional_skill_gate_task_A"], ev["gates"]["opportunity_skill_gate_task_B"]
    if len([k for k in gd if k.startswith("D")]) != 7 or len([k for k in go if k.startswith("O")]) != 6:
        bad.append("gate sets incomplete")
    if "0.505" not in gd["D1_discrimination"] or "0.55" not in go["O1_discrimination"]:
        bad.append("minimum improvement criteria missing")
    if ev["uncertainty"]["block_bootstrap"]["block_s"] != [600, 1800] or ev["uncertainty"]["effective_sample_size"]["min_n_eff_per_fold"] < 100:
        bad.append("uncertainty requirements missing")
    if not {"PnL", "Sharpe", "profit factor"} <= set(ev["selection_rule"]["never_used"]):
        bad.append("selection rule must exclude economic metrics")
    # tasks
    if set(tk["task_B_opportunity"]["barrier_variants"]) != {"V_vol_scaled", "F_fixed", "N_cost_normalized"}:
        bad.append("barrier variants changed")
    if tk["task_A_direction"]["abstention_policy"]["in_gate"] is not False or tk["task_A_direction"]["unconditional_coverage"].find("100%") < 0:
        bad.append("direction gate must be unconditional (no abstention)")
    if tk["feature_configs"]["TRAILING_NORM"].find("strictly causal") < 0 or len(tk["fingerprint_features"]) != 10:
        bad.append("feature configs / fingerprint features malformed")
    if "NEVER interpreted as directional" not in tk["task_B_opportunity"]["interpretation_constraint"]:
        bad.append("Task B must not be interpreted as directional skill")
    # fingerprints & cost
    if set(fp["tests"]) != {"FP1_month_stratified", "FP2_feed_boundary", "FP3_regime_transfer", "FP4_proxy_removal", "FP5_adversarial_validation", "FP6_placebo"}:
        bad.append("fingerprint tests incomplete")
    if "NON-CHRONOLOGICAL" not in fp["tests"]["FP3_regime_transfer"]["c_reversed_diagnostic"]:
        bad.append("reversed-time diagnostic must be labelled non-chronological")
    if co["no_profitability_claim"] is not True or "HYPOTHETICAL" not in co["status"]["C1_C2_values"] or len(co["broker_details_needed_to_establish_realistic_commissions"]) < 6:
        bad.append("cost section incomplete")
    return bad
