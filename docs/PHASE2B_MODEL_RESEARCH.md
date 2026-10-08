# Phase 2B model research - development-fold evidence (final holdout LOCKED)

**Verdict (pre-registered rule, development evidence only): NO RELIABLE SIGNAL DETECTED.** All numbers below are *predictive classification* performance on the five frozen walk-forward development folds. There is **no economic/trading evaluation**: no PnL, Sharpe or profit factor was computed, costs C1/C2 are hypothetical, and **profitability is UNVERIFIED**. The final holdout (2026-09-14..2026-10-06) was not read, scored or inspected. No order was placed and nothing was connected to MT5 execution.

Frozen specification: spec_hash `57fb49f5eb10c5f0e8fd3a3c82c1c1c39603ee915be3cf510286c35475331845`, dataset freeze `freeze-48779c44e3f2999274a2`, protocol `ed05108d113595703eb09e64094fe840c57d5adf20b75512e2c62189c5a35d4b` (committed before any model was trained, git `8c18f55`); experiments ran at git `2dc269d`.

## 1. Experiment protocol

Primary label `tb_H60` (triple barrier, 60 s) under the three frozen cost scenarios C0/C1/C2, each trained and scored separately. Decision grid: bar ends on a 30 s UTC grid (1/30 of the 1-s rows, 297,040 rows over 115 development days). Folds: the five frozen expanding folds with the frozen purge/embargo; within each training block the last 20% (by time) is held out for temperature calibration (the model is fit on the earlier 80% after an extra 1,800 s embargo), and the same 80%-model is reported uncalibrated and calibrated. Preprocessing (logistic: winsorisation at 0.1/99.9%, median/IQR scaling, median imputation) and REGIME thresholds are fit on the training block only. Decision rule: argmax of calibrated probabilities; no threshold was optimised. Experiment ids are SHA-256 hashes of (freeze, feature manifest, label + scenario, folds, model, hyper-parameters, seed, preprocessing, calibration, grid, protocol, matrix id); experiments are never overwritten. Dependence: 60 s labels on a 30 s grid overlap by 50%, so uncertainty uses a paired **block bootstrap (600 s blocks = 10 x the horizon, B=1000, resampling within fold)**, plus a non-overlapping-60 s robustness evaluation stored per fold.

## 2. Exact models and predetermined search spaces (final; never expanded)

- A. Multinomial logistic regression (L2, lbfgs, max_iter 300): C ∈ [0.01, 0.1, 1.0] × class_weight ∈ [None, 'balanced'] (6 configs).
- B. LightGBM gradient boosting: num_leaves ∈ [15, 31] × n_estimators ∈ [150, 400] (4 configs); fixed learning_rate 0.05, min_child_samples 500, subsample 0.7, colsample 0.7, reg_lambda 10, class_weight balanced.
- C. A third family was **not** used (no justification beyond a first causal-signal test).
- Feature configurations: CORE (108), CONDITIONAL (108 + 8 gated session aggregates), REGIME (108 + 2 regime codes with thresholds fit per fold on training rows) - 10 × 3 = 30 candidates per scenario, 90 model experiments + 12 baseline experiments, seed 7. Undocumented flag bits, raw volume and `last` are never used.

## 3. Selection rule (written before training)

```
{
 "choice": "highest S among gate-passers; if candidates are within 0.005 of the best S, prefer the simpler: logistic < LightGBM, CORE < CONDITIONAL < REGIME, fewer trees, fewer leaves, larger C-regularisation",
 "gates (all must hold, evaluated on calibrated probabilities)": {
  "G1_beats_baselines": "mean-over-scenarios fold macro-F1 > the best naive baseline's, in >= 4 of 5 folds",
  "G2_probabilistic_skill": "fold log-loss < training-prior log-loss in >= 4 of 5 folds",
  "G3_all_classes_predicted": "mean validation recall of every class >= 0.05",
  "G4_no_proxy_flag": "no leakage/proxy flag from the rule below"
 },
 "never_used": [
  "final holdout",
  "PnL",
  "Sharpe",
  "profit factor",
  "any threshold optimised for returns"
 ],
 "no_gate_passer": "report NO CANDIDATE; the verdict becomes NO RELIABLE SIGNAL DETECTED",
 "proxy_flag": "set if (a) a control model (session_only or spread_only) reaches >= 80% of the candidate's mean macro-F1 lift over the best baseline in the same scenario, or (b) the session + spread feature groups carry >= 50% of the candidate's total group-permutation log-loss increase",
 "score": "S = M - D - E; M = mean over scenarios and folds of macro-F1; D = mean over scenarios of the SD over folds of macro-F1; E = mean over scenarios and folds of top-label ECE (calibrated)",
 "unit": "candidate = (feature_config, model_family, hyperparameters); each trained separately for C0, C1 and C2 labels"
}
```

## 4. Baselines (all development folds)

| scenario | baseline | macro-F1 (mean ± SD over folds) | balanced acc | accuracy | log loss | predicted-class frequency SHORT / NO_TRADE / LONG |
|---|---|---|---|---|---|---|
| C0_spread_only | majority | 0.228 ± 0.005 | 0.333 | 0.519 | 1.027 | 0.00 / 1.00 / 0.00 |
| C0_spread_only | stratified_random | 0.335 ± 0.002 | 0.335 | 0.389 | 1.027 | 0.24 / 0.53 / 0.23 |
| C0_spread_only | momentum | 0.218 ± 0.006 | 0.334 | 0.242 | - | 0.50 / 0.00 / 0.50 |
| C0_spread_only | mean_reversion | 0.218 ± 0.004 | 0.333 | 0.241 | - | 0.50 / 0.00 / 0.50 |
| C1_moderate | majority | 0.243 ± 0.005 | 0.333 | 0.575 | 0.977 | 0.00 / 1.00 / 0.00 |
| C1_moderate | stratified_random | 0.334 ± 0.001 | 0.334 | 0.423 | 0.977 | 0.21 / 0.58 / 0.21 |
| C1_moderate | momentum | 0.202 ± 0.008 | 0.337 | 0.216 | - | 0.50 / 0.00 / 0.50 |
| C1_moderate | mean_reversion | 0.198 ± 0.005 | 0.330 | 0.211 | - | 0.50 / 0.00 / 0.50 |
| C2_pessimistic | majority | 0.267 ± 0.005 | 0.333 | 0.667 | 0.867 | 0.00 / 1.00 / 0.00 |
| C2_pessimistic | stratified_random | 0.333 ± 0.002 | 0.333 | 0.499 | 0.867 | 0.17 / 0.67 / 0.16 |
| C2_pessimistic | momentum | 0.171 ± 0.010 | 0.339 | 0.171 | - | 0.50 / 0.00 / 0.50 |
| C2_pessimistic | mean_reversion | 0.165 ± 0.007 | 0.328 | 0.165 | - | 0.50 / 0.00 / 0.50 |

The naive bar a model must beat is the best baseline's macro-F1 (stratified random ≈ 0.334, i.e. 1/3 for a 3-class problem); momentum and mean-reversion are at chance (balanced accuracy ≈ 0.33), so 60-second trailing return alone carries no directional information for `tb_H60`.

## 5. All candidates (mean over the 3 scenarios and 5 folds; calibrated probabilities)

S = M − D − E (M mean macro-F1, D mean fold SD, E mean top-label ECE). G1: folds beating the best baseline's macro-F1; G2: folds with log loss below the training-prior log loss; G3: smallest mean class recall.

| features | model | hyper-parameters | M | D | E | S | bal acc | log loss | G1 | G2 | G3 | passes G1-G3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CORE | lightgbm | n_estimators=400, num_leaves=31 | 0.362 | 0.004 | 0.022 | 0.337 | 0.370 | 1.069 | 5/5 | 0/5 | 0.286 | no |
| REGIME | lightgbm | n_estimators=400, num_leaves=31 | 0.362 | 0.004 | 0.022 | 0.336 | 0.369 | 1.069 | 5/5 | 0/5 | 0.284 | no |
| CORE | lightgbm | n_estimators=150, num_leaves=31 | 0.365 | 0.004 | 0.028 | 0.333 | 0.373 | 1.073 | 5/5 | 0/5 | 0.287 | no |
| CORE | lightgbm | n_estimators=400, num_leaves=15 | 0.364 | 0.004 | 0.027 | 0.333 | 0.372 | 1.072 | 5/5 | 0/5 | 0.288 | no |
| REGIME | lightgbm | n_estimators=400, num_leaves=15 | 0.363 | 0.004 | 0.027 | 0.332 | 0.371 | 1.072 | 5/5 | 0/5 | 0.287 | no |
| REGIME | lightgbm | n_estimators=150, num_leaves=31 | 0.364 | 0.004 | 0.028 | 0.332 | 0.371 | 1.072 | 5/5 | 0/5 | 0.286 | no |
| CORE | lightgbm | n_estimators=150, num_leaves=15 | 0.366 | 0.004 | 0.034 | 0.327 | 0.373 | 1.075 | 5/5 | 0/5 | 0.285 | no |
| REGIME | lightgbm | n_estimators=150, num_leaves=15 | 0.365 | 0.004 | 0.034 | 0.326 | 0.372 | 1.075 | 5/5 | 0/5 | 0.286 | no |
| CORE | logistic | C=1.0, class_weight=balanced | 0.365 | 0.008 | 0.044 | 0.314 | 0.373 | 1.083 | 5/5 | 0/5 | 0.293 | no |
| REGIME | logistic | C=0.1, class_weight=balanced | 0.366 | 0.007 | 0.045 | 0.314 | 0.373 | 1.081 | 5/5 | 0/5 | 0.283 | no |
| CORE | logistic | C=0.1, class_weight=balanced | 0.365 | 0.007 | 0.044 | 0.314 | 0.373 | 1.083 | 5/5 | 0/5 | 0.294 | no |
| REGIME | logistic | C=0.01, class_weight=balanced | 0.366 | 0.007 | 0.045 | 0.313 | 0.373 | 1.081 | 5/5 | 0/5 | 0.286 | no |
| CORE | logistic | C=0.01, class_weight=balanced | 0.365 | 0.008 | 0.044 | 0.313 | 0.373 | 1.083 | 5/5 | 0/5 | 0.293 | no |
| REGIME | logistic | C=1.0, class_weight=balanced | 0.365 | 0.007 | 0.045 | 0.313 | 0.373 | 1.081 | 5/5 | 0/5 | 0.282 | no |
| CONDITIONAL | logistic | C=0.01, class_weight=balanced | 0.355 | 0.007 | 0.044 | 0.304 | 0.371 | 1.078 | 5/5 | 0/5 | 0.232 | no |
| CONDITIONAL | logistic | C=1.0, class_weight=balanced | 0.355 | 0.007 | 0.044 | 0.304 | 0.372 | 1.078 | 5/5 | 0/5 | 0.229 | no |
| CONDITIONAL | logistic | C=0.1, class_weight=balanced | 0.355 | 0.008 | 0.044 | 0.303 | 0.371 | 1.078 | 5/5 | 0/5 | 0.232 | no |
| CONDITIONAL | lightgbm | n_estimators=400, num_leaves=31 | 0.350 | 0.019 | 0.032 | 0.299 | 0.372 | 1.081 | 4/5 | 0/5 | 0.284 | no |
| CONDITIONAL | lightgbm | n_estimators=150, num_leaves=31 | 0.352 | 0.017 | 0.036 | 0.299 | 0.372 | 1.083 | 4/5 | 0/5 | 0.275 | no |
| CONDITIONAL | lightgbm | n_estimators=150, num_leaves=15 | 0.352 | 0.016 | 0.039 | 0.298 | 0.373 | 1.083 | 4/5 | 0/5 | 0.259 | no |
| CONDITIONAL | lightgbm | n_estimators=400, num_leaves=15 | 0.349 | 0.020 | 0.035 | 0.293 | 0.372 | 1.084 | 4/5 | 0/5 | 0.267 | no |
| CORE | logistic | C=0.1, class_weight=None | 0.258 | 0.005 | 0.025 | 0.228 | 0.337 | 0.945 | 0/5 | 5/5 | 0.005 | no |
| CORE | logistic | C=1.0, class_weight=None | 0.258 | 0.005 | 0.025 | 0.228 | 0.337 | 0.945 | 0/5 | 5/5 | 0.005 | no |
| CORE | logistic | C=0.01, class_weight=None | 0.258 | 0.005 | 0.025 | 0.228 | 0.337 | 0.945 | 0/5 | 5/5 | 0.005 | no |
| REGIME | logistic | C=0.1, class_weight=None | 0.257 | 0.005 | 0.025 | 0.227 | 0.337 | 0.945 | 0/5 | 5/5 | 0.005 | no |
| REGIME | logistic | C=0.01, class_weight=None | 0.257 | 0.005 | 0.026 | 0.227 | 0.337 | 0.945 | 0/5 | 5/5 | 0.005 | no |
| REGIME | logistic | C=1.0, class_weight=None | 0.257 | 0.005 | 0.026 | 0.226 | 0.337 | 0.945 | 0/5 | 5/5 | 0.005 | no |
| CONDITIONAL | logistic | C=0.01, class_weight=None | 0.258 | 0.006 | 0.027 | 0.225 | 0.338 | 0.945 | 0/5 | 5/5 | 0.004 | no |
| CONDITIONAL | logistic | C=1.0, class_weight=None | 0.258 | 0.006 | 0.027 | 0.225 | 0.338 | 0.945 | 0/5 | 5/5 | 0.004 | no |
| CONDITIONAL | logistic | C=0.1, class_weight=None | 0.258 | 0.006 | 0.027 | 0.225 | 0.338 | 0.945 | 0/5 | 5/5 | 0.005 | no |

Per-fold results of **every** experiment (uncalibrated and calibrated) are in `research/phase2b/model_research/all_fold_results.csv`.

## 6. Selection outcome

Candidates passing G1-G3: 0 of 30. **No candidate passed the pre-registered gates, so the formal outcome is NO CANDIDATE.** 

**Why (diagnosed after the fact, reported transparently).** Every LightGBM and every class-balanced logistic candidate beats the baselines on macro-F1 in all five folds, but fails G2 because a class-weighted model's raw probabilities are prior-shifted: its log loss (~1.07) is worse than the training prior's (0.957), and single-parameter temperature scaling cannot undo a prior shift. The unweighted logistic models have genuine probabilistic skill (log loss 0.945 < 0.957) but essentially never predict LONG or SHORT (recall < 0.01), so they fail G1/G3. This is a design defect of protocol v1 (class weighting + prior-based log-loss gate + temperature-only calibration), **not** evidence about the data. I did **not** change the rule. Everything in sections 7-12 describes the **exploratory focus candidate** - the highest-S candidate passing G1 and G3 with G2 ignored - and is labelled EXPLORATORY; it does not alter the formal outcome.

Focus candidate: **CORE / lightgbm / n_estimators=400, num_leaves=31**; proxy check flag = False.

## 7. Development-fold results of the focus candidate (calibrated unless stated)

| scenario | fold | n train | n val | macro-F1 | best baseline macro-F1 | balanced acc | log loss (uncal → cal) | Brier | ECE (uncal → cal) | T | fit+predict s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C0_spread_only | F1 | 103,517 | 38,392 | 0.364 | 0.337 (stratified_random) | 0.369 | 1.087 → 1.085 | 0.657 | 0.024 → 0.013 | 1.58 | 79 |
| C0_spread_only | F2 | 141,909 | 38,024 | 0.364 | 0.334 (stratified_random) | 0.365 | 1.075 → 1.075 | 0.650 | 0.012 → 0.017 | 1.11 | 97 |
| C0_spread_only | F3 | 179,933 | 39,063 | 0.364 | 0.336 (stratified_random) | 0.368 | 1.079 → 1.079 | 0.653 | 0.013 → 0.016 | 1.13 | 113 |
| C0_spread_only | F4 | 218,996 | 39,063 | 0.362 | 0.332 (stratified_random) | 0.366 | 1.080 → 1.080 | 0.653 | 0.014 → 0.014 | 1.11 | 136 |
| C0_spread_only | F5 | 258,059 | 38,741 | 0.359 | 0.334 (stratified_random) | 0.365 | 1.089 → 1.089 | 0.659 | 0.018 → 0.016 | 1.09 | 147 |
| C1_moderate | F1 | 103,517 | 38,392 | 0.358 | 0.335 (stratified_random) | 0.367 | 1.081 → 1.080 | 0.653 | 0.013 → 0.009 | 1.23 | 79 |
| C1_moderate | F2 | 141,909 | 38,024 | 0.369 | 0.335 (stratified_random) | 0.371 | 1.064 → 1.064 | 0.642 | 0.027 → 0.028 | 1.01 | 96 |
| C1_moderate | F3 | 179,933 | 39,063 | 0.362 | 0.335 (stratified_random) | 0.368 | 1.068 → 1.068 | 0.645 | 0.029 → 0.025 | 0.94 | 113 |
| C1_moderate | F4 | 218,996 | 39,063 | 0.360 | 0.334 (stratified_random) | 0.366 | 1.074 → 1.074 | 0.649 | 0.023 → 0.015 | 0.85 | 134 |
| C1_moderate | F5 | 258,059 | 38,741 | 0.358 | 0.332 (stratified_random) | 0.365 | 1.083 → 1.083 | 0.655 | 0.016 → 0.015 | 0.95 | 147 |
| C2_pessimistic | F1 | 103,517 | 38,392 | 0.357 | 0.335 (stratified_random) | 0.373 | 1.066 → 1.067 | 0.643 | 0.016 → 0.015 | 0.89 | 79 |
| C2_pessimistic | F2 | 141,909 | 38,024 | 0.372 | 0.336 (stratified_random) | 0.378 | 1.039 → 1.039 | 0.623 | 0.058 → 0.056 | 0.98 | 97 |
| C2_pessimistic | F3 | 179,933 | 39,063 | 0.363 | 0.333 (stratified_random) | 0.374 | 1.038 → 1.031 | 0.617 | 0.080 → 0.052 | 0.73 | 113 |
| C2_pessimistic | F4 | 218,996 | 39,063 | 0.364 | 0.332 (stratified_random) | 0.377 | 1.052 → 1.049 | 0.627 | 0.063 → 0.018 | 0.60 | 133 |
| C2_pessimistic | F5 | 258,059 | 38,741 | 0.361 | 0.330 (stratified_random) | 0.374 | 1.066 → 1.067 | 0.641 | 0.041 → 0.016 | 0.69 | 146 |

## 8. Aggregate development performance with uncertainty (paired block bootstrap, 95%)

| scenario | best baseline | candidate macro-F1 | baseline macro-F1 | Δ macro-F1 [95% CI] | log loss cand / prior | Δ log loss vs prior [95% CI] (negative = better) | folds beating baseline |
|---|---|---|---|---|---|---|---|
| C0_spread_only | stratified_random | 0.363 | 0.335 | +0.029 [+0.025, +0.032] | 1.081 / 1.026 | +0.055 [+0.053, +0.058] | 5/5 |
| C1_moderate | stratified_random | 0.362 | 0.334 | +0.028 [+0.025, +0.031] | 1.074 / 0.977 | +0.097 [+0.094, +0.100] | 5/5 |
| C2_pessimistic | stratified_random | 0.364 | 0.333 | +0.031 [+0.028, +0.034] | 1.051 / 0.867 | +0.184 [+0.180, +0.187] | 5/5 |

**Exploratory: probabilities after prior-shift correction (decisions unchanged).**

| scenario | log loss cand / prior | Δ log loss vs prior [95% CI] | folds with skill |
|---|---|---|---|
| C0_spread_only | 1.018 / 1.026 | -0.0081 [-0.0094, -0.0066] | 5/5 |
| C1_moderate | 0.969 / 0.977 | -0.0077 [-0.0090, -0.0064] | 5/5 |
| C2_pessimistic | 0.857 / 0.867 | -0.0099 [-0.0111, -0.0085] | 5/5 |

Exploratory verdict if G2 used prior-corrected probabilities (NOT the pre-registered verdict): **PREDICTIVE SIGNAL DETECTED**.


**Robustness: non-overlapping 60 s evaluation, and direction-only accuracy.**

| scenario | candidate macro-F1 (non-overlapping 60 s rows) | stratified-random macro-F1 | delta | direction accuracy among true-LONG/SHORT rows predicted LONG/SHORT (0.5 = no directional information) | LONG precision / LONG base rate |
|---|---|---|---|---|---|
| C0_spread_only | 0.364 | 0.333 | +0.030 | 0.503 | 0.261 / 0.237 |
| C1_moderate | 0.361 | 0.334 | +0.028 | 0.503 | 0.236 / 0.210 |
| C2_pessimistic | 0.364 | 0.334 | +0.030 | 0.504 | 0.197 / 0.164 |

**What the signal is.** Among rows whose true class is LONG or SHORT and that the model also calls LONG or SHORT, the call is right only ~50% of the time (0.503 / 0.503 / 0.504): **the model carries no directional information.** LONG precision (0.26 / 0.24 / 0.20) is barely above the LONG base rate (0.24 / 0.21 / 0.16). The macro-F1 lift over a random classifier comes from separating NO_TRADE rows from trade rows, i.e. predicting whether a 1-sigma barrier will be reached within 60 s; this matches the importance ranking (volatility level: atr14_bps_*, realized_vol_bps_*, higher-timeframe ranges) and the small effect of removing any single feature group. It is volatility/timeout predictability, not a directional edge, and because the barrier itself is scaled by trailing volatility part of it may be a property of the label construction.


## 9. Calibration, confusion matrices, class-specific performance

| scenario | probabilities | log loss | Brier | top-label ECE |
|---|---|---|---|---|
| C0_spread_only | uncalibrated | 1.082 | 0.655 | 0.010 |
| C0_spread_only | temperature-calibrated | 1.081 | 0.654 | 0.011 |
| C1_moderate | uncalibrated | 1.074 | 0.649 | 0.015 |
| C1_moderate | temperature-calibrated | 1.074 | 0.649 | 0.014 |
| C2_pessimistic | uncalibrated | 1.052 | 0.633 | 0.050 |
| C2_pessimistic | temperature-calibrated | 1.051 | 0.630 | 0.026 |

**C0_spread_only** top-label reliability (equal-mass bins, mean confidence → accuracy): 0.34→0.34; 0.35→0.35; 0.36→0.35; 0.37→0.36; 0.38→0.39; 0.40→0.40; 0.42→0.43; 0.54→0.60

**C1_moderate** top-label reliability (equal-mass bins, mean confidence → accuracy): 0.34→0.35; 0.36→0.35; 0.37→0.37; 0.38→0.39; 0.40→0.40; 0.41→0.43; 0.44→0.47; 0.58→0.64

**C2_pessimistic** top-label reliability (equal-mass bins, mean confidence → accuracy): 0.35→0.36; 0.37→0.38; 0.39→0.40; 0.41→0.42; 0.43→0.46; 0.46→0.49; 0.50→0.57; 0.67→0.73

**C0_spread_only** (pooled over folds; rows = true, columns = predicted: SHORT / NO_TRADE / LONG)

| true \ pred | SHORT | NO_TRADE | LONG | recall |
|---|---|---|---|---|
| SHORT | 14,350 | 17,593 | 15,220 | 0.304 |
| NO_TRADE | 26,143 | 47,037 | 27,084 | 0.469 |
| LONG | 13,731 | 17,186 | 14,939 | 0.326 |
| precision | 0.265 | 0.575 | 0.261 | |

F1 per class: SHORT 0.283, NO_TRADE 0.517, LONG 0.290; macro-F1 0.363, balanced accuracy 0.366, accuracy 0.395. Predicted-class frequency 0.281 / 0.423 / 0.296 vs actual 0.244 / 0.519 / 0.237.

**C1_moderate** (pooled over folds; rows = true, columns = predicted: SHORT / NO_TRADE / LONG)

| true \ pred | SHORT | NO_TRADE | LONG | recall |
|---|---|---|---|---|
| SHORT | 11,727 | 17,169 | 12,775 | 0.281 |
| NO_TRADE | 26,546 | 56,344 | 28,227 | 0.507 |
| LONG | 11,345 | 16,453 | 12,697 | 0.314 |
| precision | 0.236 | 0.626 | 0.236 | |

F1 per class: SHORT 0.257, NO_TRADE 0.560, LONG 0.270; macro-F1 0.362, balanced accuracy 0.367, accuracy 0.418. Predicted-class frequency 0.257 / 0.465 / 0.278 vs actual 0.216 / 0.575 / 0.210.

**C2_pessimistic** (pooled over folds; rows = true, columns = predicted: SHORT / NO_TRADE / LONG)

| true \ pred | SHORT | NO_TRADE | LONG | recall |
|---|---|---|---|---|
| SHORT | 8,894 | 14,260 | 9,421 | 0.273 |
| NO_TRADE | 28,059 | 71,909 | 29,007 | 0.558 |
| LONG | 8,586 | 13,725 | 9,422 | 0.297 |
| precision | 0.195 | 0.720 | 0.197 | |

F1 per class: SHORT 0.228, NO_TRADE 0.628, LONG 0.237; macro-F1 0.364, balanced accuracy 0.376, accuracy 0.467. Predicted-class frequency 0.236 / 0.517 / 0.248 vs actual 0.169 / 0.667 / 0.164.

NO_TRADE is first-class: its precision/recall/F1 are reported above separately from LONG/SHORT; no PnL-optimised threshold was used. Pre-registered selectivity study (max-probability thresholds 0.40/0.50/0.60) is stored per fold in `result.json` (descriptive).

## 10. Feature importance, stability, drift, proxies

**chosen** (CORE / lightgbm / n_estimators=400, num_leaves=31, scenario C1_moderate): group permutation Δlog-loss micro +0.0160, volatility +0.0108, spread +0.0025, session +0.0024, mean_reversion +0.0024, momentum -0.0011, mtf -0.0012, structure -0.0044; top built-in importance: `atr14_bps_60s` 0.028, `htf_range_bps_300s` 0.023, `htf_ret_bps_300s` 0.022, `atr14_bps_300s` 0.021, `htf_close_dist_bps_300s` 0.020, `realized_vol_bps_300s` 0.018, `range_percentile_60s_1h` 0.018, `spread_mean_points_300s` 0.017; rank-Spearman between folds mean 0.97 (min 0.95), top-10 overlap 0.81. Top permutation features: `atr14_bps_60s` +0.0069, `htf_close_dist_bps_300s` +0.0024, `htf_range_bps_300s` +0.0019, `realized_vol_bps_60s` +0.0008, `realized_vol_bps_15s` +0.0008, `abs_ret_bps_300s` +0.0003, `directional_persistence_60s` +0.0002, `atr14_bps_300s` +0.0002.

Drift (PSI train→validation, mean over folds): 3 features > 0.10, 2 > 0.25; top: `relative_spread_bps` 0.8692, `directional_persistence_60s` 0.2887, `directional_persistence_15s` 0.1298, `realized_vol_bps_300s` 0.0987, `spread_mean_points_300s` 0.0811, `realized_vol_bps_60s` 0.0773. Feed boundary (pre 08-10..09-04 vs post 09-07..09-11): 1 features with PSI > 0.25; top `relative_spread_bps` 0.8086, `directional_persistence_60s` 0.2321, `directional_persistence_15s` 0.1281, `realized_vol_bps_300s` 0.0813, `down_tick_ratio_60s` 0.0737. Fold F5 pre vs post-boundary rows: pre: n=26,038 macro-F1 0.357 log loss 1.084; post: n=12,703 macro-F1 0.360 log loss 1.083.

**best_logistic** (CORE / logistic / C=1.0, class_weight=balanced, scenario C1_moderate): group permutation Δlog-loss volatility +0.0094, micro +0.0070, momentum +0.0057, spread +0.0043, structure +0.0036, mtf +0.0024, session +0.0023, mean_reversion +0.0019; top built-in importance: `atr14_bps_60s` 0.155, `realized_vol_bps_300s` 0.080, `realized_vol_bps_60s` 0.075, `is_new_york` 0.073, `atr14_bps_15s` 0.064, `is_london_new_york_overlap` 0.047, `atr14_bps_300s` 0.044, `spread_points` 0.041; rank-Spearman between folds mean 0.82 (min 0.73), top-10 overlap 0.79. Top permutation features: `atr14_bps_60s` +0.0221, `atr14_bps_15s` +0.0056, `realized_vol_bps_60s` +0.0056, `realized_vol_bps_300s` +0.0049, `is_london_new_york_overlap` +0.0027, `atr14_bps_300s` +0.0020, `htf_range_bps_300s` +0.0019, `abs_ret_bps_300s` +0.0018.

Drift (PSI train→validation, mean over folds): 3 features > 0.10, 2 > 0.25; top: `relative_spread_bps` 0.8692, `directional_persistence_60s` 0.2887, `directional_persistence_15s` 0.1298, `realized_vol_bps_300s` 0.0987, `spread_mean_points_300s` 0.0811, `realized_vol_bps_60s` 0.0773. Feed boundary (pre 08-10..09-04 vs post 09-07..09-11): 1 features with PSI > 0.25; top `relative_spread_bps` 0.8086, `directional_persistence_60s` 0.2321, `directional_persistence_15s` 0.1281, `realized_vol_bps_300s` 0.0813, `down_tick_ratio_60s` 0.0737. Fold F5 pre vs post-boundary rows: pre: n=26,038 macro-F1 0.354 log loss 1.099; post: n=12,703 macro-F1 0.359 log loss 1.094.

**best_lightgbm** (CORE / lightgbm / n_estimators=400, num_leaves=31, scenario C1_moderate): group permutation Δlog-loss micro +0.0160, volatility +0.0108, spread +0.0025, session +0.0024, mean_reversion +0.0024, momentum -0.0011, mtf -0.0012, structure -0.0044; top built-in importance: `atr14_bps_60s` 0.028, `htf_range_bps_300s` 0.023, `htf_ret_bps_300s` 0.022, `atr14_bps_300s` 0.021, `htf_close_dist_bps_300s` 0.020, `realized_vol_bps_300s` 0.018, `range_percentile_60s_1h` 0.018, `spread_mean_points_300s` 0.017; rank-Spearman between folds mean 0.97 (min 0.95), top-10 overlap 0.81. Top permutation features: `atr14_bps_60s` +0.0069, `htf_close_dist_bps_300s` +0.0024, `htf_range_bps_300s` +0.0019, `realized_vol_bps_60s` +0.0008, `realized_vol_bps_15s` +0.0008, `abs_ret_bps_300s` +0.0003, `directional_persistence_60s` +0.0002, `atr14_bps_300s` +0.0002.

Drift (PSI train→validation, mean over folds): 3 features > 0.10, 2 > 0.25; top: `relative_spread_bps` 0.8692, `directional_persistence_60s` 0.2887, `directional_persistence_15s` 0.1298, `realized_vol_bps_300s` 0.0987, `spread_mean_points_300s` 0.0811, `realized_vol_bps_60s` 0.0773. Feed boundary (pre 08-10..09-04 vs post 09-07..09-11): 1 features with PSI > 0.25; top `relative_spread_bps` 0.8086, `directional_persistence_60s` 0.2321, `directional_persistence_15s` 0.1281, `realized_vol_bps_300s` 0.0813, `down_tick_ratio_60s` 0.0737. Fold F5 pre vs post-boundary rows: pre: n=26,038 macro-F1 0.357 log loss 1.084; post: n=12,703 macro-F1 0.360 log loss 1.083.

**Month-identifiability probe** (CORE features → calendar month, day-grouped 5-fold CV): accuracy 0.664 vs majority 0.200 / uniform 0.167; top month-identifying features: `relative_spread_bps`, `spread_points`, `directional_persistence_60s`, `realized_vol_bps_300s`, `atr14_bps_300s`, `tick_count_60s`.

## 11. Proxy controls and ablations (focus candidate)

| control model (LightGBM, leading hyper-parameters) | scenario | macro-F1 | log loss |
|---|---|---|---|
| session_only | C0_spread_only | 0.329 | 1.097 |
| session_only | C1_moderate | 0.321 | 1.095 |
| session_only | C2_pessimistic | 0.300 | 1.092 |
| spread_only | C0_spread_only | 0.337 | 1.093 |
| spread_only | C1_moderate | 0.335 | 1.089 |
| spread_only | C2_pessimistic | 0.326 | 1.081 |

Proxy rule: (a) control lift ≥ 80% of candidate lift: **False**; (b) session+spread ≥ 50% of group-permutation Δ log-loss: **False** (shares {'C0_spread_only': 0.2305178858884386, 'C1_moderate': 0.1507991586568811, 'C2_pessimistic': 0.11371598901014478}). Flag: **False**.


| ablation (pre-registered list) | scenario | macro-F1 | Δ vs full | folds worse | Δ log loss |
|---|---|---|---|---|---|
| remove `spread` (11 features) | C0_spread_only | 0.363 | +0.0008 | 2/5 | +0.0023 |
| remove `spread` (11 features) | C1_moderate | 0.360 | -0.0010 | 2/5 | +0.0030 |
| remove `spread` (11 features) | C2_pessimistic | 0.361 | -0.0021 | 4/5 | +0.0035 |
| remove `session` (5 features) | C0_spread_only | 0.361 | -0.0013 | 4/5 | -0.0005 |
| remove `session` (5 features) | C1_moderate | 0.359 | -0.0020 | 4/5 | -0.0006 |
| remove `session` (5 features) | C2_pessimistic | 0.362 | -0.0010 | 3/5 | -0.0017 |
| remove `mtf` (15 features) | C0_spread_only | 0.360 | -0.0025 | 5/5 | +0.0007 |
| remove `mtf` (15 features) | C1_moderate | 0.359 | -0.0021 | 4/5 | +0.0006 |
| remove `mtf` (15 features) | C2_pessimistic | 0.360 | -0.0037 | 5/5 | +0.0022 |
| remove `regime` | - | not in the focus configuration | | |
| remove `momentum` (14 features) | C0_spread_only | 0.361 | -0.0018 | 5/5 | +0.0001 |
| remove `momentum` (14 features) | C1_moderate | 0.360 | -0.0015 | 3/5 | +0.0002 |
| remove `momentum` (14 features) | C2_pessimistic | 0.361 | -0.0020 | 5/5 | +0.0008 |
| remove `mean_reversion` (6 features) | C0_spread_only | 0.362 | -0.0009 | 3/5 | -0.0001 |
| remove `mean_reversion` (6 features) | C1_moderate | 0.361 | -0.0008 | 3/5 | -0.0000 |
| remove `mean_reversion` (6 features) | C2_pessimistic | 0.362 | -0.0009 | 4/5 | +0.0005 |
| remove `structure` (21 features) | C0_spread_only | 0.361 | -0.0019 | 4/5 | +0.0003 |
| remove `structure` (21 features) | C1_moderate | 0.362 | +0.0001 | 2/5 | +0.0003 |
| remove `structure` (21 features) | C2_pessimistic | 0.361 | -0.0023 | 4/5 | +0.0013 |
| remove `micro` (28 features) | C0_spread_only | 0.359 | -0.0033 | 5/5 | +0.0022 |
| remove `micro` (28 features) | C1_moderate | 0.358 | -0.0032 | 4/5 | +0.0023 |
| remove `micro` (28 features) | C2_pessimistic | 0.360 | -0.0037 | 5/5 | +0.0045 |

## 12. Computational performance

| kind | model | features | experiments | mean total fit+predict s (5 folds) | peak RSS MB |
|---|---|---|---|---|---|
| ablation | lightgbm | CORE | 21 | 211 | 1124 |
| baseline | majority | - | 3 | 0 | 497 |
| baseline | mean_reversion | - | 3 | 0 | 497 |
| baseline | momentum | - | 3 | 0 | 497 |
| baseline | stratified_random | - | 3 | 0 | 497 |
| control | lightgbm | CORE | 6 | 40 | 778 |
| model | lightgbm | CONDITIONAL | 12 | 381 | 969 |
| model | lightgbm | CORE | 12 | 358 | 920 |
| model | lightgbm | REGIME | 12 | 357 | 969 |
| model | logistic | CONDITIONAL | 18 | 145 | 969 |
| model | logistic | CORE | 18 | 136 | 920 |
| model | logistic | REGIME | 18 | 139 | 969 |

Matrix build: 115 days × ~3.4 s; grid: three scenario processes in parallel (2 threads each), ≈ 2 h wall-clock.

## 13. Failures and warnings

Failed experiments: 0. The first sequential grid run was stopped after one fold (too slow) and restarted as three parallel scenario processes with identical settings (10 incomplete directories removed; no completed experiment was touched). The matrix manifest step failed once on a JSON-serialization bug (fixed; day files unaffected). scikit-learn emitted lbfgs ConvergenceWarnings for some unregularised-ish logistic fits (max_iter 300, tol 1e-3, as pre-registered); predictions from those fits were used as produced.

## 14. Limitations

1. Predictive classification only: macro-F1 above the stratified-random baseline says nothing about tradable edge; costs C1/C2 are hypothetical and commission, slippage and latency are unknown.
2. 6 months, one DEMO feed, summer time only, 115 development days; the post-feed-boundary regime has 5 development days.
3. The 30 s grid keeps 50% overlap between neighbouring 60 s labels; the block bootstrap handles dependence but the effective sample size remains far below the row count.
4. Protocol v1 gate G2 is mechanically incompatible with class-weighted probability outputs (section 6); this was found after seeing results and is reported, not repaired.
5. Single seed (7); LightGBM thread count fixed at 2 for every grid experiment.
6. Permutation importance breaks feature correlations; it indicates dependence, not causation.

## 15. Final gate

**NO RELIABLE SIGNAL DETECTED** (pre-registered rule, development evidence only).

