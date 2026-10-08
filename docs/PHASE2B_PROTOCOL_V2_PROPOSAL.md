# Phase 2B Protocol v2 - PROPOSAL (DRAFT, NOT ACTIVE, NOT EXECUTED)

**Status.** This is a design proposal for review. Nothing in it has been run: no v2 model was trained, no v2 label was generated, no signal was produced, no strategy was
back-tested, nothing was connected to MT5 execution, and the final holdout (2026-09-14..2026-10-06) was not touched. The machine-readable drafts in
`research/phase2b/protocol_v2_draft/` (**draft_hash `149968e332bc5c3fcdf5b4264dbbe4b6add3b2978a0dfcdb9242aab79f57dfb3`**) are marked `DRAFT_NOT_ACTIVE`;
`fxscalp.research.protocol_v2.draft.assert_active()` raises until an activation file exists, and that file is created only after your explicit authorization (section 11).

**Objective.** Determine whether XAUUSD contains a *reproducible short-horizon directional prediction signal*, separately from volatility/opportunity prediction.

**Disclosure - read first.** Protocol v2 is a **revised protocol informed by the findings of Protocol v1 on the same development data**. It is **not an independent
confirmatory study**. The development window (2026-04-06..2026-09-11) has been examined extensively; v2 results on it carry an inflated false-positive risk that the
stricter gates only partly offset. The only data not used for any design decision is the locked final holdout - and even that was covered by the descriptive whole-dataset
profile published before the holdout protocol existed (no labels, features, predictions or outcomes inside it were ever inspected).

---

## 1. Summary

| | |
|---|---|
| v1 formal result | **NO RELIABLE SIGNAL DETECTED** (preserved; sealed and hashed; never reclassified) |
| What the v1 data actually showed | a small, consistent *volatility/timeout* predictability lift (macro-F1 0.363 vs 0.334) with **no directional information** (50.3% direction accuracy) |
| Main v1 defect | a methodological gate/calibration conflict: class weighting made probabilities prior-shifted, so gate G2 failed mechanically for every weighted model |
| v2 core idea | train **unweighted** binary models (proper scoring), calibrate with **nested out-of-fold** predictions, decide with a **separate** rule; evaluate **direction (Task A)** and **opportunity (Task B)** independently |
| v2 label-construction question | is `tb_H60` mostly a volatility/timeout problem? controlled comparison of volatility-scaled vs fixed-distance vs cost-normalized barriers |
| v2 anti-fingerprint design | month-stratified skill, feed-boundary split, regime-transfer folds, NEUTRAL and TRAILING_NORM feature sets, adversarial validation, within-day placebo |
| v2 budget | 7 model configurations x 3 feature configurations = 21 per label set (v1: 30); 84 screening experiments (+ confirmation/controls) |
| expected cost | roughly 4-5 hours wall-clock on this 8-core machine (estimate, section 8) |

## 2. What happened in v1

**Facts (all development folds, all three cost scenarios, from the sealed manifest):**
- 30 candidates per scenario (logistic + LightGBM; CORE / CONDITIONAL / REGIME) + 4 baselines + 6 controls + 21 ablations = 129 experiments, 0 failures.
- Class-weighted models beat the best baseline's macro-F1 in 5/5 folds (best exploratory candidate 0.363 vs stratified random 0.334, +0.029, 95% block-bootstrap CI +0.025..+0.032) but their log loss (~1.07) was worse than the training prior's (0.957) -> gate G2 failed; unweighted logistic models had real probabilistic skill (0.945 vs 0.957) but never predicted LONG/SHORT -> gates G1/G3 failed. **No candidate passed.**
- Post-hoc (explicitly **exploratory**, marked as such in the sealed manifest and the v1 report): prior-corrected log loss was 0.008-0.010 nats better than the prior in 5/5 folds; proxy controls (session-only, spread-only) added essentially nothing (lift <= 0.003); no single feature group carried the signal (largest ablation loss 0.0037 macro-F1).
- Direction: among rows whose true class is LONG/SHORT and that the model also called LONG/SHORT, it was right **50.3%** of the time (0.503/0.503/0.504 across scenarios). LONG precision (0.26/0.24/0.20) was barely above the base rate (0.24/0.21/0.16).
- Importance ranking was dominated by volatility level (`atr14_bps_60s`, `realized_vol_bps_*`, higher-timeframe ranges); permutation importance by group: micro +0.016, volatility +0.011 log-loss.
- The features identified the calendar month with **66.4%** accuracy (majority 20%, uniform 16.7%); top fingerprints `relative_spread_bps`, `spread_points`, `directional_persistence_60s`, `realized_vol_*`, `atr14_bps_300s`, `tick_count_60s`. Train-to-validation PSI > 0.25 for 2 features (`relative_spread_bps` 0.87, `directional_persistence_60s` 0.29).
- Label diagnostics (development data): under spread-only cost, `fh`/`opp` labels were near-degenerate (NO_TRADE 0.3-6.7% at H>=60); `tb` was balanced and non-degenerate.
- Dependence: ACF of the NO_TRADE indicator on the 30 s grid sums to 0.45 over lags 1-24 -> an effective-sample-size factor of only ~0.53 *for the label alone*; v1 did not estimate n_eff for the loss series and used a single block length (600 s).

### 2.1 Methodological problems (not evidence about the market)
1. **Gate G2 vs class weighting.** A model trained with weights 1/(K pi) estimates p(k|x)/pi_k (implied uniform prior); its log loss exceeds the prior's regardless of informativeness; one-parameter temperature scaling cannot move a prior (demonstrated exactly in `tests/unit/test_protocol_v2_draft.py`).
2. **Calibration too weak and too late.** Temperature-only, fit on the last 20% of a training block, and coupled to a class-weighted objective.
3. **One 3-class metric for two different questions.** Macro-F1 mixes direction and opportunity (NO_TRADE vs trade); a classifier can win macro-F1 by predicting volatility alone.
4. **Volatility-scaled barrier.** B = sigma*sqrt(H) makes the label NO_TRADE roughly "will realized movement exceed *trailing* volatility within 60 s?" - a vol-of-vol / timeout question by construction.
5. **Ablation list omitted the volatility group** (so the dominant group was never removed).
6. **Fingerprint checks were descriptive, not gates**; no placebo; no regime-transfer fold; no neutralised feature set.
7. **Uncertainty:** single block length, no n_eff, no day-cluster bootstrap, no multiplicity control across 30 candidates x 3 scenarios.
8. **Selection among 30 candidates by a composite score** with optimism not accounted for.

### 2.2 Genuinely informative findings
- On this data there is no evidence of *directional* information in the 108 frozen features at 60 s (50.3% direction accuracy), although a volatility/timeout lift exists.
- The lift is distributed (no single group), consistent across folds/scenarios and survives both the non-overlapping evaluation and the session-only/spread-only controls - but the features can also fingerprint the month, so period-specific structure cannot be excluded.
- Spread level/volatility structure changed between April-May and June onward; the feed changed on 2026-09-06.
- Costs matter: label balance moves strongly with the (unknown) cost scenario.

## 3. Protocol v1 vs proposed v2

| aspect | v1 (sealed) | v2 (proposed) | why |
|---|---|---|---|
| question | one 3-class label `tb_H60` (LONG/SHORT/NO_TRADE) | **two binary tasks**: A direction, B opportunity | direction and opportunity are different questions |
| training loss | class-weighted ('balanced') for most models | **unweighted**, binary log loss | proper scoring; probabilities estimate p(y|x) |
| calibration | temperature, last 20% of training block | **nested out-of-fold**; candidates none/temperature/Platt/isotonic chosen by inner OOF log loss | valid and data-appropriate; no outer-validation use |
| decision rule | argmax of (weighted, calibrated) probs | **separate**: p > 0.5 and p > training base rate (balanced point) | decouples decisions from probabilities |
| log-loss gate | log loss < training prior (conflicts with weights) | log-loss **skill** vs training base rate on *calibrated* probs, with effect floors and CIs | gate valid for unweighted models |
| barrier | volatility-scaled only | volatility-scaled **vs** fixed 150 pt **vs** cost-normalized 16x | tests the timeout/volatility confound |
| direction | implicit inside macro-F1 | **Task A**: cost-free mid direction U, plus executable A-exec and magnitude strata | isolates direction |
| selection screening | composite S with ECE/log loss on weighted probs | **calibration-invariant** screening (AUC), then calibrated confirmation | avoids calibration artefacts in selection |
| model budget | 10 hp sets x 3 feature configs = 30 per scenario | **7 x 3 = 21** per label set; cap 24; min_child_samples 1000 | smaller family, less fingerprint-fitting |
| feature configs | CORE / CONDITIONAL / REGIME | **RAW / NEUTRAL (-10 fingerprint features) / TRAILING_NORM** | targets period fingerprints |
| uncertainty | one block length (600 s), B=1000 | block 600 **and** 1800 s, day-cluster bootstrap, B=2000, **n_eff**, min n_eff per fold | dependence-aware |
| multiplicity | none | Task B: Bonferroni-equivalent 98.33% CIs over 3 variants; effect floors | development data reused |
| fingerprint tests | descriptive probe | **gates** FP1-FP6 (month-stratified, feed boundary, regime transfer, proxy removal, adversarial, placebo) | survive shifts or fail |
| ablations | 7 groups (no volatility) | 8 groups **including volatility** | signal provenance |
| verdicts | one 3-level verdict | per task: DIRECTIONAL SIGNAL / OPPORTUNITY SKILL DETECTED, WEAK/UNSTABLE, NONE | no cross-interpretation |
| costs | C0/C1/C2 | unchanged, labelled hypothetical; broker details list | no profitability claims |
| holdout | locked | **locked, no access, no automatic unlock** | unchanged |
| status | pre-registered then run | **draft; inactive until authorized** | review first |

## 4. Probability calibration (design and mathematics)

**Principle.** Separate *probability estimation* from the *decision rule*. v2 models are unweighted, so their outputs estimate p(y|x) directly; decisions use thresholds
(0.5 and the training base rate). That removes the v1 failure mode by construction.

**Methods compared conceptually (none chosen from v1's exploratory best result):**

| method | what it fixes | requirements / risks | verdict for v2 |
|---|---|---|---|
| Elkan prior correction | an exact prior shift from class weights or a changed base rate | exact only if likelihood ratios are right and the implied prior is exactly uniform; does nothing for over/under-confidence | not needed (no weights); kept only as a conceptual check |
| temperature scaling | global over/under-confidence | 1 parameter, cannot fix prior or class-specific bias | candidate |
| Platt scaling (binary) | scale + bias on the logit | 2 parameters, monotone, stable at 100k rows | candidate |
| vector/matrix/Dirichlet scaling | class-specific distortions in K>=3 | K^2+K parameters; needless for binary tasks | not used (tasks are binary) |
| isotonic regression | arbitrary monotone distortion | needs many positives, step artefacts, overfits small samples | candidate |
| class-weighted training + correction | trades calibration for recall | couples objective and decision; needs a correct implied prior | rejected |

**Selection among the four candidate calibrators (none/temperature/Platt/isotonic) is automatic per outer fold on inner out-of-fold log loss only** (ties within
0.0005 nats -> simpler). **Nested scheme:** inside each outer training block, 4 time-ordered expanding inner blocks with the frozen purge/embargo; inner block j>=2 is predicted by a model fit on blocks 1..j-1; calibrators are fit on the pooled inner OOF predictions; the base model is then refit on the whole outer training block and predicts the outer validation block, to which the selected calibrator is applied. **The outer validation block never fits the base model, preprocessing, feature normalisation or calibrator.** Known approximation: inner models see less data, so inner OOF probabilities are slightly less confident than final ones; residual miscalibration is *measured* on the outer validation fold by diagnostic gates (ECE <= 0.02, slope in [0.9, 1.1], |intercept| <= 0.10), which are used only to diagnose - never to refit and re-score. A failed calibration gate downgrades log-loss/Brier evidence to "indicative only"; discrimination (AUC, balanced accuracy) is unaffected.

## 5. Task A - direction

- **Target U:** mid(t+L+H) > mid(t+L) (rows with equality dropped and counted). Primary H=60 s, L=1 s (a hypothetical, uncalibrated latency); sensitivity L=0; robustness H=15 and 300 s **only for the already-selected candidate**.
- **Why mid:** a cost-free target removes barrier width, volatility scaling and NO_TRADE from the definition. Executable assumptions enter in the *secondary* A-exec target (v1 fixed-horizon label LONG/SHORT under scenario s, entry at the ask/bid, exit at the bid/ask, net of the scenario's extra cost) and in magnitude strata (realized |move| in multiples of entry cost).
- **Selection accounting.** The gate is **unconditional (100% coverage)**. A-exec and magnitude-stratified accuracy are reported with their coverage and an explicit caveat: the conditioning event is a *future* outcome, so such accuracy is an upper-bound diagnostic. The pre-registered abstention study (|p-0.5| < delta, delta in {0.02, 0.05}) only counts if it beats both random abstention and volatility-driven abstention at equal coverage; the unconditional metric is always printed next to it.
- **Ambiguous paths:** the endpoint target has none; for the barrier-direction secondary target, same-second ties are dropped and counted (sensitivity: assigned conservatively).
- **Class balance:** P(U=1) per fold is reported; balanced accuracy and AUC are primary; log loss is compared with the training base rate.
- **Baselines:** training base rate, momentum, mean-reversion, coin flip (seed 7).
- **Directional-skill gate (all must hold for DETECTED):** D1 pooled balanced accuracy >= 0.505 with lower 95% CI bound of (BA-0.5) > 0 under both block lengths and the day-cluster bootstrap; D2 calibrated log-loss skill > 0, lower CI > 0, point >= 0.0003 nats; D3 BA > 0.5 in >= 4/5 folds; D4 the NEUTRAL **and** TRAILING_NORM versions retain >= 50% of the RAW lift with lower CI > 0; D5 positive in >= 3 of 4 validation months and none significantly negative; D6 evaluated at 100% coverage; D7 the within-day placebo shows no skill. WEAK/UNSTABLE: pooled point estimate positive and >= 4 of 7 hold.

## 6. Task B - opportunity

- **Target E:** the LONG or SHORT profit barrier is hit strictly before its own stop within H on the executable bid/ask path (scenario cost widens the profit barrier); same-second profit+stop -> stop first; both sides hit -> E=1, direction ambiguous (counted).
- **Reported:** event probability, calibration, Brier, log loss, AUC, average precision, precision/recall at tau = training base rate and 0.5, and performance by train-defined volatility terciles, spread terciles and session. **Task B skill is never interpreted as directional skill.**
- **Opportunity-skill gate:** O1 AUC >= 0.55 with lower CI > 0.5 (98.33% interval); O2 calibrated log-loss skill with lower CI > 0 and point >= 0.005 nats; O3 ECE <= 0.02 and slope in [0.9, 1.1]; O4 AUC >= 0.52 in >= 2 of 3 volatility terciles **and** not VOLATILITY-DOMINATED; O5 AUC > 0.5 in >= 4/5 folds; O6 survives NEUTRAL and TRAILING_NORM (>= 50% of RAW lift).

### 6.1 Is the v1 target mostly a volatility/timeout problem? (label-construction comparison, pre-specified)
| variant | rule | anchor |
|---|---|---|
| V vol-scaled (v1) | B_i = max(1.0 sigma_i mid sqrt(H), 3 spread_i), sigma_i = trailing 600 s | v1 |
| F fixed | B = 150 points | ~ v1 mean H=60 barrier (148 pts), so event rates are comparable |
| N cost-normalized | B_i = 16 x (entry spread pts + extra_rt pts) | 16 x median C0 spread (9) ~ 144 pts |
Tests: **VT1** volatility-only model ratio R = lift(vol-only)/lift(full); **VT2** within-volatility-tercile AUC; **VT3** AUC of trailing sigma alone for E (descriptive, Stage S0). A variant is called VOLATILITY-DOMINATED when R >= 0.8 and skill fails in >= 2 of 3 terciles. Variants are reported side by side (Holm/Bonferroni-corrected); **none is chosen or dropped by trading profitability** (no PnL is computed anywhere in v2).

## 7. Temporal dependence

Chronological walk-forward only (the frozen F1-F5, segments, eligibility, purge, embargo; 30 s grid). Uncertainty: moving-block bootstrap with **600 s and 1800 s** blocks (B=2000, paired with the baseline; conclusions must hold under both) **and** a day-cluster bootstrap; the gate uses the most conservative interval. Effective sample size n_eff = n/(1+2 sum (1-k/(K+1)) rho_k) (Bartlett, K=120 lags = 1 h) of the per-row loss-difference series, reported per fold; a fold with n_eff < 500 is UNDERPOWERED and cannot cast a fold vote. v1 evidence: the label alone already has ESS factor ~0.53 at 30 s spacing.

## 8. Period fingerprints (all development data)
| id | test | pass |
|---|---|---|
| FP1 | skill by calendar month (Jun-Sep) | positive in >= 3/4 months, none significantly negative |
| FP2 | F5 validation split pre/post 2026-09-06 feed boundary | post-boundary point estimate >= 0, not significantly negative (5 days: underpowered, significance not required) |
| FP3 | (a) train Apr-May -> validate Jun-Aug; (b) train Jun-Aug -> validate late Aug-Sep 11; (c) reversed time (non-chronological diagnostic, excluded from gates) | (a),(b) positive and >= 50% of the main result |
| FP4 | NEUTRAL (drop the 10 v1 fingerprint features) and TRAILING_NORM (5-day causal per-feature normalisation) | >= 50% of RAW lift, CI > 0 |
| FP5 | adversarial validation (train vs validation rows) | descriptive; AUC > 0.9 flags severe shift |
| FP6 | within-day target permutation placebo (3 seeds) | placebo CI contains 0 |
The v1 month-identifiability probe is repeated on NEUTRAL and TRAILING_NORM (target: well below 0.664).

## 9. Costs

C0/C1/C2 are unchanged: observed spread is OBSERVED; commission, slippage and latency are UNKNOWN/UNCALIBRATED; C1/C2 values are hypothetical. Task A's primary target is cost-free; costs enter A-exec, magnitude strata and every Task B barrier. **No profitability claim is made.**
Broker/account details needed to establish realistic commissions: exact account type/server (the DEMO feed's account is not proof of live pricing); commission schedule (amount, currency, per lot per side vs round turn, tiers/rebates); XAUUSD contract specification (contract size, tick value, lot limits); raw vs marked-up spread structure and live-vs-demo spread comparison; swap rates; execution mode, slippage/requote/max-deviation policy; measured latency from the intended host to the trade server; news-time spread widening and stop levels; and an actual **fill sample** to estimate slippage (not available in this project; not to be generated without explicit authorization).

## 10. Models, search, selection, failure conditions
- Models: regularized logistic (C in {0.01, 0.1, 1}); LightGBM (num_leaves in {7, 15}, n_estimators in {100, 300}; lr 0.05, min_child_samples 1000, subsample/colsample 0.7, lambda 10, **unweighted**, deterministic, seed 7). 7 configs x 3 feature configs = **21 per label set** (cap 24, never expanded after results).
- Screening is calibration-invariant (mean fold AUC; gate: mean AUC > 0.5 and > 0.5 in >= 4/5 folds); score S = mean(AUC) - 0.5 - SD_folds(AUC); within 0.002 prefer NEUTRAL/TRAILING_NORM over RAW, then logistic, then fewer trees/leaves. Calibrated confirmation, gates, fingerprint tests, controls, ablations (incl. volatility) follow only for the selected candidate(s).
- Failure conditions (STOP): any fit on an outer-validation row, any holdout timestamp, spec/protocol/v1-seal hash mismatch, label-causality or segment-protection test failure, non-determinism, a fold with < 5,000 validation rows, > 10% unconverged fits, > 24 configurations attempted.

## 11. Proposed experiments and compute (estimates from v1 timings; not measured for v2)
| stage | content | experiments | est. wall-clock (8 cores) |
|---|---|---|---|
| S0 | new label matrix (U, E for V/F/N x C0/C1/C2, H15/60/300 for the selected) + label diagnostics (no models) | - | ~10-15 min build, < 5 min diagnostics |
| S1 | Task A screening | 21 | ~0.5 h |
| S2 | Task B screening (3 variants, C0) | 63 | ~1.5 h (about 1 h with 3 processes) |
| S3 | nested-OOF calibration, confirmatory metrics, bootstraps (A + 3 B variants) | 4 candidates x 5 outer x (4 inner + 1 final) fits | ~0.5-1 h |
| S4 | FP1-FP6, controls, ablations (8 groups), C1/C2 sensitivity | ~60-80 | ~1.5-2 h |
| S5 | verdicts, report, MODEL_CANDIDATE_V2 manifest (or NONE) | - | minutes |
Total roughly 4-5 h wall-clock; peak RSS ~1 GB; disk +1-2 GB (predictions/models; git-ignored). No GPU. No network. No MT5.

## 12. Anticipated limitations
1. v2 is not independent confirmation; the dev data has been reused (disclosed). 2. The directional prior from v1 is "no information at 60 s"; v2 is powered to detect an edge of about 0.5-1 percentage point balanced accuracy only if n_eff is adequate (typical n_eff will be a few thousand per fold, so a true edge below ~1 pp is unlikely to be resolved). 3. One DEMO feed, one instrument, ~5 months of development data, summer time only. 4. Costs are hypothetical; mid-direction skill does not imply executable profit. 5. NEUTRAL features were chosen using the v1 probe (informed). 6. TRAILING_NORM removes slow level information by design. 7. Fixed-barrier (150 pt) and cost-multiple (16x) values are anchored to v1 diagnostics; sensitivity to those anchors is not part of the budget. 8. The final holdout is a single 17-day window: one evaluation only. 9. Nested OOF calibration is an approximation (inner models smaller).

## 13. Holdout policy
The holdout stays LOCKED: no diagnostics, labels, features, predictions or metrics, and **no automatic unlock after v2 training**. Opening it requires a separate written authorization after v2 completes, a frozen MODEL_CANDIDATE_V2 manifest (model, hyper-parameters, features, preprocessing, calibrator, decision rule, seed) and the pre-registered gate outcome.

## 14. Deliverables and validation
- `research/phase2b/v1_immutable_manifest.json` (commit `abef322`): SHA-256 of 9 v1 artifacts and 129 experiments; marks the prior-corrected LightGBM analysis POST-HOC/EXPLORATORY.
- `research/phase2b/protocol_v2_draft/` (`protocol_v2_draft.json`, `calibration_v2.json`, `tasks_v2.json`, `evaluation_v2.json`, `fingerprint_tests_v2.json`, `cost_v2.json`, `draft_index.json`) - all `DRAFT_NOT_ACTIVE`.
- `src/fxscalp/research/protocol_v2/` (`draft.py` builder + static validator + activation guard; `stats.py` effective sample size) and `scripts/build_protocol_v2_draft.py`.
- `tests/unit/test_protocol_v2_draft.py` (28 tests): inactive status, deterministic regeneration, v1 artifacts unchanged, frozen spec/folds/embargo/holdout unchanged, no training/execution code path in the draft, 19 mutation cases the validator must catch, the class-weight/prior/temperature mathematics, ESS theory, budget arithmetic.

## 15. Precise authorization required before running v2
Reply with an explicit statement covering **all** of the following (anything omitted stays disallowed):
1. "Activate Protocol v2 draft_hash `149968e3...dfb3` as written (or with these listed changes: ...)."
2. "Run stages S0-S5 on the development folds only; the final holdout stays locked and is not read."
3. "Create the new label matrix and store experiment outputs under `data/research/` (about 1-2 GB)."
4. "Use up to ~5 hours of local compute."
5. (Optional, separate) "Permit/forbid regeneration of any v1 artifact" - the default is **forbid** (v1 is sealed).
Not covered by that authorization: opening the holdout, any backtest or PnL computation, signal generation, MT5 connection for orders, or collecting broker fill data.

## 16. Open questions for you
1. Accept the anchors (150 pt fixed barrier; 16x cost multiple; 10 NEUTRAL features)? 2. Is 60 s the right primary horizon for the intended execution style? 3. Do you want Task A restricted to H=60 (as drafted) or to add 15 s as a second *pre-registered* primary (costs one more gate and a stricter multiplicity correction)? 4. Can you supply the broker details listed in section 9?
