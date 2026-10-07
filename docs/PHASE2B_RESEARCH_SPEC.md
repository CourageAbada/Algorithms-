# Phase 2B-A research specification (FROZEN) - dataset, eligibility, segments, features, labels, costs, folds

**Status: specification and diagnostics only.** No model has been trained, no profitability measured, no signal generated, no order
placed. This document, together with the machine-readable files in `research/phase2b/` (index `spec_index.json`,
**spec_hash `57fb49f5eb10c5f0e8fd3a3c82c1c1c39603ee915be3cf510286c35475331845`**), is the contract for Phase 2B training. `fxscalp.research.spec.load_frozen_spec()` refuses to
load if any file was edited or if the code behind it no longer reproduces it (`SpecDriftError`); a design change therefore needs a
new spec version. Account identifiers and credentials are not part of any artifact.

## 1. Raw Dataset v1 - `XAUUSD_RAW_V1`
- Source: dataset `tickraw-fe5b9137d477c7c866d5` (dataset-manifest checksum `ac3c3d19d818101b...`), broker "Raw Trading Ltd", server `ICMarketsSC-Demo` (DEMO), symbol `XAUUSD`.
- 85,577,391 ticks, 190 source days (2026-03-31 .. 2026-10-06), 134 trading days. First/last normalized UTC: 2026-03-31T22:49:48.409Z / 2026-10-06T20:58:56.796Z. Schema `tick_raw/2`, quality policy `quality_policy/1` (sha256 `dbaf7e10c8247f6f...`).
- Time calibration `timecal-c9cc2a3a65a3dd80` (+10800 s, valid 2026-04-05T22:00:02Z .. 2026-10-07T19:23:06Z, DST unresolved), feature set `xauusd_core-fs1-ac928c2b1771`.
- **Freeze manifest** `research/phase2b/XAUUSD_RAW_V1.freeze.json`: `freeze_id freeze-48779c44e3f2999274a2`, code commit `e6e3e76`, 190 chunk entries each with raw-content hash and file checksums of the raw / tick_basic / tick_quality artifacts. Immutable chunks are REFERENCED by checksum; nothing is copied. `freeze_id` is deterministic (excludes the timestamp and commit).

## 2. Cross-day continuity (adjacent trading chunks)
133 boundaries between 134 trading days: 104 daily maintenance breaks (min 3661 s, median 3661 s, max 3784 s), 27 weekly closures (49.05-73.02 h; the longest is Easter), 2 US-holiday early closes (2026-05-25, 2026-09-07; 12,601-12,720 s). Result: **0** normalized or raw ordering violations, **0** identical ticks and **0** equal timestamps across a boundary, **0** normalization-rule shifts, **0** unexplained gaps; the only time-basis transition is the expected uncertain -> verified step (2026-04-02 -> 2026-04-06). Normal breaks and weekends are classified, not reported as missing data. (`scripts/check_continuity.py`, `market_data/continuity.py`; verdict OK, so the freeze proceeded.)

## 3. Phase 2B eligibility view (derived mask; raw data untouched)
EXCLUDE a tick if tagged `FUTURE_TICK, MISSING_ASK, MISSING_BID, NEGATIVE_SPREAD, NONFINITE, TIME_AMBIGUOUS, TIME_BASIS_UNCERTAIN, TIME_NONEXISTENT, TIME_REVERSAL`. RETAIN (tag stays, so later experiments test policies explicitly): `DUP_TICK, DUP_TIMESTAMP, EXTREME_SPREAD, LARGE_GAP, SEC_MSEC_MISMATCH, UNKNOWN_FLAG_BITS, ZERO_SPREAD`.
**85,061,230 of 85,577,391 ticks remain eligible (99.3968 %); 516,161 are excluded - all `TIME_BASIS_UNCERTAIN` (2026-04-01: 245,685 and 2026-04-02: 270,476).** FUTURE_TICK, TIME_REVERSAL, MISSING_BID/ASK, NEGATIVE_SPREAD, NONFINITE, TIME_NONEXISTENT/AMBIGUOUS occur 0 times. Development window: 115 trading days / 72,831,953 ticks; final holdout (counts only): 17 days / 12,229,277 ticks. View id `elig-3988b1382d2d68914e1e`; per-day `data/derived/eligibility/.../eligibility.parquet` (`eligible`, `exclusion_flags`, `segment_id`, `feed_regime`).

## 4. Segment policy (no bridging)
A gap longer than **300 s** between consecutive ELIGIBLE ticks starts a new segment; segment id = UTC ms of its first eligible tick. No rolling feature, return target, label, event, trade simulation or sequence may span a segment boundary (labels are computed per segment; Phase 2B features use `PipelineConfig(segment_gap_s=300)`).
Rationale (fixed before any model, not tuned): 300 s is the Phase-1 `LARGE_GAP` quality threshold; the largest ordinary inter-tick gap in the data is 27 s; the intra-session gaps are 871, 593, 1,341, 7,361 and 456 s (unexplained feed silences inside trading sessions); every threshold from 230 s to 450 s gives the identical segmentation (tested). Result: 137 segments over 132 eligible trading days, i.e. 5 intra-session boundaries:
- 2026-04-22: 06:50:33 -> 07:05:05 UTC (871 s)
- 2026-05-26: 16:44:09 -> 16:54:02 UTC (593 s)
- 2026-06-15: 09:13:39 -> 09:36:00 UTC (1341 s)
- 2026-06-29: 05:04:28 -> 07:07:09 UTC (7361 s)
- 2026-06-29: 12:33:40 -> 12:41:16 UTC (456 s)
The 2026-06-29 hole (**05:04:28 -> 07:07:09 UTC**) is a hard segment boundary; the day is NOT removed (it contributes 3 segments).

## 5. `FEED_REGIME_BOUNDARY` (2026-09-06T22:02:04.012Z = weekly open of source day 2026-09-07)
Raw flag bit `0x400` is absent on every earlier tick and present on every tick from this instant; `0x80` is present throughout. This is an observed change in the FEED. It does **not** show that the economic market regime changed. Descriptive pre/post comparison (development-visible days only; `diagnostics/feed_regime_comparison.json`):

| property | PRE (04-06..09-04) | PRE_MATCHED (08-10..09-04) | POST_DEV (09-07..09-11) |
|---|---|---|---|
| days | 110 | 20 | 5 |
| ticks/day (median) | 637,330 | 699,206 | 693,968 |
| tick rate Hz (median of days) | 7.708 | 8.457 | 8.4 |
| spread points: median / p95 / p99 (medians of daily) | 9 / 12 / 12 | 9 / 12 / 12 | 9 / 12 / 12 |
| max daily max spread (points) | 500 | 500 | 700 |
| duplicate-timestamp rate | 0.94 % | 1.41 % | 0.76 % |
| unknown flag bits (share of ticks) | {"0x80": 0.9781} | {"0x80": 0.9895} | {"0x400": 0.0117, "0x480": 0.9883} |
| P(bid changes) / P(ask changes) / P(both) | 0.977 / 0.976 / 0.954 | 0.988 / 0.987 / 0.976 | 0.987 / 0.986 / 0.974 |
| P(neither) / P(exactly one side) | 0.0015 / 0.044 | 0.0015 / 0.022 | 0.0007 / 0.025 |
| mean |mid change| when nonzero (points) | 3.55 | 3.23 | 2.87 |
| session share % ASIA / LON / NY / OVERLAP / OFF | 27.3 / 17.6 / 20.1 / 30.9 / 4.1 | 26.1 / 15.2 / 21.6 / 34.0 / 3.2 | 27.9 / 17.8 / 18.5 / 32.9 / 2.9 |
| median daily range bps / 1-min return std bps | 216 / 3.68 | 199 / 3.55 | 223 / 3.67 |

Feature missingness on matched sample days (4 pre, 3 post; Phase 2B pipeline config): all 119 features, max |delta NaN| = 1.091 pp, only the five session running-aggregate features exceed 1 pp (session_high_so_far, session_low_so_far, session_range_so_far_bps, dist_session_high_bps, dist_session_low_bps); 9 features have >1 % NaN pre and 9 post. Reading: apart from the flag bits, tick rate, spread distribution, quote-change frequencies, duplicate rate, session mix and feature missingness look alike; the one visible drift is a smaller mean mid change per tick (3.55 pre, 3.23 pre-matched, 2.87 post points) - on only 5 post days, so it is noted, not interpreted. The flag bits are **forbidden as features**: `flags`, `unknown_flag_bits`, `quality_flags` and every other raw/feed-internal column are in `FORBIDDEN_MODEL_COLUMNS`; `assert_model_inputs_allowed()` rejects them and a test proves all features are invariant to scrambled flags, `last`, volume and volume_real. Post-boundary data: 5 days in development (F5 validation only), the rest in the final holdout; no training fold contains post-boundary data.

## 6. Non-stationarity policy (enforced, tested)
Nothing learned from a distribution may be fit on the whole six months: scalers, normalization statistics, winsorization limits, volatility/spread/regime thresholds, outlier detectors, dimensionality reduction and feature selection are fit on a fold's TRAINING block only and merely APPLIED to validation/test. `research/preprocessing.py`: `FoldPreprocessor.fit` raises `LeakageError` if its data touch anything outside the training block (and there is deliberately no `fit_transform`), `train_only_quantile` / `train_only_baseline` do the same for thresholds and baselines, `assert_training_only` also rejects samples whose labels reach the embargo. Tests: parameters are unchanged when validation data are wrecked; contaminated fits and transform-before-fit raise. The regime codes are CONDITIONAL until their thresholds are refit train-only.

## 7. Prediction problem and candidate labels (specification; none selected by results)
Short-horizon XAUUSD scalping research. All labels use the executable bid/ask: a LONG buys at the ask and sells at the bid, a SHORT sells at the bid and buys at the ask; the entry spread is therefore paid inside the label, and **no label uses mid-price direction alone**.
Notation (per segment, 1-second grid): bar i ends at t_i (= the feature row timestamp; decision information <= t_i); `bid_c, ask_c` = prevailing quote at t_i; `bid_h/l, ask_h/l` = quote extremes inside bar i (empty second -> prevailing quote); latency L in {0,1} s (scenario); horizon H s; entry quote at t_(i+L): ask_e (LONG), bid_e (SHORT); window = bars i+L+1 .. i+L+H; a row is labelled only if i+L+H <= n-1 (else status BOUNDARY: the window would leave the segment); `label_end_ms = t_i + (L+H)*1000` (used for purge). x = extra round-trip cost in points (scenario).
- **A. Triple barrier (`tb`)**: sigma_i = std of 1-s log mid returns over the trailing 600 s (WARMUP until full); B_i = max(1.0 * sigma_i * mid * sqrt(H), 3 * spread_i). LONG profit: first second with bid_h >= ask_e + B_i + x; stop: bid_l <= ask_e - B_i. SHORT profit: ask_l <= bid_e - B_i - x; stop: ask_h >= bid_e + B_i. Per side: +1 profit strictly before stop, -1 stop first or same second (conservative), 0 timeout. Label LONG(+1) if only LONG wins or LONG wins earlier; SHORT(-1) symmetric; same second -> NO_TRADE + `ambiguous`; otherwise NO_TRADE. Timeout = no barrier by H.
- **B. Fixed-horizon cost-adjusted return (`fh`)**: r_long = (bid_c[i+L+H] - ask_c[i+L])/point - x; r_short = (bid_c[i+L] - ask_c[i+L+H])/point - x (points; also bps). LONG if r_long > 0, SHORT if r_short > 0, else NO_TRADE (r_long + r_short < 0 always, so at most one side).
- **C. Opportunity / NO_TRADE (`opp`)**: net favourable excursion F_j (LONG: (running max bid_h - ask_e)/point - x; SHORT: (bid_e - running min ask_l)/point - x), adverse excursion A_j (LONG: (ask_e - running min bid_l)/point; SHORT: (running max ask_h - bid_e)/point). Side is an opportunity if some second j has F_j >= T_i = 2 * (entry spread_points + x) and A_j <= F_j (reward:risk >= 1). Earlier opportunity wins; same second -> NO_TRADE + ambiguous. No volatility barrier, no stop.
All parameters (m = 1, k_spread = 3, W = 600 s, kappa = 2, rho = 1, margin 0) are fixed a priori from market structure and are **not to be tuned on label or model results**. Vectorised implementation == independent loop reference (tests); causality test: rewriting every quote after the window does not change a label.

## 8. Candidate horizons: 15 s, 60 s, 300 s
Justified by observed structure, not profit: median tick rate 7.8 Hz gives about 117, 470 and 2,340 ticks per horizon; they equal 3x the 5 s micro window, the 60 s feature window and the 300 s feature window; the median spread is 9 points (about $0.09, ~0.2 bps) while a typical 1-minute move is about 3.6 bps (~145 points), i.e. >10x the spread, so costs matter but do not dominate at >= 15 s; 1 s latency is <7 % of the shortest horizon. Excluded: < 15 s (below the 1 s grid/latency resolution relative to the horizon) and > 300 s (not a scalp; fewer independent samples). Three horizons x three families = 9 candidates; no search.

## 9. Cost Model v0 (`cost_model_v0.json`)
| component | status |
|---|---|
| observed spread | OBSERVED (historical bid/ask in every label) |
| commission (round turn, points) | **UNKNOWN / UNCALIBRATED** |
| slippage (points per side) | **UNKNOWN / UNCALIBRATED** |
| latency (s, quantised to the 1 s grid) | **UNKNOWN / UNCALIBRATED** (terminal API latency is not execution latency) |
Unknown components can only enter through a named sensitivity scenario (`UncalibratedCostError` otherwise): `C0_spread_only` (L=0, extra 0 - an optimistic lower bound, not a claim that costs are zero), `C1_moderate` (L=1 s, extra 7 + 2x1.5 = 10 points), `C2_pessimistic` (L=1 s, extra 15 + 2x7.5 = 30 points). The scenario numbers are **hypothetical sensitivity points, not broker facts**. Every reported result must be shown under all three.

## 10. Chronological partitions
Expanding-window walk-forward on whole source days; no random splits. Eligible days start 2026-04-06 (04-01/02 are time-basis uncertain; 04-03..05 closed). Purge: a training sample is used only if its `label_end_ms` <= validation start - embargo; embargo **1,800 s** (= the longest feature history; >= the 301 s longest label span); validation samples must finish inside their block. Training never overlaps validation; validation blocks are disjoint and increasing; every fold's validation + embargo ends before the holdout.

| fold | train days | n | train eligible ticks | validation days | n | val eligible ticks | train samples purged (longest label) |
|---|---|---|---|---|---|---|---|
| F1 | 2026-04-06 .. 2026-05-29 | 40 | 23,453,798 | 2026-06-01 .. 2026-06-19 | 15 | 10,376,403 | 0 |
| F2 | 2026-04-06 .. 2026-06-19 | 55 | 33,830,201 | 2026-06-22 .. 2026-07-10 | 15 | 9,607,816 | 0 |
| F3 | 2026-04-06 .. 2026-07-10 | 70 | 43,438,017 | 2026-07-13 .. 2026-07-31 | 15 | 8,920,083 | 0 |
| F4 | 2026-04-06 .. 2026-07-31 | 85 | 52,358,100 | 2026-08-03 .. 2026-08-21 | 15 | 9,312,260 | 0 |
| F5 | 2026-04-06 .. 2026-08-21 | 100 | 61,670,360 | 2026-08-24 .. 2026-09-11 | 15 | 11,161,593 | 0 |

Purge removes 0 samples at these fold edges because every boundary falls on a weekend (>= 48 h between a Friday close and the Monday open); the rule is nevertheless implemented and tested on adjacent-day boundaries. F5's validation block contains the feed-regime boundary (2026-09-07); no training block does.
**FINAL untouched evaluation period: 2026-09-14 .. 2026-10-06** (17 trading days, 12,229,277 eligible ticks). Chosen from chronology and coverage only: the last three weeks plus two days, leaving the boundary week 2026-09-07..11 in development so the pre/post-feed shift can be studied without touching the holdout. Development code may not read it: `assert_development_only` / `development_mask` raise `HoldoutViolation`; label diagnostics and all analyses added in this phase read development days only. **Disclosure:** the descriptive whole-dataset profile (spread, tick rate, unusual days) was published before this protocol existed and includes holdout days; no label, feature outcome or model result inside the holdout has been inspected.

## 11. Feature Set v1 - `xauusd_core-fs1-ac928c2b1771` (119 features)
Classified by provenance, never by performance: **108 ALLOWED, 11 CONDITIONAL, 0 EXCLUDED**. No feature uses undocumented flag bits, future information, depth of market or traded volume (`last`, `volume`, `volume_real` are never read; `dist_tickweighted_mean_*` weight by tick COUNTS), and none depends on a globally fitted statistic (trailing windows only; prefix-invariance proven on the real data). CONDITIONAL: `dist_session_high_bps`, `dist_session_low_bps`, `dist_session_mean_bps`, `minutes_since_session_open`, `minutes_until_session_close`, `session_high_so_far`, `session_low_so_far`, `session_range_so_far_bps`, `session_so_far_complete`, `spread_regime_code`, `volatility_regime_code`. Session running aggregates are admitted only where `session_so_far_complete == 1`; the regime codes only after their thresholds are re-fit on the training block. Feed-sensitive (tick-activity and staleness) features are flagged in the manifest. Full per-feature table: `feature_set_v1.json`.

## 12. Label diagnostics (development period only; descriptive; NO model, NO PnL)
115 development days, 120 segments, 9,454,586 1-s observation rows. C0/C1/C2 percentages are LONG / SHORT / NO_TRADE of labelled rows. Sensitivity scenarios C1/C2 were generated on every 4th development day (about one quarter of the rows).
**Scenario C0_spread_only (all development days)**
| label | labelled rows | LONG % | SHORT % | NO_TRADE % | ambiguous | lost: warm-up | lost: segment boundary | event duration |
|---|---|---|---|---|---|---|---|---|
| tb_H15 | 9,380,786 | 17.83 | 18.21 | 63.97 | 0 | 72,000 | 1,800 | median 9 s / p90 14 s |
| tb_H60 | 9,375,386 | 23.17 | 23.82 | 53.01 | 0 | 72,000 | 7,200 | median 32 s / p90 54 s |
| tb_H300 | 9,346,586 | 26.79 | 28.0 | 45.21 | 0 | 72,000 | 36,000 | median 147 s / p90 262 s |
| fh_H15 | 9,452,786 | 42.66 | 42.72 | 14.62 | 0 | 0 | 1,800 |  |
| fh_H60 | 9,447,386 | 46.38 | 46.94 | 6.67 | 0 | 0 | 7,200 |  |
| fh_H300 | 9,418,586 | 47.85 | 49.28 | 2.87 | 0 | 0 | 36,000 |  |
| opp_H15 | 9,452,786 | 42.66 | 42.92 | 14.42 | 1 | 0 | 1,800 |  |
| opp_H60 | 9,447,386 | 48.77 | 49.12 | 2.11 | 1 | 0 | 7,200 |  |
| opp_H300 | 9,418,586 | 49.65 | 50.03 | 0.32 | 1 | 0 | 36,000 |  |

**Scenario C1_moderate (every 4th day)**
| label | labelled rows | LONG % | SHORT % | NO_TRADE % | ambiguous | lost: warm-up | lost: segment boundary | event duration |
|---|---|---|---|---|---|---|---|---|
| tb_H15 | 2,354,773 | 13.98 | 14.32 | 71.7 | 0 | 19,800 | 528 | median 9 s / p90 14 s |
| tb_H60 | 2,353,288 | 20.38 | 21.1 | 58.51 | 0 | 19,800 | 2,013 | median 33 s / p90 54 s |
| tb_H300 | 2,345,368 | 25.26 | 26.56 | 48.18 | 0 | 19,800 | 9,933 | median 151 s / p90 264 s |
| fh_H15 | 2,374,573 | 35.64 | 35.68 | 28.68 | 0 | 0 | 528 |  |
| fh_H60 | 2,373,088 | 43.02 | 43.39 | 13.59 | 0 | 0 | 2,013 |  |
| fh_H300 | 2,365,168 | 46.37 | 47.73 | 5.9 | 0 | 0 | 9,933 |  |
| opp_H15 | 2,374,573 | 27.31 | 27.5 | 45.2 | 0 | 0 | 528 |  |
| opp_H60 | 2,373,088 | 44.46 | 44.98 | 10.56 | 0 | 0 | 2,013 |  |
| opp_H300 | 2,365,168 | 49.26 | 50.04 | 0.7 | 0 | 0 | 9,933 |  |

**Scenario C2_pessimistic (every 4th day)**
| label | labelled rows | LONG % | SHORT % | NO_TRADE % | ambiguous | lost: warm-up | lost: segment boundary | event duration |
|---|---|---|---|---|---|---|---|---|
| tb_H15 | 2,354,773 | 8.87 | 9.18 | 81.95 | 0 | 19,800 | 528 | median 10 s / p90 14 s |
| tb_H60 | 2,353,288 | 15.95 | 16.65 | 67.4 | 0 | 19,800 | 2,013 | median 35 s / p90 55 s |
| tb_H300 | 2,345,368 | 22.63 | 23.85 | 53.52 | 0 | 19,800 | 9,933 | median 157 s / p90 266 s |
| fh_H15 | 2,374,573 | 24.44 | 24.5 | 51.05 | 0 | 0 | 528 |  |
| fh_H60 | 2,373,088 | 36.34 | 36.72 | 26.94 | 0 | 0 | 2,013 |  |
| fh_H300 | 2,365,168 | 43.36 | 44.67 | 11.97 | 0 | 0 | 9,933 |  |
| opp_H15 | 2,374,573 | 9.76 | 9.97 | 80.27 | 0 | 0 | 528 |  |
| opp_H60 | 2,373,088 | 29.34 | 29.92 | 40.74 | 0 | 0 | 2,013 |  |
| opp_H300 | 2,365,168 | 46.82 | 47.7 | 5.49 | 0 | 0 | 9,933 |  |

Findings (documented, not tuned):
1. **`fh` and `opp` are near-degenerate as abstain labels when only the observed spread is charged**: NO_TRADE is 6.67 % (fh_H60), 2.87 % (fh_H300), 2.11 % (opp_H60) and 0.32 % (opp_H300): a move of 100+ points over >= 60 s almost always exceeds a 9-point spread. They become informative only under the (hypothetical) higher-cost scenarios, e.g. opp_H60 reaches 40.74 % NO_TRADE in C2. By the pre-registered rule (NO_TRADE < 5 % = non-informative) they are reported but cannot be primary in those scenarios.
2. **`tb` is non-degenerate everywhere** (NO_TRADE 45.21-81.95 %), symmetric (LONG ~ SHORT within about 1 pp), and the pre-registered primary is **`tb_H60` under all three cost scenarios**. Barrier hits for H=60 under C0: LONG profit-first 23.17 %, stop-first 29.7 %, timeout 47.13 %; median event duration 32 s of 60; mean barrier 148.1 points (~16x the median spread, so the 3x-spread floor is inactive).
3. **Stability** (tb_H60, C0, LONG/SHORT/NO_TRADE %): by month 2026-04: 22.89/23.59/53.52, 2026-05: 22.05/23.21/54.73, 2026-06: 23.81/24.59/51.6, 2026-07: 23.34/23.84/52.82, 2026-08: 23.5/23.43/53.07, 2026-09: 23.6/24.71/51.69. By session: ASIA: 23.62/24.33/52.05, LONDON: 23.66/24.03/52.31, NEW_YORK: 21.03/22.34/56.63, LONDON_NEW_YORK_OVERLAP: 25.56/25.98/48.46, OFF_HOURS: 20.54/20.47/58.99. By feed regime: pre: 23.15/23.76/53.09, post: 23.68/25.04/51.28. No month, session or feed-regime artefact is visible in class balance.
4. **Coverage and losses**: 98.7-99.98 % of observation rows are labelled; the loss is exactly 120 segments x (600 s warm-up for `tb`) and 120 x H at each segment end, i.e. nothing leaks across segment boundaries. Purge removes 0 training rows at the fold edges (weekend alignment). Ambiguous same-second ties: 0-1 rows.
5. Consecutive 1-second rows are strongly autocorrelated and horizons overlap: the effective sample size is far below the row count (subsampling or block resampling is required for inference).

## 13. Machine-readable specification (the runner may not deviate)
`research/phase2b/`: `policy.json` (exclusion/retention, 300 s segments, feed boundary), `cost_model_v0.json`, `labels_v0.json` (specs, pre-registration), `folds_v1.json` (folds, embargo, holdout), `feature_set_v1.json` (per-feature class), `XAUUSD_RAW_V1.freeze.json`, `eligibility_summary.json`, `spec_index.json` (per-file SHA-256 + `spec_hash`), plus `diagnostics/` (not hashed). Tests: deterministic regeneration, tamper/missing/code-drift detection, and the committed spec loads and matches the code.

## 14. Known limitations
1. DST/offset unresolved outside 2026-04-05..2026-10-07; only summer time; re-validate after 2026-11-01.
2. Commission, slippage and latency are UNKNOWN: all conclusions are conditional on the named scenarios; the C0 scenario is optimistic.
3. One DEMO feed (quote-only, no volume), 6 months; DEMO spreads/fills may differ from a live account.
4. A single 17-day final holdout is a coarse, one-shot test; the post-feed-boundary regime has only 5 development days.
5. 1-second grid: latency is quantised to 0/1 s; barrier ties within a second are resolved conservatively (stop first), intra-second order is unknown.
6. Heavy label/row autocorrelation; trailing-sigma barrier makes `tb` volatility-adaptive but couples it to the volatility estimator (600 s).
7. The June 29 outage and other feed holes are treated as unexplained; the 300 s rule is a convention, not a cause analysis.
8. April spreads were wider (p99 ~35 vs ~12 points from June): non-stationarity within the training window is real; train-only fitting does not remove it.
9. Descriptive whole-dataset statistics were published before the holdout protocol (disclosed above).
