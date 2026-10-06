# Validation Protocol

Goal: separate real, cost-robust edge from overfitting. Win rate is reported, never optimised alone. Primary criteria: net expectancy, profit factor, max drawdown, risk-adjusted return, stability, cost robustness.

## 1. Data splits (chronological only; no shuffling)

```
|-------- development (train/validation, walk-forward) --------|-- embargo --|-- FINAL HOLDOUT --|
```
- **Final holdout:** latest contiguous period (>= 3 months proposed), carved out and sealed in Phase 2 (a hash and date range committed; data not loaded by research code). Evaluated **once**, for a candidate that is already frozen. Any second look is logged and the holdout is declared burned.
- Development data uses **rolling (and anchored) walk-forward**: for fold k, train on [t0, t_k), calibrate on a following block, validate on the next block, test on the block after (never overlapping), step forward.

## 2. Leakage controls

- Labels use the future; features never. Truncation test: for random times t, `feature(data[:t])[t] == feature(data)[t]`; run in CI for every feature.
- **Purging:** remove training samples whose label horizon overlaps the validation/test window; **embargo** after test windows >= max label horizon + max feature lookback.
- Swing/pivot features are only usable after their confirmation lag.
- Scalers, calibrators, selectors, thresholds are fit on training/calibration segments only.
- Economic data are point-in-time; no revised values.
- Resampling/bar construction uses only closed bars; the in-progress bar is excluded or flagged.
- Time-of-day/session features are computed from UTC with explicit DST handling.
- Cross-validation inside training (for hyperparameters) uses purged K-fold with embargo (and combinatorial purged CV for robustness studies).

## 3. Selection-bias control

- **Trial registry:** every experiment/hyperparameter trial/threshold variant is logged (number of trials per strategy family).
- Report haircut statistics accounting for the number of trials (e.g. deflated Sharpe ratio / probability of backtest overfitting via CSCV) alongside raw metrics.
- Pre-register acceptance criteria before running the final evaluation.
- Prefer few parameters; sensitivity surfaces must be broad plateaus, not isolated peaks.

## 4. Cost realism

Bid/ask tick backtest preferred; OHLC-only results are labelled `REDUCED_FIDELITY` and cannot be used for promotion. Costs: spread (actual), commission, slippage model (calibrated later from demo execution data; pessimistic default), execution delay (latency distribution), swaps across rollover, stop/freeze-level constraints, min lot/step, rejects. Fills: market orders fill at the first available quote after `decision_time + latency`; stops trigger on bid/ask side appropriate to the position; gaps fill at the next available price, not at the stop.

## 5. Metrics (reported with uncertainty)

Totals, win/loss counts, win rate, avg win/loss, largest win/loss, expectancy (per trade, per R, per lot), profit factor, gross profit/loss, net, return, Sharpe, Sortino, max drawdown, Calmar, average holding time, fees, spread cost, slippage cost, long vs short. Segmented by instrument, session, hour, weekday, regime, volatility regime, confidence bucket. Confidence intervals by block bootstrap. Calibration: Brier, log-loss, ECE, reliability curves.

## 6. Robustness

Trade-order reshuffling (Monte Carlo drawdown distribution); block bootstrap CIs; parameter perturbation (+/-); cost stress: spread x1.25 / 1.5 / 2.0; slippage normal / elevated / extreme; latency stress (+100/+250/+500 ms); data-subset stability (by month, session, regime); random-entry null (B1) significance. A strategy whose expectancy flips negative under spread x1.25 is **not robust**.

## 7. Acceptance gates between stages

| Stage | Gate |
|---|---|
| Backtest -> Walk-forward | Positive net expectancy on validation after realistic costs; leakage tests green |
| Walk-forward -> Replay | Pre-registered criteria met on pooled folds and in a majority of folds; stress tests pass |
| Replay -> Shadow | Replay on production code path reproduces backtest trades within tolerance (trade-by-trade diff analysed) |
| Shadow -> Demo | Shadow period (>= N trading weeks, N set by human, D-6) with live metrics inside historical confidence bands; feed/latency healthy |
| Demo -> Live-readiness | Demo execution quality and PnL consistent with backtest/shadow after cost recalibration; no unresolved incidents; human review |

"Consistent" is defined by pre-registered tolerance bands; major deviations stop progression until explained.

## 8. Sample size

No claim is made from fewer than the pre-registered minimum number of out-of-sample trades (proposed 300 pooled, and a minimum per segment before segment claims). Small samples are reported as inconclusive.
