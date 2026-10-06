# XAU/USD Research Plan

No result in this document is a finding. Everything is a hypothesis to test. Thresholds from other instruments are not reused.

## 1. Questions (ordered)

1. **Data:** what do the broker's XAUUSD ticks look like (rate, spread distribution by session/hour, gaps, stale periods, symbol suffix, digits, contract size, stops level)?
2. **Cost floor:** what is the distribution of round-trip cost (spread + commission + estimated slippage) per session, in price units and as a fraction of ATR(1m/5m)? Below what move size is a trade impossible to win after costs?
3. **Opportunity:** at which horizons (e.g. 15s-10min) do moves exceed the cost floor often enough to matter? (Unconditional barrier-hit frequencies at various TP/SL multiples of ATR.)
4. **Regimes:** which regimes (trend/range/breakout/high-vol/news) show positive conditional net expectancy for simple baselines? Is it stable across months?
5. **Sessions:** London open, NY open, overlap, late NY, Asia: expectancy and spread by session (with confidence intervals).
6. **Events:** behaviour in +/-N minutes around high-impact USD releases: spread, slippage, whether to block entirely.
7. **Features:** which feature groups add out-of-sample information beyond a spread/vol/session baseline (ablation)?
8. **Models:** do GBMs beat logistic regression? do sequence models beat GBMs, out of sample, net of costs?
9. **Exits:** fixed TP/SL vs ATR vs time stop vs trailing vs break-even vs partials, evaluated on identical entries.
10. **Calibration/meta:** does a meta-model that vetoes low-quality setups improve net expectancy and drawdown?

## 2. Baselines (must be beaten)

- B0: no-trade (0 PnL).
- B1: random entry with identical SL/TP/cost model (null distribution; also used for significance).
- B2: simple momentum breakout and mean-reversion rules with session/spread filter.
- B3: logistic regression on a small feature set.
A model must beat B1 with statistical significance and beat B2/B3 out of sample after costs to be kept.

## 3. Experiments (Phase 2-6)

| ID | Experiment | Output |
|---|---|---|
| E1 | Data profile and quality report | tick rate, gaps, spread by session/hour, outliers |
| E2 | Cost-floor study | edge needed vs ATR by session |
| E3 | Barrier-hit base rates | P(TP before SL) grid over (TP, SL, horizon) per session/regime |
| E4 | Feature ablation | incremental OOS log-loss/AUC and net PnL |
| E5 | Label design comparison | triple-barrier variants, meta-label vs direct |
| E6 | Model comparison | LR/RF/LGBM/XGB/CatBoost, then TCN/GRU/LSTM/Transformer if justified |
| E7 | Timeframe stack test | does 1H/15m/5m context add anything; ablate each |
| E8 | Exit study | exit rules on fixed entries |
| E9 | Event-window study | block/reduce policy value |
| E10 | Stress | spread x1.25/1.5/2, slippage, latency, parameter perturbation |

## 4. Gold-specific considerations

Tick size/digits and typical spread differ by broker (2 vs 3 digits). Momentum bursts and false breakouts at session opens and US data; spreads can widen sharply with little warning; weekend/news gap risk; USD and rate sensitivity (macro series such as DXY/real yields only if a reliable, timestamp-correct source exists; otherwise excluded to avoid leakage via revised data). Mean-reversion vs trend regimes alternate and must be identified causally.

## 5. Acceptance criteria (decision D-6: no invented thresholds)

No profitability, win-rate, profit-factor, fold-consistency or trade-count threshold is fixed in advance of data. They are **derived from baseline empirical research** (E1-E3 and baselines B0-B3 in Phase 4/5) and then **pre-registered before** the walk-forward, replay and holdout evaluations, with the derivation recorded:

- **Null distribution:** the random-entry baseline B1 under the identical cost/exit model gives the distribution of net expectancy, profit factor and drawdown that luck alone produces. Acceptance requires the candidate to be statistically distinguishable from B1 at a pre-registered significance level, corrected for the number of trials (trial registry; deflated Sharpe / PBO).
- **Sample size:** the minimum number of out-of-sample trades comes from a power analysis using the baseline's measured per-trade variance and the cost floor (E2), not a round number.
- **Stability:** fold-consistency and regime coverage requirements are set from the observed fold-to-fold dispersion of the baselines.
- **Costs:** the candidate must remain net-positive under cost stress levels whose size is chosen from the measured spread/slippage distributions (e.g. high percentiles per session), not arbitrary multiples.
- **Risk:** drawdown must fit within the human-owned risk budget.
- **Calibration:** reliability/ECE bounds are set relative to the baseline calibrators' measured error.

Shadow validation (D-6) is judged by **trade count and regime coverage**, not elapsed days: it needs enough shadow trades (from the power analysis) spread over multiple observed regimes (each regime with a pre-registered minimum count), with live behaviour compared against historical expectations using the same statistics.

## 6. Deliverables per experiment

Config + code commit hash, dataset manifest hash, trial registry entry, report with metrics and uncertainty. No cherry-picked periods.
