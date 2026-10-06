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

## 5. Acceptance to progress to shadow (all required, evaluated on the untouched holdout and walk-forward folds)

Thresholds are *proposed* and need human approval (D-6): net expectancy > 0 with the lower bound of a bootstrap 95% CI above 0 on pooled walk-forward folds; profit factor above a pre-registered minimum (suggest >= 1.2 net); positive in at least a pre-registered fraction of folds (suggest >= 70%); max drawdown within risk budget; survives spread x1.5 and slippage stress with expectancy still positive; calibration error below a pre-registered bound; minimum trade count (suggest >= 300 OOS trades) for the claim to be considered.

## 6. Deliverables per experiment

Config + code commit hash, dataset manifest hash, trial registry entry, report with metrics and uncertainty. No cherry-picked periods.
