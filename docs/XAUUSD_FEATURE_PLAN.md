# XAU/USD Feature Plan (Phase 2A)

Scope: XAU/USD only. EUR/USD and GBP/USD are config/discovery only; no model work.

## Implemented groups (feature set `xauusd_core`, 119 features, max lookback 4500 s)

| Group | Content |
|---|---|
| micro (28) | tick count/rate/acceleration, mean inter-tick interval, up/down tick ratios, direction imbalance, directional persistence over 5s/15s/60s; consecutive up/down ticks; staleness; empty-interval flag |
| spread (11) | spread in points (from stored symbol `point`; NaN if unknown), relative spread bps, spread changes, rolling mean/median, percentile and z-score vs 30 min |
| structure (21) | rolling range width/position, distance to high/low (30s/60s/300s), compression/expansion ratios, range percentile, breakout distances, failed-breakout flags |
| volatility (8) | realized vol (15s/60s/300s), ATR14 on complete bars, vol percentile and z-score vs 1 h |
| momentum (14) | log-returns in bps (5s–300s), absolute returns, momentum alignment, EMA slope/distance, price velocity/acceleration |
| mean_reversion (6) | distance to rolling mean (time- and tick-weighted) in bps, z-scores at 60s/300s |
| mtf (15) | latest COMPLETE 5s/15s/30s/60s/300s bar return, range and close distance |
| session (14) | session code/flags, minutes since open / until close, session-so-far high/low/range, distances, completeness flag |
| regimes (2) | spread regime code, volatility regime code |

Each feature's exact definition, units, lookback, NaN behaviour and leakage assessment: `docs/FEATURE_CATALOG.md` (auto-generated).

## Rejected / deferred

- Order-book / depth features: MT5 demo ticks carry no reliable depth. Deferred.
- Tick-volume-based "order flow" claims: tick volume on FX/CFD is not traded volume; only used as an activity proxy.
- News/calendar features: need a calendar provider (D-5). Deferred.
- Cross-asset (DXY, yields, EUR/USD, GBP/USD) features: Phase 2B+ at the earliest; EUR/USD & GBP/USD explicitly not started.
- Absolute spread/vol thresholds: rejected (D-6); relative logic only until calibrated on real data.
- Labels, targets, triple-barrier: not tonight.
- Any fitted scaler/PCA/embedding: deferred until a causal fit protocol is used on real data.

## Requires real XAU/USD data (not available tonight)

Regime calibration quantiles; true tick rates and spread distributions by session; DST/server-time verification of real timestamps; feature stability/importance; session open behaviour at actual broker times; real digits/point/tick size; real gap/outage structure.
