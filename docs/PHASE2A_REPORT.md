# Phase 2A Report — Feature Pipeline (XAU/USD, synthetic data only)

**MT5 has NOT been verified.** The Phase 1 Windows DEMO verification gate is still open and was not bypassed. All results below are on synthetic / fake / deterministic fixtures and say nothing about real gold or any broker. No model trained, no strategy, no orders, no profitability or win-rate claims. EUR/USD & GBP/USD untouched.

## 1. Files
New: `src/fxscalp/features/{ops,normalize,bars,spec,session_features,groups,segment,registry,pipeline,validation,leakage,scaler,dataset}.py`, `src/fxscalp/regimes/{config,spread_regime,volatility_regime}.py`, `src/fxscalp/synthetic/xauusd.py`, `scripts/{build_features,export_feature_registry,benchmark_features}.py`, `configs/features/xauusd_core.registry.yaml`, `docs/{PHASE2A_FEATURE_ARCHITECTURE,LEAKAGE_PREVENTION,XAUUSD_FEATURE_PLAN,PHASE2A_REPORT}.md`, `docs/baselines/phase2a_synthetic_pipeline.json`, tests under `tests/unit/test_features_*.py` and `tests/leakage/`. Modified: `docs/FEATURE_CATALOG.md` (now auto-generated), `docs/IMPLEMENTATION_PLAN.md`, `README.md`, `pyproject.toml` (`slow` marker).

## 2. Architecture
See `PHASE2A_FEATURE_ARCHITECTURE.md`.

## 3. Features
119 features, feature set `xauusd_core-fs1-ac928c2b1771`: micro 28, spread 11, structure 21, volatility 8, momentum 14, mean_reversion 6, mtf 15, session 14, regimes 2. Max lookback 4500 s. Details: `FEATURE_CATALOG.md`.

## 4. Timeframes
1s, 5s, 15s, 30s, 1m, 5m (15m optional via config); UTC half-open boundaries; explicit empty bars.

## 5. Leakage protections
Causal-only primitives, static AST scan, completed-HTF-bar alignment, session-so-far only, causal scaler, target-contamination check, prefix invariance. See `LEAKAGE_PREVENTION.md`.

## 6. Prefix-invariance results
Exact (rtol=0, atol=0) at multiple cutoffs on: normal data, data with duplicates/out-of-order/missing ticks, two segments, weekend closure, long gaps. All pass. 11 deliberately leaky fixtures are all detected.

## 7. Synthetic scenarios
Trend up/down, range, volatility burst, quiet, spread widening, outages/missing, duplicates, out-of-order, weekend closure, London/NY opens, overlap, server-time offset.

## 8. Tests
Full suite: **301 passed** (186 Phase 0/1 retained + 115 new Phase 2A: bars 16, values 15, normalize/registry/regimes 15, validation/synthetic 14, dataset/scripts ~10, leakage 45). `ruff --select F,E9` clean. Registry snapshot drift check passes.

## 9. Performance baseline (synthetic, 4 CPU, Python 3.13, numpy 2.5.3, pandas 3.0.5)
| ticks | grid rows | normalize | bars | features | write | read | peak RSS |
|---|---|---|---|---|---|---|---|
| 100k | 19,277 | 0.10s | 0.62s | 0.46s (42k rows/s) | 0.50s | 0.42s | 285 MB |
| 500k | 98,649 | 0.23s | 2.47s | 1.24s (80k rows/s) | 1.28s | 0.79s | 778 MB |
| 1M | 198,156 | 0.39s | 4.43s | 3.24s (61k rows/s) | 2.81s | 1.53s | 1416 MB |
Saved in `docs/baselines/phase2a_synthetic_pipeline.json`. Hardware-specific; memory grows ~linearly (bar tables for all timeframes are held in memory).

## 10. Known limitations
- Synthetic data only; feature distributions on real gold unknown.
- Regime engines are uncalibrated placeholders (flagged in every manifest).
- Memory use at ~1.4 GB per 1M ticks; multi-day real datasets need chunked/segment-wise processing (Phase 2B).
- Grid is fixed at 1 s decision cadence; ms-level decisions not supported.
- Server-time offset/DST of real timestamps unverified (Phase 1 gate).
- Prefix invariance proves causality, not usefulness.
- Feature set version is sensitive to source fingerprint normalization; regenerate registry after any feature-code change.

## 11. Rejected / deferred
Order-book features, tick-volume "order flow" claims, news/calendar, cross-asset, labels/targets, fitted scalers/PCA, absolute spread/vol thresholds. See `XAUUSD_FEATURE_PLAN.md`.

## 12. Requires tomorrow's real XAU/USD data
Regime calibration, real spread/tick-rate distributions per session, DST/server-time verification, real digits/point/tick-size, real gap/outage structure, feature stability.

## 13. Commands tomorrow (after MT5 verification passes on Windows DEMO)
```
python -m scripts.verify_mt5            # per docs/WINDOWS_MT5_VERIFICATION.md
python -m scripts.acquire_ticks ...     # per the same doc; produces a tickraw-... dataset
python -m scripts.profile_ticks ...
python -m scripts.build_features --dataset <tickraw-id or path> --out-root data
python -m scripts.export_feature_registry --check
python -m pytest -q
```
(Run `python -m scripts.build_features --help` for exact flags.)

## 14. Proposed Phase 2B (needs approval)
Run the pipeline on verified real data; calibrate regimes on a separate reference period; feature distribution/stability profiling; chunked processing for multi-day data; then label design with purge/embargo. No profitability claims.

## 15. Recommended architecture changes
Segment-wise streaming to bound memory; optional Polars/Numba backend only after equality tests against the current implementation; keep the registry snapshot in CI; store regime calibration artefacts as versioned, hashed files.
