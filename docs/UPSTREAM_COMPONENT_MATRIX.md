# Upstream Component Matrix

Revised in Phase 0.5 after inspecting actual source (Vibe-Trading `7f6908b`, Qlib `be72549`, Freqtrade `9691649`, MetaTrader5 wheel `5.0.6231`). The full per-component table (licence, performance, Forex and XAUUSD suitability, rationale) is `PHASE0_5_UPSTREAM_DEEP_DIVE.md` §5; this file is the compact decision index.

Legend: **REUSE** (dependency, unchanged) · **ADAPT** (vendored with attribution after review and our own tests) · **WRAP** (behind our interface) · **REFERENCE_ONLY** (ideas, no code) · **REJECT**.
Nothing is vendored yet. Policy: no GPL code (Freqtrade/FreqAI) in this repository.

| # | Upstream (licence) | Component | Decision | Our module | Change vs Phase 0 |
|---|---|---|---|---|---|
| 1 | MetaTrader5 pkg (MIT) | Whole package, accessed only in `brokers/mt5` | WRAP | `brokers/mt5` | Persistent session; verification harness for unverified items |
| 2 | MetaTrader5 pkg | `Buy/Sell/Close` helpers | REJECT | - | New: hidden retries, fixed deviation 10 |
| 3 | Vibe (MIT) | `engines/forex.py`, `engines/base.py` | REJECT as engine; patterns REFERENCE_ONLY | `backtest` | **Changed from "reject" to "inspected and rejected"**: bar/target-weight, static spread table, no SL/TP |
| 4 | Vibe | `engines/_market_hooks.py` swap/spread tables | REJECT | - | Constants replaced by broker metadata |
| 5 | Vibe | `loaders/mt5_loader.py` | REFERENCE_ONLY | `brokers/mt5`, `market_data` | Bars only; symbol-ambiguity refusal pattern |
| 6 | Vibe | `trading/connectors/mt5/*` | REFERENCE_ONLY | `brokers/mt5`, `execution` | Demo/real guard, size guards, lock; reject per-call session |
| 7 | Vibe | `backtest/metrics.py` | ADAPT (small parts) | `backtest/metrics` | New: trade stats + drawdown only; add expectancy, R, MFE/MAE, costs, segmentation |
| 8 | Vibe | `backtest/validation.py` MC order shuffle | ADAPT | `backtest/robustness` | Reshuffle drawdown distribution only |
| 9 | Vibe | `backtest/validation.py` bootstrap / "walk-forward" | REJECT | - | iid bootstrap; equity-curve windows are not walk-forward |
| 10 | Vibe | `quantlib/crossvalidation.py` | **ADAPT** | `validation/purged_cv` | **New, high value**: purged/embargoed/CPCV/walk-forward + leakage detector |
| 11 | Vibe | `quantlib/multipletesting.py` | **ADAPT** | `validation/multiple_testing` | **New**: PSR, DSR, PBO, BH |
| 12 | Vibe | `governance/ledger.py` | **ADAPT** | `journal/ledger` | **New**: tamper-evident order/risk audit log |
| 13 | Vibe | `live/halt.py` | REFERENCE_ONLY | `core/safety`, `risk` | Sentinel-file HALT design |
| 14 | Vibe | `live/runtime/flatten.py`, mandate gate | REFERENCE_ONLY | `execution` | Flatten hedges on MT5; we close by ticket |
| 15 | Vibe | `quantlib/{risk,portfolio,constraints,microstructure,impact}.py` | REJECT | - | Portfolio/ADV/volume-based; invalid for FX tick volume |
| 16 | Vibe | `backtest/run_card.py`, `hypotheses/registry.py` | REFERENCE_ONLY | `models/registry`, trial registry | Provenance and status ideas |
| 17 | Vibe | agent loop, grounding, swarm, skills, MCP server | WRAP (optional read-only sidecar, Phase 10/11) | `agents` | Approved D-11; never in execution path |
| 18 | Vibe | `shadow_account/*` | REJECT | - | New: unrelated to our SHADOW mode |
| 19 | Vibe | factor zoo, other-market engines/loaders | REJECT | - | Equity/crypto/options |
| 20 | Qlib (MIT) | Dataset segments, infer/learn processors, explicit fit windows | REFERENCE_ONLY | `features`, `models` | Fit scalers on train fold only |
| 21 | Qlib | `RollingGen`, `trunc_days` | REFERENCE_ONLY | `models/lifecycle` | Superseded by label-span purging (#10) |
| 22 | Qlib | Model interface | REFERENCE_ONLY | `models` | Own `Predictor` |
| 23 | Qlib | Recorder/MLflow, online manager | REJECT | - | Registry stores equivalent facts |
| 24 | Qlib | `backtest/exchange.py` etc. | REJECT | - | Equity cost model |
| 25 | Qlib | `examples/highfreq`, Alpha158/360, RL | REFERENCE_ONLY / REJECT / REJECT | - | 1-min equity; no ticks |
| 26 | FreqAI (GPL-3.0) | Sliding-window retrain, predict-until-expiry | REFERENCE_ONLY | `models/lifecycle` | Concept only |
| 27 | FreqAI | Feature pipeline + outlier gate (SVM/DI/DBSCAN) | REFERENCE_ONLY | `models/novelty` | Own implementation; failure -> NO_TRADE |
| 28 | FreqAI | `make_train_test_datasets` | REJECT | - | **New finding**: no purge/embargo |
| 29 | FreqAI | Data drawer persistence (joblib/cloudpickle) | REFERENCE_ONLY | `models/registry` | Hash-verified artefacts, human promotion |
| 30 | FreqAI | Model wrappers, RL, torch, `data_kitchen` structure | REJECT | - | Wrap libraries directly |
| 31 | Freqtrade (GPL-3.0) | Bot, backtester, hyperopt, exchanges | REJECT | - | Crypto-specific |
