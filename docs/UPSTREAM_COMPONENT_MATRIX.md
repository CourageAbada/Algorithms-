# Upstream Component Matrix

Legend: **REUSED** (used as a dependency), **WRAPPED** (behind our interface), **ADAPTED** (code copied/modified; license permitting), **REFERENCE** (idea only), **REJECTED**.
No component is currently ADAPTED. Licenses per `UPSTREAM_AUDIT.md`.

| # | Upstream | Component / concept | Decision | Our module | Notes |
|---|---|---|---|---|---|
| 1 | MetaTrader5 pkg | `initialize/login/shutdown` | WRAPPED | `brokers/mt5` | Windows only; connection supervisor + health |
| 2 | MetaTrader5 pkg | `copy_ticks_range/from`, `copy_rates_*` | WRAPPED | `brokers/mt5`, `market_data` | Convert server time -> UTC at boundary |
| 3 | MetaTrader5 pkg | `symbol_info`, `symbol_info_tick` | WRAPPED | `brokers/mt5` -> `SymbolInfo` | Drives sizing, stops, lot step |
| 4 | MetaTrader5 pkg | `order_check`, `order_send` | WRAPPED | `brokers/mt5`, `execution` | `order_check` always before send; blocked outside DEMO/LIVE |
| 5 | MetaTrader5 pkg | `order_calc_margin/profit` | WRAPPED | `risk` via adapter | Cross-check own margin math |
| 6 | MetaTrader5 pkg | `market_book_*` | WRAPPED (experimental) | `market_data` | Only if reliable; never assumed to be full-market depth |
| 7 | Vibe-Trading | Evidence-gated / grounding-gate agent design | REFERENCE | `agents` | Agent numbers must cite tool outputs |
| 8 | Vibe-Trading | Skills + swarm of workers | REFERENCE | `agents` | Offline experiment analysis only |
| 9 | Vibe-Trading | MCP tool server | REFERENCE / optional sidecar | `agents` | Read-only access to journal/backtests |
| 10 | Vibe-Trading | Durable client-order-id recovery | REFERENCE | `execution` | Idempotent order tracking after restart |
| 11 | Vibe-Trading | Equity/A-share/crypto/option backtest engines | REJECTED | - | Wrong market microstructure |
| 12 | Vibe-Trading | Data loaders (27+) | REJECTED | - | MT5 + chosen vendor instead |
| 13 | Vibe-Trading | Broker connectors | REJECTED | - | None cover MT5 execution |
| 14 | Vibe-Trading | FastAPI + React dashboard | REFERENCE | `api`, `frontend` | Own stack; same family of tech |
| 15 | Qlib | DataHandler / Dataset / segments | REFERENCE | `features`, `models` | Raw -> processors -> train/valid/test segments |
| 16 | Qlib | Rolling / online model pipeline | REFERENCE | `models/lifecycle` | Walk-forward retraining |
| 17 | Qlib | Recorder / experiment artefacts | REFERENCE | `models/registry` | Persist params, data hash, metrics |
| 18 | Qlib | IC / rank-IC signal analysis | REFERENCE (adapted to one instrument) | `backtest/analysis` | Plus calibration-first metrics |
| 19 | Qlib | Model zoo (LGBM, XGB, LSTM, TFT...) | REFERENCE | `models` | We wrap sklearn/LightGBM/XGBoost/PyTorch directly |
| 20 | Qlib | Alpha158/360 factor sets | REJECTED | - | Equity-style daily factors; our features are microstructure/FX specific |
| 21 | Qlib | Binary data store, calendar | REJECTED | - | Parquet + UTC continuous time instead |
| 22 | Qlib | Cross-sectional portfolio strategies | REJECTED | - | Single-instrument scalping |
| 23 | FreqAI | Sliding-window train -> predict-until-expiry | REFERENCE | `models/lifecycle` | Reimplemented, clean-room |
| 24 | FreqAI | Feature fn / label fn separation | REFERENCE | `features`, `models/labels` | Labels future-only; features past-only; leakage tests |
| 25 | FreqAI | Outlier/novelty gate (DI, SVM) | REFERENCE | `models/novelty` | Mahalanobis / isolation forest / PSI; failing -> NO_TRADE |
| 26 | FreqAI | Model persistence + identifier + purge | REFERENCE | `models/registry` | Champion/candidate, immutable artefacts |
| 27 | FreqAI | Background live retraining thread | REFERENCE | `models/lifecycle` | Training in separate process; promotion gated, never automatic |
| 28 | Freqtrade | Backtester, hyperopt, strategy API | REJECTED | - | Candle-fill assumptions, crypto fees; hyperopt encourages overfitting |
| 29 | Freqtrade | Exchange/CCXT layer, pairlists | REJECTED | - | No FX/CFD/MT5 |
| 30 | Freqtrade | Telegram/WebUI | REFERENCE | `api`, `monitoring` | Possible alert channel later |
