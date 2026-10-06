# Phase 0 Final Report

Status: **Phase 0 complete; Phase 0.5 verification complete (see `PHASE0_5_REPORT.md`). Phase 1 has not been started.**

> **Phase 0.5 update.** Source-level inspection changed several conclusions below: the Vibe-Trading Forex engine was inspected and **rejected** as a base (bar/target-weight, static spread/swap tables, no SL/TP); three Vibe-Trading utilities (purged CV, multiple-testing statistics, audit ledger) are now planned **ADAPT** candidates; FreqAI's train/test split was found to have no purge/embargo; MT5 time semantics are **unverified/conflicting** and require empirical calibration; the collector/ZeroMQ design was replaced by a single-machine design (D-4); decisions D-1..D-11 are now resolved (see §19). Sections 2-5, 9, 19, 20 below are superseded where they conflict with `PHASE0_5_UPSTREAM_DEEP_DIVE.md`, `SYSTEM_ARCHITECTURE.md` and `IMPLEMENTATION_PLAN.md`.
No broker connection was made, no orders were sent, no backtest was run, and no performance or win-rate claim is made anywhere.

## 1. Architecture chosen
Own modular Python 3.11+ platform (`src/fxscalp`), one-way layered pipeline (ticks -> bars -> features -> regime -> models -> calibration -> meta-model -> LONG/SHORT/NO_TRADE -> quality filter -> **independent risk engine** -> execution -> broker). `BrokerAdapter` abstraction, MT5 first. Same code path for REPLAY/SHADOW/DEMO/LIVE with swappable execution sinks. Collector process next to MT5 (Windows), engine/research platform-independent. See `SYSTEM_ARCHITECTURE.md`, `ARCHITECTURE_AUDIT.md`.

## 2. Vibe-Trading
*(Phase 0.5: source inspected; see deep dive §1 and §5.)* MIT. Used as **reference** only, plus planned ADAPT of `crossvalidation.py`, `multipletesting.py`, `governance/ledger.py` and small parts of `metrics.py`: evidence-gated agent (every number traceable to tool output), skills/swarm for offline experiment analysis, MCP-style read-only tool access, durable client-order-id recovery, price-caliber labelling. Possible read-only sidecar in Phase 10/11. Not used: its backtest engines, data loaders, broker connectors (no FX/MT5 execution).

## 3. Qlib
MIT. **Reference** only: DataHandler/Dataset segment separation, rolling/online model pipeline, recorder-style artefacts, IC-type signal analysis. Not used: equity factor sets, binary store/calendar, cross-sectional strategies.

## 4. FreqAI
Freqtrade is copyleft (GPL-3.0, to verify in LICENSE). **Concepts only, no code copied**: sliding-window retrain with predict-until-expiry, feature/label function separation, outlier/novelty gate, model persistence/purge, background training. Reimplemented clean-room per `ML_ARCHITECTURE.md`.

## 5. MT5 integration
*(Revised in Phase 0.5.)* Official `MetaTrader5` package (v5.0.6231, **Windows-only**, verified from the package metadata), imported only inside `brokers/mt5`, a single long-lived session on one owner thread in the single-machine engine process (no collector process, no message broker). Server-time -> UTC conversion through an empirically calibrated `TimeBase` (documentation and community reports conflict). `order_check` before any order; account must verify as demo for DEMO mode. Details marked [verify] in the audit must be confirmed in Phase 1 (official docs were unreachable from this sandbox).

## 6. XAU/USD data source
Primary: the **same broker's MT5 tick history** (`copy_ticks_range`) for the demo account that will be traded, so spreads/feed match execution. Supplementary (to verify: licence, availability, quality): a third-party tick vendor for longer history and cross-feed robustness. Depth achievable is unknown until Phase 1 measures it.

## 7. Tick storage
Parquet (immutable, partitioned by symbol/broker/date, with SHA-256 manifests) for ticks/bars/features; PostgreSQL for orders, fills, trades, journal, models, experiments, risk events; TimescaleDB optional later for metrics. Rationale in `DATA_REQUIREMENTS.md`.

## 8. Feature architecture
Versioned, hypothesis-tagged, past-only features computed incrementally (O(1) ring buffers) with a tested-equal vectorised path; groups: spread/cost, tick microstructure, multi-resolution bars, mean-reversion/structure (with explicit confirmation lags), session, event-risk, optional DOM. Admission by ablation. See `FEATURE_CATALOG.md`.

## 9. ML architecture
Triple-barrier, bid/ask-aware, cost-net labels; baselines LR -> RF/LightGBM/XGBoost/CatBoost; volatility and microstructure models; sequence models only if they beat GBMs OOS; calibration (Platt/isotonic, Brier, ECE); meta-labeling for NO_TRADE; novelty gate; champion/candidate registry with human-approved promotion; drift monitoring that can only reduce risk automatically. See `ML_ARCHITECTURE.md`.

## 10. Risk architecture
Independent package, no ML imports, fail-closed pre-trade checks (data/broker/model health, session/news, dynamic spread, edge, confidence, account limits, latency/slippage, broker rules), metadata-driven sizing rounded **down** to lot step with min-lot rejection, global kill switch, human-owned limits. See `RISK_MODEL.md`.

## 11. Backtesting architecture
Event-driven bid/ask tick backtester using the production pipeline, with spread, commission, slippage, latency, swap, stop/freeze levels, lot rules; OHLC-only runs labelled REDUCED_FIDELITY; full metric set and segmentations; stress scenarios. See `VALIDATION_PROTOCOL.md`.

## 12. Walk-forward protocol
Chronological development set with rolling/anchored walk-forward (train -> calibrate -> validate -> test), purging + embargo, CPCV for robustness, sealed final holdout evaluated once, trial registry with multiple-testing haircut, pre-registered acceptance criteria, stage gates. See `VALIDATION_PROTOCOL.md`.

## 13. Look-ahead / leakage risks identified
Labels overlapping across folds; pivot/swing features confirmed late; in-progress bar usage; scalers/calibrators/thresholds fit on future data; revised economic data; DST/server-time misalignment creating future-looking session features; threshold tuning on the holdout; feature selection on full sample; multiple-testing/selection bias; train/test feed mismatch; survivorship in "good period" selection.

## 14. Execution risks
Unknown order outcome after disconnect (duplicate orders), requotes/rejects/partial fills, slippage in fast markets, stop/freeze-level violations, filling-mode mismatch, latency larger than the edge assumes, trade-context-busy retries, netting/hedging semantics, position-state divergence after restart, weekend/news gaps jumping stops.

## 15. Broker / data-feed risks
Broker-specific quotes and spreads (models not transferable), limited or low-quality tick history, symbol suffix/digits/contract differences, server-time/DST quirks, demo feed differing from live, tick volume != real volume, unreliable DOM, spread widening/markups at rollover and news, terms restricting automated/scalping activity, MT5 Windows-only + terminal dependency.

## 16. Security concerns
Credential leakage (git/logs), accidental live trading, runaway orders, stale-data trading, dependency/supply-chain (incl. agent/MCP sidecars), prompt injection into the research agent, dashboard exposure, tampering with risk config/model artefacts. Mitigations in `SECURITY_MODEL.md`; redaction and LIVE interlock already implemented and tested.

## 17. Files created
- Docs (`docs/`): UPSTREAM_AUDIT, UPSTREAM_COMPONENT_MATRIX, ARCHITECTURE_AUDIT, SYSTEM_ARCHITECTURE, IMPLEMENTATION_PLAN, FOREX_MARKET_MODEL, XAUUSD_RESEARCH_PLAN, DATA_REQUIREMENTS, FEATURE_CATALOG, ML_ARCHITECTURE, RISK_MODEL, VALIDATION_PROTOCOL, EXECUTION_MODEL, SECURITY_MODEL, PHASE0_REPORT.
- Root: `README.md` (rewritten), `.env.example`, `.gitignore`, `pyproject.toml`.
- Configs: `configs/base.yaml`, `configs/risk.example.yaml`, `configs/instruments/{xauusd,eurusd,gbpusd}.yaml`.
- Code: package skeleton for all 21 subpackages under `src/fxscalp/`; implemented only: `core/modes.py` (modes, default SHADOW, LIVE interlock), `core/config.py`, `monitoring/logging_setup.py` (JSON logs + secret redaction), `brokers/base.py` (BrokerAdapter ABC + neutral types).
- Tests: 17 unit tests (modes/interlock, redaction, config loading, abstract adapter) — passing. Leakage-test placeholder README.
- Placeholders: `frontend/`, `scripts/`, `models/`, `data/` (.gitkeep).

## 18. Dependencies proposed
See `IMPLEMENTATION_PLAN.md`. Only `pyyaml` (+ `pytest` for dev) is needed today. `MetaTrader5` is Windows-only.

## 19. Decisions (resolved by the reviewer after Phase 0)
- **D-1** MT5 on a Windows development machine; architect for a later Windows VPS.
- **D-2** MT5 DEMO account first; broker configurable.
- **D-3** Start with broker MT5 tick history; keep a `HistoricalDataProvider` abstraction for external tick data later.
- **D-4** No Kafka/Redis/message broker; simplest reliable same-machine architecture with preserved seams.
- **D-5** Economic-calendar provider stays abstract until providers are evaluated.
- **D-6** No invented profitability/win-rate thresholds; thresholds derived from baseline empirical research; shadow validation by trade count and regime coverage.
- **D-7** Two independent safety concepts: HALT_NEW_TRADES and EMERGENCY_FLATTEN (the latter needs a much stronger trigger or an authorised command).
- **D-8** Java examples moved to `legacy/java/`.
- **D-9** Private/proprietary; keep attribution and licensing records (`THIRD_PARTY_NOTICES.md`).
- **D-10** Environment-based secrets, `.gitignore`, automated secret scanning in CI.
- **D-11** React + TypeScript dashboard; Vibe-Trading only as a possible research sidecar/reference, never in the execution path.

## 20. Exact Phase 1 plan
See `IMPLEMENTATION_PLAN.md` -> "Phase 1 — exact plan" (revised in Phase 0.5: 12 steps; read-only persistent-session MT5 adapter, MT5 verification harness for every unverified item, time-base calibration, symbol discovery, resumable tick provider with manifests and `time_basis`, Fake adapter, import-boundary tests). No orders in Phase 1.

## Limitations of this Phase 0 audit
`freqtrade.io` and `mql5.com` were unreachable (egress proxy), the Vibe-Trading README was only partially read, and no upstream code was inspected. MT5 API details and the Freqtrade licence are marked [verify]. No claims about data availability or strategy viability are made.

**Stopping here per instructions. Awaiting review and approval before Phase 1.**
