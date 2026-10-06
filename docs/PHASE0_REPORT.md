# Phase 0 Final Report

Status: **complete, awaiting human review. Phase 1 has not been started.**
No broker connection was made, no orders were sent, no backtest was run, and no performance or win-rate claim is made anywhere.

## 1. Architecture chosen
Own modular Python 3.11+ platform (`src/fxscalp`), one-way layered pipeline (ticks -> bars -> features -> regime -> models -> calibration -> meta-model -> LONG/SHORT/NO_TRADE -> quality filter -> **independent risk engine** -> execution -> broker). `BrokerAdapter` abstraction, MT5 first. Same code path for REPLAY/SHADOW/DEMO/LIVE with swappable execution sinks. Collector process next to MT5 (Windows), engine/research platform-independent. See `SYSTEM_ARCHITECTURE.md`, `ARCHITECTURE_AUDIT.md`.

## 2. Vibe-Trading
MIT. Used as **reference** only: evidence-gated agent (every number traceable to tool output), skills/swarm for offline experiment analysis, MCP-style read-only tool access, durable client-order-id recovery, price-caliber labelling. Possible read-only sidecar in Phase 10/11. Not used: its backtest engines, data loaders, broker connectors (no FX/MT5 execution).

## 3. Qlib
MIT. **Reference** only: DataHandler/Dataset segment separation, rolling/online model pipeline, recorder-style artefacts, IC-type signal analysis. Not used: equity factor sets, binary store/calendar, cross-sectional strategies.

## 4. FreqAI
Freqtrade is copyleft (GPL-3.0, to verify in LICENSE). **Concepts only, no code copied**: sliding-window retrain with predict-until-expiry, feature/label function separation, outlier/novelty gate, model persistence/purge, background training. Reimplemented clean-room per `ML_ARCHITECTURE.md`.

## 5. MT5 integration
Official `MetaTrader5` package (v5.0.6231, **Windows-only**, per PyPI), imported only inside `brokers/mt5`, run in a Windows collector process that owns the terminal connection and exposes neutral dataclasses over a local channel. Server-time -> UTC conversion at the boundary. `order_check` before any order; account must verify as demo for DEMO mode. Details marked [verify] in the audit must be confirmed in Phase 1 (official docs were unreachable from this sandbox).

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

## 19. Decisions requiring human approval
- **D-1** Where does MT5 run (Windows VPS/PC)? Required for Phase 1; sandbox is Linux.
- **D-2** Broker and demo account (affects symbol names, spreads, tick depth, scalping permissions). Must be demo.
- **D-3** Required history depth and tick vendor (if any beyond the broker).
- **D-4** Collector<->engine transport (ZeroMQ vs local socket/gRPC) and process layout.
- **D-5** Economic-calendar provider (point-in-time, cost).
- **D-6** Acceptance thresholds (expectancy CI, PF, fold-consistency, min trades, shadow duration, holdout length), proposed in `XAUUSD_RESEARCH_PLAN.md` / `VALIDATION_PROTOCOL.md`.
- **D-7** Kill-switch policy for open positions (block-only vs flatten).
- **D-8** Existing `BubbleSort.java`, `QuickSort.java` in repo root: keep, move to `legacy/`, or delete.
- **D-9** Repository licence (copyleft upstreams are not copied, so any licence is possible).
- **D-10** Secret-management approach and CI secret scanning.
- **D-11** Frontend stack confirmation (React + FastAPI proposed) and whether Vibe-Trading is evaluated as a sidecar in Phase 10/11.

## 20. Exact Phase 1 plan
See `IMPLEMENTATION_PLAN.md` -> "Phase 1 - exact plan" (11 steps; read-only MT5 adapter, symbol discovery, server-time offset verification, resumable tick downloader with manifests, latency probe, Fake adapter for Linux tests, import-boundary tests, report). No orders in Phase 1.

## Limitations of this Phase 0 audit
`freqtrade.io` and `mql5.com` were unreachable (egress proxy), the Vibe-Trading README was only partially read, and no upstream code was inspected. MT5 API details and the Freqtrade licence are marked [verify]. No claims about data availability or strategy viability are made.

**Stopping here per instructions. Awaiting review and approval before Phase 1.**
