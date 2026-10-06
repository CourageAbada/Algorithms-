# System Architecture

> Revised in Phase 0.5 (see `PHASE0_5_UPSTREAM_DEEP_DIVE.md`). Changes: same-machine, no message broker (D-4); `HistoricalDataProvider` abstraction (D-3); two safety concepts (D-7); persistent single-owner MT5 session; server-time calibration as a first-class component; three vendored utilities (purged CV, multiple-testing stats, audit ledger).

## 1. Dataflow (single code path for REPLAY / SHADOW / DEMO / LIVE)

```
MT5 terminal (Windows) ──► MT5BrokerAdapter ──► Tick Engine (validate, dedupe, order, health)
                                                   │ raw ticks ─► Parquet raw store (immutable)
                                                   ▼
                                           Bar Aggregator (1s…1h, bid/ask/mid)
                                                   ▼
        Session Engine ─┐                  Feature Engine (incremental, versioned)
        News/Event Risk ┼──────────────────►        ▼
                                           Regime Engine (rules + ML)
                                                   ▼
                              Model Ensemble ─► Calibration ─► Meta-model
                                                   ▼
                              Decision: LONG / SHORT / NO_TRADE (+ expected edge)
                                                   ▼
                                    Trade-Quality Filter (cost/edge/spread)
                                                   ▼
                                    RISK ENGINE (independent, can veto/size)
                                                   ▼
                            ExecutionSink  ┬ SimulatedSink   (BACKTEST/REPLAY)
                                           ├ ShadowSink      (SHADOW: hypothetical, no order API)
                                           └ BrokerSink      (DEMO/LIVE) ─► BrokerAdapter ─► MT5
                                                   ▼
                       Journal · Metrics · Drift monitors · Dashboard API
```

## 2. Process model (Phase 0.5: simplest reliable same-machine design, D-1/D-4)

Initial deployment is **one Windows machine** (development PC; migratable to a Windows VPS later). No Kafka/Redis/message broker. The design keeps seams so that processes can be separated later without touching business logic.

- **Process A, trading engine (single process, asyncio):** MT5 adapter + tick engine + features + regime + models + signals + risk + execution + journal. The MT5 adapter runs all `MetaTrader5` calls on **one dedicated worker thread** that owns a **single long-lived terminal session** (attach once; never initialise/shutdown per call) and serialises calls with a lock (the API's thread-safety is UNVERIFIED). The rest of the engine talks to it through `BrokerAdapter`/`MarketDataSource` interfaces and in-process `asyncio.Queue` event channels.
- **Seams for later separation:** `MarketDataSource` (live MT5, replay, shadow feed), `ExecutionSink` (simulated, shadow, broker), `EventChannel` (in-process queue today; a socket/IPC implementation can be dropped in if the MT5 adapter is split into its own Windows service). Nothing outside `brokers/mt5` imports `MetaTrader5`, so on Linux (research, replay, CI) everything runs against `FakeBrokerAdapter` and Parquet data.
- **Training workers:** run as separate OS processes started by a CLI/scheduler on the same machine (heavy CPU must not stall the engine); they write candidate artefacts to the registry only.
- **API/dashboard (D-11):** FastAPI read-mostly service + **React + TypeScript** frontend. Write endpoints: engage HALT_NEW_TRADES, and *request* EMERGENCY_FLATTEN (requires a separate authorised confirmation, below).
- **Research agent / Vibe-Trading sidecar (optional, Phase 10/11):** separate process, read-only access to journal and backtest artefacts; never on the order path and never in the latency-critical loop.

## 3. Key interfaces (contracts)

- `BrokerAdapter` (see `src/fxscalp/brokers/base.py`): neutral dataclasses `SymbolInfo`, `Tick`, `AccountInfo`, `OrderRequest`, `OrderResult`.
- `HistoricalDataProvider` (see `src/fxscalp/market_data/provider.py`, D-3): `provenance()` and `get_ticks()` returning UTC ticks plus a `DatasetProvenance` (provider, broker, server, **time basis**: `utc_verified | server_time_calibrated | unverified`). First implementation: broker MT5 tick history; an external higher-quality tick vendor can be added later without touching features/models/risk. Data with an `unverified` time basis is not usable for training.
- `TimeBase` (planned): empirically estimates the MT5 server-time offset per broker (documentation and community reports conflict, see deep dive §4 #15), stores its provenance, and is the only place that converts raw MT5 timestamps to UTC.
- `FeatureSet`: `name, version, hypothesis, lookback, update(tick|bar) -> values`, pure function of past data only.
- `Predictor`: `predict(features, context) -> ProbabilityVector{long, short, no_trade}` + `model_version`, `feature_version`.
- `TradeProposal`: direction, entry reference, SL, TP, horizon, expected gross move, expected net edge, probabilities, regime, session, reasons.
- `RiskDecision`: `APPROVE(size, adjusted SL/TP) | REJECT(reason codes) | REDUCE`. Immutable, journaled.
- `ExecutionSink.submit(RiskDecision) -> ExecutionReport`.

## 4. Mode semantics

| Mode | Data | Orders | Sink |
|---|---|---|---|
| BACKTEST | historical ticks/bars | simulated | SimulatedSink |
| REPLAY | historical ticks, event-driven, speed 1x/10x/100x/max | simulated | SimulatedSink |
| SHADOW | live MT5 data | **never sent** | ShadowSink (no reference to order methods) |
| DEMO | live MT5 demo | sent to demo account | BrokerSink (account must verify `is_demo`) |
| LIVE | live | sent | BrokerSink + interlock (`core/modes.py` + further checks); disabled by default |

ShadowSink is a separate class that does not hold a `BrokerAdapter` reference at all, so a shadow bug cannot send an order.

## 5. Safety concepts and failure policy (D-7)

Two **independent** concepts (`src/fxscalp/core/safety.py`):

- **HALT_NEW_TRADES** — prevents additional exposure. Cheap, may be engaged automatically by any check: stale feed, MT5 disconnect, NaN/invalid prediction, feature failure, DB outage (journal cannot persist), news-feed outage (if policy requires), clock/time-base anomaly, abnormal spread, daily-loss or drawdown limits, model/drift failure, reconciliation mismatch, manual. Existing positions keep their broker-side SL/TP.
- **EMERGENCY_FLATTEN** — attempts controlled closure of existing exposure. Requires a **substantially stronger trigger**: an explicit authorised command (token supplied from a human-controlled secret; with no token configured it cannot be authorised at all) or a human-configured hard-loss trigger. It is never engaged by an ordinary health check, drift alarm or model failure. Closing is done **by position ticket** (never by sending a blind opposite order, which opens a hedge on hedging accounts), with bounded retries, per-position result logging, and a final reconciliation; it also halts new trades.

Releasing either state is a deliberate human act. Every transition is written to the audit ledger.

## 6. Storage (see `DATA_REQUIREMENTS.md`)

Parquet (raw ticks, bars, features) + PostgreSQL (orders, fills, trades, journal, models, experiments, risk events), TimescaleDB optional for metrics.

## 7. Observability

JSON logs with secret redaction (`monitoring/logging_setup.py`), metrics for feed rate, staleness, feature/inference/exec latencies, slippage, rejects, drawdown; alert thresholds drive the dashboard SYSTEM panel and the kill switch.

## 8. Dashboard

React + FastAPI. Panels: MODE banner, ACCOUNT, PERFORMANCE, TRADING, AI (probabilities, NO_TRADE reason, regime, model version, calibration), MARKET (3 instruments), SYSTEM (MT5, feed, model, latency, DB, kill switch). Phase 10.
