# System Architecture

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

## 2. Process model

- **Collector process (Windows, next to MT5):** owns the terminal connection, the only process that imports `MetaTrader5`. Publishes normalised ticks/account/positions over a local channel (ZeroMQ or a socket; decision D-4) and executes order commands received from the engine. Rationale: the MT5 API is blocking, Windows-only and terminal-bound.
- **Engine process:** features -> signals -> risk -> execution orchestration (asyncio event loop). Platform independent, runs on Linux for research/replay.
- **Training workers:** separate processes; never share memory with the live engine; output candidate artefacts to the registry.
- **API/dashboard:** FastAPI read-mostly service + React frontend. The only write endpoints: kill switch (engage only; release requires CLI + confirmation) and mode-neutral acknowledgements.
- **Research agent:** separate process, read-only data access.

## 3. Key interfaces (contracts)

- `BrokerAdapter` (see `src/fxscalp/brokers/base.py`): neutral dataclasses `SymbolInfo`, `Tick`, `AccountInfo`, `OrderRequest`, `OrderResult`.
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

## 5. Failure policy

Any of: stale feed, MT5 disconnect, NaN/invalid prediction, feature failure, DB outage (journal cannot persist), news-feed outage (if the policy requires it), clock anomaly, abnormal spread, kill switch -> **no new trades**. Existing positions keep their broker-side SL/TP; closing logic is explicit and tested (policy D-7).

## 6. Storage (see `DATA_REQUIREMENTS.md`)

Parquet (raw ticks, bars, features) + PostgreSQL (orders, fills, trades, journal, models, experiments, risk events), TimescaleDB optional for metrics.

## 7. Observability

JSON logs with secret redaction (`monitoring/logging_setup.py`), metrics for feed rate, staleness, feature/inference/exec latencies, slippage, rejects, drawdown; alert thresholds drive the dashboard SYSTEM panel and the kill switch.

## 8. Dashboard

React + FastAPI. Panels: MODE banner, ACCOUNT, PERFORMANCE, TRADING, AI (probabilities, NO_TRADE reason, regime, model version, calibration), MARKET (3 instruments), SYSTEM (MT5, feed, model, latency, DB, kill switch). Phase 10.
