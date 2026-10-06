# Implementation Plan

Phases are gated; a profitable-looking backtest never skips a phase. Phase 0 is complete pending review. **Nothing past Phase 0 starts without approval.**

| Phase | Scope | Exit gate |
|---|---|---|
| 0 | Audit, architecture docs, skeleton | Human review of `PHASE0_REPORT.md` and decisions D-1..D-10 |
| 1 | MT5 connection, symbol discovery, historical tick/bar acquisition, latency baseline | Reproducible tick download for XAUUSD (+EUR/GBP) with manifest; symbol metadata snapshots; measured MT5 call latency |
| 2 | Tick/bar storage, data-quality pipeline, holdout sealing, cost-floor/data profile (E1, E2) | Quality report; sealed holdout hash; spread/session statistics |
| 3 | Feature engine (incremental + vectorised, equality-tested), session engine, leakage tests | All features pass truncation tests; hypothesis registry complete |
| 4 | XAUUSD baselines (B1-B3) + ML baselines (LR, LGBM, XGB, ...), triple-barrier labels, calibration | Baselines reported with costs; no claim of edge yet |
| 5 | Realistic tick backtester + metrics + segmentation | Tested against hand-computed trades; reduced-fidelity labelling |
| 6 | Walk-forward, purged CV, robustness (MC, stress), trial registry | Pre-registered criteria evaluated; holdout touched at most once |
| 7 | Market replay (production code path, speeds 1x/10x/100x/max) | Replay-vs-backtest trade parity |
| 8 | Shadow mode on live MT5 data (never sends orders) | Shadow vs historical expectations within bands |
| 9 | MT5 demo trading, execution-quality measurement | Demo vs shadow/backtest comparison; slippage model recalibrated |
| 10 | Dashboard + monitoring + drift/alerts | Kill switch and health panels verified with fault injection |
| 11 | Advanced ensemble, meta-labeling, sequence models (if justified) | OOS improvement net of costs, statistically supported |
| 12 | EURUSD model | Independent validation |
| 13 | GBPUSD model | Independent validation |
| 14 | Controlled live-readiness assessment (not enablement) | Human decision |

## Phase 1 — exact plan (proposed; awaiting approval)

Prerequisite decisions: D-1 (where MT5 runs), D-2 (broker/demo account), D-3 (data depth), D-4 (collector<->engine transport).

1. **Dev setup:** pinned `requirements` (see below), `ruff`, `mypy`, `pytest`, pre-commit with secret scanning (gitleaks), CI workflow running lint+unit tests on Linux (MT5 tests skipped there).
2. **Import-boundary tests:** assert `risk` does not import `models`; `agents` cannot import `execution`; only `brokers/mt5` imports `MetaTrader5`.
3. **Core types:** finalise `SymbolInfo/Tick/AccountInfo/OrderRequest/OrderResult` (decide float vs Decimal for order-facing fields), error enum, UTC time utilities (`time_msc` <-> aware datetime).
4. **`FakeBrokerAdapter`:** deterministic, scriptable (ticks, rejects, disconnects) for Linux tests.
5. **`MT5BrokerAdapter` (read-only first):** `connect/disconnect/health_check`, `get_account`, `get_symbol_info`, `get_ticks` (via `copy_ticks_range`), `get_positions/get_orders`. `submit_order/modify/cancel/close` raise `NotEnabledInPhase1`. Credentials via environment only. Demo-account assertion.
6. **Symbol discovery:** resolve canonical -> broker symbol from `broker_symbol_candidates` plus a fuzzy catalogue search; `symbol_select`; persist a timestamped `SymbolInfo` snapshot; fail loudly on ambiguity.
7. **Server-time offset:** derive and verify the broker server time offset/DST rule against the latest tick vs system UTC; unit tests for DST transitions; store the offset provenance.
8. **Historical download tool (`scripts/download_ticks.py`):** chunked `copy_ticks_range` (by day), retries with bounded backoff, resumable, writes immutable Parquet partitions + manifest (counts, min/max time, SHA-256, broker, symbol, server, schema version). Bars (`copy_rates_*`) downloaded only for cross-checks.
9. **Latency/availability probe:** measure `symbol_info_tick`, `copy_ticks_from`, `order_check` (no order sent) latency distributions; record achievable tick-history depth per symbol.
10. **Tests:** adapter contract tests run against Fake and (on Windows, opt-in marker) against MT5 demo; manifest tests; time-conversion tests.
11. **Deliverable report:** `docs/PHASE1_REPORT.md` with measured depth, rates, latencies, problems. Then STOP for review.

Phase 1 sends **no orders** (adapter order methods disabled).

## Proposed dependencies (not installed in Phase 0)

- Core: Python >= 3.11, `pyyaml`, `pydantic` (config/schemas), `numpy`, `pandas`, `polars`, `pyarrow`, `duckdb` (queries), `anyio`/`asyncio`, `pyzmq` (collector channel, D-4).
- ML: `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `optuna`, `shap`, later `torch`.
- Stats: `scipy`, `statsmodels`.
- Data: `MetaTrader5` (Windows only), a tick-vendor client TBD.
- Services: `fastapi`, `uvicorn`, `sqlalchemy`, `alembic`, `psycopg`; frontend `react`, `vite`, charting library.
- Dev: `pytest`, `pytest-asyncio`, `hypothesis`, `ruff`, `mypy`, `pre-commit`, `gitleaks`.
All licenses to be checked (no GPL in the runtime closure unless approved).
