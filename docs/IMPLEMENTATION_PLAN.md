# Implementation Plan

Phases are gated; a profitable-looking backtest never skips a phase. Phase 0 and Phase 0.5 (upstream deep dive + MT5 verification) are complete pending review. **Nothing past Phase 0.5 starts without approval.**

| Phase | Scope | Exit gate |
|---|---|---|
| 0 / 0.5 | Audit, architecture docs, skeleton; source-level upstream inspection; MT5 API verification | Human review of `PHASE0_5_UPSTREAM_DEEP_DIVE.md` |
| 1 | Read-only MT5 adapter, MT5 verification harness, symbol discovery, time-base calibration, historical tick acquisition, latency baseline | Verification report with every UNVERIFIED item resolved or still flagged; reproducible tick download (XAUUSD first) with manifest and `time_basis` |
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

## Phase 1 — STATUS: implemented and tested on Linux against a fake; real-MT5 verification PENDING (see `PHASE1_REPORT.md`)

Original exact plan (revised in Phase 0.5):

Approved decisions: D-1 MT5 on a Windows development machine (VPS later); D-2 MT5 **demo** account, broker configurable; D-3 start with broker MT5 tick history behind `HistoricalDataProvider`; D-4 same-machine, no message broker; D-5 calendar provider abstract; D-6 no invented thresholds; D-7 HALT_NEW_TRADES / EMERGENCY_FLATTEN; D-8 Java moved to `legacy/java/`; D-9 proprietary; D-10 env secrets + CI secret scanning; D-11 React + TypeScript.

Principle: **verify before trusting.** Phase 1 turns every UNVERIFIED MT5 assumption (deep dive §4) into a test, and produces data only with a known time basis. Phase 1 sends **no orders**: `submit_order/modify_order/cancel_order/close_position` raise `NotEnabledInPhase1`; `order_check` may be used only for filling-mode/stop-level probing after you approve that step, and `order_send` is never called.

1. **Dev setup (Windows + Linux CI):** pinned Python/requirements, `ruff`, `mypy`, `pytest`, pre-commit with gitleaks. Confirm the CI workflow (added in Phase 0.5, untested on GitHub) passes; fix if not.
2. **Import-boundary tests:** `risk` must not import `models`; `agents` cannot import `execution`; only `brokers/mt5` imports `MetaTrader5`; only `execution` may call order methods.
3. **Core types and time:** finalise `SymbolInfo/Tick/AccountInfo/OrderRequest/OrderResult` (float vs Decimal decision for order-facing fields), neutral error enum mapped from `TRADE_RETCODE_*` and `RES_*`, UTC utilities (`time_msc` <-> aware datetime).
4. **`FakeBrokerAdapter`** (deterministic, scriptable ticks/rejects/disconnects/server-time offsets) so all non-MT5 logic is testable on Linux.
5. **`MT5BrokerAdapter` (read-only), persistent session:** one owner thread + lock; attach once; `connect/disconnect/health_check`, `get_account` (with **demo/contest/real guard** from `trade_mode`; DEMO mode refuses non-demo), `get_symbol_info`, `get_ticks` (via `copy_ticks_range`), `get_positions/get_orders`. Credentials from environment only; never logged.
6. **MT5 verification harness (`scripts/mt5_verify.py`, Windows + DEMO only):** runs on the dev machine and writes `docs/MT5_VERIFICATION_REPORT.md` (machine-generated, reviewed by you). It must check and record, per instrument (XAUUSD, EURUSD, GBPUSD): (a) struct fields and numeric types for `terminal_info/account_info/symbol_info/tick`; (b) **server-time offset**: latest tick time vs system UTC over several samples while the market is open, repeated across days, and once across a DST change if possible; (c) `copy_ticks_range` row limits and achievable history depth; (d) `symbol_info` consistency (`tick_value` vs `contract_size * tick_size`, digits/point, volume step, stops/freeze levels); (e) margin mode (netting vs hedging) and execution mode; (f) tick-flag semantics; (g) call latency distributions (`symbol_info_tick`, `copy_ticks_from`, `order_calc_margin`); (h) re-verify against the official docs from the Windows machine. Items that cannot be verified stay UNVERIFIED in the report.
7. **Symbol discovery:** resolve canonical -> broker symbol from `broker_symbol_candidates` plus wildcard search; `symbol_select`; refuse ambiguity (no guessing); snapshot `SymbolInfo` with a timestamp.
8. **`TimeBase`:** compute and persist the calibrated server-time offset with provenance and confidence; unit tests across DST transitions with the Fake adapter; flag data `time_basis=unverified` when calibration is missing.
9. **`MT5HistoricalDataProvider` (D-3):** chunked (by day), resumable, bounded-backoff `copy_ticks_range`; immutable Parquet partitions with raw timestamps preserved + manifest (counts, min/max, SHA-256, broker, server, symbol, schema, `time_basis`). Bars downloaded only for cross-checks.
10. **Vendoring decisions (only if you approve):** import `crossvalidation.py`, `multipletesting.py`, `ledger.py` with attribution (`THIRD_PARTY_NOTICES.md`) and our own property tests (e.g. no train label span overlaps a test span; DSR checked against literature values; ledger tamper tests and a Windows lock test). Otherwise defer to Phase 3/6.
11. **Tests:** adapter contract tests against Fake (CI) and against MT5 demo (opt-in Windows marker); manifest tests; time-conversion tests; guard tests (real account refused in DEMO).
12. **Report `docs/PHASE1_REPORT.md`:** measured depth, tick rates, spreads by session (descriptive only), latencies, verification results, problems. Then STOP for review.

## Phase 2A — STATUS: implemented and tested on synthetic data only (see `PHASE2A_REPORT.md`)

Raw -> normalization -> bars (1s..5m) -> 119-feature `xauusd_core` engine -> validation -> ML-ready dataset with provenance; leakage suite incl. prefix-invariance; spread/volatility regime engines (UNCALIBRATED until real data). Gate not bypassed: the Phase 1 Windows MT5 DEMO verification is still outstanding. No model training, strategy, labels or orders.

Phase 2B (proposed, needs approval; requires real verified XAU/USD data): run `build_features` on the real Phase 1 dataset, calibrate regime reference quantiles on a separate reference period, profile feature distributions, then design labels (with purge/embargo) — still no profitability claims.

## Later-phase notes carried from Phase 0.5

- Phase 3: leakage truncation tests; features per `FEATURE_CATALOG.md`.
- Phase 4: instrument-specific (XAUUSD-only) baselines first; derive validation thresholds from baseline empirics (D-6).
- Phase 5: bid/ask tick backtester built in-house (the Vibe-Trading Forex engine was inspected and rejected as a base); metrics: adapted trade statistics + expectancy, R, MFE/MAE, cost breakdown, segmentation.
- Phase 6: purged walk-forward, CPCV, DSR/PBO, block bootstrap, trade-order reshuffle drawdown.
- Phase 10: React + TypeScript dashboard; optional read-only Vibe-Trading research sidecar evaluation.

## Proposed dependencies (not installed in Phase 0)

- Core: Python >= 3.11, `pyyaml`, `pydantic` (config/schemas), `numpy`, `pandas`, `polars`, `pyarrow`, `duckdb` (queries), standard-library `asyncio`/`threading`/`queue` (no message broker, D-4). `scipy` if `multipletesting.py` is vendored.
- ML: `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `optuna`, `shap`, later `torch`.
- Stats: `scipy`, `statsmodels`.
- Data: `MetaTrader5` (Windows only), a tick-vendor client TBD.
- Services: `fastapi`, `uvicorn`, `sqlalchemy`, `alembic`, `psycopg`; frontend `react`, `vite`, charting library.
- Dev: `pytest`, `pytest-asyncio`, `hypothesis`, `ruff`, `mypy`, `pre-commit`, `gitleaks`.
All licenses to be checked (no GPL in the runtime closure unless approved).
