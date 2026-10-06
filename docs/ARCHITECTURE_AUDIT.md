# Architecture Audit

Status: Phase 0, pending human review.

## 1. Repository inspection

Initial state of `CourageAbada/Algorithms-`: `README.md` (algorithm-notes), `BubbleSort.java`, `QuickSort.java`. No Python, no history relevant to trading. The Java files are unrelated; they are left untouched in the repo root (decision D-8: move to `legacy/` or remove).

Environment constraints discovered:
- Cloud sandbox is **Linux**; the `MetaTrader5` package is **Windows-only**. The MT5 adapter therefore cannot execute in this environment. It must be developed against a Windows host, and all other layers must be testable on Linux via a `FakeBrokerAdapter` and recorded data.
- Egress is proxy-restricted (some documentation hosts blocked). Data vendors may also be unreachable from the sandbox; heavy data work happens on the user's machine/VPS.

## 2. Candidate architectures considered

| Option | Description | Verdict |
|---|---|---|
| A. Fork Freqtrade/FreqAI + add MT5 | Reuse bot shell | Rejected: GPL, crypto exchange abstraction, candle fills, no NO_TRADE/ risk independence |
| B. Fork Vibe-Trading | Reuse agent platform | Rejected for trading path: no FX engine, LLM-centric |
| C. Qlib as core | Reuse data/model/backtest | Rejected: equity/cross-sectional, calendar bars, no bid/ask |
| D. **Own modular Python core, upstream as reference; MT5 behind adapter; Parquet + Postgres; offline LLM agent** | | **Chosen** |
| E. Rust/C++ low-latency core | Max speed | Deferred: retail MT5 round-trips (tens of ms) dominate; Python+NumPy/Polars suffice. Revisit only if measured feature/inference latency is the bottleneck |

## 3. Chosen architecture (summary)

Python 3.11+ monorepo, `src/fxscalp/`, layered pipeline with one-way dependencies:

`brokers -> market_data/ticks -> bars -> features/sessions/news -> regimes -> models -> signals -> risk -> execution -> brokers`

Hard rules (enforced by import-linter-style tests in Phase 1):
1. `risk` must not import `models`; it receives only a typed `TradeProposal` and market/account state.
2. `execution` is the only package allowed to call order methods, and only through `BrokerAdapter`.
3. `agents` has no import path to `execution`, `risk` config writers, or order methods.
4. `shadow`, `replay`, `backtest` use the same `features -> regimes -> models -> signals -> risk` code as live; only the data source and the order sink differ (`ExecutionSink` implementations: Simulated, Shadow-hypothetical, Broker).
5. Every layer receives the active `TradingMode`; the order sink refuses `send` unless `mode.may_send_orders`.

## 4. Key risks found and where they are handled

| Risk | Handling |
|---|---|
| Look-ahead in features/labels | Feature purity contract + truncation tests (`tests/leakage`); labels isolated in `models/labels`; purged CV + embargo |
| Backtest optimism (candle-close fills, no spread) | Bid/ask tick backtester; OHLC-only results labelled `REDUCED_FIDELITY` |
| Broker-specific feed | Train and validate on the **same broker's** tick history that execution will use; record feed provenance in dataset manifest |
| Server-time/DST errors | UTC internally; broker offset derived and tested; session engine tested around DST transitions |
| Tiny edge vs cost | Expected-edge gate (edge > spread + commission + slippage + margin) before risk |
| Silent data staleness | Feed-health gate is part of the risk engine; stale -> no new trades |
| ML overconfidence | Calibration + meta-model + novelty gate; thresholds from validation only |
| Multiple-testing / selection bias | Trial registry; deflated-Sharpe style correction; untouched holdout |
| Operator error going live | Triple interlock + mode-aware sink + demo-account verification |
| LLM in hot path | Forbidden by import rules; agent is offline only |

## 5. Language/performance assessment

Expected load: XAUUSD retail feeds produce on the order of 10^4-10^6 ticks/day depending on broker and session (**measure in Phase 1**). Per-tick feature updates must be incremental O(1) (rolling buffers), not full-window recomputation. Target (to be validated): feature+inference < 20 ms p99 per decision on one core; end-to-end signal-to-order dominated by MT5 round trip.

## 6. Open assumptions to confirm

A-1 Broker supports tick history via MT5 for the needed depth. A-2 Demo account feed is representative of live. A-3 User can host a Windows machine/VPS for the MT5 terminal. A-4 Algorithmic trading of CFDs is permitted by the user's broker/jurisdiction (not legal advice; user's responsibility).
