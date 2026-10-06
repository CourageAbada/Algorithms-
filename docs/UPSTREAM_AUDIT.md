# Upstream Audit

Status: Phase 0 audit, **superseded by `PHASE0_5_UPSTREAM_DEEP_DIVE.md`** (source-level inspection). Where the two differ, the deep dive wins. Reviewer sign-off: pending.

## Method and limits of this (Phase 0) audit — corrected in Phase 0.5

- Sources actually read: GitHub landing pages and READMEs for Vibe-Trading, Qlib, Freqtrade; the FreqAI doc source (`docs/freqai.md` on GitHub); the PyPI page for `MetaTrader5`.
- `freqtrade.io` and `mql5.com` were **blocked by the sandbox egress proxy**. MT5 API details below that are not on PyPI come from the author's prior knowledge of the official `MetaTrader5` package and are marked **[verify]**. Phase 1 must verify each against the official docs and against a live demo terminal.
- No upstream source code was cloned or executed in Phase 0. Statements about internals are from READMEs and must be re-checked before any reuse. Licenses must be confirmed from each repo's `LICENSE` file before any code is copied.

## Summary decision

| Upstream | Decision | One-line reason |
|---|---|---|
| Vibe-Trading (MIT) | **Reference + optional sidecar** for the offline research agent | Good agent/skills/"evidence-gated" design; no FX execution engine; not for tick-level decisions |
| Qlib (MIT) | **Reference only** (ideas: dataset/handler split, rolling/online model management, recorder, IC analysis) | Equity/cross-sectional, calendar-based daily/minute; tick/bid-ask/CFD costs not native |
| Freqtrade / FreqAI (GPL-3.0 **[verify]**) | **Reference only, no code copied** | Crypto-exchange centric; copyleft would infect this codebase if code were copied |
| MetaTrader5 Python package | **Wrap behind `BrokerAdapter`** | Only practical MT5 route; Windows-only, synchronous, blocking |

## 1. Vibe-Trading (https://github.com/HKUDS/Vibe-Trading)

**What it is (from README):** a Python 3.11+ agent platform: FastAPI backend, React 19 frontend, optional Electron desktop, Docker. LLM agents with a modular *skills* system and a *swarm* of workers (retry/backoff, replay), an MCP server (README claims 74+ tools), 27+ data loaders, four backtest engines (equity, China A-share, crypto perpetuals, options), 18 broker connectors, and a "grounding gate" that cuts numeric claims not supported by tool output. License: **MIT**.

1. **Reuse directly:** nothing in the trading path. Possibly (Phase 10/11) run it as an *external sidecar* that reads our journal/backtest outputs through a read-only interface.
2. **Wrap behind an adapter:** the research agent, if adopted, behind our `agents/` interface with read-only data access and no order or risk-config write path.
3. **Architectural inspiration only:** evidence-gated answers (every number the agent states must trace to a tool result), skill allow-lists per worker, swarm for parallel experiment analysis, durable client-order-id recovery for live orders, "price caliber" labeling (adjusted/raw/unknown), warm-up vs evaluation window separation.
4. **Rewrite:** all execution, backtest, risk, feature and model code. Their engines encode equity/crypto rules (T+1, lot of 100, funding, stamp tax).
5. **Unsuitable for FX scalping:** README states MT5 is a *data source only* and "no native FX execution engine (routes through spot venues)". No bid/ask-aware tick backtest, no CFD contract/tick-value model, no swap/rollover/stop-level/freeze-level logic.
6. **Licensing:** MIT. Compatible with any license we choose; keep copyright notice if code is ever copied. Dependencies of the project are separate and not audited.
7. **Performance:** an LLM-agent platform; latency and nondeterminism rule it out of the tick-time loop. Fine for offline analysis.
8. **Crypto/equity assumptions to avoid:** 24/7 calendar or exchange trading calendar, central limit order book with a single venue price, fractional/lot-size share conventions, funding rates, adjusted-price handling, single "close" fill model.

Caveat: only the first ~100k characters of the (very long) README were read.

## 2. Microsoft Qlib (https://github.com/microsoft/qlib)

**What it is:** AI-oriented quant research platform: compact data layer, "Quant Dataset Zoo" (Alpha158/Alpha360), 25+ model implementations (LightGBM, XGBoost, LSTM, Transformer, TabNet, TFT...), a workflow runner (`qrun`), a backtester and analysis (IC, Sharpe, drawdown), and online serving. Markets in README: US and China equities, daily and 1-minute data. Python 3.8-3.12. License: **MIT**.

1. **Reuse directly:** nothing initially. Optionally individual model definitions as *reading material*; we prefer thin, tested wrappers over scikit-learn/LightGBM/XGBoost/PyTorch that we control.
2. **Wrap:** none planned. If adopted later for experiment tracking, only behind our `models/registry` interface.
3. **Inspiration:** (a) the *DataHandler/Dataset* split between raw data, processors and segments; (b) rolling retraining and online-model management as a pipeline concept; (c) *recorder* style experiment artefacts; (d) signal-quality analysis (IC / rank-IC, decile/quantile analysis) adapted to a single-instrument setting.
4. **Rewrite:** data storage (tick/bid-ask, UTC, Parquet), backtester (bid/ask, spread, slippage, latency, contract metadata), cost model.
5. **Unsuitable:** cross-sectional stock ranking (TopK-Drop style), trading-calendar assumptions, adjusted-price/ dividend handling, A-share conventions, daily-bar default. Not designed for tick-level, always-open, decentralised FX with variable spread.
6. **Licensing:** MIT; low risk.
7. **Performance:** its binary data layer is fast for cross-sectional bar queries, but adds a heavy Cython/own-format dependency and does not remove our need for tick storage. Not adopted.
8. **Equity-specific assumptions:** limit up/down, T+1, share lots, instrument universes and benchmarks, factor-neutral portfolio construction.

## 3. Freqtrade / FreqAI (https://github.com/freqtrade/freqtrade)

**What it is:** crypto trading bot (spot and futures exchanges via CCXT), Telegram/WebUI, backtesting, hyperopt. FreqAI is its adaptive-ML module: user-defined feature and label functions, automatic data cleaning/normalisation/PCA, outlier detection (dissimilarity index, SVM), **sliding-window periodic retraining**, background training thread in live mode, persisted models for fast restart, 8 example models. README: Python 3.11+. License: **GPL-3.0 [verify in LICENSE]**; the README fetched did not state it.

1. **Reuse directly:** nothing. Copying GPL code into this repo would force the combined work to be GPL-3.0.
2. **Wrap:** no. Running Freqtrade as a process is possible but it has no FX/CFD/MT5 broker support.
3. **Inspiration (concepts are not copyrightable; expression is):** train-on-sliding-window then predict-until-expiry; model identifier per (pair, train-end-time); expiring and purging stale models; outlier/novelty gate at inference (we will implement our own Mahalanobis/isolation-forest/PSI gate); separation of feature function and label function; crash-safe model persistence.
4. **Rewrite:** all of it, from the spec in `docs/ML_ARCHITECTURE.md`.
5. **Unsuitable:** exchange and pair-list abstraction, candle-only (not tick) data model, spot-style fill assumptions, no spread/commission/swap/contract-size model, "buy/sell signal" semantics with no NO_TRADE meta layer, hyperopt-on-same-data culture that invites overfitting.
6. **Licensing:** see above. Clean-room policy: contributors must not paste Freqtrade code into this repo.
7. **Performance:** FreqAI trains on candle-frequency features (10k+ features easily); feasible, but uses candles, not our tick pipeline.
8. **Crypto assumptions:** 24/7 candles, exchange-published OHLCV as truth, volume as true volume, pair lists by volume, maker/taker fee model.

## 4. MetaTrader 5 Python integration

**Facts verified from PyPI:** package `MetaTrader5`, version 5.0.6231 (released 2026-09-27), Python 3.6-3.14, **Windows x86-64 only**, maintained by MetaQuotes, license field MIT. The terminal itself is proprietary and governed by the broker/MetaQuotes terms.

**[verify] API surface:** `initialize`, `login`, `shutdown`, `version`, `last_error`, `account_info`, `terminal_info`, `symbols_total/get`, `symbol_info`, `symbol_info_tick`, `symbol_select`, `market_book_add/get/release`, `copy_rates_from/from_pos/range`, `copy_ticks_from/range`, `orders_total/get`, `positions_total/get`, `history_orders_total/get`, `history_deals_total/get`, `order_calc_margin`, `order_calc_profit`, `order_check`, `order_send`.

1. **Reuse:** the official package, unmodified, via `MT5BrokerAdapter`.
2. **Wrap:** everything. No MT5 structure may leave `brokers/mt5/`.
3. **Inspiration:** `order_check` as a pre-flight; `order_calc_margin/profit` for margin; `symbol_info` for contract metadata.
4. **Rewrite:** nothing; adapt returned named tuples/arrays into our neutral dataclasses at the boundary.
5. **Unsuitable / risks:** Windows-only (cloud Linux sandboxes cannot run it); needs a running terminal logged into the broker; the API is synchronous and blocking (isolate in a dedicated worker thread/process); returned timestamps are **broker server time encoded as epoch**, not true UTC **[verify]**, so the offset and DST must be derived per broker and tested; tick history depth and quality are broker-dependent; DOM (`market_book`) often unavailable or partial for FX/CFDs; account may be netting or hedging; filling modes and execution mode are per symbol; real-money use needs the broker's terms to permit automated trading.
6. **Licensing:** the wrapper is MIT per PyPI; broker/terminal EULAs apply.
7. **Performance:** polling `copy_ticks_from` is practical at sub-second cadence but not an event stream; expect tens of ms per call. Measure in Phase 1 (latency baseline).
8. **FX-specific assumptions:** none to remove; this is the correct domain. Symbol suffixes (`XAUUSDm`, `GOLD`) vary by broker.

## Licensing summary and policy

- Vibe-Trading, Qlib, MT5 wrapper: MIT (permissive).
- Freqtrade: copyleft (GPL-3.0, verify). **Policy: no Freqtrade code in this repo.**
- Project license for this repository: **undecided** (decision D-9 in `PHASE0_REPORT.md`).
