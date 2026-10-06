# Phase 0.5 — Upstream Deep Dive & Architecture Verification

Status: complete, awaiting review. Supersedes the "limits of this audit" caveats in `UPSTREAM_AUDIT.md`.
Audit date: 2026-10-06. No broker was contacted and no order was sent. No upstream code was copied into this repository.

## 0. What was inspected, and how

| Source | Version inspected | How | Licence (verified from file) |
|---|---|---|---|
| Vibe-Trading | `7f6908b` (main, 2026-10-06), shallow clone | Read source of `agent/backtest/**`, `agent/src/trading/connectors/mt5/**`, `agent/src/live/**`, `agent/src/quantlib/**`, `agent/src/governance/**`, agent/swarm/grounding/hypotheses packages; ran selected upstream test files | MIT (`LICENSE`). `NOTICE` says it bundles Qlib feature definitions "under Apache 2.0" (see §6) |
| Qlib | `be72549` (2026-09-16) | Read `qlib/data/dataset`, `qlib/workflow/task/gen.py`, `qlib/contrib/rolling`, `qlib/model/base.py`, `qlib/backtest/exchange.py`, `examples/highfreq` | MIT (`LICENSE`: "MIT License, Copyright (c) Microsoft Corporation") |
| Freqtrade/FreqAI | `9691649` (2026-10-06) | Read `freqtrade/freqai/{freqai_interface,data_kitchen,data_drawer}.py`; listed model zoo | **GPL-3.0** (verified: `LICENSE` is the GNU GPL v3 text). Read for understanding only |
| MetaTrader5 package | wheel `5.0.6231` (cp312 win_amd64) downloaded with `pip download --platform win_amd64` and **unpacked, not imported or run** | Read `MetaTrader5/__init__.py` (pure-Python part), `METADATA`, `WHEEL`, `LICENSE.txt`; extracted strings from `_core.pyd` | MIT per package metadata; terminal/broker EULAs separate |
| MT5 official docs | not fetchable (`mql5.com` blocked by the sandbox egress proxy) | Web search result summaries only | secondary evidence — flagged as such |

Upstream tests executed in an isolated scratch clone (pure-Python, no network, no broker):
- `tests/quantlib/test_crossvalidation.py` + `test_multipletesting.py`: **86 passed**.
- `tests/test_forex_engine.py` + `test_mt5_loader.py` + `test_mt5_connector.py`: **183 passed, 1 failed** (the failure is a missing optional `httpx` in this sandbox, unrelated to the logic).
- Governance/halt/hypothesis tests: **58 passed, 1 failed** (an environment-package-version test, unrelated).

Passing upstream tests show the code does what *its authors* specified. They do not show it meets *our* requirements; that is judged separately below.

---

## 1. Vibe-Trading — findings

### 1.1 The Forex backtest engine (`agent/backtest/engines/forex.py`, `base.py`, `_market_hooks.py`)

Verified behaviour:
1. **Bar-based, target-weight engine.** User/LLM "signal engine" code emits a signal per bar in [-1, 1]. `_align()` shifts it by one bar (`shifted_vals[1:] = sig_vals[:-1]`) and fills at the **next bar's open** (`execution_open` = `bar["open"]`). The no-look-ahead rule is sound for bar data. Valuation before the fill uses the open, not the unknown close (explicit comments).
2. **No bid/ask, no ticks.** Spread is a **static per-pair table in pips** (`_SPREAD_PIPS`: EUR/USD 1.0, GBP/USD 1.2, XAU/USD 3.2 ...), applied as `half_spread + slippage_pips(0.3)` on every fill. No variable spread, no session/news widening, no commission (`calc_commission` returns 0.0).
3. **No stop-loss, take-profit, trailing or intrabar logic.** `grep` for `stop_loss|take_profit|trailing|intrabar` over the backtest package (excluding options/perp liquidation) returns nothing. Exits happen only when the next signal changes. This cannot express triple-barrier trades, the core of our label design and exit study.
4. **Hardcoded instrument constants instead of broker metadata.** Pip size (`0.0001`, `0.01` JPY, gold `0.10`), lot size (100,000; gold 100 oz via `_METAL_SPECS`), micro-lot rounding (`/100`), swap tables `_SWAP_LONG/_SHORT` in unspecified units ("per lot") applied once per calendar-date change, triple on Wednesday. Leverage is a flat constant; margin = `size*price/leverage` with no stop-out, no margin currency, no tick value. These are exactly the "never assume" items in our requirements (tick size/value, contract size, lot step, swap mode, stop/freeze levels).
5. **Position semantics are portfolio weights** (sum of |w| <= 1), rebalance bands, plan-rejection accounting — designed for multi-asset baskets, not a single-instrument scalper with discrete entries.
6. **Good engineering worth noting:** immutable `Position/FillRecord/TradeRecord`; fills ledger (`fills.jsonl`) including rejected-plan reasons (`no_bar`, `zero_size`, `execution_blocked`); warm-up vs evaluation window separation (`evaluation_start_index`); bounded forward-fill for valuation; negative/zero price guards; XAU-specific correction (gold is not a 0.0001-pip/100k instrument, with a comment that the generic assumption understated spread ~1000x) — evidence that the authors hit the same trap we are designing around, but their fix is another constant table.

**Verdict on our question "can the Forex engine save us rebuilding validated functionality?"** Not for the scalping core. The engine's validated behaviour is *bar-open fills with static costs on target weights*; our requirements are tick-driven, bid/ask-aware, SL/TP/time-barrier trades with broker-metadata sizing. Reusing it would force us to override fill model, costs, sizing, exits, margin and swap, leaving little. The parts worth keeping are **patterns** (immutable fill/trade records, rejection-reason accounting, warm-up separation, provenance), which are small.

### 1.2 MT5 loader (`agent/backtest/loaders/mt5_loader.py`)

- **OHLCV bars only** via `copy_rates_range` (intervals 1m ... 1M). **No tick download** (`copy_ticks_*` appears nowhere in the repository's non-test Python). Volume = `tick_volume`.
- Good: lazy import returning "unavailable" when package missing; tz-aware UTC request datetimes with an explicit comment that naive datetimes are shifted by the local timezone; symbol resolution that tries the exact token, normalised base, then `symbols_get(group="BASE*")` and **refuses ambiguous matches** instead of guessing; per-symbol failure isolation; terminal-path pinning warning.
- Gap: bar `time` is converted with `pd.to_datetime(frame["time"], unit="s")` and treated as UTC; **no server-time offset calibration or DST handling**. Given the time-semantics uncertainty in §4 this is exactly where silent session/feature errors enter.
- Annualisation tables hard-code `mt5: 260` days, which is a bar-return convention, not a scalper metric.

### 1.3 MT5 trading connector (`agent/src/trading/connectors/mt5/{_client,orders,reads,profiles,symbols}.py`)

Useful, verified patterns:
- **Demo/real guard on every session**: reads `account_info().trade_mode` and refuses a "paper" profile on a real account and vice versa; **contest accounts rejected** (fail-closed). The constants `ACCOUNT_TRADE_MODE_DEMO/CONTEST/REAL` match the installed package source (§4).
- Size guards that apply on demo too; notional-to-lots **floored to `volume_step`**, never rounded up; reject below `volume_min`; reject above `volume_max`.
- `order_check` before `order_send`; every non-`DONE` retcode becomes an error (fail-closed); partial (`DONE_PARTIAL`) handled.
- Filling-mode negotiation from the symbol's `filling_mode` bitmask; `getattr(mt5, "SYMBOL_FILLING_IOC", 2)` fallbacks. **We cannot verify these `SYMBOL_FILLING_*` constants in the installed package source** (they are not in `__init__.py`), so the numeric fallbacks are UNVERIFIED and must be tested on a demo terminal.
- Honest documentation of a hedging-account trap: an opposite-side order **opens a hedge** rather than closing; closing must pin `position=ticket`. Their own `flatten` hedges instead of closing (acknowledged follow-up). This is directly relevant to our EMERGENCY_FLATTEN design.
- Global `threading.Lock` around every MT5 call (the API is not documented as thread-safe).

Not suitable for us:
- `_session()` calls `mt5.initialize(login, password, server, timeout)` **and `mt5.shutdown()` on every operation**. Initialise/attach can take seconds; per-call re-login also re-sends credentials. Wrong for a latency-sensitive loop; we need one long-lived attached session owned by one thread.
- Order requests carry **no SL/TP** and are designed for an LLM-assisted manual/agentic flow (mandate, notional caps in USD), not model-driven scalping.
- USD-notional logic, mandate gate, daily order counts are product features of a different system.

### 1.4 Safety / audit (`agent/src/live/halt.py`, `agent/src/governance/ledger.py`)

- `halt.py`: an out-of-band **sentinel file** is the kill switch; its mere existence halts (malformed content still halts); checked before every broker call; works even if the agent is wedged. This matches our `HALT_NEW_TRADES` design well. Reference only (it is wired into their mandate/CLI).
- `ledger.py`: **hash-chained, fsynced, append-only JSONL ledger** with sequence numbers, `verify_chain`, export/verify, rotation, directory fsync, POSIX `flock` or Windows `msvcrt` byte-range lock. Standard-library only (imports: hashlib, json, logging, os, dataclasses, pathlib, typing). The Windows lock path exists but **we have not exercised it on Windows**. Strong ADAPT candidate for our order-lifecycle/risk-event audit trail.

### 1.5 Metrics (`agent/backtest/metrics.py`, `models.py`)

Verified contents: `win_rate_and_stats` (win rate, avg-win/avg-loss ratio, profit factor, max consecutive losses, avg holding bars), `calc_metrics` (total/annual return, max drawdown against a high-water mark that starts at initial cash, Sharpe, Sortino with all-period downside RMS denominator, Calmar, turnover, benchmark stats), `by_symbol_stats`, `by_exit_reason_stats`. Profit factor/ratio are `None` (not 0) when there are no losing trades — a good convention.

Missing vs our requirements: **expectancy**, per-trade R multiples, MFE/MAE, spread/slippage/commission/swap cost breakdown, segmentation by session/hour/weekday/regime/confidence, long/short split, calibration metrics. Sharpe/Sortino are **per-bar return** ratios annualised by `bars_per_year` assumptions (260 days for mt5), which is not a meaningful quantity for an irregular, event-driven tick scalper (we will report trade-level and daily-PnL based ratios). `+1e-10` epsilons and `fillna(0)` returns are conventions we would need to understand case by case.
Size of reusable part: ~100 lines of pure numpy/pandas. Saves little; **ADAPT the small, well-defined pieces with our own tests**, do not take the module.

### 1.6 Validation (`agent/backtest/validation.py`, `agent/src/quantlib/crossvalidation.py`, `multipletesting.py`)

- `validation.py` (do **not** treat as walk-forward validation):
  - "Monte Carlo permutation test" reshuffles trade PnL order and compares Sharpe/max-DD of paths. It tests **path ordering**, not whether the *entries* have edge versus random entries (our B1 null). Useful only for the order-reshuffle drawdown distribution that we already specified.
  - Bootstrap Sharpe CI resamples per-bar returns **i.i.d.** (ignores autocorrelation; we need a block bootstrap).
  - "Walk-forward analysis" **splits an already-produced equity curve into N windows** and reports per-window return/Sharpe/consistency. There is no retraining, no purge/embargo, no re-fit: it is a stability check, not walk-forward validation.
- `quantlib/crossvalidation.py` (634 lines; imports only numpy/pandas): `purged_kfold_splits`, `group_purged_kfold_splits`, `combinatorial_purged_splits`, `purged_walk_forward_splits`, plus **`detect_boundary_leakage`** that makes the off-by-one boundary leak assertable. Label spans are closed on both ends (conservative). Embargo default 1% of the sample, documented as a judgement. 86 upstream tests pass. **This is the most valuable component found for our protocol** (and exactly what FreqAI lacks, §3).
- `quantlib/multipletesting.py` (needs scipy): `sharpe_ratio`, `probabilistic_sharpe_ratio`, `expected_maximum_sharpe`, `deflated_sharpe_ratio`, `benjamini_hochberg`, `probability_of_backtest_overfitting` (CSCV). Directly supports our trial-registry/selection-bias requirement.

### 1.7 Other `quantlib/` modules

- `risk.py` (VaR/CVaR/drawdown distributions, EVT tail fit): portfolio-level return-series risk. Reference only; our risk engine is pre-trade, instrument-metadata-driven.
- `microstructure.py` (VPIN, Amihud, Kyle's lambda, Roll spread): VPIN, Kyle and Amihud need traded/signed volume, and **FX/CFD tick volume is an update count**, so they are not valid as written. Roll's estimator needs prices only, but we observe the actual quoted bid/ask spread, so it adds nothing. Rejected.
- `impact.py` (fixed-bps, linear and square-root market impact by ADV participation, `delayed_execution`): impact-by-ADV is irrelevant for retail lot sizes. Reference for the `delayed_execution` idea only.
- `constraints.py`, `portfolio/*`, `optimizers/*`: multi-asset weights and broker-portfolio reading. Rejected.
- Factor zoo (`agent/src/factors`, 477 files: alpha101, gtja191, qlib158, academic): daily cross-sectional equity factors. Rejected.

### 1.8 Agent / research architecture

- `agent/loop.py` (3.6k lines) ReAct loop; `agent/grounding/*` (~7k lines): run-scoped identity lock, evidence records, figure extraction and policies to reject numbers not supported by tool results; `swarm/*`: worker pool, presets (YAML "committees" e.g. factor-research, ML quant lab with a data-leakage reviewer), retry/backoff, task store; `hypotheses/registry.py`: small JSON registry with statuses `exploring/testing/validated/rejected/monitoring`; `governance/manifest.py`; `backtest/run_card.py`: run provenance with file/JSON hashes and metric citations; 60+ markdown "skills".
- All of this is heavily coupled to their loop, provider layer and tool registry. **`shadow_account/` is unrelated to our SHADOW mode** (it extracts a user's trading patterns from a broker journal and codegens strategies); the name collision must not mislead us.
- Position: the concepts (evidence-gated numbers, hypothesis registry, run cards, leakage-reviewer persona) are valuable; the code is not a drop-in. Best use is as an **external, read-only research sidecar** after Phase 6, never in the execution path (confirmed approved decision D-11).

---

## 2. Qlib — findings (source inspected)

| Area | What the source does | Useful for XAU/USD? |
|---|---|---|
| Dataset segmentation | `DatasetH(handler, segments={"train":(a,b),"valid":..,"test":..})`; `DataHandlerLP` keeps separate `_infer` and `_learn` frames produced by `infer_processors`/`learn_processors`; `prepare(segment, col_set, data_key)` | **Concept yes.** Raw -> processors -> segments with a train/infer split of processing. We implement with plain DataFrames + scikit-learn pipelines |
| Leakage prevention | Learnable processors (`MinMaxNorm`, `ZScoreNorm`, `RobustZScoreNorm`) take **explicit `fit_start_time/fit_end_time`**, with a code comment that setting them correctly "is very important"; the fit window is the caller's responsibility | **Concept yes**: fit scalers only on the training segment. Qlib does not enforce it; we enforce by fitting inside each fold |
| Rolling training | `RollingGen(step, rtype=expanding|sliding, trunc_days)` generates tasks; `trunc_segments` truncates train/valid segments before the test start "to avoid future information leakage" (`MultiHorizonGenBase` uses `horizon + label_leak_n`) | **Concept yes**, but coarse (day-based truncation); Vibe's `purged_walk_forward_splits` with real label spans is better |
| Model interface | `Model.fit(dataset, reweighter)`, `predict(dataset, segment)`, `ModelFT.finetune` | Concept; our `Predictor` is simpler and probability-vector based |
| Experiment management | `R`/Recorder on MLflow, `SignalRecord`, collectors, `OnlineManager` | **Reject** as a dependency (heavy). Keep the idea: persist model, config, dataset hash, metrics per run (our registry) |
| Backtest | `Exchange(open_cost=0.0015, close_cost=0.0025, min_cost=5.0, trade_unit=100, limit_threshold, deal_price, impact_cost)`; account/position at daily/minute bars | **Reject**: equity cost model (rate costs, board lots, price limits), no bid/ask or spread |
| High-frequency example | `examples/highfreq`: 1-minute CSI data, `HighFreqHandler` with price features normalised by prior close, IC/ICIR/long-short benchmarks; used as an RL order-execution dataset | **Reference only**: normalisation-by-reference-price idea, IC as a diagnostic. No tick data, no FX, no spread |
| Evaluation | IC, rank-IC, ICIR, long-short return | IC as a *supplementary* diagnostic for continuous scores; our primary metrics are net expectancy, PF, DD, calibration |
| Alpha158/360 | Equity daily/min factor sets | Reject |

Conclusion: **no Qlib code is needed.** Three ideas are adopted: train-only fit of processors (explicit), label-aware truncation in rolling splits (done better by Vibe's purged splitters), and experiment records tied to data/config hashes.

## 3. FreqAI — findings (reference only; GPL-3.0 verified)

| Area | What the source does | Decision / rationale |
|---|---|---|
| Orchestration | `IFreqaiModel.start_backtesting` (retrain per sliding window) / `start_live` (background training thread, predict with last model), `train`/`fit`/`predict` abstract methods | Concept: sliding-window train, predict-until-expiry. We implement `models/lifecycle` ourselves |
| Window config | `train_period_days`, `backtest_period_days`, `live_retrain_hours`, `expiration_hours`; `check_if_model_expired` returns True after `expiration_hours`; `check_if_new_training_required` | Concept: stale-model guard -> NO_TRADE (already in our design) |
| Splitting | `make_train_test_datasets` -> scikit-learn `train_test_split(**data_split_parameters)`, default `shuffle=False`, optional `shuffle_after_split`. **A search for `label_period`, `purge`, `embargo` finds only `purge_old_models`**: no purging of labels that overlap the train/test boundary | **Important negative finding.** Their in-window split can leak through overlapping label horizons. Do not copy this scheme; our purged/embargoed splits are required |
| Feature pipeline | `define_data_pipeline` builds a datasieve `Pipeline` (normalise, optional SVM outlier removal, `DissimilarityIndex(di_threshold)`, `DBSCAN`); fit on train, `transform(outlier_check=True)` at predict; outlier -> `do_predict=0` | Concept: novelty/outlier gate at inference that forces "no prediction". We implement our own (scikit-learn) and route to NO_TRADE |
| Weights | `weight_factor` exponential recency weights | Concept; test whether recency weighting helps (E-series) |
| Persistence | `FreqaiDataDrawer`: models saved per `identifier`/pair/timestamp (`model_save_type` default `joblib`), historic predictions via `cloudpickle`, JSON metadata, `purge_old_models`; metric tracker | Concept; **deserialising pickled/joblib artefacts automatically is a supply-chain/integrity risk**: we verify artefact hashes from the registry before load and promote only through the human-approved gate. FreqAI deploys the newest trained model automatically, which our policy forbids |
| Structure | `FreqaiDataKitchen` (~1,000 lines) mixes dataframe slicing, feature building, timerange math, prediction storage | Do not imitate; keep our modules small and single-purpose |
| Model zoo | LightGBM/XGBoost/RF classifiers & regressors, PyTorch MLP/Transformer, RL learner | We wrap the underlying libraries directly; RL rejected |
| Domain | Pair lists, candle (not tick) dataframes, strategy `populate_*` hooks, crypto exchanges | Not applicable to MT5/tick |

Conclusion: FreqAI contributes **architecture ideas only**, and its splitting is weaker than what we need.

---

## 4. MT5 verification table

Evidence classes. **V-src** = verified in the actual installed package source/metadata (wheel `MetaTrader5 5.0.6231`, `__init__.py`, `METADATA`, `WHEEL`). **V-bin** = strings present in the compiled `_core` binary (proves a name/argument exists, not its type or semantics). **S** = secondary evidence only (web-search summaries of official docs/forums, upstream code comments). **UNVERIFIED** items must not become hard architectural dependencies; each has a Phase 1 test that would verify it against a real demo terminal.

| # | API / function | Purpose | Input | Output | Platform req. | Status | Source | Implementation implication |
|---|---|---|---|---|---|---|---|---|
| 1 | Package platform | Install target | - | - | **Windows x86-64 only**; needs a running MT5 terminal; `numpy>=1.7` | **Verified (V-src)** | `METADATA` (`Platform: Windows`, `Requires-Dist: numpy>=1.7`), wheel tag `win_amd64`; PyPI page | MT5 code only on Windows; all other layers testable on Linux with a fake adapter. Confirms D-1 (Windows dev machine; VPS later) |
| 2 | Python versions | Compatibility | - | - | 3.6-3.14 (classifiers) | Verified (V-src) | `METADATA` | Pin our Python; no constraint |
| 3 | Licence | Compliance | - | - | - | Verified (V-src) | `LICENSE.txt` (MIT, MetaQuotes 2000-2025) | Package is permissive; terminal/broker EULAs separate (record in licensing doc) |
| 4 | `initialize([path],[server],[login],[password])` | Attach to terminal | optional path/server/login/password; `timeout`, `portable` keyword args | bool | Windows + terminal | Signature verified (V-bin docstring); `timeout`/`portable` args present (V-bin strings `Invalid "timeout" argument`, `portable`); whether it launches the terminal: **UNVERIFIED** | `_core` strings; upstream usage | Attach once per process and keep the session; never re-initialise per call (contrast Vibe connector) |
| 5 | `login(login,[server],[password])`, `shutdown()`, `version()`, `last_error()` | Session control | per signature | bool / tuple | same | Signatures verified (V-bin); return types UNVERIFIED | `_core` docstrings | Wrap in adapter; on failure read `last_error()`; error constants below |
| 6 | `RES_*` last_error codes | Error classification | - | ints | - | Verified (V-src) | `__init__.py`: `RES_S_OK=1`, `RES_E_FAIL=-1`, `INVALID_PARAMS=-2`, `NOT_FOUND=-4`, `AUTH_FAILED=-6`, `UNSUPPORTED=-7`, `AUTO_TRADING_DISABLED=-8`, `INTERNAL_FAIL_*` | Map to our neutral error enum |
| 7 | `terminal_info()` | Terminal state | - | struct (`TerminalInfo`) | same | Exists (V-bin: `TerminalInfo` struct); field names `connected, trade_allowed, trade_expert, ping_last, community_account, dlls_allowed, build, company, language, path, data_path, commondata_path` present as strings (**which struct owns which field: UNVERIFIED**) | `_core` strings | Use for health check (`connected`, `trade_allowed`, `ping_last`) after confirming fields on a demo terminal |
| 8 | `account_info()` | Account state | - | `AccountInfo` struct | same | Exists (V-bin); fields `balance, equity, margin, margin_free, margin_level, leverage, profit, margin_so_call/so_so, margin_mode, trade_mode, currency, login, server` (string presence only; types UNVERIFIED) | `_core` strings | Basis for `AccountInfo`; the demo/real guard uses `trade_mode` |
| 9 | `ACCOUNT_TRADE_MODE_DEMO/CONTEST/REAL` | Demo vs real discriminator | - | 0 / 1 / 2 | - | **Verified (V-src)** | `__init__.py` | Hard guard: DEMO mode requires `trade_mode == DEMO`; contest and real rejected |
| 10 | `ACCOUNT_MARGIN_MODE_RETAIL_NETTING/EXCHANGE/RETAIL_HEDGING` | Position semantics | - | 0 / 1 / 2 | - | Verified (V-src) | `__init__.py` | Adapter must read margin mode at connect; **hedging vs netting changes close semantics** (see #20) |
| 11 | `symbols_total()`, `symbols_get([group])`, `symbol_select(symbol,[enable])` | Symbol discovery/visibility | group wildcard, symbol, bool | count / tuple of `SymbolInfo` / bool | same | Signatures verified (V-bin) | `_core` docstrings | Symbol resolution from config candidates + wildcard search; refuse ambiguity (pattern from Vibe loader); select symbol before requesting data |
| 12 | `symbol_info(symbol)` | Instrument metadata | symbol | `SymbolInfo` or None | same | Signature verified; fields **present by name (V-bin)**: `digits, point, spread, visible, select, trade_contract_size, trade_tick_size, trade_tick_value, trade_tick_value_profit, trade_tick_value_loss, volume_min, volume_max, volume_step, trade_stops_level, trade_freeze_level, trade_exemode, trade_mode, filling_mode, swap_mode, swap_long, swap_short, swap_rollover3days, currency_base, currency_profit, currency_margin, session_deals`; numeric types/units **UNVERIFIED** | `_core` strings | Source of sizing/cost inputs; Phase 1 dumps real values for XAUUSD/EURUSD/GBPUSD and asserts self-consistency (e.g. tick_value vs contract_size*tick_size) |
| 13 | `symbol_info_tick(symbol)` | Latest tick | symbol | `Tick` struct or None | same | Signature verified (V-bin); field names `time, bid, ask, last, volume, time_msc, flags, volume_real` (string presence) | `_core` strings | Latest-quote polling for health only; not an event stream |
| 14 | `copy_ticks_from(symbol, date_from, count, flags)`, `copy_ticks_range(symbol, date_from, date_to, flags)` | Tick history | symbol, datetime or epoch, count / end, flags | numpy structured array | same | Signatures verified (V-bin). Flags `COPY_TICKS_ALL=-1, INFO=1, TRADE=2` and tick flags `BID=0x02 ASK=0x04 LAST=0x08 VOLUME=0x10 BUY=0x20 SELL=0x40` **verified (V-src)**. Array dtype/fields: UNVERIFIED (names present) | `__init__.py`, `_core` | Phase 1 historical download uses `copy_ticks_range` in day chunks. Max rows per call and available depth: **UNVERIFIED** (measure) |
| 15 | **Time semantics of returned/accepted times** | Correct UTC handling | - | - | - | **UNVERIFIED / conflicting (S)**. Official docs (via search summary) say the terminal stores tick/bar times in UTC and naive `datetime` is shifted by the local timezone, so pass tz-aware UTC. Multiple forum threads (incl. one on MetaQuotes-Demo) report returned timestamps ~3 h ahead of true UTC, i.e. **server time encoded as epoch**. Vibe's loader comment agrees naive datetimes are mis-shifted but never calibrates | web search (mql5 forum threads, doc summary); `mt5_loader.py` | **Do not assume either.** Adapter must estimate the offset empirically per broker (latest tick time vs system UTC while market is open, repeated over several days incl. a DST change), store it with provenance (`DatasetProvenance.time_basis`), and treat `unverified` data as non-trainable. Not a hard dependency until measured |
| 16 | `copy_rates_from/from_pos/range` | Bars | symbol, timeframe, dates/pos/count | numpy structured array (`time, open, high, low, close, tick_volume, spread, real_volume`) | same | Signatures verified (V-bin); `TIMEFRAME_*` constants verified (V-src); field names present; history depth bound by terminal setting (claim from Vibe docstring) UNVERIFIED | `__init__.py`, `_core`, `mt5_loader.py` | Bars used only for cross-checks; our bars are built from ticks |
| 17 | `market_book_add/get/release(symbol)`, `BOOK_TYPE_*` | Depth of market | symbol | list of book entries | same | Functions + constants verified (V-src/V-bin). **Availability and quality per broker/symbol: UNVERIFIED** | `__init__.py`, `_core` | Optional experimental feature; no architecture depends on it |
| 18 | `order_calc_margin(action,symbol,volume,price)`, `order_calc_profit(action,symbol,volume,price_open,price_close)` | Margin/profit estimates | per signature | float or None | same | Signatures verified (V-bin) | `_core` docstrings | Use as cross-check of our own money maths; not as the only source |
| 19 | `order_check(request)` / `order_send(request)` | Pre-flight / submit | request dict (keys per docs: `action, symbol, volume, type, price, stoplimit, sl, tp, deviation, magic, comment, type_filling, type_time, expiration, position, position_by, order`) | `OrderCheckResult` / `OrderSendResult` (`retcode, deal, order, volume, price, bid, ask, comment, request_id, retcode_external`) | same | Request key names: present in binary (V-bin, incl. `sl`, `tp`) and listed by docs (S). Result struct names verified (V-bin). Semantics of each field: UNVERIFIED until tested | `_core`; search summary of docs | Always `order_check` first; map every `TRADE_RETCODE_*` (verified list in V-src: DONE=10009, REQUOTE=10004, REJECT=10006, INVALID_STOPS=10016, NO_MONEY=10019, PRICE_OFF=10021, FROZEN=10029, INVALID_FILL=10030, ...) to a neutral enum |
| 20 | Closing a position | Flatten | `TRADE_ACTION_DEAL` with opposite `type` and **`position=<ticket>`**; or `TRADE_ACTION_CLOSE_BY` | result | same | Constants verified (V-src). "Opposite order without `position` opens a hedge on hedging accounts": **S** (Vibe's docstring); must be confirmed on a demo hedging account | `__init__.py`; `orders.py` | EMERGENCY_FLATTEN always closes by position ticket (never by sending an opposite market order blindly) |
| 21 | `ORDER_FILLING_FOK/IOC/RETURN/BOC`, `ORDER_TYPE_*`, `TRADE_ACTION_*`, `DEAL_*`, `POSITION_*` | Enumerations | - | ints | - | Verified (V-src) | `__init__.py` | Reference constants only via our enum mapping inside `brokers/mt5` |
| 22 | `SYMBOL_FILLING_FOK/IOC` (bitmask values) and `filling_mode` bit meanings | Choose a valid filling mode | - | - | - | **UNVERIFIED**: constants absent from `__init__.py`; Vibe uses fallbacks `IOC=2`, `FOK=1` | `orders.py` | Phase 1 test on demo: for each symbol, `order_check` each candidate filling mode and record which pass. Do not hardcode |
| 23 | `positions_total/get`, `orders_total/get`, `history_orders_total/get`, `history_deals_total/get` | State, reconciliation, execution quality | symbol/ticket/group/date range | structs | same | Signatures verified (V-bin) | `_core` docstrings | Reconciliation after restart; commission/swap/fill price from deal history |
| 24 | Bundled helpers `Buy()`, `Sell()`, `Close()` | Convenience | - | - | - | **Verified (V-src) and rejected**: fixed `deviation=10`, up to 10 blind retries on `REQUOTE/PRICE_OFF`, `Close` loops over positions | `__init__.py` | Never use; they hide retry behaviour and fixed slippage that execution must control and record |
| 25 | Thread-safety / concurrency | Process design | - | - | - | **UNVERIFIED** (undocumented in available material) | - | One owner thread/process for all MT5 calls + a lock; never call from multiple threads |
| 26 | Call latency, tick rate, `copy_ticks` limits, history depth | Feasibility, latency budget | - | - | - | **UNVERIFIED** | - | Measure in Phase 1 (latency probe); budgets set only from measurements |
| 27 | Broker server timezone/DST rules | Session engine | - | - | - | **UNVERIFIED** (per broker) | - | Derive from data (see #15); unit-test across DST changes |
| 28 | Whether `initialize()` can start the terminal / `portable` semantics | Ops | - | - | - | UNVERIFIED | - | Start the terminal manually under the operator's control in Phase 1 |

**Net MT5 position:** function names, signatures, constants, struct names, platform and licence are verified against the package itself. **Time semantics, filling-mode bits, hedge-close behaviour, struct field types, thread-safety, limits and latency are unverified** and are turned into explicit Phase 1 verification tests (§8) rather than assumptions.

---

## 5. Component decision matrix

Decisions: **REUSE** (take as a dependency/unchanged), **ADAPT** (copy/modify with attribution after review), **WRAP** (use behind our interface), **REFERENCE_ONLY**, **REJECT**.
Licence column: MIT = copy permitted with the copyright/permission notice retained; GPL = no code may be copied.

### 5.1 Vibe-Trading (MIT)

| Source project | File / module | Purpose | Decision | Licence consideration | Performance consideration | Forex suitability | XAUUSD suitability | Rationale |
|---|---|---|---|---|---|---|---|---|
| Vibe-Trading | `backtest/engines/forex.py` + `base.py` | Bar-based FX/CFD backtest on target weights | **REJECT** (as engine); patterns **REFERENCE_ONLY** | MIT; no obstacle, but nothing to take | Vectorised numpy, fast on bars; irrelevant for tick events | Low: static spreads, no bid/ask, no SL/TP, constants not metadata | Low-medium: has gold specs (0.10 pip, 100 oz) as constants, which our own metadata makes unnecessary | Our core needs tick-driven, bid/ask, barrier exits, broker-metadata sizing; overriding all would leave nothing |
| Vibe-Trading | `engines/_market_hooks.py` (`calc_forex_swap`, `_SWAP_*`, symbol normalisation) | Static swap tables, symbol classes | **REJECT** | MIT | trivial | Low: swap in unspecified units, once per date change | Low | We use broker `swap_long/short/mode` and rollover timing |
| Vibe-Trading | `engines/base.py` (`FillRecord/TradeRecord/Position`, plan-rejection reasons, warm-up/evaluation split) | Immutable evidence records | **REFERENCE_ONLY** | MIT | n/a | Medium | Medium | Same ideas implemented with our fields (bid, ask, latency, commission, slippage, MFE/MAE) |
| Vibe-Trading | `backtest/loaders/mt5_loader.py` | MT5 bar loader | **REFERENCE_ONLY** | MIT | per-call bars; no ticks | Medium: symbol-ambiguity refusal, tz-aware requests | Medium | Bars only, no offset calibration; we need ticks and calibrated time. Re-implement symbol resolution with our config |
| Vibe-Trading | `trading/connectors/mt5/_client.py` | Session, profile guard, lock | **REFERENCE_ONLY** | MIT | initialise+shutdown per call: slow | High as a pattern (demo/real guard) | same | Take the guard and lock ideas; reject per-call session model |
| Vibe-Trading | `trading/connectors/mt5/orders.py` | Order build, size guards, filling negotiation | **REFERENCE_ONLY** | MIT | n/a | High as checklist | same | No SL/TP; filling constants unverified; hedging-close caveat documented |
| Vibe-Trading | `backtest/metrics.py` (`win_rate_and_stats`, drawdown from initial-cash HWM, Sortino convention) | Performance metrics | **ADAPT** (small parts) after our tests | MIT; keep notice | trivial | Medium: per-bar Sharpe conventions unsuitable | Medium | Reimplement/adapt trade statistics + drawdown; add expectancy, R, MFE/MAE, costs, segmentation. Do not import the module |
| Vibe-Trading | `backtest/validation.py` (MC order shuffle) | Path-order Monte Carlo | **ADAPT** (reshuffle drawdown distribution only) | MIT | O(n_sims * n_trades) fine | n/a | n/a | Not a test of edge; label accordingly |
| Vibe-Trading | `backtest/validation.py` (bootstrap, "walk-forward") | i.i.d. bootstrap; equity-curve windows | **REJECT** | MIT | n/a | n/a | n/a | i.i.d. bootstrap and non-retraining "walk-forward" mislead for autocorrelated trade series; use block bootstrap + purged walk-forward |
| Vibe-Trading | `src/quantlib/crossvalidation.py` | Purged/embargoed K-fold, group, CPCV, purged walk-forward, leakage detector | **ADAPT** (vendor with attribution + our tests) | MIT; pin upstream commit `7f6908b`, retain notice in a `THIRD_PARTY_NOTICES` entry | Pure numpy/pandas; negligible | High: label-span based, instrument agnostic | High | Directly implements our required purge/embargo; 86 upstream tests pass. Must be re-validated on our event-label spans (irregular spacing) in Phase 4 |
| Vibe-Trading | `src/quantlib/multipletesting.py` | PSR, DSR, PBO (CSCV), BH | **ADAPT** (vendor with attribution + our tests) | MIT; needs scipy (BSD) | negligible | agnostic | agnostic | Satisfies selection-bias/trial-registry requirement; verify formulas against the literature before relying on them |
| Vibe-Trading | `src/governance/ledger.py` | Hash-chained, fsynced, append-only audit ledger | **ADAPT** (vendor with attribution + Windows tests) | MIT | fsync per record: ms-level, acceptable for order events, not for ticks | agnostic | agnostic | Tamper-evident order/risk-event journal; Windows lock path untested by us |
| Vibe-Trading | `src/live/halt.py` | Sentinel-file kill switch | **REFERENCE_ONLY** | MIT | trivial | agnostic | agnostic | Matches our HALT_NEW_TRADES design; we implement ours |
| Vibe-Trading | `src/live/runtime/flatten.py`, mandate gate (`sdk_order_gate.py`, `live/mandate/*`) | Flatten and notional mandates | **REFERENCE_ONLY** | MIT | n/a | Low | Low | Product-specific; their flatten hedges on MT5 (documented) |
| Vibe-Trading | `quantlib/risk.py`, `portfolio.py`, `constraints.py` | Portfolio VaR/weights | **REJECT** | MIT | n/a | Low | Low | Not pre-trade scalping risk |
| Vibe-Trading | `quantlib/microstructure.py` (VPIN, Kyle, Amihud, Roll) | Liquidity/flow measures | **REJECT** (revisit with trade-volume data) | MIT | n/a | Low: FX tick volume is not traded volume; Roll is redundant with the quoted spread | Low | Inputs invalid or redundant for MT5 FX/CFD |
| Vibe-Trading | `quantlib/impact.py` | Slippage/impact models | **REJECT** (idea: delayed execution) | MIT | n/a | Low | Low | ADV-participation impact irrelevant at retail size |
| Vibe-Trading | `backtest/run_card.py` | Run provenance + hashes | **REFERENCE_ONLY** | MIT | n/a | agnostic | agnostic | Our manifest/registry records equivalent facts |
| Vibe-Trading | `src/hypotheses/registry.py` | Hypothesis status registry | **REFERENCE_ONLY** | MIT | n/a | agnostic | agnostic | Idea for our trial registry statuses |
| Vibe-Trading | `agent/loop.py`, `agent/grounding/*`, `swarm/*`, skills, MCP server | LLM research agent | **WRAP** (optional external sidecar, Phase 10/11, read-only) | MIT; separate process, no code in our tree | LLM latency: never in execution path | agnostic | agnostic | Approved D-11; sidecar behind a read-only interface to journal/backtest artefacts |
| Vibe-Trading | `shadow_account/*` | Pattern extraction from a user's journal | **REJECT** | MIT | n/a | n/a | n/a | Unrelated to our SHADOW mode |
| Vibe-Trading | `agent/src/factors/*` (477 files) | Equity factor zoo | **REJECT** | MIT, but NOTICE cites Apache-2.0 Qlib definitions: avoid inheriting | large | Low | Low | Daily cross-sectional equity factors |
| Vibe-Trading | other engines (china_a, crypto, options, futures, equity) and the ~25 non-MT5 loaders | Other markets | **REJECT** | MIT | n/a | n/a | n/a | Not our markets |

### 5.2 Qlib (MIT)

| Source project | File / module | Purpose | Decision | Licence consideration | Performance consideration | Forex suitability | XAUUSD suitability | Rationale |
|---|---|---|---|---|---|---|---|---|
| Qlib | `data/dataset/{__init__,handler,processor}.py` | Dataset segments, infer/learn processors, fit windows | **REFERENCE_ONLY** | MIT | pandas-heavy; MultiIndex | Medium (concept) | Medium | Same discipline via scikit-learn pipelines fitted per fold |
| Qlib | `workflow/task/gen.py` (`RollingGen`, `trunc_segments`) | Rolling task generation | **REFERENCE_ONLY** | MIT | n/a | Medium | Medium | Day-based truncation is coarser than label-span purging |
| Qlib | `contrib/rolling/base.py` | Offline rolling experiments | **REFERENCE_ONLY** | MIT | MLflow-based | Low | Low | Heavy tooling for little benefit |
| Qlib | `model/base.py` | `fit/predict(dataset,segment)` interface | **REFERENCE_ONLY** | MIT | n/a | Medium | Medium | Our `Predictor` returns calibrated probability vectors |
| Qlib | `workflow/{recorder,record_temp,online}` | Experiment recording, online models | **REJECT** (idea kept) | MIT | MLflow dependency | Low | Low | Our registry stores equivalent facts |
| Qlib | `backtest/exchange.py`, `account.py`, `position.py` | Equity-style execution/accounting | **REJECT** | MIT | n/a | Low | Low | Rate costs, board lots, limit prices; no spread |
| Qlib | `examples/highfreq/*` | 1-min equity high-frequency data/handler | **REFERENCE_ONLY** | MIT | n/a | Low | Low | Price normalisation idea; no ticks/FX |
| Qlib | `contrib/data/handler.py` (Alpha158/360) | Equity factor sets | **REJECT** | MIT (Vibe's NOTICE labels derivatives Apache-2.0: unresolved discrepancy) | n/a | Low | Low | Not relevant |
| Qlib | `qlib/rl/*` | RL order execution | **REJECT** | MIT | heavy | Low | Low | Out of scope |

### 5.3 Freqtrade / FreqAI (GPL-3.0; **no code to be copied**)

| Source project | File / module | Purpose | Decision | Licence consideration | Performance consideration | Forex suitability | XAUUSD suitability | Rationale |
|---|---|---|---|---|---|---|---|---|
| FreqAI | `freqai_interface.py` (start_backtesting/start_live, expiry) | Sliding-window retrain & predict-until-expiry | **REFERENCE_ONLY** | GPL-3.0: ideas only, clean-room | training in separate thread/process | Medium (concept) | Medium | Our lifecycle adds champion/challenger + human-approved promotion |
| FreqAI | `data_kitchen.py` (`make_train_test_datasets`, timeranges) | Splitting/window math | **REJECT** | GPL-3.0 | n/a | Low | Low | No purge/embargo; shuffle options risk leakage |
| FreqAI | `freqai_interface.py` pipeline (SVM, DI, DBSCAN outlier steps) | Inference novelty gate | **REFERENCE_ONLY** | GPL-3.0 | O(n*d) distance computations | Medium | Medium | We implement our own gate; failure -> NO_TRADE |
| FreqAI | `data_drawer.py` | Model/prediction persistence, purge old models | **REFERENCE_ONLY** | GPL-3.0 | pickle I/O | Low | Low | We use hashed, versioned artefacts with human promotion |
| FreqAI | `prediction_models/*`, `base_models/*` | Model wrappers | **REJECT** | GPL-3.0 | n/a | n/a | n/a | We wrap LightGBM/XGBoost/scikit-learn directly |
| FreqAI | `RL/*`, `torch/*` | RL, PyTorch wrappers | **REJECT** | GPL-3.0 | heavy | Low | Low | Out of scope |
| Freqtrade | core bot, backtesting, hyperopt, exchange layer | Crypto trading framework | **REJECT** | GPL-3.0 | n/a | Low | Low | Crypto candles/exchanges; hyperopt invites overfitting |

### 5.4 MetaTrader5 package (MIT wrapper; proprietary terminal)

| Source project | File / module | Purpose | Decision | Licence consideration | Performance consideration | Forex suitability | XAUUSD suitability | Rationale |
|---|---|---|---|---|---|---|---|---|
| MetaTrader5 | `MetaTrader5` (functions in §4) | Terminal access | **WRAP** (`brokers/mt5` only) | MIT wrapper; terminal EULA | synchronous IPC; latency to be measured | High | High | Only practical route; isolate behind `BrokerAdapter` |
| MetaTrader5 | `Buy/Sell/Close` helpers in `__init__.py` | Convenience | **REJECT** | - | blind retries | - | - | Hidden retries and fixed deviation |

---

## 6. Licensing record (D-9: private/proprietary)

- This repository is treated as private/proprietary. Using MIT code is compatible; obligations: keep the copyright and permission notice with any copied code (`THIRD_PARTY_NOTICES.md`).
- **No GPL code may enter the repository** (Freqtrade/FreqAI). Contributor rule: do not paste from Freqtrade.
- Planned vendored (ADAPT) files, each to carry a header with origin URL, commit, licence text reference: `crossvalidation.py`, `multipletesting.py`, `governance/ledger.py`, small parts of `metrics.py`. These are not copied in Phase 0.5.
- Discrepancy to resolve before copying anything beyond those files: Vibe's `NOTICE` says it bundles Qlib feature definitions under **Apache 2.0**, while Qlib's `LICENSE` is MIT. Irrelevant while we avoid the factor zoo; flagged.
- MetaTrader 5 terminal and broker terms (automated trading, scalping restrictions) are separate from the Python package licence; the operator must read the broker's terms. Not legal advice.

## 7. Impact on the architecture (summary; details in `SYSTEM_ARCHITECTURE.md`)

1. **Build our own tick/bid-ask backtest and metrics core** (confirmed). The Vibe Forex engine does not shortcut it.
2. **Adopt three small, tested vendored components** after review: purged CV, multiple-testing statistics, hash-chained audit ledger.
3. **MT5 adapter rewritten for a persistent session**, with demo/real guard, ticket-pinned closes, `order_check` first, empirically calibrated server time, and verification tests for every UNVERIFIED item.
4. **Same-machine simplicity (D-4):** no message broker; in-process queues behind interfaces.
5. **Two safety concepts (D-7):** HALT_NEW_TRADES (auto, cheap) and EMERGENCY_FLATTEN (authorised only).
6. **Validation thresholds come from baseline data (D-6)**, not from numbers chosen here. Earlier "suggested" thresholds were removed from the docs.

## 8. Newly discovered risks

| ID | Risk | Mitigation |
|---|---|---|
| R-1 | MT5 timestamp semantics unresolved (UTC vs server time) can silently shift sessions, news windows and features | Mandatory empirical calibration + provenance flag; tests across DST |
| R-2 | Hedging account: an opposite order opens a hedge, doubling exposure while appearing to "close" | Close by position ticket; detect margin mode; test on demo hedging and netting accounts |
| R-3 | Filling-mode bitmask constants not in the installed package | Probe with `order_check` per symbol, record result, no hardcoded numbers |
| R-4 | Upstream "walk-forward"/Monte Carlo utilities could be mistaken for real validation | Documented limits (§1.6); we use purged walk-forward + trial-count-aware statistics |
| R-5 | FreqAI-style splits without purging leak via label overlap | Never adopt; leakage detector in our CI |
| R-6 | Static spread/swap tables in upstream FX engine could be copied into ours | Policy: all cost inputs come from measured broker data |
| R-7 | Per-call `initialize/shutdown` (Vibe connector) would add seconds of latency and resend credentials | Persistent session design |
| R-8 | Vendored code correctness not guaranteed by upstream tests (they test the authors' spec) | Our own property tests (e.g. no train sample's label span overlaps a test span; DSR against known literature values) before use |
| R-9 | The ledger's Windows lock path is untested by us; fsync cost | Windows test run on the dev machine in Phase 1; ledger only for order/risk events |
| R-10 | Dependency creep (scipy for multipletesting) | Limit to scipy + numpy + pandas; decision recorded |
| R-11 | Documentation host blocked in this sandbox; MT5 docs verified only secondarily | Phase 1 first task re-verifies against official docs from the Windows machine |

## 9. Revised Phase 1 (high level; exact steps in `IMPLEMENTATION_PLAN.md`)

Verification-first: build the read-only MT5 adapter and a **MT5 verification harness** (`scripts/mt5_verify.py`, runs only on the Windows machine against a DEMO account) that tests every UNVERIFIED row of §4 and writes a signed-off report before any data is trusted. No orders are sent in Phase 1 (`order_check` is allowed only after review and never `order_send`).
