# Phase 1 Report: Read-only MT5 Data Foundation

**Status: code complete and fully tested on Linux against a fake. NOT verified against a real MetaTrader 5 terminal.**
**Real Windows/MT5 verification has not occurred.** Every statement below about real MT5 behaviour is either
"verified from the package source/metadata in Phase 0.5" or "UNVERIFIED until the Windows procedure is run". The
completion gate in the Phase 1 brief therefore cannot be declared met yet; see §17.

No trading, no order submission, no model, no strategy, no backtest and no profit/win-rate claim exists anywhere in this phase.
No Vibe-Trading, Qlib or FreqAI code was copied (the vendor/reuse decision is deferred to the validation subsystem).

## 1. Files created / modified
**Created (source)**: `brokers/{errors,accounts,symbols}.py`; `brokers/mt5/{__init__,api,owner,convert,adapter,fake}.py`;
`market_data/{timebase,schema,quality,store,acquire,calibrate,profile}.py`; `sessions/calendar.py`; `core/{provenance,env}.py`; `workflows.py`.
**Rewritten**: `brokers/base.py` (read-only interface + guarded trading interface, raw tick dtype), `market_data/provider.py` (`HistoricalDataProvider` + `MT5HistoricalDataProvider`).
**Created (scripts)**: `scripts/{verify_mt5,acquire_ticks,profile_ticks,benchmark_io}.py` (`python -m scripts.<name>`).
**Created (docs)**: `MT5_ADAPTER_AND_DATA_FOUNDATION.md`, `WINDOWS_MT5_VERIFICATION.md`, `XAUUSD_DATA_PROFILE.md` (placeholder, no data), `PHASE1_REPORT.md`, `baselines/phase1_synthetic_io.json`.
**Modified**: instrument configs (canonical `XAU_USD`/`EUR_USD`/`GBP_USD`, `aliases`, expected currencies), `core/config.py`, `monitoring/logging_setup.py` (redacts `login=`), `pyproject.toml` (numpy/pandas/pyarrow/tzdata deps, `pythonpath`), `.env.example`, `.gitignore` (`reports/`), `README.md`, `IMPLEMENTATION_PLAN.md`.

## 2. Tests added
162 new tests (186 total): `test_accounts_and_adapter` (48), `test_convert_and_symbols` (23), `test_timebase` (26), `test_quality` (15),
`test_store_and_acquire` (23), `test_sessions_profile_harness` (27). Updated: `test_configs`.

## 3. Test results
`python -m pytest -q`: **186 passed** (Python 3.13.16, Linux x86_64, numpy 2.5.3, pandas 3.0.5, pyarrow 25.0.1). `ruff check --select F,E9` clean.
The suite needs no MT5 installation. The CI workflow (`.github/workflows/ci.yml`) has still not been run on GitHub.

## 4. MT5 APIs actually verified
**Against a real terminal: none.** Verified in Phase 0.5 from the unpacked 5.0.6231 package (not executed): function names/signatures, constants
(including `ACCOUNT_TRADE_MODE_*`, `COPY_TICKS_*`, `TIMEFRAME_*`, `RES_*`), struct names, platform. Phase 1 exercises *our code* for these
functions against the fake: `initialize`, `shutdown`, `version`, `last_error`, `terminal_info`, `account_info`, `symbols_get`, `symbol_info`,
`symbol_info_tick`, `symbol_select`, `copy_ticks_range`, `copy_ticks_from`, `copy_rates_range`, `market_book_add/get/release`. That proves our
handling, not the terminal's behaviour.

## 5. APIs / behaviours still unverified (must be settled by the Windows run)
Time semantics (UTC vs server time, DST); struct field types/units; the dtype and field order of tick/rate arrays; `copy_ticks_range` `date_to`
inclusivity and per-call row cap; whether `copy_ticks_*` returns `None` or an empty array when there is no data; behaviour of `datetime` arguments;
history depth; `symbol_select` side effects; DOM availability; thread-safety; whether `initialize` can start the terminal; call latency; tick-flag
semantics; `SYMBOL_FILLING_*` bits and hedge-close behaviour (these need order functions and stay out of scope). `login` is deliberately not exposed.

## 6. XAU/USD symbol resolution
Canonical `XAU_USD` -> broker symbol by tiered, deterministic discovery over aliases `XAUUSD, XAU/USD, XAU_USD, GOLD` with recognised prefixes/suffixes
(`XAUUSDm`, `XAUUSD.a`, `mXAUUSD`, `GOLDm` ... covered by tests); ties are `ambiguous`, unrecognised affixes need review, a currency mismatch
(base/profit) is rejected, an explicit `pin` resolves ambiguity. The mapping is persisted in `configs/broker_symbols/<broker>__<server>.yaml`;
application code resolves only through `SymbolMap`. EUR_USD and GBP_USD use the same mechanism (discovery/configuration only; no model work).

## 7. Instrument metadata schema
`symbol_info/1` JSON: `raw` (all returned fields, unconverted), `typed` (digits, point, tick size/value/profit/loss, contract size, volume min/max/step/limit,
stops/freeze level, spread, spread_float, trade/exec/calc mode, filling/order/expiration mode, swap fields, currencies, path, description, book depth),
`consistency_issues`. Content-addressed; units are checked (not assumed) by `check_symbol_consistency` (tested).

## 8. Time-semantics findings
**No empirical finding exists yet** (no MT5 available). What was built: offset estimation from latest ticks vs the local UTC clock with residual/MAD
quality gates; stale-feed and clock-skew detection; optional DST inference from weekly opens (assumption-labelled: open anchored to 17:00 New York);
explicit `time_basis` (`utc_verified | server_fixed_offset | server_dst_rule | unverified`); DST fold/gap handling that flags rather than guesses.
Validated against the fake with known ground truth (fixed offsets +3/0/-5/+5.5 h; New-York-anchored rule). The documentation-vs-community conflict from
Phase 0.5 remains open: the Windows run will settle it for the operator's broker.

## 9. Tick-data schema
`tick_raw/1`: `seq, source_time, source_time_msc, normalized_utc_time, ingestion_time_utc, time_basis, bid, ask, last, volume, volume_real, flags`
(raw values never overwritten). Separate datasets `tick_derived/1` (`mid, spread, spread_points, relative_spread`) and `tick_quality/1` (`quality_flags, quarantined`).

## 10. Data-quality rules
Negative/zero spread, missing/non-positive/NaN bid or ask, extreme spread (k x batch median), duplicate timestamp, duplicate tick, timestamp reversal,
large gap, second/millisecond mismatch, unknown flag bits, ambiguous/non-existent time. Tag, never delete; severe tags quarantine. Thresholds are
recorded heuristics to be re-derived from the profile (§MT5_ADAPTER_AND_DATA_FOUNDATION 9).

## 11. Storage architecture
Parquet (zstd-3) per source-time day under `data/raw/ticks/broker=/server=/instrument=/year=/month=/day=/`, atomic and read-only after write,
chunk manifest + dataset manifest, separate `data/derived/`, checkpoint/resume by manifest + checksum, `acquisition_log.jsonl`. Layout chosen for
resumability and hash-stability (not benchmarked against alternatives: no evidence yet to depart from the plan).

## 12. Dataset provenance design
`raw_content_sha256` (broker columns only) and `file_sha256` per chunk; `dataset_id = tickraw-<hash(identity + ordered chunk content hashes)>`; dataset
manifest checksum; software versions + git commit; time-basis identity; symbol-info hash; account fingerprint; derived datasets reference their raw parent
hashes; `DatasetRef`/`LineageRecord` types defined for later feature-set and model links. Same data + same time rule -> same ID (tested).

## 13. Performance measurements (SYNTHETIC ticks, this Linux sandbox; 4 CPUs; 2,000,000 rows; best of 3)
| Stage | Time | Throughput |
|---|---|---|
| terminal array -> `RAW_TICK_DTYPE` (adapter conversion) | 0.61 s | 3.3 M rows/s |
| array -> Arrow raw table | 0.32 s | 6.3 M rows/s |
| quality assessment | 0.49 s | 4.1 M rows/s |
| Parquet write (zstd-3, fsync, checksum) | 1.23 s | 1.6 M rows/s (~13 MB/s) |
| Parquet read with checksum verification | 0.32 s | 6.3 M rows/s |
| Parquet read without verification | 0.43 s | 4.7 M rows/s |
Peak transient memory about 490 MB (tracemalloc) for a 2 M-row batch (~245 bytes/row). On-disk size 8.5 bytes/tick **on synthetic data, which
compresses far better than real ticks; do not use it for capacity planning**. Reads are page-cache warm (so read-verified < read-unverified is noise).
**MT5 tick retrieval throughput and latency were NOT measured** (needs Windows; the harness records them). Raw file: `docs/baselines/phase1_synthetic_io.json`.

## 14. Failures discovered while building
1. The DST classifier called a fixed +3 h server "New-York-anchored" because my fake's weekend closure was fixed in UTC, violating the classifier's own documented assumption. Fixed (fake now NY-anchored) and covered by tests for both server types.
2. pandas 3 changed default datetime resolution; my `asi8` unit assumptions were wrong (a 7-minute offset error appeared). Replaced by explicit-millisecond conversion + round-trip tests.
3. Blanket key-based log redaction erased the whole `account` block of the report (key "account"). The report now scrubs only secret-bearing keys + explicit secret values; tested.
4. Dataset identity was changing with calibration residuals; identity now uses only basis + rule + `dst_determined`.
5. A future-dated latest tick (wrong offset) was not flagged as stale; now it is.
6. A "noisy sample" test used a seed whose MAD was under the gate; the estimator was right, the test was too weak (strengthened).

## 15. Risks discovered
- All real-MT5 behaviour remains unverified; some assumptions may fail on a real broker (e.g. `None` vs empty array on no data, `date_to` semantics, array dtypes).
- DST inference rests on the 17:00-New-York weekly-open assumption, which may not hold for a given broker's gold CFD.
- Server-time offset estimation needs an accurate local clock and an active market; weekend runs are inconclusive by design.
- Half-hour/odd-offset servers are handled to 15-minute granularity only.
- Very large days (high tick rates) may need `--split-rows`; the per-call cap is unknown until probed.
- `copy_ticks_*` may trigger server-side history downloads, making first requests slow; the call deadline (30 s default) may need raising.
- The fake lives in `src/` and could be mistaken for evidence; reports label fake runs `FAKE_NOT_MT5` / `real_mt5_verified: false`.
- Symbol-map files contain broker/server names (not secrets) but are operator-specific.
- CI workflow untested on GitHub.

## 16. Decisions requiring approval
1. Canonical names `XAU_USD/EUR_USD/GBP_USD` (adopted from your example; older configs used `XAUUSD`).
2. Chunk = source-time day (vs UTC day); normalised UTC is a column, so UTC-day queries filter across two chunks.
3. Quality thresholds and the conventional session hours (`sessions/calendar.py`) as provisional defaults until the real profile exists.
4. `--allow-unverified-time` exists (flagged data); proposal: never use such data for training.
5. DOM probe on by default in the harness (`market_book_add` subscribes to depth; read-only); `symbol_select` adds symbols to Market Watch.
6. Keep `FakeMT5*` in `src/fxscalp/brokers/mt5/fake.py` (importable by scripts) vs moving to `tests/`.
7. Commit `configs/broker_symbols/*.yaml` to git, or ignore them (operator-specific)?
8. Parquet zstd-3 / 262k row groups as defaults.
9. Whether the harness's 30 s default call deadline is appropriate for the first history requests.

## 17. Has real Windows/MT5 verification occurred?
**No.** This environment is Linux; `MetaTrader5` is Windows-only. Steps that **must** run on Windows (see `WINDOWS_MT5_VERIFICATION.md`): `python -m scripts.verify_mt5 ...`
(connection, account class, symbol resolution for XAU/EUR/GBP, metadata, latest tick, tick/bar retrieval, tick fields/flags, time-semantics measurement,
DOM, history depth, request limits, latency) and optionally `acquire_ticks` + `profile_ticks` to produce the real `XAUUSD_DATA_PROFILE.md`.

Completion gate status: reliable connection **UNVERIFIED**; account classification **tested on fake**; XAU/USD resolution **tested on fake**; metadata capture **tested on fake**;
tick acquisition **tested on fake**; time semantics **built, not measured**; bid/ask quality analysis **tested on fake**; raw storage **tested**; provenance **tested**;
fake adapter **done**; tests **pass**; Windows procedure **written**. Phase 1 is therefore *implemented but not accepted*.

## 18. Exact Phase 2 proposal (only after the Windows report has been reviewed and the unverified items resolved)
Phase 2 = tick/bar storage hardening, data-quality pipeline, cost-floor study groundwork (no features, no models):
1. **Gate:** review `mt5_verify_*.json`; fix adapter/convert/time assumptions the real terminal contradicts; re-run until the XAU/USD checks pass.
2. **Deep XAU/USD acquisition:** acquire the history depth found by the probe (chunked, resumable), plus EUR/USD and GBP/USD only for data profiling (no models).
3. **Quality pipeline v2:** thresholds derived from the real profile (spread/gap distributions); gap classifier (weekend/rollover/holiday) using the session calendar; cross-check ticks vs MT5 bars; DST-transition-week audit; quarantine review report.
4. **Bar aggregation:** deterministic bid/ask-aware bars (1s, 5s, 15s, 30s, 1m, 5m, 15m, 1h) from raw ticks as separate derived datasets with manifests and lineage; closed bars only; tests for boundary and DST handling.
5. **Session/calendar validation:** confirm or replace the session-hour conventions from observed activity; add ROLLOVER and OFF_HOURS rules.
6. **Descriptive cost-floor study (E1/E2 groundwork):** spread distributions by session/hour, in price units and relative to short-horizon range — data description only, no trading interpretation.
7. **Dataset registry:** index of dataset IDs, derived lineage, and read API returning verified tables; seal the final-holdout date range **by hash** once depth is known (decision D-6 style: length chosen from data needs).
8. Deferred as instructed: purged-CV/multiple-testing/ledger vendoring decision (validation phase); no feature engine, regime engine, ML, backtest or strategy.
