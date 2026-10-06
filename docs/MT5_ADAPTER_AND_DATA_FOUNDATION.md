# Phase 1 Design: MT5 Adapter and Data Foundation

Scope: read-only. No orders, no positions, no order history, no strategy, no model.

## 1. Pipeline
```
MT5 terminal -> ReadOnlyMT5Api (allow-list) -> MT5Owner thread -> MT5BrokerAdapter
   connect -> verify account (DEMO gate) -> discover symbols -> symbol metadata -> time calibration
   -> TickAcquirer (chunked, resumable) -> TickStore (immutable Parquet + manifests)
   -> derived datasets (tick_basic, tick_quality) -> profile
```

## 2. Lifecycle and concurrency model
States: `DISCONNECTED -> CONNECTING -> CONNECTED (initialised AND verified) -> LOST | REJECTED | CLOSED`.
`connect()`: load package -> start owner thread -> `initialize` (once) -> `account_info` -> identity cross-check vs `.env`
-> **account class gate** -> `terminal_info.connected` check. Any failure after initialisation calls `shutdown` and stops the
owner before raising; a rejected (non-DEMO) account is never retried by `reconnect()`.

Concurrency assumptions (all documented in `brokers/mt5/owner.py`):
1. MT5 package thread-safety is **undocumented/unverified** -> every call (including `initialize`/`shutdown`) runs on ONE
   daemon owner thread, FIFO, one at a time. Callers on any thread block until result/error/deadline.
2. Calling the owner from its own thread raises (deadlock guard).
3. A blocking call cannot be cancelled. A deadline overrun marks the owner **wedged**, the adapter state `LOST`, and all later
   calls fail fast (`CallTimeoutError`/`ConnectionLostError`). `reconnect()` creates a new owner; the abandoned thread may still
   hold IPC state, so repeated wedges should be handled by restarting the process.
4. Whether initialise/use/shutdown must share a thread is unknown; sharing one is the conservative choice.
5. The session is persistent: `initialize` once, `shutdown` once (tests assert exactly one of each across many calls).
6. Per-call latency is recorded per function (count, mean, p50, p95, max) and reported by the harness.
7. `HealthMonitor` (optional thread) calls `health_check()` periodically; it records and reports, it never auto-reconnects by default.

## 3. No-trading guarantees (defence in depth)
1. The adapter only holds `ReadOnlyMT5Api`: an allow-list of read functions. `order_send`, `order_check`, `order_calc_*`,
   `Buy/Sell/Close`, `login`, positions/orders/history functions raise `TradingDisabledError`; unknown names raise `AttributeError`.
2. All `BrokerAdapter` trading methods (`submit_order`, `modify_order`, `cancel_order`, `close_position`, `get_positions`,
   `get_orders`) raise `TradingDisabledError`.
3. The fake module exposes trading functions only as tripwires and tests assert they are never called.
4. A source-scan test fails if any file other than the blocklist (`api.py`) and the fake mentions an order function, or if
   `MetaTrader5` is imported outside `brokers/mt5`.
Note: `symbol_select` adds a symbol to Market Watch (data visibility). `market_book_add/release` subscribe to depth of market.
Neither trades.

## 4. Account safety gate
`classify_trade_mode`: 0 -> DEMO, 1 -> CONTEST, 2 -> REAL (constants verified in the installed package source), anything else
-> UNKNOWN. Phase 1 allows **DEMO only**; REAL/CONTEST/UNKNOWN raise `AccountNotAllowedError`, the session is closed and no
data call is made. There is no override. Identity (masked login, fingerprint, server, company, class, margin mode, leverage,
currency) is logged and stored; the full login and password never are. The fingerprint is `sha256(login|server)[:12]`.

## 5. Symbol discovery and mapping
Canonical names: `XAU_USD`, `EUR_USD`, `GBP_USD` (aliases in `configs/instruments/*.yaml`). Tiers: exact alias (0), case-insensitive
(1), separator-insensitive (2), alias + recognised broker affix such as `m`, `.a`, `pro` (3), other affix (4, never auto-selected).
Ties are `ambiguous`; metadata currency mismatches are rejected; explicit `pins` resolve ambiguity. The result is persisted as
`configs/broker_symbols/<broker>__<server>.yaml` (deterministic, no timestamps). Application code resolves names only through
`SymbolMap` (`canonical -> broker symbol`).

## 6. Instrument metadata
Persisted per instrument as content-addressed JSON `data/metadata/symbol_info/<broker>/<server>/<CANONICAL>/<sha16>.json`:
`raw` (every field exactly as returned, no unit conversion), `typed` (projection: digits, point, tick size/value (+profit/loss),
contract size, volume min/max/step/limit, stops/freeze level, spread, spread_float (spread mode), trade/execution/calc mode,
filling/order/expiration mode, swap fields, currency base/profit/margin, path, description, book depth), `consistency_issues`.
Units are not assumed; `check_symbol_consistency` flags: point vs digits, tick size vs point, tick_value vs contract_size*tick_size
when profit currency equals account currency, volume min/step multiples, invalid/missing values.
Field names present in the installed package binary are listed in `PHASE0_5_UPSTREAM_DEEP_DIVE.md`; types/units are verified only by the Windows run.

## 7. Time semantics (measured, never assumed)
Method (`market_data/timebase.py`):
1. **Offset estimation.** Sample the latest tick N times; `delta = tick.time_msc/1000 - local_utc_at_receipt = offset - tick_age
   (+ clock error)`. Only ticks that advanced are used (stale feeds are ignored). The median is rounded to 15 minutes; the **residual**
   and MAD are reported; large residual/MAD/implausible offset -> `inconclusive` (never a guess). Compared: `time` vs `time_msc//1000`,
   MT5 tick vs local UTC. The broker's server clock is observable only through tick timestamps (no API clock).
2. **DST inference (optional, assumption-labelled).** For N past weeks fetch the first tick after Saturday 12:00 (source time)
   with `copy_ticks_from(count=1)` (the weekly open); compare the source time-of-week of the open between US-DST and standard-time weeks.
   Unchanged -> `us_dst_anchored` (server follows New York DST); shifted by 1 h -> `fixed_offset_vs_ny_open`; otherwise `inconclusive`.
   **Assumption stated in the output:** the weekly open is anchored to 17:00 New York; this is an inference, not a verification.
3. **Resolution** -> `TimeBasis`: `utc_verified` (measured offset 0), `server_fixed_offset`, `server_dst_rule` (zone rule, e.g. NY+7h),
   `unverified`. `dst_determined` says whether DST behaviour is supported by evidence.
4. **Transformation (explicit):** `normalized_utc = source_time - offset(rule, instant)`; for zone rules the offset follows the zone's DST dates;
   a repeated DST hour is resolved only when the observed tick order shows the wall clock jumping back, otherwise the ticks are flagged
   `TIME_AMBIGUOUS`; wall times that never existed are flagged `TIME_NONEXISTENT` (quarantined).
5. `TimeBase.normalize` **refuses** an `unverified` basis unless `allow_unverified=True`, in which case `time_basis=unverified` is written into every row and manifest.
The spec is saved at `data/metadata/timebase/<broker>/<server>.json`. Dataset identity includes only basis + rule + `dst_determined`
(not calibration timestamps/residuals).

## 8. Tick schema (RAW dataset `tick_raw/1`)
| Column | Type | Meaning |
|---|---|---|
| seq | int64 | row order as received (key for derived/quality datasets) |
| source_time | int64 | raw `tick.time` (s), source domain |
| source_time_msc | int64 | raw `tick.time_msc` (ms), source domain |
| normalized_utc_time | timestamp[ms, UTC] | per section 7 |
| ingestion_time_utc | timestamp[ms, UTC] | when we received it |
| time_basis | string | per section 7 |
| bid, ask, last | float64 | raw |
| volume | uint64 | raw tick volume field |
| volume_real | float64 | raw; NaN if the terminal does not provide it |
| flags | uint32 | raw tick flags |
Derived `tick_derived/1` (separate dataset): `seq, mid, spread, spread_points (= spread/point), relative_spread (= spread/mid)`.
Quality `tick_quality/1` (separate dataset): `seq, quality_flags (bitmask), quarantined`.

## 9. Data-quality rules (tag, never delete)
MISSING_BID/ASK (NaN, <=0), NEGATIVE_SPREAD (ask<bid), ZERO_SPREAD, EXTREME_SPREAD (spread > k x batch median; k=20 default),
DUP_TIMESTAMP, DUP_TICK, TIME_REVERSAL, LARGE_GAP (>300 s default; gaps >6 h listed), SEC_MSEC_MISMATCH, NONFINITE, TIME_AMBIGUOUS,
TIME_NONEXISTENT, UNKNOWN_FLAG_BITS (outside verified BID/ASK/LAST/VOLUME/BUY/SELL). **Quarantined** = MISSING_*, NEGATIVE_SPREAD, NONFINITE,
TIME_REVERSAL, TIME_NONEXISTENT. Thresholds are quality heuristics recorded in each quality manifest, to be re-derived from the data profile.
The first tick of a chunk is checked against the previous chunk's last tick when available.

## 10. Storage and provenance
Layout: `data/raw/ticks/broker=<slug>/server=<slug>/instrument=<CANONICAL>/year=YYYY/month=MM/day=DD/{ticks.parquet,manifest.json}`
(one chunk = one SOURCE-time day), `data/derived/<tick_basic|tick_quality>/<same>/`, `data/datasets/<broker>/<server>/<instrument>/<dataset_id>.json`,
`data/metadata/{symbol_info,timebase}/...`, `data/acquisition_log.jsonl`. Parquet, zstd level 3, row groups of 262,144, stats on.
Chunk completeness = manifest exists AND file sha256 matches. Writes are atomic (temp+fsync+rename); files are made read-only; a different
payload for an existing chunk raises `ChunkExistsError`. Chunk manifest: schema, broker/server/instrument/broker symbol, source day, requested
window and semantics, counts, first/last times, clipped rows, time basis + spec hash, ingestion time, account fingerprint, symbol-info hash,
request latency, `file_sha256`, `raw_content_sha256` (hash of the broker's own columns, independent of Parquet encoding and normalisation),
software versions (+ git commit).
Dataset manifest: `dataset_id = tickraw-<sha256(identity + ordered chunk content hashes)[:20]>`, chunks, record count, time basis spec,
symbol-info hash, software, collection time, lineage (parents: none), and `manifest_checksum_sha256`. Same broker data + same time rule => same
dataset ID (tested). Provenance chain to build on: dataset ID -> (later) feature set version -> model ID. `core/provenance.py` defines `DatasetRef` and `LineageRecord`.

## 11. Acquisition behaviour
Chunk window `[00:00, 24:00)` source time, half-open: requested as `[start, end]` and clipped (clipped rows counted) because `date_to` inclusivity is
unverified. A request returning >= `split_rows_threshold` rows is split recursively (so a silent per-call cap cannot truncate data; tested).
`probe_request_limits` (growing windows; row-count plateau suggests a cap) and `probe_history_depth` (gallop + bisect on Wednesday 13:00 windows;
~1-week precision; assumes no holes) are run by the harness. Resume: verified chunks are skipped; a corrupt chunk raises rather than being trusted.
Connection loss mid-range: bounded reconnect + retry, then the range stops and no dataset manifest is written.

## 12. Failure catalogue
`MT5UnavailableError` (wrong OS / 32-bit / DLL), `MT5InitializationError`, `MT5AuthError`, `AccountNotAllowedError`, `AccountMismatchError`,
`SymbolNotFoundError`, `AmbiguousSymbolError`, `NoTickDataError`, `StaleTickError` (old or future-dated tick), `ConnectionLostError`, `CallTimeoutError`,
`DataFormatError`, `TimeCalibrationError`, `TradingDisabledError`, `ChunkExistsError`, `ChunkCorruptError`. Each message states the cause and the action.
