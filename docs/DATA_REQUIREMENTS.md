# Data Requirements

## 1. Datasets

| Dataset | Source | Needed fields | Use |
|---|---|---|---|
| Raw ticks (primary) | **Same broker's MT5** `copy_ticks_range` (demo account used for trading) | time_msc, bid, ask, last, volume, flags | Training, backtest, replay, shadow |
| Raw ticks (supplementary) | Third-party tick vendor (e.g. Dukascopy-style public tick history; terms, XAUUSD availability and licensing **to verify**) | bid, ask, timestamp | Longer history, cross-feed robustness checks. Never mixed into training without a feed-id feature or separate model |
| Broker bars | MT5 `copy_rates_*` | OHLC, tick_volume, spread, real_volume | Sanity checks only; our bars are built from ticks |
| Symbol metadata | `symbol_info` snapshots | see `FOREX_MARKET_MODEL.md` | Sizing, costs; stored with timestamps (metadata changes) |
| Account/deal history | MT5 | commissions, swaps, fills | Cost calibration |
| Economic calendar | Provider TBD (D-5); requirement: event time (UTC), currency, impact, forecast/actual/previous, **point-in-time** (no retroactive revisions), revision history | | Event-risk gating |
| DOM (optional) | MT5 `market_book` if available | | Experimental only |
| Macro/cross-asset (optional) | DXY, yields, etc. | | Only if point-in-time and timestamp-correct |

## 2. Volume and history

- Ticks: depth is broker-limited; Phase 1 measures what is obtainable (D-3: start with broker MT5 tick history behind the `HistoricalDataProvider` abstraction; higher-quality external tick data can be added later). The required history is determined from the data profile: enough distinct volatility/session regimes and enough events to support disjoint train / validation / walk-forward / untouched-holdout segments with the sample sizes required by the pre-registered power analysis (D-6). No fixed number of months is assumed.
- Storage estimate (to be measured): compressed Parquet ~ 20-60 bytes/tick; at 10^5-10^6 ticks/day this is well under 1 GB/month.

## 3. Quality pipeline (Phase 2)

Checks, each producing a counted, reported finding and never silent drops:
duplicate ticks; out-of-order timestamps; bid > ask (crossed); zero/negative prices; price jumps beyond k x robust-vol; spread outliers; gaps (intra-session) and expected closures; stale periods (no update > N s while market open); timezone/offset verification; symbol-metadata change detection; flag decoding sanity; coverage by day/session. Raw data are immutable; cleaning produces a derived dataset with a manifest listing every transformation and the number of rows affected.

## 4. Layout

```
data/ticks/symbol=XAUUSD/broker=<id>/date=YYYY-MM-DD/part-*.parquet   (raw, immutable)
data/bars/symbol=.../res=1s|5s|.../date=...                              (derived)
data/features/<feature_set_version>/symbol=.../date=...                  (derived, versioned)
data/manifests/*.json   (source, range, row counts, SHA-256, schema version, code commit)
```
All timestamps UTC int64 ms (and a `time_msc_raw` column preserved exactly as returned by MT5 for audit). The conversion to UTC uses the **empirically calibrated server-time offset** (documentation and community reports conflict on whether MT5 returns UTC or server time; see `PHASE0_5_UPSTREAM_DEEP_DIVE.md` §4 #15); the dataset manifest records `time_basis` and how the offset was derived. Data with `time_basis = unverified` is not used for training.

## 5. Database choice and rationale

- **Parquet (+ Polars/DuckDB) for time series:** columnar, compressible, immutable partitions are easy to hash/version for reproducible experiments, fast sequential scans for backtest/replay, no write-amplification, trivial to move between Windows collector and Linux research.
- **PostgreSQL for transactional/relational:** orders, fills, positions, trades, journal, risk events, model registry, experiments, backtest runs: needs constraints, joins, updates, and audit history.
- **TimescaleDB (optional, later):** live metrics/health time series and dashboard queries once volume justifies it. Not required for Phase 1-5.
- Rejected: a single giant tick table in Postgres (cost, bloat, slower scans); SQLite for the live path (concurrency); proprietary tick DBs (cost, lock-in) unless measurement shows a need.

## 6. Reproducibility

Every experiment records dataset manifest hashes, feature-set version, code commit, config, random seeds. Re-running reproduces metrics within stated tolerance.
