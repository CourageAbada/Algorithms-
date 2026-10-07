# Phase 1 - Real MT5 DEMO verification result (2026-10-07)

Read-only. No orders, no order_*/Buy/Sell/login calls, no bulk acquisition, no Phase 2B, no models, no signals.
Source artifact: `reports/mt5_verify_20261007T122854Z.json` (git-ignored; contains no password and a masked login).

**Verdict: CONDITIONAL PASS** (harness itself printed PASS; this report downgrades it because of the open items below).

## Environment
- Windows 11 Home SL 10.0.26200 x64; Python 3.12.3 (venv); MetaTrader5 5.0.6231; repo commit 9624552
- Terminal: `C:\Program Files\MetaTrader 5\terminal64.exe`, build 6140, ping_last ~245 ms
- Account: class **DEMO** (trade_mode_raw 0), server `ICMarketsSC-Demo`, company "Raw Trading Ltd", login masked,
  USD, leverage 1:5000, margin_mode_raw 2 (= RETAIL_HEDGING per MT5 docs; not independently verified)
- `.env` holds only `MT5_TERMINAL_PATH` (no credentials), is git-ignored; harness attached to the logged-in terminal.

## XAU/USD (canonical `XAU_USD` -> broker `XAUUSD`, unique match)
| field | value | | field | value |
|---|---|---|---|---|
| digits | 2 | | volume min / max / step | 0.01 / 100 / 0.01 |
| point | 0.01 | | stops level / freeze level | 0 / 0 |
| tick size / value | 0.01 / 1.0 (profit & loss 1.0) | | trade_mode | 4 (FULL per docs) |
| contract size | 100 | | trade_exemode | 2 (MARKET per docs) |
| currencies (base/margin/profit) | XAU / XAU / USD | | filling_mode | 2 (bitmask; IOC per docs - bit meaning UNVERIFIED, needs order_check) |
| order_mode | 127 | | swap long / short | -58.701 / 41.441 |
Quote at report time: bid 4097.33, ask 4097.43, spread 0.10 = 10 points (matches `symbol_info.spread`=10, floating spread);
tick raw time 1791386816.002 (msc 1791386816002) = 2026-10-07 12:26:56.002 UTC after the measured -10800 s shift.

## Time semantics
- Raw MT5 tick times are **server time stored as epoch**, NOT UTC: source_time - UTC = **+10800 s** (UTC+3) on 2026-10-07.
  The MT5 docs' "UTC" claim is contradicted for this broker.
- 60 samples x 2 s: median offset 10800.66 s, MAD 0.04 s, residual 0.66 s, rounded to 15 min; sec/msec consistent. Confidence: high
  for the *current* offset, assuming the Windows clock is NTP-accurate (not independently checked).
- Example: tick msc 1791386816002 received locally 12:26:55.294 UTC -> normalised 12:26:56.002 UTC.
- Basis stored: `server_fixed_offset` (+3 h). **DST: NOT DETERMINED.** Only 8 weekly opens (all in US-DST summer) were
  obtainable; inference needs >=4 weeks each side of a DST change. Tick history starts ~April 2026, i.e. after the March 2026
  switch, and the November switch has not happened. +3 h in October is consistent with a "GMT+2 winter / GMT+3 summer"
  (NY-close) server, but that is an inference only. Any data before 2026-03-08 or after 2026-11-01 must not be converted with
  a fixed +3 h.

## Historical data
- Ticks: work. 30,256 XAUUSD ticks for the last hour fetched in ~4 ms (warm). Dtypes: time i8, bid/ask/last f8, volume u8,
  time_msc i8, flags u4, volume_real f8. `last`/volume are 0 (quote-only feed).
- Tick depth (weekly probe): data present back to 2026-04-01; 2026-03-25 and 2026-02-25 returned 0 rows (~1 week precision).
- **Cold-history behaviour (important):** in a separate read-only diagnostic, first-time requests for old ticks were slow:
  13 weeks back 36 s; 26 and 52 weeks back failed after ~95 s with `Terminal: Call failed`. The harness's 30 s call timeout
  turned this into adapter state LOST in three earlier runs (`--dst-weeks 52`, and once with 8 while the terminal was still
  busy). Warm requests are fast. The successful run used `--dst-weeks 8`, after the terminal had cached the history.
- Bars: M1 5000 bars (to 2026-10-01), H1 5000 bars (to 2025-12-02), D1 5000 bars (to 2007-10-22; 32 s cold). 100000 bars ->
  `Terminal: Invalid params` (terminal maxbars 100000). **Anomaly:** the harness's own M1 hour request returned **1 bar**
  (check still printed PASS), while a manual request moments later returned 60 - first-call/cold behaviour; the harness check is
  too lenient.
- Request limits: **not determined.** Harness reported "suspected_cap 30286" but its windows extend into the future, so row counts
  plateau only because "now" was reached; this is a probe artefact, not evidence of a cap.
- Empty-result behaviour: weeks with no history return 0 rows (fast, ~0.5 ms) in the depth probe; cold unavailable history can
  instead return None / `Call failed` after a long wait. Date-to inclusivity was **not tested**.
- Tick flags observed: 0x402 (245), 0x404 (220), 0x406 (29,791). Bit 0x400 is undocumented and appears on every tick (the
  quality module counts all 30,256 as UNKNOWN_FLAG_BITS) - semantics unresolved.

## EUR_USD / GBP_USD
Resolve uniquely: `EURUSD`, `GBPUSD` (metadata captured, 0 consistency findings, live quotes seen). Nothing further done.

## Latency (terminal API call latency - NOT network/broker or order-execution latency)
- symbol_info_tick: n=83, p50 0.12 ms, p95 0.39 ms, max 0.46 ms (latest_tick sampler n=20: p50 0.14, p95 0.22, max 0.25 ms)
- copy_ticks_range (n=16): p50 23 ms, p95 74 ms, max 74 ms; copy_ticks_from (n=8): p50 62 ms, p95 111 ms
- copy_rates_range (n=1) 182 ms; symbols_get (7,394 symbols) 82 ms; initialize 2.6 ms
- Terminal-to-broker ping reported by terminal_info: ~245 ms (a single reading). Order execution latency: not measured.
- Abnormal: cold deep-history requests, see above.

## Data quality (1-hour XAUUSD window, 30,256 ticks; nothing cleaned)
negative spread 0; zero spread 0; missing bid/ask 0; duplicate ticks 0; duplicate timestamps (same time_msc) **31**;
timestamp reversals 0; large gaps (>300 s) 0; long gaps 0; sec/msec mismatch 0; non-finite 0; extreme spread 0;
quarantined 0; median positive spread 0.10. Unknown flag bits: 30,256 (see above). Stale/future ticks: not separately
assessed beyond the offset-corrected window ending at "now"; one hour in a liquid session is not representative of
rollover, weekend open or news periods.
DOM: `market_book_get` returned 0 levels - depth of market not available (check reported "available" with 0 levels).

## Safety
- DEMO gate evaluated before any collection and passed; order functions called: none; allow-list only (attestation in report).
- Failed earlier runs (reports 121012Z, 121608Z, 122350Z) are kept as evidence; they are the cold-history LOST failures.
- Credential scan of reports/, data/, configs/, docs/: full login number not present; no "password" strings in reports.

## Still unverified
SYMBOL_FILLING bit meanings, hedge-vs-close behaviour, MT5 package thread-safety, DST behaviour, tick-flag 0x400, request
limits, date-to inclusivity, behaviour outside liquid hours, gaps over weekends, order-execution latency.

## Conditions before controlled historical tick acquisition
1. Treat time as `server_fixed_offset +3 h` valid only for the current summer-time period; resolve DST (e.g. after 2026-11-01
   plus the weekly-open probe, or a documented broker statement) before converting older/newer data. Do not assume UTC.
2. Acquisition must be chunked, resumable and tolerate cold downloads: per-call timeout well above 30 s (observed 36-95 s) and
   a retry/skip path, not a LOST state. Warm the terminal by requesting history gradually, oldest-needed first.
3. Available tick history is only ~6 months (from ~2026-04); plan datasets accordingly.
4. Re-run the harness inside the busiest hours and over a weekend open; fix the lenient bars check and the request-limit probe
   before relying on them.
5. Record the undocumented flag bit 0x400 and 31 duplicate-timestamp ticks in the profiling plan (preserve, tag, do not drop).
