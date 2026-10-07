# XAU/USD real tick dataset profile (ICMarkets DEMO via MT5), acquired 2026-10-07

**Status: descriptive profile only.** No labels, forecast horizons, regimes, thresholds, models or strategy exist. Account login and
credentials are not part of this document.

## 1. Feed and acquisition
- Broker/feed: "Raw Trading Ltd", server `ICMarketsSC-Demo`, DEMO account (gate unchanged), symbol `XAUUSD` (canonical `XAU_USD`; digits 2, point 0.01, contract 100).
- Acquired 2026-10-07 19:42-19:46 UTC with `scripts.acquire_history` (newest complete source day first, then backwards; verified immutable chunk per source day; read-only).
- Range: source days **2026-03-31 .. 2026-10-06** (190 days). The incomplete day (2026-10-07) was not requested.
- **Dataset id (deterministic): `tickraw-fe5b9137d477c7c866d5`**, manifest `data/datasets/raw_trading_ltd/icmarketssc-demo/XAU_USD/tickraw-fe5b9137d477c7c866d5.json`
  (hash of identity fields + per-day raw content hashes + time-basis spec; re-computation verified by the integrity script).
- Time calibration: `timecal-c9cc2a3a65a3dd80` (+10800 s, 60 samples, residual 0.55 s, MAD 0.15 s, valid 2026-04-05T22:00:02Z .. 2026-10-07T19:23:06Z, confidence high, `dst_status = unresolved_no_us_dst_transition_in_observed_window`).

## 2. Size and history boundary
| item | value |
|---|---|
| total raw ticks | **85,577,391** (stored = fetched; 0 clipped, 0 deleted, 0 repaired) |
| first / last raw tick (UTC) | 2026-03-31T22:49:48.409000+00:00 / 2026-10-06T20:58:56.796000+00:00 |
| history boundary | first tick 2026-03-31 22:49:48 UTC (source day 2026-04-01); source day 2026-03-31 returned a legitimate empty result |
| source days / with ticks / closed-no-ticks | 190 / 134 / 56 |
| ticks per trading day | min 207,335, p25 538,477, median 646,020, p75 752,190, max 1,115,360 |
| Parquet on disk | 1.36 GB (raw 663 MB, tick_basic 595 MB, tick_quality 99 MB) = 15.9 bytes/tick; ~23 GB disk still free |

## 3. Acquisition behaviour (monitor log `data/acquisition_monitor.jsonl`)
- 190 chunks in total: 134 days with ticks and 56 closed days (legitimate empty results); 2026-10-06 came from the earlier dry run (verified, skipped), 3 days from a supervised 3-chunk trial, the rest from the full run. **0 unavailable-history responses, 0 retries, 0 request timeouts, 0 DEGRADED/LOST transitions, 0 health checks, 0 failures.**
- Per chunk (request + convert + quality + write + verify): median 1.7 s, p95 2.5 s, max 2.9 s; 284 s in total. Each day was one request (no row cap hit; up to 1.1 M ticks per request).
- **The cold-history / retry / timeout path was NOT exercised on the real terminal in this run**: the terminal's history had already been downloaded by earlier diagnostics and verification probes (those showed 36-95 s cold loads and failures before the hardening). The path is covered only by fake-based tests. No dataset-level evidence about it exists.
- Resources: peak RSS per day 255 MB + 347 MB per million ticks (linear fit; max 699 MB at 1.12 M ticks; correlation with day size 0.82); RSS between chunks 240 MB (first 10) -> 317 MB (last 10), drift +0.87 MB/chunk - no leak/non-linear growth; processing time scales with ticks (corr 0.90, ~2.9 s per million ticks). Disk: 1.36 GB written.
- Backwards-order caveat: for a day, the previous day's last tick was usually not yet available, so the first tick of each day was not checked for gap/duplicate/reversal against its predecessor (recorded in each quality manifest as `first_tick_context`). Day boundaries coincide with the daily break/weekend, so the practical impact is nil.

## 4. Integrity (`scripts.verify_dataset_integrity`, exit 0)
All 190 manifests present and status `verified`; raw/tick_basic/tick_quality checksums, row counts and exact Parquet schemas verified; derived parent hashes match; each chunk's ticks lie inside its own source-day window, windows are contiguous and non-overlapping; no duplicate day, no duplicate chunk content, no gap in the manifest, no stray `.tmp` or out-of-range chunk; seq is 0..n-1; first tick of each day is after the previous day's last tick; sum of rows = 85,577,391 = manifest total; dataset id and manifest checksum reproduce.

## 5. Quality findings (all tagged, nothing removed; quarantined = 0)
| tag | ticks | note |
|---|---|---|
| negative spread / missing bid / missing ask / non-finite | 0 | |
| ZERO_SPREAD | 818 | concentrated in early April (2026-04-02/07/08) |
| EXTREME_SPREAD (> 20x the day's median positive spread) | 8,234 | 4,584 on 2026-05-19 alone |
| DUP_TIMESTAMP (same millisecond as previous tick) | 726,460 (0.849 %) | contents differ; kept in broker order |
| DUP_TICK (adjacent, identical content) | 0 (rate 0) | |
| TIME_REVERSAL / SEC_MSEC_MISMATCH / FUTURE_TICK / TIME_AMBIGUOUS / TIME_NONEXISTENT | 0 | |
| LARGE_GAP (> 300 s inside a day) | 6 | see section 8 |
| UNKNOWN_FLAG_BITS | 84,039,243 (98.2 %) | informational (see below) |
| TIME_BASIS_UNCERTAIN | 516,161 | 2026-04-01 and 2026-04-02 only |
`last` and `volume` are 0 on every tick (quote-only feed): there is no traded volume information.

### Undocumented flags (preserved verbatim, never interpreted)
Unknown-bit distribution: `0x80`: 68,461,015; `0x480`: 15,422,363; `0x400`: 155,865.
**Bit 0x80 is present from the first day to the last; bit 0x400 first appears on 2026-09-07** (22 days to 2026-10-06; 15,578,228 ticks carry it). Before 2026-09-07 no tick has 0x400. This is a change in the feed's flag content on or around 2026-09-07 (cause unknown). Raw flag histogram: {"0x4": 1432116, "0x82": 1567803, "0x86": 66893212, "0x2": 2326, "0x6": 103706, "0x404": 155865, "0x482": 173622, "0x486": 15248741}.

## 6. Time basis and limitations
- Ticks with VERIFIED time basis (inside the calibration record's validity period): **85,061,230 = 99.3968 %**.
- Ticks tagged TIME_BASIS_UNCERTAIN: **516,161 = 0.6032 %** (2026-04-01: 245,685 and 2026-04-02: 270,476 - before the first weekly open that supports the offset, 2026-04-05T22:00Z). They are stored with raw `source_time`, `source_time_msc`, the normalization rule `fixed+10800s` and `time_basis_id`, so they can be re-normalised.
- The +10800 s offset is supported by 26 weekly opens (time-of-week constant within 123 s) for 2026-04-05 .. 2026-10-04 and by direct sampling on 2026-10-07. **DST is unresolved**: this window contains no US DST transition (2026-03-08, 2026-11-01), so the evidence cannot show how the offset behaves across one. Anything outside the validity period (e.g. after the next transition) must be re-calibrated, not assumed. Session labels in this document are DST-aware conventions applied to the normalized UTC axis.

## 7. Spread behaviour (points; 1 point = 0.01 USD; overall n = 85,577,391)
Overall: mean 8.879, p5 5.05, median 9.05, p95 11.95, p99 15.95, p99.9 50.05, max 700 (quantiles from a 0.1-point histogram, +-0.05). Daily median ranges 8-10.
The spread is quantised: the dominant values are 5, 9-12 and 40 points. Spread widens sharply before the daily break (see windows) and at 12:30 UTC releases.

By session (labelling convention, DST-aware):
| session | ticks | share | days | p50 | p95 | p99 | p99.9 | mean |
|---|---|---|---|---|---|---|---|---|
| ASIA | 23,090,112 | 27.0% | 134 | 9.95 | 12.05 | 13.05 | 43.05 | 9.296 |
| LONDON | 15,333,060 | 17.9% | 134 | 9.05 | 11.05 | 11.95 | 39.05 | 8.497 |
| NEW_YORK | 17,434,582 | 20.4% | 134 | 9.05 | 11.05 | 39.95 | 46.05 | 9.073 |
| LONDON_NEW_YORK_OVERLAP | 26,372,407 | 30.8% | 134 | 9.05 | 10.05 | 17.05 | 65.05 | 8.476 |
| OFF_HOURS | 3,347,230 | 3.9% | 134 | 10.05 | 12.05 | 25.05 | 50.05 | 9.928 |

By UTC hour (spread p50/p95/p99 and tick share):
| hour | ticks | share | p50 | p95 | p99 |
|---|---|---|---|---|---|
| 00 | 3,524,032 | 4.1% | 10.05 | 12.05 | 13.95 |
| 01 | 4,975,940 | 5.8% | 10.05 | 12.05 | 14.05 |
| 02 | 3,411,162 | 4.0% | 10.05 | 12.05 | 12.05 |
| 03 | 2,533,562 | 3.0% | 9.05 | 12.05 | 12.05 |
| 04 | 1,913,635 | 2.2% | 9.95 | 12.05 | 12.05 |
| 05 | 3,120,611 | 3.6% | 10.05 | 12.05 | 14.05 |
| 06 | 3,611,170 | 4.2% | 9.05 | 11.05 | 14.95 |
| 07 | 3,228,712 | 3.8% | 9.05 | 11.05 | 11.95 |
| 08 | 3,378,378 | 3.9% | 9.05 | 11.05 | 12.95 |
| 09 | 2,860,238 | 3.3% | 9.05 | 11.05 | 11.95 |
| 10 | 2,669,338 | 3.1% | 9.05 | 11.05 | 11.05 |
| 11 | 3,196,394 | 3.7% | 8.95 | 10.05 | 10.05 |
| 12 | 4,975,463 | 5.8% | 9.05 | 11.05 | 23.05 |
| 13 | 7,653,690 | 8.9% | 9.95 | 10.05 | 19.05 |
| 14 | 7,815,506 | 9.1% | 8.05 | 10.05 | 14.95 |
| 15 | 5,927,748 | 6.9% | 7.95 | 10.05 | 11.95 |
| 16 | 4,492,632 | 5.2% | 9.05 | 11.05 | 11.95 |
| 17 | 4,002,514 | 4.7% | 9.05 | 11.05 | 11.95 |
| 18 | 3,835,148 | 4.5% | 9.05 | 11.05 | 11.95 |
| 19 | 3,718,172 | 4.3% | 9.05 | 11.05 | 11.05 |
| 20 | 1,386,116 | 1.6% | 9.05 | 40.05 | 40.05 |
| 21 | 0 | 0.0% | - | - | - |
| 22 | 1,668,221 | 1.9% | 9.95 | 14.05 | 39.95 |
| 23 | 1,679,009 | 2.0% | 10.05 | 12.05 | 14.05 |

Special windows:
| window | ticks | p50 | p95 | p99 | mean |
|---|---|---|---|---|---|
| rollover_pre_20_30_21_00 | 500,340 | 39.95 | 40.05 | 74.05 | 27.044 |
| rollover_post_22_00_22_30 | 976,185 | 9.95 | 15.95 | 50.05 | 10.747 |
| week_open_first_60min | 532,083 | 9.95 | 13.05 | 48.95 | 10.259 |
| week_close_last_60min | 307,462 | 9.05 | 40.05 | 40.05 | 13.9 |

By month (median across days unless noted):
| month | days | ticks | median ticks/day | median tick rate Hz | median spread | median daily p99 | max spread | median range bps | median 1-min return std bps |
|---|---|---|---|---|---|---|---|---|---|
| 2026-04 | 21 | 10,877,667 | 590,811 | 7.14 | 9 | 35 | 371 | 219 | 3.81 |
| 2026-05 | 21 | 13,092,292 | 602,240 | 7.28 | 10 | 33 | 201 | 217 | 3.71 |
| 2026-06 | 22 | 15,907,154 | 741,902 | 8.97 | 10 | 12 | 116 | 237 | 4.09 |
| 2026-07 | 23 | 12,997,148 | 546,178 | 6.60 | 9 | 12 | 261 | 216 | 3.40 |
| 2026-08 | 21 | 13,990,780 | 642,185 | 7.76 | 9 | 12 | 129 | 196 | 3.44 |
| 2026-09 | 22 | 16,062,973 | 749,294 | 9.08 | 9 | 12 | 700 | 221 | 3.60 |
| 2026-10 | 4 | 2,649,377 | 651,772 | 7.89 | 9 | 12 | 700 | 163 | 2.98 |
**Regime shift visible in the spreads:** median daily p99 spread is ~35 points in April-May and ~12 from June, i.e. the spread distribution is not stationary across the sample. Widest events: 700 pts at 2026-10-02T12:30:01Z, 700 pts at 2026-09-11T12:30:01Z, 508 pts at 2026-09-16T12:30:00Z, 500 pts at 2026-09-04T12:30:08Z, 500 pts at 2026-09-03T12:30:02Z, 500 pts at 2026-09-03T12:30:02Z (almost all at 12:30 UTC, i.e. scheduled US data releases).

## 8. Tick-rate behaviour
Mean tick rate per trading day: p5 4.2, p25 6.5, median 7.8, p75 9.1, p95 11.0 Hz. Per-second counts over 11,031,809 observed seconds (quiet seconds included): p50 5, p90 18, p99 38, max bin 246. Activity peaks 13:00-15:00 UTC (~9 % of ticks per hour) and is lowest 04:00 UTC (2.2 %). The first two weeks of April are thin (207k-270k ticks/day, 3.2-4 Hz).

## 9. Session and structure coverage
- All five sessions are represented on all 134 trading days (Asia 27.0 %, London 17.9 %, New York 20.4 %, London/New York overlap 30.8 %, off-hours 3.9 % of ticks).
- Daily rollover/break: 106 observed breaks, typically 3,661 s (about 20:59 -> 22:00 UTC; hour 21 has no ticks). 20:30-21:00 UTC spreads are about 4x normal (median 40 points); 22:00-22:30 UTC are near normal. Two US-holiday early closes (2026-05-25 and 2026-09-07: ticks stop 18:30 UTC, resume 22:00 UTC).
- Week open: 27 weekly opens in the data (Sunday ~22:00 UTC), 532,083 ticks in the first hour after open. Week close: 27 closes (Friday ~20:59 UTC); the last hour before the close is wide (median 9, p95 40). Weekly closures last 49.05 h normally; the Easter weekend 2026-04-02 -> 04-05 lasted 73.0 h. The first and last partial weeks are in the sample.
- Weekend closures and the Good Friday closure (2026-04-03) are legitimate, not data gaps.

## 10. Unusual days (descriptive, for later examination; no thresholds fitted)
- Highest range: 2026-04-02, 528.0; 2026-08-05, 483.3; 2026-06-11, 478.9; 2026-06-10, 459.0; 2026-08-19, 450.9 (bps of mid).
- Highest 1-minute return std: 2026-04-02, 7.7; 2026-06-11, 6.7; 2026-06-10, 6.3; 2026-09-04, 6.2; 2026-06-17, 5.8.
- Widest spread p99: 2026-05-19, 148.0; 2026-04-02, 46.0; 2026-05-21, 45.0; 2026-04-07, 42.0; 2026-04-01, 40.0; widest single spread: 2026-10-02, 700.0; 2026-09-11, 700.0; 2026-09-16, 508.0; 2026-09-03, 500.0; 2026-09-04, 500.0.
- Highest tick rate (Hz): 2026-06-11, 13.5; 2026-06-24, 12.9; 2026-06-10, 12.5; 2026-06-25, 12.1; 2026-08-28, 11.8; lowest (non-holiday-filtered): 2026-04-08, 3.2; 2026-04-02, 3.3; 2026-07-03, 4.7; 2026-07-10, 4.7; 2026-05-25, 4.9.
- Largest intra-day gaps: 2026-06-29, 7361.4; 2026-06-15, 1340.6; 2026-04-22, 871.2; 2026-04-01, 871.2; 2026-05-26, 592.7 s. **2026-06-29 has a ~2 hour hole (05:04 -> 07:07 UTC, 7,361 s) plus a 456 s gap at 12:33 and 162 s at 09:02 on a normal Monday; unexplained - possible feed outage.** 18 intra-day gaps exceed 60 s; 6 exceed 300 s.
- Per-day table (all required metrics): `data/profiles/XAU_USD_daily_profile.csv`; aggregates: `data/profiles/XAU_USD_profile.json` (both git-ignored data artifacts).

## 11. Representative feature-pipeline feasibility (Phase 2A, independent single-day runs)
| profile | day | ticks | rows / valid | features | validation | prefix invariance (row-values) | peak RSS MB | s |
|---|---|---|---|---|---|---|---|---|
| normal_activity | 2026-08-26 | 642,185 | 82,733 / 78,233 | 119 | OK | PASS (27,008,812) | 501 | 5.7 |
| high_volatility | 2026-04-02 | 270,476 | 82,736 / 78,236 | 119 | OK | PASS (25,415,908) | 484 | 4.3 |
| wide_spread | 2026-05-19 | 657,782 | 82,738 / 78,238 | 119 | OK | PASS (27,714,496) | 550 | 5.4 |
| widest_single_event_rollover_window | 2026-09-11 | 867,767 | 82,497 / 77,997 | 119 | OK | PASS (26,552,120) | 712 | 6.1 |
| low_activity | 2026-04-10 | 207,335 | 82,618 / 78,118 | 119 | OK | PASS (26,013,216) | 471 | 4.4 |
| week_open | 2026-07-13 | 625,266 | 82,739 / 78,239 | 119 | OK | PASS (25,137,156) | 571 | 5.3 |
| week_close | 2026-05-08 | 651,561 | 82,614 / 78,114 | 119 | OK | PASS (25,468,732) | 618 | 5.5 |
All 7 runs: 119 features, validation passes (only warning: `session_so_far_complete` constant on a single day), prefix invariance holds on the real ticks (normalized UTC axis, 4 cutoffs, full normalise + features chain), peak memory <= 712 MB (one process, cumulative). Regime thresholds remain UNCALIBRATED placeholders. Note 2026-04-02 (the highest-volatility day) is entirely TIME_BASIS_UNCERTAIN.

## 12. Unresolved issues
1. DST/offset outside 2026-04-05 .. 2026-10-07 unresolved (needs the 2026-11-01 transition or a broker statement); 516,161 ticks are uncertain-tagged.
2. Flag content changed on 2026-09-07 (0x400 appears); 0x80 and 0x400 semantics unknown. Any feature or filter must not rely on flags.
3. Non-stationary microstructure within the sample: April-May spread tail (p99 ~35 pts) vs June onwards (~12), changing tick rates (April thin).
4. Unexplained 2026-06-29 outage-like gap (and 2026-06-15 1,341 s, 2026-04-22 871 s, 2026-04-01 871 s, 2026-05-26 593 s).
5. Cold-history retry/timeout path unproven on the real terminal; first-tick-of-day context checks skipped because of backwards order.
6. Only ~6.2 months, all in US/UK summer time; one DEMO feed (quote-only, no volume); demo spreads/latency may differ from a live account; no execution or slippage data.
