# Phase 1.1 - Acquisition hardening and one-day real-data dry run (2026-10-07)

Read-only throughout: no orders, no models, no signals, no Phase 2B calibration, no multi-month acquisition. DEMO gate,
trading-disabled proxy, order restrictions and live interlocks are unchanged (diff touches none of them).

## What changed (and why)
| Area | Change |
|---|---|
| Timeouts | History requests (`copy_ticks_*`, `copy_rates_range`) use their own timeout (`history_timeout_s`, default 180 s = ~2x the worst measured cold load of 95 s) and a *soft* deadline: overrun raises `RequestTimeoutError`, the adapter goes `DEGRADED`, runs `terminal_info` as a health check (default 60 s, queued behind the still-running call) and returns to `CONNECTED` if it passes. Only a failed health check yields `LOST`/`CallTimeoutError`. Non-history calls keep the strict 30 s wedge-on-timeout behaviour. |
| Error taxonomy | legitimate empty (empty array) / `HistoryUnavailableError` (None + success/not-found code) / `RequestTimeoutError` / `TerminalRequestError` (e.g. -1 "Call failed") / `ConnectionLostError` (IPC range, failed health check). |
| Retry | `market_data/retry.py`: bounded (3 retries, 5 s x3 backoff capped 60 s), only for timeouts and terminal code -1; one confirmation retry for "no data"; connection loss is never retried here (reconnect path). Every retry is logged (`acquisition_log.jsonl`, chunk manifest `retries`/`retry_reasons`). Unavailable history is *not* stored as an (immutable) empty chunk. |
| Warming | `acquire_range` is newest-first by default (recent days, then backwards in day chunks); verified chunks are never re-downloaded (demonstrated: second run `requests=0`). |
| Bars check | `assess_bar_completeness`: every tick-minute (excluding the forming one) should have a bar; PASS >= 90 %, WARN >= 50 %, FAIL below, INCONCLUSIVE if < 10 tick-minutes; 3 attempts with 3 s pauses (cold bar build). |
| Request cap probe | Nested windows `[end - w, end]` with `end <= latest known tick`; a cap is only SUSPECTED if a larger window returns fewer rows than the smaller window plus the separately requested extra segment; otherwise `UNDETERMINED`. |
| Time policy | `TimeBaseSpec` carries a validity period (`valid_from/to_utc_ms`), `dst_status`, and the weekly-open consistency evidence; an immutable `timecal/1` record (broker, server, effective period, offset, samples, residual, MAD, DST status, confidence, calibration time) is stored in `data/metadata/timebase/<broker>/<server>/records/`. Instants outside the validity period are converted best-effort, **kept**, and tagged `TIME_BASIS_UNCERTAIN`. Raw tables now store `normalization_rule` and `time_basis_id` per row (`tick_raw/2`). |
| Weekly-open probe bug (found on the real terminal) | `copy_ticks_from(Saturday before history)` returns the first tick of *all* history; four earlier Saturdays therefore produced the same bogus "open" (2026-03-31 22:49:48 UTC) and a false "offset inconsistent". Opens later than 72 h after the probe instant are now rejected and recorded. Regression test included. |
| Flags | Raw `flags` untouched; new `unknown_flag_bits` column (`tick_quality/2`) and histogram; `UNKNOWN_FLAG_BITS` is informational, never quarantine. |
| Duplicates | `DUP_TIMESTAMP` (same ms) and `DUP_TICK` (same ms **and** identical content, adjacent) are separate tags, neither quarantines; stable broker order and `seq` preserved; summary reports `duplicate_timestamp_distinct_content`. |
| Stale / future | `assess_latest_tick_freshness` (stale limit 120 s, future tolerance 2 s + call latency + clock residual) in the harness; chunk-level `FUTURE_TICK` (normalized UTC later than ingestion + tolerance; quarantined). |
| History depth | `discover_history_depth`: one probe per week walking backwards (also warms history), first empty week confirmed by a second probe, then <= 4 weekday probes, then stop; errors are never read as a boundary; result cached with its observation date (`data/metadata/history_depth/...`). |

## Real DEMO measurements after hardening (report `reports/mt5_verify_20261007T192307Z.json`)
- Account class DEMO (gate unchanged). No request timeouts or retries occurred in this run (history was already warm).
- Weekly opens: 26 genuine opens 2026-04-05 .. 2026-10-04, source time-of-week constant within 123 s, step between halves -1 s,
  0 outliers, 4 earlier Saturdays rejected as "no data that week". => the same +10800 s offset applies across that whole period
  *if* the open is anchored to a fixed UTC instant (inference). Open observed ~22:00 UTC Sunday.
- Calibration record `timecal-c9cc2a3a65a3dd80`: offset +10800 s, 60 samples, residual 0.55 s, MAD 0.15 s, effective
  2026-04-05T22:00:02Z .. 2026-10-07T19:23:06Z, confidence **high**, `dst_status = unresolved_no_us_dst_transition_in_observed_window`.
  DST remains unresolved: the window (Apr-Oct) contains no US DST transition (March 8 and Nov 1, 2026 fall outside it).
- Tick history begins 2026-03-31 22:49:48 UTC (source day 2026-04-01); the empty weeks before it were confirmed by two probes each
  (depth discovery, 33 calls). Anything before 2026-04-05 22:00 UTC is therefore TIME_BASIS_UNCERTAIN under the current record.
- Request cap: UNDETERMINED (no truncation up to 1,442,020 rows per request, 3-day window).
- Bars: 60 bars, 59/60 tick-minutes covered (PASS). Latest tick FRESH (age 0.16 s). DOM unavailable.

## One-day dry run (source day 2026-10-06, Mon 22:02 UTC .. Tue 20:58 UTC)
| item | result |
|---|---|
| raw ticks fetched / stored | 433,422 / 433,422 (1 request, 0 clipped, 0 retries) |
| quarantined | 0 |
| tagged | DUP_TIMESTAMP 699 (1,386 rows share a ms), EXTREME_SPREAD 1, UNKNOWN_FLAG_BITS 433,422 |
| duplicate timestamp / exact duplicate tick | 699 consecutive same-ms ticks / 0 adjacent identical (2 non-adjacent identical copies in the day) |
| flags (raw) | 0x486: 423,287; 0x482: 5,370; 0x404: 4,765. Unknown bits: **0x400** (all ticks) and **0x80** (428,657 ticks) - both undocumented, preserved, uninterpreted |
| spread (points; point = 0.01) | min 5, p25 6, median 9, p75 10, p95 12, p99 12, p99.9 40, max 195, mean 8.54 |
| tick rate | mean 5.25 ticks/s; per-second p50 4, p90 12, p99 20, max 60; inter-tick ms p50 69, p90 483, p99 1,655, max 27,308; hourly 5.6k (open) .. 31.8k (14:00 UTC) |
| time anomalies | 0 reversals, 0 ambiguous/nonexistent, 0 uncertain rows, 0 future ticks, 0 sec/msec mismatches |
| gaps | none > 60 s (largest 27.3 s) |
| bid/ask | 0 bid > ask, 0 zero spread, 0 missing; `last` and `volume` always 0 |
| Parquet | raw 3.47 MB, tick_basic 2.80 MB, tick_quality 0.48 MB (6.7 MB total, ~8 bytes/tick raw) |
| acquisition cost | 2.9 s wall, 272 MB peak RSS; resume run 1.8 s, `requests=0` |
| Phase 2A features | 82,616 rows (1 s grid), 78,116 valid after warm-up, 119 features, validation OK (0 errors, 1 warning: `session_so_far_complete` constant on a single day); 6.1 s wall, 388 MB peak RSS; regime calibration still placeholder (not started) |
| prefix invariance (real data, normalized UTC axis) | PASS: 31,387,872 row-values compared over 5 cutoffs, 0 mismatches (10 s) |
| chunk verification | `verify_chunk` ok; dataset id `tickraw-08967bcc98b6d6de01f2`; ml dataset `mlds-7a1cd4133a92d75c7578` |

Extrapolation (not a measurement): ~6 months of weekdays at this size is on the order of 130 trading days x ~6.7 MB = ~0.9 GB
of Parquet and a few minutes of warm retrieval, but cold-history downloads (36-95 s each observed) are the real cost.

## Open items before/while collecting the full history
1. DST unresolved; records outside 2026-04-05 22:00 UTC .. calibration time are tagged TIME_BASIS_UNCERTAIN. Re-calibrate (and re-run
   weekly-open evidence) after 2026-11-01 to extend/resolve; a new record gets a new id.
2. Cold-history behaviour (36-95 s, failures at depth) is handled by soft timeouts + health check + bounded retry, but was only
   exercised against the fake in this phase (the real terminal's history was already warm). Expect and review retry events in the first
   backwards run, preferably in small batches.
3. Flags 0x400 and 0x80: semantics unknown; features must not interpret them.
4. Single-day sample; weekend, rollover, news and thin-liquidity behaviour not yet profiled.
