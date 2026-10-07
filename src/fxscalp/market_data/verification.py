"""Pure, unit-testable checks used by the MT5 verification harness (no broker access here).

* ``assess_latest_tick_freshness``: stale / future latest tick, with configurable tolerances that account for the
  measured receive latency and clock residual.
* ``assess_bar_completeness``: is a bar response plausibly complete for a window in which ticks are known to exist?
  (A defensible, session-gap-tolerant check: bars are expected for the minutes that actually contain ticks; the
  count is NOT required to equal the window length.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------------------------------ freshness
@dataclass(frozen=True)
class FreshnessConfig:
    stale_s: float = 120.0             # a liquid instrument (XAU/USD) quoting less often than this while open is stale
    future_tolerance_s: float = 2.0    # base tolerance for a tick time ahead of the receive time


@dataclass(frozen=True)
class Freshness:
    status: str                        # "FRESH" | "STALE" | "FUTURE"
    age_s: float                       # receive_utc - tick_utc (negative = tick is in the future)
    stale_limit_s: float
    future_limit_s: float
    note: str


def assess_latest_tick_freshness(tick_time_msc: int, offset_s: int, receive_utc_s: float, *,
                                 call_latency_s: float = 0.0, clock_residual_s: float = 0.0,
                                 cfg: FreshnessConfig = FreshnessConfig()) -> Freshness:
    """Age of the latest tick versus the local receive time, using the measured source-UTC offset.

    The tolerances absorb the call latency (the tick may have arrived while the call was in flight) and the measured
    clock residual, so a tick that is a fraction of a second ahead is not mislabelled as 'future'.
    """
    tick_utc_s = tick_time_msc / 1000.0 - offset_s
    age = receive_utc_s - tick_utc_s
    future_limit = cfg.future_tolerance_s + max(0.0, call_latency_s) + abs(clock_residual_s)
    stale_limit = cfg.stale_s + max(0.0, call_latency_s)
    if age < -future_limit:
        return Freshness("FUTURE", age, stale_limit, future_limit,
                         f"tick is {-age:.1f}s ahead of the receive time (tolerance {future_limit:.1f}s): offset or "
                         "local clock is wrong")
    if age > stale_limit:
        return Freshness("STALE", age, stale_limit, future_limit,
                         f"latest tick is {age:.0f}s old (limit {stale_limit:.0f}s): market closed, feed stalled, or offset wrong")
    return Freshness("FRESH", age, stale_limit, future_limit, "within tolerances")


# ------------------------------------------------------------------------------------------ bars
@dataclass(frozen=True)
class BarCompleteness:
    status: str                        # "PASS" | "WARN" | "FAIL" | "INCONCLUSIVE"
    n_bars: int
    minutes_with_ticks: int
    minutes_covered: int
    coverage: float | None             # minutes_covered / minutes_with_ticks
    note: str


def assess_bar_completeness(bar_time_s: np.ndarray, tick_time_msc: np.ndarray, *, bar_seconds: int = 60,
                            min_minutes: int = 10, pass_ratio: float = 0.9, warn_ratio: float = 0.5) -> BarCompleteness:
    """Compare a bar response with the ticks of the same window (both in the SAME source-time domain).

    Every bar bucket that contains at least one tick should exist as a bar. Session gaps and quiet minutes are fine
    (no ticks -> no expectation). The still-forming last bucket is excluded from the expectation.
    """
    bars = np.unique(np.asarray(bar_time_s, dtype="int64") // bar_seconds)
    tick_min = np.unique((np.asarray(tick_time_msc, dtype="int64") // 1000) // bar_seconds)
    if tick_min.size:
        tick_min = tick_min[:-1] if tick_min.size > 1 else tick_min    # the last bucket may still be forming
    n_expected = int(tick_min.size)
    if n_expected < min_minutes:
        return BarCompleteness("INCONCLUSIVE", int(bars.size), n_expected, 0, None,
                               f"only {n_expected} tick-minutes in the window (< {min_minutes}): too little data to judge")
    covered = int(np.isin(tick_min, bars).sum())
    ratio = covered / n_expected
    if ratio >= pass_ratio:
        return BarCompleteness("PASS", int(bars.size), n_expected, covered, ratio, f"{covered}/{n_expected} tick-minutes have a bar")
    if ratio >= warn_ratio:
        return BarCompleteness("WARN", int(bars.size), n_expected, covered, ratio,
                               f"bars cover only {ratio:.0%} of tick-minutes: response may be incomplete")
    return BarCompleteness("FAIL", int(bars.size), n_expected, covered, ratio,
                           f"bars cover only {ratio:.0%} of tick-minutes ({covered}/{n_expected}): the response is "
                           "obviously incomplete (terminal may still be building bar history; retry)")
