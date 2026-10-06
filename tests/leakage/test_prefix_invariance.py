"""CORE anti-look-ahead test: features at time t must not change when later data is appended."""
import datetime as dt

import numpy as np
import pytest

from fxscalp.features.leakage import assert_prefix_invariant, check_prefix_invariance
from tests.leakage.helpers import ANOMALY_CFG, cutoffs, make_compute, scenario

UTC = dt.timezone.utc


def _run(df, fracs=(0.15, 0.37, 0.52, 0.8, 0.97)):
    comp = make_compute()
    return assert_prefix_invariant(comp, df, cutoffs(len(df), fracs), ts_col="seq", key_col="timestamp_utc_ms")


def test_prefix_invariance_london_day_with_all_anomalies():
    df = scenario(start_utc=dt.datetime(2025, 10, 6, 6, 0, tzinfo=UTC), duration_s=3 * 3600, base_rate_hz=1.2, **ANOMALY_CFG)
    rep = _run(df)
    assert rep.ok and rep.cutoffs_checked == 5 and rep.rows_compared > 100_000 and rep.exceptions_used == {}


def test_prefix_invariance_across_new_york_open_and_overlap_us_dst():
    df = scenario(start_utc=dt.datetime(2025, 10, 6, 11, 30, tzinfo=UTC), duration_s=3 * 3600, base_rate_hz=1.0, seed=3)
    _run(df, (0.2, 0.45, 0.6, 0.9))


def test_prefix_invariance_across_a_long_gap_two_segments():
    df = scenario(start_utc=dt.datetime(2025, 10, 7, 8, 0, tzinfo=UTC), duration_s=3 * 3600, base_rate_hz=1.0,
                  outages=((3000, 2400),), seed=5)                       # 40 min outage > segment gap -> 2 segments
    comp = make_compute()
    full = comp(df)
    assert full["segment_id"].nunique() == 2
    n = len(df)
    gap_at = int(np.searchsorted(df["time_msc"].to_numpy(), df["time_msc"].iloc[0] + 3300 * 1000))      # inside the outage
    assert_prefix_invariant(comp, df, [gap_at, gap_at + 5000, int(n * 0.9)], ts_col="seq", key_col="timestamp_utc_ms", full=full)


@pytest.mark.slow
def test_prefix_invariance_over_a_weekend_closure():
    df = scenario(start_utc=dt.datetime(2025, 10, 10, 19, 30, tzinfo=UTC), duration_s=52 * 3600, base_rate_hz=0.2, seed=9)
    comp = make_compute()
    full = comp(df)
    assert full["segment_id"].nunique() == 2                              # Friday close .. Sunday open
    gap = full.groupby("segment_id")["timestamp_utc_ms"].agg(["min", "max"])
    assert gap.loc[1, "min"] - gap.loc[0, "max"] > 40 * 3600 * 1000         # no rows inside the closure
    n = len(df)
    assert_prefix_invariant(comp, df, [int(n * 0.05), int(n * 0.12), int(n * 0.5)], ts_col="seq", key_col="timestamp_utc_ms", full=full)


def test_harness_reports_rows_and_cutoffs_and_documents_exceptions():
    df = scenario(duration_s=2 * 3600, base_rate_hz=0.5)
    rep = check_prefix_invariance(make_compute(), df, [len(df) // 2], ts_col="seq", key_col="timestamp_utc_ms",
                                  exceptions={"session_so_far_complete": "documented example only"})
    assert rep.ok and rep.exceptions_used == {"session_so_far_complete": "documented example only"}
