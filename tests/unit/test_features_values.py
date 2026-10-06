"""Features checked against independent hand/numpy calculations on tiny deterministic inputs."""
import datetime as dt

import numpy as np
import pytest

from fxscalp.features.pipeline import PipelineConfig, build_feature_frame
from tests.unit.features_helpers import ms, norm, raw, steady

T0 = ms(2025, 10, 6, 9, 0)


@pytest.fixture(scope="module")
def ff():
    a = steady(T0, 400, step=0.10)
    return build_feature_frame(norm(a), PipelineConfig(), point=0.01), a


def row(ff_, sec):
    """feature row for the decision at T0 + sec seconds (bar [sec-1, sec))."""
    return ff_[0].df.set_index("timestamp_utc_ms").loc[T0 + sec * 1000]


def test_rows_are_decision_times_at_bar_end(ff):
    d = ff[0].df
    assert d["timestamp_utc_ms"].iloc[0] == T0 + 1000 and d["timestamp_utc_ms"].iloc[-1] == T0 + 399 * 1000   # last bar dropped (incomplete)
    assert (np.diff(d["timestamp_utc_ms"]) == 1000).all()


def test_tick_counts_rates_and_intervals(ff):
    r = row(ff, 30)
    assert r.tick_count_5s == 10 and r.tick_count_15s == 30 and r.tick_count_60s != r.tick_count_60s     # 60s window not yet full
    assert r.tick_rate_5s == 2.0 and r.tick_acceleration_5s == 0.0
    assert r.tick_interval_mean_ms_5s == 500 and r.empty_interval == 0
    assert row(ff, 100).tick_count_60s == 120


def test_directional_features_for_a_pure_uptrend(ff):
    r = row(ff, 30)
    assert r.up_tick_ratio_5s == 1.0 and r.down_tick_ratio_5s == 0.0 and r.tick_direction_imbalance_5s == 1.0
    assert r.directional_persistence_5s == 1.0
    assert r.consecutive_up_ticks == 2 * 30 - 1 and r.consecutive_down_ticks == 0


def test_spread_units_use_point_from_metadata():
    a = steady(T0, 100)
    f = build_feature_frame(norm(a, point=0.01), PipelineConfig(), point=0.01)
    r = f.df.set_index("timestamp_utc_ms").loc[T0 + 50_000]
    assert r.spread_points == pytest.approx(25.0)                      # 0.25 / 0.01
    assert r.relative_spread_bps == pytest.approx(1e4 * 0.25 / (2000 + 0.1 * 99), rel=1e-3)
    assert r.spread_mean_points_15s == pytest.approx(25.0)
    g = build_feature_frame(norm(a, point=0.001), PipelineConfig(), point=0.001)       # a different point changes the unit
    assert g.df.set_index("timestamp_utc_ms").loc[T0 + 50_000].spread_points == pytest.approx(250.0)


def test_unknown_point_gives_nan_points_not_a_guess():
    a = steady(T0, 100)
    f = build_feature_frame(norm(a, point=None), PipelineConfig(), point=None)
    r = f.df.set_index("timestamp_utc_ms").loc[T0 + 50_000]
    assert np.isnan(r.spread_points) and np.isnan(r.spread_mean_points_15s) and r.relative_spread_bps > 0


def test_returns_and_velocity_match_numpy(ff):
    f, a = ff
    mid = (a["bid"] + a["ask"]) / 2
    # as-of mid at the end of second s = mid of the last tick in that second (index 2s+1)
    asof = lambda s: mid[2 * (s - 1) + 1]                           # decision at T0+s => last bar [s-1, s) => ticks up to index 2(s-1)+1
    r = row((f, a), 100)
    assert r.ret_bps_5s == pytest.approx(1e4 * np.log(asof(100) / asof(95)))
    assert r.ret_bps_60s == pytest.approx(1e4 * np.log(asof(100) / asof(40)))
    assert r.abs_ret_bps_15s == pytest.approx(abs(r.ret_bps_15s))
    assert r.price_velocity_bps_per_s_15s == pytest.approx(r.ret_bps_15s / 15)
    assert np.isnan(r.momentum_alignment)                                # needs ALL horizons incl. 300 s (warm-up)
    assert row((f, a), 350).momentum_alignment == 1.0


def test_realized_vol_matches_numpy(ff):
    f, a = ff
    mid = ((a["bid"] + a["ask"]) / 2).astype("float64")
    lr = np.diff(np.log(mid))                                          # return of tick j (j>=1) vs j-1
    r = row((f, a), 100)
    last_tick = 2 * 99 + 1                                             # last tick before T0+100s
    window = lr[last_tick - 30:last_tick]                              # 15 s = 30 ticks: ticks last_tick-29 .. last_tick
    assert r.realized_vol_bps_15s == pytest.approx(1e4 * np.sqrt((window ** 2).sum()))


def test_range_features(ff):
    f, a = ff
    mid = ((a["bid"] + a["ask"]) / 2)
    r = row((f, a), 200)
    last = 2 * 199 + 1
    hi, lo, m = mid[last - 59:last + 1].max(), mid[last - 59:last + 1].min(), mid[last]      # 30 s = 60 ticks
    assert r.range_width_bps_30s == pytest.approx(1e4 * (hi - lo) / m)
    assert r.range_position_30s == pytest.approx(1.0) and r.dist_high_bps_30s == pytest.approx(0.0, abs=1e-9)
    assert r.dist_low_bps_30s == pytest.approx(1e4 * (m - lo) / m)
    assert r.compression_ratio_30s_300s != r.compression_ratio_30s_300s          # 300s window not full at 200s (warm-up)
    assert row((f, a), 350).compression_ratio_30s_300s == pytest.approx(
        row((f, a), 350).range_width_bps_30s / row((f, a), 350).range_width_bps_300s)


def test_breakout_distance_excludes_the_most_recent_guard_seconds(ff):
    f, a = ff
    r = row((f, a), 200)
    assert r.breakout_dist_up_bps_60s > 0                              # still rising: above the high that ended 5 s ago
    assert r.breakout_dist_down_bps_60s < 0
    assert r.failed_breakout_up_300s_30s != r.failed_breakout_up_300s_30s or r.failed_breakout_up_300s_30s == 0


def test_failed_breakout_flag_on_a_spike_and_retrace():
    n = 700
    mid = np.full(n * 2, 2000.0)
    mid += np.sin(np.arange(n * 2) / 50) * 0.2                         # small oscillation: defines the prior range
    spike = slice(2 * 400, 2 * 410)
    mid[spike] += 3.0                                                   # break out above the range for 10 s ...
    ts = (T0 + (np.arange(n)[:, None] * 1000 + np.array([100, 600])[None, :])).ravel()
    f = build_feature_frame(norm(raw(ts, mid)), PipelineConfig(), point=0.01)
    d = f.df.set_index("timestamp_utc_ms")
    during = d.loc[T0 + 405_000, "failed_breakout_up_300s_30s"]
    after = d.loc[T0 + 420_000, "failed_breakout_up_300s_30s"]
    assert during == 0 and after == 1                                    # ... and is back below the old high 10 s later
    assert d.loc[T0 + 460_000, "failed_breakout_up_300s_30s"] == 0       # the 30 s memory has expired


def test_missing_interval_is_explicit_in_features():
    ts = np.concatenate([T0 + np.arange(0, 100_000, 500), T0 + np.arange(160_000, 200_000, 500)])
    a = raw(ts, 2000 + 0.1 * np.arange(len(ts)))
    f = build_feature_frame(norm(a), PipelineConfig(), point=0.01)
    d = f.df.set_index("timestamp_utc_ms")
    gap_row = d.loc[T0 + 130_000]
    assert gap_row.empty_interval == 1 and gap_row.tick_count_5s == 0 and gap_row.staleness_s == pytest.approx(30.5, abs=1)
    assert np.isnan(gap_row.up_tick_ratio_5s) and np.isnan(gap_row.tick_interval_mean_ms_5s)      # 0/0 stays NaN, no fill
    assert gap_row.ret_bps_5s == 0.0                                      # as-of price unchanged (explicit state)
    assert d.loc[T0 + 165_000].empty_interval == 0


def test_session_clock_features_follow_local_time_and_dst():
    for day, london_open_utc_h in ((dt.datetime(2025, 7, 16), 7), (dt.datetime(2025, 1, 15), 8)):
        start = int(day.replace(hour=london_open_utc_h - 1, tzinfo=dt.timezone.utc).timestamp() * 1000)   # 1 h before London open
        a = steady(start, 7500, step=0.01)
        d = build_feature_frame(norm(a), PipelineConfig(), point=0.01).df.set_index("timestamp_utc_ms")
        r = d.loc[start + 3600_000 + 600_000]                              # 10 minutes after the open
        assert r.session_code in (1, 3) and r.is_london == 1
        assert r.minutes_since_session_open == pytest.approx(10.0 if r.session_code == 1 else 10.0 + 0, abs=0.01) or r.session_code == 3
        assert r.minutes_until_session_close == pytest.approx(9 * 60 - 10, abs=0.01) or r.session_code == 3
        assert r.session_so_far_complete == 1


def test_session_so_far_is_partial_when_data_starts_mid_session():
    start = ms(2025, 7, 16, 9, 30)                                         # London already open since 07:00 UTC
    d = build_feature_frame(norm(steady(start, 600)), PipelineConfig(), point=0.01).df
    assert (d["session_so_far_complete"] == 0).all()


def test_session_high_so_far_only_uses_the_past():
    start = ms(2025, 7, 16, 6, 55)
    mid = np.concatenate([np.linspace(2000, 2010, 1200), np.linspace(2010, 1990, 1200)])     # peak after 600 s (2 ticks/s)
    ts = (start + (np.arange(1200)[:, None] * 1000 + np.array([100, 600])[None, :])).ravel()
    d = build_feature_frame(norm(raw(ts, mid)), PipelineConfig(), point=0.01).df.set_index("timestamp_utc_ms")
    early = d.loc[start + 400_000]                                          # 07:01:40, rising, London open since 07:00
    assert early.session_high_so_far == pytest.approx(mid[2 * 400 - 1], abs=0.05) and early.session_high_so_far < 2007   # NOT the final high (2010)
    late = d.loc[start + 1190_000]
    assert late.session_so_far_complete == 1
    assert late.session_high_so_far == pytest.approx(2010, abs=0.02) and late.dist_session_high_bps > 90


def test_atr_uses_completed_bars_only():
    a = steady(T0, 1500, step=0.05)
    d = build_feature_frame(norm(a), PipelineConfig(), point=0.01).df.set_index("timestamp_utc_ms")
    assert np.isnan(d.loc[T0 + 13 * 15_000, "atr14_bps_15s"])                # only 13 complete 15 s bars so far
    assert d.loc[T0 + 14 * 15_000, "atr14_bps_15s"] > 0                       # 14th bar completes exactly at +210 s
    assert d.loc[T0 + 14 * 15_000 - 1000, "atr14_bps_15s"] != d.loc[T0 + 14 * 15_000 - 1000, "atr14_bps_15s"]
