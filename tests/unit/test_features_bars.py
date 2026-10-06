import numpy as np
import pytest

from fxscalp.features.bars import aggregate_bars, align_completed, take, tick_derived
from tests.unit.features_helpers import ms, raw, norm, steady


def bars_of(ts, mid, tf, **kw):
    a = raw(ts, mid)
    return aggregate_bars(a["time_msc"], a["bid"], a["ask"], a["volume"], tf, **kw)


# ---------------- boundaries ----------------
def test_boundary_tick_belongs_to_the_new_bar():
    b = bars_of([0, 999, 1000, 1999, 2000], [10, 11, 12, 13, 14], 1)
    assert b["n_ticks"].tolist() == [2, 2, 1]
    assert b["mid_close"].tolist()[:2] == [11.0, 13.0] and b["mid_open"].tolist()[1] == 12.0
    assert (b["start_ms"].to_numpy() % 1000 == 0).all()


@pytest.mark.parametrize("tf", [1, 5, 15, 30, 60, 300, 900])
def test_all_timeframes_are_utc_aligned_and_contiguous(tf):
    base = ms(2025, 10, 6, 9, 0)
    a = steady(base, 4000)
    b = aggregate_bars(a["time_msc"], a["bid"], a["ask"], a["volume"], tf)
    assert (b["start_ms"].to_numpy() % (tf * 1000) == 0).all()
    assert (np.diff(b["start_ms"].to_numpy()) == tf * 1000).all() and (b["end_ms"] - b["start_ms"] == tf * 1000).all()
    assert b["n_ticks"].sum() == len(a)


def test_missing_intervals_are_explicit_not_forward_filled():
    b = bars_of([0, 500, 5200, 5300], [10, 10.1, 11, 11.1], 1)
    assert len(b) == 6 and b["empty"].tolist() == [False, True, True, True, True, False]
    e = b[b["empty"]]
    assert (e["n_ticks"] == 0).all() and e[["mid_open", "mid_high", "mid_low", "mid_close", "spread_mean", "bid_close"]].isna().all().all()


def test_same_timestamp_ticks_keep_seq_order():
    b = bars_of([1000, 1000, 1000, 1500], [10, 20, 15, 16], 1)
    assert b.loc[0, "n_ticks"] == 4 and b.loc[0, "mid_open"] == 10 and b.loc[0, "mid_close"] == 16
    assert b.loc[0, "mid_high"] == 20 and b.loc[0, "mid_low"] == 10


def test_aggregation_requires_ordered_ticks_and_normalisation_fixes_it():
    with pytest.raises(ValueError):
        bars_of([1000, 500], [10, 11], 1)
    a = raw([1000, 2000, 1500, 3000], [10, 11, 12, 13])             # 3rd tick goes back in time
    df = norm(a)                                                    # quality quarantines the reversal; the rest stays ordered
    assert len(df) == 3 and (np.diff(df["timestamp_utc_ms"]) >= 0).all()


def test_last_bar_of_a_dataset_is_not_complete_but_earlier_ones_are():
    b = bars_of([0, 500, 1500, 2500], [1, 2, 3, 4], 1)
    assert b["complete"].tolist() == [True, True, False]
    assert bars_of([0, 500, 1500, 2500], [1, 2, 3, 4], 1, data_continues_after=True)["complete"].all()


def test_bar_ohlc_spread_and_micro_columns():
    a = raw([0, 100, 200, 300], [10.0, 10.2, 10.1, 10.4], spread=0.2)
    b = aggregate_bars(a["time_msc"], a["bid"], a["ask"], a["volume"], 1, micro=True, data_continues_after=True)
    r = b.iloc[0]
    assert (r.mid_open, r.mid_high, r.mid_low, r.mid_close) == (10.0, 10.4, 10.0, 10.4)
    assert r.bid_open == pytest.approx(9.9) and r.ask_close == pytest.approx(10.5) and r.spread_mean == pytest.approx(0.2)
    assert (r.n_up, r.n_down, r.n_flat, r.n_ticks) == (2, 1, 0, 4)       # first tick has no predecessor
    assert r.sum_dt == 300 and r.n_dt == 3 and r.run_close == 1           # up,down,up -> current run is 1 up
    assert r.realized_vol == pytest.approx(np.sqrt(np.log(10.2 / 10) ** 2 + np.log(10.1 / 10.2) ** 2 + np.log(10.4 / 10.1) ** 2))


def test_tick_derived_runs_ignore_flat_ticks():
    mid = np.array([1, 2, 3, 3, 4, 3, 2, 2, 2, 1.0])
    d = tick_derived(np.arange(10) * 100, mid - 0.1, mid + 0.1, mid)
    assert d["run_len"].tolist() == [0, 1, 2, 2, 3, -1, -2, -2, -2, -3]
    assert d["same_dir"].sum() == 4 and d["opp_dir"].sum() == 1


# ---------------- alignment (anti look-ahead) ----------------
def test_alignment_only_sees_completed_bars():
    b = bars_of([0, 500, 999, 1000, 1000, 2500, 5000, 5001], [1, 2, 3, 4, 5, 6, 7, 8], 1)
    idx, age = align_completed(np.array([999, 1000, 2999, 3000, 5000, 6000]), b)
    assert idx.tolist() == [-1, 0, 1, 2, 4, 4]           # bar containing t is never visible; empty bar 4 stays empty
    assert take(b["mid_close"].to_numpy(), idx)[4] != take(b["mid_close"].to_numpy(), idx)[4]     # NaN (explicit empty)


def test_regression_5m_bar_final_ohlc_not_visible_at_10_03_20():
    """At 10:03:20 the final OHLC of the 10:00-10:05 bar must NOT be used (look-ahead)."""
    from fxscalp.features.pipeline import PipelineConfig, build_feature_frame
    t0 = ms(2025, 10, 6, 9, 0)
    a = steady(t0, 4000, step=0.001)                                  # 09:00 .. 10:06:40, gentle drift
    spike_from, spike_to = ms(2025, 10, 6, 10, 4, 0), ms(2025, 10, 6, 10, 4, 30)
    m = (a["time_msc"] >= spike_from) & (a["time_msc"] < spike_to)
    for f in ("bid", "ask"):
        a[f][m] += 50.0                                               # violent move inside the 10:00-10:05 bar
    ff = build_feature_frame(norm(a), PipelineConfig(), point=0.01)
    df = ff.df.set_index("timestamp_utc_ms")
    t_before = ms(2025, 10, 6, 10, 3, 20)
    t_after = ms(2025, 10, 6, 10, 5, 0)
    calm = df.loc[t_before, "htf_range_bps_300s"]
    assert calm < 20                                                  # = the 09:55-10:00 bar range (calm)
    assert df.loc[ms(2025, 10, 6, 10, 4, 59), "htf_range_bps_300s"] == calm      # still the OLD bar one second before it completes
    assert df.loc[t_after, "htf_range_bps_300s"] > 100                # visible only once the bar is complete
    assert df.loc[t_before, "htf_ret_bps_300s"] == df.loc[ms(2025, 10, 6, 10, 4, 59), "htf_ret_bps_300s"]
    # the 1m/30s/5s bars follow the same rule
    assert df.loc[ms(2025, 10, 6, 10, 4, 10), "htf_range_bps_60s"] < 20      # 10:03-10:04 bar is calm; 10:04 bar still open
    assert df.loc[ms(2025, 10, 6, 10, 5, 0), "htf_range_bps_60s"] > 100      # 10:04-10:05 bar complete now
