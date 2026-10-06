import numpy as np
import pytest

from fxscalp.market_data.quality import Q, QualityConfig, assess, PrevTick, QUARANTINE_MASK
from tests.unit.conftest import make_ticks


def run(a, **kw):
    return assess(a["time"], a["time_msc"], a["bid"], a["ask"], a["last"], a["volume"], a["flags"], **kw)


def has(r, i, flag):
    return bool(int(r.flags[i]) & int(flag))


def test_clean_data_has_no_flags_and_nothing_is_dropped():
    a = make_ticks(500)
    r = run(a)
    assert len(r.flags) == 500 and r.summary["clean"] >= 499 and r.summary["quarantined"] == 0


def test_negative_spread_quarantined():
    a = make_ticks(10)
    a["ask"][4] = a["bid"][4] - 0.5
    r = run(a)
    assert has(r, 4, Q.NEGATIVE_SPREAD) and r.quarantined[4] and r.quarantined.sum() == 1


def test_zero_spread_tagged_not_quarantined():
    a = make_ticks(10)
    a["ask"][3] = a["bid"][3]
    r = run(a)
    assert has(r, 3, Q.ZERO_SPREAD) and not r.quarantined[3]


@pytest.mark.parametrize("field,flag", [("bid", Q.MISSING_BID), ("ask", Q.MISSING_ASK)])
def test_missing_bid_ask(field, flag):
    a = make_ticks(10)
    a[field][2] = 0.0
    a[field][5] = np.nan
    r = run(a)
    assert has(r, 2, flag) and has(r, 5, flag) and r.quarantined[2] and r.quarantined[5]
    assert has(r, 5, Q.NONFINITE)


def test_extreme_spread_tagged_relative_to_batch_median():
    a = make_ticks(100)
    a["ask"][50] = a["bid"][50] + 50.0
    r = run(a)
    assert has(r, 50, Q.EXTREME_SPREAD) and not r.quarantined[50]


def test_duplicate_timestamp_and_duplicate_tick():
    a = make_ticks(10)
    a[4] = a[3]                                    # identical in every field
    a["bid"][7] += 0.01                            # same timestamp as previous, different price
    a["time_msc"][7] = a["time_msc"][6]
    a["time"][7] = a["time"][6]
    r = run(a)
    assert has(r, 4, Q.DUP_TICK) and has(r, 4, Q.DUP_TIMESTAMP)
    assert has(r, 7, Q.DUP_TIMESTAMP) and not has(r, 7, Q.DUP_TICK)


def test_timestamp_reversal_quarantined():
    a = make_ticks(10)
    a["time_msc"][6] = a["time_msc"][5] - 3000
    a["time"][6] = a["time_msc"][6] // 1000
    r = run(a)
    assert has(r, 6, Q.TIME_REVERSAL) and r.quarantined[6]


def test_large_gap_tagged_and_listed():
    a = make_ticks(10)
    a["time_msc"][5:] += 7 * 3600 * 1000
    a["time"] = a["time_msc"] // 1000
    r = run(a, cfg=QualityConfig(large_gap_s=300, long_gap_s=3600))
    assert has(r, 5, Q.LARGE_GAP) and r.summary["n_long_gaps"] == 1 and r.summary["long_gaps"][0]["gap_s"] > 25000


def test_second_vs_millisecond_mismatch():
    a = make_ticks(5)
    a["time"][2] += 10
    assert has(run(a), 2, Q.SEC_MSEC_MISMATCH)


def test_unknown_flag_bits_and_histogram():
    a = make_ticks(5)
    a["flags"][1] = 0x800
    r = run(a)
    assert has(r, 1, Q.UNKNOWN_FLAG_BITS) and r.summary["tick_flag_histogram"]["0x800"] == 1


def test_time_ambiguity_flags_propagate():
    a = make_ticks(5)
    amb = np.array([False, True, False, False, False])
    non = np.array([False, False, True, False, False])
    r = run(a, ambiguous=amb, nonexistent=non)
    assert has(r, 1, Q.TIME_AMBIGUOUS) and not r.quarantined[1] and has(r, 2, Q.TIME_NONEXISTENT) and r.quarantined[2]


def test_previous_chunk_context_used_for_first_tick():
    a = make_ticks(3)
    prev = PrevTick(int(a["time_msc"][0]) + 10_000, 1.0, 1.1, 0.0, 0, 6)       # previous tick is AFTER the first
    r = run(a, prev=prev)
    assert has(r, 0, Q.TIME_REVERSAL)
    assert "not checked" in run(a).summary["first_tick_context"]


def test_quarantine_mask_definition():
    assert QUARANTINE_MASK & int(Q.NEGATIVE_SPREAD) and not QUARANTINE_MASK & int(Q.ZERO_SPREAD)


def test_empty_input():
    a = make_ticks(0)
    assert run(a).summary["n"] == 0
