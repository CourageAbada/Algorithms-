import datetime as dt

from fxscalp.brokers.mt5.fake import FakeScenario
from fxscalp.market_data.continuity import EdgeTick, check_boundary, check_continuity, classify_gap
from tests.unit.test_dataset_tooling import build

MS = lambda y, m, d, h=0, mi=0, s=0: int(dt.datetime(y, m, d, h, mi, s, tzinfo=dt.timezone.utc).timestamp() * 1000)  # noqa: E731


def edge(day, utc_ms, offset_s=10800, bid=100.0, ask=100.1, basis="tb-x", unc=False, flags=0x486):
    return EdgeTick(day, utc_ms + offset_s * 1000, utc_ms, bid, ask, 0.0, 0, flags, basis, unc)


def test_gap_classification_matches_documented_structure():
    t = MS(2026, 6, 10, 20, 59)
    assert classify_gap(3661, t) == "daily_break" and classify_gap(3300, t) == "daily_break"
    assert classify_gap(49.05 * 3600, MS(2026, 6, 5, 20, 59)) == "weekly_closure"
    assert classify_gap(73 * 3600, MS(2026, 4, 2, 20, 59)) == "weekly_closure"
    assert classify_gap(12_600, MS(2026, 5, 25, 18, 30)) == "holiday_schedule"      # US holiday early close
    assert classify_gap(12_600, MS(2026, 5, 26, 18, 30)) == "UNEXPECTED"            # same gap on an ordinary day
    assert classify_gap(900, t) == "UNEXPECTED" and classify_gap(10 * 3600, t) == "UNEXPECTED"


def test_normal_daily_break_and_weekend_are_not_problems():
    a = edge("2026-06-09", MS(2026, 6, 9, 20, 58, 59))
    assert check_boundary(a, edge("2026-06-10", MS(2026, 6, 9, 22, 0, 1))).problems == []
    fri = edge("2026-06-05", MS(2026, 6, 5, 20, 58, 58))
    r = check_boundary(fri, edge("2026-06-08", MS(2026, 6, 7, 22, 0, 2)))
    assert r.kind == "weekly_closure" and r.problems == [] and not r.time_basis_transition


def test_ordering_duplicates_offset_shift_and_basis_transition_are_detected():
    a = edge("2026-06-09", MS(2026, 6, 9, 20, 58))
    back = check_boundary(a, edge("2026-06-10", MS(2026, 6, 9, 20, 57)))
    assert back.kind == "ORDER_VIOLATION" and not back.normalized_ordered and back.problems
    dup = check_boundary(a, edge("2026-06-10", MS(2026, 6, 9, 20, 58)))
    assert dup.duplicate_tick and dup.duplicate_timestamp and any("identical tick" in p for p in dup.problems)
    same_ms_diff = check_boundary(a, edge("2026-06-10", MS(2026, 6, 9, 20, 58), bid=101.0))
    assert same_ms_diff.duplicate_timestamp and not same_ms_diff.duplicate_tick          # timestamp != tick duplicate
    shifted = check_boundary(a, edge("2026-06-10", MS(2026, 6, 9, 22, 0), offset_s=7200))
    assert abs(shifted.offset_shift_s) == 3600 and any("normalisation shifts" in p for p in shifted.problems)
    tr = check_boundary(a, edge("2026-06-10", MS(2026, 6, 9, 22, 0), basis="tb-y", unc=True))
    assert tr.time_basis_transition and tr.problems == []                                # a transition is reported, not an error


def test_check_continuity_on_a_store_skips_closed_days_and_flags_contiguous_data(tmp_path):
    _, st, man = build(tmp_path, days=7, scenario=FakeScenario(ticks_per_day=3000))
    res = check_continuity(st, man)
    trading = sum(c["record_count"] > 0 for c in man["chunks"])
    assert res["n_boundaries"] == trading - 1 and res["ordering_violations"] == 0
    assert res["duplicate_ticks_across_boundary"] == 0 and res["offset_shifts_across_boundary"] == 0
    # the fake feed trades continuously across midnight (no maintenance break): that is NOT an expected structure
    assert res["kinds"].get("UNEXPECTED", 0) >= 1 and not res["ok"]
