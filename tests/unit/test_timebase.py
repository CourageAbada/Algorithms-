import datetime as dt

import numpy as np
import pytest

from fxscalp.brokers.errors import TimeCalibrationError
from fxscalp.market_data.timebase import (DstInference, OffsetSample, ServerTimeRule, TimeBase, TimeBasis, estimate_offset,
                                          infer_dst_from_opens, infer_dst_from_weekly_open, resolve_time_basis,
                                          unverified_spec)

NY7 = ServerTimeRule.zone_shift("America/New_York", 7 * 3600)


def utc_ms(*a):
    return int(dt.datetime(*a, tzinfo=dt.timezone.utc).timestamp() * 1000)


def samples(offset_s, n=12, age=0.4, t0=1_760_000_000.0, skew=0.0, jitter=0.0):
    rng = np.random.default_rng(1)
    out = []
    for i in range(n):
        local = t0 + i * 1.0
        msc = int((local + skew - age + offset_s + rng.normal(0, jitter)) * 1000)
        out.append(OffsetSample(local, msc // 1000, msc, 0.01))
    return out


# ---------------- offset estimation ----------------
@pytest.mark.parametrize("off", [0, 3 * 3600, 2 * 3600, -5 * 3600, 19800, -9 * 3600])
def test_offset_recovered_including_negative_and_half_hour(off):
    e = estimate_offset(samples(off))
    assert e.status == "ok" and e.offset_s == off and abs(e.residual_s) < 5


def test_stale_feed_is_inconclusive_not_a_wrong_offset():
    s = samples(3 * 3600)
    stale = [OffsetSample(x.local_utc_s, s[0].tick_time_s, s[0].tick_time_msc, 0.0) for x in s]   # same tick repeated
    e = estimate_offset(stale)
    assert e.status == "inconclusive" and e.offset_s is None and any("stale" in n or "closed" in n for n in e.notes)


def test_too_few_samples_inconclusive():
    assert estimate_offset(samples(0, n=3)).status == "inconclusive"


def test_clock_skew_detected_via_residual():
    e = estimate_offset(samples(3 * 3600, skew=200.0))
    assert e.status == "inconclusive" and "clock" in " ".join(e.notes)


def test_noisy_samples_inconclusive():
    assert estimate_offset(samples(3 * 3600, jitter=400.0)).status == "inconclusive"


def test_implausible_offset_inconclusive():
    assert estimate_offset(samples(20 * 3600)).status == "inconclusive"


def test_sec_msec_mismatch_is_reported():
    s = samples(0)
    s[2] = OffsetSample(s[2].local_utc_s, s[2].tick_time_s + 5, s[2].tick_time_msc, 0.0)
    assert estimate_offset(s).sec_msc_consistent is False


# ---------------- conversion ----------------
def test_fixed_rule_roundtrip():
    r = ServerTimeRule.fixed(3 * 3600)
    u = np.array([utc_ms(2025, 1, 1), utc_ms(2025, 7, 1)])
    src = u + r.offset_at_utc_ms(u) * 1000
    back, amb, non = r.source_to_utc_ms(src)
    assert (back == u).all() and not amb.any() and not non.any()


def test_zone_rule_follows_us_dst_and_roundtrips_everywhere():
    assert NY7.offset_at_utc_ms(np.array([utc_ms(2024, 1, 15, 12)]))[0] == 2 * 3600
    assert NY7.offset_at_utc_ms(np.array([utc_ms(2024, 7, 1, 12)]))[0] == 3 * 3600
    # the 3 weeks between EU and US DST changes: still US rule
    assert NY7.offset_at_utc_ms(np.array([utc_ms(2024, 3, 20, 12)]))[0] == 3 * 3600
    rng = np.random.default_rng(0)
    u = np.sort(rng.integers(utc_ms(2024, 1, 1), utc_ms(2025, 1, 1), 5000)) // 1000 * 1000
    src = u + NY7.offset_at_utc_ms(u) * 1000
    back, amb, non = NY7.source_to_utc_ms(np.sort(src))
    # instants outside the DST fold/gap windows convert exactly
    good = ~(amb | non)
    assert good.mean() > 0.99
    order = np.argsort(src)
    assert (back[good] == u[order][good]).all()


def test_dst_fold_resolved_from_observed_order():
    start = utc_ms(2024, 11, 3, 4, 30)                          # 00:30 EDT, before fall-back
    u = start + np.arange(0, 4 * 3600 * 1000, 60_000, dtype="int64")
    src = u + NY7.offset_at_utc_ms(u) * 1000
    assert (np.diff(src) < 0).any()                              # wall clock really jumps back in server time
    back, amb, non = NY7.source_to_utc_ms(src)
    assert (back == u).all() and not amb.any()


def test_dst_fold_without_reversal_is_flagged_not_silently_resolved():
    # only the second occurrence of the repeated hour is present: ambiguous by construction
    u = utc_ms(2024, 11, 3, 6, 10) + np.arange(0, 30 * 60 * 1000, 60_000, dtype="int64")
    src = u + NY7.offset_at_utc_ms(u) * 1000
    back, amb, _ = NY7.source_to_utc_ms(src)
    assert amb.all() and (back != u).all()                       # flagged, and best effort is 1 h off (documented)


def test_nonexistent_wall_time_flagged():
    wall = dt.datetime(2024, 3, 10, 2, 30)                      # NY local time that never happened
    src = int((wall + dt.timedelta(hours=7)).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    _, amb, non = NY7.source_to_utc_ms(np.array([src]))
    assert non.all() and not amb.any()


# ---------------- DST inference ----------------
def weekly_open_source_times(rule, year=2024, jitter_s=240, seed=2):
    rng = np.random.default_rng(seed)
    d, out = dt.date(year, 1, 7), []                             # Sundays
    while d.year == year:
        ny = dt.datetime(d.year, d.month, d.day, 17, tzinfo=__import__("zoneinfo").ZoneInfo("America/New_York"))
        u = int(ny.timestamp()) + int(rng.integers(0, jitter_s))
        out.append(u + int(rule.offset_at_utc_ms(np.array([u * 1000]))[0]))
        d += dt.timedelta(days=7)
    return np.array(out, dtype="int64")


def test_dst_inference_fixed_offset_server():
    inf = infer_dst_from_opens(weekly_open_source_times(ServerTimeRule.fixed(3 * 3600)), 3 * 3600)
    assert inf.classification == "fixed_offset_vs_ny_open" and abs(inf.delta_s + 3600) < 900


def test_dst_inference_ny_anchored_server():
    inf = infer_dst_from_opens(weekly_open_source_times(NY7), 3 * 3600)
    assert inf.classification == "us_dst_anchored" and abs(inf.delta_s) < 900


def test_dst_inference_needs_both_regimes():
    opens = weekly_open_source_times(NY7)[:8]                    # Jan-Feb only (standard time)
    assert infer_dst_from_opens(opens, 2 * 3600).classification == "inconclusive"


def test_dst_inference_from_tick_stream_gaps():
    opens = weekly_open_source_times(ServerTimeRule.fixed(3 * 3600))
    ticks = np.sort(np.concatenate([opens, opens - 30 * 3600 - 5, opens + 60]))     # data before the gap, at the open
    inf = infer_dst_from_weekly_open(ticks, 3 * 3600)
    assert inf.classification in ("fixed_offset_vs_ny_open", "inconclusive")


def test_inference_carries_its_assumption():
    inf = infer_dst_from_opens(weekly_open_source_times(NY7), 3 * 3600)
    assert "17:00" in inf.assumption and "inference" in inf.assumption


# ---------------- resolution ----------------
def _est(off):
    return estimate_offset(samples(off))


def test_resolve_outcomes():
    when = dt.datetime(2024, 7, 1, tzinfo=dt.timezone.utc)
    s0 = resolve_time_basis(_est(0), None, broker="B", server="S", calibrated_at_utc=when)
    assert s0.basis is TimeBasis.UTC_VERIFIED and s0.rule.fixed_offset_s == 0
    s3 = resolve_time_basis(_est(3 * 3600), None, broker="B", server="S", calibrated_at_utc=when)
    assert s3.basis is TimeBasis.SERVER_FIXED_OFFSET and not s3.dst_determined and "NOT determined" in s3.notes[0]
    dst = DstInference("us_dst_anchored", 10, 10, 0.0, 0.0, 0.0)
    sd = resolve_time_basis(_est(3 * 3600), dst, broker="B", server="S", calibrated_at_utc=when)
    assert sd.basis is TimeBasis.SERVER_DST_RULE and sd.rule.shift_s == 7 * 3600 and sd.dst_determined
    winter = resolve_time_basis(_est(2 * 3600), dst, broker="B", server="S", calibrated_at_utc=dt.datetime(2024, 1, 15, tzinfo=dt.timezone.utc))
    assert winter.rule.shift_s == 7 * 3600                       # same rule recovered from a winter calibration
    fx = resolve_time_basis(_est(3 * 3600), DstInference("fixed_offset_vs_ny_open", 10, 10, 0, 0, -3600), broker="B", server="S")
    assert fx.basis is TimeBasis.SERVER_FIXED_OFFSET and fx.dst_determined
    bad = resolve_time_basis(estimate_offset(samples(0, n=2)), None, broker="B", server="S")
    assert bad.basis is TimeBasis.UNVERIFIED and bad.rule is None


def test_unverified_timebase_refuses_to_normalise_silently():
    tb = TimeBase(unverified_spec("B", "S", "no calibration"))
    with pytest.raises(TimeCalibrationError, match="UNVERIFIED"):
        tb.normalize(np.array([1_000]))
    n = tb.normalize(np.array([1_000]), allow_unverified=True)          # explicit opt-in only
    assert n.basis is TimeBasis.UNVERIFIED and n.utc_ms[0] == 1_000


def test_timebase_save_load_and_inverse(tmp_path):
    spec = resolve_time_basis(_est(3 * 3600), DstInference("us_dst_anchored", 9, 9, 0, 0, 0), broker="B", server="S")
    tb = TimeBase(spec)
    tb.save(tmp_path / "t.json")
    tb2 = TimeBase.load(tmp_path / "t.json")
    assert tb2.spec.identity() == spec.identity() and tb2.basis is TimeBasis.SERVER_DST_RULE
    u = np.array([1_720_000_000, 1_700_000_000])
    assert (tb2.utc_to_source_s(u) - u).tolist() == [3 * 3600, 2 * 3600]


def test_identity_ignores_calibration_noise():
    a = resolve_time_basis(_est(3 * 3600), None, broker="B", server="S")
    b = resolve_time_basis(estimate_offset(samples(3 * 3600, age=0.9)), None, broker="B", server="S")
    assert a.identity() == b.identity() and a.estimate.residual_s != b.estimate.residual_s
