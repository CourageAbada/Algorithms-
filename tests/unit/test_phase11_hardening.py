"""Phase 1.1 acquisition hardening: request-timeout vs connection-loss, retry/backoff, warming order, bar/limit checks,
time-calibration validity, unknown flags, duplicate-timestamp policy, stale/future ticks, history-depth discovery."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from fxscalp.brokers.errors import (CallTimeoutError, ConnectionLostError, HistoryUnavailableError, RequestTimeoutError,
                                    TerminalRequestError)
from fxscalp.brokers.mt5.adapter import AdapterState
from fxscalp.brokers.mt5.fake import FakeMT5Adapter, FakeScenario
from fxscalp.market_data import quality as Qm
from fxscalp.market_data.acquire import AcquisitionConfig, TickAcquirer, discover_history_depth
from fxscalp.market_data.quality import Q
from fxscalp.market_data.retry import NO_RETRY, RetryPolicy, call_with_retry
from fxscalp.market_data.store import ChunkKey, TickStore
from fxscalp.market_data.timebase import (OffsetEstimate, TimeBase, TimeBasis, assess_offset_consistency,
                                          build_calibration_record, dst_status, resolve_time_basis,
                                          us_dst_transition_in_window)
from fxscalp.market_data.verification import assess_bar_completeness, assess_latest_tick_freshness
from fxscalp.workflows import probe_weekly_opens_detailed
from tests.unit.conftest import DAY0, make_ticks, make_timebase

FAST = RetryPolicy(max_retries=3, backoff_base_s=0.01, backoff_factor=2.0, backoff_max_s=0.05, unavailable_retries=1)
KEY = ChunkKey("Fake Broker Ltd", "FakeBroker-Demo", "XAU_USD", DAY0)


def connected(scenario: FakeScenario, **kw) -> FakeMT5Adapter:
    a = FakeMT5Adapter(scenario, **kw)
    a.connect()
    return a


def acquirer(a, tmp_path, **cfg):
    tb = make_timebase(a)
    st = TickStore(tmp_path)
    cfg.setdefault("retry", FAST)
    return TickAcquirer(a, st, tb, broker="Fake Broker Ltd", server="FakeBroker-Demo", canonical="XAU_USD",
                        broker_symbol="XAUUSDm", symbol_info=a.get_symbol_info("XAUUSDm"), account_fingerprint="fp",
                        account_currency="USD", config=AcquisitionConfig(**cfg), sleep=lambda s: None), st


# ============================ 2. request timeout != connection lost ============================
def test_slow_history_request_does_not_mark_the_connection_lost_when_health_check_passes():
    sc = FakeScenario(slow_function="copy_ticks_range", slow_s=0.6, slow_first_n=1, weekend_closure=False)
    a = connected(sc, history_timeout_s=0.2, health_timeout_s=5.0)
    s = 1_760_000_000
    with pytest.raises(RequestTimeoutError):
        a.get_ticks_range("XAUUSDm", s, s + 600)
    assert a.state == AdapterState.CONNECTED                         # DEGRADED only while the health check ran
    ev = a.timeout_events[-1]
    assert ev["function"] == "copy_ticks_range" and ev["health_ok"] is True
    assert len(a.get_ticks_range("XAUUSDm", s, s + 600)) > 0         # same session keeps working
    assert a.fake.trading_calls == []


def test_request_timeout_becomes_lost_only_if_the_health_check_actually_fails():
    sc = FakeScenario(slow_function="copy_ticks_range", slow_s=3.0, slow_first_n=1, weekend_closure=False)
    a = connected(sc, history_timeout_s=0.2, health_timeout_s=0.3)   # the blocked call outlasts the health deadline
    s = 1_760_000_000
    with pytest.raises(CallTimeoutError):                            # a ConnectionLostError subtype
        a.get_ticks_range("XAUUSDm", s, s + 600)
    assert a.state == AdapterState.LOST
    assert a.timeout_events[-1]["health_ok"] is False


def test_non_history_call_timeout_keeps_the_strict_semantics():
    sc = FakeScenario(hang_function="symbol_info_tick", hang_s=1.0)
    a = connected(sc, call_timeout_s=0.2)
    with pytest.raises(CallTimeoutError):
        a.get_latest_tick("XAUUSDm")
    assert a.state == AdapterState.LOST


def test_history_timeout_default_is_based_on_measured_cold_loads():
    a = FakeMT5Adapter()
    assert a._history_timeout_s >= 2 * 95 - 10 and a._history_timeout_s > a._owner._default_timeout


def test_terminal_error_none_and_empty_are_distinguished():
    a = connected(FakeScenario(terminal_error_function="copy_ticks_range", terminal_error_first_n=1, none_first_n=0,
                               weekend_closure=False))
    s = 1_760_000_000
    with pytest.raises(TerminalRequestError) as ei:
        a.get_ticks_range("XAUUSDm", s, s + 60)
    assert ei.value.code == -1 and a.state == AdapterState.CONNECTED
    b = connected(FakeScenario(none_first_n=1, weekend_closure=False))
    with pytest.raises(HistoryUnavailableError):                     # None with a success code = no data, not an error
        b.get_ticks_range("XAUUSDm", s, s + 60)
    c = connected(FakeScenario())
    sunday = int(dt.datetime(2025, 10, 12, 6, tzinfo=dt.timezone.utc).timestamp())
    assert len(c.get_ticks_range("XAUUSDm", sunday, sunday + 60)) == 0   # legitimate empty array


def test_ipc_failure_is_connection_loss_not_a_retryable_request_error():
    a = connected(FakeScenario(fail_after_calls=12, weekend_closure=False))
    with pytest.raises(ConnectionLostError):
        for _ in range(20):
            a.get_ticks_range("XAUUSDm", 1_760_000_000, 1_760_000_060)
    assert a.state == AdapterState.LOST


# ============================ 3. retry / backoff ============================
def test_call_with_retry_is_bounded_logs_each_retry_and_backs_off():
    waits, events, n = [], [], {"c": 0}

    def always_timeout():
        n["c"] += 1
        raise RequestTimeoutError("slow")

    pol = RetryPolicy(max_retries=3, backoff_base_s=5.0, backoff_factor=3.0, backoff_max_s=40.0)
    with pytest.raises(RequestTimeoutError):
        call_with_retry(always_timeout, pol, label="x", sleep=waits.append, on_event=events.append)
    assert n["c"] == 4 and waits == [5.0, 15.0, 40.0]                # bounded, exponential, capped
    assert [e["event"] for e in events] == ["retry", "retry", "retry", "retry_exhausted"]
    assert all(e["reason"] == "request_timeout" for e in events)


def test_retry_never_retries_connection_loss_or_non_retryable_terminal_codes():
    calls = {"c": 0}

    def lost():
        calls["c"] += 1
        raise ConnectionLostError("gone")

    with pytest.raises(ConnectionLostError):
        call_with_retry(lost, FAST, sleep=lambda s: None)
    assert calls["c"] == 1

    def bad_params():
        calls["c"] += 1
        raise TerminalRequestError("invalid params", -2)

    with pytest.raises(TerminalRequestError):
        call_with_retry(bad_params, FAST, sleep=lambda s: None)
    assert calls["c"] == 2


def test_acquirer_retries_terminal_error_then_succeeds_and_logs(tmp_path):
    a = connected(FakeScenario(terminal_error_function="copy_ticks_range", terminal_error_first_n=2))
    acq, st = acquirer(a, tmp_path)
    r = acq.acquire_day(DAY0)
    assert r.status == "acquired" and r.rows > 0
    m = st.chunk_manifest(KEY)
    assert m["retries"] == 2 and m["retry_reasons"] == ["terminal_error_-1"]
    log = [json.loads(x) for x in (tmp_path / "acquisition_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(1 for e in log if e.get("event") == "retry") == 2


def test_acquirer_recovers_from_a_cold_load_timeout_without_reconnecting(tmp_path):
    a = connected(FakeScenario(slow_function="copy_ticks_range", slow_s=0.5, slow_first_n=1),
                  history_timeout_s=0.15, health_timeout_s=5.0)
    acq, st = acquirer(a, tmp_path)
    r = acq.acquire_day(DAY0)
    assert r.status == "acquired" and st.chunk_manifest(KEY)["retry_reasons"] == ["request_timeout"]
    assert a.state == AdapterState.CONNECTED and a.fake.calls.count("initialize") == 1


def test_retries_exhausted_fail_the_chunk_and_store_nothing(tmp_path):
    a = connected(FakeScenario(terminal_error_function="copy_ticks_range", terminal_error_first_n=99))
    acq, st = acquirer(a, tmp_path, retry=RetryPolicy(max_retries=2, backoff_base_s=0.01, backoff_max_s=0.02))
    s = acq.acquire_range(DAY0, DAY0)
    assert s.failed and st.chunk_manifest(KEY) is None and s.dataset_manifest is None


def test_unavailable_history_is_not_stored_as_an_empty_chunk(tmp_path):
    a = connected(FakeScenario(none_first_n=2))                      # first attempt + confirmation retry both say "no data"
    acq, st = acquirer(a, tmp_path)
    r = acq.acquire_day(DAY0)
    assert r.status == "unavailable" and st.chunk_manifest(KEY) is None
    assert acq.acquire_day(DAY0).status == "acquired"                # later retry works: nothing immutable was written


def test_legitimate_empty_day_is_stored_with_its_reason(tmp_path):
    a = connected(FakeScenario())
    acq, st = acquirer(a, tmp_path)
    r = acq.acquire_day(dt.date(2025, 10, 12))
    assert r.status == "empty" and st.chunk_manifest(ChunkKey("Fake Broker Ltd", "FakeBroker-Demo", "XAU_USD", dt.date(2025, 10, 12)))["empty_reason"] == "legitimate_empty_result"


# ============================ 4. gradual warming / resume ============================
def test_range_is_acquired_newest_first_and_resume_skips_verified_chunks(tmp_path):
    a = connected(FakeScenario())
    acq, st = acquirer(a, tmp_path)
    s = acq.acquire_range(DAY0, DAY0 + dt.timedelta(days=3))
    order = [json.loads(x)["day"] for x in (tmp_path / "acquisition_log.jsonl").read_text(encoding="utf-8").splitlines()
             if json.loads(x).get("status") == "acquired"]
    assert order == sorted(order, reverse=True) and not s.failed     # newest day requested first, then backwards
    n_calls = a.fake.calls.count("copy_ticks_range")
    s2 = acq.acquire_range(DAY0, DAY0 + dt.timedelta(days=3))
    assert [c.status for c in s2.chunks] == ["skipped_verified"] * 4
    assert a.fake.calls.count("copy_ticks_range") == n_calls         # verified immutable chunks never re-downloaded


# ============================ 5. bar completeness ============================
def _ticks_over_minutes(n_min, per_min=20, start=1_760_000_000):
    t0 = (start // 60) * 60 * 1000
    return np.sort(np.concatenate([t0 + m * 60_000 + np.arange(per_min) * 2000 for m in range(n_min)]))


def test_one_bar_for_an_hour_of_ticks_is_not_a_pass():
    ticks = _ticks_over_minutes(60)
    c = assess_bar_completeness(np.array([ticks[0] // 1000]), ticks)
    assert c.status == "FAIL" and c.minutes_covered <= 1


def test_complete_bars_pass_and_quiet_minutes_do_not_penalise():
    ticks = _ticks_over_minutes(60)
    bars = np.unique((ticks // 1000) // 60) * 60
    assert assess_bar_completeness(bars, ticks).status == "PASS"
    sparse = ticks[(ticks // 60_000) % 5 != 0]                       # 20% of minutes have no ticks (session gaps)
    assert assess_bar_completeness(np.unique((sparse // 1000) // 60) * 60, sparse).status == "PASS"


def test_partially_complete_bars_warn_and_tiny_samples_are_inconclusive():
    ticks = _ticks_over_minutes(60)
    bars = (np.unique((ticks // 1000) // 60) * 60)[:40]
    assert assess_bar_completeness(bars, ticks).status == "WARN"
    assert assess_bar_completeness(np.array([]), _ticks_over_minutes(4)).status == "INCONCLUSIVE"


# ============================ 7/8. time calibration validity ============================
def _est(offset=10800, resid=0.5):
    return OffsetEstimate("ok", offset, offset + resid, resid, 0.04, 60, 60, True, 900, ())


def _weekly_opens(n, offset=10800, step_after=None, step_s=3600):
    # open = Sunday 21:00 UTC (US summer 17:00 NY) expressed in source time; weekly from a known Sunday
    base = int(dt.datetime(2026, 4, 5, 21, tzinfo=dt.timezone.utc).timestamp()) + offset
    out = []
    for k in range(n):
        o = base + k * 604800
        if step_after is not None and k >= step_after:
            o += step_s
        out.append(o)
    return np.array(out, dtype="int64")


def test_consistent_weekly_opens_support_one_offset_over_the_observed_period():
    c = assess_offset_consistency(_weekly_opens(26), 10800)
    assert c.consistent and c.n_weeks == 26 and c.n_outlier_weeks == 0
    assert c.first_open_utc_ms == int(dt.datetime(2026, 4, 5, 21, tzinfo=dt.timezone.utc).timestamp() * 1000)


def test_an_offset_step_inside_the_window_makes_the_evidence_inconsistent():
    c = assess_offset_consistency(_weekly_opens(26, step_after=13), 10800)
    assert not c.consistent and abs(c.step_between_halves_s - 3600) < 1 and any("NOT constant" in n for n in c.notes)


def test_time_of_week_wraparound_is_not_an_outlier():
    base = _weekly_opens(12)
    base = base + 3 * 3600 - 5 * 60    # opens near Monday 00:00 source time ...
    jitter = np.where(np.arange(12) % 2 == 0, 0, 600)   # ... straddling the week boundary
    assert assess_offset_consistency(base + jitter, 10800).consistent


def test_validity_period_is_bounded_and_outside_is_uncertain_not_dropped():
    cons = assess_offset_consistency(_weekly_opens(26), 10800)
    calibrated = dt.datetime(2026, 10, 7, 12, tzinfo=dt.timezone.utc)
    spec = resolve_time_basis(_est(), None, broker="B", server="S", calibrated_at_utc=calibrated, consistency=cons,
                              bound_validity=True)
    assert spec.basis == TimeBasis.SERVER_FIXED_OFFSET and not spec.dst_determined
    assert spec.valid_from_utc_ms == cons.first_open_utc_ms and spec.valid_to_utc_ms == int(calibrated.timestamp() * 1000)
    assert spec.dst_status == "unresolved_no_us_dst_transition_in_observed_window"
    tb = TimeBase(spec)
    inside = int(dt.datetime(2026, 9, 1, 12, tzinfo=dt.timezone.utc).timestamp() * 1000) + 10_800_000
    before = int(dt.datetime(2026, 1, 15, 12, tzinfo=dt.timezone.utc).timestamp() * 1000) + 10_800_000
    after = int(dt.datetime(2026, 11, 20, 12, tzinfo=dt.timezone.utc).timestamp() * 1000) + 10_800_000
    n = tb.normalize(np.array([before, inside, after], dtype="int64"))
    assert n.uncertain.tolist() == [True, False, True]
    assert n.utc_ms.tolist() == [before - 10_800_000, inside - 10_800_000, after - 10_800_000]   # converted, flagged, kept


def test_without_consistent_evidence_validity_is_only_the_calibration_day():
    calibrated = dt.datetime(2026, 10, 7, 12, tzinfo=dt.timezone.utc)
    spec = resolve_time_basis(_est(), None, broker="B", server="S", calibrated_at_utc=calibrated, bound_validity=True)
    assert spec.valid_to_utc_ms - spec.valid_from_utc_ms == 24 * 3600 * 1000


def test_legacy_unbounded_spec_is_unchanged_and_unverified_rows_are_all_uncertain():
    spec = resolve_time_basis(_est(), None, broker="B", server="S")
    assert spec.valid_from_utc_ms is None and "valid_from_utc_ms" not in spec.identity()
    from fxscalp.market_data.timebase import unverified_spec
    n = TimeBase(unverified_spec("B", "S", "x")).normalize(np.array([1, 2, 3], dtype="int64"), allow_unverified=True)
    assert n.uncertain.all()


def test_calibration_record_contents_and_confidence():
    cons = assess_offset_consistency(_weekly_opens(26), 10800)
    spec = resolve_time_basis(_est(), None, broker="B", server="S", consistency=cons, bound_validity=True,
                              calibrated_at_utc=dt.datetime(2026, 10, 7, 12, tzinfo=dt.timezone.utc))
    rec = build_calibration_record(spec)
    for k in ("broker", "server", "effective_from_utc", "effective_to_utc", "offset_s", "n_samples", "residual_s", "mad_s",
              "dst_status", "confidence", "calibrated_at_utc", "basis_id", "record_id"):
        assert k in rec, k
    assert rec["offset_s"] == 10800 and rec["n_samples"] == 60 and rec["dst_status"].startswith("unresolved")
    assert rec["confidence"] == "high"                                # 60 samples, tiny MAD/residual, 26 consistent weeks


def test_dst_transition_detection():
    ms = lambda y, m, d: int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp() * 1000)  # noqa: E731
    assert not us_dst_transition_in_window(ms(2026, 4, 1), ms(2026, 10, 7))
    assert us_dst_transition_in_window(ms(2026, 2, 1), ms(2026, 4, 1))
    assert dst_status(None, ms(2026, 2, 1), ms(2026, 4, 1)) == "unresolved_transition_inside_window_inconclusive"


def test_stored_chunk_preserves_raw_time_and_records_rule_basis_id_and_uncertainty(tmp_path):
    a = connected(FakeScenario())
    tb = make_timebase(a)
    # bound the validity to a window that excludes the fake data day -> every row TIME_BASIS_UNCERTAIN, none dropped
    from dataclasses import replace
    spec = replace(tb.spec, valid_from_utc_ms=1, valid_to_utc_ms=2)
    acq = TickAcquirer(a, TickStore(tmp_path), TimeBase(spec), broker="Fake Broker Ltd", server="FakeBroker-Demo",
                       canonical="XAU_USD", broker_symbol="XAUUSDm", symbol_info=a.get_symbol_info("XAUUSDm"),
                       config=AcquisitionConfig(retry=FAST), sleep=lambda s: None)
    r = acq.acquire_day(DAY0)
    st = acq.store
    raw = st.read_raw_chunk(KEY)
    assert raw.num_rows == r.rows > 0
    assert {"source_time", "source_time_msc", "normalized_utc_time", "time_basis", "normalization_rule",
            "time_basis_id"} <= set(raw.column_names)
    assert set(raw["normalization_rule"].to_pylist()) == {spec.rule.name} and set(raw["time_basis_id"].to_pylist()) == {spec.basis_id()}
    q = st.read_derived("tick_quality", KEY)
    man = st.chunk_manifest(KEY)
    assert man["time_basis_uncertain_rows"] == r.rows and man["time_basis_id"] == spec.basis_id()
    assert (q["quality_flags"].to_numpy() & int(Q.TIME_BASIS_UNCERTAIN)).all() and not q["quarantined"].to_numpy().any()


# ============================ 9. unknown flag bits ============================
def _assess(a, **kw):
    return Qm.assess(a["time"], a["time_msc"], a["bid"], a["ask"], a["last"], a["volume"], a["flags"], **kw)


def test_unknown_flag_bit_0x400_is_preserved_tagged_not_quarantined():
    a = make_ticks(200)
    a["flags"] = 0x406
    q = _assess(a)
    assert (a["flags"] == 0x406).all()                                # raw flags untouched
    assert (q.unknown_bits == 0x400).all()
    assert q.summary["counts"]["UNKNOWN_FLAG_BITS"] == 200 and q.summary["quarantined"] == 0 and q.summary["clean"] == 0
    assert q.summary["unknown_flag_bits_histogram"] == {"0x400": 200}
    t = q.to_table(np.arange(200))
    assert t.column_names == ["seq", "quality_flags", "quarantined", "unknown_flag_bits"] and set(t["unknown_flag_bits"].to_pylist()) == {0x400}


def test_known_flags_have_no_unknown_bits():
    a = make_ticks(50)
    a["flags"] = 6
    assert (_assess(a).unknown_bits == 0).all()


# ============================ 10. duplicate timestamp vs duplicate tick ============================
def test_same_millisecond_ticks_with_different_content_are_kept_in_order_and_not_duplicates():
    a = make_ticks(10, step_ms=1000)
    a[4]["time_msc"] = a[3]["time_msc"]
    a[4]["time"] = a[3]["time"]
    a[4]["bid"], a[4]["ask"] = a[3]["bid"] + 0.1, a[3]["ask"] + 0.1   # different content
    q = _assess(a)
    assert (q.flags[4] & int(Q.DUP_TIMESTAMP)) and not (q.flags[4] & int(Q.DUP_TICK))
    assert not q.quarantined[4]                                        # not removed/quarantined
    s = q.summary
    assert s["duplicate_timestamp_total"] == 1 and s["duplicate_timestamp_distinct_content"] == 1 and s["duplicate_tick_identical"] == 0


def test_identical_repeats_are_duplicate_ticks_and_still_not_dropped():
    a = make_ticks(10, step_ms=1000)
    a[5] = a[4]
    q = _assess(a)
    assert (q.flags[5] & int(Q.DUP_TIMESTAMP)) and (q.flags[5] & int(Q.DUP_TICK)) and q.summary["duplicate_tick_identical"] == 1
    assert len(q.flags) == 10


def test_stored_order_of_same_timestamp_ticks_is_the_broker_order(tmp_path):
    from fxscalp.market_data.schema import raw_table
    from fxscalp.market_data.timebase import NormalizedTime
    a = make_ticks(6, step_ms=1000)
    a[3]["time_msc"] = a[2]["time_msc"]
    a[3]["time"] = a[2]["time"]
    a[3]["bid"] = 1.0                                                  # deliberately out-of-trend: must not be reordered
    norm = NormalizedTime(a["time_msc"].astype("int64"), np.zeros(6, bool), np.zeros(6, bool), TimeBasis.SERVER_FIXED_OFFSET)
    t = raw_table(a, norm, dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
    assert t["seq"].to_pylist() == list(range(6)) and t["bid"].to_pylist() == a["bid"].tolist()


# ============================ 11. stale / future ============================
def test_future_ticks_are_tagged_and_quarantined_with_tolerance_for_latency():
    a = make_ticks(5, step_ms=1000)
    ing = 1_760_000_100_000
    utc = np.array([ing - 5000, ing - 1000, ing + 1500, ing + 3000, ing + 60_000], dtype="int64")
    q0 = _assess(a, normalized_utc_ms=utc, ingestion_utc_ms=ing)                       # tolerance 2 s
    assert [bool(f & int(Q.FUTURE_TICK)) for f in q0.flags] == [False, False, False, True, True]
    assert q0.quarantined.tolist() == [False, False, False, True, True]
    q1 = _assess(a, normalized_utc_ms=utc, ingestion_utc_ms=ing, receive_latency_s=1.5)  # latency widens tolerance to 3.5 s
    assert [bool(f & int(Q.FUTURE_TICK)) for f in q1.flags] == [False, False, False, False, True]


def test_latest_tick_freshness_classification_and_tolerances():
    off, now = 10800, 1_760_000_000.0
    msc = lambda age: int((now - age + off) * 1000)  # noqa: E731
    assert assess_latest_tick_freshness(msc(0.5), off, now).status == "FRESH"
    assert assess_latest_tick_freshness(msc(300), off, now).status == "STALE"
    assert assess_latest_tick_freshness(msc(-30), off, now).status == "FUTURE"
    assert assess_latest_tick_freshness(msc(-2.4), off, now).status == "FUTURE"
    assert assess_latest_tick_freshness(msc(-2.4), off, now, call_latency_s=0.5).status == "FRESH"   # latency absorbed
    assert assess_latest_tick_freshness(msc(-2.4), off, now, clock_residual_s=0.66, call_latency_s=0.0).status == "FRESH"


# ============================ 12. history-depth discovery ============================
def _now_source():
    return int(dt.datetime(2025, 10, 8, tzinfo=dt.timezone.utc).timestamp()) + 3 * 3600


def test_depth_discovery_finds_boundary_with_few_calls_and_caches(tmp_path):
    start = dt.datetime(2025, 4, 1, tzinfo=dt.timezone.utc).timestamp()
    a = connected(FakeScenario(history_start_s=start, weekend_closure=False))
    cache = tmp_path / "depth.json"
    d = discover_history_depth(a, "XAUUSDm", _now_source(), cache_path=cache, policy=FAST, sleep=lambda s: None)
    assert d.status == "boundary_found" and d.earliest_data_day is not None
    assert abs((d.earliest_data_day - dt.date(2025, 4, 1)).days) <= 3
    assert d.empty_confirmed_day is not None and d.empty_confirmed_day < d.earliest_data_day
    assert d.calls <= 40 and cache.exists()
    n = a.fake.calls.count("copy_ticks_range")
    d2 = discover_history_depth(a, "XAUUSDm", _now_source(), cache_path=cache, policy=FAST, sleep=lambda s: None)
    assert d2.cached and d2.earliest_data_day == d.earliest_data_day and a.fake.calls.count("copy_ticks_range") == n
    assert json.loads(cache.read_text(encoding="utf-8"))["observed_at_utc"]
    # no probe went deeper than the confirmed boundary week
    deepest = min(dt.date.fromisoformat(s["day"]) for s in d.samples)
    assert deepest >= d.empty_confirmed_day - dt.timedelta(days=7)


def test_depth_discovery_never_reads_a_persistent_error_as_a_boundary(tmp_path):
    a = connected(FakeScenario(terminal_error_function="copy_ticks_range", terminal_error_first_n=999, weekend_closure=False))
    d = discover_history_depth(a, "XAUUSDm", _now_source(), cache_path=tmp_path / "d.json", policy=NO_RETRY, sleep=lambda s: None)
    assert d.status == "inconclusive" and d.empty_confirmed_day is None and not (tmp_path / "d.json").exists()


def test_depth_discovery_confirms_an_empty_week_with_a_second_probe(tmp_path):
    a = connected(FakeScenario(no_ticks_symbols=("XAUUSDm",)))
    d = discover_history_depth(a, "XAUUSDm", _now_source(), policy=NO_RETRY, sleep=lambda s: None)
    assert d.status == "no_recent_data" and d.earliest_data_day is None and d.calls == 2


# ============================ weekly opens: skipped weeks are recorded, no hammering ============================
def test_weekly_open_probe_records_skipped_weeks_and_stops_after_consecutive_errors():
    start = dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc).timestamp()
    a = connected(FakeScenario(history_start_s=start))
    p = probe_weekly_opens_detailed(a, "XAUUSDm", _now_source(), 30, policy=FAST, sleep=lambda s: None)
    assert len(p.opens) >= 8 and any("history starts later" in s["reason"] for s in p.skipped)
    b = connected(FakeScenario(terminal_error_function="copy_ticks_from", terminal_error_first_n=999))
    calls0 = b.fake.calls.count("copy_ticks_from")
    q = probe_weekly_opens_detailed(b, "XAUUSDm", _now_source(), 30, policy=RetryPolicy(max_retries=1, backoff_base_s=0.01),
                                    sleep=lambda s: None, max_consecutive_errors=3)
    assert len(q.opens) == 0 and any("stopped after 3 consecutive" in s.get("reason", "") for s in q.skipped)
    assert b.fake.calls.count("copy_ticks_from") - calls0 <= 3 * 2           # bounded: 3 weeks x (1 try + 1 retry)


# ============================ safety ============================
def test_hardening_never_touches_trading_functions(tmp_path):
    a = connected(FakeScenario(terminal_error_function="copy_ticks_range", terminal_error_first_n=1,
                               slow_function="copy_ticks_range", slow_s=0.3, slow_first_n=1),
                  history_timeout_s=0.1, health_timeout_s=3)
    acq, _ = acquirer(a, tmp_path)
    acq.acquire_range(DAY0, DAY0 + dt.timedelta(days=1))
    discover_history_depth(a, "XAUUSDm", _now_source(), policy=FAST, sleep=lambda s: None)
    assert a.fake.trading_calls == []
    assert not ({"order_send", "order_check", "positions_get", "orders_get"} & set(a.fake.calls))


def test_weekly_open_probe_rejects_the_first_tick_of_history_returned_for_earlier_saturdays():
    """Regression (found on the real terminal): copy_ticks_from(Saturday before history) returns the first tick of ALL
    history; repeating it for each earlier week produced bogus identical 'opens' and a false 'inconsistent offset'."""
    start = dt.datetime(2025, 7, 1, 22, 49, 48, tzinfo=dt.timezone.utc).timestamp()
    a = connected(FakeScenario(history_start_s=start))
    p = probe_weekly_opens_detailed(a, "XAUUSDm", _now_source(), 40, policy=FAST, sleep=lambda s: None)
    first_history_tick_src = int(start) + 3 * 3600
    assert first_history_tick_src not in set(p.opens.tolist())
    assert len(set(p.opens.tolist())) == len(p.opens)                   # no repeated 'open'
    assert assess_offset_consistency(p.opens, 10800).consistent         # genuine weekly opens agree
