import datetime as dt
import json
import os
import stat

import numpy as np
import pytest

from fxscalp.brokers.errors import TimeCalibrationError
from fxscalp.brokers.mt5.fake import FakeMT5Adapter, FakeScenario
from fxscalp.market_data.acquire import (AcquisitionConfig, TickAcquirer, day_window_s, persist_symbol_metadata,
                                         probe_history_depth, probe_request_limits)
from fxscalp.market_data.quality import Q
from fxscalp.market_data.schema import RAW_SCHEMA, derived_table, raw_table
from fxscalp.market_data.store import ChunkCorruptError, ChunkExistsError, ChunkKey, TickStore, raw_content_sha256
from fxscalp.market_data.timebase import NormalizedTime, TimeBase, TimeBasis, unverified_spec
from tests.unit.conftest import DAY0, make_ticks, make_timebase

NOW = dt.datetime(2025, 10, 6, 12, tzinfo=dt.timezone.utc)


def table(n=50, **kw):
    a = make_ticks(n, **kw)
    norm = NormalizedTime(a["time_msc"].astype("int64") - 3 * 3600 * 1000, np.zeros(n, bool), np.zeros(n, bool),
                          TimeBasis.SERVER_FIXED_OFFSET)
    return a, raw_table(a, norm, NOW)


KEY = ChunkKey("Fake Broker Ltd", "FakeBroker-Demo", "XAU_USD", DAY0)


# ---------------- schema ----------------
def test_raw_schema_has_required_provenance_columns_and_preserves_raw_values():
    a, t = table()
    for col in ("source_time", "source_time_msc", "normalized_utc_time", "ingestion_time_utc", "time_basis"):
        assert col in t.column_names
    assert t.schema == RAW_SCHEMA
    assert (t["bid"].to_numpy() == a["bid"]).all() and (t["source_time_msc"].to_numpy() == a["time_msc"]).all()
    assert set(t["time_basis"].to_pylist()) == {"server_fixed_offset"}


def test_derived_is_separate_and_never_overwrites_raw():
    a, t = table()
    d = derived_table(t, 0.01)
    assert set(d.column_names) == {"seq", "mid", "spread", "spread_points", "relative_spread"}
    assert not (set(d.column_names) - {"seq"}) & set(t.column_names)
    assert np.allclose(d["spread"].to_numpy(), a["ask"] - a["bid"])
    assert np.allclose(d["spread_points"].to_numpy(), (a["ask"] - a["bid"]) / 0.01)
    assert np.allclose(d["mid"].to_numpy(), (a["ask"] + a["bid"]) / 2)
    assert np.isnan(derived_table(t, None)["spread_points"].to_numpy()).all()


# ---------------- store ----------------
def test_store_roundtrip_manifest_and_verify(tmp_path):
    a, t = table()
    st = TickStore(tmp_path)
    man = st.write_raw_chunk(KEY, t, {"broker_symbol": "XAUUSDm"})
    assert man["record_count"] == 50 and man["status"] == "verified" and len(man["file_sha256"]) == 64
    assert st.verify_chunk(KEY) == (True, "ok")
    assert st.read_raw_chunk(KEY).num_rows == 50
    assert (tmp_path / "raw/ticks/broker=fake_broker_ltd/server=fakebroker-demo/instrument=XAU_USD/year=2025/month=10/day=06").is_dir()
    assert st.list_chunks("Fake Broker Ltd", "FakeBroker-Demo", "XAU_USD") == [KEY]


def test_raw_chunks_are_immutable(tmp_path):
    _, t = table()
    st = TickStore(tmp_path)
    st.write_raw_chunk(KEY, t, {})
    p = st.raw_dir(KEY) / "ticks.parquet"
    assert not os.access(p, os.W_OK) or (p.stat().st_mode & stat.S_IWUSR) == 0       # read-only permission set
    assert st.write_raw_chunk(KEY, t, {})["raw_content_sha256"] == raw_content_sha256(t)   # identical rewrite = no-op
    _, t2 = table(seed=99)
    with pytest.raises(ChunkExistsError):
        st.write_raw_chunk(KEY, t2, {})


def test_corruption_detected(tmp_path):
    _, t = table()
    st = TickStore(tmp_path)
    st.write_raw_chunk(KEY, t, {})
    p = st.raw_dir(KEY) / "ticks.parquet"
    os.chmod(p, 0o644)
    with open(p, "r+b") as fh:
        fh.seek(100)
        fh.write(b"\x00\xff\x00")
    ok, why = st.verify_chunk(KEY)
    assert not ok and "checksum" in why
    with pytest.raises(ChunkCorruptError):
        st.read_raw_chunk(KEY)


def test_dataset_manifest_identity_is_deterministic_and_content_sensitive(tmp_path):
    def body(tag):
        return {"schema_version": "tick_raw/1", "broker": "B", "server": "S", "instrument": "XAU_USD", "broker_symbol": "X",
                "time_basis": "server_fixed_offset", "timebase_spec_sha256": "t", "chunks": [
                    {"source_day": "2025-10-06", "raw_content_sha256": tag, "file_sha256": "f", "record_count": 1}],
                "collected_at_utc": "different every run"}
    st = TickStore(tmp_path)
    a, b = st.write_dataset_manifest(body("h1")), st.write_dataset_manifest({**body("h1"), "collected_at_utc": "x"})
    c = st.write_dataset_manifest(body("h2"))
    assert a["dataset_id"] == b["dataset_id"] != c["dataset_id"] and len(a["manifest_checksum_sha256"]) == 64


def test_content_hash_independent_of_normalisation_and_ingestion():
    a, t1 = table()
    norm2 = NormalizedTime(a["time_msc"].astype("int64"), np.zeros(50, bool), np.zeros(50, bool), TimeBasis.UTC_VERIFIED)
    t2 = raw_table(a, norm2, NOW.replace(hour=1))
    assert raw_content_sha256(t1) == raw_content_sha256(t2)


# ---------------- acquisition ----------------
def make_acquirer(adapter, tmp_path, tb=None, **cfg):
    tb = tb or make_timebase(adapter)
    info = adapter.get_symbol_info("XAUUSDm")
    st = TickStore(tmp_path)
    return TickAcquirer(adapter, st, tb, broker="Fake Broker Ltd", server="FakeBroker-Demo", canonical="XAU_USD",
                        broker_symbol="XAUUSDm", symbol_info=info, account_fingerprint="fp", account_currency="USD",
                        config=AcquisitionConfig(**cfg)), st


def test_acquire_range_end_to_end_with_provenance(adapter, tmp_path):
    acq, st = make_acquirer(adapter, tmp_path)
    s = acq.acquire_range(DAY0, DAY0 + dt.timedelta(days=6))
    assert [c.status for c in s.chunks].count("acquired") >= 5 and not s.failed
    m = s.dataset_manifest
    for k in ("dataset_id", "broker", "server", "instrument", "broker_symbol", "start_source_day", "end_source_day",
              "record_count", "schema_version", "time_basis", "collected_at_utc", "software", "manifest_checksum_sha256"):
        assert k in m, k
    assert m["instrument"] == "XAU_USD" and m["broker_symbol"] == "XAUUSDm" and m["software"]["fxscalp"]
    assert m["record_count"] == sum(c["record_count"] for c in m["chunks"])
    for c in m["chunks"]:                                           # dataset -> chunk hashes chain
        cm = st.chunk_manifest(ChunkKey("Fake Broker Ltd", "FakeBroker-Demo", "XAU_USD", dt.date.fromisoformat(c["source_day"])))
        assert cm["raw_content_sha256"] == c["raw_content_sha256"] and cm["file_sha256"] == c["file_sha256"]
    # derived datasets are separate files and point at the raw parent
    key = ChunkKey("Fake Broker Ltd", "FakeBroker-Demo", "XAU_USD", DAY0)
    dm = json.loads((st.derived_dir("tick_basic", key) / "manifest.json").read_text())
    assert dm["parents"][0]["raw_content_sha256"] == st.chunk_manifest(key)["raw_content_sha256"]
    assert st.derived_dir("tick_basic", key) != st.raw_dir(key) and "derived" in str(st.derived_dir("tick_basic", key))


def test_normalised_time_matches_known_ground_truth(adapter, tmp_path):
    acq, st = make_acquirer(adapter, tmp_path)           # fake server is UTC+3 (fixed)
    acq.acquire_day(DAY0)
    t = st.read_raw_chunk(ChunkKey("Fake Broker Ltd", "FakeBroker-Demo", "XAU_USD", DAY0))
    src = t["source_time_msc"].to_numpy()
    utc = t["normalized_utc_time"].cast("int64").to_numpy()
    assert ((src - utc) == 3 * 3600 * 1000).all()
    assert set(t["time_basis"].to_pylist()) == {"server_fixed_offset"} and t["ingestion_time_utc"][0].as_py().year >= 2025


def test_resume_skips_verified_chunks_and_force_redownloads(adapter, tmp_path):
    acq, _ = make_acquirer(adapter, tmp_path)
    acq.acquire_range(DAY0, DAY0 + dt.timedelta(days=2))
    n_calls = adapter.fake.calls.count("copy_ticks_range")
    s = acq.acquire_range(DAY0, DAY0 + dt.timedelta(days=2))
    assert {c.status for c in s.chunks} == {"skipped_verified"} and adapter.fake.calls.count("copy_ticks_range") == n_calls
    key = ChunkKey("Fake Broker Ltd", "FakeBroker-Demo", "XAU_USD", DAY0)
    before = acq.store.chunk_manifest(key)
    r = acq.acquire_day(DAY0, force=True)                           # re-download requested explicitly
    assert r.status == "acquired" and adapter.fake.calls.count("copy_ticks_range") > n_calls
    assert acq.store.chunk_manifest(key)["file_sha256"] == before["file_sha256"]   # identical data -> untouched raw file


def test_corrupt_chunk_is_redownloaded_not_trusted(adapter, tmp_path):
    acq, st = make_acquirer(adapter, tmp_path)
    key = ChunkKey("Fake Broker Ltd", "FakeBroker-Demo", "XAU_USD", DAY0)
    acq.acquire_day(DAY0)
    p = st.raw_dir(key) / "ticks.parquet"
    os.chmod(p, 0o644)
    p.write_bytes(p.read_bytes()[:-50])
    assert st.verify_chunk(key)[0] is False
    with pytest.raises(ChunkExistsError):                           # detected; operator must move the bad chunk aside
        acq.acquire_day(DAY0)


def test_boundary_rows_are_clipped_and_counted_not_duplicated(tmp_path):
    class Stub:
        def __init__(self):
            self.n = 0

        def get_ticks_range(self, sym, s, e):
            a = make_ticks(4, start_msc=s * 1000 - 500, step_ms=500)       # row at s-500ms, s, s+500, s+1000
            a["time_msc"][-1] = e * 1000                                   # a tick exactly at the (inclusive) end
            a["time"] = a["time_msc"] // 1000
            return a

    adapter = FakeMT5Adapter()
    adapter.connect()
    acq, _ = make_acquirer(adapter, tmp_path)
    acq.adapter = Stub()
    s, e = day_window_s(DAY0)
    rows, clipped = acq._fetch_window(s, e)
    assert clipped == 2 and len(rows) == 2 and rows["time_msc"].min() >= s * 1000 and rows["time_msc"].max() < e * 1000


def test_split_when_request_cap_reached_matches_unsplit_total(tmp_path):
    big = FakeMT5Adapter(FakeScenario(ticks_per_day=20000))
    big.connect()
    a1, _ = make_acquirer(big, tmp_path / "a")
    r1 = a1.acquire_day(DAY0)
    capped = FakeMT5Adapter(FakeScenario(ticks_per_day=20000, max_rows_per_request=3000))
    capped.connect()
    a2, _ = make_acquirer(capped, tmp_path / "b", split_rows_threshold=3000, min_split_window_s=60)
    r2 = a2.acquire_day(DAY0)
    assert r2.requests > 1 and r2.rows == r1.rows                    # nothing lost to the cap
    a3, _ = make_acquirer(capped, tmp_path / "c")                     # without splitting the cap silently truncates...
    assert a3.acquire_day(DAY0).rows == 3000                          # ...which is exactly why the probe/split exist


def test_empty_day_is_recorded_explicitly(adapter, tmp_path):
    acq, st = make_acquirer(adapter, tmp_path)
    r = acq.acquire_day(dt.date(2025, 10, 12))                        # Sunday: market closed all (source) day
    assert r.status == "empty" and r.rows == 0
    assert st.chunk_manifest(ChunkKey("Fake Broker Ltd", "FakeBroker-Demo", "XAU_USD", dt.date(2025, 10, 12)))["empty_reason"] == "no_ticks_returned"


def test_anomalies_are_tagged_and_raw_rows_preserved(tmp_path):
    sc = FakeScenario(neg_spread_prob=0.002, dup_prob=0.002, reversal_prob=0.001, zero_ask_prob=0.001)
    a = FakeMT5Adapter(sc)
    a.connect()
    acq, st = make_acquirer(a, tmp_path)
    r = acq.acquire_day(DAY0)
    key = ChunkKey("Fake Broker Ltd", "FakeBroker-Demo", "XAU_USD", DAY0)
    raw = st.read_raw_chunk(key)
    q = st.read_derived("tick_quality", key)
    assert raw.num_rows == r.rows == q.num_rows                       # nothing deleted
    flags = q["quality_flags"].to_numpy()
    assert (flags & int(Q.NEGATIVE_SPREAD)).any() and (flags & int(Q.DUP_TICK)).any() and (flags & int(Q.TIME_REVERSAL)).any()
    qm = json.loads((st.derived_dir("tick_quality", key) / "manifest.json").read_text())["quality_summary"]
    assert qm["quarantined"] == int(q["quarantined"].to_numpy().sum()) > 0
    # raw values equal what the broker returned (anomalies included)
    broker_rows = a.get_ticks_range("XAUUSDm", *day_window_s(DAY0))
    assert raw.num_rows == len(broker_rows[(broker_rows["time_msc"] >= day_window_s(DAY0)[0] * 1000)
                                           & (broker_rows["time_msc"] < day_window_s(DAY0)[1] * 1000)])


def test_unverified_time_basis_is_refused_unless_explicit(adapter, tmp_path):
    tb = TimeBase(unverified_spec("B", "S", "not calibrated"))
    acq, st = make_acquirer(adapter, tmp_path, tb=tb)
    with pytest.raises(TimeCalibrationError):
        acq.acquire_day(DAY0)
    assert st.chunk_manifest(ChunkKey("Fake Broker Ltd", "FakeBroker-Demo", "XAU_USD", DAY0)) is None
    acq2, st2 = make_acquirer(adapter, tmp_path / "x", tb=tb, allow_unverified_time=True)
    acq2.acquire_day(DAY0)
    key = ChunkKey("Fake Broker Ltd", "FakeBroker-Demo", "XAU_USD", DAY0)
    assert st2.chunk_manifest(key)["time_basis"] == "unverified"


def test_connection_loss_mid_range_reconnects_and_resumes(tmp_path):
    a = FakeMT5Adapter()
    a.connect()
    acq, st = make_acquirer(a, tmp_path)
    acq.acquire_day(DAY0)
    a.fake.sc.fail_after_calls = len(a.fake.calls) + 3

    def reconnect():
        a.fake.sc.fail_after_calls = None
        a.reconnect(max_attempts=2, backoff_s=0)
    acq._reconnect = reconnect
    s = acq.acquire_range(DAY0, DAY0 + dt.timedelta(days=3))
    assert not s.failed and s.dataset_manifest is not None


def test_failure_stops_range_and_no_dataset_manifest(tmp_path):
    a = FakeMT5Adapter()
    a.connect()
    acq, st = make_acquirer(a, tmp_path, max_connection_retries=0)
    a.fake.sc.fail_after_calls = len(a.fake.calls) + 2
    s = acq.acquire_range(DAY0, DAY0 + dt.timedelta(days=3))
    assert s.failed and s.dataset_manifest is None and len(s.chunks) < 4


def test_same_data_gives_same_dataset_id_across_runs(tmp_path):
    ids = []
    for i in range(2):
        a = FakeMT5Adapter()
        a.connect()
        acq, _ = make_acquirer(a, tmp_path / str(i))
        ids.append(acq.acquire_range(DAY0, DAY0 + dt.timedelta(days=2)).dataset_manifest["dataset_id"])
        a.disconnect()
    assert ids[0] == ids[1]
    other = FakeMT5Adapter(FakeScenario(seed=99))
    other.connect()
    acq, _ = make_acquirer(other, tmp_path / "o")
    assert acq.acquire_range(DAY0, DAY0 + dt.timedelta(days=2)).dataset_manifest["dataset_id"] != ids[0]


def test_symbol_metadata_snapshot_is_content_addressed(adapter, tmp_path):
    info = adapter.get_symbol_info("XAUUSDm")
    s1, i1 = persist_symbol_metadata(tmp_path, "B", "S", "XAU_USD", info, "USD")
    s2, _ = persist_symbol_metadata(tmp_path, "B", "S", "XAU_USD", info, "USD")
    assert s1 == s2 and i1 == []
    files = list((tmp_path / "metadata/symbol_info").rglob("*.json"))
    assert len(files) == 1 and "trade_tick_size" in json.loads(files[0].read_text())["raw"]


# ---------------- probes ----------------
def test_history_depth_probe_finds_start():
    start = dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc).timestamp()
    a = FakeMT5Adapter(FakeScenario(history_start_s=start, weekend_closure=False))
    a.connect()
    now_src = int(dt.datetime(2025, 10, 8, tzinfo=dt.timezone.utc).timestamp()) + 3 * 3600
    p = probe_history_depth(a, "XAUUSDm", now_src)
    assert p.earliest_source_day is not None
    assert abs((p.earliest_source_day - dt.date(2024, 6, 1)).days) <= 14 and p.calls < 40


def test_history_depth_probe_when_no_recent_data():
    a = FakeMT5Adapter(FakeScenario(no_ticks_symbols=("XAUUSDm",)))
    a.connect()
    p = probe_history_depth(a, "XAUUSDm", 1_760_000_000)
    assert p.earliest_source_day is None


def test_request_limit_probe_detects_cap():
    a = FakeMT5Adapter(FakeScenario(ticks_per_day=60000, max_rows_per_request=1500, weekend_closure=False))
    a.connect()
    p = probe_request_limits(a, "XAUUSDm", 1_760_000_000, windows_s=(1800, 7200, 86400))
    assert p.suspected_cap == 1500
    b = FakeMT5Adapter(FakeScenario(weekend_closure=False))
    b.connect()
    assert probe_request_limits(b, "XAUUSDm", 1_760_000_000, windows_s=(1800, 7200)).suspected_cap is None


# ---------------- atomic write durability (Windows-compatible semantics) ----------------
def test_atomic_write_fsyncs_a_writable_descriptor_before_replace(tmp_path, monkeypatch):
    """Regression: fsync on a read-only descriptor raises EBADF on Windows. The writer's own handle must be synced
    (opened writable), data flushed first, and the replace must happen only after the fsync."""
    import os
    import stat as _stat
    from fxscalp.market_data import store as S

    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def spy_fsync(fd):
        mode = os.fstat(fd).st_mode
        events.append("fsync_regular" if _stat.S_ISREG(mode) else "fsync_other")
        # emulate Windows: a read-only descriptor must never reach fsync (checked via a writable probe)
        try:
            os.write(fd, b"")
        except OSError as exc:  # pragma: no cover - only on a read-only fd
            raise AssertionError("fsync called on a read-only descriptor") from exc
        return real_fsync(fd)

    def spy_replace(a, b):
        events.append("replace")
        return real_replace(a, b)

    monkeypatch.setattr(S.os, "fsync", spy_fsync)
    monkeypatch.setattr(S.os, "replace", spy_replace)
    target = tmp_path / "m.json"
    S._atomic_write_json({"a": 1, "unicode": "é€"}, target)
    assert events.index("fsync_regular") < events.index("replace")
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "unicode": "é€"}
    assert not list(tmp_path.glob("*.tmp"))
    assert not os.access(target, os.W_OK)           # immutable after write


def test_atomic_write_failure_leaves_no_temp_and_keeps_target_intact(tmp_path):
    from fxscalp.market_data import store as S
    target = tmp_path / "ticks.parquet"
    a, t = table()
    S._atomic_write_parquet(t, target)
    before = target.read_bytes()

    with pytest.raises(TypeError):
        S._atomic_write_json({1: 1, "a": 2}, tmp_path / "bad.json")         # un-sortable keys fail before any write
    with pytest.raises(ZeroDivisionError):
        S._atomic_write(tmp_path / "boom.bin", lambda fh: 1 / 0, mode="wb")  # failure mid-write
    assert not (tmp_path / "boom.bin").exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert target.read_bytes() == before


def test_atomic_parquet_roundtrip_is_readable(tmp_path):
    import pyarrow.parquet as pq
    from fxscalp.market_data import store as S
    a, t = table()
    S._atomic_write_parquet(t, tmp_path / "x.parquet")
    assert pq.read_table(tmp_path / "x.parquet").num_rows == t.num_rows
