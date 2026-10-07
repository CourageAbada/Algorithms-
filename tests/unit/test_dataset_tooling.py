"""Deterministic logic of the dataset tooling: integrity verification, dataset id, profile aggregation, representative-day
selection, histogram utilities and the resumable backwards acquisition orchestrator."""

from __future__ import annotations

import datetime as dt
import json
import os
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxscalp.brokers.mt5.fake import FakeMT5Adapter, FakeScenario
from fxscalp.market_data.acquire import AcquisitionConfig, TickAcquirer
from fxscalp.market_data.integrity import verify_dataset_integrity
from fxscalp.market_data.profiling import BIN, NBINS, hist_add, hist_q, hist_summary, pct
from fxscalp.market_data.retry import RetryPolicy
from fxscalp.market_data.sampling import pick_days
from fxscalp.market_data.store import ChunkKey, TickStore, compute_dataset_id
from tests.unit.conftest import DAY0, make_timebase

BROKER, SERVER = "Fake Broker Ltd", "FakeBroker-Demo"
FAST = RetryPolicy(max_retries=1, backoff_base_s=0.01, backoff_max_s=0.02)


def writable(p: Path) -> None:
    os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)


def build(tmp_path: Path, days: int = 5, scenario: FakeScenario | None = None, root_name: str = "data"):
    a = FakeMT5Adapter(scenario or FakeScenario(ticks_per_day=6000))
    a.connect()
    st = TickStore(tmp_path / root_name)
    acq = TickAcquirer(a, st, make_timebase(a), broker=BROKER, server=SERVER, canonical="XAU_USD", broker_symbol="XAUUSDm",
                       symbol_info=a.get_symbol_info("XAUUSDm"), account_fingerprint="fp", account_currency="USD",
                       config=AcquisitionConfig(retry=FAST), sleep=lambda s: None)
    s = acq.acquire_range(DAY0, DAY0 + dt.timedelta(days=days - 1))
    assert s.dataset_manifest is not None and not s.failed
    return a, st, s.dataset_manifest


def key(day: dt.date) -> ChunkKey:
    return ChunkKey(BROKER, SERVER, "XAU_USD", day)


# ============================ dataset integrity ============================
def test_clean_dataset_verifies_and_counts_everything(tmp_path):
    _, st, man = build(tmp_path)
    r = verify_dataset_integrity(st, man)
    assert r["ok"] and r["problems"] == [] and r["total_rows_verified"] == man["record_count"]
    assert r["chunks_in_manifest"] == 5 and r["nonempty_chunks"] >= 3 and r["parquet_bytes"]["total"] > 0


def test_checksum_mismatch_is_detected(tmp_path):
    _, st, man = build(tmp_path)
    f = st.raw_dir(key(DAY0)) / "ticks.parquet"
    writable(f)
    b = bytearray(f.read_bytes())
    b[len(b) // 2] ^= 0xFF
    f.write_bytes(bytes(b))
    r = verify_dataset_integrity(st, man)
    assert not r["ok"] and any("verify_chunk failed" in p and "checksum" in p for p in r["problems"])


def test_missing_chunk_manifest_and_missing_derived_artifacts_are_detected(tmp_path):
    _, st, man = build(tmp_path)
    m = st.raw_dir(key(DAY0 + dt.timedelta(days=1))) / "manifest.json"
    writable(m)
    m.unlink()
    q = st.derived_dir("tick_quality", key(DAY0 + dt.timedelta(days=2))) / "tick_quality.parquet"
    writable(q)
    q.unlink()
    r = verify_dataset_integrity(st, man)
    assert any("raw manifest missing" in p for p in r["problems"])
    assert any("tick_quality artifact missing" in p for p in r["problems"])


def test_missing_and_duplicate_days_in_the_dataset_manifest_are_detected(tmp_path):
    _, st, man = build(tmp_path)
    missing = {**man, "chunks": [c for i, c in enumerate(man["chunks"]) if i != 2]}
    r = verify_dataset_integrity(st, missing)
    assert any("not contiguous" in p for p in r["problems"]) and any("not reproducible" in p for p in r["problems"])
    dup = {**man, "chunks": man["chunks"] + [man["chunks"][-1]]}
    assert any("duplicate source days" in p for p in verify_dataset_integrity(st, dup)["problems"])


def test_overlapping_source_day_windows_are_detected(tmp_path):
    _, st, man = build(tmp_path)
    k = key(DAY0 + dt.timedelta(days=1))
    mp = st.raw_dir(k) / "manifest.json"
    writable(mp)
    m = json.loads(mp.read_text(encoding="utf-8"))
    s, e = m["requested_window_source_s"]
    m["requested_window_source_s"] = [s - 3600, e]                  # starts an hour before the day: overlaps the previous day
    mp.write_text(json.dumps(m), encoding="utf-8")
    r = verify_dataset_integrity(st, man)
    assert any("requested window" in p for p in r["problems"]) and any("overlap/gap" in p for p in r["problems"])


def test_ticks_outside_their_own_day_window_are_detected_as_a_collision(tmp_path):
    """A chunk holding another day's ticks is not the same as its manifest claiming so: verify via copied content."""
    _, st, man = build(tmp_path)
    d1, d2 = DAY0, DAY0 + dt.timedelta(days=1)
    src, dst = st.raw_dir(key(d1)) / "ticks.parquet", st.raw_dir(key(d2)) / "ticks.parquet"
    writable(dst)
    dst.write_bytes(src.read_bytes())                                # day-2 chunk now contains day-1 ticks
    r = verify_dataset_integrity(st, man)
    assert not r["ok"] and any(str(d2) in p for p in r["problems"])


def test_stray_temp_files_are_reported(tmp_path):
    _, st, man = build(tmp_path)
    (st.root / "raw" / "leftover.parquet.tmp").write_bytes(b"x")
    assert any("leftover temp files" in p for p in verify_dataset_integrity(st, man)["problems"])


# ============================ deterministic dataset id ============================
def test_dataset_id_is_deterministic_across_runs_and_roots(tmp_path):
    _, _, m1 = build(tmp_path, root_name="a")
    _, _, m2 = build(tmp_path, root_name="b")
    assert m1["dataset_id"] == m2["dataset_id"] == compute_dataset_id(m1)
    assert m1["manifest_checksum_sha256"] != "" and m1["collected_at_utc"] is not None


def test_dataset_id_changes_with_content_identity_or_timebase(tmp_path):
    _, _, m = build(tmp_path)
    changed = {**m, "chunks": [{**m["chunks"][0], "raw_content_sha256": "0" * 64}] + m["chunks"][1:]}
    assert compute_dataset_id(changed) != m["dataset_id"]
    assert compute_dataset_id({**m, "timebase_spec_sha256": "f" * 64}) != m["dataset_id"]
    assert compute_dataset_id({**m, "collected_at_utc": "2030-01-01", "software": {"x": 1}}) == m["dataset_id"]   # not identity
    tampered = {**m, "dataset_id": "tickraw-deadbeef"}
    assert any("not reproducible" in p for p in verify_dataset_integrity(TickStore(tmp_path / "data"), tampered)["problems"])


# ============================ histogram utilities ============================
def test_histogram_quantiles_match_numpy_within_a_bin():
    rng = np.random.default_rng(1)
    v = np.round(rng.gamma(3.0, 3.0, 50_000), 3)
    h = np.zeros(NBINS, dtype="int64")
    hist_add(h, v[:20_000])
    hist_add(h, v[20_000:])                                          # chunked accumulation == one pass
    for q in (5, 50, 95, 99):
        assert abs(hist_q(h, (q,))[f"p{q:g}"] - np.percentile(v, q)) <= 2 * BIN
    s = hist_summary(h)
    assert s["n"] == len(v) and abs(s["mean"] - v.mean()) < BIN


def test_histogram_ignores_non_finite_and_negative_values_and_clips_the_tail():
    h = np.zeros(NBINS, dtype="int64")
    hist_add(h, np.array([np.nan, -1.0, np.inf, 5.0, 1e9]))
    assert int(h.sum()) == 2 and h[NBINS - 1] == 1 and h[int(5.0 / BIN)] == 1   # NaN, -1 and +inf dropped; 1e9 clipped into the last bin
    assert hist_summary(np.zeros(NBINS, dtype="int64")) == {"n": 0}
    assert pct(np.array([]), (50,)) == {"p50": None}


# ============================ representative-day selection ============================
def _profile_frame():
    days = pd.date_range("2026-05-04", periods=15)
    df = pd.DataFrame({"day": days.strftime("%Y-%m-%d"), "weekday": days.strftime("%a"),
                       "n_ticks": np.linspace(300_000, 700_000, 15).astype(int),
                       "range_bps": np.linspace(100, 300, 15), "spread_p99": np.linspace(10, 40, 15),
                       "spread_max": np.linspace(50, 150, 15), "week_open": False, "week_close": False})
    df.loc[df.index[[0, 7]], "week_open"] = True
    df.loc[df.index[[4, 11]], "week_close"] = True
    return df


def test_pick_days_is_deterministic_distinct_and_follows_the_extremes():
    df = _profile_frame()
    a, b = pick_days(df), pick_days(df.sample(frac=1.0, random_state=3).reset_index(drop=True))
    assert a == b                                                    # order of rows does not matter
    assert len(set(a.values())) == len(a)                            # every selected day is distinct
    assert a["high_volatility"] == df.loc[df.range_bps.idxmax(), "day"]
    assert a["wide_spread"] not in (a["high_volatility"],)           # already used -> next-best, never reused
    assert a["low_activity"] == df.loc[df.n_ticks.idxmin(), "day"]
    assert df.set_index("day").loc[a["week_open"], "week_open"] and df.set_index("day").loc[a["week_close"], "week_close"]
    assert df.set_index("day").loc[a["normal_activity"], "weekday"] in ("Tue", "Wed", "Thu")


def test_pick_days_ignores_closed_days_and_handles_missing_flags():
    df = _profile_frame().drop(columns=["week_open", "week_close"])
    closed = df.iloc[[0]].copy()
    closed["n_ticks"] = 0
    out = pick_days(pd.concat([closed.assign(day="2026-01-01", weekday="Thu"), df], ignore_index=True))
    assert "2026-01-01" not in out.values() and "week_open" not in out


# ============================ profile aggregation (end-to-end on a synthetic store) ============================
def test_profile_script_aggregates_match_independent_computation(tmp_path):
    from scripts import profile_real_dataset
    _, st, man = build(tmp_path, days=7)
    out = tmp_path / "prof"
    rc = profile_real_dataset.main(["--data-root", str(st.root), "--broker", BROKER, "--server", SERVER, "--start", str(DAY0),
                                    "--end", str(DAY0 + dt.timedelta(days=6)), "--out-dir", str(out)])
    assert rc == 0
    df = pd.read_csv(out / "XAU_USD_daily_profile.csv")
    prof = json.loads((out / "XAU_USD_profile.json").read_text(encoding="utf-8"))
    assert prof["total_raw_ticks"] == man["record_count"] == int(df["n_ticks"].sum())
    assert prof["n_source_days"] == 7 and prof["n_missing_chunks"] == 0
    assert prof["n_trading_days_with_ticks"] == int((df["n_ticks"] > 0).sum())
    # per-day numbers equal what an independent read of the chunk gives
    for _, r in df[df.n_ticks > 0].iterrows():
        k = key(dt.date.fromisoformat(r["day"]))
        spread = st.read_derived("tick_basic", k)["spread_points"].to_numpy()
        assert r["n_ticks"] == len(spread) == st.chunk_manifest(k)["record_count"]
        assert abs(r["spread_median"] - np.median(spread)) < 1e-2 and abs(r["spread_max"] - spread.max()) < 1e-2
    # the overall histogram quantile matches the pooled spreads
    pooled = np.concatenate([st.read_derived("tick_basic", key(dt.date.fromisoformat(d)))["spread_points"].to_numpy()
                             for d in df[df.n_ticks > 0]["day"]])
    assert abs(prof["spread_points_overall"]["p50"] - np.median(pooled)) <= 2 * BIN and prof["spread_points_overall"]["n"] == len(pooled)
    assert sum(prof["ticks_by_utc_hour"].values()) == prof["total_raw_ticks"] == sum(prof["ticks_by_session"].values())
    # fake data is verified time basis (unbounded legacy spec): nothing uncertain; weekend days are closed, not "missing"
    assert prof["time_basis"]["ticks_time_basis_uncertain"] == 0 and prof["n_closed_days_no_ticks"] == 2
    assert prof["missing_periods"]["weekdays_with_no_ticks"] == []


# ============================ resumable backwards orchestrator ============================
def _prepare_orchestrator(tmp_path, monkeypatch):
    from fxscalp.brokers.symbols import SymbolMap  # noqa: F401
    from fxscalp.workflows import (calibrate_timebase, load_instrument_specs, run_discovery, save_calibration_record,
                                   timebase_file)
    from scripts import acquire_history
    a = FakeMT5Adapter(FakeScenario(ticks_per_day=3000))
    ident = a.connect()
    root, cfg = tmp_path / "data", tmp_path / "cfg"
    specs = load_instrument_specs(Path(__file__).resolve().parents[2] / "configs")
    run_discovery(a, specs, cfg, ident.company, ident.server)
    spec, _ = calibrate_timebase(a, "XAUUSDm", broker=ident.company, server=ident.server, samples=8, interval_s=0.01,
                                 bound_validity=False)
    from fxscalp.market_data.timebase import TimeBase, build_calibration_record
    TimeBase(spec).save(timebase_file(root, ident.company, ident.server))
    rec_id = build_calibration_record(spec)["record_id"]
    a.disconnect()
    monkeypatch.setattr(acquire_history, "make_adapter", lambda env: FakeMT5Adapter(FakeScenario(ticks_per_day=3000)))
    args = ["--data-root", str(root), "--config-dir", str(cfg), "--env-file", str(tmp_path / "no.env"),
            "--expect-calibration", rec_id, "--min-call-interval", "0", "--start", "2025-10-06", "--end", "2025-10-10"]
    return acquire_history, root, args, rec_id


def test_orchestrator_stops_cleanly_resumes_and_never_redownloads_verified_chunks(tmp_path, monkeypatch):
    mod, root, args, _ = _prepare_orchestrator(tmp_path, monkeypatch)
    rc = mod.main(args + ["--max-chunks", "2"])
    assert rc == 3                                                   # stopped cleanly by the chunk budget
    st = TickStore(root)
    done = {d for d in (dt.date(2025, 10, 6) + dt.timedelta(days=i) for i in range(5))
            if st.chunk_manifest(key(d)) is not None}
    assert done == {dt.date(2025, 10, 10), dt.date(2025, 10, 9)}     # newest first
    first = {d: st.chunk_manifest(key(d))["raw_content_sha256"] for d in done}
    rc = mod.main(args)                                              # resume
    assert rc == 0
    for d in done:
        assert st.chunk_manifest(key(d))["raw_content_sha256"] == first[d]
    log = [json.loads(x) for x in (root / "acquisition_monitor.jsonl").read_text(encoding="utf-8").splitlines()]
    resumed = [r for r in log if r["event"] == "chunk" and r["day"] in {d.isoformat() for d in done}][-2:]
    assert [r["status"] for r in resumed] == ["skipped_verified"] * 2        # verified chunks were not fetched again
    fin = [r for r in log if r["event"] == "finished"]
    assert len(fin) == 1 and fin[0]["records"] > 0 and list(st.iter_dataset_manifests(BROKER, SERVER, "XAU_USD"))
    assert verify_dataset_integrity(st, next(st.iter_dataset_manifests(BROKER, SERVER, "XAU_USD")))["ok"]


def test_orchestrator_refuses_wrong_calibration_and_incomplete_current_day(tmp_path, monkeypatch, capsys):
    mod, root, args, rec_id = _prepare_orchestrator(tmp_path, monkeypatch)
    bad = [x if x != rec_id else "timecal-0000000000000000" for x in args]
    assert mod.main(bad) == 2
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    future = args[:-2] + ["--end", (dt.date.fromisoformat(today) + dt.timedelta(days=1)).isoformat()]
    assert mod.main(future) == 2
    assert "not a complete source day" in capsys.readouterr().err
    assert not (root / "raw").exists()                               # nothing was requested


def test_orchestrator_stops_on_a_failed_chunk_and_keeps_verified_ones(tmp_path, monkeypatch):
    mod, root, args, _ = _prepare_orchestrator(tmp_path, monkeypatch)
    state = {"n": 0}

    def flaky(env):
        # first connection: terminal error on the 3rd copy_ticks_range (after 2 good days); exhausts the 1 allowed retry
        return FakeMT5Adapter(FakeScenario(ticks_per_day=3000, terminal_error_function="copy_ticks_range",
                                           terminal_error_first_n=0))
    a = FakeMT5Adapter(FakeScenario(ticks_per_day=3000))
    orig = a.fake._enter

    def enter(name):
        if name == "copy_ticks_range":
            state["n"] += 1
            if state["n"] >= 3:
                a.fake._err = (-1, "Terminal: Call failed")
                a.fake.calls.append(name)
                return False
        return orig(name)
    a.fake._enter = enter
    monkeypatch.setattr(mod, "make_adapter", lambda env: a)
    rc = mod.main(args + ["--max-retries", "1"])
    assert rc == 1                                                   # chunk failed after bounded retries -> clean stop
    st = TickStore(root)
    assert st.chunk_manifest(key(dt.date(2025, 10, 10))) is not None and st.chunk_manifest(key(dt.date(2025, 10, 9))) is not None
    assert st.chunk_manifest(key(dt.date(2025, 10, 8))) is None
    log = [json.loads(x) for x in (root / "acquisition_monitor.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(r["event"] == "chunk_failed" for r in log) and any(r["event"] == "stopped" for r in log)
