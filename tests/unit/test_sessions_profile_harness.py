import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pytest

from fxscalp.brokers.mt5.fake import FakeMT5Adapter, FakeScenario
from fxscalp.market_data.profile import build_profile, render_markdown
from fxscalp.sessions.calendar import (DEFAULT_RULES, Session, classify_instant, classify_sessions, overlap_utc,
                                       session_flags, window_utc)
from tests.unit.conftest import DAY0
from tests.unit.test_store_and_acquire import make_acquirer

UTC = dt.timezone.utc


def at(*a):
    return dt.datetime(*a, tzinfo=UTC)


# ---------------- sessions: DST-aware, not naive UTC hours ----------------
@pytest.mark.parametrize("day,hours", [(dt.date(2025, 1, 15), 4), (dt.date(2025, 3, 19), 5),    # UK still GMT, US already EDT
                                       (dt.date(2025, 7, 16), 4), (dt.date(2025, 10, 29), 5),   # UK back to GMT, US still EDT
                                       (dt.date(2025, 11, 5), 4)])
def test_overlap_length_changes_with_mismatched_dst(day, hours):
    s, e = overlap_utc(day)
    assert (e - s) == dt.timedelta(hours=hours)


def test_london_window_utc_shifts_with_uk_dst():
    w = DEFAULT_RULES.london
    assert window_utc(dt.date(2025, 1, 15), w)[0].hour == 8
    assert window_utc(dt.date(2025, 7, 15), w)[0].hour == 7


def test_tokyo_has_no_dst():
    a = window_utc(dt.date(2025, 1, 15), DEFAULT_RULES.asia)[0].hour
    b = window_utc(dt.date(2025, 7, 15), DEFAULT_RULES.asia)[0].hour
    assert a == b == 0


@pytest.mark.parametrize("when,expected", [
    (at(2025, 7, 16, 13, 0), Session.LONDON_NEW_YORK_OVERLAP), (at(2025, 7, 16, 9, 0), Session.LONDON),
    (at(2025, 7, 16, 18, 0), Session.NEW_YORK), (at(2025, 7, 16, 2, 0), Session.ASIA),
    (at(2025, 7, 16, 21, 30), Session.OFF_HOURS), (at(2025, 7, 19, 13, 0), Session.MARKET_CLOSED),
    # same UTC clock time, different sessions because of DST:
    (at(2025, 1, 15, 12, 30), Session.LONDON), (at(2025, 7, 15, 12, 30), Session.LONDON_NEW_YORK_OVERLAP),
    (at(2025, 3, 19, 12, 30), Session.LONDON_NEW_YORK_OVERLAP), (at(2025, 11, 5, 12, 30), Session.LONDON),
])
def test_classification_examples(when, expected):
    assert classify_instant(when) is expected


def test_weekly_closure_boundaries_follow_new_york_time():
    assert classify_instant(at(2025, 7, 11, 20, 59)) is not Session.MARKET_CLOSED        # Fri 16:59 NY (EDT)
    assert classify_instant(at(2025, 7, 11, 21, 0)) is Session.MARKET_CLOSED             # Fri 17:00 NY
    assert classify_instant(at(2025, 7, 13, 20, 59)) is Session.MARKET_CLOSED            # Sun 16:59 NY
    assert classify_instant(at(2025, 7, 13, 21, 0)) is not Session.MARKET_CLOSED         # Sun 17:00 NY
    assert classify_instant(at(2025, 1, 10, 21, 59)) is not Session.MARKET_CLOSED        # winter: closes 22:00 UTC
    assert classify_instant(at(2025, 1, 10, 22, 0)) is Session.MARKET_CLOSED


def test_naive_datetime_rejected_and_vectorised_matches_scalar():
    with pytest.raises(ValueError):
        classify_instant(dt.datetime(2025, 7, 16, 13))
    base = int(at(2025, 7, 14).timestamp() * 1000)
    ms = base + np.arange(0, 7 * 86400 * 1000, 17 * 60 * 1000, dtype="int64")
    vec = classify_sessions(ms)
    for i in range(0, len(ms), 37):
        assert vec[i] == classify_instant(dt.datetime.fromtimestamp(ms[i] / 1000, tz=UTC)).value
    f = session_flags(ms)
    assert set(f) == {"in_asia", "in_london", "in_new_york", "market_closed"}


# ---------------- profile ----------------
def test_profile_is_descriptive_and_consistent_with_store(tmp_path):
    a = FakeMT5Adapter(FakeScenario(ticks_per_day=15000))
    a.connect()
    acq, st = make_acquirer(a, tmp_path)
    s = acq.acquire_range(DAY0, DAY0 + dt.timedelta(days=7))
    p = build_profile(st, "Fake Broker Ltd", "FakeBroker-Demo", "XAU_USD", DAY0, DAY0 + dt.timedelta(days=7), 0.01)
    assert p["coverage"]["ticks"] == s.dataset_manifest["record_count"]
    assert p["coverage"]["quarantined_ticks"] == 0 and not p["coverage"]["days_missing"]
    assert sum(p["tick_frequency"]["ticks_by_utc_hour"].values()) == p["coverage"]["ticks"]
    sp = p["spread"]["points"]
    assert sp["p50"] > 0 and sp["p1"] <= sp["p50"] <= sp["p99"]
    assert p["weekend_boundaries"] and p["weekend_boundaries"][0]["gap_hours"] > 40
    assert "MARKET_CLOSED" not in p["spread"]["by_session_points"] or p["spread"]["by_session_points"]["MARKET_CLOSED"]["n"] >= 0
    md = render_markdown(p)
    for h in ("## Coverage", "## Tick frequency", "## Spread", "## Weekend boundaries", "## Rollover", "## Volatility"):
        assert h in md
    assert "no trading signal" in md
    assert not re.search(r"\b(profit|alpha|edge|backtest|sharpe|win rate)\b", md, re.I)    # analysis only, no performance language
    assert json.dumps(p, default=str)


# ---------------- harness ----------------
def run_harness(tmp_path, *extra, env=None, monkeypatch=None):
    from scripts import verify_mt5
    argv = ["--fake", "--time-samples", "8", "--time-interval", "0.01", "--latency-samples", "5",
            "--output-dir", str(tmp_path / "rep"), "--data-root", str(tmp_path / "data"), "--no-probe", *extra]
    code = verify_mt5.main(argv)
    reports = sorted((tmp_path / "rep").glob("mt5_verify_*.json"))
    return code, json.loads(reports[-1].read_text()), reports[-1].read_text()


def test_harness_fake_run_passes_but_never_claims_real_verification(tmp_path):
    code, rep, _ = run_harness(tmp_path)
    assert code == 0 and rep["mode"] == "FAKE_NOT_MT5" and rep["real_mt5_verified"] is False
    assert rep["verdict"] == "PASS_FAKE_PIPELINE_ONLY" and any("fake" in u.lower() for u in rep["unverified"])
    ids = {c["id"]: c["status"] for c in rep["checks"]}
    for must in ("initialize", "account_gate", "symbol_XAU_USD", "symbol_EUR_USD", "symbol_GBP_USD", "metadata_XAU_USD",
                 "time_offset", "ticks", "bid_ask", "bars", "latency", "no_trading"):
        assert ids.get(must) in ("PASS", "WARN"), (must, ids.get(must))
    assert rep["account"]["class"] == "DEMO" and rep["time_semantics"]["spec"]["basis"] in ("server_fixed_offset", "utc_verified")
    assert rep["ticks"]["rows"] > 0 and rep["no_trading_attestation"]["order_functions_called"] == []
    assert rep["no_trading_attestation"]["fake_trading_calls"] == []


def test_harness_rejects_real_account_and_collects_nothing(tmp_path):
    code, rep, _ = run_harness(tmp_path, "--fake-trade-mode", "2")
    assert code == 3 and rep["verdict"] == "FAIL" and rep["real_mt5_verified"] is False
    assert any(c["id"] == "fatal" and "AccountNotAllowedError" in c["title"] for c in rep["checks"])
    assert "ticks" not in rep and not (tmp_path / "data" / "raw").exists()


def test_report_never_contains_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("MT5_PASSWORD", "Sup3rS3cretPW")
    monkeypatch.setenv("MT5_LOGIN", "87654321")
    code, rep, text = run_harness(tmp_path)
    assert "Sup3rS3cretPW" not in text and "87654321" not in text and "12345678" not in text
    assert rep["account"]["login_masked"] == "***678"
    assert "password" not in text.lower() or "***REDACTED***" in text or '"password"' not in text


def test_harness_detects_unusable_time_calibration(tmp_path):
    from scripts import verify_mt5
    code = verify_mt5.main(["--fake", "--time-samples", "2", "--time-interval", "0.01", "--output-dir", str(tmp_path / "r"),
                            "--data-root", str(tmp_path / "d"), "--no-probe", "--latency-samples", "3"])
    rep = json.loads(sorted((tmp_path / "r").glob("*.json"))[-1].read_text())
    assert rep["time_semantics"]["spec"]["basis"] == "unverified"
    assert next(c for c in rep["checks"] if c["id"] == "time_offset")["status"] == "FAIL" and code == 1


# ---------------- source guards: nothing can trade ----------------
SRC = Path(__file__).resolve().parents[2] / "src" / "fxscalp"
FORBIDDEN = re.compile(r"\b(order_send|order_check|order_calc_margin|order_calc_profit|positions_get|orders_get)\b")


def test_trading_functions_are_only_mentioned_in_the_blocklist_and_fake_tripwires():
    offenders = []
    for p in SRC.rglob("*.py"):
        if p.name in ("api.py", "fake.py"):
            continue
        if FORBIDDEN.search(p.read_text()):
            offenders.append(str(p.relative_to(SRC)))
    assert offenders == []


def test_only_the_mt5_package_imports_metatrader5():
    for p in SRC.rglob("*.py"):
        if "brokers/mt5" in str(p):
            continue
        assert not re.search(r"^\s*(import|from)\s+MetaTrader5", p.read_text(), re.M), p


def test_no_message_broker_or_microservice_dependencies():
    deps = (SRC.parents[1] / "pyproject.toml").read_text().lower()
    for bad in ("kafka", "redis", "rabbit", "pika", "kubernetes", "celery", "zmq"):
        assert bad not in deps
