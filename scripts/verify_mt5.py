"""MT5 verification harness (READ-ONLY): ``python -m scripts.verify_mt5``.

Verifies connection, account class, symbol discovery, metadata, ticks, bars, time semantics, DOM and latency
against a real MT5 terminal (Windows, DEMO account). ``--fake`` runs the same pipeline against the in-memory
fake: that validates OUR CODE only and is reported as ``real_mt5_verified: false``.

It never places an order, never calls any order/position/history function (the adapter exposes only a read-only
allow-list), and never writes credentials to the report.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fxscalp.brokers.accounts import AccountClass
from fxscalp.brokers.errors import BrokerError, NoTickDataError, TradingDisabledError
from fxscalp.brokers.mt5.adapter import MT5BrokerAdapter
from fxscalp.brokers.mt5.convert import compare_spread_with_tick
from fxscalp.core.env import load_dotenv
from fxscalp.core.provenance import software_versions
from fxscalp.market_data import quality as Qm
from fxscalp.market_data.acquire import persist_symbol_metadata, probe_history_depth, probe_request_limits
from fxscalp.market_data.timebase import TimeBase
from fxscalp.monitoring.logging_setup import REDACTED, configure_logging
from fxscalp.workflows import (calibrate_timebase, load_instrument_specs, make_adapter, run_discovery,
                               timebase_file)

REPORT_VERSION = "mt5_verify/1"
PRIMARY = "XAU_USD"
log = logging.getLogger("fxscalp.verify")


class Report:
    def __init__(self, fake: bool):
        self.data: dict[str, Any] = {
            "report_version": REPORT_VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "FAKE_NOT_MT5" if fake else "REAL_MT5", "real_mt5_verified": False,
            "environment": {"platform": platform.platform(), "machine": platform.machine(),
                            "python": platform.python_version(), "software": software_versions()},
            "checks": [], "symbols": {}, "unverified": []}

    def check(self, cid: str, title: str, status: str, detail: str = "", **evidence: Any) -> None:
        assert status in ("PASS", "FAIL", "WARN", "SKIP")
        self.data["checks"].append({"id": cid, "title": title, "status": status, "detail": detail, "evidence": evidence})
        print(f"[{status:4}] {cid}: {title}" + (f" - {detail}" if detail else ""))

    def failed(self) -> bool:
        return any(c["status"] == "FAIL" for c in self.data["checks"])


def _find_window_with_ticks(adapter: MT5BrokerAdapter, sym: str, now_source_s: int, hours: int) -> tuple[int, int, np.ndarray]:
    """Most recent window of ``hours`` hours (source time) that contains ticks, searching back up to 10 days."""
    for back in range(0, 10):
        if back == 0:
            s0 = now_source_s - hours * 3600
        else:   # same clock window on earlier days (midday source time: inside the trading day on weekdays)
            day = (datetime.fromtimestamp(now_source_s, tz=timezone.utc) - timedelta(days=back)).date()
            s0 = int(datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc).timestamp())
        arr = _try(adapter, sym, s0, s0 + hours * 3600)
        if len(arr):
            return s0, s0 + hours * 3600, arr
    raise NoTickDataError(f"no ticks found for {sym} in the last 10 days")


def _try(adapter: MT5BrokerAdapter, sym: str, s: int, e: int) -> np.ndarray:
    try:
        return adapter.get_ticks_range(sym, s, e)
    except NoTickDataError:
        return np.empty(0)


def run(args: argparse.Namespace) -> int:
    configure_logging("INFO")
    env = dict(os.environ)
    if not args.fake:
        load_dotenv(Path(args.env_file), env)
    rep = Report(args.fake)
    out_dir, config_dir, data_root = Path(args.output_dir), Path(args.config_dir), Path(args.data_root)
    scenario = None
    if args.fake:
        from fxscalp.brokers.mt5.fake import FakeScenario
        from fxscalp.market_data.timebase import ServerTimeRule
        scenario = FakeScenario(trade_mode=args.fake_trade_mode, rule=ServerTimeRule.fixed(args.fake_offset_hours * 3600))
    adapter: MT5BrokerAdapter | None = None
    code = 0
    try:
        adapter = make_adapter(env, fake=args.fake, fake_scenario=scenario)
        ident = adapter.connect()
        rep.check("initialize", "MT5 initialize + verification", "PASS", f"state={adapter.state.value}")
        rep.data["account"] = {"login_masked": ident.login_masked, "login_fingerprint": ident.login_fingerprint,
                               "server": ident.server, "company": ident.company, "class": ident.account_class.value,
                               "trade_mode_raw": ident.trade_mode_raw, "margin_mode_raw": ident.margin_mode_raw,
                               "leverage": ident.leverage, "currency": ident.currency}
        rep.check("account_gate", "Account class is DEMO", "PASS" if ident.account_class == AccountClass.DEMO else "FAIL",
                  ident.account_class.value)
        tinfo = adapter.get_terminal_info()
        rep.data["terminal"] = {"connected": tinfo.connected, "trade_allowed": tinfo.trade_allowed, "build": tinfo.build,
                                "version": tinfo.version, "company": tinfo.company, "name": tinfo.name,
                                "ping_last_us": tinfo.ping_last_us, "fields_returned": sorted(tinfo.raw)}
        rep.check("terminal_info", "Terminal info", "PASS" if tinfo.connected else "FAIL",
                  f"build={tinfo.build} version={tinfo.version} ping_us={tinfo.ping_last_us}")
        broker, server = ident.company or "unknown", ident.server or "unknown"

        # ---- discovery -------------------------------------------------------------------------
        specs = load_instrument_specs(config_dir)
        map_dir = Path(args.map_dir) if args.map_dir else (out_dir / "fake_symbol_maps" if args.fake else config_dir)
        results, mpath = run_discovery(adapter, specs, map_dir, broker, server, write=not args.no_write_map)
        rep.data["symbol_map_file"] = str(mpath)
        for r in results:
            rep.data["symbols"][r.canonical] = {"discovery": {"status": r.status, "chosen": r.chosen, "detail": r.detail,
                                                              "candidates": [{"name": c.name, "tier": c.tier,
                                                                              "rejected": c.rejected} for c in r.candidates]}}
            rep.check(f"symbol_{r.canonical}", f"{r.canonical} symbol resolution", "PASS" if r.ok else
                      ("FAIL" if r.canonical == PRIMARY else "WARN"), f"{r.status}: {r.chosen or r.detail}")
        # ---- per-symbol metadata + latest tick ---------------------------------------------------
        for r in results:
            if not r.ok:
                continue
            sym = r.chosen
            info = adapter.get_symbol_info(sym)
            sha, issues = persist_symbol_metadata(data_root, broker, server, r.canonical, info, ident.currency)
            entry = rep.data["symbols"][r.canonical]
            entry.update({"broker_symbol": sym, "metadata_sha256": sha, "consistency_issues": issues,
                          "metadata_typed": {k: getattr(info, k) for k in info.__dataclass_fields__ if k != "raw"},
                          "metadata_fields_returned": sorted(info.raw)})
            errs = [i for i in issues if i["severity"] == "error"]
            rep.check(f"metadata_{r.canonical}", f"{r.canonical} metadata captured", "FAIL" if errs else
                      ("WARN" if issues else "PASS"), f"{len(issues)} consistency findings", issues=issues)
            try:
                tick = adapter.get_latest_tick(sym)
                entry["latest_tick"] = {"time_raw": tick.time_raw, "time_msc_raw": tick.time_msc_raw, "bid": tick.bid,
                                        "ask": tick.ask, "last": tick.last, "flags": tick.flags,
                                        "spread_compare": compare_spread_with_tick(info, tick)}
                rep.check(f"latest_tick_{r.canonical}", f"{r.canonical} latest tick", "PASS" if tick.ask >= tick.bid > 0 else "FAIL",
                          f"bid={tick.bid} ask={tick.ask}")
            except BrokerError as exc:
                rep.check(f"latest_tick_{r.canonical}", f"{r.canonical} latest tick", "WARN", str(exc))

        primary = rep.data["symbols"].get(PRIMARY, {})
        psym = primary.get("broker_symbol")
        if psym is None:
            rep.check("primary_missing", "XAU_USD unavailable: cannot continue with time/ticks", "FAIL")
            raise SystemExit(4)

        # ---- time semantics ----------------------------------------------------------------------
        spec, ev = calibrate_timebase(adapter, psym, broker=broker, server=server, samples=args.time_samples,
                                      interval_s=args.time_interval, dst_weeks=args.dst_weeks)
        tb = TimeBase(spec)
        tb.save(timebase_file(data_root, broker, server))
        rep.data["time_semantics"] = {"spec": spec.to_json(), "evidence": ev,
                                      "interpretation": f"source_time - UTC = {spec.estimate.offset_s if spec.estimate else None} s",
                                      "documentation_claim": "MT5 docs: times are UTC",
                                      "documentation_claim_status": "tested above (not assumed)"}
        est = spec.estimate
        rep.check("time_offset", "Server-time offset measured", "PASS" if est and est.status == "ok" else "FAIL",
                  f"offset_s={est.offset_s if est else None} residual_s={None if not est else est.residual_s} "
                  f"basis={spec.basis.value}", notes=list(est.notes) if est else [])
        rep.check("time_dst", "DST behaviour determined", "PASS" if spec.dst_determined else "WARN",
                  "determined from weekly-open evidence (assumption-labelled)" if spec.dst_determined else
                  "not determined (use --dst-weeks 52 on a connected terminal with deep history)")

        # ---- ticks -------------------------------------------------------------------------------
        off = est.offset_s if est and est.offset_s is not None else 0
        now_source = int(time.time()) + off
        s0, e0, _ = _find_window_with_ticks(adapter, psym, now_source, args.tick_hours)
        t0 = time.perf_counter()
        arr = adapter.get_ticks_range(psym, s0, e0)
        dt_ticks = time.perf_counter() - t0
        q = Qm.assess(arr["time"], arr["time_msc"], arr["bid"], arr["ask"], arr["last"], arr["volume"], arr["flags"])
        norm = tb.normalize(arr["time_msc"], allow_unverified=True)
        rep.data["ticks"] = {
            "window_source_s": [s0, e0], "rows": int(len(arr)), "fetch_seconds": round(dt_ticks, 4),
            "rows_per_s": round(len(arr) / dt_ticks, 1) if dt_ticks > 0 else None,
            "dtype": [(n, str(arr.dtype[n])) for n in arr.dtype.names],
            "first_source_time_msc": int(arr["time_msc"][0]) if len(arr) else None,
            "last_source_time_msc": int(arr["time_msc"][-1]) if len(arr) else None,
            "first_normalized_utc": None if not len(arr) else str(np.datetime64(int(norm.utc_ms[0]), "ms")),
            "volume_real_present": bool(np.isfinite(arr["volume_real"]).any()) if len(arr) else None,
            "flag_histogram": q.summary["tick_flag_histogram"], "quality": q.summary}
        rep.check("ticks", f"Historical ticks ({PRIMARY})", "PASS" if len(arr) else "FAIL",
                  f"{len(arr):,} rows in {dt_ticks:.2f}s; quarantined={q.summary['quarantined']}")
        bad = q.summary["counts"]
        rep.check("bid_ask", "bid/ask validation", "PASS" if not (bad["NEGATIVE_SPREAD"] or bad["MISSING_BID"] or bad["MISSING_ASK"])
                  else "WARN", f"negative={bad['NEGATIVE_SPREAD']} zero={bad['ZERO_SPREAD']} missing={bad['MISSING_BID'] + bad['MISSING_ASK']}")
        rep.check("timestamp_fields", "time vs time_msc consistency", "PASS" if not bad["SEC_MSEC_MISMATCH"] else "WARN",
                  f"mismatches={bad['SEC_MSEC_MISMATCH']}")
        # ---- bars --------------------------------------------------------------------------------
        try:
            rates = adapter.get_rates_range(psym, "M1", s0, e0)
            rep.data["bars"] = {"timeframe": "M1", "rows": int(len(rates)), "fields": list(rates.dtype.names)}
            rep.check("bars", "Historical bars (M1)", "PASS" if len(rates) else "WARN", f"{len(rates)} bars")
        except BrokerError as exc:
            rep.check("bars", "Historical bars (M1)", "WARN", str(exc))
        # ---- DOM ---------------------------------------------------------------------------------
        if args.skip_dom:
            rep.check("dom", "Depth of market", "SKIP", "--skip-dom")
        else:
            book = adapter.get_market_book(psym)
            rep.data["dom"] = {"available": book is not None, "levels": None if book is None else len(book)}
            rep.check("dom", "Depth of market (read-only)", "PASS" if book else "WARN", "available" if book else "not available")
        # ---- limits / depth ----------------------------------------------------------------------
        if not args.no_probe:
            lim = probe_request_limits(adapter, psym, s0)
            rep.data["request_limits"] = {"rows": lim.rows, "suspected_cap": lim.suspected_cap, "note": lim.note}
            dep = probe_history_depth(adapter, psym, now_source)
            rep.data["history_depth"] = {"earliest_source_day": None if dep.earliest_source_day is None else dep.earliest_source_day.isoformat(),
                                         "calls": dep.calls, "note": dep.note, "samples": dep.samples}
            rep.check("history_depth", "XAU_USD tick history depth probe", "PASS" if dep.earliest_source_day else "WARN", dep.note
                      + f" earliest={dep.earliest_source_day}")
        # ---- latency -----------------------------------------------------------------------------
        lat = []
        for _ in range(args.latency_samples):
            t = time.perf_counter()
            adapter.get_latest_tick(psym)
            lat.append(time.perf_counter() - t)
        rep.data["latency"] = {"latest_tick_ms": {"n": len(lat), "p50": 1000 * float(np.percentile(lat, 50)),
                                                   "p95": 1000 * float(np.percentile(lat, 95)), "max": 1000 * max(lat)},
                               "per_function": adapter.latency_stats()}
        rep.check("latency", "Basic call latency", "PASS", f"latest_tick p50={rep.data['latency']['latest_tick_ms']['p50']:.1f}ms")
        # ---- no-trading attestation --------------------------------------------------------------
        try:
            adapter.submit_order(None)  # type: ignore[arg-type]
            rep.check("no_trading", "Order submission blocked", "FAIL", "submit_order did not raise")
        except TradingDisabledError:
            rep.check("no_trading", "Order submission blocked (TradingDisabledError)", "PASS")
        if args.fake:
            rep.data["no_trading_attestation"] = {"fake_trading_calls": list(adapter.fake.trading_calls)}  # type: ignore[attr-defined]
        rep.data["no_trading_attestation"] = {**rep.data.get("no_trading_attestation", {}),
                                              "api_allow_list_only": True, "order_functions_called": []}
        rep.data["real_mt5_verified"] = (not args.fake) and not rep.failed()
    except SystemExit as exc:
        code = int(exc.code or 1)
    except BrokerError as exc:
        rep.check("fatal", type(exc).__name__, "FAIL", str(exc))
        code = 3 if type(exc).__name__ == "AccountNotAllowedError" else 2
    except Exception as exc:  # noqa: BLE001 - the report must still be written
        rep.check("fatal", type(exc).__name__, "FAIL", str(exc))
        code = 2
    finally:
        if adapter is not None:
            try:
                adapter.disconnect()
            except Exception:  # noqa: BLE001
                pass
    rep.data["unverified"] = _unverified(args.fake, rep)
    rep.data["verdict"] = "FAIL" if (rep.failed() or code) else ("PASS_FAKE_PIPELINE_ONLY" if args.fake else "PASS")
    _write(rep, out_dir, env)
    return code or (1 if rep.failed() else 0)


def _unverified(fake: bool, rep: Report) -> list[str]:
    base = ["SYMBOL_FILLING_* bit values (needs order_check; not run in Phase 1)", "hedge-vs-close behaviour",
            "thread-safety of the MT5 package", "initialize() launching the terminal",
            "tick-flag semantics beyond observed flag histogram"]
    if fake:
        return ["EVERYTHING about real MT5 behaviour: this run used the in-memory fake"] + base
    ts = rep.data.get("time_semantics", {}).get("spec", {})
    if not ts.get("dst_determined"):
        base.append("DST behaviour of server time")
    return base


_SECRET_KEY = __import__("re").compile(r"(password|passwd|secret|token|api[_-]?key)", __import__("re").I)


def _scrub(obj: Any) -> Any:
    """Redact only secret-bearing keys (the report legitimately contains a masked login and the account class)."""
    if isinstance(obj, dict):
        return {k: REDACTED if _SECRET_KEY.search(str(k)) else _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def _write(rep: Report, out_dir: Path, env: dict[str, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = json.dumps(_scrub(rep.data), indent=2, sort_keys=True, default=str)
    for secret in (env.get("MT5_PASSWORD"), env.get("MT5_LOGIN")):
        if secret and secret not in ("change-me", "00000000") and secret in payload:
            payload = payload.replace(secret, "***REDACTED***")
    path = out_dir / f"mt5_verify_{'fake_' if rep.data['mode'] != 'REAL_MT5' else ''}{stamp}.json"
    path.write_text(payload)
    print(f"\nVerdict: {rep.data.get('verdict')} | real_mt5_verified={rep.data['real_mt5_verified']}")
    print(f"JSON report: {path}")
    for u in rep.data["unverified"][:6]:
        print(f"  UNVERIFIED: {u}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Read-only MT5 verification harness (no orders are ever sent).")
    p.add_argument("--fake", action="store_true", help="run against the in-memory fake (validates our code, NOT MT5)")
    p.add_argument("--fake-trade-mode", type=int, default=0, help="fake account trade_mode (2=REAL to test the gate)")
    p.add_argument("--fake-offset-hours", type=int, default=3)
    p.add_argument("--env-file", default=".env")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--data-root", default="data")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--time-samples", type=int, default=30)
    p.add_argument("--time-interval", type=float, default=1.0)
    p.add_argument("--dst-weeks", type=int, default=0, help="weekly-open probes for DST inference (e.g. 52)")
    p.add_argument("--tick-hours", type=int, default=1)
    p.add_argument("--latency-samples", type=int, default=20)
    p.add_argument("--skip-dom", action="store_true")
    p.add_argument("--no-probe", action="store_true", help="skip request-limit and history-depth probes")
    p.add_argument("--map-dir", default=None, help="where broker_symbols/<broker>__<server>.yaml is written "
                   "(default: --config-dir; with --fake: <output-dir>/fake_symbol_maps)")
    p.add_argument("--no-write-map", action="store_true")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
