"""Shared Phase 1 workflows used by the scripts (connect, discover, calibrate). Read-only."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from fxscalp.brokers.base import ReadOnlyBrokerAdapter
from fxscalp.brokers.mt5.adapter import MT5BrokerAdapter, Mt5Credentials
from fxscalp.brokers.symbols import (DiscoveryResult, InstrumentSpec, SymbolMap, build_map_document, discover, map_path,
                                     write_map)
from fxscalp.core.config import load_yaml
from fxscalp.core.provenance import slug
from fxscalp.market_data.calibrate import collect_offset_samples
from fxscalp.brokers.errors import HistoryUnavailableError, RequestTimeoutError, TerminalRequestError
from fxscalp.market_data.retry import RetryPolicy, call_with_retry
from fxscalp.market_data.timebase import (DstInference, OffsetConsistency, TimeBase, TimeBaseSpec,
                                          assess_offset_consistency, build_calibration_record, estimate_offset,
                                          infer_dst_from_opens, resolve_time_basis)

log = logging.getLogger("fxscalp.workflows")


def make_adapter(env: Mapping[str, str], *, fake: bool = False, fake_scenario: Any = None) -> MT5BrokerAdapter:
    if fake:
        from fxscalp.brokers.mt5.fake import FakeMT5Adapter
        return FakeMT5Adapter(fake_scenario)
    return MT5BrokerAdapter(Mt5Credentials.from_env(env))


def load_instrument_specs(config_dir: Path) -> dict[str, InstrumentSpec]:
    specs: dict[str, InstrumentSpec] = {}
    for p in sorted((config_dir / "instruments").glob("*.yaml")):
        cfg = load_yaml(p)
        s = InstrumentSpec.from_config(cfg)
        specs[s.canonical] = s
    return specs


def run_discovery(adapter: ReadOnlyBrokerAdapter, specs: Mapping[str, InstrumentSpec], config_dir: Path, broker: str,
                  server: str, *, write: bool = True) -> tuple[list[DiscoveryResult], Path]:
    """Discover every configured instrument; persist the deterministic mapping (pins from an existing map are kept)."""
    symbols = adapter.discover_symbols()
    path = map_path(config_dir, broker, server)
    pins = SymbolMap.load(path).pins if path.exists() else {}
    results = [discover(spec, symbols, pin=pins.get(spec.canonical)) for spec in specs.values()]
    if write:
        write_map(path, build_map_document(broker, server, results, pins))
    return results, path


def timebase_file(data_root: Path, broker: str, server: str) -> Path:
    return data_root / "metadata" / "timebase" / slug(broker) / f"{slug(server)}.json"


@dataclass
class WeeklyOpenProbe:
    opens: np.ndarray
    skipped: list[dict[str, Any]] = field(default_factory=list)   # weeks without an open and why (never silent)
    retry_events: list[dict[str, Any]] = field(default_factory=list)


def probe_weekly_opens_detailed(adapter: ReadOnlyBrokerAdapter, broker_symbol: str, now_source_s: int, weeks: int, *,
                                policy: RetryPolicy = RetryPolicy(), sleep: Callable[[float], None] = time.sleep,
                                max_consecutive_errors: int = 3, max_open_delay_s: int = 72 * 3600) -> WeeklyOpenProbe:
    """First tick after Saturday 12:00 (source time) of each of the last ``weeks`` weeks (one call each).

    Saturday midday is inside the weekly closure for any plausible server offset, so the first tick after it is the
    weekly open. Weeks are walked from the most recent backwards (progressively warming cold history). A timeout or
    terminal error is retried with backoff and, if it persists, RECORDED as a skipped week; after
    ``max_consecutive_errors`` such weeks in a row the walk stops (the terminal is not hammered).

    ``copy_ticks_from`` returns the first tick AFTER the requested instant even when that is weeks later (e.g. for a
    Saturday before the history begins it returns the first tick of the whole history). Such a tick is not that
    week's open: anything later than ``max_open_delay_s`` after the Saturday probe instant is rejected and recorded.
    """
    probe = WeeklyOpenProbe(np.array([], dtype="int64"))
    opens: list[int] = []
    now = datetime.fromtimestamp(now_source_s, tz=timezone.utc)
    last_sat = (now - timedelta(days=(now.weekday() - 5) % 7)).date()
    consecutive = 0
    for k in range(1, weeks + 1):
        sat = last_sat - timedelta(days=7 * k)
        s = int(datetime(sat.year, sat.month, sat.day, 12, tzinfo=timezone.utc).timestamp())
        try:
            arr = call_with_retry(lambda: adapter.get_ticks_from(broker_symbol, s, 1), policy, label=f"weekly_open[{sat}]",
                                  sleep=sleep, on_event=probe.retry_events.append)
            consecutive = 0
        except HistoryUnavailableError:
            probe.skipped.append({"week_saturday": sat.isoformat(), "reason": "history_unavailable"})
            consecutive = 0
            continue
        except (RequestTimeoutError, TerminalRequestError) as exc:
            probe.skipped.append({"week_saturday": sat.isoformat(), "reason": f"{type(exc).__name__} after retries"})
            consecutive += 1
            if consecutive >= max_consecutive_errors:
                probe.skipped.append({"reason": f"stopped after {consecutive} consecutive request errors"})
                break
            continue
        if len(arr):
            t_open = int(arr["time"][0])
            if t_open - s > max_open_delay_s:
                probe.skipped.append({"week_saturday": sat.isoformat(), "reason": "first tick is "
                                      f"{(t_open - s) / 3600:.0f} h after Saturday 12:00: that week has no data "
                                      "(history starts later); not a weekly open"})
                continue
            opens.append(t_open)
        else:
            probe.skipped.append({"week_saturday": sat.isoformat(), "reason": "no ticks after Saturday 12:00"})
    probe.opens = np.array(sorted(opens), dtype="int64")
    return probe


def probe_weekly_opens(adapter: ReadOnlyBrokerAdapter, broker_symbol: str, now_source_s: int, weeks: int) -> np.ndarray:
    """Backwards-compatible wrapper returning only the open instants (see ``probe_weekly_opens_detailed``)."""
    return probe_weekly_opens_detailed(adapter, broker_symbol, now_source_s, weeks).opens


def calibrate_timebase(adapter: ReadOnlyBrokerAdapter, broker_symbol: str, *, broker: str, server: str,
                       samples: int = 30, interval_s: float = 1.0, dst_weeks: int = 0,
                       sleep: Any = time.sleep, bound_validity: bool = True,
                       policy: RetryPolicy = RetryPolicy()) -> tuple[TimeBaseSpec, dict[str, Any]]:
    """Measure the server-time offset, check that ONE offset is consistent across the observed weekly opens, and
    resolve a TimeBaseSpec whose validity period is bounded by that evidence (never '+N h forever')."""
    smp = collect_offset_samples(adapter, broker_symbol, samples, interval_s, sleep=sleep)
    est = estimate_offset(smp)
    dst: DstInference | None = None
    cons: OffsetConsistency | None = None
    wprobe: WeeklyOpenProbe | None = None
    if est.status == "ok" and dst_weeks > 0 and est.offset_s is not None:
        now_source = int(time.time()) + est.offset_s
        wprobe = probe_weekly_opens_detailed(adapter, broker_symbol, now_source, dst_weeks, policy=policy, sleep=sleep)
        dst = infer_dst_from_opens(wprobe.opens, est.offset_s)
        cons = assess_offset_consistency(wprobe.opens, est.offset_s)
    spec = resolve_time_basis(est, dst, broker=broker, server=server, consistency=cons, bound_validity=bound_validity)
    evidence = {"n_samples": len(smp), "interval_s": interval_s,
                "weekly_opens_found": 0 if wprobe is None else int(len(wprobe.opens)),
                "weekly_open_weeks_requested": dst_weeks,
                "weekly_open_skipped": [] if wprobe is None else wprobe.skipped,
                "weekly_open_retries": 0 if wprobe is None else len(wprobe.retry_events),
                "offset_consistency": None if cons is None else cons.__dict__,
                "first_samples": [{"local_utc_s": round(s.local_utc_s, 3), "tick_time_msc": s.tick_time_msc,
                                   "latency_ms": round(s.call_latency_s * 1000, 2)} for s in smp[:5]]}
    return spec, evidence


def calibration_records_dir(data_root: Path, broker: str, server: str) -> Path:
    return data_root / "metadata" / "timebase" / slug(broker) / slug(server) / "records"


def save_calibration_record(data_root: Path, spec: TimeBaseSpec) -> tuple[Path, dict[str, Any]]:
    """Persist the immutable, versioned calibration record (a different measurement gets a different record id)."""
    rec = build_calibration_record(spec)
    d = calibration_records_dir(data_root, spec.broker, spec.server)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{rec['record_id']}.json"
    if not path.exists():
        path.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path, rec


def load_or_none(data_root: Path, broker: str, server: str) -> TimeBase | None:
    p = timebase_file(data_root, broker, server)
    return TimeBase.load(p) if p.exists() else None
