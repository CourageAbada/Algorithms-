"""Shared Phase 1 workflows used by the scripts (connect, discover, calibrate). Read-only."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fxscalp.brokers.base import ReadOnlyBrokerAdapter
from fxscalp.brokers.mt5.adapter import MT5BrokerAdapter, Mt5Credentials
from fxscalp.brokers.symbols import (DiscoveryResult, InstrumentSpec, SymbolMap, build_map_document, discover, map_path,
                                     write_map)
from fxscalp.core.config import load_yaml
from fxscalp.core.provenance import slug
from fxscalp.market_data.calibrate import collect_offset_samples
from fxscalp.market_data.timebase import (DstInference, TimeBase, TimeBaseSpec, estimate_offset, infer_dst_from_opens,
                                          resolve_time_basis)

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


def probe_weekly_opens(adapter: ReadOnlyBrokerAdapter, broker_symbol: str, now_source_s: int, weeks: int) -> np.ndarray:
    """First tick after Saturday 12:00 (source time) of each of the last ``weeks`` weeks (one cheap call each).

    Saturday midday is inside the weekly closure for any plausible server offset, so the first tick after it is the
    weekly open. Weeks with no data are skipped.
    """
    opens: list[int] = []
    now = datetime.fromtimestamp(now_source_s, tz=timezone.utc)
    last_sat = (now - timedelta(days=(now.weekday() - 5) % 7)).date()
    for k in range(1, weeks + 1):
        sat = last_sat - timedelta(days=7 * k)
        s = int(datetime(sat.year, sat.month, sat.day, 12, tzinfo=timezone.utc).timestamp())
        try:
            arr = adapter.get_ticks_from(broker_symbol, s, 1)
        except Exception:  # noqa: BLE001 - missing weeks are expected at the history edge
            continue
        if len(arr):
            opens.append(int(arr["time"][0]))
    return np.array(sorted(opens), dtype="int64")


def calibrate_timebase(adapter: ReadOnlyBrokerAdapter, broker_symbol: str, *, broker: str, server: str,
                       samples: int = 30, interval_s: float = 1.0, dst_weeks: int = 0,
                       sleep: Any = time.sleep) -> tuple[TimeBaseSpec, dict[str, Any]]:
    """Measure the server-time offset (and optionally infer DST behaviour) and resolve a TimeBaseSpec."""
    smp = collect_offset_samples(adapter, broker_symbol, samples, interval_s, sleep=sleep)
    est = estimate_offset(smp)
    dst: DstInference | None = None
    opens = np.array([], dtype="int64")
    if est.status == "ok" and dst_weeks > 0 and est.offset_s is not None:
        now_source = int(time.time()) + est.offset_s
        opens = probe_weekly_opens(adapter, broker_symbol, now_source, dst_weeks)
        dst = infer_dst_from_opens(opens, est.offset_s)
    spec = resolve_time_basis(est, dst, broker=broker, server=server)
    evidence = {"n_samples": len(smp), "interval_s": interval_s, "weekly_opens_found": int(len(opens)),
                "first_samples": [{"local_utc_s": round(s.local_utc_s, 3), "tick_time_msc": s.tick_time_msc,
                                   "latency_ms": round(s.call_latency_s * 1000, 2)} for s in smp[:5]]}
    return spec, evidence


def load_or_none(data_root: Path, broker: str, server: str) -> TimeBase | None:
    p = timebase_file(data_root, broker, server)
    return TimeBase.load(p) if p.exists() else None
