"""Acquire historical ticks into the immutable Parquet store (READ-ONLY).

  python -m scripts.acquire_ticks --instrument XAU_USD --start 2025-09-01 --end 2025-09-30

Chunked per source-time day, resumable (verified chunks are skipped unless --force), checksummed, with a dataset
manifest at the end. Requires a calibrated time basis (run scripts.verify_mt5 first, or pass --calibrate).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from fxscalp.brokers.errors import BrokerError
from fxscalp.brokers.symbols import SymbolMap
from fxscalp.core.env import load_dotenv
from fxscalp.market_data.acquire import AcquisitionConfig, TickAcquirer
from fxscalp.market_data.store import TickStore
from fxscalp.market_data.timebase import TimeBase, TimeBasis
from fxscalp.monitoring.logging_setup import configure_logging
from fxscalp.workflows import calibrate_timebase, load_instrument_specs, make_adapter, run_discovery, timebase_file


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Read-only historical tick acquisition (no orders are ever sent).")
    p.add_argument("--instrument", default="XAU_USD")
    p.add_argument("--start", required=True, help="first SOURCE-time day, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="last SOURCE-time day, YYYY-MM-DD (inclusive)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--force", action="store_true", help="re-download chunks even if verified")
    p.add_argument("--calibrate", action="store_true", help="(re)run time calibration now")
    p.add_argument("--time-samples", type=int, default=30)
    p.add_argument("--dst-weeks", type=int, default=0)
    p.add_argument("--allow-unverified-time", action="store_true",
                   help="store data with time_basis=unverified (UTC==source assumed, flagged in every manifest)")
    p.add_argument("--split-rows", type=int, default=None, help="split a request window if it returns >= this many rows")
    p.add_argument("--min-call-interval", type=float, default=0.0)
    p.add_argument("--fake", action="store_true")
    p.add_argument("--fake-ticks-per-day", type=int, default=20000)
    a = p.parse_args(argv)
    configure_logging("INFO")
    env = dict(os.environ)
    if not a.fake:
        load_dotenv(Path(a.env_file), env)
    scenario = None
    if a.fake:
        from fxscalp.brokers.mt5.fake import FakeScenario
        scenario = FakeScenario(ticks_per_day=a.fake_ticks_per_day)
    root, cfg = Path(a.data_root), Path(a.config_dir)
    adapter = make_adapter(env, fake=a.fake, fake_scenario=scenario)
    try:
        ident = adapter.connect()
        broker, server = ident.company or "unknown", ident.server or "unknown"
        specs = load_instrument_specs(cfg)
        map_dir = (root / "fake_symbol_maps") if a.fake else cfg
        _, mpath = run_discovery(adapter, specs, map_dir, broker, server)
        sym = SymbolMap.load(mpath).broker_symbol(a.instrument)
        tbp = timebase_file(root, broker, server)
        if a.calibrate or not tbp.exists():
            spec, _ = calibrate_timebase(adapter, sym, broker=broker, server=server, samples=a.time_samples,
                                         dst_weeks=a.dst_weeks)
            tb = TimeBase(spec)
            tb.save(tbp)
        else:
            tb = TimeBase.load(tbp)
        print(f"time basis: {tb.basis.value} rule={tb.spec.rule.name if tb.spec.rule else None} "
              f"dst_determined={tb.spec.dst_determined}")
        if tb.basis == TimeBasis.UNVERIFIED and not a.allow_unverified_time:
            print("ERROR: time basis is UNVERIFIED (calibration inconclusive). Re-run while the market is open with "
                  "--calibrate, or pass --allow-unverified-time to store flagged data.", file=sys.stderr)
            return 2
        info = adapter.get_symbol_info(sym)
        acq = TickAcquirer(adapter, TickStore(root), tb, broker=broker, server=server, canonical=a.instrument,
                           broker_symbol=sym, symbol_info=info, account_fingerprint=ident.login_fingerprint,
                           account_currency=ident.currency,
                           config=AcquisitionConfig(split_rows_threshold=a.split_rows,
                                                    min_call_interval_s=a.min_call_interval,
                                                    allow_unverified_time=a.allow_unverified_time))
        s = acq.acquire_range(date.fromisoformat(a.start), date.fromisoformat(a.end), force=a.force)
        for c in s.chunks:
            print(f"{c.key.source_day} {c.status:17} rows={c.rows:>9,} requests={c.requests} clipped={c.clipped_rows} "
                  f"quarantined={c.quarantined} {c.detail}")
        if s.dataset_manifest:
            print(f"dataset_id={s.dataset_manifest['dataset_id']} records={s.dataset_manifest['record_count']:,}")
        return 1 if s.failed else 0
    except BrokerError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    sys.exit(main())
