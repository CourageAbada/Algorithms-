"""Controlled, monitored, resumable BACKWARDS acquisition of the available tick history (READ-ONLY).

  python -m scripts.acquire_history --start 2026-03-31 [--end YYYY-MM-DD] --expect-calibration timecal-<id>

* newest complete source day first, then backwards, one immutable verified chunk per source day;
* a day is only "done" once its chunk verifies (checksum + row count) and the derived artifacts re-read with the same row count;
* verified chunks are never re-downloaded (re-running resumes);
* every chunk appends a monitor record (duration, requests, retries, timeouts, DEGRADED/LOST transitions, health checks, RSS,
  disk) to ``<data-root>/acquisition_monitor.jsonl``;
* STOPS CLEANLY (keeping every verified chunk) on: a failed chunk after the bounded retries, adapter LOST, repeated
  unavailable-history responses, low disk, or memory that is high / growing linearly-per-chunk beyond a limit.
The current (incomplete) source day is never requested. No order function exists on this code path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from fxscalp.brokers.errors import BrokerError
from fxscalp.brokers.symbols import SymbolMap, map_path
from fxscalp.core.env import load_dotenv
from fxscalp.market_data.acquire import AcquisitionConfig, TickAcquirer
from fxscalp.market_data.retry import RetryPolicy
from fxscalp.market_data.store import ChunkKey, TickStore
from fxscalp.market_data.timebase import TimeBase, TimeBasis, build_calibration_record
from fxscalp.monitoring.logging_setup import configure_logging
from fxscalp.workflows import make_adapter, timebase_file


def dir_bytes(root: Path) -> int:
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) if root.exists() else 0


class RssSampler:
    """Peak RSS of this process while a chunk is processed."""

    def __init__(self) -> None:
        self.proc = psutil.Process()
        self.peak = 0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self) -> "RssSampler":
        self.peak = self.proc.memory_info().rss
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            self.peak = max(self.peak, self.proc.memory_info().rss)

    def __exit__(self, *a: Any) -> None:
        self._stop.set()
        if self._t:
            self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Monitored backwards historical tick acquisition (read-only).")
    p.add_argument("--instrument", default="XAU_USD")
    p.add_argument("--start", required=True, help="oldest SOURCE-time day to attempt, YYYY-MM-DD")
    p.add_argument("--end", default=None, help="newest day (default: latest COMPLETE source day)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--expect-calibration", default=None, help="abort unless the stored time basis has this record id")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--slow-chunk-s", type=float, default=30.0, help="a chunk slower than this triggers a cool-down pause")
    p.add_argument("--pause-after-slow-s", type=float, default=15.0)
    p.add_argument("--min-call-interval", type=float, default=0.5)
    p.add_argument("--max-rss-mb", type=float, default=1500.0)
    p.add_argument("--rss-slope-mb-per-chunk", type=float, default=15.0)
    p.add_argument("--min-free-gb", type=float, default=5.0)
    p.add_argument("--max-consecutive-unavailable", type=int, default=2)
    p.add_argument("--max-chunks", type=int, default=None, help="stop after N new (non-skipped) chunks")
    a = p.parse_args(argv)
    configure_logging("INFO")
    env = dict(os.environ)
    load_dotenv(Path(a.env_file), env)
    root, cfg_dir = Path(a.data_root), Path(a.config_dir)
    store = TickStore(root)
    monitor = root / "acquisition_monitor.jsonl"
    monitor.parent.mkdir(parents=True, exist_ok=True)

    def emit(rec: dict[str, Any]) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), **rec}
        with open(monitor, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    adapter = make_adapter(env)
    transitions: list[dict[str, Any]] = []
    orig_set = adapter._set_state            # type: ignore[attr-defined]

    def spy(new: Any, why: str = "") -> None:
        transitions.append({"to": new.value, "why": why, "ts": datetime.now(timezone.utc).isoformat()})
        orig_set(new, why)

    adapter._set_state = spy                 # type: ignore[attr-defined,method-assign]
    rc = 0
    try:
        ident = adapter.connect()
        broker, server = ident.company or "unknown", ident.server or "unknown"
        sym = SymbolMap.load(map_path(cfg_dir, broker, server)).broker_symbol(a.instrument)
        tbp = timebase_file(root, broker, server)
        if not tbp.exists():
            print("ERROR: no stored time calibration; run scripts.verify_mt5 first", file=sys.stderr)
            return 2
        tb = TimeBase.load(tbp)
        rec = build_calibration_record(tb.spec)
        if tb.basis == TimeBasis.UNVERIFIED or (a.expect_calibration and rec["record_id"] != a.expect_calibration):
            print(f"ERROR: stored calibration {rec['record_id']} ({tb.basis.value}) != expected {a.expect_calibration}",
                  file=sys.stderr)
            return 2
        off = tb.spec.estimate.offset_s if tb.spec.estimate and tb.spec.estimate.offset_s is not None else 0
        today_source = datetime.fromtimestamp(time.time() + off, tz=timezone.utc).date()
        last_complete = today_source - timedelta(days=1)          # today's source day is still in progress
        end = date.fromisoformat(a.end) if a.end else last_complete
        if end > last_complete:
            print(f"ERROR: --end {end} is not a complete source day (latest complete: {last_complete})", file=sys.stderr)
            return 2
        start = date.fromisoformat(a.start)
        info = adapter.get_symbol_info(sym)
        acq = TickAcquirer(adapter, store, tb, broker=broker, server=server, canonical=a.instrument, broker_symbol=sym,
                           symbol_info=info, account_fingerprint=ident.login_fingerprint, account_currency=ident.currency,
                           config=AcquisitionConfig(newest_first=True, min_call_interval_s=a.min_call_interval,
                                                    retry=RetryPolicy(max_retries=a.max_retries)))
        emit({"event": "start", "start": start.isoformat(), "end": end.isoformat(), "calibration": rec["record_id"],
              "time_basis_id": tb.spec.basis_id(), "history_timeout_s": adapter._history_timeout_s,   # type: ignore[attr-defined]
              "baseline_rss_mb": round(psutil.Process().memory_info().rss / 2**20, 1)})
        print(f"acquiring {a.instrument} source days {end} -> {start} (newest first); calibration {rec['record_id']}")
        n_total = (end - start).days + 1
        rss_hist: list[float] = []
        new_chunks = unavailable_run = 0
        n_ev = n_tr = 0
        stop_reason = None
        d = end
        done = 0
        while d >= start:
            key = ChunkKey(broker, server, a.instrument, d)
            free_gb = shutil.disk_usage(root).free / 2**30
            if free_gb < a.min_free_gb:
                stop_reason = f"low disk space ({free_gb:.1f} GB free)"
                break
            t0 = time.perf_counter()
            try:
                with RssSampler() as rs:
                    r = acq.acquire_day(d)
                    status_extra: dict[str, Any] = {}
                    if r.status in ("acquired", "empty", "skipped_verified"):
                        ok, why = store.verify_chunk(key)              # re-read + checksum the written artifact
                        man = store.chunk_manifest(key) or {}
                        n_basic = store.read_derived("tick_basic", key).num_rows
                        n_qual = store.read_derived("tick_quality", key).num_rows
                        if not ok or n_basic != man.get("record_count") or n_qual != man.get("record_count"):
                            raise RuntimeError(f"post-write verification FAILED for {d}: {why}; derived rows "
                                               f"{n_basic}/{n_qual} vs {man.get('record_count')}")
                        status_extra = {"verified": True, "retries": man.get("retries", 0),
                                        "retry_reasons": man.get("retry_reasons", []),
                                        "uncertain_rows": man.get("time_basis_uncertain_rows"),
                                        "file_bytes": man.get("file_sha256") and man.get("file_bytes")}
            except BrokerError as exc:
                emit({"event": "chunk_failed", "day": d.isoformat(), "error": f"{type(exc).__name__}: {exc}",
                      "seconds": round(time.perf_counter() - t0, 2), "adapter_state": adapter.state.value})
                stop_reason = f"chunk {d} failed after bounded retries: {type(exc).__name__}: {exc}"
                rc = 1
                break
            except Exception as exc:  # noqa: BLE001
                emit({"event": "chunk_failed", "day": d.isoformat(), "error": f"{type(exc).__name__}: {exc}"})
                stop_reason = f"chunk {d} error: {type(exc).__name__}: {exc}"
                rc = 1
                break
            secs = time.perf_counter() - t0
            to_events = adapter.timeout_events[n_ev:]
            n_ev = len(adapter.timeout_events)
            trans = transitions[n_tr:]
            n_tr = len(transitions)
            rss_end = psutil.Process().memory_info().rss / 2**20
            rec_c = {"event": "chunk", "day": d.isoformat(), "status": r.status, "rows": r.rows, "requests": r.requests,
                     "seconds": round(secs, 2), "quarantined": r.quarantined, "detail": r.detail,
                     "peak_rss_mb": round(rs.peak / 2**20, 1), "rss_end_mb": round(rss_end, 1),
                     "data_bytes": dir_bytes(root / "raw") + dir_bytes(root / "derived") if r.status != "skipped_verified" else None,
                     "free_gb": round(free_gb, 1), "adapter_state": adapter.state.value,
                     "timeout_events": to_events, "state_transitions": trans, **status_extra}
            emit(rec_c)
            done += 1
            print(f"{d} {r.status:16} rows={r.rows:>9,} {secs:6.1f}s retries={status_extra.get('retries', '-')} "
                  f"rss={rss_end:6.0f}MB state={adapter.state.value} ({done}/{n_total})", flush=True)
            if r.status == "unavailable":
                unavailable_run += 1
                if unavailable_run >= a.max_consecutive_unavailable:
                    stop_reason = f"{unavailable_run} consecutive unavailable-history responses (history boundary?)"
                    break
            else:
                unavailable_run = 0
            if r.status != "skipped_verified":
                new_chunks += 1
                rss_hist.append(rss_end)
                if rss_end > a.max_rss_mb:
                    stop_reason = f"RSS {rss_end:.0f} MB above limit {a.max_rss_mb:.0f} MB"
                    break
                if len(rss_hist) >= 20:
                    slope = float(np.polyfit(np.arange(20), np.array(rss_hist[-20:]), 1)[0])
                    if slope > a.rss_slope_mb_per_chunk and rss_end > 600:
                        stop_reason = f"RSS growing {slope:.1f} MB/chunk over the last 20 chunks (non-linear/leak?)"
                        break
                if a.max_chunks and new_chunks >= a.max_chunks:
                    stop_reason = f"--max-chunks {a.max_chunks} reached"
                    break
            if secs > a.slow_chunk_s:
                emit({"event": "cooldown", "after_day": d.isoformat(), "seconds": a.pause_after_slow_s})
                time.sleep(a.pause_after_slow_s)
            d -= timedelta(days=1)
        if stop_reason:
            emit({"event": "stopped", "reason": stop_reason, "last_day": d.isoformat()})
            print(f"STOPPED CLEANLY: {stop_reason}. Verified chunks are preserved; re-run to resume.")
            rc = rc or 3
        else:
            first = start
            while first <= end and store.chunk_manifest(ChunkKey(broker, server, a.instrument, first)) is None:
                first += timedelta(days=1)       # boundary days that were unavailable are not part of the dataset
            man = acq.write_dataset_manifest(first, end)
            emit({"event": "finished", "dataset_id": man["dataset_id"], "first_day": first.isoformat(),
                  "last_day": end.isoformat(), "records": man["record_count"]})
            print(f"DONE dataset_id={man['dataset_id']} days={first}..{end} records={man['record_count']:,}")
    except BrokerError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        rc = 2
    finally:
        adapter.disconnect()
    return rc


if __name__ == "__main__":
    sys.exit(main())
