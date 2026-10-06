"""Historical tick acquisition: chunked, resumable, checksummed. READ-ONLY (uses the read-only adapter).

Chunk = one SOURCE-TIME day. Request windows are expressed in the SOURCE time domain (the domain the
terminal compares against); the TimeBase is only used to normalise results to UTC afterwards.

Window semantics: each chunk is half-open ``[start, end)`` in source milliseconds. Whether the
terminal treats ``date_to`` as inclusive is UNVERIFIED, so we request ``[start, end]`` and clip to the
half-open window; clipped rows are COUNTED in the manifest, so every tick belongs to exactly one chunk
and nothing is silently lost.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fxscalp.brokers.base import RAW_TICK_DTYPE, SymbolInfo, ReadOnlyBrokerAdapter
from fxscalp.brokers.errors import ConnectionLostError, NoTickDataError
from fxscalp.brokers.mt5.convert import check_symbol_consistency
from fxscalp.core.provenance import canonical_json, sha256_bytes, slug, software_versions
from fxscalp.market_data import quality as Qm
from fxscalp.market_data.schema import (DERIVED_SCHEMA_VERSION, QUALITY_SCHEMA_VERSION, RAW_SCHEMA_VERSION,
                                        derived_table, raw_table)
from fxscalp.market_data.store import ChunkKey, TickStore, utcnow
from fxscalp.market_data.timebase import TimeBase

log = logging.getLogger("fxscalp.acquire")
DAY_S = 86400


@dataclass(frozen=True)
class AcquisitionConfig:
    split_rows_threshold: int | None = None   # if a request returns >= this many rows, split the window
    min_split_window_s: int = 60
    min_call_interval_s: float = 0.0
    allow_unverified_time: bool = False
    max_connection_retries: int = 1
    quality: Qm.QualityConfig = field(default_factory=Qm.QualityConfig)


@dataclass
class ChunkResult:
    key: ChunkKey
    status: str                 # "acquired" | "skipped_verified" | "empty" | "failed"
    rows: int = 0
    requests: int = 0
    clipped_rows: int = 0
    seconds: float = 0.0
    detail: str = ""
    quarantined: int = 0


@dataclass
class AcquisitionSummary:
    chunks: list[ChunkResult]
    dataset_manifest: dict[str, Any] | None

    @property
    def failed(self) -> list[ChunkResult]:
        return [c for c in self.chunks if c.status == "failed"]


def day_window_s(day: date) -> tuple[int, int]:
    s = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
    return s, s + DAY_S


def persist_symbol_metadata(root: Path, broker: str, server: str, canonical: str, info: SymbolInfo,
                            account_currency: str | None) -> tuple[str, list[dict[str, str]]]:
    """Write a content-addressed metadata snapshot; returns (sha256, consistency issues)."""
    issues = [{"severity": i.severity, "code": i.code, "message": i.message}
              for i in check_symbol_consistency(info, account_currency)]
    body = {"schema_version": "symbol_info/1", "canonical": canonical, "broker": broker, "server": server,
            "account_currency": account_currency, "raw": info.raw,
            "typed": {k: getattr(info, k) for k in info.__dataclass_fields__ if k != "raw"},
            "consistency_issues": issues}
    sha = sha256_bytes(canonical_json(body).encode())
    d = root / "metadata" / "symbol_info" / slug(broker) / slug(server) / canonical
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{sha[:16]}.json"
    if not p.exists():
        p.write_text(json.dumps({**body, "collected_at_utc": utcnow().isoformat(), "sha256": sha}, indent=2,
                                sort_keys=True, default=str))
    return sha, issues


class TickAcquirer:
    def __init__(self, adapter: ReadOnlyBrokerAdapter, store: TickStore, timebase: TimeBase, *, broker: str,
                 server: str, canonical: str, broker_symbol: str, symbol_info: SymbolInfo,
                 account_fingerprint: str | None = None, account_currency: str | None = None,
                 config: AcquisitionConfig = AcquisitionConfig(),
                 reconnect: Callable[[], Any] | None = None, sleep: Callable[[float], None] = time.sleep):
        self.adapter, self.store, self.tb = adapter, store, timebase
        self.broker, self.server, self.canonical, self.sym = broker, server, canonical, broker_symbol
        self.info, self.cfg = symbol_info, config
        self.account_fp, self.account_ccy = account_fingerprint, account_currency
        self._reconnect = reconnect or getattr(adapter, "reconnect", None)
        self._sleep = sleep
        self._last_call = 0.0
        self._requests = 0
        self._latencies: list[float] = []
        self._timebase_sha = sha256_bytes(canonical_json(timebase.spec.identity()).encode())
        self._meta_sha, self._meta_issues = persist_symbol_metadata(store.root, broker, server, canonical,
                                                                    symbol_info, account_currency)

    # ---- fetching ----------------------------------------------------------------------------
    def _request(self, s: int, e: int) -> np.ndarray:
        wait = self.cfg.min_call_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        attempt = 0
        while True:
            t0 = time.perf_counter()
            try:
                arr = self.adapter.get_ticks_range(self.sym, s, e)
                self._latencies.append(time.perf_counter() - t0)
                self._requests += 1
                self._last_call = time.monotonic()
                return arr
            except NoTickDataError:
                self._latencies.append(time.perf_counter() - t0)
                self._requests += 1
                self._last_call = time.monotonic()
                return np.empty(0, dtype=RAW_TICK_DTYPE)
            except ConnectionLostError:
                attempt += 1
                if attempt > self.cfg.max_connection_retries or self._reconnect is None:
                    raise
                log.warning("connection lost during tick request; reconnecting (attempt %d)", attempt)
                self._reconnect()

    def _fetch_window(self, s: int, e: int) -> tuple[np.ndarray, int]:
        """Rows with source_time_msc in [s*1000, e*1000), in broker order. Returns (rows, clipped_count)."""
        arr = self._request(s, e)
        thr = self.cfg.split_rows_threshold
        if thr is not None and len(arr) >= thr and (e - s) > self.cfg.min_split_window_s:
            mid = s + (e - s) // 2
            a, ca = self._fetch_window(s, mid)
            b, cb = self._fetch_window(mid, e)
            return np.concatenate([a, b]), ca + cb
        lo, hi = s * 1000, e * 1000
        keep = (arr["time_msc"] >= lo) & (arr["time_msc"] < hi)
        return arr[keep], int((~keep).sum())

    # ---- one chunk -----------------------------------------------------------------------------
    def acquire_day(self, day: date, *, force: bool = False) -> ChunkResult:
        key = ChunkKey(self.broker, self.server, self.canonical, day)
        t0 = time.perf_counter()
        if not force:
            ok, _ = self.store.verify_chunk(key)
            if ok:
                m = self.store.chunk_manifest(key) or {}
                return ChunkResult(key, "skipped_verified", m.get("record_count", 0), detail="verified chunk exists")
        s, e = day_window_s(day)
        req0 = self._requests
        try:
            arr, clipped = self._fetch_window(s, e)
            ingestion = utcnow()
            norm = self.tb.normalize(arr["time_msc"], allow_unverified=self.cfg.allow_unverified_time)
            tbl = raw_table(arr, norm, ingestion)
            prev = self._previous_tick(day)
            q = Qm.assess(arr["time"], arr["time_msc"], arr["bid"], arr["ask"], arr["last"], arr["volume"],
                          arr["flags"], ambiguous=norm.ambiguous, nonexistent=norm.nonexistent, cfg=self.cfg.quality,
                          prev=prev)
            lat = self._latencies[-(self._requests - req0):] if self._requests > req0 else []
            manifest = self.store.write_raw_chunk(key, tbl, {
                "broker_symbol": self.sym, "requested_window_source_s": [s, e], "window_semantics": "[start,end) source ms",
                "requests": self._requests - req0, "clipped_rows_outside_window": clipped,
                "time_basis": norm.basis.value, "timebase_spec_sha256": self._timebase_sha,
                "timebase_rule": self.tb.spec.rule.name if self.tb.spec.rule else None,
                "dst_determined": self.tb.spec.dst_determined,
                "ambiguous_time_rows": int(norm.ambiguous.sum()), "nonexistent_time_rows": int(norm.nonexistent.sum()),
                "ingestion_time_utc": ingestion.isoformat(), "account_fingerprint": self.account_fp,
                "symbol_info_sha256": self._meta_sha, "request_latency_s": {
                    "n": len(lat), "max": max(lat) if lat else None, "mean": float(np.mean(lat)) if lat else None},
                "empty_reason": "no_ticks_returned" if len(arr) == 0 else None,
            })
            self.store.write_derived("tick_basic", key, derived_table(tbl, self.info.point), manifest,
                                     {"schema_version": DERIVED_SCHEMA_VERSION, "point": self.info.point,
                                      "transformation": "mid=(bid+ask)/2; spread=ask-bid; spread_points=spread/point; "
                                                        "relative_spread=spread/mid"})
            self.store.write_derived("tick_quality", key, q.to_table(tbl["seq"].to_numpy(zero_copy_only=False)),
                                     manifest, {"schema_version": QUALITY_SCHEMA_VERSION,
                                                "quality_summary": q.summary})
            self._log_event(key, "acquired" if len(arr) else "empty", len(arr), clipped, q)
            return ChunkResult(key, "acquired" if len(arr) else "empty", len(arr), self._requests - req0, clipped,
                               time.perf_counter() - t0, quarantined=int(q.quarantined.sum()))
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised by the range loop policy
            self._log_event(key, "failed", 0, 0, None, error=f"{type(exc).__name__}: {exc}")
            raise

    def _previous_tick(self, day: date) -> Qm.PrevTick | None:
        pk = ChunkKey(self.broker, self.server, self.canonical, day - timedelta(days=1))
        if self.store.chunk_manifest(pk) is None:
            return None
        try:
            t = self.store.read_raw_chunk(pk, ["source_time_msc", "bid", "ask", "last", "volume", "flags"])
        except Exception:  # noqa: BLE001
            return None
        if t.num_rows == 0:
            return None
        i = t.num_rows - 1
        return Qm.PrevTick(int(t["source_time_msc"][i].as_py()), float(t["bid"][i].as_py()), float(t["ask"][i].as_py()),
                           float(t["last"][i].as_py()), int(t["volume"][i].as_py()), int(t["flags"][i].as_py()))

    def _log_event(self, key: ChunkKey, status: str, rows: int, clipped: int, q: Qm.QualityResult | None,
                   error: str | None = None) -> None:
        rec = {"ts": utcnow().isoformat(), "instrument": key.instrument, "day": key.source_day.isoformat(),
               "status": status, "rows": rows, "clipped": clipped, "error": error,
               "quarantined": None if q is None else int(q.quarantined.sum())}
        p = self.store.root / "acquisition_log.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        log.info("tick chunk %s", status, extra={"ctx": rec})

    # ---- range / dataset -------------------------------------------------------------------------
    def acquire_range(self, start_day: date, end_day: date, *, force: bool = False,
                      stop_on_error: bool = True) -> AcquisitionSummary:
        results: list[ChunkResult] = []
        d = start_day
        while d <= end_day:
            try:
                results.append(self.acquire_day(d, force=force))
            except Exception as exc:  # noqa: BLE001
                results.append(ChunkResult(ChunkKey(self.broker, self.server, self.canonical, d), "failed",
                                           detail=f"{type(exc).__name__}: {exc}"))
                if stop_on_error:
                    break
            d += timedelta(days=1)
        manifest = None
        if results and not any(r.status == "failed" for r in results):
            manifest = self.write_dataset_manifest(start_day, end_day)
        return AcquisitionSummary(results, manifest)

    def write_dataset_manifest(self, start_day: date, end_day: date) -> dict[str, Any]:
        chunks = []
        total = 0
        d = start_day
        while d <= end_day:
            key = ChunkKey(self.broker, self.server, self.canonical, d)
            m = self.store.chunk_manifest(key)
            ok, why = self.store.verify_chunk(key)
            if m is None or not ok:
                raise RuntimeError(f"cannot build dataset manifest: chunk {d} not verified ({why})")
            chunks.append({"source_day": m["source_day"], "record_count": m["record_count"],
                           "raw_content_sha256": m["raw_content_sha256"], "file_sha256": m["file_sha256"]})
            total += m["record_count"]
            d += timedelta(days=1)
        spec = self.tb.spec
        body = {"manifest_type": "tick_raw_dataset", "schema_version": RAW_SCHEMA_VERSION, "broker": self.broker,
                "server": self.server, "instrument": self.canonical, "broker_symbol": self.sym,
                "start_source_day": start_day.isoformat(), "end_source_day": end_day.isoformat(),
                "record_count": total, "time_basis": spec.basis.value, "timebase_spec": spec.to_json(),
                "timebase_spec_sha256": self._timebase_sha, "symbol_info_sha256": self._meta_sha,
                "account_fingerprint": self.account_fp, "collected_at_utc": utcnow().isoformat(),
                "software": software_versions(), "chunks": chunks,
                "derived_datasets": ["tick_basic", "tick_quality"],
                "lineage": {"parents": [], "transformation": "broker copy_ticks_range acquisition"}}
        return self.store.write_dataset_manifest(body)


# --------------------------------------------------------------------------------------------
# Probes (what is the maximum history? how big can a request be?)
# --------------------------------------------------------------------------------------------
@dataclass
class DepthProbe:
    earliest_source_day: date | None
    samples: list[dict[str, Any]]
    calls: int
    note: str


def _wednesday_of(day: date) -> date:
    return day - timedelta(days=day.weekday()) + timedelta(days=2)


def probe_history_depth(adapter: ReadOnlyBrokerAdapter, broker_symbol: str, now_source_s: int, *,
                        max_days: int = 365 * 20, hour_source: int = 13, window_s: int = 3600) -> DepthProbe:
    """Gallop back in time, then bisect, using a 1-hour window on the Wednesday of each probed week.

    Assumes availability is contiguous from some start date (holes are possible: the result is the
    earliest probed week that has data, with ~1 week precision).
    """
    samples: list[dict[str, Any]] = []

    def has_data(days_back: int) -> bool:
        d = _wednesday_of((datetime.fromtimestamp(now_source_s, tz=timezone.utc) - timedelta(days=days_back)).date())
        s = int(datetime(d.year, d.month, d.day, hour_source, tzinfo=timezone.utc).timestamp())
        t0 = time.perf_counter()
        try:
            n = len(adapter.get_ticks_range(broker_symbol, s, s + window_s))
        except NoTickDataError:
            n = 0
        samples.append({"day": d.isoformat(), "rows": n, "latency_s": round(time.perf_counter() - t0, 4)})
        return n > 0

    if not has_data(7):
        return DepthProbe(None, samples, len(samples), "no ticks in the most recent probed week (market closed or no feed)")
    lo, hi = 7, None            # lo: known has-data, hi: known no-data
    step = 14
    while step <= max_days:
        if has_data(step):
            lo = step
            step *= 2
        else:
            hi = step
            break
    if hi is None:
        d = _wednesday_of((datetime.fromtimestamp(now_source_s, tz=timezone.utc) - timedelta(days=lo)).date())
        return DepthProbe(d, samples, len(samples), f"data present at the probe limit of {max_days} days")
    while hi - lo > 7:
        mid = (lo + hi) // 2
        if has_data(mid):
            lo = mid
        else:
            hi = mid
    d = _wednesday_of((datetime.fromtimestamp(now_source_s, tz=timezone.utc) - timedelta(days=lo)).date())
    return DepthProbe(d, samples, len(samples), "earliest probed week with data (~1 week precision, assumes no holes)")


@dataclass
class RequestLimitProbe:
    rows: list[dict[str, Any]]
    suspected_cap: int | None
    note: str


def probe_request_limits(adapter: ReadOnlyBrokerAdapter, broker_symbol: str, start_s: int,
                         windows_s: tuple[int, ...] = (3600, 6 * 3600, 86400, 3 * 86400)) -> RequestLimitProbe:
    """Request growing windows from the same start; a row count that stops growing hints at a per-call cap."""
    rows: list[dict[str, Any]] = []
    for w in windows_s:
        t0 = time.perf_counter()
        try:
            n = len(adapter.get_ticks_range(broker_symbol, start_s, start_s + w))
        except NoTickDataError:
            n = 0
        dt = time.perf_counter() - t0
        rows.append({"window_s": w, "rows": n, "latency_s": round(dt, 4),
                     "rows_per_s": round(n / dt, 1) if dt > 0 else None})
    cap = None
    for a, b in zip(rows, rows[1:]):
        if b["rows"] == a["rows"] and a["rows"] > 0 and b["window_s"] > a["window_s"]:
            cap = a["rows"]
    return RequestLimitProbe(rows, cap, "a plateau across growing windows suggests a per-request row cap"
                             if cap else "no plateau observed (cap not triggered at these sizes)")
