"""Deterministic synthetic XAU/USD tick generator. FOR TESTING ONLY.

It exists to exercise the pipeline (boundaries, gaps, duplicates, out-of-order ticks, session effects, regimes). It makes
NO attempt to be a realistic or profitable market and nothing measured on it is evidence about gold or any broker.
Same config + seed => byte-identical output.

Scenarios supported: trend, range (mean reverting), volatility burst, quiet, spread widening windows, tick outages
(missing ticks), duplicates, out-of-order ticks, weekend closure (New York Fri 17:00 -> Sun 17:00), London/New York open
activity bursts, London/New York overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxscalp.brokers.base import RAW_TICK_DTYPE
from fxscalp.sessions.calendar import DEFAULT_RULES, Session, classify_sessions, window_utc


@dataclass(frozen=True)
class Regime:
    kind: str                 # "trend_up" | "trend_down" | "range" | "vol_burst" | "quiet"
    duration_s: int
    sigma_bps: float = 0.4    # per-tick volatility (bps) before regime scaling
    drift_bps: float = 0.0


DEFAULT_SCHEDULE = (Regime("range", 1500), Regime("trend_up", 1200), Regime("range", 900), Regime("vol_burst", 300),
                    Regime("range", 1200), Regime("trend_down", 1200), Regime("quiet", 900))


@dataclass(frozen=True)
class SynthConfig:
    start_utc: datetime = datetime(2025, 10, 6, 6, 0, tzinfo=timezone.utc)       # a Monday
    duration_s: int = 6 * 3600
    seed: int = 7
    base_price: float = 2000.0
    point: float = 0.01
    digits: int = 2
    base_rate_hz: float = 1.5
    base_spread_points: float = 25.0
    schedule: tuple[Regime, ...] = DEFAULT_SCHEDULE
    session_activity: bool = True                      # London/NY opens, overlap, Asia/off-hours multipliers
    spread_widening: tuple[tuple[int, int, float], ...] = ()      # (offset_s, duration_s, multiplier)
    outages: tuple[tuple[int, int], ...] = ()          # (offset_s, duration_s): no ticks at all
    drop_prob: float = 0.0                             # random missing ticks
    dup_prob: float = 0.0                              # exact duplicate of the previous tick
    out_of_order_prob: float = 0.0                     # timestamp pulled 2-5 s into the past (broker order kept)
    weekend_closure: bool = True
    source_offset_s: int = 0                           # tick time domain = UTC + offset (server time simulation)


def _activity(utc_ms_sec: np.ndarray, cfg: SynthConfig) -> np.ndarray:
    """Per-second activity multiplier from the (DST-aware) session calendar."""
    lab = classify_sessions(utc_ms_sec, DEFAULT_RULES)
    mult = np.ones(len(lab))
    if cfg.session_activity:
        mult = np.select([lab == Session.ASIA.value, lab == Session.LONDON.value, lab == Session.NEW_YORK.value,
                          lab == Session.LONDON_NEW_YORK_OVERLAP.value, lab == Session.OFF_HOURS.value],
                         [0.6, 1.2, 1.2, 1.6, 0.4], default=1.0)
        # open bursts: first 30 minutes after the London and New York opens (local DST-aware opens)
        days = pd.to_datetime(utc_ms_sec[[0, -1]], unit="ms", utc=True).normalize()
        for d in pd.date_range(days[0] - pd.Timedelta(days=1), days[1] + pd.Timedelta(days=1), freq="D"):
            for w in (DEFAULT_RULES.london, DEFAULT_RULES.new_york):
                o, _ = window_utc(d.date(), w)
                o_ms = int(o.timestamp() * 1000)
                m = (utc_ms_sec >= o_ms) & (utc_ms_sec < o_ms + 30 * 60 * 1000)
                mult[m] *= 2.5
    if cfg.weekend_closure:
        mult[lab == Session.MARKET_CLOSED.value] = 0.0
    return mult


def generate_ticks(cfg: SynthConfig = SynthConfig()) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    t0 = int(cfg.start_utc.timestamp())
    secs = np.arange(cfg.duration_s, dtype="int64")
    sec_ms = (t0 + secs) * 1000
    act = _activity(sec_ms, cfg)
    # regime schedule per second (cycled)
    bounds, i, pos = [], 0, 0
    while pos < cfg.duration_s:
        r = cfg.schedule[i % len(cfg.schedule)]
        bounds.append((pos, min(cfg.duration_s, pos + r.duration_s), r))
        pos += r.duration_s
        i += 1
    rate_mult = np.ones(cfg.duration_s)
    for a, b, r in bounds:
        rate_mult[a:b] = {"vol_burst": 3.0, "quiet": 0.4}.get(r.kind, 1.0)
    rate = cfg.base_rate_hz * act * rate_mult
    for off, dur in cfg.outages:
        rate[max(0, off):max(0, off) + dur] = 0.0
    counts = rng.poisson(rate)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=RAW_TICK_DTYPE)
    sec_of_tick = np.repeat(secs, counts)
    ms_in_sec = rng.integers(0, 1000, total)
    # order ticks inside each second by millisecond (stable by (second, ms))
    order = np.lexsort((ms_in_sec, sec_of_tick))
    sec_of_tick, ms_in_sec = sec_of_tick[order], ms_in_sec[order]
    utc_ms = (t0 + sec_of_tick) * 1000 + ms_in_sec
    # ---- price path, regime by regime (continuous)
    level = np.empty(total)
    price = cfg.base_price
    for a, b, r in bounds:
        lo, hi = np.searchsorted(sec_of_tick, a, "left"), np.searchsorted(sec_of_tick, b, "left")
        n = hi - lo
        if n == 0:
            continue
        eps = rng.standard_normal(n)
        sig = r.sigma_bps * 1e-4 * price
        if r.kind in ("trend_up", "trend_down"):
            drift = (1 if r.kind == "trend_up" else -1) * (r.drift_bps or 0.15) * 1e-4 * price
            path = price + np.cumsum(drift + sig * eps)
        elif r.kind == "vol_burst":
            path = price + np.cumsum(sig * 3.0 * eps)
        elif r.kind == "quiet":
            path = price + np.cumsum(sig * 0.4 * eps)
        else:   # range: mean-reverting around the segment start
            phi = 0.995
            y = pd.Series(sig * 6.0 * eps).ewm(alpha=1 - phi, adjust=False).mean().to_numpy() / (1 - phi) * 0.1
            path = price + y
        level[lo:hi] = path
        price = float(path[-1])
    # ---- spread (points) with widening windows, quantised to the point
    t_rel = sec_of_tick
    mult = np.ones(total)
    for off, dur, m in cfg.spread_widening:
        mult[(t_rel >= off) & (t_rel < off + dur)] = m
    sp_pts = np.maximum(1, np.round(cfg.base_spread_points * mult * (1 + rng.exponential(0.12, total))))
    spread = sp_pts * cfg.point
    bid = np.round(level - spread / 2, cfg.digits)
    ask = np.round(bid + spread, cfg.digits)
    ask = np.maximum(ask, bid + cfg.point)
    # ---- random missing ticks
    keep = np.ones(total, bool)
    if cfg.drop_prob:
        keep = rng.random(total) >= cfg.drop_prob
    src_ms = utc_ms + cfg.source_offset_s * 1000
    arr = np.empty(total, dtype=RAW_TICK_DTYPE)
    arr["time_msc"], arr["time"] = src_ms, src_ms // 1000
    arr["bid"], arr["ask"], arr["last"] = bid, ask, 0.0
    arr["volume"], arr["volume_real"], arr["flags"] = 0, 0.0, 6
    arr = arr[keep]
    n = len(arr)
    if cfg.out_of_order_prob and n > 10:
        idx = np.flatnonzero(rng.random(n) < cfg.out_of_order_prob)
        idx = idx[idx > 0]
        arr["time_msc"][idx] = arr["time_msc"][idx] - rng.integers(2000, 5000, len(idx))
        arr["time"][idx] = arr["time_msc"][idx] // 1000
    if cfg.dup_prob and n > 10:
        dup = np.flatnonzero(rng.random(n) < cfg.dup_prob)
        arr = np.insert(arr, dup + 1, arr[dup])
    return arr


def synthetic_symbol_info(point: float = 0.01, digits: int = 2):
    """A SymbolInfo for the synthetic instrument (same field layout the real adapter produces)."""
    from fxscalp.brokers.mt5 import convert
    from fxscalp.brokers.mt5.fake import FakeMT5Module
    d = FakeMT5Module()._symbol("XAUUSDm")._asdict()
    d.update(point=point, digits=digits, trade_tick_size=point, name="XAUUSDm")
    return convert.to_symbol_info(d, "XAU_USD")


class _ArrayAdapter:
    """Duck-typed read-only adapter serving a prepared tick array through the REAL acquisition code path."""

    def __init__(self, ticks: np.ndarray):
        self.ticks = ticks

    def get_ticks_range(self, sym: str, s: int, e: int) -> np.ndarray:
        m = (self.ticks["time_msc"] >= s * 1000) & (self.ticks["time_msc"] <= e * 1000)
        return self.ticks[m]


def write_phase1_dataset(root: Path, ticks: np.ndarray, *, start_day: date, end_day: date, source_offset_s: int = 0,
                         broker: str = "SyntheticBroker", server: str = "Synthetic-Demo", point: float = 0.01,
                         digits: int = 2) -> dict[str, Any]:
    """Store synthetic ticks exactly like Phase 1 does (raw + derived + quality + dataset manifest). Returns the manifest."""
    from fxscalp.market_data.acquire import TickAcquirer
    from fxscalp.market_data.store import TickStore
    from fxscalp.market_data.timebase import (OffsetEstimate, ServerTimeRule, TimeBase, TimeBaseSpec, TimeBasis)
    est = OffsetEstimate("ok", source_offset_s, float(source_offset_s), 0.0, 0.0, 0, 0, True, 900,
                         ("SYNTHETIC: offset is known by construction, not measured",))
    spec = TimeBaseSpec(TimeBasis.UTC_VERIFIED if source_offset_s == 0 else TimeBasis.SERVER_FIXED_OFFSET,
                        ServerTimeRule.fixed(source_offset_s), False, datetime.now(timezone.utc).isoformat(), broker, server,
                        est, None, ("synthetic dataset",))
    acq = TickAcquirer(_ArrayAdapter(ticks), TickStore(root), TimeBase(spec), broker=broker, server=server,
                       canonical="XAU_USD", broker_symbol="XAUUSDm", symbol_info=synthetic_symbol_info(point, digits),
                       account_fingerprint="synthetic", account_currency="USD")
    summ = acq.acquire_range(start_day, end_day)
    if summ.dataset_manifest is None:
        raise RuntimeError(f"synthetic ingest failed: {[c.detail for c in summ.failed]}")
    return summ.dataset_manifest


__all__ = ["Regime", "SynthConfig", "generate_ticks", "write_phase1_dataset", "synthetic_symbol_info"]
