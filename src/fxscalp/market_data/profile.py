"""Descriptive data profile of a stored tick dataset. DATA ANALYSIS ONLY: no signals, no strategy, no
performance statements. Quarantined ticks are excluded from spread/volatility statistics but are always
counted and reported (nothing is dropped silently).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from fxscalp.market_data.store import ChunkKey, TickStore
from fxscalp.sessions.calendar import classify_sessions

PCTS = (1, 5, 25, 50, 75, 90, 95, 99, 99.9)
_RES_CAP = 400_000


class _Reservoir:
    """Bounded random sample for percentiles (seeded, deterministic)."""

    def __init__(self, cap: int = _RES_CAP, seed: int = 11):
        self.cap, self.rng, self.parts, self.n, self.total = cap, np.random.default_rng(seed), [], 0, 0

    def add(self, a: np.ndarray) -> None:
        a = a[np.isfinite(a)]
        self.total += len(a)
        if len(a) == 0:
            return
        self.parts.append(a)
        self.n += len(a)
        if self.n > 2 * self.cap:
            allv = np.concatenate(self.parts)
            self.parts, self.n = [self.rng.choice(allv, self.cap, replace=False)], self.cap

    def stats(self) -> dict[str, Any]:
        if not self.parts:
            return {"n": 0}
        v = np.concatenate(self.parts)
        out = {"n": int(self.total), "sample_n": int(len(v)), "mean": float(v.mean())}
        out.update({f"p{p}": float(np.percentile(v, p)) for p in PCTS})
        return out


def build_profile(store: TickStore, broker: str, server: str, instrument: str, start: date, end: date,
                  point: float | None = None) -> dict[str, Any]:
    chunks_total = n_ticks = n_quar = 0
    days_empty: list[str] = []
    days_missing: list[str] = []
    per_day: dict[str, int] = {}
    hour_ticks = np.zeros(24, dtype="int64")
    hour_days: dict[int, set[str]] = {h: set() for h in range(24)}
    spread_all, spread_rel = _Reservoir(), _Reservoir()
    spread_hour = {h: _Reservoir(60_000, h) for h in range(24)}
    spread_sess: dict[str, _Reservoir] = {}
    interval_ms = _Reservoir()
    ticks_sess: dict[str, int] = {}
    vol_hour: dict[int, list[np.ndarray]] = {h: [] for h in range(24)}
    vol_sess: dict[str, list[np.ndarray]] = {}
    flag_counts: dict[str, int] = {}
    long_gaps: list[dict[str, Any]] = []
    zero_spread = 0
    widest: list[dict[str, Any]] = []
    weekend: list[dict[str, Any]] = []
    prev_last_utc: int | None = None
    rollover: dict[int, _Reservoir] = {}
    first_utc = last_utc = None
    time_basis: set[str] = set()

    d = start
    while d <= end:
        key = ChunkKey(broker, server, instrument, d)
        man = store.chunk_manifest(key)
        if man is None:
            days_missing.append(d.isoformat())
            d += timedelta(days=1)
            continue
        chunks_total += 1
        time_basis.add(man.get("time_basis", "?"))
        n = man["record_count"]
        per_day[d.isoformat()] = n
        if n == 0:
            days_empty.append(d.isoformat())
            d += timedelta(days=1)
            continue
        raw = store.read_raw_chunk(key, ["seq", "source_time_msc", "normalized_utc_time"]).to_pandas()
        der = store.read_derived("tick_basic", key).to_pandas()
        qua = store.read_derived("tick_quality", key).to_pandas()
        qsum = json.loads((store.derived_dir("tick_quality", key) / "manifest.json").read_text())["quality_summary"]
        for k, v in qsum["counts"].items():
            flag_counts[k] = flag_counts.get(k, 0) + v
        long_gaps.extend(qsum.get("long_gaps", []))
        utc_ms = (raw["normalized_utc_time"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
                  .astype("datetime64[ms]").astype("int64"))
        quar = qua["quarantined"].to_numpy()
        n_ticks += n
        n_quar += int(quar.sum())
        ok = ~quar
        first_utc = utc_ms[0] if first_utc is None else first_utc
        last_utc = utc_ms[-1]
        # weekend / long gap boundaries across chunks
        if prev_last_utc is not None and utc_ms[0] - prev_last_utc > 24 * 3600 * 1000:
            weekend.append({"last_tick_utc_ms": int(prev_last_utc), "first_tick_utc_ms": int(utc_ms[0])})
        gaps_in = np.flatnonzero(np.diff(utc_ms) > 24 * 3600 * 1000)
        for g in gaps_in:
            weekend.append({"last_tick_utc_ms": int(utc_ms[g]), "first_tick_utc_ms": int(utc_ms[g + 1])})
        prev_last_utc = int(utc_ms[-1])
        hours = ((utc_ms // 3_600_000) % 24).astype(int)
        hour_ticks += np.bincount(hours, minlength=24)
        for h in np.unique(hours):
            hour_days[int(h)].add(d.isoformat())
        sess = classify_sessions(utc_ms)
        for sname, c in zip(*np.unique(sess, return_counts=True)):
            ticks_sess[str(sname)] = ticks_sess.get(str(sname), 0) + int(c)
        sp_pts = der["spread_points"].to_numpy()
        sp = der["spread"].to_numpy()
        rel = der["relative_spread"].to_numpy()
        mid = der["mid"].to_numpy()
        zero_spread += int(((sp == 0) & ok).sum())
        spread_all.add(sp_pts[ok])
        spread_rel.add(rel[ok])
        for h in np.unique(hours):
            spread_hour[int(h)].add(sp_pts[ok & (hours == h)])
        for sname in np.unique(sess):
            spread_sess.setdefault(str(sname), _Reservoir(100_000)).add(sp_pts[ok & (sess == sname)])
        top = np.argsort(np.where(ok, sp_pts, -np.inf))[-3:]
        for i in top:
            if np.isfinite(sp_pts[i]):
                widest.append({"utc_ms": int(utc_ms[i]), "spread_points": float(sp_pts[i]), "spread": float(sp[i])})
        dms = np.diff(raw["source_time_msc"].to_numpy())
        interval_ms.add(dms[(dms > 0) & (dms < 3600_000)].astype("float64"))
        # rollover view: spread by minute offset from 17:00 New York local
        ny = pd.to_datetime(utc_ms, unit="ms", utc=True).tz_convert("America/New_York")
        mnt = (ny.hour.to_numpy() * 60 + ny.minute.to_numpy()) - 17 * 60
        sel = ok & (np.abs(mnt) <= 60) & (ny.weekday.to_numpy() < 5)
        for m in np.unique(mnt[sel]):
            rollover.setdefault(int(m), _Reservoir(20_000)).add(sp_pts[sel & (mnt == m)])
        # 1-minute mid returns (bps), grouped by hour/session
        minute = utc_ms // 60_000
        df = pd.DataFrame({"m": minute[ok], "mid": mid[ok]})
        if len(df) > 1:
            last = df.groupby("m", sort=True)["mid"].last()
            r = np.diff(np.log(last.to_numpy())) * 1e4
            mm = last.index.to_numpy()[1:]
            hh = ((mm // 60) % 24).astype(int)
            ss = classify_sessions(mm * 60_000)
            for h in np.unique(hh):
                vol_hour[int(h)].append(r[hh == h])
            for sname in np.unique(ss):
                vol_sess.setdefault(str(sname), []).append(r[ss == sname])
        d += timedelta(days=1)

    widest = sorted(widest, key=lambda x: -x["spread_points"])[:20]

    def vstats(parts: list[np.ndarray]) -> dict[str, Any]:
        if not parts:
            return {"n": 0}
        v = np.concatenate(parts)
        return {"n_minutes": int(len(v)), "std_bps": float(v.std()), "mean_abs_bps": float(np.abs(v).mean())}

    # weekend boundary summary in New York local time
    wk = []
    for w in weekend[:200]:
        a = pd.Timestamp(w["last_tick_utc_ms"], unit="ms", tz="UTC").tz_convert("America/New_York")
        b = pd.Timestamp(w["first_tick_utc_ms"], unit="ms", tz="UTC").tz_convert("America/New_York")
        wk.append({"last_tick_utc": str(pd.Timestamp(w["last_tick_utc_ms"], unit="ms", tz="UTC")),
                   "first_tick_utc": str(pd.Timestamp(w["first_tick_utc_ms"], unit="ms", tz="UTC")),
                   "last_tick_ny": a.strftime("%a %H:%M"), "first_tick_ny": b.strftime("%a %H:%M"),
                   "gap_hours": round((w["first_tick_utc_ms"] - w["last_tick_utc_ms"]) / 3.6e6, 2)})
    days_active = max(1, len(per_day) - len(days_empty))
    return {
        "scope": {"broker": broker, "server": server, "instrument": instrument, "start": start.isoformat(),
                  "end": end.isoformat(), "time_basis": sorted(time_basis)},
        "coverage": {"chunks_present": chunks_total, "days_missing": days_missing, "days_empty": days_empty,
                     "ticks": n_ticks, "quarantined_ticks": n_quar, "excluded_from_spread_and_vol_stats": n_quar,
                     "first_utc_ms": None if first_utc is None else int(first_utc),
                     "last_utc_ms": None if last_utc is None else int(last_utc)},
        "tick_frequency": {"ticks_per_day": per_day, "mean_ticks_per_active_day": n_ticks / days_active,
                           "ticks_by_utc_hour": {str(h): int(hour_ticks[h]) for h in range(24)},
                           "mean_ticks_per_hour_by_utc_hour": {str(h): float(hour_ticks[h] / max(1, len(hour_days[h])))
                                                               for h in range(24)},
                           "ticks_by_session": ticks_sess, "inter_tick_ms": interval_ms.stats()},
        "spread": {"points": spread_all.stats(), "relative": spread_rel.stats(), "zero_spread_ticks": zero_spread,
                   "by_utc_hour_points": {str(h): spread_hour[h].stats() for h in range(24)},
                   "by_session_points": {k: v.stats() for k, v in sorted(spread_sess.items())},
                   "widest_ticks": widest, "point": point},
        "data_quality": {"flag_counts": flag_counts, "long_gaps_listed": long_gaps[:100],
                         "n_long_gaps_listed": len(long_gaps)},
        "weekend_boundaries": wk,
        "rollover_spread_by_minute_from_17h_ny": {str(k): v.stats() for k, v in sorted(rollover.items())
                                                  if k % 5 == 0},
        "volatility_1m_returns_bps": {"by_utc_hour": {str(h): vstats(vol_hour[h]) for h in range(24)},
                                      "by_session": {k: vstats(v) for k, v in sorted(vol_sess.items())},
                                      "note": "descriptive only; first minute of each chunk has no prior return"},
    }


def render_markdown(p: dict[str, Any], symbol_meta: dict[str, Any] | None = None, *, title: str = "XAU/USD Data Profile",
                    generated_by: str = "fxscalp.market_data.profile") -> str:
    s = p["scope"]
    L: list[str] = [f"# {title}", "",
                    f"Generated by `{generated_by}`. **Descriptive data analysis only: no trading signal, strategy or "
                    "performance claim is made or implied.**", "",
                    f"- Broker/server: `{s['broker']}` / `{s['server']}`", f"- Instrument: `{s['instrument']}`",
                    f"- Source-day range: {s['start']} .. {s['end']}", f"- Time basis: {', '.join(s['time_basis'])}", ""]
    c = p["coverage"]
    L += ["## Coverage", "", f"- Chunks present: {c['chunks_present']}; missing days: {len(c['days_missing'])}; "
          f"empty days: {len(c['days_empty'])}",
          f"- Ticks: {c['ticks']:,}; quarantined (excluded from spread/volatility statistics, never deleted): "
          f"{c['quarantined_ticks']:,}", ""]
    tf = p["tick_frequency"]
    L += ["## Tick frequency", "", f"- Mean ticks per active day: {tf['mean_ticks_per_active_day']:,.0f}",
          f"- Inter-tick interval (ms): {_fmt(tf['inter_tick_ms'])}", "", "| UTC hour | mean ticks/hour |", "|---|---|"]
    L += [f"| {h} | {v:,.0f} |" for h, v in tf["mean_ticks_per_hour_by_utc_hour"].items()]
    L += ["", "Ticks by session: " + ", ".join(f"{k}={v:,}" for k, v in sorted(tf["ticks_by_session"].items())), ""]
    sp = p["spread"]
    L += ["## Spread", "", f"- Spread in points: {_fmt(sp['points'])}", f"- Relative spread: {_fmt(sp['relative'])}",
          f"- Zero-spread ticks: {sp['zero_spread_ticks']:,}", "", "### By UTC hour (points)", "",
          "| hour | n | p50 | p95 | p99 |", "|---|---|---|---|---|"]
    for h, v in sp["by_utc_hour_points"].items():
        if v.get("n"):
            L.append(f"| {h} | {v['n']:,} | {v['p50']:.1f} | {v['p95']:.1f} | {v['p99']:.1f} |")
    L += ["", "### By session (points)", "", "| session | n | p50 | p95 | p99 |", "|---|---|---|---|---|"]
    for k, v in sp["by_session_points"].items():
        if v.get("n"):
            L.append(f"| {k} | {v['n']:,} | {v['p50']:.1f} | {v['p95']:.1f} | {v['p99']:.1f} |")
    L += ["", "### Widest spreads observed", ""]
    L += [f"- {pd.Timestamp(w['utc_ms'], unit='ms', tz='UTC')}: {w['spread_points']:.1f} points" for w in sp["widest_ticks"][:10]]
    dq = p["data_quality"]
    L += ["", "## Missing data, duplicates, quality flags", "",
          "| flag | ticks |", "|---|---|"] + [f"| {k} | {v:,} |" for k, v in dq["flag_counts"].items()]
    L += ["", f"Gaps longer than the report threshold: {dq['n_long_gaps_listed']}", "", "## Weekend boundaries", "",
          "| last tick (NY) | first tick (NY) | gap h |", "|---|---|---|"]
    L += [f"| {w['last_tick_ny']} | {w['first_tick_ny']} | {w['gap_hours']} |" for w in p["weekend_boundaries"][:20]]
    L += ["", "## Rollover behaviour (spread points by minutes from 17:00 New York, every 5 min)", "",
          "| min | n | p50 | p95 |", "|---|---|---|---|"]
    for k, v in p["rollover_spread_by_minute_from_17h_ny"].items():
        if v.get("n"):
            L.append(f"| {k} | {v['n']:,} | {v['p50']:.1f} | {v['p95']:.1f} |")
    vol = p["volatility_1m_returns_bps"]
    L += ["", "## Volatility characteristics (1-minute mid returns, bps; descriptive)", "",
          "| key | minutes | std | mean abs |", "|---|---|---|---|"]
    for k, v in {**{f"UTC {int(h):02d}h": x for h, x in vol["by_utc_hour"].items()}, **vol["by_session"]}.items():
        if v.get("n_minutes"):
            L.append(f"| {k} | {v['n_minutes']:,} | {v['std_bps']:.2f} | {v['mean_abs_bps']:.2f} |")
    if symbol_meta:
        L += ["", "## Broker symbol metadata (as returned; units not assumed)", "", "```json",
              json.dumps(symbol_meta.get("typed", symbol_meta), indent=2, sort_keys=True, default=str), "```"]
        issues = symbol_meta.get("consistency_issues") or []
        if issues:
            L += ["", "Consistency checks: " + "; ".join(f"{i['severity']} {i['code']}" for i in issues)]
    return "\n".join(L) + "\n"


def _fmt(d: dict[str, Any]) -> str:
    if not d.get("n"):
        return "n/a"
    keys = ("mean", "p1", "p5", "p50", "p95", "p99", "p99.9")
    return f"n={d['n']:,} " + " ".join(f"{k}={d[k]:.4g}" for k in keys if k in d)
