"""Session features. All "so far" quantities are running aggregates inside the current session INSTANCE (one
continuous run of a session's window); nothing about the final session range is ever visible while it is running.

Membership is evaluated at the START of the row's bar (all data in the bar then lies inside the window); minutes are
measured at the bar END (the decision time). Window hours are conventions from ``sessions/calendar.py`` (local
wall-clock in the session's own IANA zone, hence DST-correct); they are validated against real data in Phase 2B.

Reference session used for the "so far"/clock features: LONDON -> LONDON, NEW_YORK and LONDON_NEW_YORK_OVERLAP ->
NEW_YORK, ASIA -> ASIA, otherwise none (NaN).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxscalp.features import ops
from fxscalp.sessions.calendar import DEFAULT_RULES, Session, SessionRules, classify_sessions, session_flags

SESSION_CODES = {Session.ASIA.value: 0, Session.LONDON.value: 1, Session.NEW_YORK.value: 2,
                 Session.LONDON_NEW_YORK_OVERLAP.value: 3, Session.OFF_HOURS.value: 4, Session.MARKET_CLOSED.value: 5}


def _local_minutes(ms: np.ndarray, zone: str) -> np.ndarray:
    idx = pd.to_datetime(np.asarray(ms, dtype="int64"), unit="ms", utc=True).tz_convert(zone)
    return (idx.hour.to_numpy() * 60 + idx.minute.to_numpy() + idx.second.to_numpy() / 60.0
            + idx.microsecond.to_numpy() / 6e7)


def session_arrays(start_ms: np.ndarray, end_ms: np.ndarray, mid_asof: np.ndarray, mid_high: np.ndarray,
                   mid_low: np.ndarray, grid_s: int, rules: SessionRules = DEFAULT_RULES) -> dict[str, np.ndarray]:
    n = len(start_ms)
    label = classify_sessions(start_ms, rules)
    flags = session_flags(start_ms, rules)
    closed = flags["market_closed"]
    in_a, in_l, in_n = (flags["in_asia"] & ~closed), (flags["in_london"] & ~closed), (flags["in_new_york"] & ~closed)
    out: dict[str, np.ndarray] = {
        "session_code": np.array([SESSION_CODES[x] for x in label], dtype="float64") if n else np.array([]),
        "is_asia": in_a.astype("float64"), "is_london": in_l.astype("float64"), "is_new_york": in_n.astype("float64"),
        "is_london_new_york_overlap": (in_l & in_n).astype("float64")}
    ref = np.where(label == Session.ASIA.value, "A", np.where(label == Session.LONDON.value, "L",
                   np.where((label == Session.NEW_YORK.value) | (label == Session.LONDON_NEW_YORK_OVERLAP.value), "N", "")))
    wins = {"A": rules.asia, "L": rules.london, "N": rules.new_york}
    since = np.full(n, np.nan)
    until = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    mean = np.full(n, np.nan)
    complete = np.full(n, np.nan)
    for key, w in wins.items():
        member = {"A": in_a, "L": in_l, "N": in_n}[key]
        onset = member & ~np.concatenate(([False], member[:-1]))
        inst = np.cumsum(onset)
        s_min, e_min = w.minutes()
        lm_end = _local_minutes(end_ms, w.zone)
        sel = ref == key
        since[sel] = (lm_end - s_min)[sel]
        until[sel] = (e_min - lm_end)[sel]
        h = np.where(member, mid_high, np.nan)
        l_ = np.where(member, mid_low, np.nan)
        m = np.where(member, mid_asof, np.nan)
        # instance-restricted running aggregates (rows outside the session carry NaN and are not selected)
        hi_k = ops.cummax_by(h, inst)
        lo_k = ops.cummin_by(l_, inst)
        mean_k = ops.cummean_by(np.nan_to_num(m, nan=0.0), inst)           # as-of mid is never NaN inside a segment
        hi[sel], lo[sel], mean[sel] = hi_k[sel], lo_k[sel], mean_k[sel]
        # was the instance observed from (nearly) its open? first row of each instance must be within 2 grid steps
        first_since = pd.Series(np.where(member, lm_end - s_min, np.nan)).groupby(inst).transform("first").to_numpy()
        complete[sel] = (first_since <= 2 * grid_s / 60.0)[sel].astype("float64")
    out["minutes_since_session_open"] = since
    out["minutes_until_session_close"] = until
    out["session_high_so_far"] = hi
    out["session_low_so_far"] = lo
    out["session_mean_so_far"] = mean
    out["session_so_far_complete"] = complete
    return out
