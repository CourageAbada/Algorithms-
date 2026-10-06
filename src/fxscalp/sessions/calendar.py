"""Timezone-aware session classification. Canonical time is UTC; sessions are defined by LOCAL wall-clock
windows in their home market's IANA zone, so DST shifts (US and UK change on different dates) are handled by
the tz database, never by hardcoded UTC hours.

The window hours below are CONVENTIONS (commonly quoted FX session hours), not facts about any broker's
liquidity. The XAUUSD data profile is where they get validated or replaced. Rules are plain data so they can
be overridden from configuration.

Primary label precedence (documented, deterministic):
    MARKET_CLOSED > LONDON_NEW_YORK_OVERLAP > NEW_YORK > LONDON > ASIA > OFF_HOURS
Membership flags (in_asia / in_london / in_new_york) are available separately for analyses that need overlaps
of other pairs (e.g. Asia/London).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


class Session(str, Enum):
    ASIA = "ASIA"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    LONDON_NEW_YORK_OVERLAP = "LONDON_NEW_YORK_OVERLAP"
    OFF_HOURS = "OFF_HOURS"
    MARKET_CLOSED = "MARKET_CLOSED"


@dataclass(frozen=True)
class Window:
    zone: str
    start: time
    end: time          # exclusive; windows do not cross local midnight

    def minutes(self) -> tuple[int, int]:
        return self.start.hour * 60 + self.start.minute, self.end.hour * 60 + self.end.minute


@dataclass(frozen=True)
class SessionRules:
    asia: Window = Window("Asia/Tokyo", time(9, 0), time(18, 0))
    london: Window = Window("Europe/London", time(8, 0), time(17, 0))
    new_york: Window = Window("America/New_York", time(8, 0), time(17, 0))
    # nominal weekly closure in NY local time: Friday 17:00 -> Sunday 17:00
    closure_zone: str = "America/New_York"
    closure_start: time = time(17, 0)       # Friday
    closure_end: time = time(17, 0)         # Sunday


DEFAULT_RULES = SessionRules()


def _local(utc_ms: np.ndarray, zone: str) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(utc_ms, dtype="int64"), unit="ms", utc=True).tz_convert(zone)


def _in_window(utc_ms: np.ndarray, w: Window) -> np.ndarray:
    loc = _local(utc_ms, w.zone)
    m = loc.hour.to_numpy() * 60 + loc.minute.to_numpy()
    s, e = w.minutes()
    weekday = loc.weekday.to_numpy() < 5          # sessions run Mon-Fri local time
    return (m >= s) & (m < e) & weekday


def market_closed(utc_ms: np.ndarray, rules: SessionRules = DEFAULT_RULES) -> np.ndarray:
    loc = _local(utc_ms, rules.closure_zone)
    wd = loc.weekday.to_numpy()
    m = loc.hour.to_numpy() * 60 + loc.minute.to_numpy()
    cs, ce = rules.closure_start.hour * 60 + rules.closure_start.minute, rules.closure_end.hour * 60 + rules.closure_end.minute
    return ((wd == 4) & (m >= cs)) | (wd == 5) | ((wd == 6) & (m < ce))


def session_flags(utc_ms: np.ndarray, rules: SessionRules = DEFAULT_RULES) -> dict[str, np.ndarray]:
    return {"in_asia": _in_window(utc_ms, rules.asia), "in_london": _in_window(utc_ms, rules.london),
            "in_new_york": _in_window(utc_ms, rules.new_york), "market_closed": market_closed(utc_ms, rules)}


def classify_sessions(utc_ms: np.ndarray, rules: SessionRules = DEFAULT_RULES) -> np.ndarray:
    """Primary session label per UTC-millisecond timestamp (array of str)."""
    f = session_flags(utc_ms, rules)
    out = np.full(len(f["in_asia"]), Session.OFF_HOURS.value, dtype=object)
    out[f["in_asia"]] = Session.ASIA.value
    out[f["in_london"]] = Session.LONDON.value
    out[f["in_new_york"]] = Session.NEW_YORK.value
    out[f["in_london"] & f["in_new_york"]] = Session.LONDON_NEW_YORK_OVERLAP.value
    out[f["market_closed"]] = Session.MARKET_CLOSED.value
    return out


def classify_instant(utc: datetime, rules: SessionRules = DEFAULT_RULES) -> Session:
    if utc.tzinfo is None:
        raise ValueError("naive datetime: pass a timezone-aware UTC datetime")
    ms = np.array([int(utc.timestamp() * 1000)], dtype="int64")
    return Session(classify_sessions(ms, rules)[0])


def window_utc(day: date, w: Window) -> tuple[datetime, datetime]:
    """UTC start/end of a local-day window (DST-correct) for tests and reporting."""
    z = ZoneInfo(w.zone)
    s = datetime.combine(day, w.start, tzinfo=z)
    e = datetime.combine(day, w.end, tzinfo=z)
    from datetime import timezone
    return s.astimezone(timezone.utc), e.astimezone(timezone.utc)


def overlap_utc(day: date, rules: SessionRules = DEFAULT_RULES) -> tuple[datetime, datetime] | None:
    ls, le = window_utc(day, rules.london)
    ns, ne = window_utc(day, rules.new_york)
    s, e = max(ls, ns), min(le, ne)
    return (s, e) if s < e else None


