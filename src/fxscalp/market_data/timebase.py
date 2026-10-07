"""Empirical MT5 time semantics: calibration, DST inference, and explicit normalisation to UTC.

Nothing here assumes MT5 timestamps are UTC. The workflow is:

1. ``estimate_offset(samples)``: compare the latest-tick timestamp with the local UTC clock over
   many samples while the feed is live. The raw difference is ``offset - tick_age`` (plus clock
   error), so it is rounded to a plausible offset granularity and the residual is reported.
2. ``infer_dst_from_weekly_open(ticks)``: (optional, needs months of history) checks whether the
   source-time of the weekly market open shifts across the US DST change. This rests on an
   explicit ASSUMPTION (the weekly open is anchored to 17:00 New York) and yields an inference,
   never a verification.
3. ``resolve_time_basis`` combines both into a ``TimeBaseSpec`` with an explicit ``TimeBasis``.
4. ``TimeBase.normalize`` converts raw source time to UTC using that spec; ambiguous (DST fold)
   instants are flagged, never silently resolved.

Transformation (documented, never silent):  ``normalized_utc = source_time - offset(rule, instant)``.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from fxscalp.brokers.errors import TimeCalibrationError

OFFSET_GRANULARITY_S = 900            # real UTC offsets are multiples of 15 minutes
MAX_PLAUSIBLE_OFFSET_S = 14 * 3600
NY_ZONE = "America/New_York"


class TimeBasis(str, Enum):
    UNVERIFIED = "unverified"
    UTC_VERIFIED = "utc_verified"                  # measured offset is 0
    SERVER_FIXED_OFFSET = "server_fixed_offset"    # measured constant offset; DST may be undetermined
    SERVER_DST_RULE = "server_dst_rule"            # offset follows a DST rule supported by evidence


# --------------------------------------------------------------------------------------------
# Rules: offset = source_time - utc_time
# --------------------------------------------------------------------------------------------
def _ms(idx: "pd.DatetimeIndex") -> np.ndarray:
    """Epoch milliseconds of a DatetimeIndex, independent of pandas' internal resolution (NaT -> 0)."""
    arr = idx.tz_convert("UTC").tz_localize(None).to_numpy() if idx.tz is not None else idx.to_numpy()
    out = arr.astype("datetime64[ms]").astype("int64")
    return np.where(np.isnat(arr), 0, out)


@dataclass(frozen=True)
class ServerTimeRule:
    kind: str                      # "fixed" | "zone"
    name: str
    fixed_offset_s: int = 0        # kind == "fixed"
    zone: str = ""                 # kind == "zone": IANA zone providing DST dates
    shift_s: int = 0               # kind == "zone": offset = zone_utcoffset + shift_s

    @staticmethod
    def fixed(offset_s: int) -> "ServerTimeRule":
        return ServerTimeRule("fixed", f"fixed{offset_s:+d}s", fixed_offset_s=int(offset_s))

    @staticmethod
    def zone_shift(zone: str, shift_s: int, name: str | None = None) -> "ServerTimeRule":
        return ServerTimeRule("zone", name or f"{zone}{shift_s:+d}s", zone=zone, shift_s=int(shift_s))

    def offset_at_utc_ms(self, utc_ms: np.ndarray) -> np.ndarray:
        """Seconds to ADD to a UTC instant to obtain source time."""
        utc_ms = np.asarray(utc_ms, dtype="int64")
        if self.kind == "fixed":
            return np.full(utc_ms.shape, self.fixed_offset_s, dtype="int64")
        idx = pd.to_datetime(utc_ms, unit="ms", utc=True)
        local = idx.tz_convert(self.zone).tz_localize(None)
        zone_off_ms = _ms(local) - _ms(idx)
        return (zone_off_ms // 1000).astype("int64") + self.shift_s

    def source_to_utc_ms(self, source_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert source epoch-ms to UTC epoch-ms.

        Returns (utc_ms, ambiguous_mask, nonexistent_mask). In a DST fold the same wall-clock value
        occurs twice. Ticks in a fold are assigned to the first/second occurrence ONLY when the
        broker's tick order shows the wall-clock jumping back; otherwise they are flagged ambiguous
        and the first occurrence is used as a best effort.
        """
        source_ms = np.asarray(source_ms, dtype="int64")
        if self.kind == "fixed":
            z = np.zeros(source_ms.shape, dtype=bool)
            return source_ms - self.fixed_offset_s * 1000, z, z
        wall = pd.to_datetime(source_ms - self.shift_s * 1000, unit="ms")  # naive zone-local wall time
        first = wall.tz_localize(self.zone, ambiguous="NaT", nonexistent="NaT")
        bad = np.asarray(first.isna())
        utc = np.where(bad, 0, _ms(first)).astype("int64")
        if not bad.any():
            return utc, np.zeros(source_ms.shape, bool), np.zeros(source_ms.shape, bool)
        amb = np.zeros(source_ms.shape, bool)
        non = np.zeros(source_ms.shape, bool)
        # resolve folds from observed order; classify the rest as ambiguous / non-existent
        wall_dst_first = wall.tz_localize(self.zone, ambiguous=np.ones(len(wall), bool), nonexistent="NaT")
        wall_dst_second = wall.tz_localize(self.zone, ambiguous=np.zeros(len(wall), bool), nonexistent="NaT")
        is_amb = np.asarray(wall_dst_first.notna()) & bad
        is_non = bad & ~is_amb
        utc_first = _ms(wall_dst_first)
        utc_second = _ms(wall_dst_second)
        use_second = np.zeros(source_ms.shape, bool)
        pos = np.flatnonzero(is_amb)
        if pos.size:
            # contiguous runs of ambiguous ticks
            splits = np.flatnonzero(np.diff(pos) > 1) + 1
            for run in np.split(pos, splits):
                run_src = source_ms[run]
                back = np.flatnonzero(np.diff(run_src) < -1_800_000)  # wall clock jumped back > 30 min
                if back.size:
                    use_second[run[back[0] + 1:]] = True
                    # runs whose reversal was observed are resolved, not ambiguous
                else:
                    amb[run] = True
        resolved = np.where(use_second, utc_second, utc_first)
        utc = np.where(is_amb, resolved, utc)
        non[is_non] = True
        utc = np.where(is_non, source_ms - (self.shift_s * 1000), utc)  # best effort; flagged
        return utc.astype("int64"), amb, non


# --------------------------------------------------------------------------------------------
# Step 1: offset estimation
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OffsetSample:
    local_utc_s: float          # time.time() when the tick was received (midpoint of the call)
    tick_time_s: int            # raw tick.time
    tick_time_msc: int          # raw tick.time_msc
    call_latency_s: float = 0.0


@dataclass(frozen=True)
class OffsetEstimate:
    status: str                         # "ok" | "inconclusive"
    offset_s: int | None
    median_delta_s: float | None
    residual_s: float | None            # median_delta - offset  (= -tick_age + clock error)
    mad_s: float | None
    n_total: int
    n_used: int
    sec_msc_consistent: bool | None
    granularity_s: int
    notes: tuple[str, ...] = ()


def estimate_offset(samples: Sequence[OffsetSample], *, min_samples: int = 5, max_mad_s: float = 30.0,
                    max_abs_residual_s: float = 90.0) -> OffsetEstimate:
    """Estimate source_time - UTC from latest-tick samples. Never raises; returns inconclusive instead.

    Only samples where the tick advanced since the previous sample are used (a repeated tick means
    a stale/closed feed and would bias the estimate towards the past).
    """
    notes: list[str] = []
    n_total = len(samples)
    used: list[OffsetSample] = []
    prev_msc: int | None = None
    for s in samples:
        if prev_msc is None or s.tick_time_msc != prev_msc:
            used.append(s)
        prev_msc = s.tick_time_msc
    if len(used) < n_total:
        notes.append(f"{n_total - len(used)} samples ignored: tick did not advance (stale/closed feed)")
    sec_ok = all(abs(s.tick_time_msc // 1000 - s.tick_time_s) <= 1 for s in samples) if samples else None
    if sec_ok is False:
        notes.append("tick.time and tick.time_msc//1000 disagree for some samples")
    if len(used) < min_samples:
        notes.append(f"only {len(used)} usable samples (< {min_samples}); market may be closed")
        return OffsetEstimate("inconclusive", None, None, None, None, n_total, len(used), sec_ok,
                              OFFSET_GRANULARITY_S, tuple(notes))
    deltas = [s.tick_time_msc / 1000.0 - s.local_utc_s for s in used]
    med = statistics.median(deltas)
    mad = statistics.median(abs(d - med) for d in deltas)
    offset = int(round(med / OFFSET_GRANULARITY_S)) * OFFSET_GRANULARITY_S
    residual = med - offset
    if abs(offset) > MAX_PLAUSIBLE_OFFSET_S:
        notes.append(f"implausible offset {offset}s")
        return OffsetEstimate("inconclusive", None, med, residual, mad, n_total, len(used), sec_ok,
                              OFFSET_GRANULARITY_S, tuple(notes))
    if mad > max_mad_s:
        notes.append(f"sample spread too large (MAD {mad:.1f}s > {max_mad_s}s): feed jitter or stale ticks")
        return OffsetEstimate("inconclusive", None, med, residual, mad, n_total, len(used), sec_ok,
                              OFFSET_GRANULARITY_S, tuple(notes))
    if abs(residual) > max_abs_residual_s:
        notes.append(f"residual {residual:.1f}s is large: tick age or local clock error; "
                     "check Windows time sync and that the market is active")
        return OffsetEstimate("inconclusive", None, med, residual, mad, n_total, len(used), sec_ok,
                              OFFSET_GRANULARITY_S, tuple(notes))
    notes.append("offset rounded to the nearest 15 min; assumes the local clock is NTP-synchronised")
    return OffsetEstimate("ok", offset, med, residual, mad, n_total, len(used), sec_ok,
                          OFFSET_GRANULARITY_S, tuple(notes))


# --------------------------------------------------------------------------------------------
# Step 2: DST inference from the weekly open (assumption-labelled)
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DstInference:
    classification: str     # "us_dst_anchored" | "fixed_offset_vs_ny_open" | "inconclusive"
    n_weeks_dst: int
    n_weeks_std: int
    median_open_tow_dst_s: float | None
    median_open_tow_std_s: float | None
    delta_s: float | None
    assumption: str = ("The weekly market open is anchored to 17:00 America/New_York "
                       "(spot-FX convention). This may not hold for every broker/CFD; the result is "
                       "an inference, not a verification.")
    notes: tuple[str, ...] = ()


def weekly_opens(source_s: np.ndarray, gap_s: int = 24 * 3600) -> np.ndarray:
    """Source-epoch seconds of the first tick after each gap longer than ``gap_s``."""
    source_s = np.asarray(source_s, dtype="int64")
    if source_s.size < 2:
        return np.array([], dtype="int64")
    d = np.diff(source_s)
    return source_s[1:][d > gap_s]


def infer_dst_from_weekly_open(source_s: np.ndarray, anchor_offset_s: int, *, min_weeks_each: int = 4,
                               tol_s: int = 900) -> DstInference:
    """Infer from a tick stream (weekly opens = first tick after each >24 h gap)."""
    return infer_dst_from_opens(weekly_opens(source_s), anchor_offset_s, min_weeks_each=min_weeks_each, tol_s=tol_s)


def infer_dst_from_opens(opens: np.ndarray, anchor_offset_s: int, *, min_weeks_each: int = 4,
                         tol_s: int = 900) -> DstInference:
    """Infer from explicit weekly-open instants (source-epoch seconds), e.g. from copy_ticks_from probes."""
    opens = np.asarray(opens, dtype="int64")
    notes: list[str] = []
    if opens.size == 0:
        return DstInference("inconclusive", 0, 0, None, None, None, notes=("no weekly gaps found",))
    # time of week in the source domain (seconds since Sunday 00:00); epoch day 0 was a Thursday
    tow = (opens + 4 * 86400) % 604800
    approx_utc = pd.to_datetime((opens - anchor_offset_s) * 1000, unit="ms", utc=True)
    local_ny = approx_utc.tz_convert(NY_ZONE)
    is_dst = np.array([bool(t.dst()) for t in local_ny])
    # drop weeks within 8 days of a transition: classification is unreliable there
    trans_flag = np.zeros(len(opens), bool)
    for i, t in enumerate(local_ny):
        a = (t - pd.Timedelta(days=8)).dst()
        b = (t + pd.Timedelta(days=8)).dst()
        trans_flag[i] = (bool(a) != bool(t.dst())) or (bool(b) != bool(t.dst()))
    keep = ~trans_flag
    dst_t, std_t = tow[keep & is_dst], tow[keep & ~is_dst]
    if len(dst_t) < min_weeks_each or len(std_t) < min_weeks_each:
        return DstInference("inconclusive", len(dst_t), len(std_t), None, None, None,
                            notes=(f"need >= {min_weeks_each} weeks on each side of the US DST change",))
    m_dst, m_std = float(np.median(dst_t)), float(np.median(std_t))
    delta = m_dst - m_std
    if abs(delta) <= tol_s:
        cls = "us_dst_anchored"
    elif abs(delta + 3600) <= tol_s:
        cls = "fixed_offset_vs_ny_open"
    else:
        cls = "inconclusive"
        notes.append(f"weekly-open shift {delta:.0f}s matches neither 0 nor -3600s")
    return DstInference(cls, int(len(dst_t)), int(len(std_t)), m_dst, m_std, delta, notes=tuple(notes))


# --------------------------------------------------------------------------------------------
# Step 3: combine
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeBaseSpec:
    basis: TimeBasis
    rule: ServerTimeRule | None
    dst_determined: bool
    calibrated_at_utc: str
    broker: str
    server: str
    estimate: OffsetEstimate | None
    dst_inference: DstInference | None
    notes: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["basis"] = self.basis.value
        return d

    def identity(self) -> dict[str, Any]:
        """The part of the spec that determines HOW times are converted (excludes calibration timestamps and
        residuals, so re-calibrating to the same rule does not change a dataset's identity)."""
        return {"basis": self.basis.value, "rule": asdict(self.rule) if self.rule else None,
                "dst_determined": self.dst_determined}

    @staticmethod
    def from_json(d: dict[str, Any]) -> "TimeBaseSpec":
        rule = ServerTimeRule(**d["rule"]) if d.get("rule") else None
        est = OffsetEstimate(**{**d["estimate"], "notes": tuple(d["estimate"].get("notes", ()))}) if d.get("estimate") else None
        dst = DstInference(**{**d["dst_inference"], "notes": tuple(d["dst_inference"].get("notes", ()))}) if d.get("dst_inference") else None
        return TimeBaseSpec(TimeBasis(d["basis"]), rule, d["dst_determined"], d["calibrated_at_utc"], d["broker"],
                            d["server"], est, dst, tuple(d.get("notes", ())))


def unverified_spec(broker: str, server: str, reason: str) -> TimeBaseSpec:
    return TimeBaseSpec(TimeBasis.UNVERIFIED, None, False, _now_iso(), broker, server, None, None, (reason,))


def resolve_time_basis(estimate: OffsetEstimate, dst: DstInference | None, *, broker: str, server: str,
                       calibrated_at_utc: datetime | None = None) -> TimeBaseSpec:
    when = calibrated_at_utc or datetime.now(timezone.utc)
    if estimate.status != "ok" or estimate.offset_s is None:
        return TimeBaseSpec(TimeBasis.UNVERIFIED, None, False, when.isoformat(), broker, server, estimate, dst,
                            ("offset calibration inconclusive",) + estimate.notes)
    off = estimate.offset_s
    if dst is not None and dst.classification == "us_dst_anchored":
        ny_off = int(pd.Timestamp(when).tz_convert(NY_ZONE).utcoffset().total_seconds())
        return TimeBaseSpec(TimeBasis.SERVER_DST_RULE, ServerTimeRule.zone_shift(NY_ZONE, off - ny_off, "ny_anchored"),
                            True, when.isoformat(), broker, server, estimate, dst,
                            ("DST rule inferred from weekly-open evidence (assumption-labelled)",))
    fixed_supported = bool(dst and dst.classification == "fixed_offset_vs_ny_open")
    note = ("DST evidence supports a fixed offset" if fixed_supported
            else "DST behaviour NOT determined from the available data; conversion of data far from the "
                 "calibration date may be off by 1 h across DST changes")
    basis = TimeBasis.UTC_VERIFIED if off == 0 else TimeBasis.SERVER_FIXED_OFFSET
    return TimeBaseSpec(basis, ServerTimeRule.fixed(off), fixed_supported, when.isoformat(), broker, server,
                        estimate, dst, (note,))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------------------------
# Step 4: use
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class NormalizedTime:
    utc_ms: np.ndarray
    ambiguous: np.ndarray
    nonexistent: np.ndarray
    basis: TimeBasis


class TimeBase:
    """Applies a TimeBaseSpec. Refuses to normalise when the basis is UNVERIFIED (callers must opt in)."""

    def __init__(self, spec: TimeBaseSpec):
        self.spec = spec

    @property
    def basis(self) -> TimeBasis:
        return self.spec.basis

    def normalize(self, source_ms: np.ndarray, *, allow_unverified: bool = False) -> NormalizedTime:
        if self.spec.rule is None:
            if not allow_unverified:
                raise TimeCalibrationError(
                    "Time basis is UNVERIFIED: no calibrated server-time rule. Run the time calibration "
                    "(scripts.verify_mt5 --time-samples N) while the market is open, or pass "
                    "allow_unverified=True to store a best-effort 'UTC==source' assumption explicitly.")
            rule = ServerTimeRule.fixed(0)
        else:
            rule = self.spec.rule
        utc, amb, non = rule.source_to_utc_ms(source_ms)
        return NormalizedTime(utc, amb, non, self.spec.basis)

    def utc_to_source_s(self, utc_s: np.ndarray | int) -> np.ndarray:
        """Inverse mapping, used to express UTC request windows in the source domain."""
        rule = self.spec.rule or ServerTimeRule.fixed(0)
        a = np.atleast_1d(np.asarray(utc_s, dtype="int64"))
        return a + rule.offset_at_utc_ms(a * 1000)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.spec.to_json(), indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "TimeBase":
        return TimeBase(TimeBaseSpec.from_json(json.loads(path.read_text(encoding="utf-8"))))


