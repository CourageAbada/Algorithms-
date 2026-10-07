"""Chronological walk-forward folds, purge/embargo and the protected FINAL holdout (Phase 2B-A, frozen).

Everything is expressed on normalised UTC milliseconds. Partitions are whole SOURCE days (the acquisition unit) converted with
the calibrated +10800 s offset that is valid for the whole usable range; day D covers [D 00:00 source, D+1 00:00 source).

Rules (also enforced by tests):
* training strictly precedes validation; validation blocks do not overlap and increase in time;
* PURGE: a training sample is used only if the last instant of information its label uses (``label_end_ms``) is <= the
  validation start minus the embargo; a validation sample only if its label ends inside the validation block;
* EMBARGO: ``EMBARGO_S`` (1800 s = the longest feature look-back for spread/volume percentiles) between train and validation;
* labels never cross a segment boundary by construction (labels.py), so purge only has to handle partition boundaries;
* the FINAL holdout is chosen by chronology/coverage only and is closed to development: ``assert_development_only``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np

SOURCE_OFFSET_S = 10800            # calibration record timecal-c9cc2a3a65a3dd80 (valid 2026-04-05 .. 2026-10-07)
EMBARGO_S = 1800
MAX_LABEL_SPAN_S = 1 + 300         # latency 1 s + longest candidate horizon
FIRST_ELIGIBLE_DAY = date(2026, 4, 6)       # 2026-04-01/02 are TIME_BASIS_UNCERTAIN; 04-03..05 are closed (Good Friday weekend)
DEV_LAST_DAY = date(2026, 9, 11)
HOLDOUT_FIRST_DAY = date(2026, 9, 14)
HOLDOUT_LAST_DAY = date(2026, 10, 6)


class HoldoutViolation(RuntimeError):
    """Development code touched the final untouched evaluation period."""


def day_start_utc_ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000) - SOURCE_OFFSET_S * 1000


def day_end_utc_ms(day: date) -> int:
    return day_start_utc_ms(day + timedelta(days=1))


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_first_day: str
    train_last_day: str
    val_first_day: str
    val_last_day: str

    @property
    def train_start_ms(self) -> int:
        return day_start_utc_ms(date.fromisoformat(self.train_first_day))

    @property
    def train_end_ms(self) -> int:                 # exclusive end of the training block
        return day_end_utc_ms(date.fromisoformat(self.train_last_day))

    @property
    def val_start_ms(self) -> int:
        return day_start_utc_ms(date.fromisoformat(self.val_first_day))

    @property
    def val_end_ms(self) -> int:                   # exclusive
        return day_end_utc_ms(date.fromisoformat(self.val_last_day))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "train_start_ms": self.train_start_ms, "train_end_ms": self.train_end_ms,
                "val_start_ms": self.val_start_ms, "val_end_ms": self.val_end_ms}


def default_folds() -> list[FoldSpec]:
    t0 = FIRST_ELIGIBLE_DAY.isoformat()
    return [
        FoldSpec("F1", t0, "2026-05-29", "2026-06-01", "2026-06-19"),
        FoldSpec("F2", t0, "2026-06-19", "2026-06-22", "2026-07-10"),
        FoldSpec("F3", t0, "2026-07-10", "2026-07-13", "2026-07-31"),
        FoldSpec("F4", t0, "2026-07-31", "2026-08-03", "2026-08-21"),
        FoldSpec("F5", t0, "2026-08-21", "2026-08-24", "2026-09-11"),
    ]


@dataclass(frozen=True)
class FinalHoldout:
    first_day: str = HOLDOUT_FIRST_DAY.isoformat()
    last_day: str = HOLDOUT_LAST_DAY.isoformat()
    embargo_s: int = EMBARGO_S

    @property
    def start_ms(self) -> int:
        return day_start_utc_ms(date.fromisoformat(self.first_day))

    @property
    def end_ms(self) -> int:
        return day_end_utc_ms(date.fromisoformat(self.last_day))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "start_ms": self.start_ms, "end_ms": self.end_ms}


def validate_plan(folds: list[FoldSpec], holdout: FinalHoldout, embargo_s: int = EMBARGO_S) -> list[str]:
    """Structural invariants of a fold plan; returns a list of violations (empty = valid)."""
    bad: list[str] = []
    if embargo_s < MAX_LABEL_SPAN_S:
        bad.append("embargo shorter than the longest label span")
    prev_val_end = None
    for f in folds:
        if not f.train_start_ms < f.train_end_ms:
            bad.append(f"{f.name}: empty training block")
        if not f.train_end_ms <= f.val_start_ms:
            bad.append(f"{f.name}: training does not strictly precede validation")
        if f.val_start_ms >= f.val_end_ms:
            bad.append(f"{f.name}: empty validation block")
        if prev_val_end is not None and f.val_start_ms < prev_val_end:
            bad.append(f"{f.name}: validation block overlaps/precedes the previous one")
        if f.val_end_ms + embargo_s * 1000 > holdout.start_ms:
            bad.append(f"{f.name}: validation (+embargo) reaches the final holdout")
        prev_val_end = f.val_end_ms
    if holdout.start_ms >= holdout.end_ms:
        bad.append("empty final holdout")
    return bad


def train_mask(fold: FoldSpec, obs_ms: np.ndarray, label_end_ms: np.ndarray, embargo_s: int = EMBARGO_S) -> np.ndarray:
    """Samples usable for TRAINING in ``fold`` after purge + embargo."""
    obs, end = np.asarray(obs_ms, dtype="int64"), np.asarray(label_end_ms, dtype="int64")
    return (obs >= fold.train_start_ms) & (obs < fold.train_end_ms) & (end <= fold.val_start_ms - embargo_s * 1000)


def val_mask(fold: FoldSpec, obs_ms: np.ndarray, label_end_ms: np.ndarray) -> np.ndarray:
    obs, end = np.asarray(obs_ms, dtype="int64"), np.asarray(label_end_ms, dtype="int64")
    return (obs >= fold.val_start_ms) & (obs < fold.val_end_ms) & (end <= fold.val_end_ms)


def purge_counts(fold: FoldSpec, obs_ms: np.ndarray, label_end_ms: np.ndarray, embargo_s: int = EMBARGO_S) -> dict[str, int]:
    obs, end = np.asarray(obs_ms, dtype="int64"), np.asarray(label_end_ms, dtype="int64")
    in_train = (obs >= fold.train_start_ms) & (obs < fold.train_end_ms)
    in_val = (obs >= fold.val_start_ms) & (obs < fold.val_end_ms)
    return {"train_candidates": int(in_train.sum()), "train_kept": int(train_mask(fold, obs, end, embargo_s).sum()),
            "train_purged_or_embargoed": int((in_train & ~train_mask(fold, obs, end, embargo_s)).sum()),
            "val_candidates": int(in_val.sum()), "val_kept": int(val_mask(fold, obs, end).sum()),
            "val_purged": int((in_val & ~val_mask(fold, obs, end)).sum())}


def assert_development_only(ts_ms: np.ndarray, holdout: FinalHoldout | None = None, *, what: str = "data") -> None:
    """Raise HoldoutViolation if any timestamp lies in the final holdout."""
    h = holdout or FinalHoldout()
    t = np.asarray(ts_ms, dtype="int64")
    if t.size and int(t.max()) >= h.start_ms:
        raise HoldoutViolation(f"{what} reaches the FINAL untouched evaluation period (>= {h.first_day}); development may not use it")


def development_mask(ts_ms: np.ndarray, label_end_ms: np.ndarray | None = None, holdout: FinalHoldout | None = None) -> np.ndarray:
    """Rows usable in development: observation AND label end before (holdout start - embargo)."""
    h = holdout or FinalHoldout()
    ok = np.asarray(ts_ms, dtype="int64") < h.start_ms - h.embargo_s * 1000
    if label_end_ms is not None:
        ok &= np.asarray(label_end_ms, dtype="int64") <= h.start_ms - h.embargo_s * 1000
    return ok


def folds_manifest() -> dict[str, Any]:
    folds, hold = default_folds(), FinalHoldout()
    return {"version": "folds/1", "scheme": "expanding-window walk-forward, whole source days, no random splits",
            "source_offset_s": SOURCE_OFFSET_S, "embargo_s": EMBARGO_S, "max_label_span_s": MAX_LABEL_SPAN_S,
            "first_eligible_day": FIRST_ELIGIBLE_DAY.isoformat(), "development_last_day": DEV_LAST_DAY.isoformat(),
            "folds": [f.to_dict() for f in folds], "final_holdout": hold.to_dict(),
            "rules": {"train_before_validation": True, "purge": "train label_end_ms <= val_start - embargo; val label_end_ms <= val_end",
                      "embargo_s": EMBARGO_S, "segments_respected": "labels never cross a segment boundary (labels.py)",
                      "feed_boundary": {"boundary_utc_ms": 1788732124012, "folds_with_post_boundary_validation": ["F5"],
                                        "folds_with_post_boundary_training": []},
                      "preprocessing": "fit on the fold's training block only (research.preprocessing)",
                      "final_holdout": "closed to model/feature/label/threshold selection; no inspection of labels, features or "
                                       "outcomes inside it until the research design is frozen and an explicit authorization is given"},
            "selection_basis": "chronology and data coverage only: 8-week minimum first training block; 3-week validation blocks; "
                               "holdout = the last 3 weeks + 2 days (17 trading days, ~13% of eligible days), leaving the boundary "
                               "week 2026-09-07..11 in development so the pre/post-feed shift can be measured without touching the holdout",
            "violations": validate_plan(folds, hold)}
