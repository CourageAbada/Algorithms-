"""Feature-quality validation: schema, timestamps, NaN/inf, constants, ranges, static leakage scan -> report."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fxscalp.features.groups import GROUPS
from fxscalp.features.leakage import scan_source
from fxscalp.features.pipeline import FeatureFrame

META_COLUMNS = ("timestamp_utc_ms", "segment_id", "row_valid", "session", "spread_regime", "volatility_regime")


@dataclass
class FeatureQualityReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows: int = 0
    valid_rows: int = 0
    per_feature: dict[str, dict[str, Any]] = field(default_factory=dict)
    feature_set_version: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "feature_set_version": self.feature_set_version, "rows": self.rows, "valid_rows": self.valid_rows,
                "errors": self.errors, "warnings": self.warnings, "per_feature": self.per_feature}


def validate_feature_frame(ff: FeatureFrame, *, max_post_warmup_nan: float = 0.5, range_tol: float = 1e-6,
                           scan_code: bool = True) -> FeatureQualityReport:
    df, reg = ff.df, ff.registry
    rep = FeatureQualityReport(rows=len(df), feature_set_version=reg.version)
    if df.empty:
        rep.errors.append("empty feature frame")
        return rep
    # -- schema
    if df.columns.duplicated().any():
        rep.errors.append(f"duplicate columns: {sorted(set(df.columns[df.columns.duplicated()]))}")
        df = df.loc[:, ~df.columns.duplicated()]              # report, then continue on the first occurrence of each name
    expected = list(META_COLUMNS) + reg.names
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if missing:
        rep.errors.append(f"missing columns: {missing}")
    if extra:
        rep.errors.append(f"schema mismatch: unexpected columns {extra}")
    non_numeric = [c for c in reg.names if c in df.columns and df[c].dtype.kind not in "fiu"]
    if non_numeric:
        rep.errors.append(f"non-numeric feature columns: {non_numeric}")
    # -- timestamps
    ts = df["timestamp_utc_ms"].to_numpy()
    step = ff.config.grid_s * 1000
    if (ts % step).any():
        rep.errors.append("timestamps not aligned to the grid")
    for sid, g in df.groupby("segment_id"):
        d = np.diff(g["timestamp_utc_ms"].to_numpy())
        if len(d) and (d <= 0).any():
            rep.errors.append(f"segment {sid}: timestamps not strictly increasing")
        elif len(d) and (d != step).any():
            rep.errors.append(f"segment {sid}: {int((d != step).sum())} grid steps missing (timestamp misalignment)")
    if len(ts) > 1 and len(np.unique(ts)) != len(ts):
        rep.errors.append("duplicate timestamps across segments")
    valid = df["row_valid"].to_numpy() == 1
    rep.valid_rows = int(valid.sum())
    # -- per-feature
    for spec in reg.enabled:
        c = spec.name
        if c not in df.columns or df[c].dtype.kind not in "fiu":
            continue
        x = df[c].to_numpy(dtype="float64")
        xv = x[valid]
        st: dict[str, Any] = {"nan_frac_all": float(np.isnan(x).mean()),
                              "nan_frac_valid": float(np.isnan(xv).mean()) if len(xv) else None}
        n_inf = int(np.isinf(x).sum())
        if n_inf:
            rep.errors.append(f"{c}: {n_inf} inf values")
        fin = xv[np.isfinite(xv)]
        if len(fin):
            st.update(min=float(fin.min()), max=float(fin.max()), mean=float(fin.mean()), std=float(fin.std()),
                      n_unique=int(len(np.unique(fin))))
            if st["n_unique"] <= 1:
                rep.warnings.append(f"{c}: constant over valid rows")
            rng = spec.expected_range
            if rng:
                lo, hi = rng
                if (lo is not None and fin.min() < lo - range_tol) or (hi is not None and fin.max() > hi + range_tol):
                    rep.errors.append(f"{c}: values outside expected range {rng}: min={fin.min():.6g} max={fin.max():.6g}")
        if len(xv):
            frac = st["nan_frac_valid"]
            structural = any(t in spec.nan_policy for t in ("unknown", "outside", "never NaN"))
            if frac == 1.0 and not structural:
                rep.errors.append(f"{c}: NaN on every valid row")
            elif frac is not None and frac > max_post_warmup_nan and not structural:
                rep.warnings.append(f"{c}: {frac:.0%} NaN after warm-up")
        rep.per_feature[c] = st
    # -- static leakage scan of the feature code itself
    if scan_code:
        for g in GROUPS:
            for f in scan_source(inspect.getsource(g.fn)):
                rep.errors.append(f"leakage scan [{g.name}] line {f.line}: {f.kind}: {f.detail}")
    return rep


def render_report_markdown(rep: FeatureQualityReport, title: str = "Feature quality report") -> str:
    L = [f"# {title}", "", f"- Feature set version: `{rep.feature_set_version}`", f"- Rows: {rep.rows:,} (valid after warm-up: {rep.valid_rows:,})",
         f"- Status: **{'OK' if rep.ok else 'FAILED'}** ({len(rep.errors)} errors, {len(rep.warnings)} warnings)", ""]
    if rep.errors:
        L += ["## Errors", ""] + [f"- {e}" for e in rep.errors] + [""]
    if rep.warnings:
        L += ["## Warnings", ""] + [f"- {w}" for w in rep.warnings] + [""]
    L += ["## Per-feature statistics (valid rows)", "", "| feature | NaN% (valid) | min | max | mean | unique |", "|---|---|---|---|---|---|"]
    for c, s in rep.per_feature.items():
        f = lambda k: ("" if k not in s else f"{s[k]:.4g}")  # noqa: E731
        nanv = "" if s.get("nan_frac_valid") is None else f"{100 * s['nan_frac_valid']:.1f}"
        L.append(f"| {c} | {nanv} | {f('min')} | {f('max')} | {f('mean')} | {s.get('n_unique', '')} |")
    return "\n".join(L) + "\n"
