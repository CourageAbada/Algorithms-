"""Anti-look-ahead tooling.

Two independent layers:
 1. DYNAMIC (authoritative): ``check_prefix_invariance`` recomputes features on a truncated tick stream and requires every
    row that exists in both runs to be identical. Any dependence on later data changes earlier rows and is caught,
    whatever its mechanism (shift(-1), centred windows, in-progress bars, whole-data statistics, target use...).
 2. STATIC (fast, targeted): ``scan_source`` rejects feature code that contains the usual look-ahead constructs.

Only documented exceptions may differ (default: none). Known non-causal helpers that must never feed a feature:
``quality.assess`` EXTREME_SPREAD (uses the batch median) is a tag only and is not used by the pipeline.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


class LeakageDetected(AssertionError):
    pass


# ---------------------------------------------------------------- dynamic: prefix invariance
@dataclass
class PrefixReport:
    ok: bool = True
    cutoffs_checked: int = 0
    rows_compared: int = 0
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    exceptions_used: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        if self.ok:
            return f"prefix-invariant: {self.rows_compared:,} rows over {self.cutoffs_checked} cutoffs"
        m = self.mismatches[0]
        return (f"LEAKAGE: {len(self.mismatches)} mismatches, first: column={m.get('column')} at "
                f"t={m.get('first_timestamp')} ({m.get('kind')}, {m.get('n_rows')} rows)")


def _equal(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> np.ndarray:
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        a = a.astype("float64")
        b = b.astype("float64")
        if rtol == 0 and atol == 0:
            return (a == b) | (np.isnan(a) & np.isnan(b))
        return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    return a == b


def check_prefix_invariance(compute: Callable[[pd.DataFrame], pd.DataFrame], ticks: pd.DataFrame, cutoffs_ms: Iterable[int],
                            *, ts_col: str = "timestamp_utc_ms", key_col: str = "timestamp_utc_ms",
                            exceptions: dict[str, str] | None = None, rtol: float = 0.0, atol: float = 0.0,
                            full: pd.DataFrame | None = None) -> PrefixReport:
    """For each cutoff C: features(ticks[ts < C]) must equal features(ticks) on every row both runs contain, and the
    truncated run's rows must be exactly the full run's rows up to its last row (same row set)."""
    exceptions = exceptions or {}
    rep = PrefixReport(exceptions_used=dict(exceptions))
    full = compute(ticks) if full is None else full
    fk = full.set_index(key_col)
    for c in cutoffs_ms:
        part = compute(ticks[ticks[ts_col] < c])
        rep.cutoffs_checked += 1
        if part.empty:
            continue
        pk = part.set_index(key_col)
        last = pk.index.max()
        expected_rows = fk.index[fk.index <= last]
        if not np.array_equal(pk.index.to_numpy(), expected_rows.to_numpy()):
            rep.ok = False
            rep.mismatches.append({"cutoff": int(c), "column": key_col, "kind": "row_set", "n_rows": int(abs(len(pk) - len(expected_rows))),
                                   "first_timestamp": int(last)})
            continue
        fsub = fk.loc[pk.index]
        for col in pk.columns:
            if col in exceptions:
                continue
            if col not in fsub.columns:
                rep.ok = False
                rep.mismatches.append({"cutoff": int(c), "column": col, "kind": "missing_in_full", "n_rows": len(pk),
                                       "first_timestamp": int(pk.index[0])})
                continue
            eq = _equal(pk[col].to_numpy(), fsub[col].to_numpy(), rtol, atol)
            rep.rows_compared += int(len(eq))
            if not eq.all():
                rep.ok = False
                bad = np.flatnonzero(~eq)
                rep.mismatches.append({"cutoff": int(c), "column": col, "kind": "value", "n_rows": int(len(bad)),
                                       "first_timestamp": int(pk.index[bad[0]])})
    return rep


def assert_prefix_invariant(*a: Any, **k: Any) -> PrefixReport:
    rep = check_prefix_invariance(*a, **k)
    if not rep.ok:
        raise LeakageDetected(rep.summary())
    return rep


# ---------------------------------------------------------------- static scan
FORBIDDEN_CALLS = {"mean", "std", "var", "median", "quantile", "max", "min", "sum", "cumsum", "cumprod", "cummax", "cummin",
                   "percentile", "nanmean", "nanstd", "nanmedian", "nanmax", "nanmin", "nansum", "argmax", "argmin", "sort",
                   "argsort", "rank", "corrcoef", "average", "diff", "gradient", "interp", "flip", "fliplr", "flipud",
                   "expanding", "rolling", "ewm", "transform", "apply", "agg", "aggregate", "resample", "groupby",
                   "merge_asof", "reversed", "bfill", "backfill", "shift", "pct_change", "fillna", "interpolate",
                   "searchsorted", "cumcount", "first", "last"}


@dataclass(frozen=True)
class Finding:
    kind: str
    line: int
    detail: str


def scan_source(source: str) -> list[Finding]:
    """Reject look-ahead constructs and direct statistics in feature code (use ``ops`` primitives instead)."""
    tree = ast.parse(textwrap.dedent(source))
    out: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else (node.func.id if isinstance(node.func, ast.Name) else "")
            if name in FORBIDDEN_CALLS:
                out.append(Finding("forbidden_call", node.lineno, f"{name}() is not allowed in feature code; use ops.* primitives"))
            if name in ("lag",) and node.args and len(node.args) >= 2:
                a = node.args[1]
                if isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub):
                    out.append(Finding("negative_shift", node.lineno, "negative lag reads the future"))
            for kw in node.keywords:
                if kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value:
                    out.append(Finding("centered_window", node.lineno, "center=True uses future rows"))
                if kw.arg in ("periods",) and isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.op, ast.USub):
                    out.append(Finding("negative_shift", node.lineno, "negative periods"))
                if kw.arg == "direction" and isinstance(kw.value, ast.Constant) and kw.value.value in ("forward", "nearest"):
                    out.append(Finding("forward_asof", node.lineno, f"direction={kw.value.value!r}"))
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            st = node.slice.step
            if isinstance(st, ast.UnaryOp) and isinstance(st.op, ast.USub):
                out.append(Finding("reversed_slice", node.lineno, "negative-step slice reverses time"))
    return out


def scan_function(fn: Callable[..., Any]) -> list[Finding]:
    return scan_source(inspect.getsource(fn))


# ---------------------------------------------------------------- target contamination
SUSPECT_NAMES = ("target", "label", "fwd_", "forward", "future", "next_", "y_true")


def check_target_contamination(features: pd.DataFrame, target: pd.Series | np.ndarray, *, corr_threshold: float = 0.98,
                               ) -> list[Finding]:
    """Static contamination screen: suspicious names, exact copies of the target, near-perfect rank correlation.
    (Features built from future data are caught dynamically by the prefix-invariance test.)"""
    out: list[Finding] = []
    y = pd.Series(np.asarray(target, dtype="float64"), index=features.index)
    for c in features.columns:
        if any(s in str(c).lower() for s in SUSPECT_NAMES):
            out.append(Finding("suspect_name", 0, f"column {c!r} looks like a target/forward-looking column"))
            continue
        x = features[c]
        if x.dtype.kind not in "fiu":
            continue
        valid = x.notna() & y.notna()
        if valid.sum() < 30:
            continue
        if np.array_equal(x[valid].to_numpy(), y[valid].to_numpy()):
            out.append(Finding("target_copy", 0, f"column {c!r} equals the target"))
            continue
        if x[valid].nunique() < 3:
            continue
        # Spearman = Pearson on average ranks. Computed explicitly because pandas' method="spearman" lazily imports
        # scipy, which is not a declared dependency of this project.
        rho = x[valid].rank(method="average").corr(y[valid].rank(method="average"))
        if rho == rho and abs(rho) >= corr_threshold:
            out.append(Finding("near_perfect_correlation", 0, f"column {c!r} has |spearman|={abs(rho):.4f} with the target"))
    return out
