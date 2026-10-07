"""Every deliberately-bad fixture MUST be caught. If one of these tests passes a bad fixture, the leakage suite is broken."""
import inspect

import numpy as np
import pytest

from fxscalp.features import ops
from fxscalp.features.leakage import (LeakageDetected, assert_prefix_invariant, check_prefix_invariance,
                                      check_target_contamination, scan_function, scan_source)
from fxscalp.features.scaler import CausalScaler, check_scaler_fit_is_causal
from tests.leakage import bad_fixtures as bad
from tests.leakage.helpers import make_compute, scenario

import datetime as dt

DF = scenario(start_utc=dt.datetime(2025, 10, 6, 6, 0, tzinfo=dt.timezone.utc), duration_s=2 * 3600 + 1800, base_rate_hz=0.8, seed=11)
# a leak is only visible where later data changes the value, so cutoffs must precede the data that matters (incl. early ones)
CUTS = [int(len(DF) * f) + 13 for f in (0.08, 0.2, 0.31, 0.55, 0.78)]


@pytest.mark.parametrize("name", sorted(bad.ALL_BAD))
def test_prefix_invariance_catches_each_bad_fixture(name):
    fn = bad.ALL_BAD[name]
    rep = check_prefix_invariance(make_compute(extra=fn), DF, CUTS, ts_col="seq", key_col="timestamp_utc_ms")
    assert not rep.ok, f"{name}: leakage NOT detected"
    bad_cols = {m["column"] for m in rep.mismatches}
    assert any(c.startswith("bad_") for c in bad_cols), (name, bad_cols)
    assert not (bad_cols - {c for c in bad_cols if c.startswith("bad_")}), "only the injected column may differ"   # the real features stay clean
    with pytest.raises(LeakageDetected):
        assert_prefix_invariant(make_compute(extra=fn), DF, CUTS[:1], ts_col="seq", key_col="timestamp_utc_ms")


def test_good_pipeline_is_not_flagged_with_the_same_cuts():
    assert check_prefix_invariance(make_compute(), DF, CUTS, ts_col="seq", key_col="timestamp_utc_ms").ok


@pytest.mark.parametrize("name,kind", [("future_bar", "forbidden_call"), ("future_session_extreme", "forbidden_call"),
                                       ("centered_window", "centered_window"), ("negative_shift", "forbidden_call"),
                                       ("forward_asof", "forward_asof"), ("backward_asof_on_start", "forbidden_call"),
                                       ("future_normalization", "forbidden_call"), ("future_volatility", "forbidden_call"),
                                       ("reversed_rolling", "reversed_slice"), ("target_contamination", "forbidden_call")])
def test_static_scanner_flags_the_pattern(name, kind):
    findings = scan_function(bad.ALL_BAD[name])
    assert any(f.kind == kind for f in findings), (name, findings)


def test_scanner_accepts_all_production_feature_groups():
    from fxscalp.features.groups import GROUPS
    for g in GROUPS:
        assert scan_source(inspect.getsource(g.fn)) == [], g.name


def test_scanner_unit_cases():
    assert scan_source("x = ops.lag(a, -3)")[0].kind == "negative_shift"
    assert scan_source("y = s.rolling(5, center=True).mean()")
    assert scan_source("z = df.shift(periods=-2)")
    assert scan_source("w = pd.merge_asof(a, b, on='t', direction='nearest')")
    assert scan_source("v = a[::-1]")[0].kind == "reversed_slice"
    assert scan_source("u = ops.rsum(a, 5) + ops.lag(a, 3)") == []


def test_lag_refuses_negative_shift_at_runtime():
    with pytest.raises(ValueError):
        ops.lag(np.arange(5.0), -1)


# ---------------- scalers: whole-dataset fitting ----------------
def test_causal_scaler_is_prefix_invariant_and_whole_dataset_scaler_is_not():
    full = make_compute()(DF)
    cols = ["ret_bps_60s", "tick_rate_15s"]
    cut = int(full["timestamp_utc_ms"].iloc[len(full) // 2])
    assert check_scaler_fit_is_causal(lambda e: CausalScaler(e), full, cols, cut)
    assert not check_scaler_fit_is_causal(lambda e: bad.WholeDatasetScaler(e), full, cols, cut)


def test_scaler_params_use_only_rows_before_fit_end():
    full = make_compute()(DF)
    cut = int(full["timestamp_utc_ms"].iloc[len(full) // 2])
    sc = CausalScaler(cut).fit(full, ["ret_bps_60s"])
    assert sc.n_fit_rows == int((full["timestamp_utc_ms"] < cut).sum()) < len(full)


# ---------------- target contamination ----------------
def test_target_contamination_screen_flags_copies_names_and_perfect_correlations():
    full = make_compute()(DF)
    y = full["ret_bps_5s"].shift(-30)
    clean = full[["tick_rate_15s", "spread_points", "range_width_bps_60s", "ret_bps_15s"]]
    assert check_target_contamination(clean, y) == []
    contaminated = clean.assign(fwd_ret=y, a1=y, a2=y * 3 + 1)
    found = check_target_contamination(contaminated, y)
    assert {f.kind for f in found} == {"suspect_name", "target_copy", "near_perfect_correlation"}
    assert {"fwd_ret", "a1", "a2"} <= {w.strip("'") for f in found for w in f.detail.split() if w.strip("'") in ("fwd_ret", "a1", "a2")}


def test_leaky_feature_used_as_label_is_caught_dynamically_too():
    rep = check_prefix_invariance(make_compute(extra=bad.bad_target_contamination), DF, CUTS, ts_col="seq", key_col="timestamp_utc_ms")
    assert not rep.ok


# ---------------- primitives ----------------
@pytest.mark.parametrize("name,fn", [("rsum", lambda x: ops.rsum(x, 7)), ("rmean", lambda x: ops.rmean(x, 7)),
                                     ("rmax", lambda x: ops.rmax(x, 7)), ("rmin", lambda x: ops.rmin(x, 7)),
                                     ("rstd", lambda x: ops.rstd(x, 7)), ("rmedian", lambda x: ops.rmedian(x, 9)),
                                     ("rrank", lambda x: ops.rrank(x, 9)), ("ewma", lambda x: ops.ewma(x, 9)),
                                     ("lag", lambda x: ops.lag(x, 4)), ("ffill", ops.ffill_asof),
                                     ("log_ret", lambda x: ops.log_ret_bps(x + 100, 5))])
def test_every_ops_primitive_is_causal(name, fn):
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 300)
    x[rng.random(300) < 0.1] = np.nan
    full = fn(x)
    for k in (10, 57, 150, 299):
        part = fn(x[:k])
        assert np.array_equal(part, full[:k], equal_nan=True), (name, k)


def test_contamination_screen_does_not_require_scipy(monkeypatch):
    """pandas' method='spearman' lazily imports scipy (undeclared dependency); the screen must not need it."""
    import builtins
    real_import = builtins.__import__

    def no_scipy(name, *a, **k):
        if name == "scipy" or name.startswith("scipy."):
            raise ModuleNotFoundError("No module named 'scipy'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    import numpy as np
    import pandas as pd
    from fxscalp.features.leakage import check_target_contamination
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    df = pd.DataFrame({"a": rng.normal(size=200), "monotone": np.exp(y)})
    out = check_target_contamination(df, y)
    assert [f.kind for f in out] == ["near_perfect_correlation"] and "monotone" in out[0].detail
