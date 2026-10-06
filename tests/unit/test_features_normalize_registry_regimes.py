import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from fxscalp.features import registry as regmod
from fxscalp.features.groups import GROUPS
from fxscalp.features.normalize import from_raw_array, normalize_ticks, split_segments
from fxscalp.features.pipeline import PipelineConfig, build_feature_frame
from fxscalp.features.registry import build_registry, group_fingerprint, registry_yaml
from fxscalp.features.spec import FeatureConfig
from fxscalp.regimes import spread_regime as sr
from fxscalp.regimes import volatility_regime as vr
from fxscalp.regimes.config import RegimeConfig, SpreadRegimeConfig, VolatilityRegimeConfig
from tests.unit.features_helpers import ms, norm, raw, steady

ROOT = Path(__file__).resolve().parents[2]


# ---------------- normalisation ----------------
def test_normalisation_derived_fields_and_units():
    a = raw([1000, 2000], [2000.0, 2001.0], spread=0.30)
    df, rep = from_raw_array(a, point=0.01)
    assert df["mid"].tolist() == pytest.approx([2000.0, 2001.0]) and df["spread"].tolist() == pytest.approx([0.30, 0.30])
    assert df["spread_points"].tolist() == pytest.approx([30.0, 30.0])
    assert df["relative_spread"].iloc[0] == pytest.approx(0.30 / 2000.0)
    assert rep.point_known and rep.n_input == 2 and list(df.columns)[:3] == ["seq", "timestamp_utc_ms", "bid"]


def test_unknown_point_is_reported_not_guessed():
    df, rep = from_raw_array(raw([1000, 2000], [2000.0, 2001.0]), point=None)
    assert df["spread_points"].isna().all() and not rep.point_known
    df0, rep0 = from_raw_array(raw([1000], [2000.0]), point=0.0)
    assert df0["spread_points"].isna().all() and not rep0.point_known


def test_quarantined_ticks_are_excluded_and_counted_never_silently():
    a = raw([1000, 2000, 3000, 4000], [2000, 2001, 2002, 2003.0])
    a["ask"][1] = a["bid"][1] - 1          # crossed
    a["bid"][2] = 0.0                      # missing bid
    df, rep = from_raw_array(a, point=0.01)
    assert len(df) == 2 and rep.n_quarantined_excluded == 2 and rep.n_input == 4
    keep, rep2 = from_raw_array(a, point=0.01, exclude_quarantined=False)
    assert len(keep) == 4 and rep2.n_quarantined_excluded == 0


def test_out_of_order_ticks_without_quality_are_sorted_stably_and_counted():
    cols = dict(seq=np.arange(4), timestamp_utc_ms=np.array([1000, 3000, 2000, 4000]), bid=np.ones(4), ask=np.ones(4) + 0.1,
                last=np.zeros(4), volume=np.zeros(4, "uint64"), volume_real=np.zeros(4), flags=np.full(4, 6, "uint32"))
    df, rep = normalize_ticks(cols, point=0.01)
    assert df["timestamp_utc_ms"].tolist() == [1000, 2000, 3000, 4000] and rep.n_reordered == 1 and df["seq"].tolist() == [0, 2, 1, 3]


def test_duplicate_timestamps_are_counted():
    df, rep = from_raw_array(raw([1000, 1000, 1000, 2000], [1, 2, 3, 4.0]), point=0.01)
    assert rep.n_same_timestamp == 2 and df["seq"].tolist() == [0, 1, 2, 3]


def test_segments_split_on_long_gaps_only():
    ts = np.array([0, 1000, 2000, 2000 + 1801_000, 2000 + 1802_000])
    assert split_segments(ts, 1800) == [(0, 3), (3, 5)]
    assert split_segments(ts, 3600) == [(0, 5)] and split_segments(np.array([]), 10) == []


# ---------------- registry / versioning ----------------
def test_registry_is_complete_unique_and_matches_pipeline_columns():
    reg = build_registry()
    names = [s.name for s in reg.specs]
    assert len(names) == len(set(names)) >= 100
    for s in reg.specs:
        assert s.description and s.units and s.nan_policy and s.leakage_risk and s.reason and s.source and s.timeframe
        assert s.lookback_s >= 0 and s.version >= 1 and s.group in {g.name for g in GROUPS}
    a = steady(ms(2025, 10, 6, 9, 0), 300)
    ff = build_feature_frame(norm(a), PipelineConfig(), point=0.01)
    assert [c for c in ff.df.columns if c in names] == reg.names                 # same names, same order, nothing extra
    assert set(ff.df.columns) - set(names) == {"timestamp_utc_ms", "segment_id", "row_valid", "session", "spread_regime", "volatility_regime"}


def test_feature_version_is_stable_and_changes_with_config_or_definition(monkeypatch):
    v0 = build_registry().version
    assert build_registry().version == v0
    assert build_registry(dataclasses.replace(FeatureConfig(), micro_windows_s=(5, 15, 30))).version != v0
    assert build_registry(dataclasses.replace(FeatureConfig(), atr_period=10)).version != v0
    assert build_registry(None, RegimeConfig(spread=SpreadRegimeConfig(p_extreme=0.95))).version != v0
    # disabling a feature changes the version (and marks it disabled)
    r = build_registry(dataclasses.replace(FeatureConfig(), disabled_features=("tick_rate_5s",)))
    assert r.version != v0 and "tick_rate_5s" not in r.names and not r.spec("tick_rate_5s").enabled
    # editing a group definition changes its fingerprint and therefore the version
    g = next(x for x in GROUPS if x.name == "micro")
    f0 = group_fingerprint(g)
    src = regmod._src
    monkeypatch.setattr(regmod, "_src", lambda o: src(o) + ("\n# edited" if o is g.fn else ""))
    assert group_fingerprint(g) != f0


def test_source_normalisation_makes_the_version_platform_independent():
    assert regmod._norm("a = 1  \r\nb = 2\t\r\n") == regmod._norm("a = 1\nb = 2\n")


def test_committed_registry_snapshot_is_current():
    path = ROOT / "configs" / "features" / "xauusd_core.registry.yaml"
    assert path.exists(), "run: python -m scripts.export_feature_registry"
    assert path.read_text() == registry_yaml(build_registry()), \
        "feature definitions changed: regenerate with `python -m scripts.export_feature_registry` and review the diff"
    doc = yaml.safe_load(path.read_text())
    assert doc["feature_set_version"] == build_registry().version and len(doc["features"]) == len(build_registry().specs)


def test_emitted_specs_declare_leakage_assessment_and_nan_policy():
    for s in build_registry().specs:
        assert s.leakage_risk.split(":")[0] in ("LOW", "MEDIUM", "HIGH")
        assert "NaN" in s.nan_policy or s.nan_policy in ("never NaN",) or "NaN" in s.nan_policy or "0 (UNKNOWN)" in s.nan_policy


# ---------------- regimes ----------------
def test_spread_regime_uncalibrated_is_relative_and_requires_both_conditions():
    cols = {"spread_points": np.array([30, 80, 300, np.nan, 30.0]), "spread_percentile_1800s": np.array([0.5, 0.95, 0.995, 0.9, np.nan]),
            "spread_median_points_300s": np.array([30, 30, 30, 30, 30.0])}
    out = sr.classify_spread(cols)
    assert out.tolist() == [sr.NORMAL, sr.ELEVATED, sr.EXTREME, sr.UNKNOWN, sr.UNKNOWN]
    # high percentile but tiny ratio (persistently stable spread) is NOT flagged
    cols2 = {"spread_points": np.array([31.0]), "spread_percentile_1800s": np.array([0.999]), "spread_median_points_300s": np.array([30.0])}
    assert sr.classify_spread(cols2).tolist() == [sr.NORMAL]
    assert SpreadRegimeConfig().calibration_status == "UNCALIBRATED_PLACEHOLDER_DEFAULTS"


def test_spread_regime_calibrated_by_session():
    rng = np.random.default_rng(0)
    sess = np.repeat([1.0, 2.0], 2000)
    ref = np.concatenate([rng.normal(30, 2, 2000), rng.normal(60, 4, 2000)])
    cal = sr.calibrate_reference_quantiles(ref, sess, "tickraw-REFERENCE", (0.95, 0.995))
    assert set(cal.by_session_code) == {1, 2} and cal.by_session_code[2][0] > cal.by_session_code[1][0]
    cfg = SpreadRegimeConfig(calibration=cal, calibration_status="CALIBRATED:tickraw-REFERENCE")
    out = sr.classify_spread({"spread_points": np.array([30.0, 45.0, 200.0, 67.0, 31.0]), "session_code": np.array([1, 1, 1, 2, 4.0])}, cfg)
    assert out.tolist() == [sr.NORMAL, sr.EXTREME, sr.EXTREME, sr.ELEVATED, sr.UNKNOWN]      # session 4 has no reference -> UNKNOWN
    assert sr.calibrate_reference_quantiles(ref[:10], sess[:10], "x").by_session_code == {}   # too few observations


def test_volatility_regime_states_and_calibration():
    cols = {"realized_vol_bps_60s": np.array([1, 5, 9, 12, np.nan, 2.0]), "vol_percentile_60s_1h": np.array([0.1, 0.5, 0.9, 0.99, 0.5, np.nan]),
            "vol_zscore_60s_1h": np.array([-1, 0, 1, 2.5, 0, 0.0])}
    assert vr.classify_volatility(cols).tolist() == [vr.LOW, vr.NORMAL, vr.HIGH, vr.EXTREME, vr.UNKNOWN, vr.UNKNOWN]
    v = np.concatenate([np.linspace(1, 10, 3000)] * 2)
    s = np.repeat([1.0, 3.0], 3000)
    cal = vr.calibrate_reference_quantiles(v, s, "tickraw-REF")
    cfg = VolatilityRegimeConfig(calibration=cal)
    out = vr.classify_volatility({"realized_vol_bps_60s": np.array([1.2, 5.0, 8.5, 9.9]), "session_code": np.full(4, 1.0)}, cfg)
    assert out.tolist() == [vr.LOW, vr.NORMAL, vr.HIGH, vr.EXTREME]


def test_regime_config_is_part_of_the_manifest_and_flags_uncalibrated():
    d = RegimeConfig().to_dict()
    assert d["spread"]["calibration_status"].startswith("UNCALIBRATED") and d["volatility"]["calibration"] is None
    json.dumps(d)
