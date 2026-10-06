import datetime as dt

import numpy as np
import pytest

from fxscalp.features.pipeline import FeatureFrame, PipelineConfig, build_feature_frame
from fxscalp.features.validation import render_report_markdown, validate_feature_frame
from fxscalp.synthetic.xauusd import SynthConfig, generate_ticks, synthetic_symbol_info
from tests.unit.features_helpers import norm

UTC = dt.timezone.utc


@pytest.fixture(scope="module")
def good():
    cfg = SynthConfig(start_utc=dt.datetime(2025, 10, 6, 6, 0, tzinfo=UTC), duration_s=2 * 3600 + 900, base_rate_hz=0.8, seed=4)
    return build_feature_frame(norm(generate_ticks(cfg)), PipelineConfig(), point=0.01)


def mutated(ff, fn):
    df = ff.df.copy()
    fn(df)
    return FeatureFrame(df, ff.registry, ff.segments, ff.config)


# ---------------- validator ----------------
def test_clean_frame_passes_validation(good):
    rep = validate_feature_frame(good)
    assert rep.ok, rep.errors
    assert rep.valid_rows > 0 and rep.per_feature["tick_rate_5s"]["n_unique"] > 3
    md = render_report_markdown(rep)
    assert "Feature quality report" in md and "tick_rate_5s" in md


def test_detects_inf(good):
    r = validate_feature_frame(mutated(good, lambda d: d.__setitem__("tick_rate_5s", d["tick_rate_5s"].where(d.index != 10, np.inf))))
    assert any("inf" in e for e in r.errors)


def test_detects_nan_everywhere_post_warmup(good):
    r = validate_feature_frame(mutated(good, lambda d: d.__setitem__("ret_bps_60s", np.nan)))
    assert any("ret_bps_60s" in e and "NaN on every valid row" in e for e in r.errors)


def test_detects_constant_feature_as_warning(good):
    r = validate_feature_frame(mutated(good, lambda d: d.__setitem__("range_width_bps_60s", 1.0)))
    assert any("range_width_bps_60s" in w and "constant" in w for w in r.warnings)


def test_detects_out_of_range(good):
    r = validate_feature_frame(mutated(good, lambda d: d.__setitem__("up_tick_ratio_15s", d["up_tick_ratio_15s"] + 2.0)))
    assert any("up_tick_ratio_15s" in e and "outside expected range" in e for e in r.errors)


def test_detects_missing_extra_and_duplicate_columns(good):
    r = validate_feature_frame(mutated(good, lambda d: d.drop(columns=["tick_rate_5s"], inplace=True)))
    assert any("missing columns" in e and "tick_rate_5s" in e for e in r.errors)
    r = validate_feature_frame(mutated(good, lambda d: d.__setitem__("surprise_col", 1.0)))
    assert any("schema mismatch" in e and "surprise_col" in e for e in r.errors)
    df = good.df.copy()
    dup = FeatureFrame(__import__("pandas").concat([df, df[["tick_rate_5s"]]], axis=1), good.registry, good.segments, good.config)
    assert any("duplicate columns" in e for e in validate_feature_frame(dup).errors)


def test_detects_timestamp_misalignment_and_gaps(good):
    r = validate_feature_frame(mutated(good, lambda d: d.__setitem__("timestamp_utc_ms", d["timestamp_utc_ms"] + 137)))
    assert any("not aligned" in e for e in r.errors)
    r = validate_feature_frame(mutated(good, lambda d: d.drop(index=[50, 51], inplace=True)))
    assert any("grid steps missing" in e for e in r.errors)
    r = validate_feature_frame(mutated(good, lambda d: d.__setitem__("timestamp_utc_ms", d["timestamp_utc_ms"].where(d.index != 20, d["timestamp_utc_ms"].iloc[5]))))
    assert any("not strictly increasing" in e for e in r.errors)


def test_detects_non_numeric_feature_column(good):
    r = validate_feature_frame(mutated(good, lambda d: d.__setitem__("tick_rate_5s", "x")))
    assert any("non-numeric" in e for e in r.errors)


def test_validator_includes_static_leakage_scan(good, monkeypatch):
    import fxscalp.features.validation as v

    def leaky_fn(ctx, cfg, prior, emit):
        return (ctx.x).shift(-1)
    monkeypatch.setattr(v, "GROUPS", [type("G", (), {"name": "leaky", "fn": staticmethod(leaky_fn)})()])
    r = validate_feature_frame(good)
    assert any("leakage scan [leaky]" in e for e in r.errors)


# ---------------- synthetic generator ----------------
def test_synthetic_is_deterministic_and_seeded():
    a, b = generate_ticks(SynthConfig(seed=1, duration_s=1800)), generate_ticks(SynthConfig(seed=1, duration_s=1800))
    c = generate_ticks(SynthConfig(seed=2, duration_s=1800))
    assert a.tobytes() == b.tobytes() and a.tobytes() != c.tobytes()


def test_synthetic_scenarios_have_the_requested_properties():
    base = dict(start_utc=dt.datetime(2025, 10, 7, 8, 0, tzinfo=UTC), duration_s=7200, seed=3)
    plain = generate_ticks(SynthConfig(**base))
    assert (np.diff(plain["time_msc"]) >= 0).all() and (plain["ask"] > plain["bid"]).all() and plain["flags"][0] == 6
    out = generate_ticks(SynthConfig(**base, outages=((1000, 300),)))
    t0 = base["start_utc"].timestamp() * 1000
    assert ((out["time_msc"] >= t0 + 1000_000) & (out["time_msc"] < t0 + 1300_000)).sum() == 0                  # no ticks in the outage
    wide = generate_ticks(SynthConfig(**base, spread_widening=((3000, 600, 6.0),)))
    sp = (wide["ask"] - wide["bid"]) / 0.01
    inside = (wide["time_msc"] >= t0 + 3000_000) & (wide["time_msc"] < t0 + 3600_000)
    assert sp[inside].mean() > 3 * sp[~inside].mean()
    dup = generate_ticks(SynthConfig(**base, dup_prob=0.01))
    assert len(dup) > len(plain) * 1.005 and (np.diff(dup["time_msc"]) == 0).sum() > (np.diff(plain["time_msc"]) == 0).sum()
    ooo = generate_ticks(SynthConfig(**base, out_of_order_prob=0.01))
    assert (np.diff(ooo["time_msc"]) < 0).sum() > 10
    dropped = generate_ticks(SynthConfig(**base, drop_prob=0.3))
    assert len(dropped) < 0.8 * len(plain)
    shifted = generate_ticks(SynthConfig(**base, source_offset_s=10800))
    assert (shifted["time_msc"] - plain["time_msc"]).min() == 10800_000 and len(shifted) == len(plain)


def test_synthetic_weekend_closure_and_session_activity():
    fri = generate_ticks(SynthConfig(start_utc=dt.datetime(2025, 10, 10, 19, 0, tzinfo=UTC), duration_s=52 * 3600, base_rate_hz=0.2, seed=1))
    d = np.diff(fri["time_msc"])
    assert d.max() > 47 * 3600 * 1000                                         # Friday 21:00 UTC .. Sunday 21:00 UTC (US DST) closure
    last_fri = fri["time_msc"][np.argmax(d)]
    assert dt.datetime.fromtimestamp(last_fri / 1000, tz=UTC).hour == 20       # last tick just before 21:00 UTC (17:00 EDT)
    day = generate_ticks(SynthConfig(start_utc=dt.datetime(2025, 7, 16, 0, 0, tzinfo=UTC), duration_s=86400, base_rate_hz=0.5, seed=2))
    hour = ((day["time_msc"] // 3_600_000) % 24).astype(int)
    c = np.bincount(hour, minlength=24)
    assert c[7] > 1.5 * c[3] and c[13] > 1.5 * c[3] and c[12] > c[21]          # London open (07 UTC summer) / NY open+overlap busy vs quiet Asia/off hours


def test_synthetic_symbol_info_is_a_valid_symbolinfo():
    from fxscalp.brokers.mt5.convert import check_symbol_consistency
    si = synthetic_symbol_info()
    assert si.point == 0.01 and si.canonical == "XAU_USD" and check_symbol_consistency(si, "USD") == []


def test_pipeline_handles_dst_transition_week_bars_utc_aligned():
    """US DST starts 2024-03-10 07:00 UTC: UTC bars are unaffected, session labels shift."""
    from fxscalp.features.bars import aggregate_bars
    a = generate_ticks(SynthConfig(start_utc=dt.datetime(2024, 3, 10, 5, 0, tzinfo=UTC), duration_s=4 * 3600, base_rate_hz=1.0, seed=6, weekend_closure=False))
    for tf in (60, 300, 3600):
        b = aggregate_bars(a["time_msc"], a["bid"], a["ask"], a["volume"], tf)
        assert (np.diff(b["start_ms"].to_numpy()) == tf * 1000).all()
    x = lambda d: build_feature_frame(norm(generate_ticks(SynthConfig(start_utc=d, duration_s=1800, base_rate_hz=1.0, seed=1, weekend_closure=False))), PipelineConfig(), point=0.01).df
    est = x(dt.datetime(2025, 3, 3, 12, 15, tzinfo=UTC))             # both UK and US on winter time: 12:15 UTC = London only
    edt = x(dt.datetime(2025, 3, 19, 12, 15, tzinfo=UTC))            # US already on summer time, UK not: 12:15 UTC = overlap
    assert est["session"].iloc[100] == "LONDON" and edt["session"].iloc[100] == "LONDON_NEW_YORK_OVERLAP"
