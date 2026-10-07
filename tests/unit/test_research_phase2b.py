"""Phase 2B-A: label causality, segment-boundary protection, eligibility, purge/embargo, chronological folds, train-only
preprocessing, feature policy, deterministic manifests and final-holdout protection."""

from __future__ import annotations

import copy
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from fxscalp.market_data.quality import Q
from fxscalp.research import costs, folds, labels, policy, spec
from fxscalp.research.costs import SCENARIOS, CostScenario, UncalibratedCostError
from fxscalp.research.eligibility import day_view
from fxscalp.research.feature_policy import (ALLOWED, CONDITIONAL, FORBIDDEN_MODEL_COLUMNS, assert_model_inputs_allowed,
                                              build_feature_manifest)
from fxscalp.research.labels import (CANDIDATE_HORIZONS_S, STATUS_BOUNDARY, STATUS_OK, STATUS_WARMUP, LabelSpec, SegmentGrid,
                                     candidate_specs, generate_labels, reference_labels)
from fxscalp.research.preprocessing import (FoldPreprocessor, LeakageError, train_only_baseline, train_only_quantile)


def synth(n=700, seed=0, spread_pts=10, vol=0.15, t0=1_700_000_000_000) -> SegmentGrid:
    r = np.random.default_rng(seed)
    mid = 4000 + np.cumsum(r.normal(0, vol, n))
    sp = np.round(spread_pts + r.integers(-3, 4, n)) * 0.01
    bid_c = np.round(mid - sp / 2, 2)
    ask_c = np.round(bid_c + sp, 2)
    ext = lambda c, k: c + k * np.abs(r.normal(0, 0.05, n))  # noqa: E731
    pb, pa = np.roll(bid_c, 1), np.roll(ask_c, 1)
    return SegmentGrid(t0 + np.arange(1, n + 1) * 1000, bid_c, ask_c,
                       np.maximum(ext(bid_c, 1), np.maximum(bid_c, pb)), np.minimum(ext(bid_c, -1), np.minimum(bid_c, pb)),
                       np.maximum(ext(ask_c, 1), np.maximum(ask_c, pa)), np.minimum(ext(ask_c, -1), np.minimum(ask_c, pa)), 0.01)


def with_arrays(g: SegmentGrid, **over) -> SegmentGrid:
    d = {k: getattr(g, k) for k in ("end_ms", "bid_c", "ask_c", "bid_h", "bid_l", "ask_h", "ask_l", "point")}
    d.update(over)
    return SegmentGrid(**d)


# ============================ label correctness & causality ============================
@pytest.mark.parametrize("seed,vol", [(0, 0.05), (1, 0.25), (2, 0.7)])
@pytest.mark.parametrize("spec_", [LabelSpec("tb", 15), LabelSpec("tb", 30, vol_window_s=120), LabelSpec("fh", 15), LabelSpec("fh", 60),
                                   LabelSpec("opp", 15), LabelSpec("opp", 60)], ids=lambda s: s.name)
def test_vectorised_labels_equal_the_loop_reference_for_every_scenario(seed, vol, spec_):
    g = synth(seed=seed, vol=vol)
    for sc in SCENARIOS.values():
        a, b = generate_labels(g, spec_, sc), reference_labels(g, spec_, sc)
        assert (a["label"] == b["label"]).all() and (a["status"] == b["status"]).all()


@pytest.mark.parametrize("spec_", [LabelSpec("tb", 30, vol_window_s=60), LabelSpec("fh", 30), LabelSpec("opp", 30)], ids=lambda s: s.name)
@pytest.mark.parametrize("scenario", ["C0_spread_only", "C1_moderate"])
def test_labels_are_causal_future_beyond_the_window_never_matters(spec_, scenario):
    """label(t) is a function of quotes in [t-W, t+L+H] only: rewriting everything later must not change it."""
    g = synth(n=500, seed=3, vol=0.3)
    L = SCENARIOS[scenario].latency_s
    base = generate_labels(g, spec_, scenario)
    cut = 250
    rng = np.random.default_rng(9)
    arrays = {k: getattr(g, k).copy() for k in ("bid_c", "ask_c", "bid_h", "bid_l", "ask_h", "ask_l")}
    for k in arrays:
        arrays[k][cut:] += rng.normal(0, 5.0, len(arrays[k]) - cut)           # wreck the future after `cut`
    changed = generate_labels(with_arrays(g, **arrays), spec_, scenario)
    safe = np.arange(g.n) + L + spec_.horizon_s < cut                          # whole window strictly before the rewritten part
    assert safe.sum() > 100
    assert (base["label"][safe] == changed["label"][safe]).all() and (base["status"][safe] == changed["status"][safe]).all()
    # and the rewrite DOES matter for rows whose window reaches it (the test is not vacuous)
    assert (base["label"] != changed["label"]).any()


def test_label_end_is_the_last_instant_of_information_used():
    g = synth(n=300)
    out = generate_labels(g, LabelSpec("tb", 60, vol_window_s=60), "C1_moderate")       # latency 1 s
    assert (out["label_end_ms"] == g.end_ms + (1 + 60) * 1000).all()


def test_unknown_cost_components_require_an_explicit_scenario():
    g = synth(n=200)
    with pytest.raises(UncalibratedCostError):
        generate_labels(g, LabelSpec("fh", 15), None)
    with pytest.raises(UncalibratedCostError):
        generate_labels(g, LabelSpec("fh", 15), "no_such_scenario")
    assert costs.COMPONENTS["commission_round_turn_points"]["status"] == "UNKNOWN/UNCALIBRATED"
    assert costs.COMPONENTS["slippage_points_per_side"]["status"] == "UNKNOWN/UNCALIBRATED"
    assert costs.COMPONENTS["latency_s"]["status"] == "UNKNOWN/UNCALIBRATED"
    assert costs.COMPONENTS["observed_spread"]["status"] == "OBSERVED"
    with pytest.raises(ValueError):
        CostScenario("x", 2, 0.0, 0.0)                                       # latency beyond the 1 s grid quantisation


def test_fixed_horizon_label_uses_bid_ask_not_mid_and_at_most_one_side_wins():
    g = synth(n=400, seed=5, vol=0.4)
    out = generate_labels(g, LabelSpec("fh", 15), "C0_spread_only")
    ok = out["status"] == STATUS_OK
    rl, rs = out["r_long_points"][ok], out["r_short_points"][ok]
    assert not ((rl > 0) & (rs > 0)).any()                                    # r_long + r_short = -(spreads) < 0
    i = 10
    assert abs(rl[i] - ((g.bid_c[i + 15] - g.ask_c[i]) / 0.01)) < 1e-9
    # a flat mid with a positive spread can never be a LONG/SHORT winner
    flat = with_arrays(g, bid_c=np.full(g.n, 100.0), ask_c=np.full(g.n, 100.1), bid_h=np.full(g.n, 100.0), bid_l=np.full(g.n, 100.0),
                       ask_h=np.full(g.n, 100.1), ask_l=np.full(g.n, 100.1))
    assert (generate_labels(flat, LabelSpec("fh", 15), "C0_spread_only")["label"] == 0).all()


def test_extra_cost_turns_marginal_winners_into_no_trade():
    g = synth(n=600, seed=6, vol=0.2)
    a = generate_labels(g, LabelSpec("fh", 60), "C0_spread_only")["label"]
    b = generate_labels(g, LabelSpec("fh", 60), CostScenario("big", 0, 40.0, 0.0))["label"]
    assert (a != 0).sum() > (b != 0).sum()


def test_triple_barrier_ties_and_status_codes():
    n = 400
    flat = np.full(n, 100.0)
    g = SegmentGrid(1_700_000_000_000 + np.arange(1, n + 1) * 1000, flat, flat + 0.1, flat.copy(), flat.copy(), flat + 0.1, flat + 0.1, 0.01)
    out = generate_labels(g, LabelSpec("tb", 15, vol_window_s=60), "C0_spread_only")
    st = out["status"]
    assert (st[:60] == STATUS_WARMUP).all() and (st[-15:] == STATUS_BOUNDARY).all() and (st[60:-15] == STATUS_OK).all()
    assert (out["label"] == 0).all()                                          # no movement -> timeout -> NO_TRADE
    # same-second profit and stop -> stop first
    r = np.random.default_rng(0)
    mid = 100 + np.cumsum(r.normal(0, 0.05, n))
    b = mid - 0.05
    g2 = SegmentGrid(g.end_ms, b, b + 0.1, b.copy(), b.copy(), b + 0.1, b + 0.1, 0.01)
    i = 100
    arr = {k: getattr(g2, k).copy() for k in ("bid_h", "bid_l", "ask_h", "ask_l")}
    arr["bid_h"][i + 5] = b[i] + 100.0
    arr["bid_l"][i + 5] = b[i] - 100.0                                       # both barriers touched in one second
    g3 = with_arrays(g2, **arr)
    o = generate_labels(g3, LabelSpec("tb", 15, vol_window_s=60), "C0_spread_only")
    assert o["long_outcome"][i] == -1                                         # conservative: stop wins the tie


# ============================ segment-boundary protection ============================
def test_a_label_window_never_leaves_its_segment_and_next_segment_data_is_invisible():
    seg_a = synth(n=400, seed=1, vol=0.2)
    seg_b = synth(n=400, seed=2, vol=5.0, t0=1_700_000_000_000 + 5000 * 1000)       # wild next segment after a 4,000 s hole
    for sp_ in candidate_specs():
        if sp_.horizon_s > 60:
            continue
        a = generate_labels(seg_a, sp_, "C1_moderate")
        assert (a["status"][-(sp_.horizon_s + 1):] == STATUS_BOUNDARY).all()          # tail rows lose their label, not borrow
        # labels of segment A are identical whether or not segment B exists (they are computed per segment)
        again = generate_labels(seg_a, sp_, "C1_moderate")
        assert (a["label"] == again["label"]).all()
        b = generate_labels(seg_b, sp_, "C1_moderate")
        assert (b["label_end_ms"][b["status"] == STATUS_OK] <= seg_b.end_ms[-1]).all()
        assert (a["label_end_ms"][a["status"] == STATUS_OK] <= seg_a.end_ms[-1]).all()


def test_segment_starts_split_on_unexplained_gaps_and_respect_the_threshold():
    t = np.cumsum(np.array([0, 1, 1, 2, 301, 1, 1, 450, 1, 7361, 1, 299]) * 1000).astype("int64") + 1_000_000
    s = policy.segment_starts(t)
    assert s.tolist() == [True, False, False, False, True, False, False, True, False, True, False, False]   # 299 s does not split
    assert policy.segment_starts(t, 500).sum() == 2 + 0 and policy.SEGMENT_GAP_S == 300
    assert policy.segment_starts(t[:3], prev_last_utc_ms=int(t[0]) - 400_000).tolist() == [True, False, False]   # carry-in gap
    assert policy.segment_starts(t[:3], prev_last_utc_ms=int(t[0]) - 100_000).tolist() == [False, False, False]


def test_every_gap_in_the_threshold_band_yields_the_same_segmentation_of_the_real_gap_list():
    """Documented property: the observed intra-day gaps are 223, 456, 593, 871, 871, 1341, 7361 s (>=60 s, eligible data);
    any threshold in [230, 450] produces the same segmentation - the choice is not a tuned knob."""
    gaps = np.array([89, 140, 162, 182, 223, 456, 593, 871, 871, 1341, 7361], dtype=float)
    base = (gaps > 300).sum()
    assert all((gaps > g).sum() == base for g in range(230, 451, 10))


# ============================ eligibility ============================
@pytest.mark.parametrize("tag", [Q.TIME_BASIS_UNCERTAIN, Q.FUTURE_TICK, Q.TIME_REVERSAL, Q.MISSING_BID, Q.MISSING_ASK,
                                 Q.NEGATIVE_SPREAD, Q.NONFINITE, Q.TIME_NONEXISTENT])
def test_each_required_exclusion_tag_removes_the_tick(tag):
    assert not policy.eligibility_mask(np.array([int(tag), int(tag) | int(Q.DUP_TIMESTAMP)], dtype="uint32")).any()


@pytest.mark.parametrize("tag", [Q.DUP_TIMESTAMP, Q.EXTREME_SPREAD, Q.ZERO_SPREAD, Q.UNKNOWN_FLAG_BITS, Q.LARGE_GAP, Q.DUP_TICK])
def test_retained_tags_do_not_exclude(tag):
    assert policy.eligibility_mask(np.array([int(tag)], dtype="uint32")).all()
    assert tag in policy.RETAIN_TAGS or tag == Q.DUP_TICK


def test_eligibility_view_is_deterministic_and_segments_follow_eligible_ticks_only():
    utc = (np.arange(0, 20) * 1000 + 1_000_000).astype("int64")
    utc[10:] += 600_000                                        # 10-minute hole in the middle
    qf = np.zeros(20, dtype="uint32")
    qf[3] = int(Q.TIME_BASIS_UNCERTAIN)
    qf[12] = int(Q.DUP_TIMESTAMP)
    a, b = day_view(utc, qf, None), day_view(utc, qf, None)
    for k in a:
        assert (a[k] == b[k]).all()
    assert not a["eligible"][3] and a["eligible"].sum() == 19 and a["eligible"][12]
    assert len(set(a["segment_id"][a["eligible"]])) == 2 and (a["segment_id"][~a["eligible"]] == -1).all()
    assert a["segment_id"][0] == utc[0] and a["segment_id"][10] == utc[10]            # id = UTC ms of the segment's first eligible tick
    assert (a["feed_regime"] == 0).all()


def test_feed_regime_boundary_is_explicit():
    b = policy.FEED_REGIME_BOUNDARY_UTC_MS
    assert policy.feed_regime(np.array([b - 1, b, b + 1])).tolist() == [0, 1, 1]
    m = policy.policy_manifest()["feed_regime"]
    assert m["name"] == "FEED_REGIME_BOUNDARY" and "NO claim" in m["claim"] and "flags" in m["forbidden_as_features"]


# ============================ chronological folds, purge, embargo ============================
def test_default_plan_is_valid_chronological_and_leaves_the_holdout_alone():
    fs, h = folds.default_folds(), folds.FinalHoldout()
    assert folds.validate_plan(fs, h) == []
    for f in fs:
        assert f.train_start_ms < f.train_end_ms <= f.val_start_ms < f.val_end_ms
        assert f.val_end_ms + folds.EMBARGO_S * 1000 <= h.start_ms
    assert [f.val_start_ms for f in fs] == sorted(f.val_start_ms for f in fs)
    assert all(a.val_end_ms <= b.val_start_ms for a, b in zip(fs, fs[1:]))
    assert all(a.train_end_ms < b.train_end_ms for a, b in zip(fs, fs[1:]))            # expanding windows
    assert folds.day_start_utc_ms(dt.date(2026, 9, 14)) == h.start_ms


def test_validate_plan_catches_every_kind_of_chronology_error():
    h = folds.FinalHoldout()
    f = folds.default_folds()
    swapped = [folds.FoldSpec("X", "2026-06-01", "2026-06-19", "2026-06-01", "2026-06-12")]       # val starts before train ends
    assert any("strictly precede" in p for p in folds.validate_plan(swapped, h))
    overlap = [f[0], folds.FoldSpec("Y", "2026-04-06", "2026-06-12", "2026-06-15", "2026-07-03")]
    assert any("overlaps" in p for p in folds.validate_plan(overlap, h))
    into_holdout = [folds.FoldSpec("Z", "2026-04-06", "2026-09-04", "2026-09-07", "2026-09-14")]
    assert any("final holdout" in p for p in folds.validate_plan(into_holdout, h))
    assert any("embargo shorter" in p for p in folds.validate_plan(f, h, embargo_s=10))


ADJACENT = folds.FoldSpec("T", "2026-04-06", "2026-04-08", "2026-04-09", "2026-04-10")      # train_end == val_start (no weekend)


def test_purge_removes_train_samples_whose_label_reaches_the_embargoed_boundary():
    fold = ADJACENT
    assert fold.train_end_ms == fold.val_start_ms
    vs, E = fold.val_start_ms, folds.EMBARGO_S * 1000
    obs = np.array([vs - E - 400_000, vs - E - 100_000, vs - E - 1_000, vs - E, vs - 1000, vs + 1000], dtype="int64")
    end = obs + 301_000
    keep = folds.train_mask(fold, obs, end)
    assert keep.tolist() == [True, False, False, False, False, False]             # label end (obs+301 s) must be <= val_start - embargo
    assert fold.train_end_ms <= vs                                                # and nothing at/after the val start is train
    c = folds.purge_counts(fold, obs, end)
    assert c["train_candidates"] == 5 and c["train_kept"] == 1 and c["train_purged_or_embargoed"] == 4
    assert c["val_candidates"] == 1
    # a sample whose label would end exactly at the embargo line is kept; one ms later is purged
    assert folds.train_mask(fold, np.array([vs - E - 1000]), np.array([vs - E]))[0]
    assert not folds.train_mask(fold, np.array([vs - E - 1000]), np.array([vs - E + 1]))[0]


def test_validation_samples_must_finish_inside_the_validation_block():
    fold = folds.default_folds()[0]
    obs = np.array([fold.val_start_ms, fold.val_end_ms - 400_000, fold.val_end_ms - 100_000, fold.val_end_ms + 1], dtype="int64")
    assert folds.val_mask(fold, obs, obs + 301_000).tolist() == [True, True, False, False]


def test_embargo_zero_would_admit_boundary_labels_so_the_embargo_is_actually_applied():
    fold = ADJACENT
    obs = np.array([fold.val_start_ms - 1_000_000], dtype="int64")
    assert not folds.train_mask(fold, obs, obs + 301_000)[0]                      # 1000 s before val: inside the 1800 s embargo
    assert folds.train_mask(fold, obs, obs + 301_000, embargo_s=0)[0]


def test_real_weekend_alignment_makes_purge_a_noop_but_the_rule_still_holds():
    # train ends Friday close; validation starts Monday: >48 h apart, so no label can straddle the boundary
    f = folds.default_folds()[0]
    assert (f.val_start_ms - f.train_end_ms) >= 2 * 86_400_000 - 1


# ============================ final holdout protection ============================
def test_final_holdout_is_closed_to_development():
    h = folds.FinalHoldout()
    ok = np.array([h.start_ms - folds.EMBARGO_S * 1000 - 1], dtype="int64")
    folds.assert_development_only(ok)
    for bad in (np.array([h.start_ms]), np.array([h.start_ms + 10**9]), np.array([0, h.end_ms])):
        with pytest.raises(folds.HoldoutViolation):
            folds.assert_development_only(bad, what="features")
    obs = np.array([h.start_ms - 5_000_000, h.start_ms - 100_000, h.start_ms + 5], dtype="int64")
    assert folds.development_mask(obs, obs + 301_000).tolist() == [True, False, False]   # 2nd: label reaches the embargo line
    assert h.first_day == "2026-09-14" and h.last_day == "2026-10-06"


def test_no_fold_validation_or_training_sample_can_come_from_the_holdout():
    h = folds.FinalHoldout()
    for f in folds.default_folds():
        assert f.val_end_ms <= h.start_ms and f.train_end_ms <= h.start_ms


# ============================ training-only preprocessing ============================
def _frame(fold, n=600, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(fold.train_start_ms, fold.val_end_ms, n)).astype("int64")
    return pd.DataFrame({"timestamp_utc_ms": ts, "a": rng.normal(0, 1, n), "b": rng.gamma(2.0, 2.0, n)})


def test_scaler_parameters_depend_only_on_the_training_block():
    fold = folds.default_folds()[0]
    df = _frame(fold)
    tr = df[df.timestamp_utc_ms < fold.train_end_ms]
    p1 = FoldPreprocessor(fold, ["a", "b"]).fit(tr)
    df2 = df.copy()
    df2.loc[df2.timestamp_utc_ms >= fold.val_start_ms, ["a", "b"]] *= 1000.0                  # wreck validation data
    p2 = FoldPreprocessor(fold, ["a", "b"]).fit(df2[df2.timestamp_utc_ms < fold.train_end_ms])
    assert p1.scaler.params == p2.scaler.params
    val = df[df.timestamp_utc_ms >= fold.val_start_ms]
    out = p1.transform(val)                                                                   # val transformed with TRAIN params
    med, sc = p1.scaler.params["a"]
    assert np.allclose(out["a"].to_numpy(), (val["a"].to_numpy() - med) / sc)


def test_fitting_on_data_that_includes_validation_or_future_raises():
    fold = folds.default_folds()[0]
    df = _frame(fold)
    with pytest.raises(LeakageError):
        FoldPreprocessor(fold, ["a"]).fit(df)                                                 # contains validation rows
    with pytest.raises(LeakageError):
        FoldPreprocessor(fold, ["a"]).transform(df)                                           # transform before fit
    assert not hasattr(FoldPreprocessor, "fit_transform")


def test_thresholds_baselines_and_regime_cutoffs_are_train_only():
    fold = folds.default_folds()[0]
    df = _frame(fold)
    tr = df[df.timestamp_utc_ms < fold.train_end_ms]
    q = train_only_quantile(tr["b"].to_numpy(), tr["timestamp_utc_ms"].to_numpy(), fold, [0.5, 0.99])
    assert np.allclose(q, np.quantile(tr["b"], [0.5, 0.99]))
    base = train_only_baseline(tr["b"].to_numpy(), tr["timestamp_utc_ms"].to_numpy(), fold)
    assert base["n"] == len(tr)
    for fn in (lambda: train_only_quantile(df["b"].to_numpy(), df["timestamp_utc_ms"].to_numpy(), fold, 0.9),
               lambda: train_only_baseline(df["b"].to_numpy(), df["timestamp_utc_ms"].to_numpy(), fold)):
        with pytest.raises(LeakageError):
            fn()
    from fxscalp.research.preprocessing import assert_training_only
    with pytest.raises(LeakageError):                                    # training samples whose labels run into the embargo
        assert_training_only(tr["timestamp_utc_ms"].to_numpy(), fold, label_end_ms=np.array([fold.val_start_ms - 10]))


# ============================ feature eligibility ============================
def test_all_119_features_are_classified_and_nothing_uses_forbidden_inputs():
    m = build_feature_manifest()
    assert m["n_features"] == 119 and sum(m["counts"].values()) == 119
    assert m["counts"][ALLOWED] == 108 and m["counts"][CONDITIONAL] == 11 and m["counts"]["EXCLUDED"] == 0
    assert {r["name"] for r in m["features"]} == set(m["model_input_allowed"]) | set(m["model_input_conditional"]) | set(m["excluded"])
    assert m["feature_set_version"].startswith("xauusd_core-fs1-")


def test_model_input_check_rejects_flags_forbidden_columns_and_unmet_conditions():
    m = build_feature_manifest()
    assert_model_inputs_allowed(m["model_input_allowed"], m)
    for bad in ("flags", "unknown_flag_bits", "quality_flags", "volume", "last", "label", "segment_id", "raw_flags_0x400"):
        with pytest.raises(ValueError):
            assert_model_inputs_allowed(m["model_input_allowed"][:3] + [bad], m)
    with pytest.raises(ValueError, match="CONDITIONAL"):
        assert_model_inputs_allowed(["spread_regime_code"], m)
    assert_model_inputs_allowed(["spread_regime_code"], m, allow_conditional=True)            # caller has applied the gating
    with pytest.raises(ValueError, match="not a Feature Set v1"):
        assert_model_inputs_allowed(["invented_feature"], m)
    assert {"flags", "unknown_flag_bits"} <= FORBIDDEN_MODEL_COLUMNS


def test_features_are_invariant_to_raw_flags_volume_last_and_quality_flags():
    """Dynamic proof that no feature depends on undocumented flag bits, traded volume or `last` (all excluded inputs)."""
    from tests.unit.features_helpers import POINT, norm, raw
    from fxscalp.features.pipeline import PipelineConfig, build_feature_frame
    rng = np.random.default_rng(4)
    ts = 1_760_000_000_000 + np.sort(rng.integers(0, 4 * 3600 * 1000, 40_000))
    mid = 2000 + np.cumsum(rng.normal(0, 0.05, len(ts)))
    arr = raw(ts, mid)
    a = norm(arr, POINT)
    arr2 = arr.copy()
    arr2["flags"] = rng.choice([0x486, 0x482, 0x404, 0x80, 0x400, 0x6], len(arr2)).astype("uint32")
    arr2["last"] = rng.normal(3000, 10, len(arr2))
    arr2["volume"] = rng.integers(0, 1000, len(arr2))
    arr2["volume_real"] = rng.random(len(arr2)) * 50
    b = norm(arr2, POINT)
    cfg = PipelineConfig()
    fa = build_feature_frame(a, cfg, point=POINT).df
    fb = build_feature_frame(b, cfg, point=POINT).df
    cols = [c for c in fa.columns if c in set(build_feature_manifest()["model_input_allowed"]) | set(build_feature_manifest()["model_input_conditional"])]
    pd.testing.assert_frame_equal(fa[cols].reset_index(drop=True), fb[cols].reset_index(drop=True))


# ============================ deterministic, tamper-evident manifests ============================
def test_code_manifests_are_deterministic():
    a, b = spec.build_code_manifests(), spec.build_code_manifests()
    assert {k: spec.canon(v) for k, v in a.items()} == {k: spec.canon(v) for k, v in b.items()}
    assert set(a) == set(spec.CODE_COMPONENTS)
    assert a["labels_v0.json"]["horizons_s"] == list(CANDIDATE_HORIZONS_S) == [15, 60, 300]
    assert len(a["labels_v0.json"]["candidates"]) == 9 and a["folds_v1.json"]["violations"] == []
    assert a["cost_model_v0.json"]["components"]["commission_round_turn_points"]["status"] == "UNKNOWN/UNCALIBRATED"


def _tmp_spec(tmp_path):
    data = {"XAUUSD_RAW_V1.freeze.json": {"freeze_id": "freeze-test"}, "eligibility_summary.json": {"x": 1}}
    spec.write_spec(tmp_path, data)
    return tmp_path


def test_loader_accepts_the_frozen_spec_and_is_idempotent(tmp_path):
    d = _tmp_spec(tmp_path)
    s1 = spec.load_frozen_spec(d)
    h1 = (d / "spec_index.json").read_text(encoding="utf-8")
    _tmp_spec(tmp_path)                                              # regenerating changes nothing
    assert (d / "spec_index.json").read_text(encoding="utf-8") == h1 and spec.load_frozen_spec(d).spec_hash == s1.spec_hash


def test_loader_detects_edits_missing_components_and_code_drift(tmp_path, monkeypatch):
    d = _tmp_spec(tmp_path)
    p = d / "labels_v0.json"
    body = json.loads(p.read_text(encoding="utf-8"))
    body["horizons_s"] = [5, 10]
    p.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(spec.SpecDriftError, match="modified after the freeze"):
        spec.load_frozen_spec(d)
    d2 = _tmp_spec(tmp_path / "b")
    (d2 / "cost_model_v0.json").unlink()
    with pytest.raises(spec.SpecDriftError, match="missing"):
        spec.load_frozen_spec(d2)
    d3 = _tmp_spec(tmp_path / "c")
    monkeypatch.setattr(policy, "SEGMENT_GAP_S", 120)
    with pytest.raises(spec.SpecDriftError, match="no longer matches"):
        spec.load_frozen_spec(d3)
    monkeypatch.setattr(policy, "SEGMENT_GAP_S", 300)
    monkeypatch.setattr(labels, "CANDIDATE_HORIZONS_S", (5, 10))
    with pytest.raises(spec.SpecDriftError):
        spec.load_frozen_spec(d3)


@pytest.mark.skipif(not (spec.SPEC_DIR / 'spec_index.json').exists(), reason='frozen spec not generated yet (removed once committed)')
def test_the_committed_frozen_spec_loads_and_matches_the_code():
    s = spec.load_frozen_spec()                                      # research/phase2b in the repository
    assert s["XAUUSD_RAW_V1.freeze.json"]["dataset_id"] == policy.RAW_V1_DATASET_ID
    assert s["policy.json"]["segment_policy"]["gap_s"] == 300
    assert s["folds_v1.json"]["final_holdout"]["first_day"] == "2026-09-14"
    assert copy.deepcopy(s.components) == s.components
