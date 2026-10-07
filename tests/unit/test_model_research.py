"""Phase 2B model research framework: determinism, experiment ids, fold isolation, holdout protection, preprocessing isolation,
serialization, prediction reproducibility, metric correctness, probability handling and the pre-registered protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxscalp.research import folds as F
from fxscalp.research import policy
from fxscalp.research.model_research import metrics as M
from fxscalp.research.model_research import protocol as P
from fxscalp.research.model_research.experiment import (ExpConfig, ExperimentExists, ExperimentStore, Runner, experiment_id, hp_tuple)
from fxscalp.research.model_research.models import (SESSION_AGGREGATES, ModelPipeline, apply_temperature, feature_matrix,
                                                    fit_regime_config, fit_temperature)
from fxscalp.research.preprocessing import LeakageError

FEAT = json.loads((Path(__file__).resolve().parents[2] / "research/phase2b/feature_set_v1.json").read_text(encoding="utf-8"))
ALLOWED = FEAT["model_input_allowed"]
FROZEN = {"raw_dataset_freeze": P.EXPECTED_FREEZE_ID, "spec_hash": P.EXPECTED_SPEC_HASH, "protocol_hash": P.protocol_hash()}


def synth_matrix(n_days=120, per_day=70, seed=0) -> pd.DataFrame:
    """Random matrix with the real column layout and a weak planted signal; timestamps span the development window."""
    rng = np.random.default_rng(seed)
    start, end = F.day_start_utc_ms(F.FIRST_ELIGIBLE_DAY), F.day_end_utc_ms(F.DEV_LAST_DAY)
    ts = np.sort(rng.choice(np.arange(start // 30000, end // 30000), size=n_days * per_day, replace=False) * 30000).astype("int64")
    n = len(ts)
    cols = {"ts": ts, "segment_id": np.zeros(n, dtype="int64")}
    for c in set(ALLOWED) | {*SESSION_AGGREGATES, "session_so_far_complete", "spread_regime_code", "volatility_regime_code"}:
        cols[c] = rng.normal(0, 1, n).astype("float32")
    cols["session_code"] = rng.integers(0, 5, n).astype("float32")
    cols["spread_points"] = np.abs(rng.normal(9, 2, n)).astype("float32")
    cols["realized_vol_bps_60s"] = np.abs(rng.normal(3, 1, n)).astype("float32")
    cols["session_so_far_complete"] = (rng.random(n) > 0.1).astype("float32")
    sig = cols["ret_bps_60s"] * 0.8 + rng.normal(0, 1, n)
    y = np.where(sig > 0.7, 2, np.where(sig < -0.7, 0, 1)).astype("int8")
    for s in P.SCENARIOS:
        cols[f"y_{s}"], cols[f"st_{s}"], cols[f"le_{s}"] = y, np.zeros(n, dtype="int8"), ts + 61_000
    return pd.DataFrame(cols)


@pytest.fixture(scope="module")
def df():
    return synth_matrix()


def runner(df, tmp_path, name="exp"):
    return Runner(df, "matrix-test", FROZEN, ALLOWED, ExperimentStore(tmp_path / name), n_jobs=1)


LOG_HP = {"family": "logistic", "C": 1.0, "class_weight": None}
LGB_HP = {"family": "lightgbm", "num_leaves": 15, "n_estimators": 20}


# ============================ experiment ids ============================
def test_experiment_id_is_deterministic_and_sensitive_to_every_component(df):
    ctx = runner(df, Path("/nonexistent")).ctx
    base = ExpConfig("model", "CORE", "logistic", hp_tuple(LOG_HP), "C0_spread_only")
    assert experiment_id(base, ctx) == experiment_id(ExpConfig("model", "CORE", "logistic", hp_tuple(LOG_HP), "C0_spread_only"), ctx)
    changed = [ExpConfig("model", "REGIME", "logistic", hp_tuple(LOG_HP), "C0_spread_only"),
               ExpConfig("model", "CORE", "lightgbm", hp_tuple(LGB_HP), "C0_spread_only"),
               ExpConfig("model", "CORE", "logistic", hp_tuple({**LOG_HP, "C": 0.1}), "C0_spread_only"),
               ExpConfig("model", "CORE", "logistic", hp_tuple(LOG_HP), "C1_moderate"),
               ExpConfig("model", "CORE", "logistic", hp_tuple(LOG_HP), "C0_spread_only", seed=8),
               ExpConfig("ablation", "CORE", "logistic", hp_tuple(LOG_HP), "C0_spread_only", columns=("spread_points",), tag="x")]
    ids = {experiment_id(c, ctx) for c in changed} | {experiment_id(base, ctx)}
    assert len(ids) == len(changed) + 1
    for key in ("freeze_id", "feature_manifest_sha", "folds_sha", "protocol_hash", "matrix_id", "grid_step_s", "label", "spec_hash"):
        c2 = dict(ctx)
        c2[key] = "different"
        assert experiment_id(base, c2) != experiment_id(base, ctx), key


def test_experiments_are_deterministic_stored_completely_and_never_overwritten(df, tmp_path):
    r1, r2 = runner(df, tmp_path, "a"), runner(df, tmp_path, "b")
    id1 = r1.run_models("C1_moderate", "CORE", [LOG_HP])[0]
    id2 = r2.run_models("C1_moderate", "CORE", [LOG_HP])[0]
    assert id1 == id2
    p1, p2 = r1.store.predictions(id1), r2.store.predictions(id2)
    pd.testing.assert_frame_equal(p1, p2)                                    # identical predictions
    res = r1.store.load(id1)
    for k in ("config", "context", "protocol_hash", "git", "env", "folds", "aggregate_calibrated", "total_duration_s", "peak_rss_mb"):
        assert k in res, k
    assert len(res["folds"]) == 5 and all("importance" in f and "model_file" in f and "duration_s" in f for f in res["folds"])
    d = r1.store.dir(id1)
    assert (d / "predictions.parquet").exists() and (d / "model_F5.joblib").exists()
    with pytest.raises(ExperimentExists):
        r1.store.save(id1, res, p1)
    mtime = (d / "result.json").stat().st_mtime_ns
    assert r1.run_models("C1_moderate", "CORE", [LOG_HP]) == [id1]           # re-running loads, does not recompute or overwrite
    assert (d / "result.json").stat().st_mtime_ns == mtime


# ============================ fold isolation / preprocessing isolation ============================
def test_fold_rows_respect_chronology_purge_embargo_and_never_touch_the_holdout(df, tmp_path):
    r = runner(df, tmp_path)
    for fold in r.folds:
        rows = r.fold_rows("C0_spread_only", fold)
        ts, le = df["ts"].to_numpy(), df["le_C0_spread_only"].to_numpy()
        assert ts[rows["train"]].max() < fold.val_start_ms <= ts[rows["val"]].min() and ts[rows["val"]].max() < fold.val_end_ms
        assert le[rows["train"]].max() <= fold.val_start_ms - F.EMBARGO_S * 1000
        assert not set(rows["train"]) & set(rows["val"]) and not set(rows["fit"]) & set(rows["cal"])
        assert set(rows["fit"]) | set(rows["cal"]) <= set(rows["train"])
        assert ts[rows["fit"]].max() < ts[rows["cal"]].min()                  # calibration slice is strictly later than the fit slice
        assert ts[rows["val"]].max() < F.FinalHoldout().start_ms


def test_fitted_artifacts_depend_on_training_rows_only(df, tmp_path):
    r = runner(df, tmp_path)
    fold = r.folds[1]
    rows = r.fold_rows("C0_spread_only", fold)

    def fit_all(frame):
        tr = frame.iloc[rows["train"]]
        regime = fit_regime_config({k: tr[k].to_numpy("float64") for k in ("spread_points", "session_code", "realized_vol_bps_60s")},
                                   tr["ts"].to_numpy(), fold)
        X, names = feature_matrix(tr, "REGIME", ALLOWED, regime)
        y = frame[f"y_C0_spread_only"].to_numpy()[rows["train"]].astype("int64")
        pipe = ModelPipeline("logistic", {"C": 1.0, "class_weight": None}, names).fit_preprocessing(X).fit(X, y)
        return regime.to_dict(), pipe.prep["median"], pipe.estimator.coef_.copy()

    a = fit_all(df)
    wrecked = df.copy()
    val_or_later = np.ones(len(df), dtype=bool)
    val_or_later[rows["train"]] = False                                       # everything that is NOT a training row
    for c in ALLOWED + ["spread_points", "session_code", "realized_vol_bps_60s"]:
        wrecked.loc[val_or_later, c] = 1e6
    wrecked.loc[val_or_later, "y_C0_spread_only"] = 0
    b = fit_all(wrecked)
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])      # untouched by validation/future
    wrecked2 = df.copy()
    wrecked2.loc[rows["train"], "spread_points"] += 50
    assert fit_all(wrecked2)[0] != a[0]                                       # but they DO depend on the training rows


def test_regime_thresholds_refuse_non_training_data(df, tmp_path):
    r = runner(df, tmp_path)
    fold = r.folds[0]
    rows = r.fold_rows("C0_spread_only", fold)
    mixed = df.iloc[np.concatenate([rows["train"], rows["val"]])]
    with pytest.raises(LeakageError):
        fit_regime_config({k: mixed[k].to_numpy("float64") for k in ("spread_points", "session_code", "realized_vol_bps_60s")}, mixed["ts"].to_numpy(), fold)


def test_conditional_features_are_masked_unless_the_gate_holds(df):
    X, names = feature_matrix(df.iloc[:400], "CONDITIONAL", ALLOWED)
    gate = df["session_so_far_complete"].to_numpy()[:400] == 1.0
    for c in SESSION_AGGREGATES:
        col = X[:, names.index(c)]
        assert np.isnan(col[~gate]).all() and np.isfinite(col[gate]).all()
    assert len(names) == 108 + 8 and feature_matrix(df.iloc[:5], "CORE", ALLOWED)[0].shape[1] == 108


# ============================ holdout protection ============================
def test_runner_and_matrix_loader_refuse_holdout_rows(df, tmp_path):
    bad = df.copy()
    bad.loc[len(bad) - 1, "ts"] = F.FinalHoldout().start_ms + 1000
    with pytest.raises(F.HoldoutViolation):
        runner(bad, tmp_path)


# ============================ serialization / reproducibility ============================
@pytest.mark.parametrize("hp", [LOG_HP, LGB_HP], ids=["logistic", "lightgbm"])
def test_models_serialize_and_reproduce_predictions(df, tmp_path, hp):
    X, names = feature_matrix(df.iloc[:6000], "CORE", ALLOWED)
    y = df["y_C0_spread_only"].to_numpy()[:6000].astype("int64")
    pipe = ModelPipeline(hp["family"], {k: v for k, v in hp.items() if k != "family"}, names, P.SEED, 1).fit_preprocessing(X).fit(X, y)
    pipe.temperature = 1.3
    p = pipe.predict_calibrated(X[:500])
    path = tmp_path / "m.joblib"
    pipe.save(path)
    loaded = ModelPipeline.load(path)
    assert np.array_equal(loaded.predict_calibrated(X[:500]), p) and loaded.temperature == 1.3
    again = ModelPipeline(hp["family"], {k: v for k, v in hp.items() if k != "family"}, names, P.SEED, 1).fit_preprocessing(X).fit(X, y)
    again.temperature = 1.3
    assert np.array_equal(again.predict_calibrated(X[:500]), p)               # same seed + data -> identical predictions


def test_lightgbm_importance_is_normalised_and_logistic_importance_is_scaled(df):
    X, names = feature_matrix(df.iloc[:5000], "CORE", ALLOWED)
    y = df["y_C0_spread_only"].to_numpy()[:5000].astype("int64")
    lg = ModelPipeline("lightgbm", {"num_leaves": 15, "n_estimators": 20}, names, P.SEED, 1).fit_preprocessing(X).fit(X, y)
    assert abs(sum(lg.importance().values()) - 1.0) < 1e-6
    lr = ModelPipeline("logistic", {"C": 1.0, "class_weight": None}, names).fit_preprocessing(X).fit(X, y)
    top = max(lr.importance(), key=lr.importance().get)
    assert top == "ret_bps_60s"                                                # the planted signal is recovered


# ============================ metric correctness ============================
def test_metrics_match_sklearn_and_hand_computation():
    from sklearn import metrics as sk
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 4000)
    p = rng.dirichlet([1, 1, 1], 4000) * 0.6 + np.eye(3)[y] * 0.4
    yh = p.argmax(axis=1)
    m = M.full_metrics(y, yh, p)
    assert abs(m["macro_f1"] - sk.f1_score(y, yh, average="macro")) < 1e-12
    assert abs(m["balanced_accuracy"] - sk.balanced_accuracy_score(y, yh)) < 1e-12
    assert abs(m["accuracy"] - sk.accuracy_score(y, yh)) < 1e-12
    assert abs(m["log_loss"] - sk.log_loss(y, p, labels=[0, 1, 2])) < 1e-9
    assert np.allclose(m["precision"], sk.precision_score(y, yh, average=None)) and np.allclose(m["recall"], sk.recall_score(y, yh, average=None))
    assert np.allclose(m["f1"], sk.f1_score(y, yh, average=None)) and np.array_equal(np.array(m["confusion"]), sk.confusion_matrix(y, yh))
    brier_ref = np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))
    assert abs(m["brier"] - brier_ref) < 1e-12
    assert abs(sum(m["pred_freq"]) - 1) < 1e-12 and abs(sum(m["class_dist"]) - 1) < 1e-12


def test_ece_is_zero_for_calibrated_and_large_for_overconfident_probabilities():
    rng = np.random.default_rng(2)
    n = 60000
    p = rng.dirichlet([2, 2, 2], n)
    y = np.array([rng.choice(3, p=row) for row in p[:20000]])
    assert M.ece_top(y, p[:20000])[0] < 0.02                                   # outcomes drawn from the stated probabilities
    over = np.eye(3)[rng.integers(0, 3, 20000)] * 0.98 + 0.01                  # always 98% sure, right only 1/3 of the time
    assert M.ece_top(rng.integers(0, 3, 20000), over)[0] > 0.5


def test_block_bootstrap_sufficient_statistics_equal_direct_metrics():
    rng = np.random.default_rng(3)
    ts = np.arange(5000) * 30_000
    y, yh = rng.integers(0, 3, 5000), rng.integers(0, 3, 5000)
    p = rng.dirichlet([1, 1, 1], 5000)
    blocks = M.block_counts(ts, y, yh, p, 600)
    point = M.pooled_point([blocks])
    direct = M.full_metrics(y, yh, p)
    assert abs(point["macro_f1"] - direct["macro_f1"]) < 1e-12 and abs(point["log_loss"] - direct["log_loss"]) < 1e-9
    assert blocks[:, -1].sum() == 5000 and len(blocks) == 5000 * 30 // 600
    # identical model and baseline -> the paired difference is exactly zero in every replicate
    out = M.paired_block_bootstrap([blocks], [blocks], B=50, seed=1)
    assert out["macro_f1"]["diff"]["lo95"] == out["macro_f1"]["diff"]["hi95"] == 0.0
    # a model better than the baseline has a positive difference whose CI excludes zero
    good = M.block_counts(ts, y, y, np.eye(3)[y] * 0.9 + 0.033, 600)
    d = M.paired_block_bootstrap([good], [blocks], B=100, seed=1)
    assert d["macro_f1"]["diff"]["lo95"] > 0 and d["log_loss"]["diff"]["hi95"] < 0


# ============================ probability handling ============================
def test_temperature_scaling_fixes_overconfidence_without_changing_decisions():
    rng = np.random.default_rng(4)
    true_p = rng.dirichlet([2, 2, 2], 30000)
    y = np.array([rng.choice(3, p=row) for row in true_p])
    z = np.log(true_p) * 3.0                                                   # sharpened -> overconfident
    over = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    T = fit_temperature(over, y)
    cal = apply_temperature(over, T)
    assert 2.5 < T < 3.5 and abs(T - 3.0) < 0.3
    assert M.log_loss(y, cal) < M.log_loss(y, over) and M.ece_top(y, cal)[0] < M.ece_top(y, over)[0]
    assert np.array_equal(cal.argmax(axis=1), over.argmax(axis=1))
    assert np.allclose(cal.sum(axis=1), 1.0) and (cal >= 0).all() and np.allclose(apply_temperature(over, 1.0), over)


def test_experiment_probabilities_are_valid_distributions(df, tmp_path):
    r = runner(df, tmp_path)
    eid = r.run_models("C0_spread_only", "CORE", [LOG_HP])[0]
    p = r.store.predictions(eid)
    for pref in ("u", "c"):
        pr = p[[f"{pref}0", f"{pref}1", f"{pref}2"]].to_numpy("float64")
        assert np.allclose(pr.sum(axis=1), 1.0, atol=1e-5) and (pr >= 0).all()
    assert set(p["fold"]) == {"F1", "F2", "F3", "F4", "F5"}


def test_baselines_use_training_priors_and_are_causal(df, tmp_path):
    r = runner(df, tmp_path)
    ids = r.run_baselines("C0_spread_only")
    res = {r.store.load(i)["config"]["family"]: r.store.load(i) for i in ids}
    assert set(res) == {"majority", "stratified_random", "momentum", "mean_reversion"}
    maj = res["majority"]["folds"][0]["calibrated"]
    assert maj["recall"].count(1.0) == 1 and abs(maj["pred_freq"][int(np.argmax(maj["pred_freq"]))] - 1.0) < 1e-12
    mom, rev = res["momentum"]["folds"][0]["calibrated"], res["mean_reversion"]["folds"][0]["calibrated"]
    assert np.allclose(mom["pred_freq"], rev["pred_freq"][::-1])              # mean reversion is the exact mirror of momentum
    assert "log_loss" not in mom                                               # hard rules carry no probabilities


# ============================ the pre-registered protocol ============================
def test_protocol_is_pre_registered_stable_and_matches_the_committed_file():
    m = P.protocol_manifest()
    assert m["n_candidates_per_scenario"] == 30 == len(P.hp_grid()) * 3 and len(P.hp_grid()) == 10
    committed = json.loads((Path(__file__).resolve().parents[2] / "research/phase2b/model_research/protocol_v1.json").read_text(encoding="utf-8"))
    assert committed["protocol_hash"] == P.protocol_hash() and committed["written_before_training"] is True
    assert m["final_holdout"].startswith("LOCKED") and m["primary_label"] == "tb_H60" and m["third_family"].startswith("NOT used")
    assert P.EXPECTED_SPEC_HASH == "57fb49f5eb10c5f0e8fd3a3c82c1c1c39603ee915be3cf510286c35475331845"
    assert set(m["selection_rule"]["never_used"]) >= {"final holdout", "PnL", "Sharpe", "profit factor"}


def test_frozen_state_verification_fails_closed_on_drift(monkeypatch):
    from fxscalp.research.model_research import frozen
    ok = frozen.verify_frozen_state(check_data=False)
    assert ok["spec_hash"] == P.EXPECTED_SPEC_HASH and ok["primary_label"] == "tb_H60"
    monkeypatch.setattr(policy, "SEGMENT_GAP_S", 200)
    with pytest.raises(frozen.FrozenStateError, match="drift"):
        frozen.verify_frozen_state(check_data=False)
    monkeypatch.setattr(policy, "SEGMENT_GAP_S", 300)
    monkeypatch.setattr(P, "EXPECTED_SPEC_HASH", "0" * 64)
    with pytest.raises(frozen.FrozenStateError, match="spec_hash"):
        frozen.verify_frozen_state(check_data=False)


# ============================ selection rule, verdict, diagnostics ============================
def _fake_result(scn, macro, ll, ece, rec=(0.3, 0.5, 0.3), fam="logistic", fc="CORE", hp=(("C", 1.0),)):
    folds = [{"fold": f"F{i + 1}", "calibrated": {"macro_f1": macro[i], "log_loss": ll[i], "ece_top": ece, "balanced_accuracy": macro[i],
                                                   "accuracy": 0.5, "recall": list(rec)}} for i in range(5)]
    return {"config": {"kind": "model", "feature_config": fc, "family": fam, "hp": [list(x) for x in hp], "scenario": scn, "columns": None, "tag": ""},
            "folds": folds, "experiment_id": f"e-{fam}-{scn}"}


def _fake_baselines(f1=0.334, ll=1.0):
    out = {}
    for b in ("majority", "stratified_random", "momentum", "mean_reversion"):
        out[b] = {s: {"folds": [{"calibrated": {"macro_f1": f1 if b == "stratified_random" else 0.2, "log_loss": ll}} for _ in range(5)]}
                  for s in P.SCENARIOS}
    return out


def test_selection_gates_score_and_tie_break_follow_the_pre_registered_rule():
    from fxscalp.research.model_research import selection as S
    base = _fake_baselines()
    def cand(macro, ll, ece=0.02, rec=(0.3, 0.5, 0.3), fam="logistic", fc="CORE", hp=(("C", 1.0),)):
        return {s: _fake_result(s, macro, ll, ece, rec, fam, fc, hp) for s in P.SCENARIOS}
    cands = {
        ("CORE", "logistic", (("C", 1.0),), None, ""): cand([0.40] * 5, [0.95] * 5),                         # passes everything
        ("CORE", "logistic", (("C", 0.1),), None, ""): cand([0.30] * 5, [0.95] * 5, hp=(("C", 0.1),)),         # G1: never beats baseline
        ("CORE", "lightgbm", (("n_estimators", 150),), None, ""): cand([0.40, 0.41, 0.40, 0.30, 0.30], [0.95] * 5, fam="lightgbm"),   # only 3 folds beat
        ("REGIME", "logistic", (("C", 1.0),), None, ""): cand([0.41] * 5, [1.05] * 5, fc="REGIME"),             # G2: worse than the prior
        ("CORE", "logistic", (("C", 10.0),), None, ""): cand([0.45] * 5, [0.9] * 5, rec=(0.01, 0.99, 0.0)),    # G3: never predicts LONG
    }
    df = S.evaluate(cands, base)
    ok = df[df["gates_123"]]
    assert len(ok) == 1 and ok.iloc[0]["hp"] == {"C": 1.0} and ok.iloc[0]["feature_config"] == "CORE"
    assert df[(df.family == "lightgbm")].iloc[0]["G1_folds_beating_baseline"] == 3 and not df[df.feature_config == "REGIME"].iloc[0]["G2"]
    assert not df[df.hp.apply(lambda h: h.get("C") == 10.0)].iloc[0]["G3"]
    row = ok.iloc[0]
    assert abs(row["S"] - (0.40 - 0.0 - 0.02)) < 1e-9                                                       # S = M - D - E
    # tie-break: within 0.005 prefer logistic over LightGBM, CORE over REGIME, fewer trees
    two = S.evaluate({("CORE", "logistic", (("C", 1.0),), None, ""): cand([0.400] * 5, [0.95] * 5),
                      ("CORE", "lightgbm", (("n_estimators", 150),), None, ""): cand([0.403] * 5, [0.95] * 5, fam="lightgbm")}, base)
    assert S.tie_break(two[two.gates_123]).family == "logistic"
    far = S.evaluate({("CORE", "logistic", (("C", 1.0),), None, ""): cand([0.400] * 5, [0.95] * 5),
                      ("CORE", "lightgbm", (("n_estimators", 150),), None, ""): cand([0.420] * 5, [0.95] * 5, fam="lightgbm")}, base)
    assert S.tie_break(far[far.gates_123]).family == "lightgbm"                                           # clearly better beats simplicity


def _boot(lo, mean_diff, ll_hi, cand_f1, base_f1, cand_ll, prior_ll, folds=5):
    return {"macro_f1_vs_best_baseline": {"diff": {"mean": mean_diff, "lo95": lo, "hi95": mean_diff + 0.01}},
            "log_loss_vs_prior": {"diff": {"mean": cand_ll - prior_ll, "lo95": cand_ll - prior_ll - 0.01, "hi95": ll_hi}},
            "pooled_candidate": {"macro_f1": cand_f1, "log_loss": cand_ll}, "pooled_best_baseline": {"macro_f1": base_f1},
            "pooled_prior": {"log_loss": prior_ll}, "folds_beating_best_baseline": folds}


def test_verdict_rule_matches_the_pre_registered_definitions():
    from fxscalp.research.model_research import selection as S
    good = {s: _boot(0.03, 0.06, -0.02, 0.40, 0.334, 0.95, 1.00) for s in P.SCENARIOS}
    assert S.verdict(good, False)["verdict"] == "PREDICTIVE SIGNAL DETECTED"
    assert S.verdict(good, True)["verdict"] == "WEAK/UNSTABLE SIGNAL"                                      # proxy flag blocks DETECTED
    assert S.verdict(good, False)["holdout_evaluation_justified"] and not S.verdict(good, True)["holdout_evaluation_justified"]
    tiny = {s: _boot(0.001, 0.003, -0.001, 0.337, 0.334, 0.995, 1.00) for s in P.SCENARIOS}                # significant-looking but below the effect floors
    assert S.verdict(tiny, False)["verdict"] == "WEAK/UNSTABLE SIGNAL"
    unstable = dict(good)
    unstable["C2_pessimistic"] = _boot(-0.01, 0.0, 0.02, 0.33, 0.334, 1.01, 1.00)
    assert S.verdict(unstable, False)["verdict"] == "WEAK/UNSTABLE SIGNAL"
    few_folds = {s: _boot(0.03, 0.06, -0.02, 0.40, 0.334, 0.95, 1.00, folds=3) for s in P.SCENARIOS}
    assert S.verdict(few_folds, False)["verdict"] == "WEAK/UNSTABLE SIGNAL"
    none = {s: _boot(-0.03, -0.01, 0.03, 0.31, 0.334, 1.02, 1.00) for s in P.SCENARIOS}
    assert S.verdict(none, False)["verdict"] == "NO RELIABLE SIGNAL DETECTED"


def test_permutation_importance_finds_the_planted_feature_and_group_structure(df):
    from fxscalp.research.model_research import diagnostics as D
    X, names = feature_matrix(df, "CORE", ALLOWED)
    y = df["y_C0_spread_only"].to_numpy().astype("int64")
    pipe = ModelPipeline("lightgbm", {"num_leaves": 15, "n_estimators": 60}, names, P.SEED, 1).fit_preprocessing(X).fit(X, y)
    pipe.temperature = 1.0
    gi = {"signal": [names.index("ret_bps_60s")], "noise": [names.index("tick_rate_5s"), names.index("spread_points")]}
    out = D.permutation_importance(pipe, X, y, gi, [names.index("ret_bps_60s")], seed=1, n_rows=6000, repeats=2)
    assert out["groups"]["signal"]["d_log_loss"] > 10 * abs(out["groups"]["noise"]["d_log_loss"]) and out["groups"]["signal"]["d_macro_f1"] > 0.05
    assert "ret_bps_60s" in out["features"]
    assert abs(D.spearman(np.arange(20), np.arange(20)) - 1.0) < 1e-12 and D.spearman(np.arange(20), -np.arange(20)) < -0.99


def test_psi_detects_shift_and_ignores_identical_distributions():
    from fxscalp.research.model_research import diagnostics as D
    rng = np.random.default_rng(5)
    a, b = rng.normal(0, 1, 20000), rng.normal(0, 1, 20000)
    assert D.psi(a, b) < 0.02 and D.psi(a, b + 1.5) > 0.5 and D.psi(a, np.full(50, np.nan)) != D.psi(a, np.full(50, np.nan))   # NaN when too few


def test_prior_shift_correction_recovers_true_posteriors_from_a_class_balanced_model():
    """Exploratory-analysis helper: a class-weighted ('balanced') model reports p_w ∝ p_true / prior; the Elkan correction undoes it."""
    from fxscalp.research.model_research import selection as S
    rng = np.random.default_rng(6)
    prior = np.array([0.25, 0.5, 0.25])
    p_true = rng.dirichlet([3, 3, 3], 1000) * 0.5 + prior * 0.5
    p_true /= p_true.sum(axis=1, keepdims=True)
    w = p_true / prior
    p_w = w / w.sum(axis=1, keepdims=True)
    assert np.allclose(S.prior_corrected(p_w, prior), p_true, atol=1e-12) and np.allclose(S.prior_corrected(p_w, prior).sum(axis=1), 1.0)
    y = np.array([rng.choice(3, p=r) for r in p_true])
    assert M.log_loss(y, S.prior_corrected(p_w, prior)) < M.log_loss(y, p_w)           # the weighted model's raw probabilities are worse
