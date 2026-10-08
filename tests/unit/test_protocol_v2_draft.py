"""Static validation of the Protocol v2 DRAFT: it must stay inactive, preserve v1 and the frozen spec, keep the holdout locked and be
internally consistent. Nothing here trains a model on market data or reads the holdout."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from fxscalp.research.protocol_v2 import draft
from fxscalp.research.protocol_v2.stats import effective_sample_size

ROOT = Path(__file__).resolve().parents[2]


def fresh(tmp_path):
    d = tmp_path / "draft"
    draft.write_draft(d)
    return d


def edit(d, name, fn):
    p = d / name
    body = json.loads(p.read_text(encoding="utf-8"))
    fn(body)
    p.write_text(draft.canon(body), encoding="utf-8")
    idx = json.loads((d / "draft_index.json").read_text(encoding="utf-8"))      # re-index: the problem found must be the SEMANTIC one
    idx["files"][name] = draft.sha(draft.canon(body))
    idx["draft_hash"] = draft.sha(draft.canon(idx["files"]))
    (d / "draft_index.json").write_text(draft.canon(idx), encoding="utf-8")


# ============================ committed draft ============================
def test_committed_draft_is_valid_and_inactive():
    assert draft.validate_draft() == []
    with pytest.raises(draft.ProtocolNotActive):
        draft.assert_active()
    assert not (draft.DRAFT_DIR / "protocol_v2_ACTIVATION.json").exists()
    m = json.loads((draft.DRAFT_DIR / "protocol_v2_draft.json").read_text(encoding="utf-8"))
    assert m["status"] == "DRAFT_NOT_ACTIVE" and m["active"] is False


def test_regeneration_is_content_identical(tmp_path):
    d = fresh(tmp_path)
    for name in (*draft.FILES, "draft_index.json"):
        a = (d / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        b = (draft.DRAFT_DIR / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert a == b, name


# ============================ v1 preserved, frozen spec and holdout untouched ============================
def test_v1_artifacts_are_unchanged_since_sealing():
    from scripts.seal_v1_results import OUT, TEXT_FILES, bytes_sha, text_sha
    sealed = json.loads((ROOT / OUT).read_text(encoding="utf-8"))
    assert sealed["protocol_v1_formal_verdict"] == "NO RELIABLE SIGNAL DETECTED" and sealed["n_experiments"] == 129
    for f in TEXT_FILES:
        assert text_sha(ROOT / f) == sealed["files"][f], f"v1 artifact changed: {f}"
    assert any("POST-HOC" in s for s in sealed["post_hoc_exploratory_material"])
    exp = ROOT / "data/research/experiments"
    if exp.exists():                                                                  # local experiment store (not in git)
        for eid, h in list(sealed["experiments"].items())[:20]:
            assert bytes_sha(exp / eid / "result.json") == h["result_json"]


def test_draft_matches_the_frozen_specification_and_leaves_the_holdout_locked():
    from fxscalp.research import spec
    fs = spec.load_frozen_spec()
    m = json.loads((draft.DRAFT_DIR / "protocol_v2_draft.json").read_text(encoding="utf-8"))
    fz = m["frozen_and_unchanged"]
    assert fz["spec_hash"] == fs.spec_hash and fz["embargo_s"] == fs["folds_v1.json"]["embargo_s"]
    assert fz["segment_gap_s"] == fs["policy.json"]["segment_policy"]["gap_s"]
    assert fz["feature_set"] == fs["feature_set_v1.json"]["feature_set_version"]
    assert fz["folds"] == [f["name"] for f in fs["folds_v1.json"]["folds"]]
    assert list(fs["cost_model_v0.json"]["scenarios"]) == fz["cost_scenarios"]
    h = fs["folds_v1.json"]["final_holdout"]
    assert (m["holdout"]["first_day"], m["holdout"]["last_day"]) == (h["first_day"], h["last_day"])
    assert m["holdout"]["status"] == "LOCKED" and m["holdout"]["v2_access"] == "none"
    fp = json.dumps(json.loads((draft.DRAFT_DIR / "fingerprint_tests_v2.json").read_text(encoding="utf-8")))
    for day in re.findall(r"2026-\d\d-\d\d", fp):                                    # every date used lies inside the development window
        assert "2026-04-06" <= day <= "2026-09-11", day


def test_draft_code_contains_no_training_or_execution_paths():
    src = (ROOT / "src/fxscalp/research/protocol_v2/draft.py").read_text(encoding="utf-8")
    for bad in ("import lightgbm", "import sklearn", "from sklearn", "MetaTrader5", "order_send", "load_matrix", "TickStore", "FinalHoldout("):
        assert bad not in src, bad


# ============================ the validator catches every kind of violation ============================
@pytest.mark.parametrize("name,mutate,needle", [
    ("protocol_v2_draft.json", lambda b: b.update(status="ACTIVE"), "DRAFT_NOT_ACTIVE"),
    ("protocol_v2_draft.json", lambda b: b["holdout"].update(status="OPEN"), "LOCKED"),
    ("protocol_v2_draft.json", lambda b: b["holdout"].update(v2_access="read"), "LOCKED"),
    ("protocol_v2_draft.json", lambda b: b["disclosure"].update(statement="v2 confirms v1"), "disclosure"),
    ("protocol_v2_draft.json", lambda b: b["frozen_and_unchanged"].update(embargo_s=60), "frozen specification"),
    ("protocol_v2_draft.json", lambda b: b["preserves"].update(v1_formal_verdict="SIGNAL DETECTED"), "v1 formal verdict"),
    ("calibration_v2.json", lambda b: b["training_loss"].update(class_weight="balanced"), "unweighted"),
    ("calibration_v2.json", lambda b: b["nested_scheme"].update(final="fit calibrator on the outer validation block"), "nested"),
    ("calibration_v2.json", lambda b: b.update(candidate_calibrators=["none", "temperature"]), "calibrator list"),
    ("evaluation_v2.json", lambda b: b.update(configs_per_label_set=40), "budget"),
    ("evaluation_v2.json", lambda b: b["selection_rule"].update(never_used=["holdout"]), "economic"),
    ("evaluation_v2.json", lambda b: b["gates"]["directional_skill_gate_task_A"].pop("D4_robust_to_fingerprints"), "gate sets"),
    ("evaluation_v2.json", lambda b: b["uncertainty"]["block_bootstrap"].update(block_s=[600]), "uncertainty"),
    ("tasks_v2.json", lambda b: b["task_A_direction"]["abstention_policy"].update(in_gate=True), "unconditional"),
    ("tasks_v2.json", lambda b: b["task_B_opportunity"]["barrier_variants"].pop("F_fixed"), "barrier variants"),
    ("tasks_v2.json", lambda b: b["task_B_opportunity"].update(interpretation_constraint="directional skill"), "directional"),
    ("fingerprint_tests_v2.json", lambda b: b["tests"].pop("FP6_placebo"), "fingerprint tests"),
    ("fingerprint_tests_v2.json", lambda b: b["tests"]["FP3_regime_transfer"].update(c_reversed_diagnostic="train late, test early"), "non-chronological"),
    ("cost_v2.json", lambda b: b.update(no_profitability_claim=False), "cost section"),
])
def test_validator_detects_violations(tmp_path, name, mutate, needle):
    d = fresh(tmp_path)
    assert draft.validate_draft(d) == []
    edit(d, name, mutate)
    problems = draft.validate_draft(d)
    assert problems and any(needle in p for p in problems), problems


def test_validator_detects_unindexed_edits_activation_and_missing_files(tmp_path):
    d = fresh(tmp_path)
    p = d / "evaluation_v2.json"
    b = json.loads(p.read_text(encoding="utf-8"))
    b["n_model_configs"] = 99
    p.write_text(draft.canon(b), encoding="utf-8")
    assert any("hash differs" in x for x in draft.validate_draft(d))
    d2 = fresh(tmp_path / "x")
    (d2 / "protocol_v2_ACTIVATION.json").write_text("{}", encoding="utf-8")
    assert any("activation file exists" in x for x in draft.validate_draft(d2))
    draft.assert_active(d2)                                                            # only an activation file makes it usable
    d3 = fresh(tmp_path / "y")
    (d3 / "cost_v2.json").unlink()
    assert any("missing" in x for x in draft.validate_draft(d3))


# ============================ the mathematics the draft relies on ============================
def test_class_weighting_distorts_probabilities_prior_correction_undoes_it_and_temperature_cannot():
    """Exact (known-likelihood) demonstration of the v1 failure mode: with weights 1/(K*pi) the estimated posterior is p/pi (normalised)."""
    rng = np.random.default_rng(0)
    pi = np.array([0.25, 0.50, 0.25])
    n = 200_000
    y = rng.choice(3, size=n, p=pi)
    mu = np.array([-0.35, 0.0, 0.35])                                 # weak signal in one feature: x | y ~ N(mu_y, 1)
    x = rng.normal(mu[y], 1.0)
    lik = np.exp(-0.5 * (x[:, None] - mu[None, :]) ** 2)
    p = lik * pi
    p /= p.sum(axis=1, keepdims=True)                                 # true posterior (what an unweighted model targets)
    pw = lik / lik.sum(axis=1, keepdims=True)                         # class-weighted model: implied uniform prior

    def ll(q):
        return float(-np.mean(np.log(q[np.arange(n), y])))

    prior_ll = float(-np.mean(np.log(pi[y])))
    assert ll(p) < prior_ll < ll(pw)                                  # the informative posterior beats the prior; the weighted one is WORSE than the prior
    pc = pw * pi
    pc /= pc.sum(axis=1, keepdims=True)
    assert abs(ll(pc) - ll(p)) < 1e-9                                 # the Elkan correction recovers the posterior exactly
    best = min(ll((lambda z: z / z.sum(axis=1, keepdims=True))(pw ** (1 / T))) for T in np.linspace(0.3, 6, 80))
    assert best > prior_ll                                            # no temperature fixes it: a prior shift is not a scale error
    assert (np.argmax(p / pi, axis=1) == np.argmax(pw, axis=1)).all() # the balanced DECISION is a separate, prior-adjusted rule


def test_effective_sample_size_matches_theory():
    rng = np.random.default_rng(1)
    n = 40_000
    iid = rng.normal(size=n)
    assert 0.9 * n < effective_sample_size(iid) <= n
    rho = 0.8
    e = rng.normal(size=n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = rho * ar[i - 1] + e[i]
    theory = n * (1 - rho) / (1 + rho)                                # about n/9
    assert 0.6 * theory < effective_sample_size(ar, max_lag=120) < 1.6 * theory
    assert effective_sample_size(np.ones(100)) == 100 and effective_sample_size(np.array([1.0, 2.0])) == 2
    assert abs(1 / (1 + 2 * 0.45) - 0.526) < 0.01                     # v1 label ACF sum (lags 1..24) of 0.45 -> ESS factor ~0.53


def test_budget_arithmetic_and_stage_counts_are_consistent():
    ev = draft.evaluation()
    assert ev["n_model_configs"] == 7 and ev["configs_per_label_set"] == 21 <= ev["hyperparameter_budget_cap_per_label_set"]
    st = {s["id"]: s for s in draft.master()["stages"]}
    assert st["S1"]["experiments"] == 21 and st["S2"]["experiments"] == 63 == 3 * 21
