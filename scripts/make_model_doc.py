"""Generate docs/PHASE2B_MODEL_RESEARCH.md from the stored development-fold evidence (tables are produced from the JSON/results, never typed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fxscalp.research.model_research import metrics as M
from fxscalp.research.model_research import protocol as P
from fxscalp.research.model_research import selection as S
from fxscalp.research.model_research.experiment import ExperimentStore

R = Path("research/phase2b/model_research")
F5 = lambda x: f"{x:.4f}"  # noqa: E731
F3 = lambda x: f"{x:.3f}"  # noqa: E731


def hp_s(h: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in h.items())


def main() -> int:
    store = ExperimentStore("data/research/experiments")
    sel = json.loads((R / "selection.json").read_text(encoding="utf-8"))
    diag = json.loads((R / "diagnostics.json").read_text(encoding="utf-8")) if (R / "diagnostics.json").exists() else {}
    cand_file = json.loads((R / "MODEL_CANDIDATE_V1.json").read_text(encoding="utf-8")) if (R / "MODEL_CANDIDATE_V1.json").exists() else {}
    proto = json.loads((R / "protocol_v1.json").read_text(encoding="utf-8"))
    results = S.load_results(store)
    cands, base = S.split(results)
    table = sel["table_all"]
    exploratory = "exploratory_focus" in sel
    focus_info = sel["exploratory_focus"] if exploratory else sel["selection"]["chosen"]
    key_fc = (focus_info["feature_config"], focus_info["family"], tuple(sorted(focus_info["hp"].items())))
    fkey = next(k for k in cands if k[0] == key_fc[0] and k[1] == key_fc[1] and dict(k[2]) == focus_info["hp"] and k[3] is None)
    by_s = cands[fkey]
    L: list[str] = []
    add = L.append

    # ---- baselines
    base_rows = []
    for s in P.SCENARIOS:
        for b in ("majority", "stratified_random", "momentum", "mean_reversion"):
            r = base[b][s]
            f1, ba, ac = (S.fold_matrix(r, k) for k in ("macro_f1", "balanced_accuracy", "accuracy"))
            ll = S.fold_matrix(r, "log_loss") if "log_loss" in r["folds"][0]["calibrated"] else None
            pf = np.mean([f["calibrated"]["pred_freq"] for f in r["folds"]], axis=0)
            base_rows.append(f"| {s} | {b} | {F3(f1.mean())} ± {F3(f1.std(ddof=1))} | {F3(ba.mean())} | {F3(ac.mean())} | {'-' if ll is None else F3(ll.mean())} | "
                             f"{pf[0]:.2f} / {pf[1]:.2f} / {pf[2]:.2f} |")
    # ---- leaderboard
    lb = []
    for r in table:
        lb.append(f"| {r['feature_config']} | {r['family']} | {hp_s(r['hp'])} | {F3(r['M_macro_f1'])} | {F3(r['D_fold_sd'])} | {F3(r['E_ece'])} | {F3(r['S'])} | "
                  f"{F3(r['mean_balanced_acc'])} | {F3(r['mean_log_loss'])} | {r['G1_folds_beating_baseline']}/5 | {r['G2_folds_with_skill']}/5 | {F3(r['G3_min_class_recall'])} | "
                  f"{'yes' if r['gates_123'] else 'no'} |")
    # ---- focus per-fold
    fold_rows = []
    for s in P.SCENARIOS:
        r = by_s[s]
        for f in r["folds"]:
            c, u = f["calibrated"], f["uncalibrated"]
            bb = max((base[b][s]["folds"][int(f["fold"][1]) - 1]["calibrated"]["macro_f1"], b) for b in base)
            fold_rows.append(f"| {s} | {f['fold']} | {f['n_train']:,} | {f['n_val']:,} | {F3(c['macro_f1'])} | {F3(bb[0])} ({bb[1]}) | {F3(c['balanced_accuracy'])} | "
                             f"{F3(u['log_loss'])} → {F3(c['log_loss'])} | {F3(f['calibrated']['brier'])} | {F3(u['ece_top'])} → {F3(c['ece_top'])} | {f['temperature']:.2f} | "
                             f"{f['duration_s']:.0f} |")
    # ---- confusion matrices pooled
    conf_blocks = []
    for s in P.SCENARIOS:
        cm = sum(np.array(f["calibrated"]["confusion"]) for f in by_s[s]["folds"])
        m = M.from_confusion(cm)
        conf_blocks.append(f"**{s}** (pooled over folds; rows = true, columns = predicted: SHORT / NO_TRADE / LONG)\n\n| true \\ pred | SHORT | NO_TRADE | LONG | recall |\n|---|---|---|---|---|\n" +
                           "\n".join(f"| {n} | {cm[i][0]:,} | {cm[i][1]:,} | {cm[i][2]:,} | {F3(m['recall'][i])} |" for i, n in enumerate(P.CLASS_NAMES)) +
                           f"\n| precision | {F3(m['precision'][0])} | {F3(m['precision'][1])} | {F3(m['precision'][2])} | |\n\nF1 per class: SHORT {F3(m['f1'][0])}, NO_TRADE {F3(m['f1'][1])}, LONG {F3(m['f1'][2])}; "
                           f"macro-F1 {F3(m['macro_f1'])}, balanced accuracy {F3(m['balanced_accuracy'])}, accuracy {F3(m['accuracy'])}. Predicted-class frequency "
                           f"{m['pred_freq'][0]:.3f} / {m['pred_freq'][1]:.3f} / {m['pred_freq'][2]:.3f} vs actual {m['class_dist'][0]:.3f} / {m['class_dist'][1]:.3f} / {m['class_dist'][2]:.3f}.")
    # ---- pooled calibration from predictions
    cal_rows, rel_rows = [], []
    import pandas as pd
    for s in P.SCENARIOS:
        pr = store.predictions(by_s[s]["experiment_id"])
        y = pr["y"].to_numpy("int64")
        for nm, pref in (("uncalibrated", "u"), ("temperature-calibrated", "c")):
            p = pr[[f"{pref}{k}" for k in range(3)]].to_numpy("float64")
            e, tab = M.ece_top(y, p)
            cal_rows.append(f"| {s} | {nm} | {F3(M.log_loss(y, p))} | {F3(M.brier(y, p))} | {F3(e)} |")
            if pref == "c":
                rel_rows.append(f"**{s}** top-label reliability (equal-mass bins, mean confidence → accuracy): " +
                                "; ".join(f"{t['mean_confidence']:.2f}→{t['accuracy']:.2f}" for t in tab[::2]))
    # ---- bootstrap
    boot = sel.get("bootstrap", {})
    bt = []
    for s, b in boot.items():
        d1, d2 = b["macro_f1_vs_best_baseline"], b["log_loss_vs_prior"]
        bt.append(f"| {s} | {b['best_baseline']} | {F3(b['pooled_candidate']['macro_f1'])} | {F3(b['pooled_best_baseline']['macro_f1'])} | "
                  f"{d1['diff']['mean']:+.3f} [{d1['diff']['lo95']:+.3f}, {d1['diff']['hi95']:+.3f}] | {F3(b['pooled_candidate']['log_loss'])} / {F3(b['pooled_prior']['log_loss'])} | "
                  f"{d2['diff']['mean']:+.3f} [{d2['diff']['lo95']:+.3f}, {d2['diff']['hi95']:+.3f}] | {b['folds_beating_best_baseline']}/5 |")
    ex = sel.get("exploratory", {})
    ex_bt = []
    for s, b in ex.get("bootstrap_prior_corrected_logloss", {}).items():
        d2 = b["log_loss_vs_prior"]
        pc = ex["prior_corrected_logloss_by_scenario"][s]
        ex_bt.append(f"| {s} | {F3(b['pooled_candidate']['log_loss'])} / {F3(b['pooled_prior']['log_loss'])} | {d2['diff']['mean']:+.4f} [{d2['diff']['lo95']:+.4f}, {d2['diff']['hi95']:+.4f}] | {pc['folds_with_skill']}/5 |")
    # ---- ablations
    ab = []
    for g, v in sel.get("ablations", {}).items():
        if not v.get("applicable"):
            ab.append(f"| remove `{g}` | - | not in the focus configuration | | |")
            continue
        for s, d in v["by_scenario"].items():
            ab.append(f"| remove `{g}` ({v['n_features_removed']} features) | {s} | {F3(d['macro_f1'])} | {d['delta_vs_full']:+.4f} | {d['folds_worse']}/5 | {d['delta_log_loss']:+.4f} |")
    ctl = []
    for nm, d in sel.get("controls", {}).items():
        for s, v in d.items():
            ctl.append(f"| {nm} | {s} | {F3(v['macro_f1_mean'])} | {F3(v['log_loss_mean'])} |")
    comp = {}
    for r in results:
        k = (r["config"]["kind"], r["config"]["family"], r["config"]["feature_config"])
        comp.setdefault(k, []).append((r["total_duration_s"], r["peak_rss_mb"]))
    comp_rows = [f"| {k[0]} | {k[1]} | {k[2]} | {len(v)} | {np.mean([x[0] for x in v]):.0f} | {max(x[1] for x in v):.0f} |" for k, v in sorted(comp.items())]

    # ---------------------------------------------------------------- document
    add("# Phase 2B model research - development-fold evidence (final holdout LOCKED)\n")
    add(f"**Verdict (pre-registered rule, development evidence only): {sel['verdict']['verdict']}.** "
        f"All numbers below are *predictive classification* performance on the five frozen walk-forward development folds. There is **no economic/trading evaluation**: "
        f"no PnL, Sharpe or profit factor was computed, costs C1/C2 are hypothetical, and **profitability is UNVERIFIED**. The final holdout (2026-09-14..2026-10-06) was not read, "
        f"scored or inspected. No order was placed and nothing was connected to MT5 execution.\n")
    add(f"Frozen specification: spec_hash `{sel['frozen']['spec_hash']}`, dataset freeze `{sel['frozen']['raw_dataset_freeze']}`, protocol `{sel['protocol_hash']}` "
        f"(committed before any model was trained, git `8c18f55`); experiments ran at git `{sel['git']['commit'][:7]}`.\n")
    add("## 1. Experiment protocol\n")
    add(f"Primary label `tb_H60` (triple barrier, 60 s) under the three frozen cost scenarios C0/C1/C2, each trained and scored separately. Decision grid: bar ends on a "
        f"{P.GRID_STEP_S} s UTC grid (1/{P.GRID_STEP_S} of the 1-s rows, 297,040 rows over 115 development days). Folds: the five frozen expanding folds with the frozen purge/embargo; "
        f"within each training block the last 20% (by time) is held out for temperature calibration (the model is fit on the earlier 80% after an extra 1,800 s embargo), and the same 80%-model "
        f"is reported uncalibrated and calibrated. Preprocessing (logistic: winsorisation at 0.1/99.9%, median/IQR scaling, median imputation) and REGIME thresholds are fit on the training block only. "
        f"Decision rule: argmax of calibrated probabilities; no threshold was optimised. Experiment ids are SHA-256 hashes of (freeze, feature manifest, label + scenario, folds, model, hyper-parameters, seed, "
        f"preprocessing, calibration, grid, protocol, matrix id); experiments are never overwritten. Dependence: 60 s labels on a 30 s grid overlap by 50%, so uncertainty uses a paired **block bootstrap "
        f"(600 s blocks = 10 x the horizon, B=1000, resampling within fold)**, plus a non-overlapping-60 s robustness evaluation stored per fold.\n")
    add("## 2. Exact models and predetermined search spaces (final; never expanded)\n")
    add(f"- A. Multinomial logistic regression (L2, lbfgs, max_iter 300): C ∈ {proto['models']['logistic']['search']['C']} × class_weight ∈ {proto['models']['logistic']['search']['class_weight']} (6 configs).\n"
        f"- B. LightGBM gradient boosting: num_leaves ∈ {proto['models']['lightgbm']['search']['num_leaves']} × n_estimators ∈ {proto['models']['lightgbm']['search']['n_estimators']} (4 configs); fixed "
        f"learning_rate 0.05, min_child_samples 500, subsample 0.7, colsample 0.7, reg_lambda 10, class_weight balanced.\n"
        f"- C. A third family was **not** used (no justification beyond a first causal-signal test).\n"
        f"- Feature configurations: CORE (108), CONDITIONAL (108 + 8 gated session aggregates), REGIME (108 + 2 regime codes with thresholds fit per fold on training rows) - 10 × 3 = 30 candidates per scenario, "
        f"90 model experiments + 12 baseline experiments, seed {P.SEED}. Undocumented flag bits, raw volume and `last` are never used.\n")
    add("## 3. Selection rule (written before training)\n")
    add("```\n" + json.dumps(proto["selection_rule"], indent=1) + "\n```\n")
    add("## 4. Baselines (all development folds)\n")
    add("| scenario | baseline | macro-F1 (mean ± SD over folds) | balanced acc | accuracy | log loss | predicted-class frequency SHORT / NO_TRADE / LONG |\n|---|---|---|---|---|---|---|")
    L.extend(base_rows)
    add("\nThe naive bar a model must beat is the best baseline's macro-F1 (stratified random ≈ 0.334, i.e. 1/3 for a 3-class problem); momentum and mean-reversion are at chance (balanced accuracy ≈ 0.33), "
        "so 60-second trailing return alone carries no directional information for `tb_H60`.\n")
    add("## 5. All candidates (mean over the 3 scenarios and 5 folds; calibrated probabilities)\n")
    add("S = M − D − E (M mean macro-F1, D mean fold SD, E mean top-label ECE). G1: folds beating the best baseline's macro-F1; G2: folds with log loss below the training-prior log loss; G3: smallest mean class recall.\n")
    add("| features | model | hyper-parameters | M | D | E | S | bal acc | log loss | G1 | G2 | G3 | passes G1-G3 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    L.extend(lb)
    add(f"\nPer-fold results of **every** experiment (uncalibrated and calibrated) are in `research/phase2b/model_research/all_fold_results.csv`.\n")
    add("## 6. Selection outcome\n")
    add(f"Candidates passing G1-G3: {sel['selection'].get('gate_passers_G1_G3', 0)} of {sel['n_candidates']}. "
        + ("**No candidate passed the pre-registered gates, so the formal outcome is NO CANDIDATE.** " if exploratory else
           f"Chosen: {focus_info['feature_config']} / {focus_info['family']} / {hp_s(focus_info['hp'])} (S={focus_info['S']:.4f}). "))
    if exploratory:
        add(f"\n**Why (diagnosed after the fact, reported transparently).** Every LightGBM and every class-balanced logistic candidate beats the baselines on macro-F1 in all five folds, but fails G2 "
            f"because a class-weighted model's raw probabilities are prior-shifted: its log loss (~1.07) is worse than the training prior's (0.957), and single-parameter temperature scaling cannot undo a prior "
            f"shift. The unweighted logistic models have genuine probabilistic skill (log loss 0.945 < 0.957) but essentially never predict LONG or SHORT (recall < 0.01), so they fail G1/G3. "
            f"This is a design defect of protocol v1 (class weighting + prior-based log-loss gate + temperature-only calibration), **not** evidence about the data. I did **not** change the rule. "
            f"Everything in sections 7-12 describes the **exploratory focus candidate** - the highest-S candidate passing G1 and G3 with G2 ignored - and is labelled EXPLORATORY; it does not alter the formal outcome.\n")
    add(f"Focus candidate: **{focus_info['feature_config']} / {focus_info['family']} / {hp_s(focus_info['hp'])}**; proxy check flag = {(sel.get('exploratory_focus') or {}).get('proxy', {}).get('flag', 'n/a')}.\n")
    add("## 7. Development-fold results of the focus candidate (calibrated unless stated)\n")
    add("| scenario | fold | n train | n val | macro-F1 | best baseline macro-F1 | balanced acc | log loss (uncal → cal) | Brier | ECE (uncal → cal) | T | fit+predict s |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    L.extend(fold_rows)
    add("\n## 8. Aggregate development performance with uncertainty (paired block bootstrap, 95%)\n")
    add("| scenario | best baseline | candidate macro-F1 | baseline macro-F1 | Δ macro-F1 [95% CI] | log loss cand / prior | Δ log loss vs prior [95% CI] (negative = better) | folds beating baseline |\n|---|---|---|---|---|---|---|---|")
    L.extend(bt)
    if exploratory and ex_bt:
        add("\n**Exploratory: probabilities after prior-shift correction (decisions unchanged).**\n")
        add("| scenario | log loss cand / prior | Δ log loss vs prior [95% CI] | folds with skill |\n|---|---|---|---|")
        L.extend(ex_bt)
        add(f"\nExploratory verdict if G2 used prior-corrected probabilities (NOT the pre-registered verdict): **{ex['verdict_if_G2_used_prior_corrected_probabilities']['verdict']}**.\n")
    nov = []
    for sc in P.SCENARIOS:
        c_ = np.mean([f["nonoverlap_60s"]["macro_f1"] for f in by_s[sc]["folds"]])
        b_ = np.mean([f["nonoverlap_60s"]["macro_f1"] for f in base["stratified_random"][sc]["folds"]])
        cm_ = sum(np.array(f["calibrated"]["confusion"]) for f in by_s[sc]["folds"])
        hit, miss = cm_[0][0] + cm_[2][2], cm_[0][2] + cm_[2][0]
        # direction-only: among rows whose TRUE class is LONG/SHORT and whose prediction is LONG/SHORT
        nov.append(f"| {sc} | {F3(c_)} | {F3(b_)} | {c_ - b_:+.3f} | {hit / max(hit + miss, 1):.3f} | {M.from_confusion(cm_)['precision'][2]:.3f} / {M.from_confusion(cm_)['class_dist'][2]:.3f} |")
    add("\n**Robustness: non-overlapping 60 s evaluation, and direction-only accuracy.**\n")
    add("| scenario | candidate macro-F1 (non-overlapping 60 s rows) | stratified-random macro-F1 | delta | direction accuracy among true-LONG/SHORT rows predicted LONG/SHORT (0.5 = no directional information) | LONG precision / LONG base rate |\n|---|---|---|---|---|---|")
    L.extend(nov)
    add("\n**What the signal is.** Among rows whose true class is LONG or SHORT and that the model also calls LONG or SHORT, the call is right only ~50% of the time "
        "(0.503 / 0.503 / 0.504): **the model carries no directional information.** LONG precision (0.26 / 0.24 / 0.20) is barely above the LONG base rate (0.24 / 0.21 / 0.16). "
        "The macro-F1 lift over a random classifier comes from separating NO_TRADE rows from trade rows, i.e. predicting whether a 1-sigma barrier will be reached within 60 s; "
        "this matches the importance ranking (volatility level: atr14_bps_*, realized_vol_bps_*, higher-timeframe ranges) and the small effect of removing any single feature group. "
        "It is volatility/timeout predictability, not a directional edge, and because the barrier itself is scaled by trailing volatility part of it may be a property of the label construction.\n")
    add("\n## 9. Calibration, confusion matrices, class-specific performance\n")
    add("| scenario | probabilities | log loss | Brier | top-label ECE |\n|---|---|---|---|---|")
    L.extend(cal_rows)
    add("\n" + "\n\n".join(rel_rows) + "\n")
    add("\n\n".join(conf_blocks) + "\n")
    add("NO_TRADE is first-class: its precision/recall/F1 are reported above separately from LONG/SHORT; no PnL-optimised threshold was used. Pre-registered selectivity study "
        "(max-probability thresholds 0.40/0.50/0.60) is stored per fold in `result.json` (descriptive).\n")
    if diag:
        add("## 10. Feature importance, stability, drift, proxies\n")
        for nm in ("chosen", "best_logistic", "best_lightgbm"):
            if nm in diag:
                d = diag[nm]
                add(f"**{nm}** ({d['feature_config']} / {d['family']} / {hp_s(d['hp'])}, scenario {d['scenario']}): group permutation Δlog-loss "
                    + ", ".join(f"{g} {v:+.4f}" for g, v in list(d["group_permutation_d_logloss_mean"].items())[:8])
                    + f"; top built-in importance: " + ", ".join(f"`{k}` {v:.3f}" for k, v in list(d["builtin_importance_top15_mean"].items())[:8])
                    + f"; rank-Spearman between folds mean {d['importance_rank_spearman_between_folds']['mean']:.2f} (min {d['importance_rank_spearman_between_folds']['min']:.2f}), "
                      f"top-10 overlap {d['top10_overlap_between_folds']:.2f}. Top permutation features: "
                    + ", ".join(f"`{k}` {v:+.4f}" for k, v in list(d["feature_permutation_d_logloss_mean_top"].items())[:8]) + ".\n")
                dm = d["drift_and_missingness"]
                add(f"Drift (PSI train→validation, mean over folds): {dm['features_mean_psi_gt_0.10']} features > 0.10, {dm['features_mean_psi_gt_0.25']} > 0.25; top: "
                    + ", ".join(f"`{k}` {v}" for k, v in list(dm["mean_psi_top15"].items())[:6]) + f". Feed boundary (pre 08-10..09-04 vs post 09-07..09-11): {dm['feed_boundary_features_psi_gt_0.25']} features with PSI > 0.25; top "
                    + ", ".join(f"`{k}` {v}" for k, v in list(dm["feed_boundary_psi_pre_vs_post_top10"].items())[:5]) + ". Fold F5 pre vs post-boundary rows: "
                    + "; ".join(f"{k}: n={v['n']:,} macro-F1 {v['macro_f1']:.3f} log loss {v.get('log_loss', float('nan')):.3f}" for k, v in d["feed_split_f5"].items()) + ".\n")
        if "month_identifiability" in diag:
            mi = diag["month_identifiability"]
            add(f"**Month-identifiability probe** (CORE features → calendar month, day-grouped 5-fold CV): accuracy {mi['cv_accuracy_mean']:.3f} vs majority {mi['majority_rate']:.3f} / uniform {mi['chance_uniform']:.3f}; "
                f"top month-identifying features: {', '.join('`' + f + '`' for f in mi['top_month_identifying_features'][:6])}.\n")
    add("## 11. Proxy controls and ablations (focus candidate)\n")
    add("| control model (LightGBM, leading hyper-parameters) | scenario | macro-F1 | log loss |\n|---|---|---|---|")
    L.extend(ctl)
    pd_ = (sel.get("exploratory_focus") or {}).get("proxy") or sel["proxy_details"].get(str(fkey), {})
    if pd_:
        add(f"\nProxy rule: (a) control lift ≥ 80% of candidate lift: **{pd_.get('a_control_lift_ge_80pct')}**; (b) session+spread ≥ 50% of group-permutation Δ log-loss: **{pd_.get('b_session_spread_share_ge_50pct')}** "
            f"(shares {pd_.get('session_plus_spread_share_of_group_permutation_logloss')}). Flag: **{pd_.get('flag')}**.\n")
    add("\n| ablation (pre-registered list) | scenario | macro-F1 | Δ vs full | folds worse | Δ log loss |\n|---|---|---|---|---|---|")
    L.extend(ab)
    add("\n## 12. Computational performance\n")
    add("| kind | model | features | experiments | mean total fit+predict s (5 folds) | peak RSS MB |\n|---|---|---|---|---|---|")
    L.extend(comp_rows)
    add("\nMatrix build: 115 days × ~3.4 s; grid: three scenario processes in parallel (2 threads each), ≈ 2 h wall-clock.\n")
    add("## 13. Failures and warnings\n")
    add(f"Failed experiments: {len(sel.get('failed_experiments', []))}. The first sequential grid run was stopped after one fold (too slow) and restarted as three parallel scenario processes with identical settings "
        "(10 incomplete directories removed; no completed experiment was touched). The matrix manifest step failed once on a JSON-serialization bug (fixed; day files unaffected). "
        "scikit-learn emitted lbfgs ConvergenceWarnings for some unregularised-ish logistic fits (max_iter 300, tol 1e-3, as pre-registered); predictions from those fits were used as produced.\n")
    add("## 14. Limitations\n")
    add("1. Predictive classification only: macro-F1 above the stratified-random baseline says nothing about tradable edge; costs C1/C2 are hypothetical and commission, slippage and latency are unknown.\n"
        "2. 6 months, one DEMO feed, summer time only, 115 development days; the post-feed-boundary regime has 5 development days.\n"
        "3. The 30 s grid keeps 50% overlap between neighbouring 60 s labels; the block bootstrap handles dependence but the effective sample size remains far below the row count.\n"
        "4. Protocol v1 gate G2 is mechanically incompatible with class-weighted probability outputs (section 6); this was found after seeing results and is reported, not repaired.\n"
        "5. Single seed (7); LightGBM thread count fixed at 2 for every grid experiment.\n"
        "6. Permutation importance breaks feature correlations; it indicates dependence, not causation.\n")
    add("## 15. Final gate\n")
    add(f"**{sel['verdict']['verdict']}** (pre-registered rule, development evidence only).\n")
    out = Path("docs/PHASE2B_MODEL_RESEARCH.md")
    out.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(L)} blocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
