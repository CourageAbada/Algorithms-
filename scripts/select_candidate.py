"""Apply the PRE-REGISTERED selection rule, proxy checks, diagnostics, ablations and verdict (development folds only).

  python -m scripts.select_candidate
Reads stored experiments; runs the pre-registered controls / ablations through the same deterministic runner. The final holdout
is never read and no PnL is computed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from fxscalp.research import folds as F
from fxscalp.research.model_research import diagnostics as D
from fxscalp.research.model_research import protocol as P
from fxscalp.research.model_research import selection as S
from fxscalp.research.model_research.data import load_matrix
from fxscalp.research.model_research.experiment import ExperimentStore, Runner
from fxscalp.research.model_research.frozen import verify_frozen_state
from fxscalp.research.model_research.models import ModelPipeline

OUT = Path("research/phase2b/model_research")


def _default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, np.bool_)):
        return float(o) if isinstance(o, np.floating) else bool(o)
    return str(o)


def j(o: Any) -> Any:
    return json.loads(json.dumps(o, default=_default))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/research/matrix_v1")
    ap.add_argument("--store", default="data/research/experiments")
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args(argv)
    frozen = verify_frozen_state("data")
    df, man = load_matrix(Path(a.matrix))
    feat = json.loads(Path("research/phase2b/feature_set_v1.json").read_text(encoding="utf-8"))
    groups_of = D.feature_groups(feat)
    store = ExperimentStore(a.store)
    runner = Runner(df, man["matrix_id"], frozen, feat["model_input_allowed"], store, a.n_jobs)
    t0 = time.perf_counter()
    cands, base = S.split(S.load_results(store))
    table = S.evaluate(cands, base)
    print(f"{len(table)} complete candidates (of {P.protocol_manifest()['n_candidates_per_scenario']}); gate-passers 1-3: {int(table['gates_123'].sum())}")
    res: dict[str, Any] = {"protocol_hash": P.protocol_hash(), "frozen": {k: v for k, v in frozen.items() if k != "git"}, "git": frozen["git"],
                           "n_candidates": len(table), "failed_experiments": [p.parent.name for p in store.root.glob("*/failed.json")]}

    # ---- leading LightGBM hp for the controls (highest S among LightGBM candidates)
    lgb_rows = table[table["family"] == "lightgbm"]
    lead_lgb = lgb_rows.iloc[0]
    lgb_hp = {"family": "lightgbm", **lead_lgb["hp"]}
    spread_cols = tuple(c for c in feat["model_input_allowed"] if groups_of[c] == "spread")
    session_cols = tuple(c for c in D.SESSION_COLS if c in feat["model_input_allowed"])
    controls = {}
    for scn in P.SCENARIOS:
        for nm, cols in (("session_only", session_cols), ("spread_only", spread_cols)):
            eid = runner.run_models(scn, "CORE", [lgb_hp], kind="control", columns=cols, tag=f"control:{nm}", log=lambda *x: None)[0]
            controls.setdefault(nm, {})[scn] = store.load(eid)
    refs = S.baseline_refs(base)

    def proxy_check(row) -> dict[str, Any]:
        by_s = cands[row["key"]]
        lifts = {}
        flag_a = False
        for scn in P.SCENARIOS:
            cand_lift = float(S.fold_matrix(by_s[scn], "macro_f1").mean() - refs[scn]["best_baseline_macro_f1"].mean())
            ctl = {nm: float(S.fold_matrix(controls[nm][scn], "macro_f1").mean() - refs[scn]["best_baseline_macro_f1"].mean()) for nm in controls}
            lifts[scn] = {"candidate_lift": cand_lift, "controls": ctl}
            if cand_lift > 0 and any(v >= 0.8 * cand_lift for v in ctl.values()):
                flag_a = True
        shares = {}
        flag_b = False
        for scn in P.SCENARIOS:
            res_s = by_s[scn]
            tot, ss = 0.0, 0.0
            for fold in runner.folds:
                rec = next(f for f in res_s["folds"] if f["fold"] == fold.name)
                pipe = ModelPipeline.load(store.dir(res_s["experiment_id"]) / rec["model_file"])
                X, y, _ = D.fold_validation_matrix(runner, scn, row["feature_config"], fold, pipe.feature_names)
                gidx: dict[str, list[int]] = {}
                for k, n in enumerate(pipe.feature_names):
                    gidx.setdefault(groups_of.get(n, "regimes" if "regime" in n else "other"), []).append(k)
                pi = D.permutation_importance(pipe, X, y, gidx, [], seed=P.SEED, n_rows=15000, repeats=2)
                inc = {g: max(v["d_log_loss"], 0.0) for g, v in pi["groups"].items()}
                tot += sum(inc.values())
                ss += inc.get("session", 0.0) + inc.get("spread", 0.0)
            shares[scn] = ss / tot if tot > 0 else float("nan")
            flag_b |= bool(tot > 0 and ss / tot >= 0.5)
        return {"flag": bool(flag_a or flag_b), "a_control_lift_ge_80pct": flag_a, "b_session_spread_share_ge_50pct": flag_b,
                "control_lifts": lifts, "session_plus_spread_share_of_group_permutation_logloss": shares}

    # ---- sequential evaluation of the gate-passers in descending S
    passers = table[table["gates_123"]]
    proxy: dict[tuple, dict[str, Any]] = {}
    chosen = None
    candidates_after_proxy = []
    if len(passers):
        for _, row in passers.iterrows():
            proxy[row["key"]] = proxy_check(row)
            if not proxy[row["key"]]["flag"]:
                candidates_after_proxy.append(row)
                break
        if candidates_after_proxy:
            top = candidates_after_proxy[0]
            near = passers[passers["S"] >= top["S"] - S.TIE]
            ok = [top]
            for _, row in near.iterrows():
                if row["key"] == top["key"]:
                    continue
                proxy[row["key"]] = proxy.get(row["key"]) or proxy_check(row)
                if not proxy[row["key"]]["flag"]:
                    ok.append(row)
            chosen = min(ok, key=S.simplicity)
    res["selection"] = {"gate_passers_G1_G3": int(len(passers)), "proxy_checks": {str(k): v["flag"] for k, v in proxy.items()},
                        "chosen": None if chosen is None else {"feature_config": chosen["feature_config"], "family": chosen["family"], "hp": chosen["hp"], "S": float(chosen["S"])}}
    res["table_top20"] = j([{k: v for k, v in r.items() if k != "key"} for r in table.head(20).to_dict("records")])
    res["table_all"] = j([{k: v for k, v in r.items() if k != "key"} for r in table.to_dict("records")])
    res["controls"] = {nm: {s: {"macro_f1_mean": float(S.fold_matrix(v, 'macro_f1').mean()), "per_fold": S.fold_matrix(v, 'macro_f1').tolist(),
                                "log_loss_mean": float(S.fold_matrix(v, 'log_loss').mean())} for s, v in d.items()} for nm, d in controls.items()}
    res["proxy_details"] = j({str(k): v for k, v in proxy.items()})
    res["baseline_refs"] = j({s: {"best_baseline_macro_f1_per_fold": r["best_baseline_macro_f1"], "prior_log_loss_per_fold": r["prior_log_loss"],
                                  "best_baseline_name_per_fold": r["best_baseline_name"]} for s, r in refs.items()})
    exploratory = chosen is None
    if exploratory:
        # FORMAL outcome of protocol v1: no candidate passed the gates. Everything below this line is labelled EXPLORATORY and does not change it.
        res["verdict"] = {"verdict": "NO RELIABLE SIGNAL DETECTED", "reason": "no candidate passed the pre-registered gates (protocol v1)",
                          "note": "formal, pre-registered outcome; see 'exploratory' for the post-hoc analysis of WHY (class-weight probability shift vs gate G2)"}
        pool = table[table["G1"] & table["G3"]]
        chosen = pool.iloc[0] if len(pool) else table.iloc[0]
        if chosen["key"] not in proxy:
            proxy[chosen["key"]] = proxy_check(chosen)
        res["exploratory_focus"] = {"definition": "highest-S candidate passing G1 and G3 (G2 ignored) - NOT a pre-registered selection",
                                    "feature_config": chosen["feature_config"], "family": chosen["family"], "hp": chosen["hp"], "S": float(chosen["S"]),
                                    "proxy": j(proxy[chosen["key"]])}
    key = chosen["key"]
    by_s = cands[key]
    print("FORMAL CHOSEN:" if not exploratory else "EXPLORATORY FOCUS:", key[:3], f"S={chosen['S']:.4f}")

    # ---- uncertainty + verdict
    boot = S.bootstrap_vs_baseline(store, by_s, base)
    res["bootstrap"] = j(boot)
    if not exploratory:
        res["verdict"] = j(S.verdict(boot, proxy[key]["flag"]))
    else:
        boot_pc = S.bootstrap_vs_baseline(store, by_s, base, prior_correct=True)
        ev = S.verdict(boot_pc, proxy[key]["flag"])
        pc_ll = {}
        for scn in P.SCENARIOS:
            pri = [f["prior"] for f in base["majority"][scn]["folds"]]
            pred = store.predictions(by_s[scn]["experiment_id"])
            per = []
            for i, f in enumerate(sorted(pred["fold"].unique())):
                d = pred[pred["fold"] == f]
                from fxscalp.research.model_research import metrics as MM
                per.append(MM.log_loss(d["y"].to_numpy("int64"), S.prior_corrected(d[["u0", "u1", "u2"]].to_numpy("float64"), np.array(pri[i]))))
            pc_ll[scn] = {"prior_corrected_log_loss_per_fold": per, "prior_log_loss_per_fold": refs[scn]["prior_log_loss"].tolist(),
                          "folds_with_skill": int((np.array(per) < refs[scn]["prior_log_loss"]).sum())}
        res["exploratory"] = j({"label": "POST-HOC / EXPLORATORY - not part of the pre-registered selection or verdict",
                                "finding": "class-weighted models have prior-shifted probabilities; single-parameter temperature scaling cannot correct that, so gate G2 "
                                           "(log loss below the training prior) fails mechanically. Elkan prior correction is applied to the log-loss ONLY; decisions are the "
                                           "model's argmax decisions as pre-registered.",
                                "bootstrap_prior_corrected_logloss": boot_pc, "verdict_if_G2_used_prior_corrected_probabilities": ev,
                                "prior_corrected_logloss_by_scenario": pc_ll})
    from fxscalp.research.model_research import metrics as M
    yv = runner.y("C1_moderate")[runner.fold_rows("C1_moderate", runner.folds[2])["val"]]
    res["label_autocorrelation_NO_TRADE_indicator"] = j(M.label_autocorrelation(yv, P.GRID_STEP_S)[:24])
    # ---- ablations (pre-registered list) on the chosen candidate
    names = by_s["C0_spread_only"]["folds"][0]["feature_names"]
    hp = {"family": chosen["family"], **chosen["hp"]}
    abl: dict[str, Any] = {}
    for g in P.ABLATIONS:
        gname = D.GROUP_ALIASES.get(g, g)
        drop = [n for n in names if groups_of.get(n, "regimes" if "regime" in n else "other") == gname]
        if not drop:
            abl[g] = {"applicable": False, "note": f"group '{gname}' not in the chosen feature configuration"}
            continue
        keep = tuple(n for n in names if n not in drop)
        per = {}
        for scn in P.SCENARIOS:
            eid = runner.run_models(scn, chosen["feature_config"], [hp], kind="ablation", columns=keep, tag=f"ablate:{g}", log=lambda *x: None)[0]
            r = store.load(eid)
            full = S.fold_matrix(by_s[scn], "macro_f1")
            m = S.fold_matrix(r, "macro_f1")
            per[scn] = {"macro_f1": float(m.mean()), "delta_vs_full": float(m.mean() - full.mean()), "folds_worse": int((m < full).sum()),
                        "log_loss": float(S.fold_matrix(r, "log_loss").mean()), "delta_log_loss": float(S.fold_matrix(r, "log_loss").mean() - S.fold_matrix(by_s[scn], "log_loss").mean())}
        abl[g] = {"applicable": True, "n_features_removed": len(drop), "removed_sample": drop[:6], "by_scenario": per}
    res["ablations"] = abl
    print(f"selection+ablations done {time.perf_counter() - t0:.0f}s")
    (OUT / "selection.json").write_text(json.dumps(res, indent=1, default=_default), encoding="utf-8")

    # ---- diagnostics for serious candidates: chosen, best logistic, best LightGBM (scenario C1 = middle; stability across folds)
    serious = {"chosen": chosen, "best_logistic": table[table["family"] == "logistic"].iloc[0], "best_lightgbm": lead_lgb}
    diag: dict[str, Any] = {}
    for nm, row in serious.items():
        key2 = row["key"]
        scn = "C1_moderate"
        res_s = cands[key2][scn]
        per_fold_imp, per_fold_perm, group_perm = [], [], []
        for fold in runner.folds:
            rec = next(f for f in res_s["folds"] if f["fold"] == fold.name)
            pipe = ModelPipeline.load(store.dir(res_s["experiment_id"]) / rec["model_file"])
            X, y, _ = D.fold_validation_matrix(runner, scn, row["feature_config"], fold, pipe.feature_names)
            imp = pipe.importance()
            top = [pipe.feature_names.index(n) for n, _ in sorted(imp.items(), key=lambda kv: -kv[1])[:15]]
            gidx: dict[str, list[int]] = {}
            for k, n in enumerate(pipe.feature_names):
                gidx.setdefault(groups_of.get(n, "regimes" if "regime" in n else "other"), []).append(k)
            pi = D.permutation_importance(pipe, X, y, gidx, top, seed=P.SEED, n_rows=40000, repeats=3)
            per_fold_imp.append(imp)
            per_fold_perm.append(pi)
        names2 = list(per_fold_imp[0])
        mat = np.array([[imp[n] for n in names2] for imp in per_fold_imp])
        sp = [D.spearman(mat[a_], mat[b_]) for a_ in range(5) for b_ in range(a_ + 1, 5)]
        top_sets = [set(np.array(names2)[np.argsort(-mat[f])[:10]]) for f in range(5)]
        overlap = float(np.mean([len(top_sets[a_] & top_sets[b_]) / 10 for a_ in range(5) for b_ in range(a_ + 1, 5)]))
        gp = {g: [pf["groups"][g]["d_log_loss"] for pf in per_fold_perm] for g in per_fold_perm[0]["groups"]}
        fp: dict[str, list[float]] = {}
        for pf in per_fold_perm:
            for n, v in pf["features"].items():
                fp.setdefault(n, []).append(v["d_log_loss"])
        diag[nm] = {"feature_config": row["feature_config"], "family": row["family"], "hp": row["hp"], "scenario": scn,
                    "builtin_importance_top15_mean": dict(sorted({n: float(mat[:, k].mean()) for k, n in enumerate(names2)}.items(), key=lambda kv: -kv[1])[:15]),
                    "importance_rank_spearman_between_folds": {"mean": float(np.nanmean(sp)), "min": float(np.nanmin(sp))},
                    "top10_overlap_between_folds": overlap,
                    "group_permutation_d_logloss_mean_by_fold": {g: [round(x, 5) for x in v] for g, v in gp.items()},
                    "group_permutation_d_logloss_mean": {g: float(np.mean(v)) for g, v in sorted(gp.items(), key=lambda kv: -np.mean(kv[1]))},
                    "feature_permutation_d_logloss_mean_top": dict(sorted({n: float(np.mean(v)) for n, v in fp.items()}.items(), key=lambda kv: -kv[1])[:15]),
                    "feed_split_f5": {k: {kk: vv for kk, vv in v.items() if kk in ("n", "macro_f1", "balanced_accuracy", "log_loss", "ece_top", "pred_freq", "class_dist")}
                                      for k, v in next(f for f in res_s["folds"] if f["fold"] == "F5")["feed_split"].items()},
                    "drift_and_missingness": D.drift_and_missingness(runner, scn, row["feature_config"], pipe.feature_names)}
        print(f"diagnostics {nm} done {time.perf_counter() - t0:.0f}s", flush=True)
    diag["month_identifiability"] = D.month_identifiability(df, feat["model_input_allowed"])
    (OUT / "diagnostics.json").write_text(json.dumps(j(diag), indent=1), encoding="utf-8")
    print("done", f"{time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
