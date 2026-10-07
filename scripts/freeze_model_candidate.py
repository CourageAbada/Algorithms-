"""Freeze MODEL_CANDIDATE_V1: fit the chosen configuration ONCE on the whole DEVELOPMENT window and write a hashed manifest.

  python -m scripts.freeze_model_candidate
Uses development rows only (the matrix loader refuses holdout rows). The final holdout is NOT evaluated here: no prediction is made
on any row at or after 2026-09-14, and no metric is computed other than a reproducibility checksum on a fixed development subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from fxscalp.core.provenance import sha256_file
from fxscalp.research import folds as F
from fxscalp.research.model_research import protocol as P
from fxscalp.research.model_research.data import load_matrix
from fxscalp.research.model_research.experiment import ExperimentStore, Runner, env_info
from fxscalp.research.model_research.frozen import verify_frozen_state
from fxscalp.research.model_research.models import ModelPipeline, feature_matrix, fit_regime_config, fit_temperature

OUT = Path("research/phase2b/model_research")


def final_pseudo_fold() -> F.FoldSpec:
    """The whole development window as ONE training block (validation fields point at the holdout dates but no holdout data is read)."""
    return F.FoldSpec("FINAL", F.FIRST_ELIGIBLE_DAY.isoformat(), F.DEV_LAST_DAY.isoformat(), F.HOLDOUT_FIRST_DAY.isoformat(), F.HOLDOUT_LAST_DAY.isoformat())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/research/matrix_v1")
    ap.add_argument("--store", default="data/research/experiments")
    ap.add_argument("--out-models", default="data/research/model_candidate_v1")
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args(argv)
    frozen = verify_frozen_state("data")
    sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    ch = sel["selection"]["chosen"]
    if ch is None:
        (OUT / "MODEL_CANDIDATE_V1.json").write_text(json.dumps({"model_candidate": "NONE", "reason": "no candidate passed the pre-registered gates",
                                                                 "protocol_hash": P.protocol_hash()}, indent=1), encoding="utf-8")
        print("no candidate: manifest records NONE")
        return 0
    df, man = load_matrix(Path(a.matrix))
    feat = json.loads(Path("research/phase2b/feature_set_v1.json").read_text(encoding="utf-8"))
    runner = Runner(df, man["matrix_id"], frozen, feat["model_input_allowed"], ExperimentStore(a.store), a.n_jobs)
    fold = final_pseudo_fold()
    hp = {k: v for k, v in ch["hp"].items()}
    models = {}
    out_dir = Path(a.out_models)
    for scn in P.SCENARIOS:
        ts, le, st = df["ts"].to_numpy(), df[f"le_{scn}"].to_numpy(), df[f"st_{scn}"].to_numpy()
        tr = (st == 0) & F.train_mask(fold, ts, le)
        idx = np.flatnonzero(tr)
        assert ts[idx].max() < fold.train_end_ms and ts[idx].max() < F.FinalHoldout().start_ms
        t_c = np.quantile(ts[idx], 0.8)
        fit = np.flatnonzero(tr & (ts < t_c) & (le <= t_c - F.EMBARGO_S * 1000))
        cal = np.flatnonzero(tr & (ts >= t_c))
        tr_df = df.iloc[idx]
        regime = None
        if ch["feature_config"] == "REGIME":
            regime = fit_regime_config({k: tr_df[k].to_numpy("float64") for k in ("spread_points", "session_code", "realized_vol_bps_60s")}, tr_df["ts"].to_numpy(), fold)
        X, names = feature_matrix(tr_df, ch["feature_config"], feat["model_input_allowed"], regime)
        pos = {g: k for k, g in enumerate(idx)}
        fp, cp = np.array([pos[g] for g in fit]), np.array([pos[g] for g in cal])
        y = runner.y(scn)[idx]
        pipe = ModelPipeline(ch["family"], hp, names, P.SEED, a.n_jobs).fit_preprocessing(X).fit(X[fp], y[fp])
        pipe.temperature = fit_temperature(pipe.predict_proba(X[cp]), y[cp])
        path = out_dir / scn / "model.joblib"
        pipe.save(path)
        # reproducibility checksum on a FIXED development subset (first 4000 training rows): reload must reproduce it exactly
        chk = hashlib.sha256(np.round(pipe.predict_calibrated(X[:4000]), 6).tobytes()).hexdigest()
        again = hashlib.sha256(np.round(ModelPipeline.load(path).predict_calibrated(X[:4000]), 6).tobytes()).hexdigest()
        assert chk == again, "serialization round-trip changed predictions"
        models[scn] = {"artifact": str(path), "artifact_sha256": sha256_file(path), "temperature": pipe.temperature, "n_fit": int(len(fp)),
                       "n_cal": int(len(cp)), "train_first_ts": int(ts[idx].min()), "train_last_ts": int(ts[idx].max()),
                       "regime_config": regime.to_dict() if regime else None, "reproducibility_checksum_dev_subset": chk}
        print(f"{scn}: fit {len(fp):,} cal {len(cp):,} T={pipe.temperature:.3f} sha {models[scn]['artifact_sha256'][:12]}", flush=True)
    body = {
        "name": "MODEL_CANDIDATE_V1", "status": "FROZEN - final holdout NOT evaluated", "protocol_hash": P.protocol_hash(),
        "spec_hash": frozen["spec_hash"], "raw_dataset_freeze": frozen["raw_dataset_freeze"], "primary_label": P.PRIMARY_LABEL,
        "model_family": ch["family"], "hyperparameters": hp, "seed": P.SEED, "feature_configuration": ch["feature_config"],
        "features": names, "n_features": len(names),
        "preprocessing": P.PREPROCESSING[ch["family"] if ch["family"] in P.PREPROCESSING else "logistic"],
        "regime_configuration": ("thresholds = reference quantiles fitted on the TRAINING block only (per session code); fitted values stored per scenario below"
                                 if ch["feature_config"] == "REGIME" else "not used"),
        "probability_calibration": P.CALIBRATION, "decision_policy": P.DECISION_POLICY,
        "training_window": {"first_day": F.FIRST_ELIGIBLE_DAY.isoformat(), "last_day": F.DEV_LAST_DAY.isoformat(), "embargo_s": F.EMBARGO_S,
                            "note": "trained once on the whole development window; evaluation on 2026-09-14..2026-10-06 requires separate authorization"},
        "scenarios": models, "development_evidence": {"selection_score_S": ch["S"], "verdict": sel["verdict"].get("verdict"),
                                                       "selection_file": "research/phase2b/model_research/selection.json"},
        "env": env_info(), "git": frozen["git"], "matrix_id": man["matrix_id"]}
    body["candidate_hash"] = hashlib.sha256(json.dumps({k: v for k, v in body.items() if k not in ("env", "git")}, sort_keys=True, default=str).encode()).hexdigest()
    (OUT / "MODEL_CANDIDATE_V1.json").write_text(json.dumps(body, indent=1, default=str), encoding="utf-8")
    print("MODEL_CANDIDATE_V1 candidate_hash", body["candidate_hash"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
