"""Fail-closed verification of the frozen research state before any experiment."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from fxscalp.features.dataset import find_dataset_manifest
from fxscalp.market_data.store import TickStore
from fxscalp.research import folds, labels, policy, spec
from fxscalp.research.freeze import build_freeze_manifest
from fxscalp.research.model_research import protocol as P


class FrozenStateError(RuntimeError):
    """The frozen specification, data or protocol does not match what the experiment expects."""


def git_state() -> dict[str, Any]:
    def g(*a: str) -> str:
        return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain", "--untracked-files=no")),
            "branch": g("rev-parse", "--abbrev-ref", "HEAD")}


def verify_frozen_state(data_root: Path | str = "data", *, check_data: bool = True) -> dict[str, Any]:
    """Raises FrozenStateError on ANY drift. Returns the confirmations recorded with every experiment."""
    try:
        fs = spec.load_frozen_spec()
    except spec.SpecDriftError as exc:
        raise FrozenStateError(f"frozen specification drift: {exc}") from exc
    if fs.spec_hash != P.EXPECTED_SPEC_HASH:
        raise FrozenStateError(f"spec_hash {fs.spec_hash} != expected {P.EXPECTED_SPEC_HASH}")
    freeze = fs["XAUUSD_RAW_V1.freeze.json"]
    if freeze["freeze_id"] != P.EXPECTED_FREEZE_ID or freeze["dataset_id"] != policy.RAW_V1_DATASET_ID:
        raise FrozenStateError("raw dataset freeze differs from the expected XAUUSD_RAW_V1")
    pol, flds, lab, cost, feat = (fs["policy.json"], fs["folds_v1.json"], fs["labels_v0.json"], fs["cost_model_v0.json"],
                                  fs["feature_set_v1.json"])
    checks = {
        "raw_dataset_freeze": freeze["freeze_id"],
        "eligibility_policy": pol["eligibility_policy_version"], "segment_policy_gap_s": pol["segment_policy"]["gap_s"],
        "feature_manifest": feat["feature_set_version"], "n_features": feat["n_features"],
        "primary_label": f"{lab['pre_registration']['primary']['family']}_H{lab['pre_registration']['primary']['horizon_s']}",
        "folds": [f["name"] for f in flds["folds"]], "embargo_s": flds["embargo_s"],
        "final_holdout": f"{flds['final_holdout']['first_day']}..{flds['final_holdout']['last_day']} (LOCKED)",
        "cost_scenarios": list(cost["scenarios"]), "spec_hash": fs.spec_hash,
    }
    if checks["primary_label"] != P.PRIMARY_LABEL or tuple(cost["scenarios"]) != P.SCENARIOS:
        raise FrozenStateError("primary label or cost scenarios differ from the protocol")
    if flds["violations"] or flds["embargo_s"] != folds.EMBARGO_S or len(flds["folds"]) != 5:
        raise FrozenStateError("fold specification invalid or changed")
    if tuple(labels.CANDIDATE_HORIZONS_S) != (15, 60, 300):
        raise FrozenStateError("candidate horizons changed")
    if check_data:
        store = TickStore(Path(data_root))
        man = find_dataset_manifest(store, policy.RAW_V1_DATASET_ID)
        now = build_freeze_manifest(store, man, feature_set_ref=feat["feature_set_version"], code_commit="x", created_at_utc="x")
        if now["freeze_id"] != freeze["freeze_id"]:
            raise FrozenStateError("recomputed freeze_id differs: the data under data/ no longer matches the freeze")
        checks["data_checksums_match_freeze"] = True
    checks["protocol_hash"] = P.protocol_hash()
    checks["git"] = git_state()
    return checks
