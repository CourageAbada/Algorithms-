"""The frozen Phase 2B specification: machine-readable manifests + a loader that refuses silent deviation.

``research/phase2b/*.json`` are generated once (scripts.build_phase2b_artifacts), committed, and indexed by ``spec_index.json``
with a SHA-256 per file and a combined ``spec_hash``. An experiment runner must call ``load_frozen_spec()``: it raises
``SpecDriftError`` if any file was edited, if a component is missing, or if the in-code definitions (policies, labels, costs,
folds, feature classification) no longer reproduce the frozen files. Changing the design therefore requires a new, explicit
spec version - not a quiet edit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fxscalp.research import costs, folds, labels, policy
from fxscalp.research.feature_policy import build_feature_manifest

SPEC_VERSION = "phase2b_spec/1"
SPEC_DIR = Path(__file__).resolve().parents[3] / "research" / "phase2b"
CODE_COMPONENTS = ("policy.json", "cost_model_v0.json", "labels_v0.json", "folds_v1.json", "feature_set_v1.json")
DATA_COMPONENTS = ("XAUUSD_RAW_V1.freeze.json", "eligibility_summary.json")


class SpecDriftError(RuntimeError):
    """The loaded specification differs from the frozen one."""


def canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True, default=str) + "\n"


def sha(obj_or_text: Any) -> str:
    text = obj_or_text if isinstance(obj_or_text, str) else canon(obj_or_text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def labels_manifest() -> dict[str, Any]:
    specs = labels.candidate_specs()
    return {
        "version": "labels/0", "status": "CANDIDATES - specified, diagnostics only; none selected; no model trained",
        "grid": {"seconds": 1, "observation_time": "bar end t_i (= feature row timestamp)", "decision_information": "<= t_i"},
        "candidates": [s.to_dict() for s in specs],
        "horizons_s": list(labels.CANDIDATE_HORIZONS_S),
        "horizon_rationale": {
            "15": "very short: 2x the 5 s micro window, ~100 ticks at the median 7.8 Hz; 1 s latency is still <7% of the horizon",
            "60": "short: equals the 60 s feature window and ~470 ticks; typical 1-min move (~3.5 bps = ~140 points) is >10x the median spread",
            "300": "medium-short: equals the 300 s feature window; upper end of an intended scalp, still ~2,300 ticks and well inside a segment",
            "excluded": "<15 s: below the 1 s grid/latency resolution relative to horizon; >300 s: not a scalp and cuts the number of independent samples",
            "basis": "observed tick rate, spread distribution (median 9 points) and feature timeframes - NOT historical profit",
        },
        "families": {
            "tb": "triple barrier on the executable side (bid for long exits, ask for short exits), barrier = max(m*sigma*mid*sqrt(H), 3*spread)",
            "fh": "fixed-horizon net return: bid_exit - ask_entry (long), bid_entry - ask_exit (short), minus extra round-trip cost",
            "opp": "opportunity/NO_TRADE: cost-covering (kappa=2x) favourable excursion with adverse <= favourable/rho (rho=1) inside H",
        },
        "classes": {"1": "LONG", "-1": "SHORT", "0": "NO_TRADE"},
        "status_codes": {"0": "OK", "1": "WARMUP (trailing sigma window incomplete)", "2": "BOUNDARY (window leaves the segment)"},
        "causality": "label at t uses only quotes in (t, t+L+H] for the outcome and quotes <= t for sigma/spread; never crosses a segment; "
                     "label_end_ms = t + (L+H)*1000 is used for purge/embargo",
        "tie_rules": "same-second profit and stop -> stop first (conservative); both sides succeed in the same second -> NO_TRADE + ambiguous",
        "parameters_fixed_a_priori": True, "tuning_on_results": "FORBIDDEN",
        "pre_registration": {
            "purpose": "close the forking path: the label is NOT chosen from model results",
            "primary": {"family": "tb", "horizon_s": 60, "cost_scenarios": ["C0_spread_only", "C1_moderate", "C2_pessimistic"],
                        "basis": "the only family whose NO_TRADE share is non-degenerate (45-82%) at every horizon and every cost scenario "
                                 "in the development-period label diagnostics; the middle horizon; selected from label balance only, "
                                 "no model or PnL result consulted"},
            "secondary": "all other candidates, reported as robustness; none may replace the primary without a new spec version",
            "degeneracy_rule": "a label whose NO_TRADE share is below 5% in a scenario is NON-INFORMATIVE as an abstain label there: it is "
                               "reported, never used as primary, and never rescued by re-tuning its parameters",
            "cost_reporting": "every reported result must be shown under all three cost scenarios; unknown costs are never defaulted to zero",
            "diagnostics_source": "research/phase2b/diagnostics/label_diagnostics.json (development period only)"},
    }


def build_code_manifests() -> dict[str, dict[str, Any]]:
    return {"policy.json": policy.policy_manifest(), "cost_model_v0.json": costs.cost_model_manifest(),
            "labels_v0.json": labels_manifest(), "folds_v1.json": folds.folds_manifest(),
            "feature_set_v1.json": build_feature_manifest()}


def write_spec(directory: Path, data_components: dict[str, dict[str, Any]]) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    files = {**build_code_manifests(), **data_components}
    index: dict[str, Any] = {"spec_version": SPEC_VERSION, "files": {}}
    for name, body in sorted(files.items()):
        text = canon(body)
        (directory / name).write_text(text, encoding="utf-8", newline="\n")
        index["files"][name] = sha(text)
    index["spec_hash"] = sha(index["files"])
    (directory / "spec_index.json").write_text(canon(index), encoding="utf-8", newline="\n")
    return index


@dataclass(frozen=True)
class FrozenSpec:
    spec_hash: str
    components: dict[str, dict[str, Any]]

    def __getitem__(self, name: str) -> dict[str, Any]:
        return self.components[name]


def load_frozen_spec(directory: Path | None = None, *, verify_code: bool = True) -> FrozenSpec:
    d = directory or SPEC_DIR
    idx_path = d / "spec_index.json"
    if not idx_path.exists():
        raise SpecDriftError(f"no frozen specification at {d}")
    index = json.loads(idx_path.read_text(encoding="utf-8"))
    comps: dict[str, dict[str, Any]] = {}
    for name in (*CODE_COMPONENTS, *DATA_COMPONENTS):
        p = d / name
        if name not in index["files"] or not p.exists():
            raise SpecDriftError(f"frozen component {name} is missing")
        text = p.read_text(encoding="utf-8")
        if sha(text) != index["files"][name]:
            raise SpecDriftError(f"{name} was modified after the freeze (hash mismatch)")
        comps[name] = json.loads(text)
    if sha(index["files"]) != index["spec_hash"]:
        raise SpecDriftError("spec_index.json is inconsistent")
    if verify_code:
        for name, body in build_code_manifests().items():
            if canon(body) != canon(comps[name]):
                raise SpecDriftError(f"the in-code definition behind {name} no longer matches the frozen file; "
                                     "a design change needs a new explicit spec version")
    return FrozenSpec(index["spec_hash"], comps)
