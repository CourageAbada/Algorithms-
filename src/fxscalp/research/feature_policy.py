"""Feature Set v1 policy: every one of the 119 features is ALLOWED, CONDITIONAL or EXCLUDED, and model inputs are checked
against that classification. Classification is by PROVENANCE (what a feature is computed from and whether it needs fitting),
never by how well it might perform.

EXCLUDED (never a model input) if it relies on: undocumented raw MT5 flag bits, future information, depth of market,
traded volume, or statistics fitted on the whole dataset. CONDITIONAL features are admitted only when their stated
condition holds (and are otherwise masked, not dropped from the manifest). Raw/derived feed-internal columns are forbidden
as model inputs outright.
"""

from __future__ import annotations

from typing import Any

from fxscalp.features.pipeline import PipelineConfig
from fxscalp.features.registry import build_registry
from fxscalp.features.spec import FeatureSpec

ALLOWED, CONDITIONAL, EXCLUDED = "ALLOWED", "CONDITIONAL", "EXCLUDED"
FEATURE_SET_POLICY_VERSION = "feature_policy/1"

#: Columns that may NEVER enter a model feature matrix (feed-internal / undocumented / identifiers / unavailable data).
FORBIDDEN_MODEL_COLUMNS = frozenset({
    "flags", "unknown_flag_bits", "quality_flags", "quarantined", "volume", "volume_real", "last", "seq", "source_time",
    "source_time_msc", "ingestion_time_utc", "time_basis", "time_basis_id", "normalization_rule", "segment_id",
    "feed_regime", "eligible", "label", "label_end_ms", "status",
})
_EXCLUDE_TOKENS = ("flag", "dom_", "depth_of_market", "traded_volume", "vwap_traded")
_SESSION_RUNNING = ("minutes_since_session_open", "minutes_until_session_close", "session_high_so_far", "session_low_so_far",
                    "session_range_so_far_bps", "dist_session_high_bps", "dist_session_low_bps", "dist_session_mean_bps")
_FEED_SENSITIVE_PREFIX = ("tick_count_", "tick_rate_", "tick_acceleration_", "tick_interval_mean_ms_", "staleness_s", "empty_interval")


def classify(spec: FeatureSpec) -> dict[str, Any]:
    n = spec.name
    reasons: list[str] = []
    notes: list[str] = []
    # ---- EXCLUDED by provenance
    low = (n + " " + spec.source + " " + spec.description).lower()
    if any(t in n.lower() or t in spec.source.lower() for t in _EXCLUDE_TOKENS):
        reasons.append("relies on raw flags / DOM / traded volume")
    if spec.leakage_risk.upper().startswith("HIGH"):
        reasons.append("declared HIGH leakage risk")
    if "future" in low and "no future" not in low and "never" not in low:
        reasons.append("description references future information")
    if not spec.enabled:
        reasons.append("disabled in the registry")
    if reasons:
        return {"class": EXCLUDED, "reasons": reasons, "condition": None, "notes": notes}
    # ---- CONDITIONAL
    if n in ("spread_regime_code", "volatility_regime_code"):
        return {"class": CONDITIONAL, "reasons": ["regime thresholds are UNCALIBRATED_PLACEHOLDER_DEFAULTS"],
                "condition": "thresholds must be re-fit on the fold's TRAINING block (research.preprocessing) before use; the "
                             "placeholder codes are not a valid model input", "notes": notes}
    if n in _SESSION_RUNNING or n == "session_so_far_complete":
        return {"class": CONDITIONAL, "reasons": ["running aggregate inside the current session instance (MEDIUM leakage risk by design); "
                                                  "partial when a session is observed mid-way (segment start, data start)"],
                "condition": "use only where session_so_far_complete == 1 for the sample (else mask); never the session's final value",
                "notes": notes}
    # ---- ALLOWED
    if n.startswith(_FEED_SENSITIVE_PREFIX):
        notes.append("feed-sensitive: its distribution shifted across FEED_REGIME_BOUNDARY / April-May vs June; fit-dependent "
                     "transforms must be train-only")
    if "tickweighted" in n:
        notes.append("weights are tick COUNTS (update counts); traded volume is never used")
    if spec.group == "mtf":
        notes.append("uses completed higher-timeframe bars only (as-of alignment)")
    return {"class": ALLOWED, "reasons": ["trailing/as-of computation over past rows only (prefix-invariance tested)"],
            "condition": None, "notes": notes}


def build_feature_manifest() -> dict[str, Any]:
    cfg = PipelineConfig()
    reg = build_registry(cfg.features, cfg.regimes)
    rows = []
    for s in reg.enabled:
        c = classify(s)
        rows.append({"name": s.name, "group": s.group, "family": s.family, "source": s.source, "timeframe": s.timeframe,
                     "lookback_s": s.lookback_s, "version": s.version, "leakage_risk": s.leakage_risk.split(":")[0],
                     **c})
    counts = {k: sum(r["class"] == k for r in rows) for k in (ALLOWED, CONDITIONAL, EXCLUDED)}
    return {
        "version": FEATURE_SET_POLICY_VERSION, "feature_set_name": reg.name, "feature_set_version": reg.version,
        "n_features": len(rows), "counts": counts,
        "classification_basis": "provenance (inputs, look-back, fitting needs); never model performance",
        "pipeline": {"segment_gap_s": 300, "grid_s": 1, "config_hash_note": "PipelineConfig(segment_gap_s=300) for Phase 2B"},
        "model_input_allowed": [r["name"] for r in rows if r["class"] == ALLOWED],
        "model_input_conditional": {r["name"]: r["condition"] for r in rows if r["class"] == CONDITIONAL},
        "excluded": {r["name"]: r["reasons"] for r in rows if r["class"] == EXCLUDED},
        "forbidden_model_columns": sorted(FORBIDDEN_MODEL_COLUMNS),
        "fitting_policy": "no feature or transform may be fit on the whole dataset; scalers/thresholds/baselines are fit per fold on the training block only",
        "features": rows,
    }


def assert_model_inputs_allowed(columns: list[str], manifest: dict[str, Any] | None = None, *, allow_conditional: bool = False) -> None:
    """Raise if a model feature matrix would contain forbidden raw columns, EXCLUDED/unknown features, or (unless the caller
    has applied the gating condition) CONDITIONAL features."""
    m = manifest or build_feature_manifest()
    by = {r["name"]: r for r in m["features"]}
    bad = []
    for c in columns:
        if c in FORBIDDEN_MODEL_COLUMNS or any(t in c.lower() for t in ("flag", "0x400", "0x80")):
            bad.append(f"{c}: forbidden raw/feed-internal column")
        elif c not in by:
            bad.append(f"{c}: not a Feature Set v1 feature")
        elif by[c]["class"] == EXCLUDED:
            bad.append(f"{c}: EXCLUDED ({by[c]['reasons']})")
        elif by[c]["class"] == CONDITIONAL and not allow_conditional:
            bad.append(f"{c}: CONDITIONAL ({by[c]['condition']})")
    if bad:
        raise ValueError("model input policy violation: " + "; ".join(bad))
