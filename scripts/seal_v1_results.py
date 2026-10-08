"""Seal Protocol v1 results: SHA-256 of every v1 artifact and experiment, so v1 cannot be overwritten or silently reclassified.

Text files are hashed with newlines normalised (stable across LF/CRLF checkouts); experiment files are hashed as raw bytes.
Writes research/phase2b/v1_immutable_manifest.json. Verification: ``python -m scripts.seal_v1_results --verify``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TEXT_FILES = [
    "research/phase2b/model_research/protocol_v1.json", "research/phase2b/model_research/selection.json",
    "research/phase2b/model_research/diagnostics.json", "research/phase2b/model_research/MODEL_CANDIDATE_V1.json",
    "research/phase2b/model_research/all_fold_results.csv", "docs/PHASE2B_MODEL_RESEARCH.md", "docs/PHASE2B_RESEARCH_SPEC.md",
    "research/phase2b/spec_index.json", "research/phase2b/XAUUSD_RAW_V1.freeze.json",
]
OUT = Path("research/phase2b/v1_immutable_manifest.json")


def text_sha(p: Path) -> str:
    return hashlib.sha256(p.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest()


def bytes_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(store: Path) -> dict:
    exps = {}
    if store.exists():
        for d in sorted(store.iterdir()):
            if (d / "result.json").exists():
                exps[d.name] = {"result_json": bytes_sha(d / "result.json"),
                                "predictions": bytes_sha(d / "predictions.parquet") if (d / "predictions.parquet").exists() else None}
    return {"files": {f: text_sha(Path(f)) for f in TEXT_FILES}, "experiments": exps}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="data/research/experiments")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    cur = collect(Path(a.store))
    if a.verify:
        m = json.loads(OUT.read_text(encoding="utf-8"))
        bad = [f for f, h in m["files"].items() if cur["files"].get(f) != h]
        if cur["experiments"]:
            bad += [e for e, h in m["experiments"].items() if cur["experiments"].get(e) != h]
        print("v1 intact" if not bad else f"v1 CHANGED: {bad[:5]}")
        return 1 if bad else 0
    body = {
        "purpose": "Protocol v1 is IMMUTABLE. Nothing here may be overwritten or reclassified.",
        "sealed_at_git_commit": "9d5c4b3", "protocol_v1_formal_verdict": "NO RELIABLE SIGNAL DETECTED",
        "protocol_v1_formal_outcome": "no candidate passed gates G1-G3 (G2 failed mechanically for every class-weighted model); MODEL_CANDIDATE_V1 = NONE",
        "post_hoc_exploratory_material": ["selection.json: 'exploratory', 'exploratory_focus', 'bootstrap_prior_corrected_logloss' and the prior-corrected "
                                          "'verdict_if_G2_used_prior_corrected_probabilities' - POST-HOC, computed after seeing v1 results, NOT part of v1's pre-registered "
                                          "selection or verdict",
                                          "docs/PHASE2B_MODEL_RESEARCH.md sections 6-11 describe the exploratory focus candidate under that label"],
        "n_experiments": len(cur["experiments"]), "files": cur["files"], "experiments": cur["experiments"],
        "final_holdout": "2026-09-14..2026-10-06 LOCKED: never read",
    }
    OUT.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    print(f"sealed {len(cur['files'])} files and {len(cur['experiments'])} experiments -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
