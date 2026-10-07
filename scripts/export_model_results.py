"""Export every development-fold result (all candidates, baselines, controls, ablations) to one long CSV (the full record)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from fxscalp.research.model_research.experiment import ExperimentStore
from fxscalp.research.model_research.selection import load_results


def main(argv: list[str] | None = None) -> int:
    store = ExperimentStore(Path(argv[0]) if argv else "data/research/experiments")
    out = Path("research/phase2b/model_research/all_fold_results.csv")
    cols = ["experiment_id", "kind", "tag", "scenario", "feature_config", "family", "hp", "fold", "n_train", "n_val", "variant", "macro_f1", "balanced_accuracy",
            "accuracy", "log_loss", "brier", "ece_top", "recall_short", "recall_no_trade", "recall_long", "pred_freq_short", "pred_freq_no_trade",
            "pred_freq_long", "temperature", "duration_s", "peak_rss_mb"]
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in load_results(store):
            c = r["config"]
            for f in r["folds"]:
                for variant in ("uncalibrated", "calibrated"):
                    m = f[variant]
                    w.writerow([r["experiment_id"], c["kind"], c.get("tag", ""), c["scenario"], c["feature_config"], c["family"],
                                ";".join(f"{k}={v}" for k, v in c["hp"]), f["fold"], f["n_train"] if "n_train" in f else "", f["n_val"], variant,
                                *(round(m.get(k, float("nan")), 6) if k in m else "" for k in ("macro_f1", "balanced_accuracy", "accuracy", "log_loss", "brier", "ece_top")),
                                *(round(x, 6) for x in m["recall"]), *(round(x, 6) for x in m["pred_freq"]), f.get("temperature", ""), f.get("duration_s", ""), f.get("peak_rss_mb", "")])
                    n += 1
    print(f"wrote {n} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
