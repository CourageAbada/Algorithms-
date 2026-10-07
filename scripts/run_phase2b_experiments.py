"""Run the pre-registered Phase 2B model-research grid on the DEVELOPMENT folds (no holdout, no PnL).

  python -m scripts.run_phase2b_experiments --stage baselines|grid|all
Fails closed on any drift of the frozen specification. Experiments are deterministic, never overwritten, resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from fxscalp.research.model_research import protocol as P
from fxscalp.research.model_research.data import load_matrix
from fxscalp.research.model_research.experiment import ExperimentStore, Runner
from fxscalp.research.model_research.frozen import verify_frozen_state


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="data/research/matrix_v1")
    p.add_argument("--store", default="data/research/experiments")
    p.add_argument("--stage", choices=["baselines", "grid", "all"], default="all")
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--only-config", default=None)
    a = p.parse_args(argv)
    frozen = verify_frozen_state("data")
    if frozen["git"]["dirty"]:
        print("WARNING: git working tree has uncommitted changes (recorded in every experiment)", file=sys.stderr)
    print("frozen state verified:", json.dumps({k: v for k, v in frozen.items() if k in ("spec_hash", "raw_dataset_freeze", "protocol_hash")}))
    df, man = load_matrix(Path(a.matrix))
    feat = json.loads(Path("research/phase2b/feature_set_v1.json").read_text(encoding="utf-8"))
    runner = Runner(df, man["matrix_id"], frozen, feat["model_input_allowed"], ExperimentStore(a.store), a.n_jobs)
    t0 = time.perf_counter()
    for scn in P.SCENARIOS:
        if a.stage in ("baselines", "all"):
            runner.run_baselines(scn)
            print(f"baselines {scn} done {time.perf_counter() - t0:.0f}s", flush=True)
    if a.stage in ("grid", "all"):
        for scn in P.SCENARIOS:
            for fc in P.FEATURE_CONFIGS:
                if a.only_config and fc != a.only_config:
                    continue
                runner.run_models(scn, fc, P.hp_grid())
                print(f"grid {scn} {fc} done {time.perf_counter() - t0:.0f}s", flush=True)
    print("complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
