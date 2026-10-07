"""Build the Phase 2B model-research matrix (development days only; verifies the frozen state first)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fxscalp.features.dataset import find_dataset_manifest
from fxscalp.market_data.store import TickStore
from fxscalp.research import policy
from fxscalp.research.model_research.data import build_matrix
from fxscalp.research.model_research.frozen import verify_frozen_state


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the model-research matrix (30 s grid, tb_H60 labels C0/C1/C2).")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="data/research/matrix_v1")
    a = p.parse_args(argv)
    verify_frozen_state(a.data_root)
    store = TickStore(Path(a.data_root))
    man = build_matrix(store, find_dataset_manifest(store, policy.RAW_V1_DATASET_ID), Path(a.out))
    print(f"matrix {man['matrix_id']}: {man['rows']:,} rows over {len(man['days'])} development days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
