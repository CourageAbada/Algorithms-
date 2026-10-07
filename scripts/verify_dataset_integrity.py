"""End-to-end integrity verification of an acquired raw tick dataset (READ-ONLY).

  python -m scripts.verify_dataset_integrity --dataset-id tickraw-<id>

Checks every chunk named by the dataset manifest: manifests present, checksums, row counts, Parquet readable with the exact
schema, derived artifacts, source-day window ownership (no overlap/collision), chronological continuity across days,
unique chunk identity, dataset-id determinism and manifest checksum, orphan/temp files. Exit 0 only if everything holds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fxscalp.features.dataset import find_dataset_manifest
from fxscalp.market_data.integrity import verify_dataset_integrity
from fxscalp.market_data.store import TickStore


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verify the integrity of an acquired tick dataset.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--json-output", default=None)
    a = p.parse_args(argv)
    root = Path(a.data_root)
    store = TickStore(root)
    man = find_dataset_manifest(store, a.dataset_id)
    info = verify_dataset_integrity(store, man)
    print(json.dumps(info, indent=2))
    if a.json_output:
        Path(a.json_output).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
