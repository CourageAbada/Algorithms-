"""Cross-day continuity validation of an acquired raw tick dataset (READ-ONLY). Exit 0 only without serious problems."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fxscalp.features.dataset import find_dataset_manifest
from fxscalp.market_data.continuity import check_continuity
from fxscalp.market_data.store import TickStore


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate continuity across adjacent trading chunks.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--json-output", default=None)
    a = p.parse_args(argv)
    store = TickStore(Path(a.data_root))
    res = check_continuity(store, find_dataset_manifest(store, a.dataset_id))
    text = json.dumps(res, indent=2)
    print(text if len(text) < 6000 else json.dumps({k: v for k, v in res.items() if k != "time_basis_transitions"}, indent=2))
    if a.json_output:
        Path(a.json_output).write_text(text, encoding="utf-8")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
