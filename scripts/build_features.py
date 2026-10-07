"""Build the ML-ready XAU/USD feature dataset from a Phase 1 raw tick dataset. READ-ONLY w.r.t. raw data; no broker, no
MT5, no trading permissions needed.

  python -m scripts.build_features --data-root data --dataset-id tickraw-<id>
  python -m scripts.build_features --data-root data --broker "<company>" --server "<server>" --start 2026-10-05 --end 2026-10-09

Writes data/derived-style outputs under <out-root>/features/<instrument>/<ml_dataset_id>/ (tick_norm, bars, features,
feature_quality_report, manifest) with full provenance. Exit code 1 if feature validation reports errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fxscalp.features.dataset import DatasetError, build_from_store, find_dataset_manifest
from fxscalp.features.pipeline import PipelineConfig
from fxscalp.features.spec import FeatureConfig
from fxscalp.market_data.store import TickStore


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build features from a Phase 1 raw tick dataset (offline; no MT5).")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-root", default=None, help="default: <data-root>/derived")
    p.add_argument("--dataset-id", help="Phase 1 dataset id (tickraw-...)")
    p.add_argument("--manifest", help="path to a Phase 1 dataset manifest json (alternative to --dataset-id)")
    p.add_argument("--instrument", default="XAU_USD")
    p.add_argument("--grid-s", type=int, default=1)
    p.add_argument("--segment-gap-s", type=int, default=1800)
    p.add_argument("--max-days", type=int, default=None, help="use only the first N chunks (quick checks)")
    p.add_argument("--keep-quarantined", action="store_true", help="do not exclude quarantined ticks (not recommended)")
    p.add_argument("--no-bars", action="store_true", help="do not persist bar datasets")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    store = TickStore(Path(a.data_root))
    out_root = Path(a.out_root) if a.out_root else Path(a.data_root) / "derived"
    try:
        if a.manifest:
            import json
            man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
        elif a.dataset_id:
            man = find_dataset_manifest(store, a.dataset_id)
        else:
            p.error("give --dataset-id or --manifest")
        cfg = PipelineConfig(instrument=a.instrument, grid_s=a.grid_s, segment_gap_s=a.segment_gap_s, features=FeatureConfig())
        res = build_from_store(store, man, out_root, cfg, exclude_quarantined=not a.keep_quarantined,
                               write_bars=not a.no_bars, max_days=a.max_days, force=a.force)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    m = res.manifest
    print(f"ml_dataset_id      {m['ml_dataset_id']}")
    print(f"raw dataset        {m['raw_dataset']['dataset_id']}  (time_basis={m['raw_dataset']['time_basis']})")
    print(f"feature set        {m['feature_set_version']}  ({len(m['feature_columns'])} features)")
    print(f"rows               {m['rows']:,} (valid after warm-up: {m['valid_rows']:,}); segments: {len(m['segments'])}")
    print(f"regime calibration {m['regime_calibration']}")
    print(f"quality            ok={m['quality']['ok']} errors={len(m['quality']['errors'])} warnings={len(m['quality']['warnings'])}")
    for e in m["quality"]["errors"][:10]:
        print("  ERROR:", e)
    print(f"output             {res.out_dir}")
    return 0 if m["quality"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
