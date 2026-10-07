"""Write the committed feature-registry snapshot and the generated feature catalog.

  python -m scripts.export_feature_registry            # rewrites configs/features/xauusd_core.registry.yaml + docs/FEATURE_CATALOG.md
  python -m scripts.export_feature_registry --check    # exit 1 if the committed snapshot differs (CI / drift guard)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fxscalp.features.registry import build_registry, registry_yaml

ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = ROOT / "configs" / "features" / "xauusd_core.registry.yaml"
CATALOG_PATH = ROOT / "docs" / "FEATURE_CATALOG.md"


def catalog_markdown() -> str:
    reg = build_registry()
    L = ["# Feature Catalog (generated)", "",
         "Generated from the feature registry by `python -m scripts.export_feature_registry`; do not edit by hand.",
         f"Feature set: `{reg.name}`  version: `{reg.version}`  ({len(reg.enabled)} enabled of {len(reg.specs)}).", "",
         "Row semantics: a row at time t describes information available at the END of grid bar t (ticks with timestamp < t only). "
         "Distances/returns are in basis points (bps); spread in points = spread/point with `point` from stored symbol metadata. "
         "Warm-up = `lookback_s` seconds after each segment start (NaN, `row_valid = 0`). Every feature is validated for NaN/inf/range/"
         "constants, and the whole set for prefix invariance (see `LEAKAGE_PREVENTION.md`).", ""]
    fam: dict[str, list] = {}
    for s in reg.specs:
        fam.setdefault(s.family, []).append(s)
    for f, specs in fam.items():
        L += [f"## {f}", "", "| name | definition | source / tf | lookback s | units | expected range | NaN policy | leakage | reason |",
              "|---|---|---|---|---|---|---|---|---|"]
        for s in specs:
            r = "-" if s.expected_range is None else f"[{'' if s.expected_range[0] is None else s.expected_range[0]}, {'' if s.expected_range[1] is None else s.expected_range[1]}]"
            L.append(f"| `{s.name}` | {s.description} | {s.source} / {s.timeframe} | {s.lookback_s} | {s.units} | {r} | "
                     f"{s.nan_policy} | {s.leakage_risk} | {s.reason} |")
        L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    a = p.parse_args(argv)
    y = registry_yaml(build_registry())
    c = catalog_markdown()
    if a.check:
        ok = YAML_PATH.exists() and YAML_PATH.read_text(encoding="utf-8") == y
        print("registry snapshot is up to date" if ok else "registry snapshot DRIFT: run python -m scripts.export_feature_registry")
        return 0 if ok else 1
    YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    YAML_PATH.write_text(y, encoding="utf-8")
    CATALOG_PATH.write_text(c + "\n", encoding="utf-8")
    print(f"wrote {YAML_PATH} and {CATALOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
