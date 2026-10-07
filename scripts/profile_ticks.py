"""Build the data profile (descriptive statistics only) from stored ticks.

  python -m scripts.profile_ticks --broker "<company>" --server "<server>" --instrument XAU_USD \
      --start 2025-09-01 --end 2025-09-30 --output docs/XAUUSD_DATA_PROFILE.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from fxscalp.core.provenance import slug
from fxscalp.market_data.profile import build_profile, render_markdown
from fxscalp.market_data.store import TickStore


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--broker", required=True)
    p.add_argument("--server", required=True)
    p.add_argument("--instrument", default="XAU_USD")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--output", default=None, help="markdown output path (default: print to stdout)")
    p.add_argument("--json-output", default=None)
    a = p.parse_args(argv)
    root = Path(a.data_root)
    store = TickStore(root)
    meta_dir = root / "metadata" / "symbol_info" / slug(a.broker) / slug(a.server) / a.instrument
    meta = None
    point = None
    if meta_dir.exists():
        files = sorted(meta_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)
        if files:
            meta = json.loads(files[-1].read_text(encoding="utf-8"))
            point = (meta.get("typed") or {}).get("point")
    prof = build_profile(store, a.broker, a.server, a.instrument, date.fromisoformat(a.start),
                         date.fromisoformat(a.end), point)
    md = render_markdown(prof, meta, title=f"{a.instrument.replace('_', '/')} Data Profile")
    if a.output:
        Path(a.output).write_text(md, encoding="utf-8")
        print(f"wrote {a.output}")
    else:
        print(md)
    if a.json_output:
        Path(a.json_output).write_text(json.dumps(prof, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
