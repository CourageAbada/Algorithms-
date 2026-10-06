"""Performance baseline of the Phase 2A pipeline on SYNTHETIC ticks of increasing size (no MT5, no real data).

Stages: tick normalisation (+ quality tagging), bar aggregation (grid + all multi-timeframe bars), feature computation
(all groups + regimes), Parquet write and read of the feature table. Records rows/s, elapsed time and RSS. These numbers are
a baseline for later optimisation only; real-data performance is measured tomorrow with `--from-store`.

  python -m scripts.benchmark_features --sizes 100000 500000 1000000 --json-output docs/baselines/phase2a_synthetic_pipeline.json
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from fxscalp.features.groups import GROUPS, Emitter
from fxscalp.features.normalize import from_raw_array
from fxscalp.features.pipeline import PipelineConfig
from fxscalp.features.segment import build_context
from fxscalp.synthetic.xauusd import SynthConfig, generate_ticks


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


def run_one(n_ticks: int) -> dict:
    proc = psutil.Process()
    rate = 5.0
    cfg = SynthConfig(start_utc=datetime(2025, 10, 6, 0, 0, tzinfo=timezone.utc), duration_s=int(n_ticks / rate * 1.4) + 60,
                      base_rate_hz=rate, session_activity=False, weekend_closure=False, seed=1)
    arr, t_gen = timed(lambda: generate_ticks(cfg))
    arr = arr[:n_ticks]
    pcfg = PipelineConfig()
    rss0 = proc.memory_info().rss
    (ticks, nrep), t_norm = timed(lambda: from_raw_array(arr, point=0.01))
    rss_norm = proc.memory_info().rss
    ctx, t_bars = timed(lambda: build_context(ticks["timestamp_utc_ms"].to_numpy(), ticks["bid"].to_numpy(), ticks["ask"].to_numpy(),
                                              ticks["volume"].to_numpy(), grid_s=1, point=0.01, cfg=pcfg.features,
                                              regime_cfg=pcfg.regimes, data_continues_after=True))
    rss_bars = proc.memory_info().rss

    def feats():
        prior, out = {}, {}
        for g in GROUPS:
            em = Emitter(g.name, g.definition_version, ())
            g.fn(ctx, pcfg.features, prior, em)
            for spec, a in em.items:
                prior[spec.name] = a
                out[spec.name] = a
        return pd.DataFrame(out)

    df, t_feat = timed(feats)
    rss_feat = proc.memory_info().rss
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "features.parquet"
        _, t_w = timed(lambda: df.to_parquet(p, compression="zstd", compression_level=3))
        mb = p.stat().st_size / 1e6
        _, t_r = timed(lambda: pd.read_parquet(p))
    rows = len(df)
    return {"ticks": n_ticks, "grid_rows": rows, "features": df.shape[1], "gen_s": t_gen,
            "normalize_s": t_norm, "normalize_ticks_per_s": n_ticks / t_norm,
            "bars_s": t_bars, "bars_ticks_per_s": n_ticks / t_bars,
            "features_s": t_feat, "features_rows_per_s": rows / t_feat, "features_ticks_per_s": n_ticks / t_feat,
            "parquet_write_s": t_w, "parquet_write_rows_per_s": rows / t_w, "parquet_mb": mb,
            "parquet_read_s": t_r, "parquet_read_rows_per_s": rows / t_r,
            "end_to_end_ticks_per_s": n_ticks / (t_norm + t_bars + t_feat + t_w),
            "rss_mb": {"before": rss0 / 1e6, "after_normalize": rss_norm / 1e6, "after_bars": rss_bars / 1e6,
                       "after_features": rss_feat / 1e6},
            "peak_rss_mb_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[100_000, 500_000, 1_000_000])
    p.add_argument("--json-output", default=None)
    a = p.parse_args(argv)
    res = {"synthetic": True, "note": "synthetic ticks; baseline for later optimisation only; says nothing about real data or MT5",
           "python": platform.python_version(), "machine": platform.machine(), "platform": platform.platform(),
           "cpu_count": psutil.cpu_count(), "total_ram_mb": psutil.virtual_memory().total / 1e6,
           "versions": {"numpy": np.__version__, "pandas": pd.__version__}, "runs": []}
    for n in a.sizes:
        r = run_one(n)
        res["runs"].append(r)
        print(f"{n:>9,} ticks -> {r['grid_rows']:>8,} rows | normalize {r['normalize_s']:.2f}s | bars {r['bars_s']:.2f}s | "
              f"features {r['features_s']:.2f}s ({r['features_rows_per_s']:,.0f} rows/s) | write {r['parquet_write_s']:.2f}s | "
              f"read {r['parquet_read_s']:.2f}s | peak RSS {r['peak_rss_mb_process']:.0f} MB", flush=True)
    if a.json_output:
        Path(a.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_output).write_text(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
