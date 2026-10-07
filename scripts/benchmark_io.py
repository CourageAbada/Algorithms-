"""Performance baseline for the data pipeline on SYNTHETIC ticks (no MT5 involved).

Measures: structure->table conversion, quality assessment, Parquet write, Parquet read (+checksum verify),
and peak memory (RSS delta / tracemalloc peak). These numbers are the baseline for later optimisation; they say
NOTHING about MT5 retrieval throughput (that is measured on Windows by scripts.verify_mt5).

  python -m scripts.benchmark_io --rows 1000000 --json-output docs/baselines/phase1_synthetic_io.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
import tracemalloc
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import psutil

from fxscalp.brokers.base import RAW_TICK_DTYPE
from fxscalp.brokers.mt5.convert import ticks_to_array
from fxscalp.market_data import quality as Qm
from fxscalp.market_data.schema import derived_table, raw_table
from fxscalp.market_data.store import ChunkKey, TickStore
from fxscalp.market_data.timebase import NormalizedTime, TimeBasis


def synth(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.empty(n, dtype=RAW_TICK_DTYPE)
    t = 1_760_000_000_000 + np.cumsum(rng.exponential(400, n)).astype("int64")
    a["time_msc"], a["time"] = t, t // 1000
    mid = 2000 + np.cumsum(rng.normal(0, 0.02, n))
    sp = np.round(0.25 + rng.exponential(0.05, n), 2)
    a["bid"], a["ask"] = np.round(mid - sp / 2, 2), np.round(mid + sp / 2, 2)
    a["last"], a["volume"], a["volume_real"] = 0.0, 0, 0.0
    a["flags"] = rng.choice([6, 2, 4], n)
    return a


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1_000_000)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--json-output", default=None)
    a = p.parse_args(argv)
    proc = psutil.Process()
    res: dict = {"synthetic": True, "rows": a.rows, "python": platform.python_version(), "machine": platform.machine(),
                 "platform": platform.platform(), "cpu_count": psutil.cpu_count(), "runs": []}
    arr = synth(a.rows)
    norm = NormalizedTime(arr["time_msc"].astype("int64") - 3 * 3600 * 1000, np.zeros(a.rows, bool), np.zeros(a.rows, bool),
                          TimeBasis.SERVER_FIXED_OFFSET)
    for _ in range(a.repeat):
        rss0 = proc.memory_info().rss
        tracemalloc.start()
        _, t_adapt = timed(lambda: ticks_to_array(arr))
        tbl, t_conv = timed(lambda: raw_table(arr, norm, datetime.now(timezone.utc)))
        _, t_der = timed(lambda: derived_table(tbl, 0.01))
        q, t_q = timed(lambda: Qm.assess(arr["time"], arr["time_msc"], arr["bid"], arr["ask"], arr["last"], arr["volume"], arr["flags"]))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        with tempfile.TemporaryDirectory() as d:
            store = TickStore(Path(d))
            key = ChunkKey("bench", "bench", "XAU_USD", date(2025, 10, 6))
            man, t_w = timed(lambda: store.write_raw_chunk(key, tbl, {"broker_symbol": "SYNTH"}))
            _, t_r = timed(lambda: store.read_raw_chunk(key, verify=True))
            _, t_r_nov = timed(lambda: store.read_raw_chunk(key, verify=False))
            _, t_rc = timed(lambda: store.read_raw_chunk(key, ["source_time_msc", "bid", "ask"], verify=False))
            mb = man["file_bytes"] / 1e6
        res["runs"].append({
            "adapter_ticks_to_array_s": t_adapt, "convert_to_arrow_s": t_conv, "derive_s": t_der, "quality_assess_s": t_q,
            "parquet_write_s": t_w, "parquet_read_verified_s": t_r, "parquet_read_unverified_s": t_r_nov,
            "parquet_read_3cols_s": t_rc, "file_mb": mb,
            "rows_per_s": {"adapter_convert": a.rows / t_adapt, "convert": a.rows / t_conv, "quality": a.rows / t_q, "write": a.rows / t_w,
                           "read_verified": a.rows / t_r, "read_unverified": a.rows / t_r_nov},
            "write_mb_per_s": mb / t_w, "bytes_per_tick_on_disk": man["file_bytes"] / a.rows,
            "tracemalloc_peak_mb": peak / 1e6, "rss_delta_mb": (proc.memory_info().rss - rss0) / 1e6})
    best = {k: min(r[k] for r in res["runs"]) for k in ("adapter_ticks_to_array_s", "convert_to_arrow_s", "quality_assess_s", "parquet_write_s",
                                                        "parquet_read_verified_s", "parquet_read_unverified_s")}
    res["best"] = best
    print(json.dumps(res, indent=2, default=float))
    if a.json_output:
        Path(a.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_output).write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
