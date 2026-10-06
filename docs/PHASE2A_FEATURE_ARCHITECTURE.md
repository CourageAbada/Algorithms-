# Phase 2A — Feature Architecture (XAU/USD)

Status: implemented and tested on synthetic / fake data only. **MT5 has not been verified** (Phase 1 Windows gate still open); nothing here depends on a live MT5. No model is trained, no strategy exists, no order path is touched.

## Pipeline

```
RAW (Phase 1 Parquet, immutable)
  -> NORMALIZATION   features/normalize.py   (unit-aware, quarantined ticks excluded + counted)
  -> SEGMENTS        gaps > segment_gap_s (1800 s) split the data; nothing is computed across a gap
  -> BARS            features/bars.py        (1s 5s 15s 30s 1m 5m [15m], explicit UTC half-open boundaries)
  -> FEATURE ENGINE  features/groups.py      (only causal primitives from features/ops.py)
  -> VALIDATION      features/validation.py  (quality report + static leakage scan)
  -> ML-READY DATASET features/dataset.py    (features.parquet + manifest with provenance)
```

Raw data is never modified. Derived data is written under `<out_root>/features/<instrument>/<mlds-id>/` (tick_norm, bars_*, features, quality report, manifest). The `mlds-` id hashes: raw dataset id, feature set version, config hash, pipeline version, schema version, quarantine policy. Same inputs => same id and byte-identical content hash (`features_content_sha256`).

## Decision-row semantics

A row at time `t` is the END of grid bar `t` and uses only ticks with timestamp `< t`. Bars are `[k*tf, (k+1)*tf)` UTC; a tick exactly on a boundary belongs to the new bar. The final bar of a dataset is not complete and is never used.

## Empty intervals and staleness

Empty intervals are explicit rows (`empty=True`, NaN prices). No silent forward-fill into bars. "As-of" state (last mid/spread) is carried explicitly and always accompanied by `staleness_s` and an empty-interval flag.

## Multi-timeframe alignment

`align_completed` maps a decision time to the latest COMPLETE higher-timeframe bar with `end <= t` (`searchsorted(side='right')` over complete bars). Regression test: at 10:03:20 the 10:00–10:05 five-minute bar is invisible; the 09:55–10:00 bar is the latest visible.

## Causal primitives

Feature code may only use `ops` primitives (rsum/rmean/rmax/rmin/rstd/rmedian/rrank/ewma/lag/ffill_asof/cummax_by...). `lag` rejects negative shifts; windows are trailing; the first `k-1` rows of a window are NaN (`warm`). A static AST scanner rejects direct pandas/numpy statistics, `shift`, `rolling`, `ewm`, `groupby`, `merge_asof`, negative lag, `center=True`, and reversed slices in feature-group code.

## Registry and versioning

Every group emits `(FeatureSpec, array)` through an `Emitter`, so metadata (definition, units, lookback, NaN behaviour, leakage assessment, reason) and computation cannot diverge. `FEATURE_SET_VERSION = xauusd_core-fs1-<12 hex>` hashes group definition versions, normalised source fingerprints, `FeatureConfig`, `RegimeConfig`, schema versions and the specs. A committed snapshot (`configs/features/xauusd_core.registry.yaml`) is drift-tested; regenerate with `python -m scripts.export_feature_registry` (`--check` to verify). The auto-generated catalog is `docs/FEATURE_CATALOG.md`.

## Regimes

Spread (NORMAL/ELEVATED/EXTREME/UNKNOWN) and volatility (LOW/NORMAL/HIGH/EXTREME/UNKNOWN) engines. Uncalibrated mode is purely relative (percentile AND ratio conditions) and every default is flagged `UNCALIBRATED_PLACEHOLDER_DEFAULTS`. Calibrated mode takes per-session `ReferenceQuantiles` computed from a separate real reference dataset (`calibrate_reference_quantiles`) — to be done with real XAU/USD data, not tonight.

## Sessions

Membership evaluated at bar START, minutes at bar END, from the DST-aware calendar. Session "so far" features use only expanding extremes inside the current session; `session_so_far_complete` marks partial history.

## Provenance (manifest)

Raw dataset id, stage hashes with parents, git commit + package versions, full config, regime calibration status, quality summary, schema/feature/pipeline versions, content hash. `load_normalized_ticks` verifies the Phase 1 hash chain and refuses `time_basis=unverified` data.

## Entry points

- API: `build_feature_frame(ticks, PipelineConfig, point)`; `build_from_store(...)`.
- CLI: `python -m scripts.build_features --dataset <id|path> ...` — read-only, no trading permissions, no MT5 import.
- `python -m scripts.export_feature_registry [--check]`, `python -m scripts.benchmark_features`.
