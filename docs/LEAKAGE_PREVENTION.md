# Leakage Prevention (Phase 2A)

Principle: a feature at decision time `t` must be a function of data with timestamp `< t` only. We enforce this by construction, by static scan, and by dynamic test.

## Layers

1. **By construction** — only causal `ops` primitives; `lag(k<0)` raises; trailing windows only; HTF bars visible only when complete.
2. **Static scan** (`leakage.scan_source/scan_function`, also run inside `validate_feature_frame`) — rejects statistics outside `ops`, `shift`, `rolling`, `ewm`, `groupby`, `merge_asof`, negative lags, `center=True`, reversed slices.
3. **Dynamic prefix-invariance** (`check_prefix_invariance`) — build features on ticks T0→T100 and T0→T200 (truncating by arrival order); every row present in both must be bit-identical (rtol=0, atol=0), and the truncated run's row set must match the full run's up to its last row. Run at multiple cutoffs because leakage only shows where later data changes a value. Covered: normal data, anomalies (duplicates, out-of-order, missing), two segments, weekend closure, long gaps.
4. **Scaler discipline** (`CausalScaler(fit_end_ms)`, `check_scaler_fit_is_causal`) — statistics are fitted only on data before a cutoff; whole-dataset fitting is detected.
5. **Target contamination** (`check_target_contamination`) — features must not be derived from, or share rows with, label horizons.

## Leakage classes tested (each with a deliberately bad fixture that MUST be detected)

future bar info · future session high/low · centered rolling · negative shift · backward as-of mistake · future normalization · whole-dataset scaler fit · future volatility · target contamination (11 bad fixtures in `tests/leakage/bad_fixtures.py`; `test_leakage_detection.py` asserts each is caught).

## Known limits

- The quality tag `EXTREME_SPREAD` in Phase 1 is batch-median based (uses the whole batch). It is **never** used as a feature; documented and excluded.
- Prefix invariance proves no dependence on later *ticks*; it cannot prove economic soundness or absence of selection bias.
- Regime calibration on real data must use a reference period disjoint from any evaluation period (Phase 2B rule).
