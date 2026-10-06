# ML Architecture

Fast numerical models make real-time decisions. An LLM never does (see §9).

### 1. Decision structure

```
features ─► primary models ─► raw scores ─► calibration ─► meta-model ─► {LONG, SHORT, NO_TRADE} + expected edge
regime engine, novelty gate, spread/session state ─► (inputs to meta-model and hard vetoes)
```
Output is a probability vector over {LONG, SHORT, NO_TRADE}. Thresholds are **not** hardcoded; they are selected on validation folds by maximising net expectancy subject to drawdown and trade-count constraints, then frozen before the holdout is touched.

### 2. Labels (models/labels; the only code allowed to look forward)

Event-based candidate entries (e.g. every N seconds, on volatility-filtered events, or on setup triggers). For each candidate and each side, with barriers set by volatility (ATR multiples), simulated on **bid/ask**:
- `tp_before_sl` (triple barrier: TP, SL, vertical time barrier H),
- net return after spread + commission + estimated slippage,
- MFE, MAE, time-to-exit.
Classes: LONG-wins, SHORT-wins, NEITHER/timeout (-> NO_TRADE target). Multiple (TP,SL,H) variants are compared (E5). Overlapping labels are handled with sample weights (uniqueness) and purging/embargo in CV.

### 3. Expected edge

`expected_net_edge = E[gross move | side] - spread - commission - E[slippage] - uncertainty_margin`, where gross move comes from the target-hit model / return regressor, slippage from measured execution data (default pessimistic until measured). Proceed only if `expected_net_edge > min_edge` (min_edge from validation, per instrument and session).

### 4. Model zoo (staged)

1. Logistic regression (regularised) — floor baseline.
2. Random forest (where useful), LightGBM, XGBoost, CatBoost — primary candidates.
3. Volatility model (realised-vol forecast; used to size barriers and as a regime input).
4. Microstructure/tick model (features B only) — checked for independent value.
5. Sequence models (TCN, GRU, LSTM, temporal transformer) **only after** GBM baselines are established, and kept only if they beat them statistically out of sample after costs (paired test on fold results, e.g. block bootstrap of PnL differences). Complexity must be justified.

### 5. Calibration

Platt (sigmoid) and isotonic, fitted on a calibration segment that is chronologically after training and before validation scoring; compared by Brier score, log-loss, ECE, reliability diagrams (per class, per session, per regime). Calibration drift is monitored live. Isotonic is avoided on small samples.

### 6. Meta-labeling

Primary model proposes a side; a meta-model (GBM or logistic) predicts whether *that* proposal will be profitable net of costs, using: primary confidence/margin, spread state, volatility, session, regime, news proximity, trend alignment, recent execution quality, trailing performance. Trade only if meta-probability > threshold (validation-selected). NO_TRADE is the default outcome.

### 7. Regime engine

Baseline: transparent rules (ADX/trend slope, vol percentile, range ratios, spread/liquidity stress, event state, abnormality flags) mapping to `TRENDING_UP, TRENDING_DOWN, RANGING, BREAKOUT, HIGH_VOLATILITY, LOW_VOLATILITY, LIQUIDITY_STRESS, NEWS_RISK, ABNORMAL, UNKNOWN`. ML regime classifier (e.g. HMM/GMM/GBM on causal features) is compared to the rules; if it does not add value it is not used. Regimes can only restrict trading, never override the risk engine.

### 8. Novelty / safety gates

Input distribution gate (robust Mahalanobis / isolation forest / PSI vs training set); NaN/Inf guard on features and outputs; probability-vector sanity (sums to 1, in [0,1]); stale-model guard (model older than `expiry` -> NO_TRADE). Any failure -> NO_TRADE and a risk event.

### 9. LLM role (agents/)

Allowed: research, experiment design, backtest interpretation, failure analysis, model comparison, report generation, journal analysis, monitoring summaries. Not allowed: tick-level decisions, order placement, risk-limit edits, promotion of models. Agent outputs must cite artefact IDs/queries (evidence-gated, cf. Vibe-Trading's grounding idea). Agent-suggested experiments run through the same validation protocol.

### 10. Model lifecycle

```
DATA ─► FEATURES ─► TRAIN CANDIDATE ─► VALIDATE ─► WALK-FORWARD ─► COMPARE vs CHAMPION ─► PROMOTE | REJECT
```
Registry record: model id, model version, feature version, dataset manifest hash, train/val/test periods, hyperparameters, seeds, metrics (incl. calibration, cost-stress), code commit, created_at, status (candidate/champion/retired). Promotion needs: beats champion on pre-registered metrics over matched walk-forward periods with CI, passes stress tests, passes drift/novelty checks, **plus explicit human approval** (D-6). Never auto-promote because training metrics improved. Retraining is scheduled in separate worker processes (FreqAI-like cadence as a concept); the live engine only loads registry-approved artefacts.

### 11. Drift monitoring and response

Monitored: feature drift (PSI/KS), prediction/confidence drift, calibration drift, win-rate/expectancy/profit-factor drift (with sequential tests, accounting for small samples), spread and slippage drift, regime frequency. Responses are tiered and automatic only in the safe direction: **reduce risk -> stop new trades**. Raising risk or changing limits is human-only. The system never edits its own risk constraints or production code.

### 12. Reproducibility and tooling

Deterministic seeds, pinned dependencies, dataset hashes, experiment tracker in Postgres (or MLflow if approved). Hyperparameter search (Optuna) is limited by a trial budget logged to the trial registry; selection uses inner walk-forward folds only.
