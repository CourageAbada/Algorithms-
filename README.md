# fxscalp — AI Forex & Gold Scalping Research/Execution Platform

> **Status: Phase 1 (read-only MT5 data foundation) and Phase 2A (XAU/USD feature pipeline, synthetic-data only) implemented and tested on Linux against fakes/synthetic data; NOT yet verified against a real MT5 terminal (Windows run required, gate still open). No trading logic, no orders, no trained models, no backtest results, no performance claims.**
> This project does not guarantee or claim profitability or any win rate. Trading leveraged CFD/FX products carries a high risk of loss. Development is demo/shadow only; LIVE trading is disabled by default and not part of the early phases.

## Goal
A modular platform that identifies short-horizon opportunities in **XAU/USD** (primary), **EUR/USD** and **GBP/USD** (secondary), with `LONG / SHORT / NO_TRADE` as first-class outcomes, an independent risk engine, MetaTrader 5 as the first (replaceable) broker, and a strict research path:

`RESEARCH -> BACKTEST -> WALK-FORWARD -> REPLAY -> SHADOW -> MT5 DEMO -> FORWARD VALIDATION -> LIVE-READINESS ASSESSMENT`

## Principles
Correctness before complexity · risk before profit · out-of-sample evidence before claims · execution costs before headline returns · NO_TRADE over a low-quality trade · the ML model and the LLM can never bypass the risk engine.

## Modes
`BACKTEST | REPLAY | SHADOW (default) | DEMO | LIVE`. Only DEMO/LIVE may send orders. Safety states: `HALT_NEW_TRADES` (blocks new exposure) and `EMERGENCY_FLATTEN` (authorised closure of existing exposure). LIVE requires `TRADING_MODE=LIVE`, `LIVE_TRADING_ENABLED=true` and a matching human-set `LIVE_ACKNOWLEDGEMENT`, plus runtime safety checks (see `docs/SECURITY_MODEL.md`).

## Repository layout
```
src/fxscalp/   core, market_data, ticks, bars, features, sessions, news, regimes, models,
               signals, risk, execution, brokers, backtest, replay, shadow, journal,
               agents, monitoring, database, api
configs/       base.yaml, risk.example.yaml, instruments/{xauusd,eurusd,gbpusd}.yaml
docs/          architecture, audits, models, protocols (start with PHASE0_5_REPORT.md)
tests/         unit, integration, leakage
frontend/ scripts/ models/ data/   (placeholders; data/ and models/ are git-ignored)
```
The unrelated Java examples that predate this project live in `legacy/java/` (decision D-8).

## Documentation index
`docs/PHASE2A_REPORT.md` (latest) · `PHASE2A_FEATURE_ARCHITECTURE` · `LEAKAGE_PREVENTION` · `XAUUSD_FEATURE_PLAN` · `docs/PHASE1_REPORT.md` · `WINDOWS_MT5_VERIFICATION` · `MT5_ADAPTER_AND_DATA_FOUNDATION` · `XAUUSD_DATA_PROFILE` (pending real data) · `PHASE0_5_REPORT` · `PHASE0_5_UPSTREAM_DEEP_DIVE` · `PHASE0_REPORT` · `UPSTREAM_AUDIT` · `UPSTREAM_COMPONENT_MATRIX` · `ARCHITECTURE_AUDIT` · `SYSTEM_ARCHITECTURE` · `IMPLEMENTATION_PLAN` · `FOREX_MARKET_MODEL` · `XAUUSD_RESEARCH_PLAN` · `DATA_REQUIREMENTS` · `FEATURE_CATALOG` · `ML_ARCHITECTURE` · `RISK_MODEL` · `VALIDATION_PROTOCOL` · `EXECUTION_MODEL` · `SECURITY_MODEL`

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # dev extras: pytest, ruff, mypy, ...
cp .env.example .env         # fake placeholders; use DEMO credentials only, never commit .env
python -m pytest -q                                   # no MT5 needed
python -m scripts.verify_mt5 --fake                    # pipeline check against the in-memory fake (NOT MT5)
python -m scripts.build_features --help                # Phase 2A: build ML-ready features from a Phase 1 dataset
python -m scripts.export_feature_registry --check      # feature registry drift check
```
Real MT5 verification (Windows, DEMO account, read-only): see `docs/WINDOWS_MT5_VERIFICATION.md`.
The `MetaTrader5` package is Windows-only (`pip install -e ".[dev,mt5]"` there).

## Security
Never commit credentials. `.env` is git-ignored; logs redact secrets; CI runs a secret scan. See `docs/SECURITY_MODEL.md`. This repository is private/proprietary; third-party attributions are in `THIRD_PARTY_NOTICES.md`.
