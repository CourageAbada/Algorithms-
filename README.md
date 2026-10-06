# fxscalp — AI Forex & Gold Scalping Research/Execution Platform

> **Status: Phase 0 (architecture & skeleton). No trading logic, no backtest results, no performance claims.**
> This project does not guarantee or claim profitability or any win rate. Trading leveraged CFD/FX products carries a high risk of loss. Development is demo/shadow only; LIVE trading is disabled by default and not part of the early phases.

## Goal
A modular platform that identifies short-horizon opportunities in **XAU/USD** (primary), **EUR/USD** and **GBP/USD** (secondary), with `LONG / SHORT / NO_TRADE` as first-class outcomes, an independent risk engine, MetaTrader 5 as the first (replaceable) broker, and a strict research path:

`RESEARCH -> BACKTEST -> WALK-FORWARD -> REPLAY -> SHADOW -> MT5 DEMO -> FORWARD VALIDATION -> LIVE-READINESS ASSESSMENT`

## Principles
Correctness before complexity · risk before profit · out-of-sample evidence before claims · execution costs before headline returns · NO_TRADE over a low-quality trade · the ML model and the LLM can never bypass the risk engine.

## Modes
`BACKTEST | REPLAY | SHADOW (default) | DEMO | LIVE`. Only DEMO/LIVE may send orders. LIVE requires `TRADING_MODE=LIVE`, `LIVE_TRADING_ENABLED=true` and a matching human-set `LIVE_ACKNOWLEDGEMENT`, plus runtime safety checks (see `docs/SECURITY_MODEL.md`).

## Repository layout
```
src/fxscalp/   core, market_data, ticks, bars, features, sessions, news, regimes, models,
               signals, risk, execution, brokers, backtest, replay, shadow, journal,
               agents, monitoring, database, api
configs/       base.yaml, risk.example.yaml, instruments/{xauusd,eurusd,gbpusd}.yaml
docs/          architecture, audits, models, protocols (start with PHASE0_REPORT.md)
tests/         unit, integration, leakage
frontend/ scripts/ models/ data/   (placeholders; data/ and models/ are git-ignored)
```
The two Java files in the root predate this project and are unrelated (see decision D-8); their old README is preserved as `README_JAVA_LEGACY.md`.

## Documentation index
`docs/PHASE0_REPORT.md` (start here) · `UPSTREAM_AUDIT` · `UPSTREAM_COMPONENT_MATRIX` · `ARCHITECTURE_AUDIT` · `SYSTEM_ARCHITECTURE` · `IMPLEMENTATION_PLAN` · `FOREX_MARKET_MODEL` · `XAUUSD_RESEARCH_PLAN` · `DATA_REQUIREMENTS` · `FEATURE_CATALOG` · `ML_ARCHITECTURE` · `RISK_MODEL` · `VALIDATION_PROTOCOL` · `EXECUTION_MODEL` · `SECURITY_MODEL`

## Quick start (Phase 0)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # dev extras: pytest, ruff, mypy, ...
cp .env.example .env         # fake placeholders; use DEMO credentials only, never commit .env
pytest
```
The `MetaTrader5` package is Windows-only and is not needed until Phase 1.

## Security
Never commit credentials. `.env` is git-ignored; logs redact secrets. See `docs/SECURITY_MODEL.md`.
