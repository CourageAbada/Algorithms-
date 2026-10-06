"""Trading modes and the LIVE safety interlock.

SHADOW is the default. LIVE requires three independent, explicit settings and is
never reachable by default or by a typo.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import Enum


class TradingMode(str, Enum):
    BACKTEST = "BACKTEST"
    REPLAY = "REPLAY"
    SHADOW = "SHADOW"
    DEMO = "DEMO"
    LIVE = "LIVE"

    @property
    def may_send_orders(self) -> bool:
        """Only DEMO and LIVE may reach a broker order endpoint."""
        return self in (TradingMode.DEMO, TradingMode.LIVE)


DEFAULT_MODE = TradingMode.SHADOW


class ModeConfigError(ValueError):
    """Raised when the mode configuration is invalid or unsafe."""


def resolve_mode(env: Mapping[str, str] | None = None) -> TradingMode:
    """Resolve the active mode from the environment, failing closed.

    - Unset TRADING_MODE -> SHADOW.
    - Unknown value -> ModeConfigError (never silently falls through to a riskier mode).
    - LIVE requires LIVE_TRADING_ENABLED=true AND LIVE_ACKNOWLEDGEMENT matching
      LIVE_ACKNOWLEDGEMENT_EXPECTED (both set by a human; no default exists).
    Further LIVE preconditions (account verified, data healthy, ...) are enforced
    by the execution layer in later phases; this is only the first gate.
    """
    env = os.environ if env is None else env
    raw = env.get("TRADING_MODE", "").strip().upper()
    if not raw:
        return DEFAULT_MODE
    try:
        mode = TradingMode(raw)
    except ValueError as exc:
        raise ModeConfigError(f"Unknown TRADING_MODE {raw!r}") from exc

    if mode is TradingMode.LIVE:
        enabled = env.get("LIVE_TRADING_ENABLED", "").strip().lower() == "true"
        ack = env.get("LIVE_ACKNOWLEDGEMENT", "")
        expected = env.get("LIVE_ACKNOWLEDGEMENT_EXPECTED", "")
        if not enabled:
            raise ModeConfigError("LIVE requires LIVE_TRADING_ENABLED=true")
        if not expected or ack != expected:
            raise ModeConfigError("LIVE requires a matching LIVE_ACKNOWLEDGEMENT")
    return mode
