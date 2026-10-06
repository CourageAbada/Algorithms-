"""Explicit, actionable broker/data errors. Every error says what failed and what to do next."""

from __future__ import annotations


class BrokerError(RuntimeError):
    """Base class for broker-layer failures."""


class TradingDisabledError(BrokerError):
    """An order/position operation was attempted while trading is disabled (Phase 1: always)."""


class MT5UnavailableError(BrokerError):
    """The MetaTrader5 package cannot be used here (missing, wrong OS/architecture, wrong Python)."""


class MT5InitializationError(BrokerError):
    """mt5.initialize() failed (terminal not installed/running, wrong path, IPC failure)."""


class MT5AuthError(BrokerError):
    """Login failed (invalid credentials, wrong server, account disabled)."""


class AccountNotAllowedError(BrokerError):
    """The connected account class is not permitted for the current phase (e.g. REAL in Phase 1)."""


class AccountMismatchError(BrokerError):
    """The terminal is logged into a different account/server than configured."""


class SymbolNotFoundError(BrokerError):
    """No broker symbol matches the canonical instrument."""


class AmbiguousSymbolError(BrokerError):
    """Several broker symbols match equally well; an explicit choice in configuration is required."""


class NoTickDataError(BrokerError):
    """The terminal returned no tick data for the request."""


class StaleTickError(BrokerError):
    """The latest tick is older than the allowed age (market closed, feed down or clock problem)."""


class ConnectionLostError(BrokerError):
    """The terminal connection was lost or the owner thread is wedged; reconnect required."""


class CallTimeoutError(ConnectionLostError):
    """A blocking MT5 call exceeded its deadline; the session is treated as lost."""


class DataFormatError(BrokerError):
    """The terminal returned data in an unexpected shape/dtype (corrupt or unsupported version)."""


class TimeCalibrationError(RuntimeError):
    """Server-time calibration is impossible or inconclusive."""
