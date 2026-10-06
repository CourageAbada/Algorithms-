"""Account classification and the Phase 1 demo-only safety gate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

from fxscalp.brokers.errors import AccountNotAllowedError


class AccountClass(str, Enum):
    DEMO = "DEMO"
    CONTEST = "CONTEST"
    REAL = "REAL"
    UNKNOWN = "UNKNOWN"


# MT5 ENUM_ACCOUNT_TRADE_MODE values: verified in the installed MetaTrader5 5.0.6231 source.
_TRADE_MODE_TO_CLASS = {0: AccountClass.DEMO, 1: AccountClass.CONTEST, 2: AccountClass.REAL}

#: Phase 1 may collect data from DEMO accounts only. There is deliberately no override.
PHASE1_ALLOWED_ACCOUNT_CLASSES = frozenset({AccountClass.DEMO})


def classify_trade_mode(trade_mode: object) -> AccountClass:
    """Map an MT5 ``account_info().trade_mode`` to an AccountClass; anything unexpected is UNKNOWN."""
    if isinstance(trade_mode, bool):
        return AccountClass.UNKNOWN
    try:
        return _TRADE_MODE_TO_CLASS.get(int(trade_mode), AccountClass.UNKNOWN)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return AccountClass.UNKNOWN


def mask_login(login: int | str | None) -> str:
    """Last 3 characters only, for logs/reports (account numbers are sensitive)."""
    if login is None:
        return "***"
    s = str(login)
    return "***" + s[-3:] if len(s) > 3 else "***"


def login_fingerprint(login: int | str | None, server: str | None) -> str:
    """Short stable fingerprint to correlate datasets with an account without storing the number."""
    h = hashlib.sha256(f"{login}|{server}".encode()).hexdigest()
    return h[:12]


@dataclass(frozen=True)
class AccountIdentity:
    """Non-secret account identification, safe to log and persist."""

    login_masked: str
    login_fingerprint: str
    server: str
    company: str
    account_class: AccountClass
    trade_mode_raw: int | None
    margin_mode_raw: int | None
    leverage: int | None
    currency: str


def assert_account_allowed(identity: AccountIdentity,
                           allowed: frozenset[AccountClass] = PHASE1_ALLOWED_ACCOUNT_CLASSES) -> None:
    """Raise AccountNotAllowedError unless the account class is in ``allowed``."""
    if identity.account_class not in allowed:
        raise AccountNotAllowedError(
            f"Account {identity.login_masked} on server {identity.server!r} is classified "
            f"{identity.account_class.value} (trade_mode={identity.trade_mode_raw!r}); this phase "
            f"allows only {sorted(c.value for c in allowed)}. Log the MT5 terminal into a DEMO "
            "account and retry. No data was collected."
        )
