"""Two independent safety concepts (decision D-7).

HALT_NEW_TRADES  - blocks additional exposure. Cheap to engage; may be engaged
                   automatically by any risk/health check. Existing positions are
                   untouched (they keep their broker-side SL/TP).
EMERGENCY_FLATTEN - attempts controlled closure of existing exposure. Much harder
                   to trigger: only an explicit, authorised command or a
                   human-configured hard-loss trigger. Never engaged by an
                   ordinary health check, drift alarm or model failure.

Neither concept implies the other, except that engaging EMERGENCY_FLATTEN also
halts new trades. Releasing either is a deliberate human act (not done here).
This module holds pure state/decision logic only; closing positions is the
execution layer's job (ticket-pinned closes, see docs/EXECUTION_MODEL.md).
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from enum import Enum


class SafetyCommand(str, Enum):
    HALT_NEW_TRADES = "HALT_NEW_TRADES"
    EMERGENCY_FLATTEN = "EMERGENCY_FLATTEN"


class FlattenNotAuthorised(PermissionError):
    """EMERGENCY_FLATTEN requested without a valid authorisation."""


@dataclass
class SafetyState:
    halt_new_trades: bool = False
    emergency_flatten: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def may_open_new_trades(self) -> bool:
        return not (self.halt_new_trades or self.emergency_flatten)

    def halt(self, reason: str) -> None:
        """Engage HALT_NEW_TRADES. Idempotent; any component may call it."""
        self.halt_new_trades = True
        self.reasons.append(f"HALT_NEW_TRADES: {reason}")

    def flatten(self, reason: str, *, token: str | None, expected_token: str | None) -> None:
        """Engage EMERGENCY_FLATTEN. Requires an explicit authorisation token.

        The expected token is human-configured (environment/secret store). An
        empty/unset expected token means flattening cannot be authorised at all
        (fail closed), so automated code paths cannot trigger it by accident.
        """
        if not expected_token or token is None or not hmac.compare_digest(token, expected_token):
            raise FlattenNotAuthorised("EMERGENCY_FLATTEN requires a valid authorisation token")
        self.emergency_flatten = True
        self.halt_new_trades = True
        self.reasons.append(f"EMERGENCY_FLATTEN: {reason}")
