"""Loading the MetaTrader5 package and the read-only API proxy.

``ReadOnlyMT5Api`` is the ONLY object the adapter uses to reach the package. It exposes an explicit
allow-list of read functions and raises TradingDisabledError for any trading function, so no code path
in Phase 1 can place, check, modify or close an order even by accident.
"""

from __future__ import annotations

import struct
import sys
from types import ModuleType
from typing import Any, Callable

from fxscalp.brokers.errors import MT5UnavailableError, TradingDisabledError

#: Functions Phase 1 may call (all verified to exist in MetaTrader5 5.0.6231).
ALLOWED_FUNCTIONS = frozenset({
    "initialize", "shutdown", "version", "last_error",
    "terminal_info", "account_info",
    "symbols_total", "symbols_get", "symbol_info", "symbol_info_tick", "symbol_select",
    "copy_ticks_from", "copy_ticks_range", "copy_rates_from", "copy_rates_from_pos", "copy_rates_range",
    "market_book_add", "market_book_get", "market_book_release",
})

#: Anything that can create, check, change or close trades/orders, or send requests to the server.
FORBIDDEN_FUNCTIONS = frozenset({
    "order_send", "order_check", "order_calc_margin", "order_calc_profit",
    "Buy", "Sell", "Close", "login",
    "positions_get", "positions_total", "orders_get", "orders_total",
    "history_orders_get", "history_orders_total", "history_deals_get", "history_deals_total",
})


def load_mt5_module() -> ModuleType:
    """Import MetaTrader5 or raise MT5UnavailableError with an actionable message."""
    if sys.platform != "win32":
        raise MT5UnavailableError(
            f"The MetaTrader5 Python package only works on Windows (this is {sys.platform!r}). "
            "Run MT5 verification on the Windows machine (see docs/WINDOWS_MT5_VERIFICATION.md), "
            "or use --fake for pipeline tests.")
    if struct.calcsize("P") * 8 != 64:
        raise MT5UnavailableError("MetaTrader5 requires 64-bit Python; this interpreter is 32-bit. "
                                  "Install the 64-bit Python and recreate the virtualenv.")
    try:
        import MetaTrader5  # type: ignore[import-not-found]
    except ImportError as exc:
        msg = str(exc)
        if "DLL load failed" in msg or "not a valid Win32" in msg:
            raise MT5UnavailableError(
                f"MetaTrader5 failed to load ({msg}). This usually means the wrong CPU architecture "
                "(ARM Windows) or a Python version without a matching wheel. Use 64-bit x86 Python 3.9-3.13.") from exc
        raise MT5UnavailableError(
            f"MetaTrader5 is not installed ({msg}). Run: pip install MetaTrader5") from exc
    return MetaTrader5


class ReadOnlyMT5Api:
    """Allow-list proxy around the MetaTrader5 module (or a fake with the same surface)."""

    def __init__(self, module: Any):
        object.__setattr__(self, "_module", module)

    def __getattr__(self, name: str) -> Any:
        if name in FORBIDDEN_FUNCTIONS:
            raise TradingDisabledError(
                f"MT5 function {name!r} is disabled in Phase 1 (read-only data foundation; no trading, "
                "no order checks, no account history). It will not be enabled without an approved phase.")
        if name in ALLOWED_FUNCTIONS or name.isupper():
            return getattr(self._module, name)
        raise AttributeError(f"MT5 function {name!r} is not on the Phase 1 read-only allow-list")

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        raise AttributeError("ReadOnlyMT5Api is immutable")

    def function(self, name: str) -> Callable[..., Any]:
        return getattr(self, name)
