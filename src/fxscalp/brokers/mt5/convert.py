"""MT5 structures -> neutral types. The conversion layer.

Rules
-----
* No unit conversion and no unit assumption: values are carried exactly as the terminal returned
  them; ``raw`` keeps every field. Typed attributes are a projection, ``None`` when a field is absent
  or not convertible.
* Whether numbers mean what we expect (e.g. tick_value vs contract_size * tick_size) is *checked*,
  not assumed: see ``check_symbol_consistency``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fxscalp.brokers.accounts import AccountIdentity, classify_trade_mode, login_fingerprint, mask_login
from fxscalp.brokers.base import AccountInfo, RAW_TICK_DTYPE, SymbolInfo, TerminalInfo, Tick
from fxscalp.brokers.errors import DataFormatError

REQUIRED_TICK_FIELDS = ("time", "bid", "ask", "last", "volume", "time_msc", "flags")
RAW_RATE_DTYPE = np.dtype([("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
                           ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")])
_SECRET_FIELDS = {"password"}  # defensive: never copy if a struct ever carried one


def struct_to_dict(obj: Any) -> dict[str, Any]:
    """Dict of all fields of an MT5 struct (PyStructSequence/namedtuple/mapping), secrets removed."""
    if obj is None:
        raise DataFormatError("terminal returned None where a structure was expected")
    if hasattr(obj, "_asdict"):
        d = dict(obj._asdict())
    elif isinstance(obj, Mapping):
        d = dict(obj)
    else:
        raise DataFormatError(f"unsupported structure type {type(obj).__name__}")
    return {k: v for k, v in d.items() if k not in _SECRET_FIELDS}


def _f(d: Mapping[str, Any], k: str) -> float | None:
    try:
        v = d.get(k)
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _i(d: Mapping[str, Any], k: str) -> int | None:
    try:
        v = d.get(k)
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _b(d: Mapping[str, Any], k: str) -> bool | None:
    v = d.get(k)
    return None if v is None else bool(v)


def _s(d: Mapping[str, Any], k: str) -> str:
    v = d.get(k)
    return "" if v is None else str(v)


def to_terminal_info(obj: Any, version: Any = None) -> TerminalInfo:
    d = struct_to_dict(obj)
    ver = None
    if version is not None:
        try:
            ver = tuple(int(x) for x in version)
        except (TypeError, ValueError):
            ver = None
    return TerminalInfo(connected=bool(d.get("connected", False)), trade_allowed=_b(d, "trade_allowed"),
                        ping_last_us=_i(d, "ping_last"), build=_i(d, "build"), company=_s(d, "company"),
                        name=_s(d, "name"), version=ver, raw={k: _jsonable(v) for k, v in d.items()
                                                              if k not in {"path", "data_path", "commondata_path"}})


def to_account_info(obj: Any) -> AccountInfo:
    d = struct_to_dict(obj)
    login = d.get("login")
    ident = AccountIdentity(
        login_masked=mask_login(login), login_fingerprint=login_fingerprint(login, _s(d, "server")),
        server=_s(d, "server"), company=_s(d, "company"),
        account_class=classify_trade_mode(d.get("trade_mode")), trade_mode_raw=_i(d, "trade_mode"),
        margin_mode_raw=_i(d, "margin_mode"), leverage=_i(d, "leverage"), currency=_s(d, "currency"))
    return AccountInfo(identity=ident, balance=_f(d, "balance") or 0.0, equity=_f(d, "equity") or 0.0,
                       margin=_f(d, "margin") or 0.0, margin_free=_f(d, "margin_free") or 0.0,
                       raw_fields_present=tuple(sorted(k for k in d if k not in {"login", "name"})))


def to_symbol_info(obj: Any, canonical: str | None = None) -> SymbolInfo:
    d = struct_to_dict(obj)
    return SymbolInfo(
        canonical=canonical, broker_symbol=_s(d, "name"), digits=_i(d, "digits"), point=_f(d, "point"),
        tick_size=_f(d, "trade_tick_size"), tick_value=_f(d, "trade_tick_value"),
        tick_value_profit=_f(d, "trade_tick_value_profit"), tick_value_loss=_f(d, "trade_tick_value_loss"),
        contract_size=_f(d, "trade_contract_size"), volume_min=_f(d, "volume_min"), volume_max=_f(d, "volume_max"),
        volume_step=_f(d, "volume_step"), volume_limit=_f(d, "volume_limit"),
        stops_level=_i(d, "trade_stops_level"), freeze_level=_i(d, "trade_freeze_level"),
        spread=_i(d, "spread"), spread_float=_b(d, "spread_float"), trade_mode=_i(d, "trade_mode"),
        trade_exemode=_i(d, "trade_exemode"), trade_calc_mode=_i(d, "trade_calc_mode"),
        filling_mode=_i(d, "filling_mode"), order_mode=_i(d, "order_mode"), expiration_mode=_i(d, "expiration_mode"),
        swap_mode=_i(d, "swap_mode"), swap_long=_f(d, "swap_long"), swap_short=_f(d, "swap_short"),
        swap_rollover3days=_i(d, "swap_rollover3days"), currency_base=_s(d, "currency_base"),
        currency_profit=_s(d, "currency_profit"), currency_margin=_s(d, "currency_margin"),
        description=_s(d, "description"), path=_s(d, "path"), visible=_b(d, "visible"), select=_b(d, "select"),
        ticks_bookdepth=_i(d, "ticks_bookdepth"), raw={k: _jsonable(v) for k, v in d.items()})


def to_tick(obj: Any) -> Tick:
    d = struct_to_dict(obj)
    missing = [k for k in REQUIRED_TICK_FIELDS if k not in d]
    if missing:
        raise DataFormatError(f"tick structure missing fields {missing}; terminal/package version mismatch?")
    return Tick(time_raw=int(d["time"]), time_msc_raw=int(d["time_msc"]), bid=float(d["bid"]), ask=float(d["ask"]),
                last=float(d["last"]), volume=int(d["volume"]), volume_real=float(d.get("volume_real", float("nan"))),
                flags=int(d["flags"]))


def ticks_to_array(arr: Any) -> np.ndarray:
    """Normalise a terminal tick array to RAW_TICK_DTYPE without altering values or order."""
    if not isinstance(arr, np.ndarray) or arr.dtype.names is None:
        raise DataFormatError(f"expected a structured numpy array of ticks, got {type(arr).__name__}")
    missing = [k for k in REQUIRED_TICK_FIELDS if k not in arr.dtype.names]
    if missing:
        raise DataFormatError(f"tick array missing fields {missing}; present: {list(arr.dtype.names)}")
    out = np.empty(len(arr), dtype=RAW_TICK_DTYPE)
    for name in RAW_TICK_DTYPE.names:
        if name in arr.dtype.names:
            out[name] = arr[name]
        else:
            out[name] = np.nan  # volume_real absent on some builds: recorded as NaN, not 0
    return out


def rates_to_array(arr: Any) -> np.ndarray:
    if not isinstance(arr, np.ndarray) or arr.dtype.names is None:
        raise DataFormatError(f"expected a structured numpy array of bars, got {type(arr).__name__}")
    missing = [k for k in RAW_RATE_DTYPE.names if k not in arr.dtype.names]
    if missing:
        raise DataFormatError(f"bar array missing fields {missing}")
    out = np.empty(len(arr), dtype=RAW_RATE_DTYPE)
    for name in RAW_RATE_DTYPE.names:
        out[name] = arr[name]
    return out


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    return v


# --------------------------------------------------------------------------------------------
# Consistency checks (units are checked, not assumed)
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ConsistencyIssue:
    severity: str      # "warn" | "error"
    code: str
    message: str


def check_symbol_consistency(info: SymbolInfo, account_currency: str | None = None,
                             rel_tol: float = 0.01) -> list[ConsistencyIssue]:
    issues: list[ConsistencyIssue] = []

    def add(sev: str, code: str, msg: str) -> None:
        issues.append(ConsistencyIssue(sev, code, msg))

    if info.point is None or info.point <= 0:
        add("error", "POINT_INVALID", f"point={info.point!r}")
    elif info.digits is not None and abs(info.point - 10.0 ** (-info.digits)) > 1e-12 * max(1.0, 1 / info.point):
        add("warn", "POINT_NE_10_POW_MINUS_DIGITS", f"point={info.point} but digits={info.digits}")
    if info.tick_size is None or info.tick_size <= 0:
        add("error", "TICK_SIZE_INVALID", f"tick_size={info.tick_size!r}")
    elif info.point and info.point > 0:
        ratio = info.tick_size / info.point
        if abs(ratio - round(ratio)) > 1e-6:
            add("warn", "TICK_SIZE_NOT_MULTIPLE_OF_POINT", f"tick_size/point={ratio}")
    if info.contract_size is None or info.contract_size <= 0:
        add("error", "CONTRACT_SIZE_INVALID", f"contract_size={info.contract_size!r}")
    if info.tick_value is None or info.tick_value <= 0:
        add("error", "TICK_VALUE_INVALID", f"tick_value={info.tick_value!r}")
    elif (info.contract_size and info.tick_size and account_currency
          and info.currency_profit.upper() == account_currency.upper()):
        expected = info.contract_size * info.tick_size
        if abs(info.tick_value - expected) > rel_tol * expected:
            add("warn", "TICK_VALUE_VS_CONTRACT", f"tick_value={info.tick_value} but contract_size*tick_size="
                f"{expected} although profit currency equals account currency: units/conversion differ from "
                "the simple expectation; do not size risk from the simple formula")
    elif account_currency and info.currency_profit and info.currency_profit.upper() != account_currency.upper():
        add("warn", "PROFIT_CCY_NE_ACCOUNT_CCY", f"profit currency {info.currency_profit} != account currency "
            f"{account_currency}: tick_value includes a conversion; cannot be cross-checked offline")
    if info.volume_step is None or info.volume_step <= 0:
        add("error", "VOLUME_STEP_INVALID", f"volume_step={info.volume_step!r}")
    else:
        if info.volume_min is None or info.volume_min <= 0:
            add("error", "VOLUME_MIN_INVALID", f"volume_min={info.volume_min!r}")
        elif abs(info.volume_min / info.volume_step - round(info.volume_min / info.volume_step)) > 1e-6:
            add("warn", "VOLUME_MIN_NOT_MULTIPLE_OF_STEP", f"min={info.volume_min} step={info.volume_step}")
        if info.volume_max is None or (info.volume_min and info.volume_max < info.volume_min):
            add("error", "VOLUME_MAX_INVALID", f"volume_max={info.volume_max!r}")
    for nm, v in (("stops_level", info.stops_level), ("freeze_level", info.freeze_level)):
        if v is None or v < 0:
            add("warn", f"{nm.upper()}_INVALID", f"{nm}={v!r}")
    for nm in ("currency_base", "currency_profit", "currency_margin"):
        if not getattr(info, nm):
            add("warn", f"{nm.upper()}_EMPTY", "field empty")
    return issues


def compare_spread_with_tick(info: SymbolInfo, tick: Tick) -> dict[str, Any]:
    """Informational: reported spread (points) vs the spread implied by the latest tick."""
    if not info.point or info.point <= 0:
        return {"comparable": False, "reason": "point unavailable"}
    implied = (tick.ask - tick.bid) / info.point
    return {"comparable": True, "reported_points": info.spread, "tick_implied_points": implied,
            "difference_points": None if info.spread is None else implied - info.spread,
            "spread_float": info.spread_float}
