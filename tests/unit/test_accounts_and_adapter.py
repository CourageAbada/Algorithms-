import logging
import threading

import pytest

from fxscalp.brokers.accounts import (AccountClass, AccountIdentity, assert_account_allowed, classify_trade_mode,
                                      mask_login)
from fxscalp.brokers.errors import (AccountMismatchError, AccountNotAllowedError, CallTimeoutError, ConnectionLostError,
                                    MT5AuthError, MT5InitializationError, MT5UnavailableError, NoTickDataError,
                                    StaleTickError, SymbolNotFoundError, TradingDisabledError)
from fxscalp.brokers.mt5 import AdapterState, HealthMonitor, MT5BrokerAdapter, Mt5Credentials
from fxscalp.brokers.mt5.api import ReadOnlyMT5Api, load_mt5_module
from fxscalp.brokers.mt5.fake import FakeMT5Adapter, FakeMT5Module, FakeScenario


@pytest.mark.parametrize("mode,cls", [(0, AccountClass.DEMO), (1, AccountClass.CONTEST), (2, AccountClass.REAL),
                                       (3, AccountClass.UNKNOWN), (-1, AccountClass.UNKNOWN), (None, AccountClass.UNKNOWN),
                                       ("x", AccountClass.UNKNOWN), (True, AccountClass.UNKNOWN)])
def test_classification(mode, cls):
    assert classify_trade_mode(mode) is cls


def _ident(cls):
    return AccountIdentity("***678", "fp", "S", "C", cls, None, None, None, "USD")


@pytest.mark.parametrize("cls", [AccountClass.REAL, AccountClass.CONTEST, AccountClass.UNKNOWN])
def test_gate_rejects_everything_but_demo(cls):
    with pytest.raises(AccountNotAllowedError):
        assert_account_allowed(_ident(cls))
    assert_account_allowed(_ident(AccountClass.DEMO))


def test_mask_login():
    assert mask_login(12345678) == "***678" and mask_login(None) == "***" and "1234" not in mask_login(12345678)


# ---------------- lifecycle / persistence ----------------
def test_persistent_session_single_initialize_single_owner_thread(adapter):
    for _ in range(25):
        adapter.get_latest_tick("XAUUSDm")
        adapter.get_symbol_info("XAUUSDm")
    adapter.get_ticks_range("XAUUSDm", 1_760_000_000, 1_760_003_600)
    f = adapter.fake
    assert f.calls.count("initialize") == 1 and "shutdown" not in f.calls
    assert len(f.threads) == 1 and threading.get_ident() not in f.threads    # all calls on ONE non-caller thread
    adapter.disconnect()
    assert f.calls.count("shutdown") == 1 and adapter.state is AdapterState.CLOSED


def test_concurrent_callers_are_serialised_on_the_owner_thread(adapter):
    errors, results = [], []

    def work():
        try:
            for _ in range(20):
                results.append(adapter.get_latest_tick("XAUUSDm").bid)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and len(results) == 120 and len(adapter.fake.threads) == 1


@pytest.mark.parametrize("mode,state", [(2, AdapterState.REJECTED), (1, AdapterState.REJECTED), (9, AdapterState.REJECTED)])
def test_non_demo_accounts_rejected_and_session_closed(mode, state):
    a = FakeMT5Adapter(FakeScenario(trade_mode=mode))
    with pytest.raises(AccountNotAllowedError) as e:
        a.connect()
    assert a.state is state and "DEMO" in str(e.value)
    assert a.fake.calls[-1] == "shutdown"                            # session closed immediately
    assert "symbols_get" not in a.fake.calls and "copy_ticks_range" not in a.fake.calls   # no data collected
    with pytest.raises(ConnectionLostError):
        a.get_latest_tick("XAUUSDm")


def test_real_account_never_retried_by_reconnect():
    a = FakeMT5Adapter(FakeScenario(trade_mode=2))
    with pytest.raises(AccountNotAllowedError):
        a.reconnect(max_attempts=3, backoff_s=0)
    assert a.fake.calls.count("initialize") == 1


def test_mt5_unavailable_wrong_platform_and_missing_package(monkeypatch):
    def boom():
        raise MT5UnavailableError("wrong platform")
    a = MT5BrokerAdapter(module_loader=boom)
    with pytest.raises(MT5UnavailableError):
        a.connect()
    monkeypatch.setattr("sys.platform", "linux")
    with pytest.raises(MT5UnavailableError, match="Windows"):
        load_mt5_module()


def test_wrong_architecture_and_import_errors(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("struct.calcsize", lambda fmt: 4)
    with pytest.raises(MT5UnavailableError, match="64-bit"):
        load_mt5_module()
    monkeypatch.undo()
    monkeypatch.setattr("sys.platform", "win32")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "MetaTrader5":
            raise ImportError("DLL load failed while importing _core")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(MT5UnavailableError, match="architecture"):
        load_mt5_module()


def test_initialization_failure_is_actionable():
    a = FakeMT5Adapter(FakeScenario(init_ok=False))
    with pytest.raises(MT5InitializationError, match="terminal"):
        a.connect()
    assert a.state is AdapterState.DISCONNECTED


def test_invalid_credentials():
    a = FakeMT5Adapter(FakeScenario(auth_ok=False))
    with pytest.raises(MT5AuthError, match="DEMO"):
        a.connect()


def test_account_mismatch_refused():
    a = FakeMT5Adapter(FakeScenario(login=11111111), credentials=Mt5Credentials(login=99999999, password="x"))
    with pytest.raises(AccountMismatchError):
        a.connect()
    assert a.state is not AdapterState.CONNECTED


def test_terminal_not_connected_to_server():
    a = FakeMT5Adapter(FakeScenario(terminal_connected=False))
    with pytest.raises(MT5InitializationError, match="not connected"):
        a.connect()


# ---------------- read API failures ----------------
def test_symbol_missing(adapter):
    with pytest.raises(SymbolNotFoundError):
        adapter.get_symbol_info("NOPE")
    with pytest.raises(SymbolNotFoundError):
        adapter.get_latest_tick("NOPE")


def test_no_tick_data(adapter):
    a = FakeMT5Adapter(FakeScenario(no_ticks_symbols=("XAUUSDm",)))
    a.connect()
    with pytest.raises(NoTickDataError):
        a.get_latest_tick("XAUUSDm")
    with pytest.raises(NoTickDataError):
        a.get_ticks_range("XAUUSDm", 1_760_000_000, 1_760_003_600)


def test_stale_latest_tick():
    a = FakeMT5Adapter(FakeScenario(latest_tick_age_s=900))
    a.connect()
    with pytest.raises(StaleTickError):
        a.get_latest_tick("XAUUSDm", max_age_s=60, offset_s=3 * 3600)
    assert a.get_latest_tick("XAUUSDm", max_age_s=3600, offset_s=3 * 3600).bid > 0


def test_stale_tick_with_wrong_offset_is_detected(adapter):
    with pytest.raises(StaleTickError):
        adapter.get_latest_tick("XAUUSDm", max_age_s=60, offset_s=0)   # 3 h offset not applied -> looks 3 h in the future/past


def test_historical_request_empty_is_empty_array_not_error(adapter):
    arr = adapter.get_ticks_range("XAUUSDm", 1_760_184_000 + 3 * 3600, 1_760_187_600 + 3 * 3600)   # weekend (closed)
    assert len(arr) == 0


def test_corrupt_return_type(adapter):
    adapter.fake.sc.corrupt_return_type = True
    from fxscalp.brokers.errors import DataFormatError
    with pytest.raises(DataFormatError):
        adapter.get_ticks_range("XAUUSDm", 1_760_000_000, 1_760_003_600)


def test_connection_loss_marks_session_lost_and_reconnect_recovers():
    a = FakeMT5Adapter()
    a.connect()
    a.get_latest_tick("XAUUSDm")
    a.fake.sc.fail_after_calls = len(a.fake.calls)         # every later call fails with an IPC error
    with pytest.raises(ConnectionLostError):
        a.get_latest_tick("XAUUSDm")
    with pytest.raises(ConnectionLostError):
        a.get_ticks_range("XAUUSDm", 1_760_000_000, 1_760_003_600)
    assert a.state is AdapterState.LOST
    assert not a.health_check().ok
    a.fake.sc.fail_after_calls = None
    a.reconnect(max_attempts=2, backoff_s=0)
    assert a.state is AdapterState.CONNECTED and a.get_latest_tick("XAUUSDm").bid > 0


def test_hung_call_times_out_and_session_is_lost():
    a = FakeMT5Adapter(FakeScenario(hang_function="symbol_info_tick", hang_s=1.0), call_timeout_s=0.2)
    a.connect()
    with pytest.raises(CallTimeoutError):
        a.get_latest_tick("XAUUSDm")
    assert a.state is AdapterState.LOST
    with pytest.raises(ConnectionLostError):
        a.get_symbol_info("XAUUSDm")


def test_health_monitor_flags_degradation():
    a = FakeMT5Adapter()
    a.connect()
    seen = []
    m = HealthMonitor(a, interval_s=999, failures_to_degrade=2, on_degraded=seen.append)
    assert m.check_once().ok and not seen
    a.fake.sc.fail_after_calls = len(a.fake.calls)
    m.check_once()
    m.check_once()
    assert len(seen) >= 1 and not m.last.ok


# ---------------- NO TRADING ----------------
def test_trading_methods_raise_trading_disabled(adapter):
    for fn, args in ((adapter.submit_order, (None,)), (adapter.modify_order, ("1", None, None)),
                     (adapter.cancel_order, ("1",)), (adapter.close_position, ("1",)),
                     (adapter.get_positions, ()), (adapter.get_orders, ())):
        with pytest.raises(TradingDisabledError):
            fn(*args)


@pytest.mark.parametrize("fn", ["order_send", "order_check", "order_calc_margin", "Buy", "Sell", "Close", "login",
                                "positions_get", "orders_get", "history_deals_get"])
def test_readonly_api_blocks_trading_functions(fn):
    api = ReadOnlyMT5Api(FakeMT5Module())
    with pytest.raises(TradingDisabledError):
        getattr(api, fn)
    with pytest.raises(AttributeError):
        api.some_unknown_function


def test_full_workflow_never_touches_trading_functions(adapter):
    adapter.discover_symbols()
    adapter.get_terminal_info()
    adapter.get_account_info()
    adapter.get_symbol_info("XAUUSDm")
    adapter.get_latest_tick("XAUUSDm")
    adapter.get_ticks_range("XAUUSDm", 1_760_000_000, 1_760_003_600)
    adapter.get_ticks_from("XAUUSDm", 1_760_000_000, 5)
    adapter.get_rates_range("XAUUSDm", "M1", 1_760_000_000, 1_760_003_600)
    adapter.get_market_book("XAUUSDm")
    adapter.health_check()
    assert adapter.fake.trading_calls == []
    assert not (set(adapter.fake.calls) & {"order_send", "order_check", "positions_get", "orders_get"})


# ---------------- secrets ----------------
def test_credentials_repr_never_shows_password():
    c = Mt5Credentials(login=12345678, password="S3cretPass!", server="Srv")
    assert "S3cretPass" not in repr(c) and "12345678" not in repr(c) and "S3cretPass" not in str(c)


def test_logs_never_contain_credentials(capsys):
    from fxscalp.monitoring.logging_setup import configure_logging
    configure_logging("INFO")
    a = FakeMT5Adapter(FakeScenario(login=12345678), credentials=Mt5Credentials(login=12345678, password="S3cretPass!", server="FakeBroker-Demo"))
    a.connect()
    a.get_latest_tick("XAUUSDm")
    with pytest.raises(AccountMismatchError):
        FakeMT5Adapter(FakeScenario(login=1), credentials=Mt5Credentials(login=12345678, password="S3cretPass!")).connect()
    a.disconnect()
    out = capsys.readouterr().out
    assert "mt5" in out                      # logging actually happened
    assert "S3cretPass" not in out and "12345678" not in out


def test_redacting_logger_still_redacts_login_in_messages():
    import json
    from fxscalp.monitoring.logging_setup import JsonFormatter
    rec = logging.LogRecord("t", logging.INFO, "f", 1, "connect login=12345678 password=abc", None, None)
    msg = json.loads(JsonFormatter().format(rec))["msg"]
    assert "12345678" not in msg and "abc" not in msg
