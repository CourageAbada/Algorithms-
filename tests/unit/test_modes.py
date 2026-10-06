import pytest

from fxscalp.core.modes import ModeConfigError, TradingMode, resolve_mode


def test_default_is_shadow():
    assert resolve_mode({}) is TradingMode.SHADOW


def test_unknown_mode_fails_closed():
    with pytest.raises(ModeConfigError):
        resolve_mode({"TRADING_MODE": "YOLO"})


def test_live_requires_all_interlocks():
    with pytest.raises(ModeConfigError):
        resolve_mode({"TRADING_MODE": "LIVE"})
    with pytest.raises(ModeConfigError):
        resolve_mode({"TRADING_MODE": "LIVE", "LIVE_TRADING_ENABLED": "true"})
    with pytest.raises(ModeConfigError):  # empty expected value must not match empty ack
        resolve_mode({"TRADING_MODE": "LIVE", "LIVE_TRADING_ENABLED": "true",
                      "LIVE_ACKNOWLEDGEMENT": "", "LIVE_ACKNOWLEDGEMENT_EXPECTED": ""})
    with pytest.raises(ModeConfigError):
        resolve_mode({"TRADING_MODE": "LIVE", "LIVE_TRADING_ENABLED": "true",
                      "LIVE_ACKNOWLEDGEMENT": "x", "LIVE_ACKNOWLEDGEMENT_EXPECTED": "y"})


def test_live_with_full_interlock():
    env = {"TRADING_MODE": "live", "LIVE_TRADING_ENABLED": "true",
           "LIVE_ACKNOWLEDGEMENT": "I-ACCEPT", "LIVE_ACKNOWLEDGEMENT_EXPECTED": "I-ACCEPT"}
    assert resolve_mode(env) is TradingMode.LIVE


@pytest.mark.parametrize("mode,expected", [
    ("BACKTEST", False), ("REPLAY", False), ("SHADOW", False), ("DEMO", True), ("LIVE", True)])
def test_only_demo_live_may_send_orders(mode, expected):
    assert TradingMode(mode).may_send_orders is expected
