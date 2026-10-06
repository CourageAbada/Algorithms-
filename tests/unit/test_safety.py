import pytest

from fxscalp.core.safety import FlattenNotAuthorised, SafetyState


def test_halt_blocks_new_trades_but_not_flatten():
    s = SafetyState()
    assert s.may_open_new_trades
    s.halt("spread abnormal")
    assert not s.may_open_new_trades
    assert s.halt_new_trades and not s.emergency_flatten


@pytest.mark.parametrize("token,expected", [(None, "secret"), ("wrong", "secret"), ("x", None), ("", ""), (None, None)])
def test_flatten_requires_valid_authorisation(token, expected):
    s = SafetyState()
    with pytest.raises(FlattenNotAuthorised):
        s.flatten("test", token=token, expected_token=expected)
    assert not s.emergency_flatten and s.may_open_new_trades


def test_flatten_with_token_also_halts():
    s = SafetyState()
    s.flatten("hard loss limit", token="secret", expected_token="secret")
    assert s.emergency_flatten and s.halt_new_trades and not s.may_open_new_trades
