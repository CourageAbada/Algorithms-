import pytest

from fxscalp.brokers.base import BrokerAdapter


def test_adapter_is_abstract():
    with pytest.raises(TypeError):
        BrokerAdapter()  # type: ignore[abstract]
