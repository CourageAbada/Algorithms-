import datetime as dt

import pytest

from fxscalp.brokers.mt5.fake import FakeMT5Adapter
from fxscalp.market_data.calibrate import collect_offset_samples
from fxscalp.market_data.timebase import TimeBase, estimate_offset, resolve_time_basis


@pytest.fixture
def adapter():
    a = FakeMT5Adapter()
    a.connect()
    yield a
    a.disconnect()


def make_timebase(adapter, symbol="XAUUSDm"):
    est = estimate_offset(collect_offset_samples(adapter, symbol, 8, 0.01), min_samples=5)
    assert est.status == "ok", est
    return TimeBase(resolve_time_basis(est, None, broker="Fake Broker Ltd", server="FakeBroker-Demo"))


@pytest.fixture
def timebase(adapter):
    return make_timebase(adapter)


DAY0 = dt.date(2025, 10, 6)   # a Monday


import numpy as np

from fxscalp.brokers.base import RAW_TICK_DTYPE


def make_ticks(n=100, start_msc=1_760_000_000_000, step_ms=500, bid0=2000.0, spread=0.25, seed=3):
    rng = np.random.default_rng(seed)
    a = np.empty(n, dtype=RAW_TICK_DTYPE)
    a["time_msc"] = start_msc + np.arange(n) * step_ms
    a["time"] = a["time_msc"] // 1000
    a["bid"] = np.round(bid0 + rng.normal(0, 0.05, n).cumsum(), 2)
    a["ask"] = np.round(a["bid"] + spread, 2)
    a["last"], a["volume"], a["volume_real"], a["flags"] = 0.0, 0, 0.0, 6
    return a
