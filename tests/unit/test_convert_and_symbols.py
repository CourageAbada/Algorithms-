import numpy as np
import pytest
import yaml

from fxscalp.brokers.base import RAW_TICK_DTYPE
from fxscalp.brokers.errors import AmbiguousSymbolError, DataFormatError, SymbolNotFoundError
from fxscalp.brokers.mt5 import convert
from fxscalp.brokers.mt5.fake import FakeMT5Module
from fxscalp.brokers.symbols import (InstrumentSpec, SymbolMap, build_map_document, discover, map_path, write_map)

XAU = InstrumentSpec("XAU_USD", ("XAUUSD", "XAU/USD", "XAU_USD", "GOLD"), "XAU", "USD")


def _info(name="XAUUSDm", **ov):
    m = FakeMT5Module()
    d = m._symbol(name)._asdict()
    d.update(ov)
    return convert.to_symbol_info(d)


# ---------------- conversion layer ----------------
def test_symbol_info_projection_and_raw_preserved():
    si = _info()
    assert (si.broker_symbol, si.digits, si.point, si.tick_size, si.contract_size) == ("XAUUSDm", 2, 0.01, 0.01, 100.0)
    assert si.volume_min == 0.01 and si.volume_step == 0.01 and si.spread_float is True
    assert si.currency_base == "XAU" and si.currency_profit == "USD"
    assert si.raw["trade_tick_size"] == 0.01 and "trade_calc_mode" in si.raw      # EVERY field kept, no unit conversion


def test_missing_and_unconvertible_fields_become_none_not_guesses():
    si = convert.to_symbol_info({"name": "X", "digits": "abc", "point": None})
    assert si.digits is None and si.point is None and si.tick_size is None and si.currency_base == ""


def test_secret_fields_never_copied():
    d = convert.struct_to_dict({"name": "a", "password": "p"})
    assert "password" not in d


def test_to_tick_requires_fields():
    with pytest.raises(DataFormatError):
        convert.to_tick({"time": 1, "bid": 1.0})
    t = convert.to_tick({"time": 1, "time_msc": 1500, "bid": 1.0, "ask": 1.1, "last": 0.0, "volume": 0, "flags": 6})
    assert t.volume_real != t.volume_real                                          # absent volume_real -> NaN, not 0


def test_ticks_to_array_validation_and_fidelity():
    with pytest.raises(DataFormatError):
        convert.ticks_to_array([(1, 2)])
    with pytest.raises(DataFormatError):
        convert.ticks_to_array(np.zeros(3, dtype=[("time", "<i8")]))
    src = np.zeros(2, dtype=[("flags", "<u4"), ("time", "<i8"), ("bid", "<f8"), ("ask", "<f8"), ("last", "<f8"),
                             ("volume", "<u8"), ("time_msc", "<i8")])               # different order, no volume_real
    src["time_msc"], src["bid"], src["ask"] = [5, 6], [1.0, 2.0], [1.5, 2.5]
    out = convert.ticks_to_array(src)
    assert out.dtype == RAW_TICK_DTYPE and list(out["time_msc"]) == [5, 6] and np.isnan(out["volume_real"]).all()


def test_rates_to_array_validation():
    with pytest.raises(DataFormatError):
        convert.rates_to_array(np.zeros(1, dtype=[("time", "<i8")]))


def test_consistency_clean_for_plausible_xau():
    assert convert.check_symbol_consistency(_info(), "USD") == []


def test_consistency_detects_unit_surprises():
    codes = {i.code for i in convert.check_symbol_consistency(_info(trade_tick_value=10.0), "USD")}
    assert "TICK_VALUE_VS_CONTRACT" in codes                                         # tick_value != contract*tick_size
    codes = {i.code for i in convert.check_symbol_consistency(_info(point=0.001), "USD")}
    assert "POINT_NE_10_POW_MINUS_DIGITS" in codes
    codes = {i.code for i in convert.check_symbol_consistency(_info(volume_min=0.015), "USD")}
    assert "VOLUME_MIN_NOT_MULTIPLE_OF_STEP" in codes
    codes = {i.code for i in convert.check_symbol_consistency(_info(trade_tick_size=0.0), "USD")}
    assert "TICK_SIZE_INVALID" in codes
    codes = {i.code for i in convert.check_symbol_consistency(_info(currency_profit="EUR"), "USD")}
    assert "PROFIT_CCY_NE_ACCOUNT_CCY" in codes                                      # cannot be cross-checked offline
    codes = {i.code for i in convert.check_symbol_consistency(_info(trade_contract_size=0.0), "USD")}
    assert "CONTRACT_SIZE_INVALID" in codes


def test_spread_comparison_is_informational():
    from fxscalp.brokers.base import Tick
    c = convert.compare_spread_with_tick(_info(spread=25), Tick(1, 1000, 2000.0, 2000.25, 0.0, 0, 0.0, 6))
    assert c["comparable"] and round(c["tick_implied_points"]) == 25 and abs(c["difference_points"]) < 1e-6


# ---------------- discovery ----------------
@pytest.mark.parametrize("names,chosen", [
    (["XAUUSDm", "EURUSDm"], "XAUUSDm"), (["XAUUSD.a"], "XAUUSD.a"), (["mXAUUSD"], "mXAUUSD"), (["GOLD"], "GOLD"),
    (["XAU/USD"], "XAU/USD"), (["xauusd#"], "xauusd#"), (["XAUUSD.pro", "XAUUSD"], "XAUUSD"), (["GOLDm"], "GOLDm"),
])
def test_discovery_finds_gold_variants(names, chosen):
    r = discover(XAU, names)
    assert r.ok and r.chosen == chosen


def test_discovery_is_conservative():
    assert discover(XAU, ["EURUSD", "GBPUSDm"]).status == "not_found"
    assert discover(XAU, ["GOLDEUR", "XAUEUR"]).status == "not_found"             # different instruments
    assert discover(XAU, ["XAUUSDm", "XAUUSDc"]).status == "ambiguous"
    assert discover(XAU, ["GOLD", "XAUUSD"]).status == "ambiguous"
    r = discover(XAU, ["XAUUSDfoo"])
    assert r.status == "needs_review" and r.chosen is None


def test_discovery_rejects_wrong_currency_and_honours_pin():
    wrong = _info("XAUUSDm", currency_profit="EUR")
    assert discover(XAU, [wrong]).status == "not_found"
    both = [_info("XAUUSDm"), _info("XAUUSDc")]
    assert discover(XAU, both).status == "ambiguous"
    r = discover(XAU, both, pin="XAUUSDc")
    assert r.status == "pinned" and r.chosen == "XAUUSDc"
    assert discover(XAU, both, pin="nope").status == "not_found"


def test_eurusd_gbpusd_discovery():
    eur = InstrumentSpec("EUR_USD", ("EURUSD", "EUR/USD"), "EUR", "USD")
    gbp = InstrumentSpec("GBP_USD", ("GBPUSD", "GBP/USD"), "GBP", "USD")
    names = ["XAUUSDm", "EURUSDm", "GBPUSDm", "EURGBPm"]
    assert discover(eur, names).chosen == "EURUSDm" and discover(gbp, names).chosen == "GBPUSDm"


def test_mapping_is_deterministic_and_order_independent(tmp_path):
    a = ["XAUUSDm", "EURUSDm", "GBPUSDm"]
    d1 = build_map_document("B", "S", [discover(XAU, a)])
    d2 = build_map_document("B", "S", [discover(XAU, list(reversed(a)))])
    assert yaml.safe_dump(d1, sort_keys=True) == yaml.safe_dump(d2, sort_keys=True)
    p = map_path(tmp_path, "Broker Ltd", "Srv-Demo")
    write_map(p, d1)
    assert p.read_bytes() == (write_map(p, d2) or p.read_bytes())                  # idempotent bytes
    assert SymbolMap.load(p).broker_symbol("XAU_USD") == "XAUUSDm"                 # canonical -> broker, from config only


def test_symbol_map_errors_are_actionable(tmp_path):
    p = map_path(tmp_path, "B", "S")
    write_map(p, build_map_document("B", "S", [discover(XAU, ["XAUUSDm", "XAUUSDc"]),
                                               discover(InstrumentSpec("EUR_USD", ("EURUSD",)), ["XAUUSDm"])]))
    m = SymbolMap.load(p)
    with pytest.raises(AmbiguousSymbolError):
        m.broker_symbol("XAU_USD")
    with pytest.raises(SymbolNotFoundError):
        m.broker_symbol("EUR_USD")
    with pytest.raises(SymbolNotFoundError):
        m.broker_symbol("UNKNOWN")


def test_instrument_configs_provide_discovery_specs():
    from pathlib import Path
    from fxscalp.workflows import load_instrument_specs
    specs = load_instrument_specs(Path(__file__).resolve().parents[2] / "configs")
    assert set(specs) == {"XAU_USD", "EUR_USD", "GBP_USD"} and "GOLD" in specs["XAU_USD"].aliases
