from pathlib import Path

import pytest

from fxscalp.core.config import ConfigError, load_instrument, load_yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["xauusd", "eurusd", "gbpusd"])
def test_instrument_configs_load(name):
    cfg = load_instrument(ROOT / "configs" / "instruments" / f"{name}.yaml")
    assert cfg["canonical"] == name[:3].upper() + "_" + name[3:].upper()
    assert len(cfg["aliases"]) >= 2  # discovery aliases, never a single hardcoded broker symbol


def test_base_config_defaults_to_shadow():
    assert load_yaml(ROOT / "configs" / "base.yaml")["mode_default"] == "SHADOW"


def test_missing_keys_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("canonical: X\n")
    with pytest.raises(ConfigError):
        load_instrument(p)
