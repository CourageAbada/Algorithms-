import datetime as dt
import json
import os
from pathlib import Path

import pytest

from fxscalp.features.dataset import DatasetError, build_from_store, read_ml_dataset
from fxscalp.features.pipeline import PipelineConfig
from fxscalp.features.spec import FeatureConfig
from fxscalp.market_data.store import ChunkKey, TickStore
from fxscalp.synthetic.xauusd import SynthConfig, generate_ticks, write_phase1_dataset

UTC = dt.timezone.utc
DAY = dt.date(2025, 10, 6)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("p1")
    cfg = SynthConfig(start_utc=dt.datetime(2025, 10, 6, 6, 0, tzinfo=UTC), duration_s=3 * 3600, base_rate_hz=0.8, seed=21,
                      dup_prob=0.001, out_of_order_prob=0.0005)
    man = write_phase1_dataset(root, generate_ticks(cfg), start_day=DAY, end_day=DAY)
    return TickStore(root), man, root


def test_build_from_phase1_store_with_full_provenance(store, tmp_path):
    st, man, root = store
    res = build_from_store(st, man, tmp_path)
    m = res.manifest
    assert m["labels"] is None and "no labels" in m["note"]
    assert m["raw_dataset"]["dataset_id"] == man["dataset_id"] and m["raw_dataset"]["manifest_checksum_sha256"] == man["manifest_checksum_sha256"]
    for k in ("ml_dataset_id", "schema_version", "feature_set_version", "pipeline_config", "config_hash", "code", "stages", "regime_calibration",
              "registry_groups", "feature_columns", "quality", "features_content_sha256"):
        assert k in m, k
    assert m["code"]["fxscalp"] and "python" in m["code"] and m["regime_calibration"]["spread"].startswith("UNCALIBRATED")
    names = [s["name"] for s in m["stages"]]
    assert names[0] == "tick_norm" and "features" in names and "bars_1s" in names and "bars_300s" in names
    for s in m["stages"]:
        assert len(s["sha256"]) == 64 and s["parents"] and (res.out_dir / s["file"]).exists()
    assert m["stages"][0]["parents"][0]["dataset_id"] == man["dataset_id"]
    assert [s for s in m["stages"] if s["name"] == "features"][0]["parents"][0]["sha256"] == m["stages"][0]["sha256"]
    assert m["quality"]["ok"], m["quality"]["errors"]
    df, m2 = read_ml_dataset(res.out_dir)
    assert len(df) == m["rows"] and m2["ml_dataset_id"] == m["ml_dataset_id"]
    assert (res.out_dir / "feature_quality_report.md").exists() and (res.out_dir / "feature_quality_report.json").exists()
    assert m["stages"][0]["normalization"]["n_quarantined_excluded"] > 0           # out-of-order ticks were quarantined & counted


def test_ml_dataset_id_is_deterministic_and_idempotent_but_config_sensitive(store, tmp_path):
    st, man, _ = store
    a = build_from_store(st, man, tmp_path, write_bars=False)
    b = build_from_store(st, man, tmp_path, write_bars=False)              # same inputs -> same id, existing identical output reused
    assert a.manifest["ml_dataset_id"] == b.manifest["ml_dataset_id"] and a.manifest["features_content_sha256"] == b.manifest["features_content_sha256"]
    other = build_from_store(st, man, tmp_path, PipelineConfig(features=FeatureConfig(micro_windows_s=(5, 15, 30))), write_bars=False)
    assert other.manifest["ml_dataset_id"] != a.manifest["ml_dataset_id"] and other.manifest["feature_set_version"] != a.manifest["feature_set_version"]
    c = build_from_store(st, man, tmp_path, PipelineConfig(grid_s=5), write_bars=False)
    assert c.manifest["ml_dataset_id"] != a.manifest["ml_dataset_id"] and c.manifest["rows"] < a.manifest["rows"]


def test_raw_data_is_not_modified_by_the_pipeline(store, tmp_path):
    st, man, root = store
    key = ChunkKey(man["broker"], man["server"], man["instrument"], DAY)
    before = (st.chunk_manifest(key)["file_sha256"], (st.raw_dir(key) / "ticks.parquet").stat().st_mtime_ns)
    build_from_store(st, man, tmp_path, write_bars=False)
    after = (st.chunk_manifest(key)["file_sha256"], (st.raw_dir(key) / "ticks.parquet").stat().st_mtime_ns)
    assert before == after and st.verify_chunk(key)[0]
    assert not any(str(tmp_path) in str(p) for p in [st.raw_dir(key)])         # derived output lives elsewhere


def test_hash_chain_breaks_are_refused(store, tmp_path):
    st, man, _ = store
    bad = json.loads(json.dumps(man))
    bad["chunks"][0]["raw_content_sha256"] = "0" * 64
    with pytest.raises(DatasetError, match="hash chain"):
        build_from_store(st, bad, tmp_path)


def test_unverified_time_basis_is_refused(store, tmp_path):
    st, man, _ = store
    bad = dict(man, time_basis="unverified")
    with pytest.raises(DatasetError, match="unverified"):
        build_from_store(st, bad, tmp_path)


def test_tampered_chunk_file_is_detected(tmp_path):
    root = tmp_path / "p1"
    cfg = SynthConfig(start_utc=dt.datetime(2025, 10, 6, 6, 0, tzinfo=UTC), duration_s=1800, seed=2)
    man = write_phase1_dataset(root, generate_ticks(cfg), start_day=DAY, end_day=DAY)
    st = TickStore(root)
    p = st.raw_dir(ChunkKey(man["broker"], man["server"], man["instrument"], DAY)) / "ticks.parquet"
    os.chmod(p, 0o644)
    p.write_bytes(p.read_bytes()[:-40])
    with pytest.raises(Exception):
        build_from_store(st, man, tmp_path / "out")


def test_unknown_point_flows_through_as_nan_units(tmp_path):
    root = tmp_path / "p1"
    cfg = SynthConfig(start_utc=dt.datetime(2025, 10, 6, 6, 0, tzinfo=UTC), duration_s=1800, seed=2)
    man = write_phase1_dataset(root, generate_ticks(cfg), start_day=DAY, end_day=DAY)
    meta_file = next((root / "metadata" / "symbol_info").rglob("*.json"))
    d = json.loads(meta_file.read_text())
    d["typed"]["point"] = None
    meta_file.write_text(json.dumps(d))
    res = build_from_store(TickStore(root), man, tmp_path / "out", write_bars=False)
    assert res.frame.df["spread_points"].isna().all() and res.frame.df["relative_spread_bps"].notna().any()
    assert res.manifest["stages"][0]["normalization"]["point_known"] is False


def test_build_features_script(store, tmp_path, capsys):
    from scripts import build_features
    st, man, root = store
    code = build_features.main(["--data-root", str(root), "--dataset-id", man["dataset_id"], "--out-root", str(tmp_path), "--no-bars"])
    out = capsys.readouterr().out
    assert code == 0 and "ml_dataset_id" in out and man["dataset_id"] in out and "UNCALIBRATED" in out
    assert build_features.main(["--data-root", str(root), "--dataset-id", "tickraw-nope", "--out-root", str(tmp_path)]) == 2


def test_script_needs_no_broker_or_mt5():
    import fxscalp.features.dataset as ds
    import sys
    src = Path(ds.__file__).read_text() + (Path(__file__).resolve().parents[2] / "scripts" / "build_features.py").read_text()
    assert "MetaTrader5" not in src and "brokers.mt5" not in src and "ReadOnlyBrokerAdapter" not in src
    assert "MetaTrader5" not in sys.modules


def test_export_registry_check_script():
    from scripts import export_feature_registry
    assert export_feature_registry.main(["--check"]) == 0
