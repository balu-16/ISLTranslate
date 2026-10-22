from pathlib import Path

import numpy as np
import pytest

from isl_slt.config import Config
from isl_slt.data.dataset import PoseDataset
from isl_slt.features.cache import read_features, write_features
from isl_slt.features.pose import Prepared
from isl_slt.utils.io import file_hash


def cached_example(tmp_path, record_factory):
    record = record_factory()
    Path(record.pose_path).write_bytes(b"synthetic raw bytes")
    record.asset_hashes = {"pose": file_hash(record.pose_path)}
    config = Config()
    config.data.feature_dir = str(tmp_path / "features")
    valid = np.ones((2, 1), bool)
    prepared = Prepared(np.ones((2, 2), np.float32), np.array([0., .04]), valid, ~valid, valid,
                        valid, np.zeros(2, int), np.zeros(2, int), np.ones((2, 2), bool),
                        {"input_dim": 2, "continuous_channels": [0, 1]})
    write_features(config, record, prepared, [record])
    return record, config


def test_validated_cache_survives_raw_removal(tmp_path, record_factory):
    record, config = cached_example(tmp_path, record_factory)
    Path(record.pose_path).unlink()
    dataset = PoseDataset([record], config)
    assert len(dataset) == 1
    assert dataset[0]["features"].shape == (2, 2)
    with pytest.raises(FileNotFoundError):
        read_features(config, record, verify_assets=True)


def test_manifest_changes_cannot_reuse_cache(tmp_path, record_factory):
    record, config = cached_example(tmp_path, record_factory)
    changed = record.model_copy(update={"fps": 999.})
    with pytest.raises(ValueError, match="metadata changed"):
        read_features(config, changed)
