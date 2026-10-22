from pathlib import Path

import pytest

from isl_slt.config import load_config


def test_overfit_is_unregularized_and_unfrozen():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/overfit.yaml")
    assert config.model.dropout == 0
    assert config.training.weight_decay == config.training.pose_noise_std == config.lexical.weight == 0
    assert len(config.training.stages) == 1
    assert not config.training.stages[0].freeze_t5
    assert config.generation.beams == [1]
    assert config.training.debug_only


def test_debug_training_rejects_missing_uids_before_data_loading(monkeypatch):
    from isl_slt.training.engine import run_training
    def forbidden(*args):
        raise AssertionError("Split loading must not occur")
    monkeypatch.setattr("isl_slt.training.engine.checked_splits", forbidden)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/overfit.yaml")
    with pytest.raises(ValueError, match="overfit --uids"):
        run_training(config)
