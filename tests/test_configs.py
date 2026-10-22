from pathlib import Path

from isl_slt.config import load_config


def test_overfit_is_unregularized_and_unfrozen():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/overfit.yaml")
    assert config.model.dropout == 0
    assert config.training.weight_decay == config.training.pose_noise_std == config.lexical.weight == 0
    assert len(config.training.stages) == 1
    assert not config.training.stages[0].freeze_t5
    assert config.generation.beams == [1]
