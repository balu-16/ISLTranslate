from isl_slt.training.checkpoint import guarded_improvement
from isl_slt.training.checkpoint import inference_payload, load_checkpoint, save_checkpoint, tokenizer_hash
from isl_slt.training.checkpoint import validate_resume_compatibility
import pytest
import torch
from isl_slt.training.checkpoint import publish_snapshots


def test_code_override_cannot_bypass_environment_or_data():
    saved = {"architecture": "same", "execution_identity": {"code": {"hash": "old"}, "environment": {"lock": "same"}}}
    changed = {"architecture": "same", "execution_identity": {"code": {"hash": "new"}, "environment": {"lock": "same"}}}
    with pytest.raises(ValueError, match="code identity"):
        validate_resume_compatibility(saved, changed)
    assert validate_resume_compatibility(saved, changed, True)
    changed["execution_identity"]["environment"] = {"lock": "different"}
    with pytest.raises(ValueError, match="environment"):
        validate_resume_compatibility(saved, changed, True)


def test_inference_snapshot_retains_reconstruction_fields(tmp_path):
    payload = {"format": "isl-slt-checkpoint-v1", "tokenizer": {},
               "compatibility": {"tokenizer": tokenizer_hash({})},
               "input_dim": 384, "t5_config": {"d_model": 8}, "lexicon": None,
               "state": {"selected_beams": 1}, "optimizer": {}, "scheduler": {}, "scaler": {}, "rng": {}}
    snapshot = inference_payload(payload)
    assert not {"optimizer", "scheduler", "scaler", "rng"} & snapshot.keys()
    path = tmp_path / "best.pt"
    save_checkpoint(path, snapshot)
    loaded = load_checkpoint(path)
    assert loaded["input_dim"] == 384
    assert loaded["t5_config"] == {"d_model": 8}
    assert loaded["state"]["selected_beams"] == 1


def test_guard_uses_historical_chrf_maximum():
    assert guarded_improvement(11, 29.5, 10, 30)
    assert not guarded_improvement(11, 29.49, 10, 30)
    assert not guarded_improvement(10, 31, 10, 30)
    assert not guarded_improvement(float("nan"), 30, 10, 30)


def test_identical_best_snapshots_share_disk_and_replacement_is_independent(tmp_path):
    paths = [tmp_path / f"best-{name}.pt" for name in ("bleu", "chrf", "guarded")]
    payload = {"format": "isl-slt-checkpoint-v1", "tokenizer": {}, "compatibility": {"tokenizer": tokenizer_hash({})},
               "model": {"weight": torch.tensor([1.0])}, "state": {"selected_beams": 1}}
    modes = publish_snapshots(paths, payload)
    assert set(modes.values()) == {"serialized", "hardlink"}
    assert len({path.stat().st_ino for path in paths}) == 1
    payload["model"] = {"weight": torch.tensor([2.0])}
    publish_snapshots(paths[:1], payload)
    assert load_checkpoint(paths[1])["model"]["weight"].item() == 1.0
    publish_snapshots([tmp_path / "bf16.pt"], payload, "bf16")
    saved = load_checkpoint(tmp_path / "bf16.pt")
    assert saved["model"]["weight"].dtype == torch.bfloat16
    assert saved["snapshot_precision"]["lossy_conversion"]
    assert saved["state"]["selected_beams"] is None
