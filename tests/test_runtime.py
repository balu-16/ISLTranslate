import torch

from isl_slt.utils.runtime import seed_everything


def test_strict_workspace_policy(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        seed_everything(7, True)
        import os
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        seed_everything(7, True, True)
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
def test_environment_probe_includes_local_t5_and_configured_precision():
    from isl_slt.config import Runtime
    from isl_slt.utils.runtime import environment_report
    report = environment_report(Runtime(device="cpu", precision="fp32"))
    assert report["probes"]["fp32"]["ok"]
    assert report["t5_probes"]["fp32"]["ok"]
    assert report["t5_probes"]["fp32"]["pretrained_weights"] is False
    assert report["configured_precision_ok"]
    assert not report["t5_probes"]["fp16"]["ok"]


