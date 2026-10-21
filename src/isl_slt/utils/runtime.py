"""Backend-neutral precision and reproducibility. Operations run only on explicit calls."""

import random
import time
from contextlib import nullcontext

import numpy as np
import torch

from isl_slt.config import Runtime


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False


def device_for(runtime: Runtime) -> torch.device:
    name = "cuda" if runtime.device == "auto" and torch.cuda.is_available() else runtime.device
    if name == "auto":
        name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU PyTorch is unavailable; install the appropriate uv backend extra")
    return torch.device(name)


def autocast(runtime: Runtime, device: torch.device):
    if runtime.precision == "fp32":
        return nullcontext()
    if runtime.precision == "fp16" and device.type == "cpu":
        raise ValueError("Use FP32 or BF16 on CPU")
    if runtime.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This backend does not report BF16 support; select FP32/FP16 explicitly")
    return torch.autocast(device.type, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[runtime.precision])


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "gpu": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["gpu"] is not None:
        if not torch.cuda.is_available() or len(state["gpu"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the saved GPU topology")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["gpu"]])


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Budget:
    def __init__(self, seconds: float | None):
        self.seconds, self.started = seconds, time.monotonic()

    @property
    def expired(self) -> bool:
        return self.seconds is not None and time.monotonic() - self.started >= self.seconds


def environment_report(runtime: Runtime) -> dict:
    device = device_for(runtime)
    report = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
              "device": str(device), "precision": runtime.precision,
              "recurrent_fp32_policy": runtime.recurrent_fp32, "probes": {}}
    if device.type == "cuda":
        report.update(name=torch.cuda.get_device_name(device), memory_bytes=torch.cuda.get_device_properties(
            device).total_memory, bf16=torch.cuda.is_bf16_supported())
    for precision in ("fp32", "bf16", "fp16"):
        try:
            policy = runtime.model_copy(update={"precision": precision})
            layer = torch.nn.LSTM(4, 4, num_layers=2, bidirectional=True, batch_first=True).to(device)
            projection = torch.nn.Linear(4, 4).to(device)
            with autocast(policy, device):
                values = projection(torch.ones(2, 3, 4, device=device))
                recurrent_context = torch.autocast(device.type, enabled=False) if runtime.recurrent_fp32 else nullcontext()
                with recurrent_context:
                    packed = torch.nn.utils.rnn.pack_padded_sequence(values.float() if runtime.recurrent_fp32 else values,
                        torch.tensor([3, 2]), batch_first=True, enforce_sorted=False)
                    output, _ = layer(packed)
                    output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
                output.float().sum().backward()
            synchronize(device)
            report["probes"][precision] = {"ok": True, "projection_dtype": str(values.dtype),
                                           "recurrent_output_dtype": str(output.dtype),
                                           "operation": "two-layer packed bidirectional LSTM forward/backward"}
        except (RuntimeError, ValueError, NotImplementedError) as error:
            report["probes"][precision] = {"ok": False, "error": str(error)}
    return report
