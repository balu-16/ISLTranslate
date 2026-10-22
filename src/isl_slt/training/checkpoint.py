"""Atomic, self-describing local checkpoints. Load only checkpoints from trusted local runs."""

from pathlib import Path
import math

import torch

from isl_slt.config import Config
from isl_slt.features.cache import preprocessing_fingerprint
from isl_slt.utils.io import atomic_open, file_hash, fingerprint
from isl_slt.features.statistics import load_statistics


def guarded_improvement(bleu, chrf, best_bleu, best_chrf, tolerance=0.5) -> bool:
    """New historical BLEU maximum with chrF within tolerance of historical chrF maximum."""
    return all(math.isfinite(value) for value in (bleu, chrf, best_bleu, best_chrf, tolerance)) and (
        tolerance >= 0 and bleu > best_bleu and chrf >= best_chrf - tolerance)


def bundle_tokenizer(directory: str | Path) -> dict[str, bytes]:
    directory = Path(directory)
    return {str(path.relative_to(directory)): path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()}


def tokenizer_hash(bundle: dict[str, bytes]) -> str:
    import hashlib
    return fingerprint({name: hashlib.sha256(value).hexdigest() for name, value in bundle.items()})


def restore_tokenizer(bundle: dict[str, bytes], directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in bundle.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe tokenizer filename in checkpoint")
        with atomic_open(directory / path) as stream:
            stream.write(value)
    return directory


def compatibility(config: Config, input_dim: int, split_metadata: dict, tokenizer: dict,
                  lexicon: dict | None, rgb_initializer: dict | None = None,
                  feature_artifacts: dict | None = None) -> dict:
    statistics = load_statistics(config, split_metadata["hashes"]["train"])
    return {"statistics": statistics["signature"] if statistics else None,
            "architecture": fingerprint({"model": config.model.model_dump(mode="json"), "input_dim": input_dim,
                                         "rgb": config.rgb.model_dump(mode="json")}),
            "feature_schema": preprocessing_fingerprint(config), "split_hashes": split_metadata["hashes"],
            "protocol": split_metadata["protocol"], "tokenizer": tokenizer_hash(tokenizer),
            "rgb_initializer": rgb_initializer,
            "feature_artifacts": feature_artifacts,
            "numerical_policy": {"precision": config.runtime.precision,
                                  "recurrent_fp32": config.runtime.recurrent_fp32,
                                  "deterministic": config.runtime.deterministic},
            "lexicon": fingerprint(lexicon) if lexicon is not None else None,
            "experiment": fingerprint({"training": config.training.model_dump(mode="json"),
                                        "lexical": config.lexical.model_dump(mode="json"),
                                        "generation": config.generation.model_dump(mode="json")})}


def save_checkpoint(path: str | Path, payload: dict) -> None:
    with atomic_open(path) as stream:
        torch.save(payload, stream)


def inference_payload(payload: dict) -> dict:
    """Keep every reconstruction/provenance field; remove only continuation state."""
    return {key: value for key, value in payload.items()
            if key not in {"optimizer", "scheduler", "scaler", "rng"}} | {
                "inference_only": True, "checkpoint_kind": "inference"}


def load_checkpoint(path: str | Path, expected: dict | None = None) -> dict:
    # RNG/optimizer state contains Python/NumPy structures: explicitly trusted local pickle is necessary.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "isl-slt-checkpoint-v1":
        raise ValueError("Unsupported checkpoint format")
    if tokenizer_hash(checkpoint["tokenizer"]) != checkpoint["compatibility"]["tokenizer"]:
        raise ValueError("Checkpoint tokenizer fingerprint mismatch")
    if expected is not None and checkpoint["compatibility"] != expected:
        raise ValueError("Incompatible architecture, tokenizer, features, splits or experiment on resume")
    return checkpoint


def average_checkpoints(paths: list[str], destination: str) -> dict:
    if len(paths) < 2:
        raise ValueError("Averaging needs at least two compatible checkpoints")
    first = load_checkpoint(paths[0])
    result = {}
    nonfloating = {}
    for name, tensor in first["model"].items():
        if tensor.is_floating_point():
            result[name] = tensor.double() / len(paths)
        else:
            nonfloating[name] = tensor.clone()
    for path in paths[1:]:
        other = load_checkpoint(path)
        if other["compatibility"] != first["compatibility"] or other["run_id"] != first["run_id"]:
            raise ValueError("Averaging requires the same compatible training trajectory")
        if other["model"].keys() != first["model"].keys():
            raise ValueError("Checkpoint tensor names differ")
        for name, tensor in other["model"].items():
            if tensor.shape != first["model"][name].shape or tensor.dtype != first["model"][name].dtype:
                raise ValueError(f"Checkpoint tensor schema differs: {name}")
            if tensor.is_floating_point():
                result[name] += tensor.double() / len(paths)
            elif not torch.equal(tensor, nonfloating[name]):
                raise ValueError(f"Nonfloating buffer differs and cannot be averaged: {name}")
    first["model"] = {name: value.to(first["model"][name].dtype) for name, value in result.items()} | nonfloating
    first = inference_payload(first)
    first.update(checkpoint_kind="averaged",
                 averaged_from=[{"path": path, "sha256": file_hash(path)} for path in paths])
    # Parent-specific decoding is invalid for averaged weights; evaluation must first select on validation.
    first["state"]["selected_beams"] = None
    save_checkpoint(destination, first)
    return {"output": destination, "parents": first["averaged_from"], "requires_validation_decoding_selection": True}
