"""Canonical fingerprints and atomic local artifacts. No network side effects."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def atomic_open(path: str | Path, mode: str = "wb") -> Iterator:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, mode) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: str | Path, value: Any, immutable: bool = False) -> None:
    if immutable and Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"Refusing to replace immutable artifact: {path}")
        return
    with atomic_open(path) as stream:
        stream.write(canonical(value) + b"\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict], immutable: bool = False) -> None:
    if immutable and Path(path).exists():
        if read_jsonl(path) != rows:
            raise ValueError(f"Refusing to replace immutable manifest: {path}")
        return
    with atomic_open(path) as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")


@lru_cache(maxsize=1)
def versions() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("isl-slt", "torch", "torchvision", "transformers", "numpy", "pose-format",
                 "sacrebleu", "tokenizers", "pydantic"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    try:
        result["git"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        result["git"] = None
    return result


def append_event(path: str | Path, **event: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as stream:
        stream.write(canonical(event) + b"\n")
