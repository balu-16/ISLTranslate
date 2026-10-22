"""Non-pickle feature arrays with content/revision-aware cache validation."""

from pathlib import Path

import numpy as np

from isl_slt.config import Config
from isl_slt.data.manifests import Record
from isl_slt.features.pose import Prepared
from isl_slt.utils.io import atomic_open, file_hash, fingerprint, read_json, versions, write_json

CACHE_VERSION = "pose-feature-cache-v2"


def preprocessing_fingerprint(config: Config) -> str:
    return fingerprint({"version": CACHE_VERSION, "features": config.features.model_dump(mode="json"),
                        "context": config.context.model_dump(mode="json")})


def record_signature(record: Record, verify_assets: bool = True) -> dict:
    """Visual identity deliberately excludes all target and interpreter text."""
    if "pose" not in record.asset_hashes or (record.timestamps_path and "timestamps" not in record.asset_hashes):
        raise ValueError(f"{record.uid}: audited manifest asset checksums are required")
    pose_hash = file_hash(record.pose_path) if verify_assets else record.asset_hashes["pose"]
    timestamps_hash = (file_hash(record.timestamps_path) if verify_assets else record.asset_hashes["timestamps"]) if record.timestamps_path else None
    for name, actual in (("pose", pose_hash), ("timestamps", timestamps_hash)):
        if name in record.asset_hashes and record.asset_hashes[name] != actual:
            raise ValueError(f"{record.uid}: {name} changed since immutable manifest creation")
    return {"uid": record.uid, "dataset": record.dataset_name, "revision": record.dataset_revision,
            "annotation_revision": record.annotation_revision, "source": record.source_id,
            "source_verified": record.source_verified, "source_provenance": record.source_provenance,
            "pose_sha256": pose_hash, "fps": record.fps,
            "timestamps_sha256": timestamps_hash,
            "bounds": [record.start_seconds, record.end_seconds], "timing_verified": record.timing_verified,
            "source_duration_seconds": record.source_duration_seconds,
            "coordinate_space": record.coordinate_space, "person_index": record.person_index}


def cache_paths(config: Config, record: Record) -> tuple[Path, Path]:
    base = Path(config.data.feature_dir) / preprocessing_fingerprint(config) / record.key
    return base.with_suffix(".npz"), base.with_suffix(".json")


def write_features(config: Config, record: Record, prepared: Prepared, dependencies: list[Record]) -> dict:
    arrays_path, metadata_path = cache_paths(config, record)
    metadata = {"version": CACHE_VERSION, "preprocessing": preprocessing_fingerprint(config),
                "record_key": record.key, "uid": record.uid, "source_revision": record.dataset_revision,
                "dependencies": [record_signature(row) for row in sorted(dependencies, key=lambda r: r.key)],
                "frames": len(prepared.times), "writer_versions": versions(), **prepared.metadata}
    with atomic_open(arrays_path) as stream:
        np.savez_compressed(stream, values=prepared.values, times=prepared.times, observed=prepared.observed,
            imputed=prepared.imputed, position_valid=prepared.position_valid, velocity_valid=prepared.velocity_valid,
            regions=prepared.regions, segments=prepared.segments, continuous_valid=prepared.continuous_valid)
    metadata["arrays_sha256"] = file_hash(arrays_path)
    write_json(metadata_path, metadata)
    return metadata


def read_features(config: Config, record: Record, dependencies: list[Record] | None = None,
                  verify_assets: bool = False, verify_arrays: bool = True) -> Prepared:
    arrays_path, metadata_path = cache_paths(config, record)
    metadata = read_json(metadata_path)
    if metadata["version"] != CACHE_VERSION or metadata["preprocessing"] != preprocessing_fingerprint(config):
        raise ValueError(f"{record.uid}: stale/incompatible feature cache")
    if metadata["record_key"] != record.key or (verify_arrays and file_hash(arrays_path) != metadata["arrays_sha256"]):
        raise ValueError(f"{record.uid}: cache identity or array checksum mismatch")
    candidates = dependencies if dependencies is not None else [record]
    signatures = [record_signature(row, verify_assets) for row in sorted(candidates, key=lambda r: r.key)]
    if signatures != metadata["dependencies"]:
        raise ValueError(f"{record.uid}: manifest/source metadata changed; regenerate features")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        return Prepared(**{name: arrays[name].copy() for name in (
            "values", "times", "observed", "imputed", "position_valid", "velocity_valid", "regions", "segments", "continuous_valid")},
            metadata=metadata)
