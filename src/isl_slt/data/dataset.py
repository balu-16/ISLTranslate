"""Full-sequence datasets, reference-free visual batches and deterministic length bucketing."""

import math
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Sampler

from isl_slt.config import Config
from isl_slt.data.context import assemble_context
from isl_slt.data.lexical import BPETokenizer, Lexicon
from isl_slt.data.manifests import Record
from isl_slt.data.pose import read_pose
from isl_slt.features.cache import read_features
from isl_slt.features.pose import resample_sequence
from isl_slt.utils.io import file_hash, fingerprint, read_json


def dependencies_for(record: Record, records: list[Record], config: Config) -> list[Record]:
    if not config.context.enabled:
        return [record]
    return [r for r in records if r.source_id == record.source_id and r.split == record.split]


def raw_sequence(record: Record, records: list[Record], config: Config):
    sequence = assemble_context(record, dependencies_for(record, records, config), config.context,
                                lambda row: read_pose(row, config.features))
    return resample_sequence(sequence, config.features.resample_fps)


class PoseDataset(Dataset):
    def __init__(self, records: list[Record], config: Config, lexicon: Lexicon | None = None,
                 training: bool = False, preflight: bool = True, context_records: list[Record] | None = None):
        self.records, self.config, self.lexicon, self.training = records, config, lexicon, training
        self.epoch = 0
        self.context_records = context_records if context_records is not None else records
        self.lengths, self.input_dim, cache_identities, video_hashes = [], None, [], {}
        for record in records:
            prepared = read_features(config, record, dependencies_for(record, self.context_records, config), verify_assets=preflight)
            self.lengths.append(len(prepared.times))
            if self.input_dim is not None and self.input_dim != prepared.values.shape[1]:
                raise ValueError("Mixed feature dimensions in one dataset")
            self.input_dim = prepared.values.shape[1]
            cache_identity = {"key": record.key, "arrays": prepared.metadata["arrays_sha256"],
                              "dependencies": prepared.metadata["dependencies"]}
            if config.rgb.mode == "cached" and preflight:
                from isl_slt.features.rgb import read_rgb, rgb_paths
                read_rgb(config, record, dependencies_for(record, self.context_records, config), prepared.times)
                _, rgb_metadata = rgb_paths(config, record)
                cache_identity["rgb"] = read_json(rgb_metadata)
            if config.rgb.mode != "off" and preflight:
                for dependency in dependencies_for(record, self.context_records, config):
                    if not dependency.video_path or dependency.image_transform is None:
                        raise ValueError(f"{dependency.uid}: RGB requires video and coordinate-transform metadata")
                    if dependency.video_path not in video_hashes:
                        video_hashes[dependency.video_path] = file_hash(dependency.video_path)
                    actual = video_hashes[dependency.video_path]
                    if dependency.asset_hashes.get("video", actual) != actual:
                        raise ValueError(f"{dependency.uid}: video changed since immutable manifest creation")
                    cache_identity.setdefault("videos", {})[dependency.key] = actual
            cache_identities.append(cache_identity)
        self.cache_fingerprint = fingerprint(sorted(cache_identities, key=lambda row: row["key"]))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        prepared = read_features(self.config, record, verify_assets=False)
        features = torch.from_numpy(prepared.values)
        generator = torch.Generator().manual_seed(self.config.training.seed + self.epoch * 1000003 + index)
        if self.training and self.config.training.pose_noise_std:
            # Only observed position channels receive small noise; masks and RGB vectors are untouched.
            joints = prepared.observed.shape[1]
            mask = torch.from_numpy(prepared.position_valid).repeat_interleave(2, dim=1)
            features[:, :joints * 2] += torch.randn(features[:, :joints * 2].shape, generator=generator) * mask * self.config.training.pose_noise_std
        item = {"features": features, "record": record, "lexical": None}
        if self.lexicon is not None:
            item["lexical"] = torch.from_numpy(self.lexicon.encode(record.text, record.uid))
        if self.config.rgb.mode == "cached":
            from isl_slt.features.rgb import read_rgb
            item["rgb"], item["rgb_mask"] = read_rgb(self.config, record,
                dependencies_for(record, self.context_records, self.config), prepared.times, verify_assets=False)
        elif self.config.rgb.mode == "images":
            from isl_slt.features.rgb import load_crops
            item["images"], item["rgb_mask"] = load_crops(record, self.context_records,
                raw_sequence(record, self.context_records, self.config), self.config)
            if self.training and self.config.rgb.pixel_augmentation:
                # Shared photometric gain across the entire sequence/views; no geometric misalignment.
                gain = 0.95 + torch.rand((), generator=generator).item() * 0.1
                mean = item["images"].new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
                std = item["images"].new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
                item["images"] = ((item["images"] * std + mean).mul(gain).clamp(0, 1) - mean) / std
        return item


def pad_time(tensors: list[torch.Tensor], value=0) -> torch.Tensor:
    maximum = max(tensor.shape[0] for tensor in tensors)
    output = tensors[0].new_full((len(tensors), maximum, *tensors[0].shape[1:]), value)
    for index, tensor in enumerate(tensors):
        output[index, :len(tensor)] = tensor
    return output


@dataclass
class Batch:
    visual: dict[str, torch.Tensor]
    records: list[Record]
    labels: torch.Tensor | None = None
    lexical: torch.Tensor | None = None

    def to(self, device: torch.device):
        # Keep lengths on CPU for pack_padded_sequence, including ROCm.
        visual = {key: value if key == "lengths" else value.to(device) for key, value in self.visual.items()}
        return Batch(visual, self.records, self.labels.to(device) if self.labels is not None else None,
                     self.lexical.to(device) if self.lexical is not None else None)


class Collator:
    def __init__(self, tokenizer=None, labels: bool = True):
        self.tokenizer, self.labels = tokenizer, labels

    def __call__(self, items: list[dict]) -> Batch:
        lengths = torch.tensor([len(item["features"]) for item in items], dtype=torch.long)
        visual = {"features": pad_time([item["features"] for item in items]), "lengths": lengths}
        for key in ("rgb", "images", "rgb_mask"):
            if key in items[0]:
                visual[key] = pad_time([item[key] for item in items])
        records, labels = [item["record"] for item in items], None
        if self.labels:
            if self.tokenizer is None:
                raise ValueError("Training collation requires an explicit tokenizer")
            if isinstance(self.tokenizer, BPETokenizer):
                encoded = [torch.tensor(self.tokenizer.encode(record.text)) for record in records]
                labels = pad_time(encoded, -100)
            else:
                encoded = self.tokenizer([record.text for record in records], padding=True,
                                         truncation=False, return_tensors="pt")
                labels = encoded["input_ids"].masked_fill(~encoded["attention_mask"].bool(), -100)
        lexical = torch.stack([item["lexical"] for item in items]) if items[0].get("lexical") is not None else None
        return Batch(visual, records, labels, lexical)


class LengthBuckets(Sampler[list[int]]):
    """Frame-budget batches, no clipping/dropping. Single long examples retain their entire sequence."""
    def __init__(self, lengths: list[int], batch_size: int, frame_budget: int, seed: int, shuffle: bool):
        if any(length < 1 for length in lengths):
            raise ValueError("Cannot batch empty sequences")
        self.lengths, self.batch_size, self.frame_budget = lengths, batch_size, frame_budget
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0

    def batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        ordered = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(ordered)
        chunks = [ordered[i:i + self.batch_size * 50] for i in range(0, len(ordered), self.batch_size * 50)]
        batches = []
        for chunk in chunks:
            chunk.sort(key=lambda index: self.lengths[index])
            current, maximum = [], 0
            for index in chunk:
                proposed = max(maximum, self.lengths[index])
                if current and (len(current) >= self.batch_size or proposed * (len(current) + 1) > self.frame_budget):
                    batches.append(current)
                    current, maximum = [], 0
                current.append(index)
                maximum = max(maximum, self.lengths[index])
            if current:
                batches.append(current)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return len(self.batches())

    def updates(self, accumulation: int) -> int:
        return math.ceil(len(self) / accumulation)
