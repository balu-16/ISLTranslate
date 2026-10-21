"""Reference-free raw-pose/video predictor, reconstructed entirely from a trusted checkpoint."""

from pathlib import Path
from typing import Literal

import torch
from pydantic import Field

from isl_slt.config import Config, Runtime, Strict
from isl_slt.data.dataset import Collator, raw_sequence
from isl_slt.data.manifests import Record
from isl_slt.evaluation.generation import decode_tokens
from isl_slt.features.pose import prepare_pose
from isl_slt.models.factory import build_model, load_tokenizer
from isl_slt.training.checkpoint import load_checkpoint, restore_tokenizer
from isl_slt.utils.io import file_hash
from isl_slt.utils.runtime import autocast, device_for


class VisualInput(Strict):
    uid: str
    pose_path: str
    source_id: str | None = None
    source_verified: bool = False
    source_provenance: str | None = None
    dataset_name: str = "inference"
    dataset_revision: str = "user-supplied"
    video_path: str | None = None
    coordinate_space: Literal["image_pixels", "normalized_image"] = "image_pixels"
    fps: float | None = Field(default=None, gt=0)
    timestamps_path: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    source_duration_seconds: float | None = None
    timing_verified: bool = False
    image_transform: list[list[float]] | None = None
    person_index: int | None = None

    def as_record(self) -> Record:
        return Record(**self.model_dump(), text="", split="test")


class Predictor:
    def __init__(self, model, tokenizer, config: Config, device: torch.device, beams: int | None):
        self.model, self.tokenizer, self.config, self.device, self.beams = model, tokenizer, config, device, beams
        self.model.eval()
        self.cached_rgb_encoder = None

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, runtime: Runtime | None = None):
        payload = load_checkpoint(checkpoint_path)
        config = Config.model_validate(payload["config"])
        if runtime is not None:
            config = config.model_copy(update={"runtime": runtime})
        tokenizer_dir = Path(checkpoint_path).parent / f"tokenizer-{payload['compatibility']['tokenizer'][:16]}"
        restore_tokenizer(payload["tokenizer"], tokenizer_dir)
        tokenizer = load_tokenizer(config, tokenizer_dir)
        lexical_size = len(payload["lexicon"]["words"]) if payload["lexicon"] else 0
        model = build_model(config, payload["input_dim"], tokenizer, lexical_size,
                            checkpoint_t5_config=payload["t5_config"], restoring=True)
        model.load_state_dict(payload["model"], strict=True)
        device = device_for(config.runtime)
        predictor = cls(model.to(device), tokenizer, config, device, payload["state"]["selected_beams"])
        predictor.checkpoint = payload
        return predictor

    def prepare(self, inputs: list[VisualInput]) -> list[dict]:
        records = [item.as_record() for item in inputs]
        items = []
        for record in records:
            sequence = raw_sequence(record, records, self.config)
            prepared = prepare_pose(sequence, self.config.features, self.config.context.enabled)
            if prepared.values.shape[1] != self.model.visual.cnn.convolutions[0].in_channels:
                raise ValueError("Inference features do not match checkpoint input schema")
            item = {"features": torch.from_numpy(prepared.values), "record": record}
            if self.config.rgb.mode != "off":
                from isl_slt.features.rgb import RGBEncoder, load_crops
                images, mask = load_crops(record, records, sequence, self.config)
                item["rgb_mask"] = mask
                if self.config.rgb.mode == "images":
                    item["images"] = images
                else:
                    if self.cached_rgb_encoder is None:
                        expected = self.checkpoint["rgb_initializer"]
                        if expected is None or file_hash(self.config.rgb.weights_path) != expected["sha256"]:
                            raise ValueError("Cached-RGB prediction requires the exact training CNN initializer checksum")
                        self.cached_rgb_encoder = RGBEncoder(self.config.rgb.weights_path).eval().to(self.device)
                    with torch.no_grad(), autocast(self.config.runtime, self.device):
                        item["rgb"] = self.cached_rgb_encoder(images.to(self.device)).float().cpu()
            items.append(item)
        return items

    @torch.no_grad()
    def predict_items(self, items: list[dict], beams: int | None = None) -> list[dict]:
        selected = self.beams if beams is None else beams
        if selected is None:
            raise ValueError("Select decoding on validation before predicting with averaged weights")
        output = []
        # Respect full sequence/frame budgets for raw prediction too.
        from isl_slt.data.dataset import LengthBuckets
        batches = LengthBuckets([len(item["features"]) for item in items], self.config.training.micro_batch_size,
                                self.config.training.max_frames_per_batch, self.config.training.seed, False)
        indexed = {}
        for indices in batches:
            batch = Collator(labels=False)([items[index] for index in indices]).to(self.device)
            with autocast(self.config.runtime, self.device):
                tokens = self.model.generate(**batch.visual, beams=selected,
                    max_new_tokens=self.config.generation.max_new_tokens, length_penalty=self.config.generation.length_penalty)
            rows = decode_tokens(self.tokenizer, tokens, self.config.generation.max_new_tokens)
            for index, row in zip(indices, rows):
                indexed[index] = {"uid": items[index]["record"].uid, **row}
        output.extend(indexed[index] for index in range(len(items)))
        return output

    def predict(self, inputs: list[VisualInput], beams: int | None = None) -> list[dict]:
        return self.predict_items(self.prepare(inputs), beams)
