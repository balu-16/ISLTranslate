"""Local-asset model construction and explicit checkpoint reconstruction."""

from pathlib import Path

from isl_slt.config import Config
from isl_slt.data.lexical import BPETokenizer
from isl_slt.models.encoders import VisualEncoder
from isl_slt.models.lstm import AttentionLSTM
from isl_slt.models.t5 import T5Translator
from isl_slt.utils.io import file_hash, read_json


def load_tokenizer(config: Config, directory: str | Path | None = None):
    directory = Path(directory or config.model.t5_dir)
    if config.model.decoder == "lstm":
        return BPETokenizer.from_pretrained(directory)
    from transformers import AutoTokenizer
    if not directory.is_dir():
        raise FileNotFoundError("Run the explicit prepare-model command before loading T5 assets")
    return AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)


def build_model(config: Config, input_dim: int, tokenizer, lexical_size: int = 0,
                checkpoint_t5_config: dict | None = None, restoring: bool = False):
    visual = VisualEncoder(input_dim, config, initialize_rgb=not restoring)
    if config.model.decoder == "lstm":
        model = AttentionLSTM(visual, len(tokenizer), config.model.decoder_hidden, config.model.decoder_embedding,
                             config.model.dropout, lexical_size,
                             word_dropout=config.model.word_dropout, ctc_weight=config.model.ctc_weight,
                             rank_weight=config.model.rank_weight, rank_margin=config.model.rank_margin)
        if config.model.init_checkpoint and not restoring:
            import torch
            saved = torch.load(config.model.init_checkpoint, map_location="cpu", weights_only=False)
            state = saved.get("model") or saved.get("model_state") or saved
            keep = {k: v for k, v in state.items() if k.startswith(("visual.", "lexical_head."))}
            missing = model.load_state_dict(keep, strict=False)
            print(f"init_checkpoint: loaded {len(keep)} encoder tensors", flush=True)
        return model
    from transformers import T5Config, T5ForConditionalGeneration
    if restoring:
        if checkpoint_t5_config is None:
            raise ValueError("Checkpoint reconstruction requires its saved T5 configuration")
        # Transient initialization is immediately replaced by strict checkpoint loading; never a fallback.
        t5 = T5ForConditionalGeneration(T5Config.from_dict(checkpoint_t5_config))
    else:
        directory = Path(config.model.t5_dir)
        metadata = read_json(directory / "asset.json")
        if metadata["repository"] != config.model.t5_repo or metadata["requested_revision"] != config.model.t5_revision:
            raise ValueError("Prepared T5 repository/revision differs from configuration")
        for name, expected in metadata["files"].items():
            if file_hash(directory / name) != expected:
                raise ValueError(f"Prepared T5 checksum changed: {name}")
        t5 = T5ForConditionalGeneration.from_pretrained(str(directory), local_files_only=True,
                                                       trust_remote_code=False, use_safetensors=True)
    return T5Translator(visual, t5, lexical_size)
