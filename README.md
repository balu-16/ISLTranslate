# Continuous ISL → English

Training and evaluation package for continuous Indian Sign Language translation. Baseline: **pose features → temporal CNN → two-layer BiLSTM → FLAN-T5-small**, with translation cross-entropy only. Optional experiments use explicit configurations. There is no pseudo-gloss CTC or guaranteed BLEU claim.

**Execution remains pending.** No project code, training, evaluation, installations, downloads or paid compute have been run. One unexecuted runner notebook provides a step-by-step interface to the CLI; local tests remain removed. The guides in `docs/` and directory README files document setup, data contracts, experiments and operations.

## Repository

```text
pyproject.toml           uv dependencies, backend indexes and package entry point
.python-version         Python 3.11 selection
.env.example            local credential/resource-path template
configs/                baseline and distinct experiment settings
docs/                   architecture, data contracts, experiments, AMD setup, runbook and status
notebooks/run_pipeline.ipynb  one unexecuted CLI runner; every command defaults to preview
data/, splits/          input and split documentation; local assets are ignored
artifacts/, runs/       asset and run documentation; generated outputs are ignored
src/isl_slt/
  cli.py                command-line entry points
  config.py             validated configuration and YAML inheritance
  workflows.py          preparation, auditing and evaluation orchestration
  data/                 manifests, splits, pose loading, context, batches and vocabularies
  features/             normalization, motion, anatomy, caches and RGB extraction
  models/               visual encoder, T5, attention-LSTM and construction
  losses/               translation CE and optional lexical BCE
  training/             staged optimization, checkpointing and resume
  evaluation/           generation, metrics, diagnostics and run comparisons
  inference/            checkpoint restoration and prediction used by evaluation
  utils/                atomic I/O, fingerprints, logging, precision and RNG
```

Every retained source module contributes to these paths or their optional experiments. Package `__init__.py` files define the installable package structure. Runtime folders retain tracked README files; local inputs and generated outputs inside them are ignored.

## Documentation

- [Runbook](docs/runbook.md): future preparation, verification, training, evaluation and recovery commands.
- [Data contracts](docs/data_contracts.md): input schemas, source identity, timing, splits and cache integrity.
- [Architecture](docs/architecture.md): model structure, training invariants and engineering decisions.
- [Experiments](docs/experiments.md): explicit optional configurations and comparison protocols.
- [AMD setup](docs/amd_setup.md): uv backend selection, ROCm configuration and environment probes.
- [Implementation status](docs/implementation_status.md): implemented capabilities and pending execution checks.
- [Pipeline notebook](notebooks/run_pipeline.ipynb): configuration, preparation, training, resume, evaluation and saved reports.

Directory guides explain [data](data/README.md), [splits](splits/README.md), [artifacts](artifacts/README.md) and [run outputs](runs/README.md).

## Configure the data

`configs/baseline.yaml` is the single shared configuration. Set dataset revisions, paths and source mappings before preparation; `dataset_revision: REQUIRED` intentionally fails audit. Other experiments inherit it. Paths are relative to the repository working directory.

Expected inputs:

```text
data/raw/ISLTranslate.csv
data/raw/ISL-signer_validation.csv
data/raw/poses/<uid>.pose
data/metadata/source_mapping.csv
data/metadata/target.csv                  optional timing/coordinate metadata
data/raw/iSign_v1.1.csv                   optional transfer annotations
data/raw/isign-poses/<uid>.pose           optional transfer poses
data/metadata/isign_source_mapping.csv    optional transfer identity mapping
data/raw/videos/<uid>.mp4                 optional RGB
```

- Annotation defaults are `uid,text`. Configure exact aliases in `data.target.columns` and `data.pretrain.columns` for the actual exports; ambiguous aliases fail. Interpreter defaults are `uid,Transcribed Text,Gold Translation`; change them to match the supplied headers.
- Source-map CSV requires `uid,source_id,verified,provenance`. Verified rows need evidence provenance; source IDs identify the same parents across datasets. A parser can infer a final numeric sequence suffix while preserving hyphens, but inferred mappings remain unverified. Default source-held-out training requires verified identities.
- Canonical manifests record immutable dataset/annotation revisions, labels, paths, source provenance, optional interpreter references and sequence metadata. Audit records asset SHA256 checksums and available exact pose/video duplicate fingerprints. Repeated text alone is retained.
- Optional metadata CSV has unique `uid` rows and canonical field names. Useful fields: `fps`, `timestamps_path`, `start_seconds`, `end_seconds`, `source_duration_seconds`, `timing_verified`, `coordinate_space`, `person_index`, `image_transform`, `annotation_revision`, `duplicate_fingerprints`. Unknown fields fail. JSON-valued fields use CSV-escaped JSON; booleans are `true`/`false`.
- Timestamp files contain one finite, strictly increasing local frame offset in seconds per pose frame. Otherwise the reader needs actual positive extraction FPS. Verified bounds convert offsets to source time. Context requires verified identity, continuity, timing and coverage; UID adjacency is insufficient. Source duration identifies true endpoints; unknown gaps fail by default.
- Coordinates default to image-pixel XY. Use `normalized_image` only when confirmed by extraction provenance; header dimensions then convert XY to pixels. RGB requires an explicit 2×3 affine `image_transform` mapping pose-header pixels to video pixels. Identity transforms need actual geometry verification.
- Actual `pose-format` headers determine joints, dimensions and people. Masks, finite coordinates and configured component confidence thresholds are combined. Multiple people require an explicit person index. Missing video does not block pose-only operation.
- Raw annotations stay immutable. Corrections are separate JSONL entries with exactly `uid,text,revision,provenance`. Include needed timing/RGB metadata before freezing manifests. Later changes require a new audited artifact revision; training seeds reuse the same immutable memberships.

## Future environment setup

These commands are **not executed**. Install uv separately and choose one backend. The explicit GPU indexes currently propose ROCm 7.2 and CUDA 13.0; check the target machine's compatibility before setup. No dependency lock has been generated yet.

```bash
uv lock
uv sync --locked --extra cpu
```

For the compatible future AMD host, use `uv sync --locked --extra rocm` instead; `--extra cuda` selects the corresponding NVIDIA environment. Add `--extra rgb`, `--extra lexical` or `--extra semantic` when needed. Backend extras conflict intentionally; do not use `--all-extras`. Retain the real `uv.lock` after resolution.

Run commands using `uv run --no-sync` after explicit setup so execution does not automatically modify the environment. The optional `notebooks` extra adds JupyterLab and ipykernel; test and lint tooling remain removed.

Copy `.env.example` to a local `.env` and supply your existing approved credential through `HF_TOKEN` or a local Hugging Face login. Credentials do not belong in YAML, source, logs or checkpoints. `--env-file .env` loads credentials and resource paths when needed. Dataset acquisition/extraction is manual; ordinary imports and model loading never download assets.

## Run through Jupyter

After confirming AMD compatibility, supply the required data/configuration and use these future commands from the repository root:

```bash
uv lock
uv sync --locked --extra rocm --extra notebooks
uv run --no-sync --env-file .env jupyter lab notebooks/run_pipeline.ipynb
```

Select the project's `.venv` Python kernel; CPU users select `--extra cpu` instead of `--extra rocm`. Include any experiment extras in the same sync command. The [runbook](docs/runbook.md#notebook-workflow) covers kernel registration and VS Code.

In the notebook, choose the config once, run the configuration/helper cells, and inspect command previews. Change `execute=False` to `execute=True` only in the stage cell you intend to run. Model preparation additionally requires `allow_network=True`. Audit and check leakage before training. Resume is a separate cell; saved reports are displayed only when they exist.

## Prepare, train and evaluate

Run from the repository root after supplying the inputs/configuration above. Inspect audit and leakage reports before training.

```bash
uv run --no-sync python -m isl_slt env-check --config configs/baseline.yaml
uv run --no-sync python -m isl_slt plan --config configs/baseline.yaml
uv run --no-sync python -m isl_slt audit --config configs/baseline.yaml
uv run --no-sync python -m isl_slt split --config configs/baseline.yaml
uv run --no-sync python -m isl_slt check-leakage --config configs/baseline.yaml
uv run --no-sync python -m isl_slt prepare-features --config configs/baseline.yaml
uv run --no-sync --env-file .env python -m isl_slt prepare-model --config configs/baseline.yaml --allow-network
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml --dry-run
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml
```

Only `prepare-model --allow-network` downloads weights. It resolves the requested revision to a concrete commit and records checksums. Default initialization is generic `google/flan-t5-small`; arbitrary task-specific checkpoints are not assumed leakage-free. Full METEOR needs local WordNet, prepared explicitly later:

```bash
uv run --no-sync --env-file .env python -m nltk.downloader -d artifacts/nltk_data wordnet
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --partition test --output runs/baseline/evaluation/test
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --partition gold --output runs/baseline/evaluation/gold
uv run --no-sync python -m isl_slt diagnostics --config configs/diagnostics.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --partition test --output runs/baseline/diagnostics/test
```

Baseline already contains full evaluation settings, so a duplicate evaluation YAML is unnecessary. Missing metric resources are reported, never replaced with zeros. Predictions are saved before scoring. Gold transcript and interpreter scores are separate; gold is not the main generalization test.

Outputs include `data/manifests/*.audit.json`, immutable split manifests and `leakage.json`, fingerprinted feature caches, and run configurations, events, tokenizer files, validation predictions and checkpoints. `last.pt`, `best-bleu.pt` and `best-chrf.pt` have distinct roles.

## Experiments

| Configuration | Contribution |
| --- | --- |
| `baseline.yaml` | Complete shared settings; CE-only pose CNN/BiLSTM/T5 |
| `local_hand.yaml` | Wrist-local handshape, keeping body-relative wrists |
| `bone.yaml` | Anatomical bone vectors and masks |
| `face.yaml` | Sparse brow/eye/mouth landmarks |
| `lexical.yaml` | Weak English-presence BCE, weight 0.1 with a ramp |
| `transfer.yaml` | Filtered extra-iSign pretraining, then target fine-tuning |
| `context.yaml` | Verified previous/following source context |
| `rgb_cached.yaml` | Frozen, synchronized ResNet18 vectors |
| `rgb_images.yaml` | Actual image access and final-CNN-block tuning |
| `attention_lstm.yaml` | Training-only BPE and attention-LSTM decoding |
| `diagnostics.yaml` | Paired visual evaluation without optional linguistic resources |

Use the same experiment configuration for preparation, training and evaluation:

```bash
uv run --no-sync python -m isl_slt prepare-features --config configs/local_hand.yaml
uv run --no-sync python -m isl_slt train --config configs/local_hand.yaml
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/local_hand.yaml --checkpoint runs/local-hand/checkpoints/best-bleu.pt --partition test --output runs/local-hand/evaluation/test
```

Set context duration or combine anatomy flags in the retained configs instead of keeping files for every numerical setting/combination. Change experiment name and run directory for a different experiment and retain its resolved config. Precomputed lexical annotations use JSONL `uid,words`; spaCy needs an explicit local pipeline. BERTScore needs its optional dependency, local model and recorded revision. Neither resource downloads automatically.

For transfer, configure extra assets/mappings, then run `audit`, `split`, `check-leakage`, `prepare-features` and `train` with `configs/transfer.yaml`. Compare target memberships with baseline. Chronology is generic initialization → filtered extra-iSign training → target training. Its contamination report excludes validation/test/gold parents, all target UIDs, available duplicates and unresolved identities; no fixed eligible-pool size is assumed.

For RGB, supply videos/transforms in frozen manifests, then use:

```bash
uv sync --locked --extra rocm --extra rgb
uv run --no-sync --env-file .env python -m isl_slt prepare-model --config configs/rgb_cached.yaml --allow-network
uv run --no-sync python -m isl_slt prepare-rgb --config configs/rgb_cached.yaml
uv run --no-sync python -m isl_slt train --config configs/rgb_cached.yaml
uv run --no-sync python -m isl_slt train --config configs/rgb_images.yaml
```

Cached vectors cannot fine-tune the CNN. Image mode reads actual crops, keeps crop-validity masks, and uses an upper-body fallback for missing hands. Only the explicit final stage unfreezes the last residual block.

## Resume and compare

```bash
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml --resume runs/baseline/checkpoints/last.pt
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml --seed 13
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml --seed 42
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml --seed 2026
```

Seed overrides write under `runs/baseline/seed-<seed>`, reusing immutable splits. Evaluate each checkpoint separately, then use `summarize --reports ... --output ...` with real report paths. Do not pool seed predictions. `bootstrap --first ... --second ... --output ...` compares saved predictions by paired source-group resampling.

`average --checkpoints ... --output ...` accepts compatible checkpoints from one trajectory and preserves nonfloating buffers. Averaged weights cannot resume training; validate decoding before test:

```bash
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/averaged.pt --partition validation --output runs/baseline/averaged/validation
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/averaged.pt --partition test --decoding runs/baseline/averaged/validation/decoding.json --output runs/baseline/averaged/test
```

Exact resume checks numerical policy, backend, manifests, tokenizer and caches, restoring optimizer/scheduler/scaler/RNG state. Wall-time limits save at accumulation boundaries. An OOM fails visibly; change an explicit batch/frame/resampling configuration and record the new trajectory instead of dropping difficult examples.

`predict --config ... --checkpoint ... --inputs ... --output ...` accepts reference-free visual JSONL records; `benchmark` uses the same inputs and records model-only/end-to-end latency. `estimate` requires measured throughput and validation time. The optional `overfit` command can inspect 32–64 chosen training UIDs with an explicitly supplied configuration; no debug-only preset or local test files are retained.

## Scientific and implementation invariants

- The 48-joint baseline selects shoulders, elbows, wrists and both hands. Shoulder normalization, short-gap interpolation and actual-time velocity retain observation, imputation and motion masks. Padding is separate. Full sequences remain intact through length bucketing.
- CNN strides total two by default, with exact odd-length masks and packed recurrence. The bridge uses loaded T5 `d_model`. Labels retain EOS and mask padding with `-100`. Freezing T5 does not detach adapter gradients; unfreezing preserves optimizer groups.
- Losses use effective-window token/label denominators, including the final partial window. Skipped/nonfinite updates do not advance scheduling. Checkpoints/manifests use atomic writes and compatibility fingerprints.
- Splits reserve whole parents containing interpreter examples, then approximate 80/10/10 by source. Supplied exact paper splits are a separately named protocol; equal counts are not paper membership. Lexical/BPE resources fit training data only.
- Validation uses free generation for greedy/beam selection. Test references never enter prediction. BLEU-1/2/3/4 are cumulative corpus scores with 13a/signatures, alongside chrF++, TER, ROUGE, METEOR and corpus WER. Reports document scales; WER/TER can exceed 100. Truncation is recorded.
- Diagnostics compare real, blank, incorrect and shuffled input. They establish input dependence, not semantic faithfulness. The BiLSTM is noncausal; latency is not a streaming claim.
- ROCm uses `torch.cuda`. `env-check` probes packed recurrent forward/backward behavior. Keep recurrent operations in FP32 when needed. No NVIDIA-only extension is required.

Setup references: [uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/), [T5 interfaces](https://huggingface.co/docs/transformers/v4.57.1/model_doc/t5), [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html). Dependency resolution, data assumptions, numerical correctness, ROCm compatibility and translation quality remain unverified until execution.
