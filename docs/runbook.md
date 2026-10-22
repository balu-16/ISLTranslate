# Preparation, training and evaluation runbook

All commands here are for the future execution phase and have not been run. They assume the repository root as the working directory. Use uv for setup and execution. This guide follows the retained configurations; the complete common settings now live in `configs/baseline.yaml`.

## 1. Supply inputs and configure paths

Read the [data contracts and local storage layout](data_contracts.md#local-storage-layout). Supply annotations, matching pose files, verified parent-source mappings, the interpreter table and exact dataset revisions. Verify actual CSV headers instead of assuming the defaults match an export.

Set paths, revisions, column mappings, coordinate conventions and any metadata in the baseline before auditing. Distinct experiments inherit these settings. Local machine overrides can use an ignored `configs/*.local.yaml` file with `extends: baseline.yaml`; preserve the resolved config for every actual run.

When context or RGB will be evaluated, include its required timing/geometry metadata before freezing manifests. A changed immutable input needs new manifest/split paths and a recorded revision.

## 2. Prepare an environment

Install uv separately, then resolve dependencies and select exactly one backend. The offline CPU acceptance setup is:

```bash
uv lock
uv sync --locked --extra cpu --extra dev
uv run --no-sync pytest
# Create .env from .env.example only if it does not already exist.
```

Populate local credentials/resource paths as needed. No lockfile was fabricated during implementation. Subsequent commands use `--no-sync` so they do not implicitly install or change dependencies. One optional runner notebook and an offline CPU pytest suite are available. Tests have not been run and their duration remains unmeasured.

### AMD and backend selection

`pyproject.toml` declares mutually exclusive `cpu`, `rocm` and `cuda` extras, using explicit torch/torchvision indexes with suffixes `cpu`, `rocm7.2` and `cu130`. These are proposed choices with compatibility still unverified. Before resolution, check the intended host's GPU architecture, OS, driver/ROCm version and supported wheel pair; adjust the dependency declaration deliberately if needed and retain the real lockfile.

After `uv lock`, choose the AMD environment with:

```bash
uv sync --locked --extra rocm
```

Use `--extra cpu` or `--extra cuda` for the other declared backends. Do not combine backend extras or use `--all-extras`. Include all needed optional extras in the same sync: `rgb` for video access, `lexical` for spaCy, `semantic` for BERTScore, `notebooks` for JupyterLab/ipykernel, and `dev` for pytest. The baseline CLI and regex lexical path need none of those optional resources.

On ROCm, PyTorch uses the `torch.cuda` API; `runtime.device: cuda` requires the GPU, while `auto` can fall back to CPU. Reports expose `torch.version.hip`. The baseline stays FP32 with `recurrent_fp32: true`; packed recurrence stays outside autocast even in mixed-precision experiments. The explicit probe in step 3 reports FP32/BF16/FP16 forward/backward results and backend information. Successful probing precedes any precision change and does not establish full model correctness. BF16 support is checked; FP16 is unsupported on CPU and uses scaling on GPU.

`runtime.determinism_mode: strict` remains default. `warn` is an explicit debugging mode. NVIDIA workspace setup occurs before CUDA use; HIP workspace/recurrent behavior needs its own host checks. Keep measured backend/package inventory, seed, resolved configuration and lockfile together. Precision, batch/frame limits or resampling changes require a named experiment and fresh run directory; do not change a resumed trajectory to bypass compatibility checks. A wall-time budget saves at accumulation boundaries and does not allocate hardware.

References: [uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/) and [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html).

### Credentials and local assets

Create private `.env` from `.env.example` only if it does not already exist. Set `HF_TOKEN` there or in the process environment; the template also declares `HF_HOME`, `NLTK_DATA` and tokenizer parallelism. Credentials stay out of YAML, notebook outputs, checkpoints and logs.

Add `--env-file .env` to commands needing these settings. Ordinary model loading uses local assets. Only explicit `prepare-model --allow-network` obtains project model weights; dataset acquisition and optional language resources have separate preparation commands below. See [local storage layout](data_contracts.md#local-storage-layout) for placement.

## Notebook workflow

The [single pipeline notebook](../notebooks/run_pipeline.ipynb) runs the same commands described below. It contains no separate model or training implementation. Before launching it, set up the backend with the `notebooks` extra and prepare the local `.env` file:

```bash
uv lock
uv sync --locked --extra rocm --extra notebooks
uv run --no-sync --env-file .env jupyter lab notebooks/run_pipeline.ipynb
```

Replace `rocm` with `cpu` or the configured `cuda` backend as appropriate. Include all required optional experiment extras in the same sync command.

Select the project's `.venv` interpreter. In VS Code, use **Select Kernel → Python Environments → .venv/bin/python**. If JupyterLab does not offer the project interpreter, register it explicitly later, then select **ISL SLT (uv)**:

```bash
uv run --no-sync python -m ipykernel install --user --name isl-slt --display-name "ISL SLT (uv)"
```

The notebook resolves the repository from the root or `notebooks/` working directory. The default uv mode requires the project kernel; `RUNNER="managed"` validates the host Python/packages/backend instead. See [managed setup](#managed-colabkaggle-setup). Change `CONFIG_PATH` in its configuration cell; derived run/checkpoint/output paths follow the selected config. An optional seed override mirrors the CLI's `seed-<seed>` directory convention.

Every CLI cell starts with `execute=False` and only prints its argument list as a readable command. Set it to `True` for the stage you intend to execute. Model downloads also require `allow_network=True`; notebook cells never install dependencies. The uv runner loads `.env` without displaying it; managed mode inherits the kernel environment. Subprocess output is streamed into the cell, nonzero exits raise errors, and kernel interruption stops the running subprocess group on POSIX.

Audit, inspect the report, freeze splits and check leakage before preparing features and training. Execute either fresh training or resume as appropriate. For a resumable time limit, configure `runtime.wall_time_seconds`; an interrupted cell may leave only a previously saved checkpoint.

Notebook evaluation uses `runs/<experiment>/evaluation/<checkpoint-stem>/<partition>/`, with the seed subdirectory when selected, so differently named checkpoints have separate outputs. Set `DECODING_FILE` when a checkpoint needs an explicit saved validation selection, including averaged weights. Changing config/seed/partition requires rerunning the configuration/helper cells. Missing saved reports are displayed as absent rather than replaced with example values.

The notebook is stored with empty outputs and null execution counts. Future verification should cover opening it from both working directories, baseline and local-hand path selection, preview behavior, missing inputs/model assets, a failing subprocess, interruption, matching resume and reference-isolated evaluation. None of these notebook scenarios has been executed.

This kernel setup follows [uv's Jupyter guidance](https://docs.astral.sh/uv/guides/integration/jupyter/).

### Managed Colab/Kaggle setup

Keep the managed host's torch build. During future authorized setup, install uv separately, clone the repository and install the package into the active host interpreter:

```bash
uv pip install --system -e ".[notebooks]"
```

Choose no backend extra for this managed installation. The notebook checks Python, required package versions and the host torch build; the explicit `env-check` stage probes recurrent behavior. Resolve incompatibilities deliberately without silently replacing torch. Open the notebook with the managed kernel and set `RUNNER="managed"`; subprocesses use its `sys.executable`. Supply `HF_TOKEN` through the platform environment/secret facility; managed mode does not read local `.env`.

For Colab, mount Drive with platform controls or a future mount cell and keep the checkout plus `data/`, `splits/`, `artifacts/` and `runs/` on persistent storage. For Kaggle, use writable persistent/exported storage and reference read-only input mounts through absolute paths in an ignored local YAML override. Set data/cache/split/run paths before auditing, then open from the checkout or `notebooks/`.

No installer, clone or mount command executes automatically from the runner. Generate a real lock in the later setup phase and retain the actual managed package inventory. Do not fabricate a managed requirements export from an unresolved environment. Managed GPU/package compatibility remains unverified.

## 3. Probe and inspect the resolved plan

```bash
uv run --no-sync python -m isl_slt env-check --config configs/baseline.yaml
uv run --no-sync python -m isl_slt plan --config configs/baseline.yaml
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml --dry-run
```

`env-check` executes synthetic recurrent forward/backward probes. `plan` and `--dry-run` print configuration/planning information without loading datasets or models. A printed plan does not validate files, cache contents, losses or convergence.

## 4. Audit and freeze membership

First generate candidate source mappings, then review their real naming/cross-dataset evidence and supply a separate authoritative map:

```bash
uv run --no-sync python -m isl_slt make-source-map --config configs/baseline.yaml
uv run --no-sync python -m isl_slt make-source-map --config configs/transfer.yaml --dataset pretrain
```

Candidates remain unverified. Unresolved UIDs fail after report export. The command never overwrites an authoritative map. Exact duplicate rows are collapsed with kept/dropped row numbers and hashes; conflicting same-UID rows fail and raw annotations remain unchanged.

After pinning the dataset revision and reviewing the [publisher's UID convention](https://huggingface.co/datasets/Exploration-Lab/iSign/blob/main/README.md), promote with recorded evidence. DEF identities require the separate approval shown here; omit it for sentence-only input. No existing authoritative mapping is overwritten:

```bash
uv run --no-sync python -m isl_slt make-source-map --config configs/baseline.yaml --promote --evidence "REPLACE with reviewed revision-specific naming and DEF evidence" --approve-def
```

```bash
uv run --no-sync python -m isl_slt audit --config configs/baseline.yaml
uv run --no-sync python -m isl_slt split --config configs/baseline.yaml
uv run --no-sync python -m isl_slt check-leakage --config configs/baseline.yaml
```

Inspect `data/manifests/target.audit.json` before proceeding, including missing/invalid poses, labels, source verification, interpreter disagreements and any requested context/RGB errors. Check actual source/group counts and split memberships in `splits/source-v1/`. A report containing issues is not a successful scientific validation.

Membership is immutable and source-disjoint; different training seeds reuse it. Do not delete or overwrite a frozen split to make leakage checks pass. Revisions require distinct artifact paths.

## 5. Prepare features and generic model assets

```bash
uv run --no-sync python -m isl_slt prepare-features --config configs/baseline.yaml
uv run --no-sync python -m isl_slt verify-cache --config configs/baseline.yaml
uv run --no-sync --env-file .env python -m isl_slt prepare-model --config configs/baseline.yaml --allow-network
```

The feature command writes fingerprinted arrays and metadata. Model preparation is an explicit future download of the configured generic initializer. It resolves the requested T5 revision to a commit and records checksums in `artifacts/models/flan-t5-small/asset.json`. Ordinary training/evaluation loads local assets.

Review feature dimensions, lengths, masks, preparation settings and the initializer before training. Ordinary pose/cached-RGB training validates arrays and manifest dependency signatures without reopening raw assets. Arrays are hashed once at construction and file identities are checked during item reads. `verify-cache` defaults to full raw verification; `--no-verify-assets` checks caches alone. Train/evaluate/diagnostics accept `--verify-assets` explicitly.

Pose cache v2 adds continuous validity; old caches need regeneration. For optional standardization, prepare features with its config first:

```bash
uv run --no-sync python -m isl_slt prepare-features --config configs/standardized.yaml
uv run --no-sync python -m isl_slt prepare-stats --config configs/standardized.yaml
uv run --no-sync python -m isl_slt train --config configs/standardized.yaml
```

Statistics fit immutable target-training data only, preserve masks/invalid zeros, and are embedded in checkpoints. Image-mode RGB and raw prediction still require real assets.

## 6. Inspect a small training subset

Choose 32–64 unique manually inspected training UIDs and write one per line to `data/overfit_uids.txt`. The dedicated preset disables dropout, weight decay, noise and lexical loss, with T5 unfrozen:

```bash
uv run --no-sync python -m isl_slt plan --config configs/overfit.yaml
uv run --no-sync python -m isl_slt overfit --config configs/overfit.yaml --uids data/overfit_uids.txt
```

The protocol fits and selects on those training examples, records a diagnostic partition and excludes extra pretraining. Inspect loss/gradients/generated text; there is no BLEU threshold or held-out quality claim.

The preset is `training.debug_only: true`: ordinary training without an explicit UID file fails before split loading. In the notebook select `configs/overfit.yaml`, set `OVERFIT_UIDS_PATH` and use the dedicated overfit cell. Debug resume uses the same `overfit --uids` invocation plus `--resume`.

## 7. Train and resume the baseline

```bash
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml
```

Default training freezes T5 for one adapter epoch, then trains jointly for up to 19 more epochs. Per-epoch validation uses generation.train_beams (default greedy); the final validation sweep uses generation.beams (default 1 and 4). BLEU-4 controls early stopping in the final stage. Monitor `events.jsonl` and saved validation predictions rather than teacher-forced losses alone.

Continue a matching saved trajectory with:

```bash
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml --resume runs/baseline/checkpoints/last.pt
```

Resume checks data/cache/tokenizer/experiment/numerical fingerprints, backend, source/Git identity, actual packages and lock/pyproject hashes. --allow-code-change records only a code transition; it cannot bypass environment or data mismatches. A new run cannot reuse an existing run directory without an explicit resume. If a wall-time limit returns `wall_time_limit`, retain `last.pt` and resume with the same compatible setup.

## 8. Prepare scoring resources and evaluate

The baseline requests METEOR, which needs local NLTK WordNet. Prepare it explicitly later:

```bash
uv run --no-sync --env-file .env python -m nltk.downloader -d artifacts/nltk_data wordnet
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --partition validation --output runs/baseline/evaluation/validation
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --decoding runs/baseline/evaluation/validation/decoding.json --partition test --output runs/baseline/evaluation/test
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --decoding runs/baseline/evaluation/validation/decoding.json --partition gold --output runs/baseline/evaluation/gold
uv run --no-sync python -m isl_slt diagnostics --config configs/diagnostics.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --decoding runs/baseline/evaluation/validation/decoding.json --partition test --output runs/baseline/diagnostics/test
```

The training checkpoint retains per-epoch selection; run the final validation beam sweep and pass decoding.json to every final prediction entry point. Test references are used only for scoring. Metrics and predictions are saved under the output path. Missing requested metric resources are recorded in the report; `evaluate` then exits with an error rather than reporting a substituted zero.

Gold reports retain all-parent-holdout transcript scores plus transcript/interpreter scores on exactly the same annotated UID subset, with shared counts/hashes. Diagnostics save real, blank, incorrect-input and temporal-shuffle predictions; they test input dependence rather than proving semantic faithfulness. For a nonbaseline experiment, use a diagnostic override inheriting that experiment's config so its model/features/RGB settings match the checkpoint.

## 9. Run seeds and summarize

Use seeds 13, 42 and 2026 with the same frozen memberships:

```bash
for training_seed in 13 42 2026; do
  uv run --no-sync python -m isl_slt train --config configs/baseline.yaml --seed "$training_seed"
  uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint "runs/baseline/seed-$training_seed/checkpoints/best-bleu.pt" --partition validation --output "runs/baseline/seed-$training_seed/evaluation/validation"
  uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint "runs/baseline/seed-$training_seed/checkpoints/best-bleu.pt" --partition test --decoding "runs/baseline/seed-$training_seed/evaluation/validation/decoding.json" --output "runs/baseline/seed-$training_seed/evaluation/test"
done
uv run --no-sync python -m isl_slt summarize --reports runs/baseline/seed-13/evaluation/test/metrics.json runs/baseline/seed-42/evaluation/test/metrics.json runs/baseline/seed-2026/evaluation/test/metrics.json --output runs/baseline/seed-summary.json
```

The seed override places outputs under `seed-<seed>`. Summaries require distinct seeds and identical effective decoding, chosen beams, numerical policy and metric signatures. If validation selects different beams, use a predeclared fixed-beam evaluation protocol for a strict comparable summary. Report individual runs and variation; do not concatenate seed predictions into a single score.

## 10. Average and reselect decoding

Only average at least two retained compatible checkpoints from the same trajectory. The glob below uses actual retained epoch files, not assumed epoch numbers:

```bash
uv run --no-sync python -m isl_slt average --checkpoints runs/baseline/checkpoints/epoch-*.pt --output runs/baseline/checkpoints/averaged.pt
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/averaged.pt --partition validation --output runs/baseline/averaged/validation
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/averaged.pt --partition test --decoding runs/baseline/averaged/validation/decoding.json --output runs/baseline/averaged/test
```

Averaged weights are inference-only and cannot resume optimizer training. Decoding selection is bound to the actual checkpoint checksum, generation configuration and validation membership.

For two compatible saved prediction files, source-group paired bootstrap is available:

```bash
uv run --no-sync python -m isl_slt bootstrap --first runs/baseline/evaluation/test/predictions.jsonl --second runs/local-hand/evaluation/test/predictions.jsonl --samples 1000 --seed 2026 --output runs/comparisons/local-hand-bootstrap.json
```

Run this only after both corresponding evaluations exist with aligned UIDs, references and source IDs.

## Prediction, measurement and recovery

`predict` and `benchmark` require reference-free visual JSONL records conforming to `VisualInput`. Required identities/paths and optional timing/geometry metadata must describe actual assets; text/reference fields are rejected.

```bash
uv run --no-sync python -m isl_slt predict --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --decoding runs/baseline/evaluation/validation/decoding.json --inputs data/prediction_inputs.jsonl --output runs/baseline/predictions.jsonl
uv run --no-sync python -m isl_slt benchmark --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --decoding runs/baseline/evaluation/validation/decoding.json --inputs data/prediction_inputs.jsonl --output runs/baseline/benchmark.json
```

Benchmark reports distinguish model-only from end-to-end timing and record warmup/iteration settings. The BiLSTM is noncausal. The `estimate` command accepts actual measured throughput, validation time, example/epoch counts and an optional hourly price; no time, credit or quality guarantee is implied.

For missing assets, fix the configured paths or prepare the declared resources. For cache or membership mismatches, identify the changed inputs and use properly revised artifacts. For memory failures, make an explicit new batch/frame/resampling experiment. For resume incompatibility, restore the original environment or start a documented new trajectory. No recovery step silently drops examples or bypasses integrity checks.

See [experiments](experiments.md) for optional run sequences and [implementation status](implementation_status.md) for outstanding verification.

## Asset acquisition and layout

These commands are for later explicit network authorization. Listing does not download the pose parts; CSV and full-part flags are separate:

```bash
uv run --no-sync --env-file .env python scripts/fetch_isign.py --list --allow-network
uv run --no-sync --env-file .env python scripts/fetch_isign.py --csv --allow-network
```

The four pose files are ordered byte parts of one ZIP. Only a deliberately requested `--parts --allow-network` downloads them. Local extraction validates paths/UID uniqueness, builds the member index and decodes one selected sample before bulk extraction. For initial extraction, use the real annotation CSV as a UID request, avoiding immutable manifests with missing/fake asset checksums:

```bash
uv run --no-sync python scripts/fetch_isign.py --archives data/raw/isign-acquisition/iSign-poses_v1.1_part_aa data/raw/isign-acquisition/iSign-poses_v1.1_part_ab data/raw/isign-acquisition/iSign-poses_v1.1_part_ac data/raw/isign-acquisition/iSign-poses_v1.1_part_ad --uids-csv data/raw/ISLTranslate.csv --config configs/baseline.yaml --out data/raw/poses
uv run --no-sync python scripts/inspect_pose.py data/raw/poses/ACTUAL_UID.pose --config configs/baseline.yaml
```

Replace ACTUAL_UID with an extracted UID. `--manifest` can replace `--uids-csv` for a canonical selection. Extraction refuses overwrites, defaults to a 50 GiB selected-byte limit and records the selection/member index/output checksums. Change `--max-bytes` deliberately if justified. Configure `pose_pattern` and canonical-to-stored `features.component_names` from observed headers. The real archive layout and sample decoding remain unverified here.

## Measured training summaries and declared defaults

```bash
uv run --no-sync python scripts/summarize_run.py runs/baseline --output runs/baseline/training-summary.json
```

Logs contain actual applied group LRs, synchronized update seconds, examples/frames throughput, allocated GPU peaks and separate epoch train/validation time. Update throughput excludes input-window loading; epoch training time includes it. Setup/checkpoint I/O is excluded. Supply measured values to `estimate`; inference latency is not training throughput.

Only `last.pt` retains continuation state. Best/epoch/averaged checkpoints keep complete inference reconstruction metadata without optimizer/RNG state. `best-guarded.pt` additionally uses the historical chrF tolerance policy.

`evaluation.repeat_seeds` and `evaluation.bootstrap_samples` are declarative notes, not automatic loops or CLI-default overrides. Operative controls are `train --seed`, explicit repetitions and `bootstrap --samples`. Workers remain zero until profiled; persistent workers/GPU pinning are supported. Pose noise changes only position-valid normalized/standardized feature channels; it does not edit raw poses or recompute velocity/bones.
