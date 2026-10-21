# Preparation, training and evaluation runbook

All commands here are for the future execution phase and have not been run. They assume the repository root as the working directory. Use uv for setup and execution. This guide follows the retained configurations; the complete common settings now live in `configs/baseline.yaml`.

## 1. Supply inputs and configure paths

Read the [data contracts](data_contracts.md) and [data directory guide](../data/README.md). Supply annotations, matching pose files, verified parent-source mappings, the interpreter table and exact dataset revisions. Verify actual CSV headers instead of assuming the defaults match an export.

Set paths, revisions, column mappings, coordinate conventions and any metadata in the baseline before auditing. Distinct experiments inherit these settings. Local machine overrides can use an ignored `configs/*.local.yaml` file with `extends: baseline.yaml`; preserve the resolved config for every actual run.

When context or RGB will be evaluated, include its required timing/geometry metadata before freezing manifests. A changed immutable input needs new manifest/split paths and a recorded revision.

## 2. Prepare an environment

Install uv separately, then resolve dependencies and select exactly one backend. This CPU example is suitable for an initial environment; use the [AMD guide](amd_setup.md) for the intended ROCm host.

```bash
uv lock
uv sync --locked --extra cpu
cp .env.example .env
```

Populate local credentials/resource paths as needed. No lockfile was fabricated during implementation. Subsequent commands use `--no-sync` so they do not implicitly install or change dependencies. One optional runner notebook is available; local tests remain removed and there is currently no test-suite command.

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

The notebook checks its kernel environment and resolves the repository from either the root or `notebooks/` working directory. Change `CONFIG_PATH` in its configuration cell; derived run/checkpoint/output paths follow the selected config. An optional seed override mirrors the CLI's `seed-<seed>` directory convention.

Every CLI cell starts with `execute=False` and only prints its argument list as a readable command. Set it to `True` for the stage you intend to execute. Model downloads also require `allow_network=True`; notebook cells never install dependencies. The runner loads `.env` through uv when the file exists and does not display its contents. Subprocess output is streamed into the cell, nonzero exits raise errors, and kernel interruption stops the running subprocess group on POSIX.

Audit, inspect the report, freeze splits and check leakage before preparing features and training. Execute either fresh training or resume as appropriate. For a resumable time limit, configure `runtime.wall_time_seconds`; an interrupted cell may leave only a previously saved checkpoint.

Notebook evaluation uses `runs/<experiment>/evaluation/<checkpoint-stem>/<partition>/`, with the seed subdirectory when selected, so differently named checkpoints have separate outputs. Set `DECODING_FILE` when a checkpoint needs an explicit saved validation selection, including averaged weights. Changing config/seed/partition requires rerunning the configuration/helper cells. Missing saved reports are displayed as absent rather than replaced with example values.

The notebook is stored with empty outputs and null execution counts. Future verification should cover opening it from both working directories, baseline and local-hand path selection, preview behavior, missing inputs/model assets, a failing subprocess, interruption, matching resume and reference-isolated evaluation. None of these notebook scenarios has been executed.

## 3. Probe and inspect the resolved plan

```bash
uv run --no-sync python -m isl_slt env-check --config configs/baseline.yaml
uv run --no-sync python -m isl_slt plan --config configs/baseline.yaml
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml --dry-run
```

`env-check` executes synthetic recurrent forward/backward probes. `plan` and `--dry-run` print configuration/planning information without loading datasets or models. A printed plan does not validate files, cache contents, losses or convergence.

## 4. Audit and freeze membership

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
uv run --no-sync --env-file .env python -m isl_slt prepare-model --config configs/baseline.yaml --allow-network
```

The feature command writes fingerprinted arrays and metadata. Model preparation is an explicit future download of the configured generic initializer. It resolves the requested T5 revision to a commit and records checksums in `artifacts/models/flan-t5-small/asset.json`. Ordinary training/evaluation loads local assets.

Review feature dimensions, sequence lengths, masks, preparation settings and the resolved initializer before using them in a training run.

## 6. Inspect a small training subset

Choose 32–64 unique manually inspected UIDs from the immutable training partition and write one UID per line to `data/overfit_uids.txt`. Supply a separate configuration, for example `configs/overfit.local.yaml`:

```yaml
extends: baseline.yaml
experiment: inspected-training-subset
run_dir: runs/inspected-training-subset
training:
  micro_batch_size: 4
  gradient_accumulation_steps: 1
  early_stopping_patience: 20
```

Then execute later:

```bash
uv run --no-sync python -m isl_slt overfit --config configs/overfit.local.yaml --uids data/overfit_uids.txt
```

The inspected-subset protocol uses the chosen training examples for both fitting and diagnostic selection, records the special selection partition, and excludes additional pretraining. It cannot establish held-out quality. Inspect loss, gradient behavior and generated outputs; do not report its metrics as validation/test results.

## 7. Train and resume the baseline

```bash
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml
```

Default training freezes T5 for one adapter epoch, then trains jointly for up to 19 more epochs. Validation generation chooses greedy or beam-4 decoding, and BLEU-4 controls early stopping in the final stage. Monitor `events.jsonl` and saved validation predictions rather than teacher-forced losses alone.

Continue a matching saved trajectory with:

```bash
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml --resume runs/baseline/checkpoints/last.pt
```

Resume checks data/cache/tokenizer/experiment/numerical fingerprints and the backend. A new run cannot reuse an existing run directory without an explicit resume. If a wall-time limit returns `wall_time_limit`, retain `last.pt` and resume with the same compatible setup.

## 8. Prepare scoring resources and evaluate

The baseline requests METEOR, which needs local NLTK WordNet. Prepare it explicitly later:

```bash
uv run --no-sync --env-file .env python -m nltk.downloader -d artifacts/nltk_data wordnet
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --partition test --output runs/baseline/evaluation/test
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --partition gold --output runs/baseline/evaluation/gold
uv run --no-sync python -m isl_slt diagnostics --config configs/diagnostics.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --partition test --output runs/baseline/diagnostics/test
```

The normal trajectory checkpoint retains validation-selected decoding. Test references are used only for scoring. Metrics and predictions are saved under the output path. Missing requested metric resources are recorded in the report; `evaluate` then exits with an error rather than reporting a substituted zero.

Gold transcript/interpreter scores remain separate. Diagnostics save real, blank, incorrect-input and temporal-shuffle predictions; they test input dependence rather than proving semantic faithfulness. For a nonbaseline experiment, use a diagnostic override inheriting that experiment's config so its model/features/RGB settings match the checkpoint.

## 9. Run seeds and summarize

Use seeds 13, 42 and 2026 with the same frozen memberships:

```bash
for training_seed in 13 42 2026; do
  uv run --no-sync python -m isl_slt train --config configs/baseline.yaml --seed "$training_seed"
  uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint "runs/baseline/seed-$training_seed/checkpoints/best-bleu.pt" --partition test --output "runs/baseline/seed-$training_seed/evaluation/test"
done
uv run --no-sync python -m isl_slt summarize --reports runs/baseline/seed-13/evaluation/test/metrics.json runs/baseline/seed-42/evaluation/test/metrics.json runs/baseline/seed-2026/evaluation/test/metrics.json --output runs/baseline/seed-summary.json
```

The seed override places outputs under `seed-<seed>`. Summaries require compatible reports from distinct seeds. Report individual runs and variation; do not concatenate seed predictions into a single score.

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
uv run --no-sync python -m isl_slt predict --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --inputs data/prediction_inputs.jsonl --output runs/baseline/predictions.jsonl
uv run --no-sync python -m isl_slt benchmark --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --inputs data/prediction_inputs.jsonl --output runs/baseline/benchmark.json
```

Benchmark reports distinguish model-only from end-to-end timing and record warmup/iteration settings. The BiLSTM is noncausal. The `estimate` command accepts actual measured throughput, validation time, example/epoch counts and an optional hourly price; no time, credit or quality guarantee is implied.

For missing assets, fix the configured paths or prepare the declared resources. For cache or membership mismatches, identify the changed inputs and use properly revised artifacts. For memory failures, make an explicit new batch/frame/resampling experiment. For resume incompatibility, restore the original environment or start a documented new trajectory. No recovery step silently drops examples or bypasses integrity checks.

See [experiments](experiments.md) for optional run sequences and [implementation status](implementation_status.md) for outstanding verification.
