# Continuous ISL → English

Baseline: **pose → temporal CNN → two-layer BiLSTM → generic FLAN-T5-small**, trained with translation cross-entropy only. Optional methods use explicit configurations. There is no pseudo-gloss CTC or guaranteed BLEU result.

**Implementation is unexecuted.** Source, tests and one runner notebook are written. Only syntax/structure and text inspection have been performed; no dependencies, dataset/model assets, project tests, notebook cells, training or paid compute have been run or obtained.

## Repository and guides

```text
src/isl_slt/               data, features, models, losses, training, evaluation, inference and utilities
configs/                   complete baseline plus explicit experiments
notebooks/run_pipeline.ipynb  one CLI runner, preview by default
scripts/                   explicit acquisition, pose inspection and run summaries
tests/                     offline CPU synthetic acceptance tests; not executed
docs/                      contracts, architecture, experiments, environments, operations and status
data/, splits/             documented local inputs and immutable memberships
artifacts/, runs/           documented prepared assets, caches and run outputs
```

Start with the [runbook](docs/runbook.md), [data contracts](docs/data_contracts.md), [architecture](docs/architecture.md), [experiments](docs/experiments.md), [AMD setup](docs/amd_setup.md) and [implementation status](docs/implementation_status.md). The [review ledger](docs/review_fixes.md) records every task and the corrections applied to the supplied fix plan. [Managed notebook setup](docs/colab.md) covers Colab/Kaggle.

Directory guides: [data](data/README.md), [splits](splits/README.md), [artifacts](artifacts/README.md), [runs](runs/README.md). Generated assets and credentials are ignored; directory README files are retained.

## Required inputs

Configure exact revisions, export columns and paths in `configs/baseline.yaml` before preparation. `dataset_revision: REQUIRED` deliberately fails auditing. Required external inputs are the ISLTranslate annotations, matching real poses, authoritative source identities and the interpreter-reference table. The full public gold headers and short aliases are supported; ambiguous aliases fail.

Source maps use `uid,source_id,verified,provenance`. Candidate sentence/DEF parsers preserve internal/trailing hyphens and never automatically claim verification. Cross-dataset parent identity, timing, coordinate conventions and RGB geometry require supporting evidence.

Default inputs are `data/raw/ISLTranslate.csv`, `data/raw/ISL-signer_validation.csv`, `data/raw/poses/<uid>.pose` and `data/metadata/source_mapping.csv`. Transfer adds extra-iSign annotations/poses/mappings; context needs verified physical timing; RGB needs actual synchronized videos and pose-to-video transforms. Dataset counts quoted in the supplied review have not been independently established here.

## Future setup and verification

Commands below are for later authorized execution. Install uv separately. Resolve a genuine lockfile and choose exactly one backend:

```bash
uv lock
uv sync --locked --extra cpu --extra dev
uv run --no-sync pytest
```

Use `--extra rocm` or the configured `--extra cuda` instead of `cpu` after checking the host's compatibility. Backend extras conflict; do not use `--all-extras`. No lockfile has been fabricated or resolved. Tests use small synthetic CPU fixtures without pretrained downloads; their pass status and runtime remain unmeasured.

Create a private `.env` from `.env.example` only if one does not exist. Supply HF credentials through that file or the process environment; never put them in configs, notebook outputs or checkpoints.

For the AMD notebook workflow:

```bash
uv sync --locked --extra rocm --extra notebooks
uv run --no-sync --env-file .env jupyter lab notebooks/run_pipeline.ipynb
```

Select the project's `.venv` kernel. Managed hosts select `RUNNER="managed"` and keep their supported torch build; use uv for package setup. Every CLI cell defaults to `execute=False`; model downloads additionally require `allow_network=True`. No installer runs automatically.

## Future preparation and training

Generate source candidates before auditing, then review identities and record evidence in a separate authoritative mapping:

```bash
uv run --no-sync python -m isl_slt make-source-map --config configs/baseline.yaml
```

The candidate CSV remains unverified; unresolved UIDs produce an error after a report is saved. The command never overwrites the authoritative map. For archive listing/extraction and a real sample pose check, see the [runbook](docs/runbook.md#asset-acquisition-and-layout).

After supplying real assets and verified mappings:

```bash
uv run --no-sync python -m isl_slt env-check --config configs/baseline.yaml
uv run --no-sync python -m isl_slt plan --config configs/baseline.yaml
uv run --no-sync python -m isl_slt audit --config configs/baseline.yaml
uv run --no-sync python -m isl_slt split --config configs/baseline.yaml
uv run --no-sync python -m isl_slt check-leakage --config configs/baseline.yaml
uv run --no-sync python -m isl_slt prepare-features --config configs/baseline.yaml
uv run --no-sync python -m isl_slt verify-cache --config configs/baseline.yaml
uv run --no-sync --env-file .env python -m isl_slt prepare-model --config configs/baseline.yaml --allow-network
uv run --no-sync python -m isl_slt overfit --config configs/overfit.yaml --uids data/overfit_uids.txt
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml
```

Inspect audit/leakage reports before training. The overfit list must contain 32–64 inspected immutable training UIDs; those diagnostic scores are not generalization results.

Ordinary training/evaluation validates caches against audited manifests without reopening raw poses or cached-RGB videos. Arrays are hashed at dataset construction; item reads check immutable file identities. `verify-cache` performs full raw verification by default; `--verify-assets` opts training/evaluation into it. Image-mode RGB and raw prediction still require real raw assets.

## Future evaluation

Per-epoch validation defaults to greedy decoding. Run the final validation beam sweep for each selected checkpoint before test/gold, and reuse its bound selection:

```bash
uv run --no-sync --env-file .env python -m nltk.downloader -d artifacts/nltk_data wordnet
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --partition validation --output runs/baseline/evaluation/validation
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --partition test --decoding runs/baseline/evaluation/validation/decoding.json --output runs/baseline/evaluation/test
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --partition gold --decoding runs/baseline/evaluation/validation/decoding.json --output runs/baseline/evaluation/gold
uv run --no-sync python -m isl_slt diagnostics --config configs/diagnostics.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --partition test --decoding runs/baseline/evaluation/validation/decoding.json --output runs/baseline/diagnostics/test
```

Metrics preserve cased BLEU as primary and add lowercase BLEU/chrF++ with signatures. Gold reports contain all-partition transcript scores plus paired transcript/interpreter scores on exactly the annotated UID subset. Missing resources are unavailable, never zero substitutes. Saved predictions precede scoring errors.

Use the [runbook](docs/runbook.md) for seed repeats, averaging, raw prediction, benchmarking and measured run summaries. All prediction entry points accept the same validation-bound `--decoding` contract, including averaged weights. Effective evaluation fingerprints prevent incompatible seed summaries.

## Experiments and persistence

Anatomy variants: local hand, bone and sparse face. Other explicit configs cover weak lexical BCE, filtered transfer, verified context, cached/image RGB and attention-LSTM. `standardized.yaml` fits continuous statistics on target training only after feature preparation; masks stay unchanged and invalid/padded values stay zero. Statistics are embedded in checkpoints and reused at inference.

The baseline keeps per-second velocity (384 channels). `no_velocity.yaml` removes both velocity values and their validity channels (240 channels); `velocity.yaml` names the original velocity-enabled representation. Scaling/removal do not silently replace the baseline.

`last.pt` retains full continuation state; best/epoch snapshots retain complete inference metadata without optimizer/RNG state. `best-guarded.pt` adds the documented chrF tolerance policy. Resume checks data, numerical policy, code, actual package inventory and lock identity. `--allow-code-change` records a code transition without bypassing environment/data checks.

Pose cache v2 adds explicit continuous validity; old caches need regeneration. Legacy external protocol configs migrate to `external-membership-gold-safe`, with new split artifacts required. Neither equal dataset counts nor this protocol establishes paper reproduction.
