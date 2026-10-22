# AMD and uv environment setup

This guide records the environment configuration implemented in the repository. No dependency resolution, installation, GPU probe or training has been executed. Run these commands only during the later execution phase on the intended host.

## Select one backend

`pyproject.toml` defines mutually exclusive `cpu`, `rocm` and `cuda` extras. Both torch and torchvision use explicit backend-specific indexes. The configured index suffixes are `cpu`, `rocm7.2` and `cu130`; these are proposed environment choices, not verified compatibility statements.

Before resolution, check the future host's GPU architecture, operating system, installed driver/ROCm version and supported PyTorch wheels. Confirm the configured index and torch/torchvision pair support that host. Update the dependency configuration deliberately if the machine requires a different combination, and retain the resolved lockfile.

The intended AMD setup commands, after installing uv separately, are:

```bash
uv lock
uv sync --locked --extra rocm
```

Choose `uv sync --locked --extra cpu` for CPU or `uv sync --locked --extra cuda` for the configured NVIDIA backend. Do not combine backend extras or use `--all-extras`. No `uv.lock` exists yet; generating one requires future dependency resolution.

Optional runtime extras compose with the selected backend:

```bash
uv sync --locked --extra rocm --extra rgb
uv sync --locked --extra rocm --extra lexical
uv sync --locked --extra rocm --extra semantic
uv sync --locked --extra rocm --extra notebooks
```

Include all extras needed by a given environment in the same sync command. RGB needs video access; `lexical` supports the optional spaCy path; `semantic` supplies BERTScore; `notebooks` supplies JupyterLab and ipykernel. The baseline CLI and regex lexical experiment need none of these extra resources.

After notebook setup, launch from the repository root with `uv run --no-sync --env-file .env jupyter lab notebooks/run_pipeline.ipynb` and select the project's `.venv` Python kernel. See the [notebook workflow](runbook.md#notebook-workflow) for kernel registration and preview controls. This launches the same pipeline through a notebook; backend and numerical checks remain required.

## Credentials and offline assets

Copy the checked-in template and fill it locally:

```bash
cp .env.example .env
```

Set `HF_TOKEN` through the local file or an existing approved login. `.env` is ignored. The template also sets `HF_HOME`, `NLTK_DATA` and tokenizer parallelism. Keep credentials out of YAML, documentation examples, checkpoints and logs.

Use `uv run --no-sync` after explicit setup. Add `--env-file .env` when a command needs those environment settings. Ordinary model loading uses local assets. Only the explicit `prepare-model --allow-network` command downloads project model weights; obtaining datasets and optional language resources is a separate preparation task.

See the [artifact guide](../artifacts/README.md) for paths and the [runbook](runbook.md) for model preparation.

## Device and precision policy

On a ROCm build, the package uses the PyTorch `torch.cuda` API for device selection, synchronization, RNG and memory/device reporting. Set `runtime.device: cuda` when a GPU is required; `auto` can fall back to CPU. The report exposes `torch.version.hip` so the actual backend can be recorded.

The initial baseline remains `precision: fp32`, `recurrent_fp32: true`. The latter executes packed recurrent operations outside autocast even when other layers use mixed precision. The environment probe exercises a two-layer packed bidirectional LSTM forward/backward operation with the configured recurrent policy:

```bash
uv run --no-sync python -m isl_slt env-check --config configs/baseline.yaml
```

It reports FP32/BF16/FP16 outcomes, device information and backend versions. A successful probe is a prerequisite for considering mixed precision; it does not establish complete model correctness or reproducibility. Do not claim a precision mode works until it succeeds on the actual host.

When changing precision, create an explicit experiment override with a fresh run directory. BF16 support is checked; FP16 is unsupported on CPU, and GPU FP16 uses gradient scaling. Do not edit a resumed trajectory's precision policy to bypass compatibility checks.

## Operational decisions

- Start with measured full-sequence batches and the configured frame budget. If memory is insufficient, lower the batch/frame limit or choose explicit full-duration resampling and record the changed experiment.
- Keep deterministic settings, package versions, the real lockfile and resolved run configuration together. Determinism and exact resume still require numerical checks on the same backend.
- An explicit wall-time budget stops at accumulation boundaries and saves `last.pt`; it does not reserve hardware or allocate compute.
- Treat a failed backend/kernel probe as a setup issue to resolve before training. No NVIDIA-only extension is required by the implementation.

Reference documentation: [uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/) and [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html). Compatibility of the configured environment remains pending.

## Strict reproducibility and future acceptance

`runtime.determinism_mode: strict` is the default. Explicit `warn` mode permits nondeterministic operations and is for debugging; its use is recorded. NVIDIA workspace configuration is applied before CUDA use. ROCm workspace behavior and recurrent numerical behavior need separate host probes; NVIDIA failure claims are not treated as established AMD failures.

For future CPU acceptance, resolve/sync `--extra cpu --extra dev`, then use `uv run --no-sync pytest`. Tests have not been executed and their runtime is not guaranteed. A real lock is still required; do not fabricate one during implementation. Retain the measured environment inventory used for managed hosts.
