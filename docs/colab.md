# Managed notebooks with uv

The runner supports RUNNER="managed" for Colab/Kaggle and RUNNER="uv" for the local AMD environment. This is unexecuted setup guidance, not a claim that a managed host's packages/GPU have been validated.

Keep the managed host's torch build. During future authorized setup, install uv separately, clone the repository, and use uv to install the package into the active host interpreter:

```bash
uv pip install --system -e ".[notebooks]"
```

Do not select a CPU/ROCm/CUDA extra for this managed installation. The notebook checks Python and required package versions plus the host's torch build, then the explicit env-check stage probes recurrent/backend behavior. If the versions are incompatible, change the documented environment deliberately instead of silently replacing torch.

Open notebooks/run_pipeline.ipynb, select the managed kernel and set RUNNER="managed". Its commands use that kernel's sys.executable. Supply HF_TOKEN through the platform's environment/secret facility; managed mode does not read a local .env. Keep all execution cells in preview until preparation is authorized.

For Colab, mount Drive using the platform UI or your own future mount cell, clone the repository inside the mounted project directory, and retain data/, artifacts/, splits/ and runs/ on persistent storage. On Kaggle, place the checkout and generated outputs in writable persistent/exported storage; input mounts can be referenced using explicit absolute paths in a local YAML override.

Set dataset, cache, split and run paths in that override before auditing. Notebook root resolution requires opening from the checkout or notebooks/ directory. All raw assets remain external. No installation/mount/clone command executes automatically from this notebook.

A missing uv.lock is expected during this implementation-only phase. Generate and retain a real lock during later environment setup. Do not fabricate requirements-colab.txt from an unresolved environment; retain the managed environment's actual package inventory with the run.
