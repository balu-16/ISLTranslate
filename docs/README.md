# Continuous ISL → English

The baseline is pose → temporal CNN → two-layer BiLSTM → FLAN-T5-small, trained with translation cross-entropy. Optional experiments use explicit configurations.

**Execution remains pending.** No dependencies, datasets or model weights have been installed/downloaded; tests, notebook cells and training have not been run. No numerical results or BLEU guarantees are claimed.

| Guide | Contents |
| --- | --- |
| [Runbook](runbook.md) | Required inputs, uv/AMD/managed setup, notebook launch, preparation, training, resume and evaluation commands |
| [Data contracts](data_contracts.md) | Local storage layout, provenance, immutable manifests/splits, cache/model assets and output files |
| [Architecture](architecture.md) | Model, losses, masks, checkpoint formats and reproducibility decisions |
| [Experiments](experiments.md) | Explicit optional configurations and comparison protocols |
| [Implementation status](implementation_status.md) | External assets, review task ledger and deferred verification |

```text
configs/                     baseline and explicit experiment settings
docs/                        operating guides and the only README
notebooks/run_pipeline.ipynb  one CLI runner with preview defaults
scripts/                     acquisition, pose inspection and run summaries
src/isl_slt/                  data, features, models, training, evaluation and inference
tests/                       offline synthetic CPU checks for future execution
```

Local inputs and generated outputs use ignored `data/`, `splits/`, `artifacts/` and `runs/` paths. Their [layout](data_contracts.md#local-storage-layout) is documented centrally; preparation creates output directories when needed. No empty placeholder folders are tracked.

Start with the [runbook](runbook.md). Use the [runner notebook](../notebooks/run_pipeline.ipynb) for an interactive interface to the same CLI; all stages require explicit execution opt-in.
