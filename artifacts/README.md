# Local model assets and feature artifacts

This directory holds prepared external assets and reproducible caches. Only this README is intended to be tracked. No weights, resources or caches have been downloaded/generated.

## Expected layout

```text
artifacts/
  models/
    flan-t5-small/                 prepared generic T5/tokenizer files and asset.json
    resnet18/
      resnet18-f37072fd.pth        optional RGB initializer
  features/<fingerprint>/         prepared pose .npz arrays, JSON metadata and index
  rgb/                            optional frozen RGB vectors and cache metadata
  huggingface/                    local HF cache when using .env.example
  nltk_data/                      explicitly prepared WordNet for METEOR
```

Paths can be overridden by configuration/environment settings. Optional spaCy/BERTScore resources must use declared local paths; no example files or synthetic weights stand in for them.

`prepare-model --allow-network` is the explicit future model-download command. T5's `asset.json` records the requested revision, resolved commit and file checksums. The RGB initializer has its own checked checksum/revision. Ordinary model loading uses local assets.

`prepare-features` produces content/revision-aware pose caches. `prepare-rgb` writes frozen vectors only for `rgb.mode: cached`; cached vectors cannot support CNN fine-tuning. Metadata ties caches to source records, timing/geometry, preprocessing settings, context dependencies and relevant initializers.

Preserve model provenance and signatures with any retained experiment. Rebuild stale caches from the declared inputs instead of changing metadata to bypass integrity checks. Training checkpoints belong under [runs](../runs/README.md).

See the [runbook](../docs/runbook.md), [data contracts](../docs/data_contracts.md) and [AMD/uv setup](../docs/amd_setup.md) for exact future preparation commands.

Pose cache v2 adds continuous validity and needs regeneration from older caches. Optional statistics live at features.statistics_path and bind the immutable training membership. Ordinary cached training/evaluation can operate after raw-asset removal; verify-cache defaults to explicit full raw verification. Image-mode RGB and raw prediction retain their real-asset requirements.
