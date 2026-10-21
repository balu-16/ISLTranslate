# Training and evaluation outputs

Future training/evaluation commands create experiment directories here. Only this README is intended to be tracked. No checkpoints, predictions or results have been produced.

## Typical experiment outputs

```text
runs/<experiment>/
  resolved.json                   complete resolved configuration
  compatibility.json              data/model/tokenizer/cache/numerical fingerprints
  events.jsonl                    update and validation events
  tokenizer/                      saved tokenizer files
  lexicon.json                    optional train-only lexical vocabulary
  validation/epoch-*.jsonl         free-generation validation predictions
  validation/epoch-*.json          decoding-selection candidates
  checkpoints/
    last.pt                       complete continuation state
    best-bleu.pt                  validation BLEU-4 selection
    best-chrf.pt                  validation chrF++ selection
    epoch-*.pt                    retained trajectory checkpoints
    pretrain/                     optional pretraining best checkpoints
  evaluation/<partition>/
    predictions.jsonl             generated text and scoring references
    metrics.json                  actual scores, signatures and resource status
  diagnostics/<partition>/        paired real/blank/incorrect/shuffled outputs
```

Evaluation output paths are explicitly supplied; the tree shows the documented convention. Validation evaluation can additionally write `decoding.json`. Seed overrides write under `runs/<experiment>/seed-<seed>/`.

Keep checkpoints, exact data memberships/revisions, prepared-asset identities, resolved configs, dependency/backend versions and prediction reports together. Use trusted local checkpoint files only. Resume requires the original compatible trajectory state; averaged checkpoints are inference-only and require fresh validation decoding selection.

Missing metric resources remain unavailable in reports. A directory or report file alone does not establish successful validation. Do not add placeholder scores or assumed performance values.

See [training/evaluation commands](../docs/runbook.md), [experiment protocols](../docs/experiments.md) and [implementation status](../docs/implementation_status.md).
