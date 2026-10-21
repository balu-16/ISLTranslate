# Immutable split artifacts

Future `split` commands write source-group memberships here. Only this README is intended to be tracked; no split memberships have been generated.

The default directory is `splits/source-v1/`. Transfer uses `splits/source-v1-transfer/`. Each contains:

- `train.jsonl`, `validation.jsonl`, `test.jsonl` and `gold.jsonl`: canonical records assigned to immutable partitions.
- `splits.json`: protocol, allocation settings and membership hashes.
- `leakage.json`: source, UID, known-content and source-verification checks.
- For transfer, `pretrain.jsonl` and `pretraining-filter.json`: eligible extra examples and the contamination report.

The standard protocol reserves complete parent sources associated with interpreter examples as gold, then approximates 80/10/10 among the remaining parents. Whole groups stay together. An original-paper protocol requires exact external membership; equal counts do not establish paper membership.

Train seeds reuse the same split files. Inspect source provenance and reports before training; unresolved identities or leakage cannot be solved by silently reshuffling clips.

Membership and canonical record content are hashed and checked when loading. Input/metadata corrections need new versioned manifests and split paths; do not edit frozen memberships in place.

See [data contracts](../docs/data_contracts.md) and the [runbook](../docs/runbook.md).
