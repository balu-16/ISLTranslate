# Review fixes and execution boundary

This ledger records implementation decisions against ISL_Fix_Plan.md. Source inspection only: no installation, dependency resolution, downloads, project/test/notebook execution, training or paid compute is performed in this phase. A real uv.lock and measured acceptance results remain future work. The supplied review document is preserved.

Corrections applied to the proposed plan: candidate source mappings require evidence before verification; determinism remains strict; uv is retained in managed environments; velocity stays per-second and present in the baseline; scaling/removal are opt-in experiments; inference snapshots retain all reconstruction metadata.

| Task | Implemented decision and deferred acceptance |
| --- | --- |
| P0-1 | Full public gold headers plus short aliases; ambiguous aliases still fail. Offline three-row attachment/ambiguity tests written, not run. |
| P0-2 | Collapse only byte-equivalent parsed CSV rows with the same UID; preserve whitespace/raw CSV and distinct UIDs. Audit kept/dropped row numbers and row hashes; conflicting duplicates fail. Tests written, not run. |
| P0-3 | Ordered sentence/DEF candidates preserve internal/trailing hyphens. make-source-map exports a separate unverified CSV and rule/revision/hash report; unresolved UIDs fail after reporting. Authoritative evidence remains required, including cross-dataset double-hyphen resolution. |
| P0-4 | Strict determinism remains default; explicit warn mode is debugging only. NVIDIA workspace setup precedes CUDA use; HIP workspace/backend probe remains separate. env-check reports policy and configured workspaces. |
| P0-5 | Managed notebooks validate declared packages/Python/torch instead of requiring .venv, retain the host torch build and use uv for installation. Lock resolution and real environment checks are deferred; no fabricated lock/requirements file or automatic installer cell. |
| P0-6 | Explicit HF listing/CSV/part acquisition binds a resolved revision. Ordered ZIP byte parts use a seekable concatenation, safe UID indexing, bounded selective extraction and one decoded sample before bulk work. Header inspection and canonical-to-stored component aliases are implemented. Actual archive naming/schema remain unverified. |
