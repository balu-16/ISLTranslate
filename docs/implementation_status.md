# Implementation status and pending verification

The repository contains an implementation of the training/evaluation pipeline and operational documentation. It has not been executed or numerically validated. No dependencies, datasets or model weights have been installed/downloaded, no notebooks or tests have been run, and no training or paid compute has been started.

## Implemented source

| Area | Implemented behavior |
| --- | --- |
| Configuration | Strict schemas, relative YAML inheritance, cross-feature/stage checks and explicit experiment configs |
| Data | CSV adapters, revisioned records, corrections, source provenance, interpreter references and pose-header inspection |
| Integrity | Asset checksums, immutable manifests, source-group splits, membership hashes, leakage checks and transfer filtering |
| Features | Shoulder-normalized pose/motion channels, validity masks, anatomy variants, verified context and fingerprinted caches |
| Baseline | Masked temporal CNN, packed two-layer BiLSTM, bridge and full generic FLAN-T5-small with translation CE |
| Optional methods | Weak lexical BCE, staged transfer, sparse face/bones/local hands, cached/live RGB and attention-LSTM |
| Training | Token-normalized accumulation, staged AdamW groups, precision policy, clipping, early stopping and wall-time exits |
| Persistence | Atomic checkpoints, tokenizer/RNG/optimizer/scheduler/scaler state, compatibility checks and exact-resume contracts |
| Evaluation | Free generation, validation decoding selection, corpus metrics, separate gold references and visual diagnostics |
| Analysis | Seed summaries, source-paired bootstrap, inference-only checkpoint averaging, reference-free prediction and timing |
| Environment | uv dependency/backend declarations, local credential template and recurrent environment probe |
| Notebook | One unexecuted CLI runner with per-stage previews, optional model-download opt-in, derived paths and saved-report inspection |

An implemented path is not a verified path. The tables describe source coverage, not test results or completed experiments.

## Retained documentation

The six guides cover [architecture](architecture.md), [data contracts](data_contracts.md), [experiments](experiments.md), [AMD setup](amd_setup.md), [operations](runbook.md) and this status record. Directory README files describe local [data](../data/README.md), [splits](../splits/README.md), [artifacts](../artifacts/README.md) and [run outputs](../runs/README.md).

Offline CPU tests are restored under tests/ with a pytest dev extra; none have been executed. One [pipeline runner notebook](../notebooks/run_pipeline.ipynb) supports uv and managed modes and retains empty outputs/null execution counts. Lint tooling and unrelated presets are not restored.

## External inputs still required

- Actual ISLTranslate annotations, matching pose files, exact revisions and authoritative parent-source mappings.
- The interpreter-reference table with verified column mapping.
- Prepared generic T5 assets with a resolved commit and checksums.
- For transfer: additional-iSign annotations/poses, revision and verified cross-dataset identities.
- For context: verified timing, continuity, source duration where needed and actual requested coverage.
- For RGB: videos, pose-to-video transforms, synchronization metadata and the declared ResNet18 initializer.
- For optional linguistic metrics/labels: local WordNet, a declared local spaCy pipeline or a local BERTScore model/revision as configured.
- A compatible future compute environment and a genuinely resolved `uv.lock`.

## Future verification work

| Check | Evidence needed before claiming success |
| --- | --- |
| Dependency resolution | Real lockfile and successful uv sync with exactly one supported backend |
| Backend support | Actual device/version report and recurrent forward/backward probe results |
| Dataset adaptation | Inspected export columns, dataset revisions, pose headers and audit reports |
| Leakage protection | Verified source provenance, frozen membership hashes and reviewed contamination/duplicate coverage |
| Feature correctness | Inspected coordinate units, normalization, gap/motion masks, odd-length padding and cache invalidation |
| Model gradients | Actual loss/gradient checks, including flow through frozen T5 into visual layers |
| Accumulation/precision | Numerical checks for token normalization, partial windows, nonfinite skipping and mixed precision |
| Resume | Interrupted versus uninterrupted continuation on a matching environment, with state/RNG checks |
| Small-subset fit | Inspected 32–64 training examples, loss behavior and generated outputs under the diagnostic protocol |
| Evaluation isolation | Free-generation predictions, validation-only selection and separately scored gold references |
| Optional paths | Real eligible transfer pool, timed context, actual RGB images/caches and train-only lexical/BPE fitting |
| Analysis | Compatible seed reports, valid source-paired comparisons, averaged decoding reselection and measured timing |
| Notebook | Correct uv kernel, root resolution, baseline/local-hand paths, preview defaults, subprocess failures/interruption, resume and evaluation |

Exact commands for executable preparation and checks are in the [runbook](runbook.md). The future test command is `uv run --no-sync pytest` after explicit `uv lock` and `uv sync --locked --extra cpu --extra dev`. The suite is written; pass status and runtime remain unmeasured. Text inspection cannot substitute for numerical validation.

## Reporting boundaries

No experiment scores, throughput measurements, latency values, memory benchmarks, eligible-pool counts or compute costs have been produced. No BLEU guarantee is made. A successful input-dependence diagnostic would still not establish semantic faithfulness, and full-sequence bidirectional inference is not a streaming result.

Future results should retain dataset/annotation/model revisions, exact memberships, resolved configs, dependency lock, package/backend versions, seeds, checkpoint checksums, decoding selection, metric signatures and raw predictions. Record measured failures and limitations alongside successful runs.

## Review fix status

The [task ledger](review_fixes.md) records P0–P3 decisions. Implemented changes cover CSV aliases/deduplication, evidence-bearing source candidates, explicit archive inspection/selective extraction, managed notebook profiles, strict backend policy, paired gold scoring, effective metric/decoding fingerprints, optional scaling/velocity ablations, overfit/guarded presets, validated cache-only operation, indexed signatures, lightweight snapshots, shared decoding, code/environment-bound resume and measured logging.

Integration checks also enforce matching scaling and RGB initializer identities across training/evaluation, keep full-verification mode from changing scientific cache identity, and provide annotation-UID extraction requests before immutable auditing.

Static validation parses source/test/script/notebook Python syntax, notebook JSON and pyproject TOML, and checks references/whitespace. It does not import/run the project. No test passes, real dataset counts, sample pose compatibility, convergence, resume equivalence or backend performance are claimed.

The definition of done in the supplied plan requires future executed CPU tests and real-data checks. Those remain deferred under the implementation-only instruction. No uv.lock, pinned managed export or real split-count report has been fabricated.
