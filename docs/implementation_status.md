# Implementation status and pending verification

The supplied [final review](ISL_Final_Review.md) reports **48/48 pytest cases passing**, real annotation/source/split checks, synthetic poses with a real FLAN-T5-small CPU chain, and bit-identical interrupted/resumed final weights at **commit `2dd30be`**. These are reviewer-reported results, not execution performed in this session and not validation of subsequent fixes.

This round remains implementation-only: no dependency resolution/installations, dataset/model downloads, project/test/notebook execution, training or paid compute. Current acceptance and numerical validation remain pending; task 10 (tested backend pins and a real `uv.lock`) is explicitly deferred.

## Implemented scope

The package covers revisioned data/provenance, immutable source splits, pose/RGB caches, the CE-only baseline, explicit experiments, training/resume, reference-free prediction and scored diagnostics. See [architecture](architecture.md) for contracts, [experiments](experiments.md) for optional paths and the [runbook](runbook.md) for execution commands.

Offline synthetic CPU tests remain under `tests/`; the single [runner notebook](../notebooks/run_pipeline.ipynb) supports uv and managed kernels with preview defaults and empty outputs/null execution counts. Source coverage is not evidence of passing tests or completed experiments.

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

Exact commands for executable preparation and checks are in the [runbook](runbook.md). The future test command is `uv run --no-sync pytest` after explicit `uv lock` and `uv sync --locked --extra cpu --extra dev`. The original suite's reported pass status applies only to `2dd30be`; new coverage and revised code remain unexecuted. Text inspection cannot substitute for numerical validation.

## Reporting boundaries

No real-pose experiment scores, AMD throughput/latency/memory measurements or compute costs have been produced in this round. The review's dataset counts and synthetic measurements are attributed to its earlier environment. No BLEU guarantee is made. A successful input-dependence diagnostic would still not establish semantic faithfulness, and full-sequence bidirectional inference is not a streaming result.

Future results should retain dataset/annotation/model revisions, exact memberships, resolved configs, dependency lock, package/backend versions, seeds, checkpoint checksums, decoding selection, metric signatures and raw predictions. Record measured failures and limitations alongside successful runs.

## Review task ledger

The following ledger records the earlier implementation against `ISL_Fix_Plan.md` at `2dd30be`; its historical 'not run' notes describe the original implementation phase, before the reviewer's reported execution. The user's existing deletion of that older plan is preserved. Current work follows [ISL_Final_Review.md](ISL_Final_Review.md).

Corrections applied to the proposed plan: candidate source mappings require evidence before verification; determinism remains strict; uv is retained in managed environments; velocity stays per-second and present in the baseline; scaling/removal are opt-in experiments; inference snapshots retain all reconstruction metadata.

| Task | Implemented decision and deferred acceptance |
| --- | --- |
| P0-1 | Full public gold headers plus short aliases; ambiguous aliases still fail. Offline three-row attachment/ambiguity tests written, not run. |
| P0-2 | Collapse only byte-equivalent parsed CSV rows with the same UID; preserve whitespace/raw CSV and distinct UIDs. Audit kept/dropped row numbers and row hashes; conflicting duplicates fail. Tests written, not run. |
| P0-3 | Ordered sentence/DEF candidates preserve internal/trailing hyphens. make-source-map exports a separate unverified CSV and rule/revision/hash report; unresolved UIDs fail after reporting. Authoritative evidence remains required, including cross-dataset double-hyphen resolution. |
| P0-4 | Strict determinism remains default; explicit warn mode is debugging only. NVIDIA workspace setup precedes CUDA use; HIP workspace/backend probe remains separate. env-check reports policy and configured workspaces. |
| P0-5 | Managed notebooks validate declared packages/Python/torch instead of requiring .venv, retain the host torch build and use uv for installation. Lock resolution and real environment checks are deferred; no fabricated lock/requirements file or automatic installer cell. |
| P0-6 | Explicit HF listing/CSV/part acquisition binds a resolved revision. Ordered ZIP byte parts use a seekable concatenation, safe UID indexing, bounded selective extraction and one decoded sample before bulk work. Header inspection and canonical-to-stored component aliases are implemented. Actual archive naming/schema remain unverified. |
| P1-1 | Full-partition transcript scores remain primary; transcript/interpreter scores on the exact annotated subset now share UID count/hash and source count. No hard-coded 397/291 assumptions. Paired synthetic scoring test written. |
| P1-2 | Reports fingerprint effective generation, selected beam, requested metrics/signatures, normalization, semantic resource identity and numerical policy. Seed summaries reject mismatches or legacy reports without this identity. |
| P1-3 | Cased BLEU remains primary; cumulative BLEU-1..4 and chrF++ additionally expose _lc scores with independent signatures. The test uses four tokens. Lowercasing does not establish paper-membership/metric equivalence. |
| P1-4 | Standardization is opt-in. Explicit continuous-channel validity fits target-training-only statistics; masks stay unchanged and invalid values stay zero. Statistics bind training membership/schema and are embedded in checkpoints for raw inference. Pose cache v2 adds continuous validity; v1 caches must be regenerated. |
| P1-5 | Existing baseline keeps per-second velocity. Baseline/no_velocity settings toggle both 96 value and 48 validity channels (384 versus 240 dimensions). No undocumented v7 baseline replacement. Actual-time invariance/dimension tests written. |
| P1-6 | Restored overfit.yaml with no dropout/weight decay/noise/lexical loss, one unfrozen stage, greedy decoding and a separate run directory. Supply 32–64 inspected training UIDs; there is no BLEU success assertion. |
| P1-7 | best-guarded requires a new historical BLEU maximum and chrF within 0.5 percentage points of the historical chrF maximum before the update. Tolerance is explicit; early stopping/primary checkpoint policies remain separate. |
| P2-1 | Ordinary pose/cached-RGB datasets validate arrays and dependency signatures against audited immutable checksums without raw files. verify-cache defaults to full raw verification; train/evaluate/diagnostics support --verify-assets. Image-mode RGB and raw prediction still need actual assets. |
| P2-2 | Array SHA checks occur at dataset construction; item reads check immutable file stat identities instead of rehashing. RGB initializer consistency is enforced. Worker/persistent/pinned-memory support is explicit; workers=0 remains the unprofiled debugging default. Two-worker/hash-call tests written. |
| P2-3 | Per-epoch validation uses generation.train_beams (default greedy); full generation.beams is reserved for explicit validation selection. Test/gold require the final selection when search policies differ. This avoids silently treating greedy development selection as the final sweep. |
| P2-4 | Source/partition timelines are indexed once for datasets/preparation; command-scoped signature/hash sessions verify each raw asset once and reject in-command mutation. A 200-clip source test counts raw hash calls. |
| P2-5 | Only last.pt retains optimizer/scheduler/scaler/RNG continuation state. Best/epoch/averaged snapshots keep all model, tokenizer, lexicon, T5 config, input dimensions, statistics, compatibility and decoding/provenance fields, and are explicitly inference-only. Reports/summaries expose checkpoint bytes. |
| P3-1 | Restored offline CPU-only pytest coverage, including T/2/T/4 padding, frozen-T5 gradients, accumulation, source leakage/splits, reference-free generation, two-window RNG/optimizer resume and paired diagnostics. Tiny T5 is constructed from a local synthetic config, never downloaded. Suite runtime/pass status remain unmeasured. |
| P3-2 | Renamed the external protocol external-membership-gold-safe, with algorithm=external-membership. Legacy config aliases warn; old split metadata must be re-exported into a new directory. Exact paper reproduction is not established by supplied counts or external membership. |
| P3-3 | Shared validation-selection identity checks now serve evaluate/predict/benchmark/diagnostics. All expose --decoding; averaged weights and changed generation policies require the matching checkpoint checksum, validation hash and effective generation hash. |
| P3-4 | Resume binds source hash, Git commit/dirty state, Python/platform, actual package inventory and lock/pyproject hashes. --allow-code-change records only code transitions and never bypasses environment/data/numerical incompatibility. Legacy unbound resumes fail explicitly. |
| P3-5 | Update logs include group LRs, measured synchronized update time, examples/frames throughput and peak allocated GPU bytes; epochs separate training/validation time. summarize_run.py reports actual measurements/checkpoint sizes, leaving unmeasured values null. |
| P3-6 | Consolidated operating guides in docs/ and updated the preview-only notebook for the corrected data/cache/environment/decoding workflow. Documented RGB slot masks, feature-only noise and declarative repeat/bootstrap fields. Integration checks bind evaluation statistics/RGB identities, memoize RGB hashing and preserve cache identity across verification modes. |

Integration checks also enforce matching scaling/RGB initializer identities across training/evaluation, keep full-verification mode from changing cache identity, and provide annotation-UID extraction requests before immutable auditing.

The earlier reviewer-reported tests/resume/split checks do not satisfy acceptance for these revised sources. No lockfile, tested backend pins or new numerical results are fabricated. The notebook itself remains unexecuted with empty outputs and null counts.

## Final review implementation round

| Task | Change | Acceptance in this round |
| --- | --- | --- |
| 1–2 | Audit validation before freezing, rejected draft envelopes; explicit evidence-bearing source promotion and DEF approval | Tests written; execution pending |
| 3–4 | Debug UID guard/dedicated notebook stage; neutral ROCm environment template | Tests written; execution pending |
| 5–6 | Exact archive sizes, bounded range/staged acquisition, strict source/text/budget selection, per-file extraction journal | Local range/interruption/corruption tests written; no acquisition performed |
| 7–8 | Zero default epoch retention, linked snapshots, optional BF16 storage, owned temp cleanup; scoped resume dependencies and logged overrides | Tests written; numerical/disk checks pending |
| 9 | Random local tiny-T5 plus recurrent precision/determinism probes | Written; no probe executed |
| 10 | Tested backend pins and genuine uv.lock | Deferred by implementation-only instruction |
| 11 | Pose-format, accumulation/stages/filter/BPE/attention, averaged reconstruction and loader-resume coverage; slow offline CLI chain | Written; no tests executed |
| 12–14 | Completed-epoch stage measurements, sentence/DEF/unknown reports and unique/per-stage parameter counts | Tests written; measured results pending |
| 15 | Consolidated docs, attributed earlier validation, removed duplicate velocity preset; one docs-only README | Static inspection only |
| 16–17 | Window-indexed context dependencies and actual BERTScore asset identities | Implementation/static review in this round; runtime validation pending |

Static acceptance checks syntax, local references, notebook structure/preview defaults and whitespace without importing the project. Future checks are in the runbook; no passing result is asserted for this revised branch.

Implementation choice: remote Hub metadata resolves the immutable archive identity, while a small HTTP reader checks exact `206`/`Content-Range` responses before consuming payloads. This prevents a filesystem adapter from silently buffering a complete part when ranges are ignored. `ArchivePart` still accepts HfFileSystem-compatible seekable openers. POSIX advisory locks protect extraction/staging/run writers. Selected raw poses are retained; all network, tests and hardware checks remain future opt-ins.
