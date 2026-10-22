# ISLTranslate: final consolidated review (commit 2dd30be)

Branch `fix/review-2026-10` @ `2dd30bed`. `master` is still at `7b58341`.
Sources merged: (A) Hark's executed review (ISL_Review_v2.md) and (B) the static review ISL_Repository_Review_2dd30be.md. Hark re-checked every new finding in (B) against the code.

## 1. Verdict
The fix round succeeded. The pipeline imports, its tests pass, and it runs end to end on synthetic poses. Do not restart the implementation. A set of mostly small fixes stands between this branch and a first real-pose run. Most of them are about data flow and storage, not modelling.

## 2. What is verified by execution (closes the "unexecuted" caveats in B)
- pytest: **48/48 pass** (B's 40 test functions; parametrization yields 48). `compileall` OK; pyflakes reports one unused import.
- Real ISLTranslate CSV: 31,221 UIDs (28,530 sentence + 2,691 DEF), 0 unresolved, 796 sources, identical duplicate `UbhONRGm-90--61` collapsed, 291 gold rows attached, 0 transcript conflicts.
- Source-held-out split: train 24,657 (638 sources) / val 3,084 (77) / test 3,083 (77) / gold 397 (4 sources). `check-leakage` passes.
- Synthetic `.pose` files + real Flan-T5-small on CPU ran this chain: audit → split → check-leakage → prepare-features → verify-cache → train (adapter + joint) → evaluate (val/test/gold) → diagnostics → predict → benchmark → average + validation reselect → overfit → summarize. This exercises in practice several paths B lists as untested in N4 (pose-format read/decode, averaging → Predictor, full `run_training` resume).
- Exact resume after a wall-time interrupt: final weights **bit-identical** (max diff 0.0).
- Test/gold evaluation without `--decoding` correctly refuses.
- Tokenizer: 7/31,222 texts contain `<unk>`; max 122 tokens (< 128 limit).

Executed coverage is not committed coverage, so N4 below still asks for real tests.

## 3. Findings check (B's new items, confirmed by Hark in code)
| ID | Confirmed? | Evidence |
|---|---|---|
| N1 context audit O(N²) | Yes, context mode only | `workflows.py:116` passes the raw list to `raw_sequence`, and `dependencies_for` (`dataset.py:25`) builds a new `SourceIndex` per record. The default baseline (context off) is unaffected. |
| N2 `.env.example` NVIDIA var | Yes | Unconditional `CUBLAS_WORKSPACE_CONFIG=:4096:8`; runbook uses `--env-file .env` for AMD/Jupyter. |
| N3 extraction not resumable | Yes | `acquisition.py:129` refuses if any output exists; report written only at the end. |
| N4 test gaps | Yes as committed tests | Partly offset by the executed chain in §2. |
| N5 notebook overfit trap | Yes | Notebook cell 30 always runs `train`; `overfit_uids` is set only by the `overfit` CLI (`cli.py:73`). `configs/overfit.yaml` in the notebook trains the full train set unregularized. |

Items A found that B missed: H1–H4, M1–M6 below.

## 4. Final task list for the coding agent
One commit per task, CPU tests only. No downloads of the full archive, no training, no push without approval.

### P0: before any real data
1. **Audit must not freeze a manifest when poses are missing (A-H2).** Default: fail and write nothing immutable. Add an explicit `--allow-missing-poses` draft mode that writes a non-frozen manifest.
   *Accept:* an audit with missing poses leaves no frozen manifest, and the same audit succeeds after the poses exist.
2. **Source-map promotion (A-H1).** `make-source-map --promote --evidence "<text>"` writes `data/metadata/source_mapping.csv` for sentence-suffix UIDs (evidence: the iSign HF README documents `[video_id]-[sequence_number]`, dataset revision pinned). DEF mappings need a separate explicit `--approve-def` flag. Never overwrite an existing map.
   *Accept:* real CSV goes candidates → promote → audit → split with no manual editing.
3. **Notebook overfit guard (B-N5).** Add an explicit notebook overfit cell that calls `overfit --uids <path>`. Make `train` fail when the config is the debug preset (an explicit `training.debug_only: true` flag) and `overfit_uids` is unset.
   *Accept:* a test shows `train` with overfit.yaml and no UIDs raises before data loading.
4. **Env template (B-N2).** Remove the active `CUBLAS_WORKSPACE_CONFIG` line (runtime.py already sets it for CUDA only), or ship `.env.cuda` / `.env.rocm`.
   *Accept:* a test loads `.env.example` and asserts no NVIDIA workspace var on the ROCm path.

### P1: before acquiring poses
5. **Selective remote acquisition (A-H3) + size report.** Make `MultipartReader` accept range-readable remote files (`huggingface_hub.HfFileSystem` / fsspec) for the 4 parts (48.3+48.3+48.3+25.3 GB). Read only the ZIP central directory, then stream only the selected members. Add `fetch_isign.py --plan` to print the selected member count and total compressed and uncompressed bytes **before** any download.
   *Accept:* a fake remote split ZIP served by a local HTTP range server; the test asserts the bytes transferred ≈ the central directory plus the selected members.
   *Fallback if range access fails:* process one part at a time (read the central directory from part_ad first, extract the members fully inside each part, keep the next part for members that cross a boundary, then delete). Peak disk ≈ one or two parts (~50–100 GB), not 170 GB.
6. **Resumable extraction (B-N3).** Add a `--resume` mode with a per-file journal bound to the revision, archive identity and selection hash. Reuse a file only when its size and CRC/sha match the ZIP entry; reject anything else. Finalize `extraction.json` at the end.
   *Accept:* interrupt after one member → resume reuses it, extracts the rest, and rejects a corrupted file.
7. **Checkpoint disk (A-H4).** Default `retain_epoch_checkpoints: 0`. Hard-link identical best-* snapshots from the same epoch. Add optional recorded bf16 inference snapshots. Clean stale `.last.pt.*` temp files at start (A-L7).
   *Accept:* one epoch improving all three metrics uses one snapshot's worth of disk; currently a run is ~2.8 GB (last.pt 929 MB, each snapshot 312 MB).

### P2: before paid AMD runs
8. **Resume environment identity (A-M1).** Compare declared deps + torch/CUDA/HIP versions only. Add `--allow-environment-change` with a logged full diff.
9. **ROCm determinism probe (A-M2).** Add a tiny-T5 forward/backward/generate check to `env-check`.
10. **Real lockfile (A-M3).** Pin the tested torch/torchvision minors per extra and commit `uv.lock`. Hark's `uv lock` failed on a `triton-rocm` conflict with torch 2.14 rocm7.2.
11. **Committed tests for N4 gaps (B-N4).** Write and read a pose-format fixture; interpolation, anchor and velocity-gap cases; production `update_window` with uneven and partial windows; `optimizer_for` across freeze/unfreeze; `filter_pretraining`; BPE and attention-LSTM decode; average → Predictor; full `run_training` resume through a loader. Turn Hark's synthetic CLI chain into one slow-marked integration test.
12. **Completed-epoch throughput (B-A3).** Report examples/s per stage for completed epochs, including data loading, and feed it to `estimate`.
13. **DEF vs sentence metrics (A-M5).** Add `clip_type` and report both. About 8% of val/test are DEF clips.
14. **Parameter counts (A-M4).** Record total, visual, T5 and per-stage trainable counts.

### P3: cleanup
15. Remove the unused `file_hash` import in dataset.py (A-L1). Update stale "tests not run" claims in docs and the notebook (A-L2). Clarify or remove the duplicate `configs/velocity.yaml` (A-L3). Restore a minimal root `README.md` → `docs/README.md` (A-L5). Move `ISL_Fix_Plan.md` and the review files to `docs/reviews/` (A-L6).
16. **Context scaling (B-N1 / A-L4)**, only before enabling context. Build `SourceIndex` once in the audit, limit dependencies to clips overlapping the time window, and store source identity once.
17. BERTScore resource hashing (B-A2). Optional.

### Decisions deferred until real poses exist
- Velocity scale / standardization as baseline (A-M6). Run train-only stats first.
- Seed aggregation (B-A1): predeclare one fixed final beam for the 3-seed protocol (simplest, honest). Keep per-seed tuning only as a secondary analysis.
- RGB memory (B-A4): measure only if the RGB system is pursued.

## 5. Agreed data plan
1. **ISLTranslate first.** Select the UIDs for train / val / test / gold from the audited split (31,221 clips, 796 sources).
2. **Optional iSign subset later.** Choose it reproducibly **by source video** with a fixed seed. Exclude **every** ISLTranslate source video (not only the held-out parents) and exact or normalized-text duplicates. Size it by a **byte or hours budget** read from `--plan`, not a flat 15%: clip lengths vary, so 15% of clips is not 15% of bytes.
3. **Measure before downloading.** `fetch_isign --plan` reads the archive index and reports exact sizes. The earlier 2–5 GB cache figure was an estimate only.
4. **Fetch only the selected members** by range access (P1 #5), or use the part-by-part fallback.
5. **Build caches, verify, then clean up.** Check coverage, checksums and cache-only loading. Then delete the archive parts and temp files. **Keep the selected raw poses** and back them up to a private HF dataset repo or Drive. Each cache is built for one feature setup, so the face / bone / local_hand / no_velocity / standardized experiments need the raw poses to rebuild theirs.

## 6. After the tasks
Re-run the full pytest suite and the synthetic CLI chain. Then `fetch_isign --plan` for ISLTranslate (+ the chosen iSign pretraining subset) → selective fetch → inspect one real pose → audit → split → 32–64-clip overfit → short measured baseline. Merge to `master` via PR only after approval.
