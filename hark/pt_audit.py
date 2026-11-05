"""Parallel audit of the pretrain pool -> data/manifests/pretrain.jsonl (records with readable poses only; failures listed)."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
from isl_slt.config import load_config
from isl_slt.data.manifests import adapt_csv, save_manifest, audit_records
CFG = load_config("configs/r5-src.yaml"); AUD = {}
RECS = adapt_csv(CFG.data.pretrain, AUD)
def work(i):
    from isl_slt.utils.io import file_hash
    from isl_slt.data.pose import read_pose
    from isl_slt.features.pose import prepare_pose
    r = RECS[i]
    if not Path(r.pose_path).is_file(): return i, None, "missing_pose"
    h = file_hash(r.pose_path)
    try:
        seq = read_pose(r, CFG.features); prepare_pose(seq, CFG.features, CFG.context.enabled)
    except (ValueError, OSError) as e:
        return i, h, repr(e)[:200]
    return i, h, None
if __name__ == "__main__":
    ok, bad = [], []
    with Pool(18) as p:
        for i, h, err in p.imap_unordered(work, range(len(RECS)), chunksize=64):
            r = RECS[i]
            if err: bad.append({"uid": r.uid, "error": err}); continue
            ok.append(r.model_copy(update={"asset_hashes": {"pose": h},
                "duplicate_fingerprints": sorted(set(r.duplicate_fingerprints) | {f"pose-file-sha256:{h}"})}))
    ok.sort(key=lambda r: r.uid)
    save_manifest(CFG.data.pretrain_manifest, ok)
    rep = {**audit_records(ok), **AUD, "kept": len(ok), "excluded": bad, "pool": len(RECS)}
    json.dump(rep, open("data/manifests/pretrain.audit.json", "w"), indent=1, default=str)
    print("PT_AUDIT kept", len(ok), "excluded", len(bad), bad[:3])
