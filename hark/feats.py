"""Parallel feature cache build: feats.py CONFIG PARTITION[,PARTITION...] [WORKERS]. Writes index.json only when building all target partitions."""
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
cfg_path, parts = sys.argv[1], sys.argv[2].split(","); W = int(sys.argv[3]) if len(sys.argv) > 3 else 18
from isl_slt.config import load_config
from isl_slt.data.splits import checked_splits
from isl_slt.data.context import SourceIndex
CFG = load_config(cfg_path); PARTS, _ = checked_splits(CFG.data)
IDX = {n: SourceIndex(r) for n, r in PARTS.items()}
def work(item):
    from isl_slt.data.dataset import dependencies_for, raw_sequence
    from isl_slt.features.cache import write_features, cache_paths
    from isl_slt.features.pose import prepare_pose
    name, i = item; rec = PARTS[name][i]
    try:
        arr, meta = cache_paths(CFG, rec)
        if Path(meta).exists() and Path(arr).exists():
            m = json.load(open(meta)); return name, rec.uid, m["frames"], m["input_dim"], None
        seq = raw_sequence(rec, IDX[name], CFG); prep = prepare_pose(seq, CFG.features, CFG.context.enabled)
        md = write_features(CFG, rec, prep, dependencies_for(rec, IDX[name], CFG))
        return name, rec.uid, md["frames"], md["input_dim"], None
    except Exception as e:
        return name, rec.uid, 0, 0, repr(e)[:300]
if __name__ == "__main__":
    t = time.time(); items = [(n, i) for n in parts for i in range(len(PARTS[n]))]
    counts = {n: [] for n in parts}; errors = []
    with Pool(W) as p:
        for k, (n, uid, fr, dim, err) in enumerate(p.imap_unordered(work, items, chunksize=32)):
            if err: errors.append((n, uid, err))
            else: counts[n].append({"uid": uid, "frames": fr, "input_dim": dim})
            if k % 5000 == 0: print(k, len(items), len(errors), round(time.time() - t), flush=True)
    from isl_slt.features.cache import preprocessing_fingerprint
    fp = preprocessing_fingerprint(CFG); out = Path(CFG.data.feature_dir) / fp
    json.dump({"errors": errors, "counts": {n: len(v) for n, v in counts.items()}, "seconds": time.time() - t},
              open(out / f"build-{'-'.join(parts)}.json", "w"), indent=1)
    if set(parts) >= {"train", "validation", "test", "gold"} and not errors:
        from isl_slt.utils.io import write_json
        write_json(out / "index.json", {"preprocessing_fingerprint": fp, "partitions": counts})
    print("DONE", fp, {n: len(v) for n, v in counts.items()}, "errors", len(errors), errors[:5])
