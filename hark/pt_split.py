"""pt_split.py BASE_SPLIT CONFIG: same train/val/test/gold as BASE_SPLIT plus the filtered pretrain partition."""
import os, sys, json
base, cfgp = sys.argv[1], sys.argv[2]
if "random" in base: os.environ["ISL_ALLOW_SOURCE_OVERLAP"] = "1"
from isl_slt.config import load_config
from isl_slt.data.manifests import load_manifest
from isl_slt.data.splits import export_splits, filter_pretraining
cfg = load_config(cfgp)
parts = {n: load_manifest(f"{base}/{n}.jsonl") for n in ("train", "validation", "test", "gold")}
kept, rep = filter_pretraining(load_manifest(cfg.data.pretrain_manifest), parts)
parts["pretrain"] = kept
meta = export_splits(parts, cfg.data, rep)
old = json.load(open(f"{base}/splits.json"))["hashes"]
assert all(old[k] == meta["hashes"][k] for k in old), "target membership changed"
print("PT_SPLIT", cfg.data.split_dir, meta["counts"], "excluded", len(rep["excluded"]))
