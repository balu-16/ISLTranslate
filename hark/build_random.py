"""Rebuild splits/random-v1 exactly as the Kaggle round-3 script did (seed 2026, duplicate groups kept together), and check membership against the saved train/validation."""
import json, os, random, shutil, sys
from pathlib import Path
os.environ["ISL_ALLOW_SOURCE_OVERLAP"] = "1"
from isl_slt.config import load_config
from isl_slt.data.manifests import load_manifest
from isl_slt.data.splits import export_splits
src = {n: load_manifest(f"splits/source-v1/{n}.jsonl") for n in ("train", "validation", "test", "gold")}
groups = {}
for r in src["train"] + src["validation"] + src["test"]:
    groups.setdefault(r.duplicate_fingerprints[0] if r.duplicate_fingerprints else r.uid, []).append(r)
keys = sorted(groups); random.Random(2026).shuffle(keys)
total = sum(len(g) for g in groups.values()); parts = {"train": [], "validation": [], "test": []}; n = 0
for k in keys:
    name = "train" if n < 0.8 * total else ("validation" if n < 0.9 * total else "test")
    parts[name] += groups[k]; n += len(groups[k])
parts["gold"] = src["gold"]
parts = {k: [r.model_copy(update={"split": k}) for r in v] for k, v in parts.items()}
old = Path("splits/random-v1")
for k in ("train", "validation"):
    saved = [json.loads(l)["uid"] for l in open(old / f"{k}.jsonl")]
    assert sorted(saved) == sorted(r.uid for r in parts[k]), f"{k} membership differs from Kaggle random-v1"
shutil.move(str(old), "splits/random-v1.kaggle-partial")
cfg = load_config("configs/r4-rand.yaml")
meta = export_splits(parts, cfg.data)
json.dump({"note": "random clip split for comparison only; sources overlap by design (rebuilt on droplet, train/val identical to Kaggle round 3)", "counts": meta["counts"]}, open("splits/random-v1/NOTE.json", "w"))
print("RANDOM OK", meta["counts"])
