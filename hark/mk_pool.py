"""Write data/raw/iSign.pretrain_pool.csv: iSign v1.1 rows whose source video is NOT any ISLTranslate source and whose pose exists."""
import csv, json, re, sys, zipfile
from pathlib import Path
from isl_slt.data.archives import local_part, MultipartReader
from isl_slt.data.acquisition import pose_index
from isl_slt.config import Adapter
from isl_slt.data.manifests import source_candidate
parts = [local_part(f"/mnt/scratch/isign/iSign-poses_v1.1_part_{s}") for s in ("aa", "ab", "ac", "ad")]
with MultipartReader(parts) as combined, zipfile.ZipFile(combined) as z:
    index = pose_index(z)
json.dump(sorted(index), open("artifacts/isign_pose_uids.json", "w"))
pats = Adapter().candidate_uid_patterns
target_src = {r["source_id"] for r in csv.DictReader(open("data/metadata/source_mapping.csv"))}
target_uids = {r["uid"] for r in csv.DictReader(open("data/raw/ISLTranslate.csv"))}
rows = list(csv.DictReader(open("data/raw/iSign_v1.1.csv")))
out, stats = [], {"rows": len(rows), "target_source": 0, "target_uid": 0, "no_pose": 0, "unresolved": 0, "dup_uid": 0}
seen = set()
for r in rows:
    m, rule, _ = source_candidate(r["uid"], pats)
    if m is None: stats["unresolved"] += 1; continue
    if m["source"] in target_src: stats["target_source"] += 1; continue
    if r["uid"] in target_uids: stats["target_uid"] += 1; continue
    if r["uid"] not in index: stats["no_pose"] += 1; continue
    if r["uid"] in seen: stats["dup_uid"] += 1; continue
    seen.add(r["uid"]); out.append((r["uid"], r["text"]))
with open("data/raw/iSign.pretrain_pool.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["uid", "text"]); w.writerows(out)
stats["kept"] = len(out); stats["archive_poses"] = len(index)
print(json.dumps(stats)); json.dump(stats, open("artifacts/pretrain_pool_stats.json", "w"), indent=1)
