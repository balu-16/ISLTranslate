#!/bin/bash
# Sharded extraction of ISLTranslate poses (6 shards) and the pretrain pool (10 shards) from local iSign parts.
cd /mnt/scratch/ISLTranslate; P=/mnt/scratch/islenv/bin/python; A="/mnt/scratch/isign/iSign-poses_v1.1_part_aa /mnt/scratch/isign/iSign-poses_v1.1_part_ab /mnt/scratch/isign/iSign-poses_v1.1_part_ac /mnt/scratch/isign/iSign-poses_v1.1_part_ad"
D=artifacts/acquisition; mkdir -p $D/sh
[ -s data/raw/iSign.pretrain_pool.csv ] || { $P hark/mk_pool.py > $D/pool.out 2>&1; echo "POOL rc=$?" >> /root/extract.status; }
[ -s data/metadata/isign_source_mapping.csv ] || { $P -m isl_slt make-source-map --config configs/r5-src.yaml --dataset pretrain --promote --approve-def \
  --evidence "iSign README at revision e4ee6c5f0d9dfcbc74205e3f1388ce94da26c298 defines UID as [video_id]-[sequence_number] and recommends splitting by video_id. Same convention as the promoted ISLTranslate map: sentence rows source = UID prefix before the final -N; DEF rows (_w/_d/_eN) source = 11-char YouTube id. Pool excludes every ISLTranslate source video (pretrain pool built 2026-11-05)." > $D/srcmap_pretrain.json 2>&1; echo "SRCMAP rc=$?" >> /root/extract.status; }
$P - <<'PY'
import csv
for name, src, n in (("t", "data/raw/ISLTranslate.available.csv", 6), ("p", "data/raw/iSign.pretrain_pool.csv", 10)):
    rows = list(csv.DictReader(open(src))); hdr = list(rows[0].keys())
    for k in range(n):
        with open(f"artifacts/acquisition/sh/{name}{k}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, hdr); w.writeheader(); w.writerows(rows[k::n])
PY
run(){ # kind shard
  if [ $1 = t ]; then extra="--config configs/baseline.yaml"; out=data/raw/poses-sh/$2; else extra="--config configs/r5-src.yaml --dataset pretrain --budget-bytes 10000000000000"; out=data/raw/isign-poses-sh/$2; fi
  $P scripts/fetch_isign.py --archives $A --uids-csv $D/sh/$1$2.csv $extra --out $out --max-bytes 400000000000 > $D/sh/x$1$2.json 2> $D/sh/x$1$2.err
  echo "$1$2 rc=$?" >> /root/extract.shards
}
gather(){ mkdir -p $2; for d in $1/*; do find $d -maxdepth 1 -name '*.pose' -exec mv -t $2 {} +; done; }
( for k in 0 1 2 3 4 5; do run t $k & done; wait
  if [ $(grep -c "^t[0-9] rc=0" /root/extract.shards) = 6 ]; then gather data/raw/poses-sh data/raw/poses; echo "TARGET rc=0 files=$(ls data/raw/poses | wc -l)" >> /root/extract.status; else echo "TARGET rc=1" >> /root/extract.status; fi ) &
( for k in 0 1 2 3 4 5 6 7 8 9; do run p $k & done; wait
  if [ $(grep -c "^p[0-9] rc=0" /root/extract.shards) = 10 ]; then gather data/raw/isign-poses-sh data/raw/isign-poses; echo "PRETRAIN rc=0 files=$(ls data/raw/isign-poses | wc -l)" >> /root/extract.status; else echo "PRETRAIN rc=1" >> /root/extract.status; fi ) &
wait
echo EXTRACT_DONE >> /root/extract.status
