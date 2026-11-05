#!/bin/bash
set -uo pipefail
exec > >(tee -a /root/pipeline.log) 2>&1
M=/mnt/scratch/ISLTranslate; RW=/mnt/scratch/rand; P=/mnt/scratch/islenv/bin/python
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
until grep -q SETUP_DONE /root/setup.status 2>/dev/null; do sleep 20; done; log setup done
cd $M; set -a; . ./.env; set +a; export PYTHONUNBUFFERED=1
bash hark/extract.sh &
until grep -q "TARGET rc=" /root/extract.status 2>/dev/null; do sleep 15; done
grep -q "TARGET rc=0" /root/extract.status || { log TARGET EXTRACT FAILED; tail -20 artifacts/acquisition/extract_target.err; exit 1; }
log target poses extracted: $(ls data/raw/poses | wc -l)
$P hark/build_random.py || { log RANDOM SPLIT FAILED; exit 1; }
$P hark/feats.py configs/r4-src.yaml train,validation,test,gold 18 || exit 1
$P -m isl_slt prepare-stats --config configs/r4-src.yaml | tail -c 300; log stats src done
# random-split workdir: own stats + runs, shared features/poses
mkdir -p $RW && cd $M && tar --exclude=./artifacts/features --exclude=./data/raw/poses --exclude=./data/raw/isign-poses --exclude=./runs --exclude=./artifacts/stats -cf - . | tar -xf - -C $RW
ln -sfn $M/artifacts/features $RW/artifacts/features; ln -sfn $M/data/raw/poses $RW/data/raw/poses; ln -sfn $M/data/raw/isign-poses $RW/data/raw/isign-poses
cd $RW && ISL_ALLOW_SOURCE_OVERLAP=1 $P -m isl_slt prepare-stats --config configs/r4-rand.yaml | tail -c 300; log stats rand done
cd $M
tmux new-window -d -t pipe -n r4src "bash hark/train.sh $M configs/r4-src.yaml"
tmux new-window -d -t pipe -n r4rand "bash hark/train.sh $RW configs/r4-rand.yaml random"
tmux new-window -d -t pipe -n r6jsrc "bash hark/train.sh $M configs/r6-joint-src.yaml"
tmux new-window -d -t pipe -n r6jrand "bash hark/train.sh $RW configs/r6-joint-rand.yaml random"
tmux new-window -d -t pipe -n r6fsrc "bash hark/train.sh $M configs/r6-frozen-src.yaml"
log R4 R6 launched
until grep -q "EXTRACT_DONE" /root/extract.status 2>/dev/null; do sleep 30; done
grep -q "PRETRAIN rc=0" /root/extract.status || { log PRETRAIN EXTRACT FAILED; cat /root/extract.status; tail -20 artifacts/acquisition/extract_pretrain.err artifacts/acquisition/srcmap_pretrain.json artifacts/acquisition/pool.out; exit 1; }
log pretrain poses: $(ls data/raw/isign-poses | wc -l)
$P hark/pt_audit.py || exit 1
$P hark/pt_split.py splits/source-v1 configs/r5-src.yaml || exit 1
cp -r data/manifests/pretrain.* $RW/data/manifests/; cp data/metadata/isign_source_mapping.csv $RW/data/metadata/; cp data/raw/iSign.pretrain_pool.csv $RW/data/raw/
(cd $RW && $P $M/hark/pt_split.py splits/random-v1 configs/r5-rand.yaml) || exit 1
$P hark/feats.py configs/r5-src.yaml pretrain 14 || exit 1
tmux new-window -d -t pipe -n r5src "bash hark/train.sh $M configs/r5-src.yaml"
tmux new-window -d -t pipe -n r5rand "bash hark/train.sh $RW configs/r5-rand.yaml random"
log R5 launched
