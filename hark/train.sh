#!/bin/bash
# train.sh WORKDIR CONFIG [random]  -> runs training in foreground, logs to /root/logs/<config>.log
wd=$1; cfg=$2; name=$(basename $cfg .yaml); mkdir -p /root/logs
cd $wd; set -a; . ./.env; set +a
export PYTHONUNBUFFERED=1 MIOPEN_USER_DB_PATH=/mnt/scratch/miopen-$name
[ "${3:-}" = random ] && export ISL_ALLOW_SOURCE_OVERLAP=1
/mnt/scratch/islenv/bin/python -m isl_slt train --config $cfg > /root/logs/$name.log 2>&1
echo "$name rc=$? $(date -u +%FT%TZ)" >> /root/train.status
