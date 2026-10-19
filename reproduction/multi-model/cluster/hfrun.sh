#!/bin/bash
# usage: GPU=5 hfrun.sh <log-name> <script.py> [args...]  -- HF/torch jobs on athena02 with the fla/sklearn overlay
source /data/user/rh_probe/env.sh
export HF_HUB_OFFLINE=1 PYTHONPATH=/data/user/rh_probe/pylib CUDA_VISIBLE_DEVICES=${GPU:?set GPU} TMPDIR=/data/user/rh_probe/tmp TRITON_CACHE_DIR=/data/user/rh_probe/tmp/triton
N=$1; shift
cd /data/user/rh_probe && exec $PY "$@" > logs/$N.log 2>&1
