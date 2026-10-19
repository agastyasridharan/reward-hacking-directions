#!/bin/bash
source /data/user/rh_probe/env.sh
export HF_HUB_OFFLINE=1 PYTHONPATH=/data/user/rh_probe/pylib CUDA_VISIBLE_DEVICES=${GPUS:?} TMPDIR=/data/user/rh_probe/tmp TRITON_CACHE_DIR=/data/user/rh_probe/tmp/triton
M=$(ls -d /data/user/hf_cache_latekey/models--zai-org--GLM-4.5-Air/snapshots/*)
cd /data/user/rh_probe
exec $PY t3_steer.py --family glm --model $M --dom dom/glm/dom_vectors.npz --layer 33 --out t3/glm_cal_think --think --conds 0,1,2 --limit 8 --max-new 2048 --batch 8 > logs/t3_glm_cal_think.log 2>&1
