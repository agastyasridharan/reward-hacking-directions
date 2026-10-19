# usage: GPU=4 PORT=8300 COND=neg2 bash steer_neg/serve.sh   (COND=0: no steering)
source /data/user/rh_probe/env.sh
V=/data/user/latekey/vllm-env
export CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13 PATH=$V/lib/python3.12/site-packages/nvidia/cu13/bin:$V/bin:$PATH
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=/data/hf_cache TMPDIR=/data/user/rh_probe/tmp FLASHINFER_WORKSPACE_BASE=/data/user/rh_probe/tmp
export VLLM_CACHE_ROOT=/data/user/rh_probe/vllm_cache_steer_$COND VLLM_DISABLE_COMPILE_CACHE=1
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/data/user/rh_probe/steer_neg/steerplug STEER_LAYER=33
[ "$COND" != "0" ] && export STEER_VEC=/data/user/rh_probe/steer_neg/$COND.npy
exec vllm serve $(ls -d /data/hf_cache/models--zai-org--GLM-4.5-Air-FP8/snapshots/*) \
  --served-model-name glm-4.5-air --host 127.0.0.1 --port $PORT --api-key $(cat /data/user/rh_probe/api_key) \
  --tensor-parallel-size 1 --max-model-len 73728 --gpu-memory-utilization 0.93 --enable-prefix-caching \
  --disable-custom-all-reduce
