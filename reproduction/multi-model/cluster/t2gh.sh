source /data/user/rh_probe/env.sh
export HF_HUB_OFFLINE=0 HF_HOME=/data/user/rh_probe/hf_home PYTHONPATH=/data/user/rh_probe/pylib CUDA_VISIBLE_DEVICES=5 TMPDIR=/data/user/rh_probe/tmp TRITON_CACHE_DIR=/data/user/rh_probe/tmp/triton
cd /data/user/rh_probe && exec $PY t2_acts_gptoss.py --model $(ls -d /data/user/hf_cache_latekey/models--openai--gpt-oss-120b/snapshots/*) --dom dom/gptoss_h/dom_vectors.npz --rollouts out_gptoss_full.jsonl --out t2/gptoss_h > logs/t2_gptoss_h.log 2>&1
