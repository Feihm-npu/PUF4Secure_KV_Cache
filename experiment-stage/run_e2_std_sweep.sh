#!/usr/bin/env bash
# E2: affine-mask std tradeoff. For each std: security (gram_l2 + norm_l2
# candidate top1, the strongest invariant attacks) vs numerical error (fp32
# wrapper sanity max logit diff). Builds the security-vs-precision curve that
# justifies the std choice instead of a single unprincipled value.
set -u
cd /home/feihm/llm-fei/PUF4Secure_KVcache
export CUDA_VISIBLE_DEVICES="${1:-4}" PYTHONPATH=src HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
PY=/home/feihm/llm-fei/.llm/bin/python
RUNS=experiments/runs
MODEL=/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B
LOG=experiment-stage/e2_sweep.log; : > "$LOG"

for STD in 4 16 64 128 256 512; do
  echo "[$(date '+%H:%M:%S')] === std=$STD ===" | tee -a "$LOG"
  # numerical error (sanity)
  $PY scripts/run_wrapper_sanity.py --model-cache-dir $MODEL --force-fp32 \
    --affine-mask --mask-std $STD \
    --out $RUNS/wave13_e2_sanity_std${STD}_qwen3_0p6b_fp32.json >>"$LOG" 2>&1
  # security: strongest invariant attack (gram) + norm baseline
  for DM in gram_l2 norm_l2; do
    $PY scripts/run_candidate_secret_collision.py --model-cache-dir $MODEL \
      --samples 40 --candidates 32 --secret-types all --layers 0,mid,last \
      --include-v --force-fp32 --distance-mode $DM \
      --affine-mask --mask-std $STD --modes plain protected_native \
      --out $RUNS/wave13_e2_${DM}_std${STD}_qwen3_0p6b_40x32_fp32.json >>"$LOG" 2>&1
  done
  echo "[$(date '+%H:%M:%S')] std=$STD done" | tee -a "$LOG"
done
echo "[$(date '+%H:%M:%S')] E2 SWEEP DONE" | tee -a "$LOG"
