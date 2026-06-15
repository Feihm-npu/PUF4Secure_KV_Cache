#!/usr/bin/env bash
# RQ7 large-candidate confidentiality sweep (Route B). Validates Thm A (orthogonal
# top-1 = 1.0 at every N) and Thm B (affine ~ 1/N, advantage independent of N).
set -u
cd /home/feihm/llm-fei/PUF4Secure_KVcache
PY=/home/feihm/llm-fei/.llm/bin/python
M=${MODEL:-/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B}
TAG=${TAG:-qwen3}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=src
COMMON="--model-cache-dir $M --secret-types verification_code --layers 0 \
  --distance-modes norm_l2,gram_l2 --include-v --force-fp32 --fp32-cache --device-id device_A"

run() {  # N samples variant
  local N=$1 S=$2 V=$3 AF=""
  [ "$V" = "affine" ] && AF="--affine-mask --mask-std 128"
  local MODES="protected_native"
  [ "$V" = "orth" ] && MODES="plain protected_native"
  echo "=== N=$N samples=$S variant=$V ==="
  $PY scripts/run_candidate_secret_collision.py $COMMON \
    --candidates "$N" --samples "$S" --modes $MODES $AF \
    --out "experiments/runs/rq7_${TAG}_N${N}_${V}.json" || echo "FAILED N=$N V=$V"
}

run 32   30 orth
run 32   30 affine
run 100  30 orth
run 100  30 affine
run 1000 20 orth
run 1000 20 affine
echo "RQ7 DONE"
