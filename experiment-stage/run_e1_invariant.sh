#!/usr/bin/env bash
# E1: right-orthogonal-invariant attacks (Gram matrix + singular-value spectrum).
# Predicts: orthogonal cache leaks (top1 high) under gram_l2/svd_l2, affine-mask
# suppresses to near chance. Args: <gpu> <model_cache_dir> <tag>
set -u
GPU="$1"; MODEL="$2"; TAG="$3"
cd /home/feihm/llm-fei/PUF4Secure_KVcache
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=src HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
PY=/home/feihm/llm-fei/.llm/bin/python
RUNS=experiments/runs
LOG="experiment-stage/e1_${TAG}.log"; : > "$LOG"
COMMON="--model-cache-dir $MODEL --samples 60 --candidates 32 --secret-types all --layers 0,mid,last --include-v --force-fp32"

run() {
  local dm="$1" extra="$2" out="$3"
  echo "[$(date '+%H:%M:%S')] $out" | tee -a "$LOG"
  $PY scripts/run_candidate_secret_collision.py $COMMON \
     --distance-mode "$dm" $extra --modes plain protected_native \
     --out "$RUNS/$out" >>"$LOG" 2>&1
  echo "[$(date '+%H:%M:%S')] exit $?" | tee -a "$LOG"
}

run gram_l2 ""                         "wave13_invariant_gram_l2_ortho_${TAG}_60x32_layers0midlast_fp32.json"
run svd_l2  ""                         "wave13_invariant_svd_l2_ortho_${TAG}_60x32_layers0midlast_fp32.json"
run gram_l2 "--affine-mask --mask-std 128" "wave13_invariant_gram_l2_affine_std128_${TAG}_60x32_layers0midlast_fp32.json"
run svd_l2  "--affine-mask --mask-std 128" "wave13_invariant_svd_l2_affine_std128_${TAG}_60x32_layers0midlast_fp32.json"
echo "[$(date '+%H:%M:%S')] E1 $TAG DONE" | tee -a "$LOG"
