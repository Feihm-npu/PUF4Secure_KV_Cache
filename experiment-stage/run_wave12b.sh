#!/usr/bin/env bash
# Wave-12b: remaining jobs after J1/J2 completed (J3 affine perf, J4/J5 perf
# reruns w/ longer horizon, J6 MMLU). Sequential for clean perf timing.
set -u
cd /home/feihm/llm-fei/PUF4Secure_KVcache
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=src
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
PY=/home/feihm/llm-fei/.llm/bin/python
RUNS=experiments/runs
LOG=experiment-stage/wave12b.log
: > "$LOG"

run() {
  echo "=================================================================" | tee -a "$LOG"
  echo "[$(date '+%H:%M:%S')] START: $1" | tee -a "$LOG"
  shift
  "$@" >>"$LOG" 2>&1
  echo "[$(date '+%H:%M:%S')] EXIT $?" | tee -a "$LOG"
}

run "J3 affine perf (plain vs affine, long horizon)" \
  $PY scripts/run_performance_eval.py \
  --out $RUNS/wave12_affine_perf_qwen3_0p6b_std128_fp32.json \
  --modes plain wrapped --force-fp32 --affine-mask --mask-std 128 \
  --prompt-lengths 64 256 1024 --decode-tokens 128 --warmup 2 --repeats 8

run "J4 orthogonal perf rerun Qwen3 (long horizon)" \
  $PY scripts/run_performance_eval.py \
  --out $RUNS/wave12_perf_qwen3_0p6b_fp32_long_horizon.json \
  --modes plain wrapped --force-fp32 \
  --prompt-lengths 64 256 1024 --decode-tokens 128 --warmup 2 --repeats 8

run "J5 orthogonal perf rerun Qwen2.5-7B (long horizon)" \
  $PY scripts/run_performance_eval.py \
  --model-cache-dir /home/feihm/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct \
  --out $RUNS/wave12_perf_qwen2p5_7b_fp32_long_horizon.json \
  --modes plain wrapped --force-fp32 \
  --prompt-lengths 64 256 --decode-tokens 128 --warmup 2 --repeats 8

run "J6 MMLU orthogonal wrapper (256 samples)" \
  $PY scripts/run_utility_eval.py \
  --out $RUNS/wave12_mmlu_qwen3_0p6b_fp32.json \
  --modes plain wrapped --force-fp32 \
  --ppl-samples 0 --mc-samples 0 --long-decode-prompts 0 --max-new-tokens 0 \
  --mmlu-samples 256

echo "[$(date '+%H:%M:%S')] ALL DONE" | tee -a "$LOG"
