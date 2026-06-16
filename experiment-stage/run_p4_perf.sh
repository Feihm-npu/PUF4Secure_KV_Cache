#!/usr/bin/env bash
set -u; cd /home/feihm/llm-fei/PUF4Secure_KVcache
PY=/home/feihm/llm-fei/.llm/bin/python; M=/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B
export PYTHONPATH=src
C="--model-cache-dir $M --prompt-lengths 256 1024 --decode-tokens 128 --repeats 8 --warmup 2 --force-fp32 --fp32-cache"
echo "### plain + orthogonal"; $PY scripts/run_performance_eval.py $C --modes plain wrapped --out experiments/runs/p4_perf_orth.json 2>&1 | grep -iE "prompt_len|tokens_per_s|mode|wrapped|plain" | grep -viE "skip|warn"
echo "### affine eager";       $PY scripts/run_performance_eval.py $C --modes wrapped --affine-mask --mask-std 128 --attn-backend eager        --out experiments/runs/p4_perf_affine_eager.json 2>&1 | grep -iE "prompt_len|tokens_per_s" | grep -viE "skip|warn"
echo "### affine fused";       $PY scripts/run_performance_eval.py $C --modes wrapped --affine-mask --mask-std 128 --attn-backend triton_fused --out experiments/runs/p4_perf_affine_fused.json 2>&1 | grep -iE "prompt_len|tokens_per_s" | grep -viE "skip|warn"
echo "P4 PERF DONE"
