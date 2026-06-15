# Experiment Plan: PUF-Bound KV-Cache Hardening

This plan tracks the reviewer-facing experiment reinforcement requested for the PUF-bound non-migratable KV-cache paper. Real PUF hardware is explicitly out of scope for this round; all PUF behavior uses the existing simulator until hardware traces are provided.

## Claims and Anti-Claims

| ID | Claim | Anti-claim to rule out | Required evidence |
| --- | --- | --- | --- |
| C1 | The Level-3 attention-basis construction preserves model behavior when computed with sufficient precision. | The wrapper changes model outputs or only works on Qwen3-0.6B. | Cross-model fp32 equivalence, perplexity/MC deltas, long-decode agreement. |
| C2 | Attacker-view rotated native caches suppress inversion, matching, and replay. | The defense only hides the three toy prompts or a Qwen3-specific square projection. | Multi-model attack baseline/defense runs and a 500-1000 prompt synthetic PII benchmark. |
| C3 | Session refresh is the security boundary under adaptive profiling. | A stronger alignment, known-plaintext, or cross-session attacker strips the basis. | Native Hungarian, known-plaintext, and cross-session profiling matrices. |
| C4 | The design has a plausible systems path. | The rotation cost makes serving impractical. | Prefill/decode latency, tokens/s, KV memory, and transform overhead. |
| C5 | The security comparison is well scoped. | AES/TEE or software-only secrets dominate the proposed mechanism. | Threat-model table, analytical comparison, and formal session-basis argument. |

## Model Matrix

| Model | Local cache | Family | Scale | Purpose |
| --- | --- | --- | ---: | --- |
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | qwen3 | 0.6B | Existing baseline and regression target. |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B` | llama | 1B | Cross-family GQA sanity and utility. |
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | qwen2 | 1.5B | Mid-scale Qwen2 GQA. |
| Llama-2-7b-chat | `meta-llama/Llama-2-7b-chat-hf` | llama | 7B | MHA large-model coverage. |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | qwen2 | 7B | Large Qwen2 GQA coverage. |

## Execution Waves

1. Wave 0: refactor `puf_attention.py` into a family-dispatched wrapper for qwen3, qwen2, and llama; sanity-check fp32 equivalence on one short prompt per family.
2. Wave 1: run cross-model utility and leakage baselines using small sanity settings, then expand sample counts after sanity passes.
3. Wave 2: add synthetic PII benchmark and long-decode divergence-rate benchmark; report exact leakage, digit recall, Wilson intervals, and divergence-step curves.
4. Wave 3: add native-cache Hungarian and known-plaintext/cross-session profiling experiments.
5. Wave 4: add end-to-end latency, throughput, KV-memory, and transform-overhead benchmarks.
6. Wave 5: update the paper with a formal threat model, comparison table, and an explicit security argument; no fabricated empirical claims.

## Resource Rules

- Use `/home/feihm/llm-fei/.llm` and `HF_DATASETS_OFFLINE=1`.
- Check GPU state before non-trivial runs. Default to `CUDA_VISIBLE_DEVICES=1` unless occupied.
- For every script, save JSON outputs with model path, command settings, resource snapshots, and cleanup status.
- Run sanity checks before large sweeps. Do not mark a wave complete without parseable artifacts.
