# Experiment Audit

## Current Status

WARN with one failed security subclaim and two prototype mitigation directions. Wave-0 cross-model sanity, Wave-1 multi-model utility, Wave-2 PII/precision, Wave-3 adaptive profiling, Wave-4 prototype performance, Wave-7 real-context PII, Wave-9 Llama real-context PII, Wave-10 unit-norm-cache sanity/attack artifacts, and Wave-11 no-sidecar transform artifacts are parseable. However, broad finite-candidate cache confidentiality for the current orthogonal-only design is FAILED by Wave-9 norm-L2 attacks: protected top1/MRR are 1.0 on Qwen3-0.6B, Llama-3.2-1B, and Qwen2.5-1.5B. The Wave-10 unit-norm sidecar prototype blocks the tested norm-L2 attack on Qwen3 and Llama, but needs trusted sidecar state. The Wave-11 affine-mask prototype avoids sidecar state and reduces tested norm/raw-L2 candidate recovery near chance, but it requires PUF unmasking inside attention and has not yet been audited for overhead, stronger attacks, or long-decode numerical stability. Llama-2-7B full utility, 1,000-prompt PII, real PUF validation, optimized-kernel performance, and full mitigation validation remain pending and must not be claimed as completed.

## Audit Notes

- PASS: `puf_attention.py` now supports qwen3/qwen2/llama and has fp32 sanity artifacts on five local models.
- PASS: Qwen3 500-prompt PII benchmark has exact leakage metrics and Wilson intervals; Llama-3.2-1B has a 120-prompt cross-family PII run.
- PASS: bf16 divergence is quantified with a matched fp32 control.
- PASS: KPA confirms same-session learnability and session-refresh non-transfer on Qwen3 and Llama-3.2-1B.
- PASS: Qwen2.5-7B performance exists and shows unchanged KV memory with 9.7-12.2% decode throughput loss in the fp32 prototype.
- PASS: Real-context seeded PII on CC-News contexts supports the injection-leakage claim beyond template-only prompts; Qwen3 and Llama-3.2-1B both have 300-prompt runs with protected keyword leakage 0/300.
- PASS: Candidate-secret collision gives an architecture-independent attack harness for non-square `v_proj` models.
- FAIL: The broad claim that orthogonal cache rotation prevents finite-candidate secret matching is false under norm-only matching. Orthogonal maps preserve per-token/head/layer vector norms, and Wave-9 `norm_l2` recovers protected secrets with top1/MRR 1.0 on Qwen3-0.6B, Llama-3.2-1B, and Qwen2.5-1.5B.
- WARN: Wave-8 raw-L2 candidate-collision suppression should be described only as direction-space mismatch, not as cache confidentiality.
- WARN: Random scalar and diagonal norm blinding are insufficient in the tested Qwen3 setup; protected top1 remains 0.117-0.250 depending on variant.
- PROTOTYPE: Unit-normalized exposed cache plus a private in-process norm sidecar preserves fp32 logits on Qwen3/Llama sanity prompts and reduces norm-L2 protected top1 to 0.000 on Qwen3 and 0.033 on Llama for 60x32 multi-secret runs. This can motivate a redesign, but it adds trusted sidecar state and needs stronger attacker/performance audits.
- NEGATIVE: Bounded non-orthogonal `O D` linear transforms need no sidecar and preserve fp32 sanity, but they do not sufficiently break norm ranking; Qwen3 protected top1 remains 0.267 even at log range 6.
- PROTOTYPE: PUF-derived affine masking needs no sidecar. At mask std 128, Qwen3 norm-L2 protected top1 drops to 0.050 and Llama drops to 0.000; raw-L2 top1 drops to 0.083 on Qwen3 and 0.000 on Llama. It should be treated as a prototype because it subtracts PUF masks inside attention and introduces cancellation/overhead risk.
- WARN: Fast Givens is correct but slower than dense in the current standalone PyTorch/Triton implementation, so optimized-overhead claims remain unsupported.
- WARN: Performance numbers use the unoptimized PyTorch monkey-patch, not a fused serving kernel.
- WARN: PII benchmark is synthetic; real PII datasets are not used.

## Integrity Rules

- Do not report a metric unless a JSON artifact exists.
- Mark missing multi-model, performance, and PII results as TODO in the paper until measured.
- Preserve model path, sample count, split, precision, and mode for each result.
- Do not claim A0 cache-only confidentiality against finite-candidate matching for the orthogonal-only design unless a norm-blinding mitigation is implemented and evaluated.
- Do not present unit-norm sidecar as final unless the paper explicitly models the sidecar as trusted non-exported state and evaluates stronger sidecar-aware attacks, utility, and overhead.
- Do not present affine masking as final until the threat model accepts PUF unmasking inside attention and the implementation passes utility, long-decode, performance, and stronger attack audits.
