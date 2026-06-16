# Experiment Progress

This document tracks the paper-oriented experiment hardening started from
`comments.md`. All runs used the environment `/home/feihm/llm-fei/.llm` and
checked GPU availability before execution. GPU memory was released after each
run as confirmed by `nvidia-smi`.

## 2026-06-04: Utility and Profiling Batch 1

### Scripts added or extended

| File | Purpose |
| --- | --- |
| `scripts/run_utility_eval.py` | Plain vs Level-3 wrapped utility benchmark: perplexity, HellaSwag-style conditional log-likelihood accuracy, and long greedy decode agreement. |
| `scripts/run_profiling_native.py` | Procrustes profiling directly against native K/V caches produced by `puf_attention.install_puf_attention`. |
| `scripts/run_profiling_large.py` | Added optional Hungarian/profile alignment scenarios with `--include-hungarian` and `--scenario-filter`. |

### Artifacts

| Artifact | Scope |
| --- | --- |
| `experiments/runs/utility_sanity_qwen3_bf16.json` | 4-sample bf16 sanity for utility script. |
| `experiments/runs/utility_sanity_qwen3_fp32.json` | 4-sample fp32 sanity for utility script. |
| `experiments/runs/utility_qwen3_bf16_128.json` | 128-sample bf16 utility run on Qwen3-0.6B. |
| `experiments/runs/utility_qwen3_fp32_128.json` | 128-sample fp32 utility run on Qwen3-0.6B. |
| `experiments/runs/profiling_native_sanity.json` | 50-prompt native profiling sanity. |
| `experiments/runs/profiling_native_summary.json` | 1000-prompt native profiling run. |
| `experiments/runs/profiling_large_hungarian_sanity.json` | 20-prompt Hungarian sanity over cache-level layout defenses. |
| `experiments/runs/profiling_large_hungarian_200.json` | 200-prompt Hungarian adaptive profiling over row/block layout defenses. |

### Key results

| Experiment | Plain | Wrapped / Attacked | Delta / Result |
| --- | ---: | ---: | --- |
| Qwen3 bf16 PPL, ag_news 128 | 57.6048 | 57.5335 | -0.0712 |
| Qwen3 bf16 HellaSwag 128 | 0.4219 | 0.4219 | 0.0000 |
| Qwen3 bf16 long decode | - | - | Prompt 0 diverges at step 4; prompt 2 at step 12. |
| Qwen3 fp32 PPL, ag_news 128 | 57.5900 | 57.5900 | +0.000001 |
| Qwen3 fp32 HellaSwag 128 | 0.4219 | 0.4219 | 0.0000 |
| Qwen3 fp32 long decode | - | - | All 3 prompts match for 64/64 tokens. |
| Native fixed-session profiling | - | V-inv 1.000 | Native rotated cache is profileable if the session basis is reused. |
| Native session-refresh profiling | - | V-inv 0.000 | Basis learned in one session does not transfer to another session. |
| Row layout + Hungarian, 200 prompts | - | V-inv 0.073 | Row layout lets the attacker recover the basis but not the target row order. |
| Block layout + block-aware Hungarian, 200 prompts | - | V-inv 1.000 | Block layout alone is broken by block-aware assignment. |

### Current interpretation

1. The Level-3 wrapper is exact in the fp32 path on the tested utility tasks and
   long-decode prompts. This is the strongest current utility evidence.
2. The bf16 path does not significantly affect aggregate PPL or small-sample
   HellaSwag accuracy, but it still causes long-decode trajectory divergence on
   low-margin prompts. This should be reported as a numerical deployment risk,
   not as a protocol error.
3. Native Level-3 caches behave like the cache-level transform under profiling:
   a fixed basis is broken, while session refresh blocks transfer.
4. Hungarian/profile alignment strengthens the negative results for layout-only
   defenses. Block layout alone should not be claimed as secure; session refresh
   is mandatory.

### Next experiment block

1. Expand utility beyond Qwen3-0.6B once wrappers exist for additional local
   models, starting with Llama-3.2-1B and Qwen2.5-7B.
2. Add a privacy prompt benchmark with hundreds of synthetic PII examples and
   exact secret-leakage metrics.
3. Add end-to-end performance measurement for prefill/decode latency, tokens/s,
   and KV memory overhead under plain, bf16 wrapped, and fp32 wrapped paths.

## 2026-06-05: Reviewer Hardening Waves 0-4

### Scripts added or extended

| File | Purpose |
| --- | --- |
| `src/puf4secure_kvcache/puf_attention.py` | Refactored Level-3 wrapper from Qwen3-only to `qwen3`, `qwen2`, and `llama` RoPE decoder models. |
| `scripts/run_wrapper_sanity.py` | Wave-0 plain-vs-wrapped first-forward logit and short greedy-decode sanity. |
| `scripts/run_pii_benchmark.py` | Wave-2 synthetic PII injection benchmark with keyword/digit leakage and Wilson intervals. |
| `scripts/run_decode_divergence_benchmark.py` | Wave-2 many-prompt long-decode divergence curve for bf16/fp32 precision diagnosis. |
| `scripts/run_performance_eval.py` | Wave-4 prefill/decode latency, decode tokens/s, and KV-memory benchmark. |
| `scripts/run_kpa_native.py` | Wave-3 known-plaintext native-cache profiling sample-complexity benchmark. |
| `scripts/run_profiling_native.py` | Added multi-family support and optional `--alignment hungarian`. |

### Artifacts

| Artifact | Scope |
| --- | --- |
| `experiments/runs/wave0_wrapper_sanity_qwen3_0p6b_fp32.json` | Qwen3-0.6B fp32 wrapper regression. |
| `experiments/runs/wave0_wrapper_sanity_llama3p2_1b_fp32.json` | Llama-3.2-1B fp32 wrapper sanity. |
| `experiments/runs/wave0_wrapper_sanity_qwen2p5_1p5b_fp32.json` | Qwen2.5-1.5B fp32 wrapper sanity. |
| `experiments/runs/wave0_wrapper_sanity_llama2_7b_fp32.json` | Llama-2-7B-chat fp32 wrapper sanity. |
| `experiments/runs/wave0_wrapper_sanity_qwen2p5_7b_fp32.json` | Qwen2.5-7B-Instruct fp32 wrapper sanity. |
| `experiments/runs/wave1_utility_smoke_llama3p2_1b_fp32.json` | Llama-3.2-1B utility smoke. |
| `experiments/runs/wave1_utility_llama3p2_1b_fp32_128.json` | Llama-3.2-1B 128-sample fp32 utility run. |
| `experiments/runs/wave1_utility_smoke_qwen2p5_1p5b_fp32.json` | Qwen2.5-1.5B utility smoke. |
| `experiments/runs/wave1_utility_qwen2p5_1p5b_fp32_128.json` | Qwen2.5-1.5B 128-sample fp32 utility run. |
| `experiments/runs/wave1_utility_qwen2p5_7b_fp32_128.json` | Qwen2.5-7B 128-sample fp32 utility run. |
| `experiments/runs/wave2_pii_qwen3_0p6b_60_fp32.json` | 60-prompt synthetic PII injection benchmark. |
| `experiments/runs/wave2_pii_qwen3_0p6b_500_fp32.json` | 500-prompt synthetic PII injection benchmark. |
| `experiments/runs/wave2_pii_llama3p2_1b_120_fp32.json` | Llama-3.2-1B 120-prompt synthetic PII injection benchmark. |
| `experiments/runs/wave2_decode_divergence_qwen3_0p6b_bf16_32x32.json` | bf16 32-prompt/32-token divergence curve. |
| `experiments/runs/wave2_decode_divergence_qwen3_0p6b_fp32_32x32.json` | fp32 matched divergence control. |
| `experiments/runs/wave3_kpa_native_qwen3_0p6b_4_16_64.json` | KPA sample-complexity residuals for 4/16/64 known prompts. |
| `experiments/runs/wave3_kpa_native_llama3p2_1b_4_16.json` | Llama-3.2-1B KPA sample-complexity residuals for 4/16 known prompts. |
| `experiments/runs/wave3_native_profiling_qwen3_0p6b_hungarian_64.json` | Native Level-3 Hungarian profiling with 64 prompts. |
| `experiments/runs/wave4_perf_qwen3_0p6b_fp32_64_256.json` | Qwen3 fp32 serving performance at 64/256-token prefill. |
| `experiments/runs/wave4_perf_qwen2p5_7b_fp32_64_256.json` | Qwen2.5-7B fp32 serving performance at 64/256-token prefill. |

### Key results

| Experiment | Plain / baseline | Wrapped / defended | Result |
| --- | ---: | ---: | --- |
| Qwen3-0.6B wrapper sanity | - | max logit diff 3.77e-05 | 4/4 or 8/8 greedy tokens match in fp32 sanity. |
| Llama-3.2-1B wrapper sanity | - | max logit diff 1.60e-05 | 8/8 greedy tokens match. |
| Qwen2.5-1.5B wrapper sanity | - | max logit diff 7.77e-05 | 8/8 greedy tokens match. |
| Llama-2-7B wrapper sanity | - | max logit diff 2.34e-05 | 4/4 greedy tokens match. |
| Qwen2.5-7B wrapper sanity | - | max logit diff 1.54e-04 | 4/4 greedy tokens match. |
| Llama-3.2-1B utility smoke PPL | 52.588617 | 52.588650 | Delta +0.000033; 4/4 decode tokens match. |
| Llama-3.2-1B fp32 PPL, ag_news 128 | 22.4110846 | 22.4110857 | Delta +0.0000011. |
| Llama-3.2-1B fp32 HellaSwag 128 | 0.4141 | 0.4141 | Delta 0.0000; long-decode text matches on 3 prompts. |
| Qwen2.5-1.5B utility smoke PPL | 35.656425 | 35.656362 | Delta -0.000063; 4/4 decode tokens match. |
| Qwen2.5-1.5B fp32 PPL, ag_news 128 | 19.4575599 | 19.4575587 | Delta -0.0000012. |
| Qwen2.5-1.5B fp32 HellaSwag 128 | 0.4531 | 0.4531 | Delta 0.0000; long-decode text matches on 3 prompts. |
| Qwen2.5-7B fp32 PPL, ag_news 128 | 15.1818894 | 15.1818911 | Delta +0.0000017. |
| Qwen2.5-7B fp32 HellaSwag 128 | 0.4844 | 0.4844 | Delta 0.0000; long-decode text matches on 3 prompts. |
| Qwen3 PII injection, 60 prompts | 30/60 keyword leaks | 0/60 keyword leaks | Plain 50.0% [37.7, 62.3] vs protected 0.0% [0.0, 6.0] Wilson 95%. |
| Qwen3 PII digit leakage, 60 prompts | 23/60 exact digit leaks | 0/60 exact digit leaks | Mean digit recall drops from 0.615 to 0.000. |
| Qwen3 PII injection, 500 prompts | 261/500 keyword leaks | 0/500 keyword leaks | Plain 52.2% [47.8, 56.5] vs protected 0.0% [0.0, 0.76] Wilson 95%. |
| Qwen3 PII digit leakage, 500 prompts | 190/500 exact digit leaks | 0/500 exact digit leaks | Mean digit recall drops from 0.6020 to 0.0043. |
| Llama-3.2-1B PII injection, 120 prompts | 67/120 keyword leaks | 0/120 keyword leaks | Plain 55.8% [46.9, 64.4] vs protected 0.0% [0.0, 3.1] Wilson 95%. |
| Llama-3.2-1B PII digit leakage, 120 prompts | 35/120 exact digit leaks | 0/120 exact digit leaks | Mean digit recall drops from 0.5229 to 0.0063. |
| Qwen3 bf16 decode, 32x32 | - | divergence rate 0.5625 | Mean divergence step among diverged prompts is 11.44. |
| Qwen3 fp32 decode, 32x32 | - | divergence rate 0.0000 | Matched fp32 control has no divergence. |
| Qwen3 KPA, 4 known prompts | same-session residual 0.0519 | refresh residual 1.4166 | Too few rows partly recover same session; refresh does not transfer. |
| Qwen3 KPA, 16 known prompts | same-session residual 4.21e-05 | refresh residual 1.4167 | Same session essentially solved; refresh remains random-basis mismatch. |
| Qwen3 KPA, 64 known prompts | same-session residual 3.04e-05 | refresh residual 1.4177 | Same session remains solved; refresh remains random-basis mismatch. |
| Llama-3.2-1B KPA, 4 known prompts | same-session residual 7.31e-06 | refresh residual 1.4154 | Cross-family KPA confirms fixed-basis profileability and refresh non-transfer. |
| Llama-3.2-1B KPA, 16 known prompts | same-session residual 3.87e-06 | refresh residual 1.4165 | Same conclusion at higher sample count. |
| Qwen3 native Hungarian, 64 prompts | same-session V-inv 1.000 | refresh V-inv 0.000 | Hungarian does not change the mandatory session-refresh conclusion. |
| Qwen3 perf, 64-token prefill | 29.40 tok/s decode | 27.16 tok/s decode | Wrapped decode throughput -7.6%, KV memory unchanged. |
| Qwen3 perf, 256-token prefill | 29.88 tok/s decode | 25.87 tok/s decode | Wrapped decode throughput -13.4%, KV memory unchanged. |
| Qwen2.5-7B perf, 64-token prefill | 34.51 tok/s decode | 30.30 tok/s decode | Wrapped decode throughput -12.2%, KV memory unchanged. |
| Qwen2.5-7B perf, 256-token prefill | 34.49 tok/s decode | 31.13 tok/s decode | Wrapped decode throughput -9.7%, KV memory unchanged. |

### Current interpretation

1. The Level-3 wrapper now has fp32 equivalence sanity on five locally cached models across Qwen3, Qwen2, and Llama families, including 7B-scale models and MHA/GQA variants.
2. The PII benchmark turns the three-prompt case study into a statistical leakage result: in the 500-prompt Qwen3 batch, plaintext injection leaks exact keywords in 52.2% of prompts, while protected native caches leak none. The Llama-3.2-1B 120-prompt batch shows the same qualitative result.
3. The bf16 risk is now quantified as a curve rather than an anecdote: 56.25% of 32 AG News prompts diverge within 32 greedy tokens in bf16, while the fp32 control remains exact.
4. KPA results strengthen the threat model: a same-session known-plaintext attacker can learn the basis with as few as 16 random prompts, but the learned basis does not transfer to a refreshed session.
5. Initial end-to-end performance numbers show no KV-memory increase and a measurable decode throughput cost on the current PyTorch wrapper. The 7B Qwen2.5 run shows 9.7-12.2% decode throughput loss under the fp32 prototype path. These are prototype overheads, not optimized kernel overheads.

### Remaining full-scale runs before submission

1. Add Llama-2-7B full 128-sample utility if a second 7B-family utility point is desired; Qwen2.5-7B is now complete.
2. Scale PII from 500 to 1,000 prompts if the paper needs the upper end of the planned range; Qwen3 has 500 and Llama-3.2-1B has 120.
3. Run bf16 and fp32-cache performance variants on Qwen2.5-7B if deployment overhead must separate precision policy from wrapper overhead.
4. Run native profiling on Llama without `--skip-downstream` only after defining a non-square inversion metric; residual/KPA evidence is already cross-family.

## 2026-06-05: Claim Hardening Batch 2

This batch targets three previously weak claims: optimized serving overhead,
real-context privacy generalization, and architecture-independent attack metrics
for non-square value projections. PUF hardware remains user-owned future work.

### Scripts added or extended

| File | Purpose |
| --- | --- |
| `src/puf4secure_kvcache/puf_attention.py` | Added `fast_givens=True` path with vectorized/Triton pairwise Givens rotations. |
| `scripts/run_performance_eval.py` | Added `wrapped_fast` mode to compare plain, dense wrapped, and fast-Givens wrapped paths. |
| `scripts/run_pii_benchmark.py` | Added `--prompt-source real_context` using public real-text contexts with seeded, exactly scored secrets. |
| `scripts/run_candidate_secret_collision.py` | Added architecture-independent candidate-secret collision attack for finite secret spaces. |

### Artifacts

| Artifact | Scope |
| --- | --- |
| `experiments/runs/wave6_fast_givens_triton_sanity_qwen3_0p6b_fp32.json` | Triton fast-Givens sanity; Qwen3 logit diff and greedy match. |
| `experiments/runs/wave6_perf_qwen3_0p6b_dense_vs_triton_fast_fp32.json` | Qwen3 plain vs dense wrapped vs Triton fast wrapped performance. |
| `experiments/runs/wave7_real_context_pii_qwen3_0p6b_300_fp32.json` | 300-prompt real-context seeded PII benchmark from cached CC-News contexts. |
| `experiments/runs/wave8_candidate_collision_llama3p2_1b_30x32_fp32.json` | Llama-3.2-1B candidate-secret collision, 30 prompts x 32 candidates. |
| `experiments/runs/wave8_candidate_collision_qwen2p5_1p5b_30x32_fp32.json` | Qwen2.5-1.5B candidate-secret collision, 30 prompts x 32 candidates. |
| `experiments/runs/wave8_candidate_collision_qwen2p5_7b_20x16_fp32.json` | Qwen2.5-7B candidate-secret collision, 20 prompts x 16 candidates. |

### Key results

| Experiment | Plain / dense baseline | Fast / protected result | Interpretation |
| --- | ---: | ---: | --- |
| Qwen3 fast-Givens sanity | - | max logit diff 3.81e-05, 8/8 tokens match | Correctness holds. |
| Qwen3 perf, 64-token prefill | dense wrapped 27.85 tok/s | Triton fast 26.67 tok/s | Fast path is slower than dense in current prototype. |
| Qwen3 perf, 1024-token prefill | dense wrapped 27.81 tok/s | Triton fast 26.98 tok/s | Triton reduces elementwise overhead but still does not beat dense einsum. |
| Real-context Qwen3 PII, 300 prompts | 69/300 keyword leaks (23.0%) | 0/300 keyword leaks (0.0%) | Real-text contexts preserve non-trivial plaintext leakage; protection remains exact-zero. |
| Real-context Qwen3 digit leakage | 50/300 exact digit leaks (16.7%) | 0/300 exact digit leaks (0.0%) | Mean digit recall drops from 0.3120 to 0.0093. |
| Llama-3.2-1B candidate collision | top1 1.000, MRR 1.000 | top1 0.033, MRR 0.101 | Architecture-independent finite-secret matching is suppressed. |
| Qwen2.5-1.5B candidate collision | top1 1.000, MRR 1.000 | top1 0.033, MRR 0.110 | Same result on non-square Qwen2 model. |
| Qwen2.5-7B candidate collision | top1 1.000, MRR 1.000 | top1 0.000, MRR 0.089 | Same trend at 7B scale. |

### Interpretation

1. The optimized-overhead claim is not yet strengthened. Pairwise Givens is algebraically correct, but PyTorch/Triton standalone kernels do not beat dense `einsum` inside the current monkey-patched attention loop. A credible optimized-overhead claim needs rotation fused into the attention/QKV kernel rather than launched as separate kernels.
2. The real-context privacy claim is strengthened. Using cached CC-News contexts with seeded secrets, plaintext cache injection leaks exact keywords in 23.0% of Qwen3 prompts, while protected native caches leak 0.0%.
3. The non-square-architecture attack claim is strengthened. Candidate-secret collision does not depend on square `v_proj` and works on Llama and Qwen2 plaintext caches; protected native caches reduce top-1 recovery to near chance.

## 2026-06-05: Claim Hardening Batch 3 / Norm Side-Channel Audit

This batch continued the high-confidence security experiments and then audited a
basis-invariant attack missed by the raw-L2 candidate-collision metric.

### Scripts added or extended

| File | Purpose |
| --- | --- |
| `scripts/run_candidate_secret_collision.py` | Added `--secret-types` for six finite secret families, `--layers 0,mid,last`, by-type summaries, and `--distance-mode norm_l2` for basis-invariant norm matching. |

### Artifacts

| Artifact | Scope |
| --- | --- |
| `experiments/runs/wave9_real_context_pii_llama3p2_1b_300_fp32.json` | 300-prompt Llama-3.2-1B real-context seeded PII benchmark from cached CC-News contexts. |
| `experiments/runs/wave9_candidate_collision_secret_types_smoke_llama3p2_1b_fp32.json` | Multi-secret candidate-collision smoke test. |
| `experiments/runs/wave9_candidate_collision_secret_types_llama3p2_1b_60x32_layers0midlast_fp32.json` | Six-secret-family raw-L2 candidate-collision run on Llama-3.2-1B. |
| `experiments/runs/wave9_candidate_collision_norm_l2_llama3p2_1b_60x32_layers0midlast_fp32.json` | Norm-only candidate matching on Llama-3.2-1B. |
| `experiments/runs/wave9_candidate_collision_norm_l2_qwen2p5_1p5b_60x32_layers0midlast_fp32.json` | Norm-only candidate matching on Qwen2.5-1.5B. |
| `experiments/runs/wave9_candidate_collision_norm_l2_qwen3_0p6b_60x32_layers0midlast_fp32.json` | Norm-only candidate matching on Qwen3-0.6B. |

### Key results

| Experiment | Plain / baseline | Protected result | Interpretation |
| --- | ---: | ---: | --- |
| Real-context Llama PII, 300 prompts | 69/300 keyword leaks (23.0%) | 0/300 keyword leaks (0.0%) | Cross-model real-context injection result matches Qwen3. |
| Real-context Llama digit leakage | 45/300 exact digit leaks (15.0%) | 0/300 exact digit leaks (0.0%) | Mean digit recall drops from 0.2639 to 0.0322. |
| Multi-secret Llama collision, raw L2, layers 0/mid/last | top1 1.000, MRR 1.000 | top1 0.100, MRR 0.221 | Direction-space mismatch weakens the attack, but protected top1 is not zero. |
| Multi-secret Llama collision, norm L2 | top1 1.000, MRR 1.000 | top1 1.000, MRR 1.000 | Orthogonal-only protection fully preserves the finite-candidate norm signature. |
| Multi-secret Qwen2.5-1.5B collision, norm L2 | top1 1.000, MRR 1.000 | top1 1.000, MRR 1.000 | Norm side-channel is not Llama-specific. |
| Multi-secret Qwen3-0.6B collision, norm L2 | top1 1.000, MRR 1.000 | top1 1.000, MRR 1.000 | Norm side-channel also breaks the main Qwen3 model. |

### Interpretation

1. The real-context PII injection result is now cross-model: Qwen3 and Llama-3.2-1B both show plaintext leakage on CC-News seeded contexts and 0/300 protected keyword leakage.
2. The earlier candidate-collision result was incomplete. Raw vector L2 measures basis mismatch, but orthogonal transforms preserve per-token/head/layer norms exactly.
3. A cache-only attacker with a finite candidate set and known prompt template can compute plaintext candidate norm signatures and compare them directly to the protected cache. In the tested 60x32 six-secret-family setup, this recovers the secret with top1/MRR 1.0 on Qwen3, Llama, and Qwen2.5.
4. The paper must not claim broad finite-candidate cache confidentiality for the current orthogonal-only design. Either scope the claim to wrong-device replay/injection without candidate norm matching, or add and evaluate a norm-blinding mitigation.

## 2026-06-05: Claim Hardening Batch 4 / Norm Mitigation Prototype

Wave-9 found a real side channel, so this batch tested mitigation options before
touching the paper. The key distinction is that random scaling still leaves a
content-correlated norm, while unit-normalizing the exposed cache removes the
content norm from the dumped KV tensors. The unit-norm prototype keeps the
original norms in a private in-process sidecar and restores them inside attention.

### Scripts added or extended

| File | Purpose |
| --- | --- |
| `src/puf4secure_kvcache/puf_attention.py` | Added optional `norm_blind` diagonal scaling and `unit_norm_cache` sidecar modes. Defaults remain unchanged. |
| `scripts/run_wrapper_sanity.py` | Added `--norm-blind`, `--norm-log-range`, and `--unit-norm-cache` flags for correctness checks. |
| `scripts/run_candidate_secret_collision.py` | Added the same mitigation flags to attack protected native caches. |

### Artifacts

| Artifact | Scope |
| --- | --- |
| `experiments/runs/wave10_norm_blind_sanity_qwen3_0p6b_fp32.json` | Scalar random scaling sanity, log range 1. |
| `experiments/runs/wave10_candidate_collision_norm_l2_norm_blind_qwen3_0p6b_60x32_layers0midlast_fp32.json` | Scalar random scaling attack, log range 1. |
| `experiments/runs/wave10_candidate_collision_norm_l2_norm_blind_qwen3_0p6b_60x32_layers0midlast_log3_fp32.json` | Scalar random scaling attack, log range 3. |
| `experiments/runs/wave10_candidate_collision_norm_l2_norm_blind_qwen3_0p6b_60x32_layers0midlast_log6_fp32.json` | Scalar random scaling attack, log range 6. |
| `experiments/runs/wave10_diag_blind_sanity_qwen3_0p6b_log3_fp32.json` | Per-dimension diagonal random scaling sanity. |
| `experiments/runs/wave10_candidate_collision_norm_l2_diag_blind_qwen3_0p6b_60x32_layers0midlast_log3_fp32.json` | Per-dimension diagonal random scaling attack. |
| `experiments/runs/wave10_unit_norm_cache_sanity_qwen3_0p6b_fp32.json` | Unit-norm sidecar correctness sanity on Qwen3. |
| `experiments/runs/wave10_unit_norm_cache_sanity_llama3p2_1b_fp32.json` | Unit-norm sidecar correctness sanity on Llama. |
| `experiments/runs/wave10_candidate_collision_norm_l2_unit_norm_cache_qwen3_0p6b_60x32_layers0midlast_fp32.json` | Norm-L2 attack against unit-norm sidecar Qwen3 cache. |
| `experiments/runs/wave10_candidate_collision_norm_l2_unit_norm_cache_llama3p2_1b_60x32_layers0midlast_fp32.json` | Norm-L2 attack against unit-norm sidecar Llama cache. |

### Key results

| Experiment | Plain / baseline | Protected result | Interpretation |
| --- | ---: | ---: | --- |
| Scalar random scaling, log range 1 | norm-L2 plain top1 1.000 | protected top1 0.683, MRR 0.789 | Insufficient. |
| Scalar random scaling, log range 3 | norm-L2 plain top1 1.000 | protected top1 0.183, MRR 0.343 | Better but still above chance. |
| Scalar random scaling, log range 6 | norm-L2 plain top1 1.000 | protected top1 0.117, MRR 0.262 | Still residual ranking signal. |
| Per-dimension diagonal scaling, log range 3 | norm-L2 plain top1 1.000 | protected top1 0.250, MRR 0.391 | Random weighted norms remain content-correlated. |
| Unit-norm sidecar Qwen3 sanity | - | max logit diff 3.24e-05, 8/8 tokens match on 2 prompts | Correctness holds in fp32 sanity. |
| Unit-norm sidecar Llama sanity | - | max logit diff 2.77e-05, 8/8 tokens match on 2 prompts | Cross-family sanity holds. |
| Unit-norm sidecar Qwen3 norm-L2 attack | plain top1 1.000, MRR 1.000 | protected top1 0.000, MRR 0.083 | Blocks the tested norm-only candidate attack. |
| Unit-norm sidecar Llama norm-L2 attack | plain top1 1.000, MRR 1.000 | protected top1 0.033, MRR 0.130 | Near chance for 32 candidates. |

### Interpretation

1. Randomly scaling exposed cache vectors is not sufficient because the resulting norm remains correlated with the content norm.
2. Unit-normalizing the exposed cache removes the content norm from the dumped KV tensors and restores norms only inside the legitimate process. This directly addresses the Wave-9 invariant attack under a cache-only threat model.
3. The unit-norm sidecar is a prototype, not a final paper claim. It introduces private sidecar state; the paper must model whether this sidecar is protected, migrated, dumped, or recomputable.
4. If this route is adopted, next experiments should cover stronger sidecar-aware candidate attacks, utility beyond sanity, decode-length stress, and performance overhead.

## 2026-06-05: Claim Hardening Batch 5 / No-Sidecar Mitigation Attempts

The sidecar mitigation raised a valid systems objection: if there is already a
trusted space for sidecar norms, why not store the KV cache there? This batch
therefore tested PUF-derived transforms that require no additional trusted
content storage beyond the PUF root/session derivation.

### Scripts added or extended

| File | Purpose |
| --- | --- |
| `src/puf4secure_kvcache/puf_attention.py` | Added optional `nonorth_log_range` for bounded `O D` transforms and optional `affine_mask` for PUF-derived additive cache masks. Defaults remain unchanged. |
| `scripts/run_wrapper_sanity.py` | Added `--nonorth-log-range`, `--affine-mask`, and `--mask-std` flags. |
| `scripts/run_candidate_secret_collision.py` | Added the same flags for protected native cache attacks. |

### Artifacts

| Artifact | Scope |
| --- | --- |
| `experiments/runs/wave11_nonorth_sanity_qwen3_0p6b_log{1,3,6}_fp32.json` | Qwen3 fp32 sanity for bounded non-orthogonal `O D` transforms. |
| `experiments/runs/wave11_candidate_collision_norm_l2_nonorth_qwen3_0p6b_60x32_layers0midlast_log{1,3,6}_fp32.json` | Norm-L2 candidate attacks against bounded non-orthogonal transforms. |
| `experiments/runs/wave11_affine_mask_sanity_qwen3_0p6b_std{4,32,128,256}_fp32.json` | Qwen3 fp32 sanity for PUF-derived affine mask strengths. |
| `experiments/runs/wave11_candidate_collision_norm_l2_affine_mask_qwen3_0p6b_60x32_layers0midlast_std{4,32,128,256}_fp32.json` | Qwen3 norm-L2 candidate attacks against affine mask strengths. |
| `experiments/runs/wave11_candidate_collision_raw_l2_affine_mask_qwen3_0p6b_60x32_layers0midlast_std128_fp32.json` | Qwen3 raw-L2 candidate attack against affine mask std128. |
| `experiments/runs/wave11_affine_mask_sanity_llama3p2_1b_std128_fp32.json` | Llama fp32 sanity for affine mask std128. |
| `experiments/runs/wave11_candidate_collision_{norm_l2,raw_l2}_affine_mask_llama3p2_1b_60x32_layers0midlast_std128_fp32.json` | Llama candidate attacks against affine mask std128. |

### Key results

| Experiment | Plain / baseline | Protected result | Interpretation |
| --- | ---: | ---: | --- |
| Non-orthogonal `O D`, Qwen3 log1 sanity | - | max diff 3.10e-05, 8/8 tokens match | Correctness holds. |
| Non-orthogonal `O D`, Qwen3 log3 sanity | - | max diff 3.15e-05, 8/8 tokens match | Correctness holds. |
| Non-orthogonal `O D`, Qwen3 log6 sanity | - | max diff 3.53e-05, 8/8 tokens match | Correctness holds. |
| Norm-L2 attack, non-orthogonal log1 | plain top1 1.000 | protected top1 0.900, MRR 0.947 | Negative. |
| Norm-L2 attack, non-orthogonal log3 | plain top1 1.000 | protected top1 0.350, MRR 0.509 | Still too high. |
| Norm-L2 attack, non-orthogonal log6 | plain top1 1.000 | protected top1 0.267, MRR 0.420 | Still too high despite stronger conditioning. |
| Affine mask Qwen3 std128 sanity | - | max diff 4.90e-04, 8/8 tokens match | Correct in fp32 sanity, but numerical error is higher. |
| Affine mask Llama std128 sanity | - | max diff 3.49e-04, 8/8 tokens match | Cross-family sanity holds. |
| Norm-L2 attack, affine mask Qwen3 std128 | plain top1 1.000 | protected top1 0.050, MRR 0.145 | Near chance for 32 candidates. |
| Raw-L2 attack, affine mask Qwen3 std128 | plain top1 1.000 | protected top1 0.083, MRR 0.179 | Strong reduction, slightly above chance. |
| Norm-L2 attack, affine mask Llama std128 | plain top1 1.000 | protected top1 0.000, MRR 0.098 | Near/below chance. |
| Raw-L2 attack, affine mask Llama std128 | plain top1 1.000 | protected top1 0.000, MRR 0.092 | Near/below chance. |

### Interpretation

1. Bounded non-orthogonal linear transforms are not enough. Even though `K'=KOD`, `Q'=QOD^{-1}`, `V'=VOD`, and output recovery are algebraically valid, the exposed norms remain sufficiently correlated with candidate plaintext norms.
2. PUF-derived affine masking is the first no-sidecar prototype that substantially suppresses both norm-L2 and raw-L2 finite-candidate matching in this setup. It stores `rotated_KV + mask(session, layer, head, position)` in the cache and regenerates/subtracts the mask inside legitimate attention.
3. This avoids storing secret sidecar data, but it changes the system story: the legitimate path now uncloaks the cache inside attention rather than computing purely over masked coordinates.
4. The std128 setting is the current best tradeoff among tested values. Std32 leaves residual norm signal; std256 does not improve Qwen3 norm-L2 over std128 and increases numerical error.
5. Before paper integration, affine masking needs utility, long-decode, performance, and stronger attacker audits. It should be described as a promising redesign candidate, not a final validated defense.

## 2026-06-09: Review-Driven Hardening Waves 12-13 (autonomous)

This batch implemented the seven review insights raised after the Batch-5 review:
reposition the contribution to physical non-migratability, generalize the
norm side channel to the full basis-invariant class, prove affine masking
removes that class, add a migration head-to-head, a security/precision sweep,
a fused-kernel (SDPA) compatibility PoC, longer-horizon performance, MMLU, and
an O(n) affine-mask implementation.

### Code added or extended

| File | Purpose |
| --- | --- |
| `src/puf4secure_kvcache/puf_attention.py` | Replaced O(n^2) per-position affine mask with `_affine_mask_cached` (per-(layer,purpose) continued generator + incremental position cache, O(n) over a decode). Added `_sdpa_attention` and an `attn_backend="sdpa"` option routing the orthogonal path through `F.scaled_dot_product_attention`. |
| `scripts/run_candidate_secret_collision.py` | Added `gram_l2` (Gram matrix `XX^T`) and `svd_l2` (singular spectrum) distance modes -- the strictly-stronger basis-invariant attacks. |
| `scripts/run_utility_eval.py` | Added a standard MMLU evaluator (`--mmlu-samples`) and affine-mask flags. |
| `scripts/run_performance_eval.py`, `run_decode_divergence_benchmark.py` | Added affine-mask flags. |
| `scripts/run_migration_comparison.py` | New: KV-Cloak (software key) vs PUF-Cache (physical root) under cache+software migration, scored by layer-0 V-inversion top-1. |
| `scripts/run_wrapper_sanity.py` | Added `--attn-backend sdpa`. |

### Key results

| Experiment | Result | Interpretation |
| --- | --- | --- |
| Affine mask O(n) fix | L1024 affine perf 6h hang -> 12s; correctness preserved (max logit diff 3.9e-4, token match 1.0) | Naive per-position re-derivation was O(n^2); cached generation is O(n). |
| Invariant attacks, orthogonal | Gram top1 1.000 and SVD-spectrum top1 1.000 on Qwen3-0.6B and Llama-3.2-1B | Orthogonal caches leak the WHOLE right-invariant class, not just norms. |
| Invariant attacks, affine std128 | Qwen3 gram/svd top1 0.033/0.033; Llama 0.050/0.067 | Affine mask removes the whole class to ~chance (1/32=0.031). |
| Std sweep (Qwen3, 40x32) | gram top1 0.975 (std4) -> 0.125 (std16) -> 0.050 (std>=64); fp32 err 3.4e-5 -> 1.0e-3 | Security plateaus near chance by std~64; std128 conservative knee; affine is fp32-only. |
| Migration head-to-head (Qwen3) | plain 1.000; cloak key-withheld 0.000; cloak key-migrated 1.000; PUF wrong-device 0.000 (V relL2 1.44); PUF legit 1.000 | Software-key obfuscation breaks on key migration; PUF-Cache stays bound with full software stack. |
| Affine utility (Qwen3 fp32) | PPL delta +2.85e-5; HellaSwag/MMLU delta 0; long-decode 3/3 exact; 32x32 divergence 0% | Affine path is fp32-exact -- closes the previous "needs utility/long-decode" gap. |
| MMLU orthogonal (256) | plain 0.3828 == wrapped 0.3828 | Standard benchmark beyond AG News/HellaSwag; exact equivalence. |
| SDPA backend (orthogonal) | max logit diff 3.96e-5, token match 1.0 | Orthogonal rotations compose with the fused-kernel attention API. |
| Perf long-horizon (decode128 x8) | Qwen3 ortho -14.5..-15.1%; Qwen2.5-7B ortho -17.6..-18.3%; Qwen3 affine -31.9..-33.2% | Prefill-independent decode; robust (std<0.6 tok/s). Old short-decode estimate understated overhead. |

### Interpretation

1. The orthogonal cache leaks every right-orthogonal-invariant (norm = Gram
   diagonal, full Gram, singular spectrum), all at top1 1.000. This generalizes
   the Wave-9 norm finding into a class result and is the paper's sharpened
   negative result.
2. Affine masking removes the entire class to near chance AND is now
   utility-validated (fp32-exact on PPL/HellaSwag/MMLU/long-decode) with measured
   O(n) overhead (~2x the orthogonal path). Its remaining gaps are fp16
   robustness and a fused kernel.
3. The migration experiment isolates the actual PUF contribution: physical
   non-migratability with no extractable key, not "no decrypt-before-attend"
   (affine masking deliberately decrypts inside attention).
4. The orthogonal path composes with `scaled_dot_product_attention`; the
   affine path needs a kernel-level de-masking hook for paged/flash.
5. Paper updated accordingly: abstract/intro repositioned to non-migratability;
   sec3 generalizes the invariant class and adds the decrypt-before-attend
   honesty; sec4 adds RQ2b (migration), the invariant-class audit table, the
   std-sweep table, MMLU, and long-horizon perf; sec5 adds fused-kernel
   compatibility and scope subsections. References filled and verified via
   Google Scholar. `latexmk` builds clean (19 pages, 9 bibitems).

## 2026-06-09: Remaining-Work Wave 14 (autonomous)

Attempted the residual future-work items from the Wave 12-13 wrap-up.

| Item | Result | Interpretation |
| --- | --- | --- |
| F1 affine + SDPA | max logit diff 4.0e-4, token match 1.0 | The affine de-mask is a cheap elementwise op before the fused kernel, so the affine path already composes with `scaled_dot_product_attention` (a pre-attention de-masking hook). Closes the "affine needs kernel hook" question. |
| F2 fp16 precision | fp16 divergence 15.6% (vs bf16 56.25%, fp32 0%) | fp16 (10 mantissa bits) is ~3.6x more stable than bf16 (7 bits). Since FlashAttention uses fp16, low-precision deployment is more viable than the bf16 result implied. |
| F3 long-context utility | PG-19 @ 4096 tokens: plain vs wrapped PPL delta +1.4e-6 | Wrapper equivalence does NOT degrade with sequence length; rotation error does not accumulate over long real documents. |
| F4 Enron real corpus | plain 14/200 (7.0%) keyword leaks, protected 0/200 | Real Enron emails: plaintext still leaks, protected exact-zero. Lower plain rate than CC-News (23%) because email text is a noisier injection setting. Closes "not real data" partially (secrets still seeded). |
| F5 real PUF hardware | not attempted | Requires a physical FPGA PUF stream + fuzzy-extractor calibration; no hardware available. Documented as a genuine hardware dependency in Limitations, not faked. |

### Code added or extended

| File | Purpose |
| --- | --- |
| `scripts/run_decode_divergence_benchmark.py` | Added `--cast-dtype {none,fp16,bf16}` for the fp16 precision study. |
| (existing) `run_pii_benchmark.py` `--context-dataset` | Reused to point the real-context PII benchmark at the Enron corpus. |
| (existing) `run_utility_eval.py` perplexity | Reused at `--ppl-max-length 4096` on PG-19 for long-context utility. |

Paper updated: RQ3 gains the long-context PG-19 result and the fp16 nuance;
the real-context PII table adds an Enron row; the discussion's fused-kernel
subsection now reports the verified affine+SDPA composition and the fp16
stability. `latexmk` clean (19 pages). Genuinely remaining: real PUF hardware,
a fused affine-mask cache-load kernel, and secrets mined from real private
records rather than seeded.

## 2026-06-17: P4b split-K fused kernel + RQ12 + fp16 fused + Llama-2-7B utility (autonomous)

Completed the remaining non-hardware engineering items from the FUSED_KERNEL_PLAN
P4b/P5 scope, plus the fp16-fused numerics and the optional Llama-2-7B utility point.

### Code added or extended

| File | Purpose |
| --- | --- |
| `src/puf4secure_kvcache/fused_attn.py` | Added split-K (flash-decoding) variant: `_get_splitk_kernel` (grid=(B,Hq,n_splits), partial (m_i,l_i,acc) per chunk) + `_get_reduce_kernel` (log-sum-exp combine). `_plan_splits` auto-tunes n_splits≤8 for one-SM-wave occupancy. `fused_demask_decode` dispatches: n_splits=1 → original single-program kernel; n_splits>1 → split-K + reduce. |
| `scripts/test_fused_kernel.py` | Added `check_splitk` validating the split-K kernel against SDPA across n_splits∈{2,4,8,auto} and all shape cases including N=1024 (the P4 failure regime). |
| `paper_latex/sections/sec4_evaluation.tex` | New `\subsection{RQ12: Fused de-masking kernel}` with Table tab:fused-kernel. Llama-2-7B row added to tab:utility-multimodel. RQ3 answer updated with 5 models + fp16 fused number. Open Gaps updated. |
| `paper_latex/sections/sec3_methodology.tex` | sec:design affine-kernel sentence updated from "resists naive fusion / future work" to "we implement this as a fused Triton flash-decoding kernel". |
| `paper_latex/sections/sec5_conclusion.tex` | Discussion "Compatibility with fused attention kernels" rewritten to report the implemented split-K kernel + fp16 14.1% number. Limitations #2 and #5 updated. Conclusion's "doubles decode overhead" → precise "$\approx$14pp over orthogonal". |

### Artifacts

| Artifact | Scope |
| --- | --- |
| `experiments/runs/p4_perf_orth.json` | plain + orthogonal, prefill 256/1024, decode 128, 8 reps (GPU 2 clean) |
| `experiments/runs/p4_perf_affine_eager.json` | affine eager de-mask, same operating point |
| `experiments/runs/p4_perf_affine_fused.json` | affine split-K fused kernel, same operating point |
| `experiments/runs/p3_divergence_{eager,fused}_{qwen3,llama}.json` | 64-prompt/64-token divergence: fused == eager == 0.0 (fp32) |
| `experiments/runs/f2_fp16_affine_{eager,fused}_qwen3.json` | fp16 model + fp32 cache + affine: eager 20.3%, fused 14.1% divergence |
| `experiments/runs/wave1_utility_llama2_7b_fp32_128.json` | Llama-2-7B 128-sample AG News + HellaSwag, fp32 |

### Key results

| Experiment | Before | After | Interpretation |
| --- | --- | --- | --- |
| Fused kernel N=1024 decode overhead | naive −68.5% (10.64 tok/s) | split-K −30.2% (23.76 tok/s) | Split-K fixes the catastrophic long-context regression (+123% throughput) |
| Fused vs eager decode | eager −33% at all N | fused −29% at all N | Fused is consistently 5-7% faster; no long-context degradation |
| Fused kernel correctness | (new) | max diff 2.7e-6 vs SDPA, token-identical Qwen3+Llama | Split-K + log-sum-exp reduce is numerically equivalent to single-program |
| fp16 model divergence | eager affine 20.3%, orth 15.6% | fused affine 14.1% | Fused is MORE stable in fp16 (entire de-mask stays fp32 inside kernel) |
| Llama-2-7B utility | (not measured) | PPL Δ+1.0e-6, HellaSwag Δ0.0 | 5th model confirms fp32-exact equivalence |

### Interpretation

1. The split-K flash-decoding kernel is the correct fix for the naive
   single-program-per-head design: it lifts SM occupancy from 16 to 128+ programs
   and eliminates the serial-over-N bottleneck that caused the −68%@1024 collapse.
   The counter-based Philox mask is position-addressable, so each split regenerates
   exactly the mask rows the write path produced — no HBM-stored mask needed.
2. The fused kernel is now numerically validated (1e-6 vs SDPA, token-identical on
   real models) AND performance-competitive: ~29% decode overhead vs ~33% eager,
   stable across context lengths. The residual ~14pp gap to orthogonal (−15%) is
   the inherent fp32 de-mask ALU cost.
3. Under fp16 model weights, the fused path diverges LESS than the orthogonal path
   (14.1% vs 15.6%), because the fused kernel performs the entire de-mask and
   attention in fp32, avoiding the intermediate fp16 rounding the eager/PyTorch
   path inherits.
4. Llama-2-7B extends the cross-model fp32-exact utility story to 5 models across
   3 families (Qwen3, Qwen2, Llama) and 2 scales (0.6B-7B).
5. Paper now reports RQ12 (fused kernel) as a completed result rather than future
   work; sec3/sec5/discussion/limitations all updated. `latexmk` clean (29 pages).
