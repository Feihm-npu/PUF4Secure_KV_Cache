# Results Table

## Wave 0: Cross-Model Wrapper Sanity

| Model | Family | Hidden | Max logit diff | Greedy match |
| --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B | qwen3 | 1024 | 3.77e-05 | 1.000 |
| Llama-3.2-1B | llama | 2048 | 1.60e-05 | 1.000 |
| Qwen2.5-1.5B-Instruct | qwen2 | 1536 | 7.77e-05 | 1.000 |
| Llama-2-7B-chat | llama | 4096 | 2.34e-05 | 1.000 |
| Qwen2.5-7B-Instruct | qwen2 | 3584 | 1.54e-04 | 1.000 |

## Wave 1: Utility

| Model | Metric | Plain | Wrapped | Delta |
| --- | --- | ---: | ---: | ---: |
| Llama-3.2-1B | AG News PPL, 128 | 22.4110846 | 22.4110857 | +0.0000011 |
| Llama-3.2-1B | HellaSwag acc., 128 | 0.4141 | 0.4141 | 0.0000 |
| Qwen2.5-1.5B | AG News PPL, 128 | 19.4575599 | 19.4575587 | -0.0000012 |
| Qwen2.5-1.5B | HellaSwag acc., 128 | 0.4531 | 0.4531 | 0.0000 |
| Qwen2.5-7B | AG News PPL, 128 | 15.1818894 | 15.1818911 | +0.0000017 |
| Qwen2.5-7B | HellaSwag acc., 128 | 0.4844 | 0.4844 | 0.0000 |

## Wave 2: PII and Precision

| Experiment | Plain | Protected / Wrapped | Artifact |
| --- | ---: | ---: | --- |
| PII keyword leaks, 60 prompts | 30/60 = 0.500 | 0/60 = 0.000 | `wave2_pii_qwen3_0p6b_60_fp32.json` |
| PII exact digit leaks, 60 prompts | 23/60 = 0.383 | 0/60 = 0.000 | `wave2_pii_qwen3_0p6b_60_fp32.json` |
| Qwen3 PII keyword leaks, 500 prompts | 261/500 = 0.522 | 0/500 = 0.000 | `wave2_pii_qwen3_0p6b_500_fp32.json` |
| Qwen3 PII exact digit leaks, 500 prompts | 190/500 = 0.380 | 0/500 = 0.000 | `wave2_pii_qwen3_0p6b_500_fp32.json` |
| Llama-3.2-1B PII keyword leaks, 120 prompts | 67/120 = 0.558 | 0/120 = 0.000 | `wave2_pii_llama3p2_1b_120_fp32.json` |
| Llama-3.2-1B PII exact digit leaks, 120 prompts | 35/120 = 0.292 | 0/120 = 0.000 | `wave2_pii_llama3p2_1b_120_fp32.json` |
| Decode divergence, bf16 32x32 | n/a | 0.5625 | `wave2_decode_divergence_qwen3_0p6b_bf16_32x32.json` |
| Decode divergence, fp32 32x32 | n/a | 0.0000 | `wave2_decode_divergence_qwen3_0p6b_fp32_32x32.json` |

## Wave 3: Adaptive Profiling

| Experiment | Same-session | Session refresh | Artifact |
| --- | ---: | ---: | --- |
| KPA 4 known prompts residual | 0.0519 | 1.4166 | `wave3_kpa_native_qwen3_0p6b_4_16_64.json` |
| KPA 16 known prompts residual | 4.21e-05 | 1.4167 | `wave3_kpa_native_qwen3_0p6b_4_16_64.json` |
| KPA 64 known prompts residual | 3.04e-05 | 1.4177 | `wave3_kpa_native_qwen3_0p6b_4_16_64.json` |
| Llama KPA 4 known prompts residual | 7.31e-06 | 1.4154 | `wave3_kpa_native_llama3p2_1b_4_16.json` |
| Llama KPA 16 known prompts residual | 3.87e-06 | 1.4165 | `wave3_kpa_native_llama3p2_1b_4_16.json` |
| Native Hungarian V-inv | 1.000 | 0.000 | `wave3_native_profiling_qwen3_0p6b_hungarian_64.json` |

## Wave 4: Performance

| Prompt len | Plain decode tok/s | Wrapped decode tok/s | Throughput delta | KV memory |
| ---: | ---: | ---: | ---: | --- |
| 64 | 29.40 | 27.16 | -7.6% | unchanged |
| 256 | 29.88 | 25.87 | -13.4% | unchanged |
| Qwen2.5-7B, 64 | 34.51 | 30.30 | -12.2% | unchanged |
| Qwen2.5-7B, 256 | 34.49 | 31.13 | -9.7% | unchanged |

## Wave 6: Fast Givens Performance Hardening

| Model / prefill | Dense wrapped tok/s | Fast-Givens tok/s | Result |
| --- | ---: | ---: | --- |
| Qwen3-0.6B, 64 | 27.85 | 26.67 | Fast path slower. |
| Qwen3-0.6B, 1024 | 27.81 | 26.98 | Fast path slower. |

## Wave 7: Real-Context Seeded PII

| Dataset / model | Plain keyword leaks | Protected keyword leaks | Plain digit leaks | Protected digit leaks |
| --- | ---: | ---: | ---: | ---: |
| CC-News contexts / Qwen3-0.6B, 300 prompts | 69/300 = 0.230 | 0/300 = 0.000 | 50/300 = 0.167 | 0/300 = 0.000 |

## Wave 8: Candidate-Secret Collision

| Model | Candidates | Plain top1 | Protected top1 | Plain MRR | Protected MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-1B | 32 | 1.000 | 0.033 | 1.000 | 0.101 |
| Qwen2.5-1.5B | 32 | 1.000 | 0.033 | 1.000 | 0.110 |
| Qwen2.5-7B | 16 | 1.000 | 0.000 | 1.000 | 0.089 |

## Wave 9: Real-Context Cross-Model and Norm-Side-Channel Audit

| Dataset / model | Plain keyword leaks | Protected keyword leaks | Plain digit leaks | Protected digit leaks |
| --- | ---: | ---: | ---: | ---: |
| CC-News contexts / Llama-3.2-1B, 300 prompts | 69/300 = 0.230 | 0/300 = 0.000 | 45/300 = 0.150 | 0/300 = 0.000 |

| Attack / model | Distance | Candidates | Plain top1 | Protected top1 | Plain MRR | Protected MRR | Claim impact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Multi-secret collision / Llama-3.2-1B | raw L2, layers 0/mid/last | 32 | 1.000 | 0.100 | 1.000 | 0.221 | Direction-space matching is weakened but not zero. |
| Multi-secret collision / Llama-3.2-1B | norm L2, layers 0/mid/last | 32 | 1.000 | 1.000 | 1.000 | 1.000 | Orthogonal-only cache fully leaks finite candidates through norm signatures. |
| Multi-secret collision / Qwen2.5-1.5B | norm L2, layers 0/mid/last | 32 | 1.000 | 1.000 | 1.000 | 1.000 | Same norm-side-channel failure on Qwen2. |
| Multi-secret collision / Qwen3-0.6B | norm L2, layers 0/mid/last | 32 | 1.000 | 1.000 | 1.000 | 1.000 | Same norm-side-channel failure on the main Qwen3 model. |

## Wave 10: Norm-Side-Channel Mitigation Prototype

| Mitigation / model | Correctness max logit diff | Greedy match | Artifact |
| --- | ---: | ---: | --- |
| Unit-norm cache sidecar / Qwen3-0.6B | 3.24e-05 | 1.000 | `wave10_unit_norm_cache_sanity_qwen3_0p6b_fp32.json` |
| Unit-norm cache sidecar / Llama-3.2-1B | 2.77e-05 | 1.000 | `wave10_unit_norm_cache_sanity_llama3p2_1b_fp32.json` |

| Attack / model | Cache variant | Plain top1 | Protected top1 | Plain MRR | Protected MRR | Claim impact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Norm-L2 collision / Qwen3-0.6B | Orthogonal-only | 1.000 | 1.000 | 1.000 | 1.000 | Broken by norm signatures. |
| Norm-L2 collision / Qwen3-0.6B | Unit-norm sidecar | 1.000 | 0.000 | 1.000 | 0.083 | Prototype blocks this norm-only attack. |
| Norm-L2 collision / Llama-3.2-1B | Orthogonal-only | 1.000 | 1.000 | 1.000 | 1.000 | Broken by norm signatures. |
| Norm-L2 collision / Llama-3.2-1B | Unit-norm sidecar | 1.000 | 0.033 | 1.000 | 0.130 | Prototype is near chance for 32 candidates. |

Intermediate random-scaling variants were insufficient: scalar blinding with log range 3/6 left Qwen3 protected top1 at 0.183/0.117, and per-dimension diagonal blinding with log range 3 left protected top1 at 0.250.

## Wave 11: No-Sidecar Transform Attempts

| Transform / model | Parameter | Correctness max logit diff | Greedy match | Artifact |
| --- | ---: | ---: | ---: | --- |
| Bounded non-orthogonal `O D` / Qwen3-0.6B | log range 1 | 3.10e-05 | 1.000 | `wave11_nonorth_sanity_qwen3_0p6b_log1_fp32.json` |
| Bounded non-orthogonal `O D` / Qwen3-0.6B | log range 3 | 3.15e-05 | 1.000 | `wave11_nonorth_sanity_qwen3_0p6b_log3_fp32.json` |
| Bounded non-orthogonal `O D` / Qwen3-0.6B | log range 6 | 3.53e-05 | 1.000 | `wave11_nonorth_sanity_qwen3_0p6b_log6_fp32.json` |
| PUF affine mask / Qwen3-0.6B | std 128 | 4.90e-04 | 1.000 | `wave11_affine_mask_sanity_qwen3_0p6b_std128_fp32.json` |
| PUF affine mask / Llama-3.2-1B | std 128 | 3.49e-04 | 1.000 | `wave11_affine_mask_sanity_llama3p2_1b_std128_fp32.json` |

| Attack / model | Cache variant | Protected top1 | Protected MRR | Interpretation |
| --- | --- | ---: | ---: | --- |
| Norm-L2 / Qwen3-0.6B | non-orthogonal `O D`, log1 | 0.900 | 0.947 | Negative; bounded nonorth remains norm-correlated. |
| Norm-L2 / Qwen3-0.6B | non-orthogonal `O D`, log3 | 0.350 | 0.509 | Improved but not near chance. |
| Norm-L2 / Qwen3-0.6B | non-orthogonal `O D`, log6 | 0.267 | 0.420 | Still too high. |
| Norm-L2 / Qwen3-0.6B | affine mask std32 | 0.117 | 0.239 | Improved, but residual signal remains. |
| Norm-L2 / Qwen3-0.6B | affine mask std128 | 0.050 | 0.145 | Near chance for 32 candidates; no sidecar. |
| Norm-L2 / Qwen3-0.6B | affine mask std256 | 0.050 | 0.145 | No clear security gain over std128; larger numerical error. |
| Raw-L2 / Qwen3-0.6B | affine mask std128 | 0.083 | 0.179 | Strong reduction, but slightly above chance. |
| Norm-L2 / Llama-3.2-1B | affine mask std128 | 0.000 | 0.098 | Near/below chance. |
| Raw-L2 / Llama-3.2-1B | affine mask std128 | 0.000 | 0.092 | Near/below chance. |

The affine-mask path uses no sidecar: the exported cache stores `rotated_KV + mask`, and the legitimate process regenerates the PUF-derived mask from `(session, layer, head, position)` and subtracts it inside attention. This changes the earlier no-decloak story and needs overhead/numerical audit.
