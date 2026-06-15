# Experiment Tracker

| Run ID | Wave | Claim | Status | Command / Artifact | Notes |
| --- | --- | --- | --- | --- | --- |
| wave0-plan | 0 | C1 | DONE | `experiment-stage/EXPERIMENT_PLAN.md` | Plan initialized. |
| wave0-wrapper | 0 | C1 | DONE | `src/puf4secure_kvcache/puf_attention.py` | Added qwen3/qwen2/llama wrapper support. |
| wave0-sanity-qwen3 | 0 | C1 | DONE | `experiments/runs/wave0_wrapper_sanity_qwen3_0p6b_fp32.json` | max diff 3.77e-05, tokens match. |
| wave0-sanity-llama | 0 | C1 | DONE | `wave0_wrapper_sanity_llama3p2_1b_fp32.json`, `wave0_wrapper_sanity_llama2_7b_fp32.json` | 1B and 7B tokens match. |
| wave0-sanity-qwen2 | 0 | C1 | DONE | `wave0_wrapper_sanity_qwen2p5_1p5b_fp32.json`, `wave0_wrapper_sanity_qwen2p5_7b_fp32.json` | 1.5B and 7B tokens match. |
| wave1-utility-smoke | 1 | C1 | DONE | `wave1_utility_smoke_llama3p2_1b_fp32.json`, `wave1_utility_smoke_qwen2p5_1p5b_fp32.json` | Cross-family utility pipeline works. |
| wave1-utility-llama1b | 1 | C1 | DONE | `wave1_utility_llama3p2_1b_fp32_128.json` | PPL delta +1.08e-06, HellaSwag delta 0. |
| wave1-utility-qwen2-1p5b | 1 | C1 | DONE | `wave1_utility_qwen2p5_1p5b_fp32_128.json` | PPL delta -1.18e-06, HellaSwag delta 0. |
| wave1-utility-qwen2-7b | 1 | C1 | DONE | `wave1_utility_qwen2p5_7b_fp32_128.json` | PPL delta +1.68e-06, HellaSwag delta 0. |
| wave2-pii-60 | 2 | C2 | DONE | `experiments/runs/wave2_pii_qwen3_0p6b_60_fp32.json` | Plain 30/60 leaks; protected 0/60. |
| wave2-pii-500 | 2 | C2 | DONE | `experiments/runs/wave2_pii_qwen3_0p6b_500_fp32.json` | Plain 261/500 leaks; protected 0/500. |
| wave2-pii-llama1b | 2 | C2 | DONE | `experiments/runs/wave2_pii_llama3p2_1b_120_fp32.json` | Plain 67/120 leaks; protected 0/120. |
| wave2-divergence-32 | 2 | C1 | DONE | `wave2_decode_divergence_qwen3_0p6b_{bf16,fp32}_32x32.json` | bf16 diverges 56.25%; fp32 0%. |
| wave3-kpa | 3 | C3 | DONE | `wave3_kpa_native_qwen3_0p6b_4_16_64.json` | Same-session KPA succeeds; refresh fails. |
| wave3-kpa-llama1b | 3 | C3 | DONE | `wave3_kpa_native_llama3p2_1b_4_16.json` | Same-session KPA succeeds; refresh fails. |
| wave3-native-hungarian | 3 | C3 | DONE | `wave3_native_profiling_qwen3_0p6b_hungarian_64.json` | Same-session V-inv 1.0; refresh 0.0. |
| wave4-performance | 4 | C4 | DONE | `wave4_perf_qwen3_0p6b_fp32_64_256.json` | Decode throughput cost 7.6-13.4%; KV memory unchanged. |
| wave4-performance-7b | 4 | C4 | DONE | `wave4_perf_qwen2p5_7b_fp32_64_256.json` | Decode throughput cost 9.7-12.2%; KV memory unchanged. |
| wave1-full-multimodel | 1 | C1/C2 | PARTIAL | see utility rows | Qwen3, Llama-1B, Qwen2.5-1.5B, Qwen2.5-7B complete; Llama-2-7B optional. |
| wave6-fast-givens | 6 | C4 | NEGATIVE | `wave6_perf_qwen3_0p6b_dense_vs_triton_fast_fp32.json` | Correct but slower than dense; optimized overhead remains unresolved. |
| wave7-real-context-pii | 7 | C2 | DONE | `wave7_real_context_pii_qwen3_0p6b_300_fp32.json` | Plain 69/300 leaks; protected 0/300. |
| wave8-candidate-collision | 8 | C2 | SUPERSEDED | `wave8_candidate_collision_*.json` | Raw-L2 direction-space collision is suppressed, but Wave9 norm-L2 attack breaks broad finite-candidate confidentiality. |
| wave9-real-context-pii-llama | 9 | C2 | DONE | `wave9_real_context_pii_llama3p2_1b_300_fp32.json` | CC-News contexts: plain 69/300 keyword leaks, protected 0/300. |
| wave9-multisecret-l2 | 9 | C2 | WARN | `wave9_candidate_collision_secret_types_llama3p2_1b_60x32_layers0midlast_fp32.json` | Six secret types, 32 candidates: raw-L2 protected top1 0.100, MRR 0.221. |
| wave9-norm-side-channel | 9 | C2 | FAIL | `wave9_candidate_collision_norm_l2_{qwen3_0p6b,llama3p2_1b,qwen2p5_1p5b}_60x32_layers0midlast_fp32.json` | Norm-only candidate matching recovers protected secrets with top1/MRR 1.0 on all three models. |
| wave10-random-norm-blinding | 10 | C2 | NEGATIVE | `wave10_candidate_collision_norm_l2_{norm_blind,diag_blind}_qwen3_*.json` | Random scalar/diagonal scaling reduces but does not eliminate norm-ranking leakage. |
| wave10-unit-norm-cache | 10 | C2 | PROTOTYPE | `wave10_unit_norm_cache_sanity_*.json`, `wave10_candidate_collision_norm_l2_unit_norm_cache_*.json` | Unit-normalized exposed cache plus private norm sidecar keeps fp32 outputs equivalent and blocks norm-L2 on Qwen3/Llama; needs broader audit. |
| wave11-nonorth-linear | 11 | C2 | NEGATIVE | `wave11_nonorth_*`, `wave11_candidate_collision_norm_l2_nonorth_*` | Bounded non-orthogonal `O D` needs no sidecar and preserves fp32 sanity, but norm-L2 protected top1 remains 0.267-0.900. |
| wave11-affine-mask | 11 | C2 | PROTOTYPE | `wave11_affine_mask_*`, `wave11_candidate_collision_*_affine_mask_*` | No-sidecar PUF additive mask reduces norm/raw-L2 candidate recovery near chance at std128 on Qwen3/Llama, but requires PUF unmask inside attention and numerical/performance audit. |
| wave5-paper-argument | 5 | C5 | PENDING | `paper_latex/` | Formal threat model and AES/TEE comparison. |
