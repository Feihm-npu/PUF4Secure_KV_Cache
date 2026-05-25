# PUF-Bound KV-cache: Technical Design and Experiment Plan

This document turns `documents/proposal.tex` into an executable PoC plan. The research goal remains the proposal goal: make leaked KV-cache non-migratable and unusable for inversion, collision, and injection/replay attacks by binding its representation to a physical device basis. The implementation details are allowed to change based on experimental evidence.

## 1. Current Starting Point

The repository already has a working Qwen3-0.6B reproduction harness:

- Plaintext KV capture from Hugging Face `past_key_values`.
- Layer-0 V inversion attack.
- Collision attack with top-k pruning, batched candidate scoring, and 3-sigma diagnostics.
- Injection attack by passing leaked `past_key_values` plus an attacker instruction.
- KV-Cloak-style reversible cache transform `S P (K + A)` and decloak path.
- Metrics: token accuracy, ROUGE-L, char F1, recovery error, latency.

The most important reproduction fact is that Qwen3-0.6B has square `v_proj`, so layer-0 V-cache can be inverted exactly. This gives a strong local testbed for any new defense.

## 2. Design Principle

The proposal should be tested as a family of device-bound attention-equivalent transformations, not as one fixed construction.

The central invariant is:

```text
Q' = Q O,  K' = K O        => Q' K'^T = Q K^T
V' = V U,  W_o' = U^T W_o => A V' W_o' = A V W_o
```

where `O` and `U` are generated from a device-specific PUF root plus session and layer/group context.

The security goal is not only confidentiality. The cache should become:

- Hard to invert without the device basis.
- Hard to collision-match against a standard local model.
- Hard to replay or inject on another device or standard model.
- Usable on the original device after regenerating the same basis.

## 3. Threat Model for PoC

### Attacker Has

- Model weights, tokenizer, model config, and source code.
- Protected KV-cache copied from storage, offloading buffer, or memory dump after the session.
- Public helper data or public PUF reliability metadata.
- Ability to run the same model locally on another device.
- Ability to run inversion, collision, injection, replay, and chosen-input profiling experiments.

### Attacker Does Not Have in the Base PoC

- The original physical PUF oracle after cache migration.
- Live runtime matrices for the target session.
- The legitimate device's regenerated basis during decode.

### Explicitly Out of Scope for Base PoC

- Full live compromise where the attacker reads all live basis matrices and fused weights during inference.
- Side-channel extraction of PUF response bits.
- A complete production secure boot / TEE chain.

## 4. Implementation Architecture

### 4.1 Simulated PUF Root

First use a deterministic simulated PUF instead of FPGA hardware.

Inputs:

```text
device_id, session_nonce, layer_id, kv_group_id, block_id, purpose
```

Outputs:

```text
seed material for O, U, row permutation, stochastic knobs
```

The simulator should support three modes:

- `stable`: exact deterministic regeneration.
- `noisy`: bit flips with configurable BER before reconstruction.
- `wrong_device`: different device root, used for non-migratability tests.

This lets us test device binding before FPGA integration.

### 4.2 Transformation Levels

We should implement transformations at three increasingly realistic levels.

Level 1: cache-only attacker-view transform.

- Transform captured K/V tensors after plaintext capture.
- Use existing attack scripts unchanged by feeding transformed KV as leaked cache.
- Good for measuring attack breakage.
- Not sufficient for legitimate decode unless decloak or model-side transform is added.

Level 2: same-device replay transform.

- Transform stored cache.
- Before legitimate decode, regenerate basis and either decloak or run a matched model wrapper.
- Good for testing utility and cache reload.

Level 3: model-integrated attention-basis transform.

- Modify attention computation or projection weights so K/V are generated and consumed in the PUF basis.
- Required to demonstrate true attention-equivalent non-migratable cache.
- Start with post-RoPE hooks/wrappers before attempting weight fusion.

## 5. Candidate Matrix Structures

We should compare multiple structures because security, RoPE compatibility, and speed may conflict.

### A. Signed Permutation

Definition:

```text
O = P D
```

where `P` is a head-dimension permutation and `D` is a +/-1 diagonal sign matrix.

Expected properties:

- Very fast.
- Orthogonal and easy to invert.
- Likely weak against statistical attacks because it preserves magnitudes and coordinate distribution.
- Useful as a lower-bound baseline.

### B. Block-Diagonal Givens Rotation

Definition:

```text
O = BlockDiag(R(theta_1), ..., R(theta_{d_h/2}))
```

Expected properties:

- RoPE-compatible candidate.
- Easy to fuse with RoPE-like pair structure.
- Limited mixing capacity.
- Important to test because it may be the best engineering compromise.

### C. Hadamard-like Mixing

Definition:

```text
O = D1 H D2 P
```

Expected properties:

- Fast `O(d log d)` if implemented with fast Walsh-Hadamard transform.
- Strong global mixing.
- RoPE fusion is uncertain.
- Good candidate for V/O path where RoPE is not involved.

### D. Few Householder Products

Definition:

```text
O = H_1 H_2 ... H_k
```

Expected properties:

- Stronger mixing as `k` grows.
- Orthogonal by construction.
- Slower and harder to fuse.
- Useful as an upper-bound security candidate.

### E. Dense Orthogonal QR

Definition:

```text
O = qr(random_matrix).Q
```

Expected properties:

- Strong mixing baseline.
- Too expensive for production if applied online.
- Useful to measure whether weaker structures lose security.

## 6. PoC Variants

The experiments should not assume the full proposal is necessary. Test incremental variants and keep only what adds measurable security.

### Variant P0: Plaintext

No protection. Existing reproduction baseline.

### Variant P1: Software KV-Cloak

Existing `S P (K + A)` transform. This is the software-secret baseline.

### Variant P2: PUF-V Only

Only transform V-cache:

```text
V_d = V U_d
```

Motivation:

- Current strongest inversion result is layer-0 V inversion.
- V-only is minimal and may be enough to stop exact inversion and injection value interpretation.
- It does not protect K-only collision on deep layers.

Expected test result:

- V inversion should drop from 100% to near 0.
- Injection may degrade but could still leak via K attention structure or model behavior.

### Variant P3: PUF-KV Basis

Transform both K and V:

```text
K_d = K O_d
V_d = V U_d
```

Motivation:

- Breaks both collision and injection because both score computation and value interpretation are wrong for attackers.

Expected test result:

- Inversion, collision, and injection all fail on wrong device or standard model.

### Variant P4: PUF-KV + Row Layout

Add PUF-defined row/block permutation:

```text
K_d = P_b K O_d
V_d = P_b V U_d
```

Motivation:

- Breaks token-position correspondence used by collision attack and profiling.
- Helps resist Procrustes if attacker cannot align plaintext/protected rows.

Expected test result:

- Collision target rank loses separability.
- Profiling requires substantially more samples or fails under session/block nonce.

### Variant P5: Session-Refreshed PUF Basis

Add session nonce to every basis generation.

Motivation:

- Prevents long-term profiling from accumulating pairs across sessions.

Expected test result:

- Procrustes recovery trained on one session does not transfer to another session.

### Variant P6: Noisy PUF Exact Path

Simulate BER and reconstruction.

Motivation:

- Validate that basis regeneration error is smaller than cross-device mismatch.

Expected test result:

- Same-device noisy basis preserves generation under acceptable BER after reconstruction.
- Wrong-device basis fails by a large margin.

## 7. How to Integrate With Qwen3-0.6B

Qwen3-0.6B cache shape from reproduction:

```text
[batch=1, num_kv_heads=8, seq_len, head_dim=128]
```

Use `kv_group_id = kv_head_id` for cache-level transforms. For GQA, query heads sharing the same KV head must use the same `O` in the Q/K path.

### Recommended First Implementation

Start with cache-level transforms:

```text
src/puf4secure_kvcache/puf_sim.py
src/puf4secure_kvcache/puf_basis.py
scripts/run_puf_basis_eval.py
scripts/run_profiling_attack.py
```

Functions to add:

```text
derive_seed(device_id, session_nonce, layer, group, block, purpose)
make_orthogonal(kind, head_dim, seed)
protect_kv_cache(kv, device_id, session_nonce, kind, include_k, include_v, layout)
recover_kv_cache(protected, same parameters)
wrong_device_replay(model, tokenizer, protected_kv, device_id_B)
```

### Later Model-Integrated Implementation

Add a small attention wrapper or monkey patch:

- Apply `O` to Q and K after RoPE.
- Apply `U` to V before caching.
- Apply `U^T` before or inside output projection.

This is safer than immediate weight fusion and lets us measure exactness first.

Only after post-RoPE wrapper is correct should we attempt:

- RoPE-compatible fused Q/K basis.
- V/O weight fusion.
- GPU/vectorized implementation.

## 8. Experiment Plan

### Experiment 1: Attack Baseline Reconfirmation

Goal: verify that the current attack baseline remains stable.

Run:

- Inversion on layer-0 V.
- Collision on layer-0 V and at least one mid-layer K/V.
- Injection with four instructions.

Success criterion:

- V inversion is near 100% on plaintext.
- Collision and injection show non-trivial leakage.

Why this matters:

- Every defense result should be compared against the same baseline captures.

### Experiment 2: Orthogonal Structure Security Sweep

Goal: identify which matrix structures actually break attacks.

Factors:

| Factor | Values |
| --- | --- |
| Transform | V-only, K-only, K+V |
| Matrix kind | signed permutation, Givens, Hadamard, Householder-k, dense QR |
| Layer | 0, 13, 27 |
| Target | K, V |
| Layout | none, row permutation, block permutation |
| Session | fixed, refreshed |

Metrics:

- Inversion token accuracy.
- Collision token accuracy.
- True-token rank and `d_target / d_other` separability.
- Injection ROUGE-L and char F1.
- Runtime ms/MB.

Decision rule:

- Reject structures that reduce utility or fail to suppress collision/injection.
- Prefer the cheapest structure that reaches near KV-Cloak-level attack suppression.

### Experiment 3: Legitimate Same-Device Utility

Goal: ensure protected cache can still be used by the legitimate device.

Cases:

- Plain decode from captured cache.
- Protect then recover then decode.
- Wrong-device recover then decode.
- Model-integrated same-device replay once implemented.

Metrics:

- Generated text exact match against plaintext.
- Next-token KL divergence.
- Top-1 next-token agreement.
- Relative L2 recovery error.
- ROUGE-L between legitimate output and plaintext output.

Decision rule:

- Same-device exact path should be identical or nearly identical.
- Wrong-device path should fail injection/replay and show high KL or low output agreement.

### Experiment 4: Cross-Device Non-Migratability

Goal: directly validate the proposal's main claim.

Setup:

- Generate protected cache on simulated device A.
- Try to use it with device B basis, device C basis, and standard model.
- Repeat across multiple prompts and sessions.

Metrics:

- Cache replay continuation success.
- Injection leakage score.
- Output agreement with same-device legitimate output.
- Device-Binding Margin:

```text
DBM = E[D(output_A, output_wrong_device)] - E[D(output_A, output_same_device_noisy)]
```

Decision rule:

- DBM should be positive and large.
- Wrong-device replay should not reveal secrets.

### Experiment 5: Chosen-Input Profiling / Procrustes Attack

Goal: test the strongest obvious weakness.

Attack:

Given plaintext/protected pairs, solve:

```text
min_O ||K_plain O - K_protected||_F, O^T O = I
```

and similarly for V/U.

Settings:

- Known aligned rows, no layout.
- Unknown row permutation.
- Block permutation with known block id but unknown row order.
- Session-fixed basis.
- Session-refreshed basis.
- Per-block basis.

Metrics:
- Basis recovery error `||O_hat O^T - I||_F / sqrt(d)`.
- Attack success after applying recovered basis.
- Number of chosen prompts needed to recover useful basis.
- Transfer success across sessions.

Decision rule:

- If fixed basis without layout is easily recovered, that is expected.
- A viable scheme must make recovery fail or not transfer under session/per-block nonce and layout.

### Experiment 6: PUF Noise Tolerance

Goal: determine how much basis regeneration noise can be tolerated.

Simulate:

- BER from `0` to `1e-1` before reconstruction.
- Reconstruction success/failure.
- Small-angle basis perturbation.
- Row permutation errors.

Metrics:

- Basis reconstruction error.
- Next-token KL.
- Output exact-match rate.
- Injection/replay leakage under noisy legitimate basis.
- DBM under noise.

Decision rule:

- Exact path must keep utility stable under realistic BER after helper-data reconstruction.
- If exact orthogonal matrices are too noise-sensitive, restrict stable PUF output to seeds reconstructed by fuzzy extractor and keep noisy bits only for stochastic path.

### Experiment 7: Performance and Vectorization

Goal: identify which candidate can be made practical.

Measurements:

- ms/MB protect/recover.
- prefill overhead.
- decode overhead.
- GPU memory overhead.
- basis generation time.

Compare:

- Python loop cache transform.
- Batched PyTorch transform over `[layer, head, block]`.
- Signed permutation / Givens / Hadamard / Householder / dense QR.

Decision rule:

- Dense QR may be acceptable only as a security upper-bound.
- Production candidate should target vectorized signed permutation, Givens, or Hadamard-like mixing.

### Experiment 8: FPGA PUF Feasibility

Goal: verify physical source feasibility after software PoC narrows design choices.

Collect:

- Reliability.
- Uniqueness.
- BER across reboot, temperature, voltage.
- Min-entropy.
- Helper-data reconstruction latency.

Use PUF data to drive:

- Device roots.
- Session seeds.
- Matrix parameter generation.
- Noise model calibration.

Decision rule:

- FPGA stage should validate the selected software design, not search the entire design space.

## 9. Minimal PoC Success Criteria

The first publishable PoC should demonstrate all of the following on Qwen3-0.6B:

| Requirement | Target |
| --- | --- |
| Plain V inversion | near 100% token recovery |
| Protected V inversion | near 0% token recovery |
| Plain collision | clear leakage above random |
| Protected collision | near random or no separability |
| Plain injection | secret appears or semantic leakage is high |
| Protected injection on standard/wrong device | no secret leakage |
| Same-device recovery | output identical or next-token KL near 0 |
| Wrong-device replay | low output agreement and low leakage |
| Profiling | fixed no-layout may fail, session/layout variant resists |
| Performance | identify at least one vectorizable candidate |

## 10. Expected Failure Modes and Pivot Rules

### If V-only is enough

V-only is attractive because Qwen3 layer-0 V inversion is the strongest result. Still test K/collision and injection. If K-only collision or injection remains strong, move to K+V.

### If Givens is too weak

Use Givens only for Q/K RoPE-compatible path and stronger Hadamard/Householder for V/O.

### If Hadamard breaks RoPE fusion

Keep Hadamard for V/O and use post-RoPE Givens for Q/K. Treat Q/K fusion as future work.

### If profiling recovers fixed basis

Do not abandon the idea. Move to session nonce, per-block basis, row layout, and runtime policy. A fixed reusable basis is likely not defensible.

### If PUF noise breaks exactness

Use fuzzy reconstruction for exact path. Do not feed raw noisy bits directly into matrix generation. Keep noisy bits only for bounded stochastic path.

### If online overhead is too high

Separate security validity from production performance. First prove non-migratability with cache-level transforms, then vectorize and fuse only the winning construction.

## 11. Implementation Order

1. Add simulated PUF and orthogonal matrix generator.
2. Add `protect_kv_cache` / `recover_kv_cache` supporting V-only, K+V, layout, and multiple matrix kinds.
3. Add `run_puf_basis_eval.py` to run inversion/collision/injection/fidelity under P0-P6 variants.
4. Add wrong-device replay tests.
5. Add Procrustes profiling attack.
6. Vectorize the best two candidate transforms.
7. Add post-RoPE attention wrapper for model-integrated same-device basis.
8. Add simulated PUF noise and DBM metric.
9. Only then connect FPGA PUF data.

## 12. Near-Term Code Tasks

Concrete files to add or modify:

| File | Purpose |
| --- | --- |
| `src/puf4secure_kvcache/puf_sim.py` | simulated stable/noisy/wrong-device PUF roots |
| `src/puf4secure_kvcache/puf_basis.py` | matrix generators and KV protection/recovery |
| `src/puf4secure_kvcache/profiling.py` | Procrustes basis recovery attacks |
| `scripts/run_puf_basis_eval.py` | one command for PUF-basis defense evaluation |
| `scripts/run_profiling_attack.py` | chosen-input profiling experiments |
| `configs/puf_basis.yaml` | experiment matrix defaults |

## 13. Recommended First Experiment Batch

Use the existing three synthetic prompts first.

Run variants:

- P0 plaintext.
- P1 KV-Cloak baseline.
- P2 V-only with signed permutation, Givens, Hadamard, dense QR.
- P3 K+V with Givens for K and Hadamard/dense QR for V.
- P4 K+V plus row/block layout.

Run attacks:

- Layer-0 V inversion.
- Layer-0 V collision with top 1% vocab.
- Mid-layer K collision on layer 13.
- Injection with `Repeat the previous content.` and `Summarize the previous content.`

Stop criteria for this batch:

- At least one PUF-basis variant suppresses all three attacks while preserving same-device recovery.
- If none do, inspect whether failure is due to K path, V path, layout, or incorrect legitimate recovery.

## 14. Working Hypothesis

The most likely viable first scheme is:

```text
Q/K path: post-RoPE group-level block-diagonal Givens basis
V/O path: Hadamard-like or Householder basis
Layout: session/block PUF-controlled row permutation
PUF: stable reconstructed seed for exact path, noisy components only for optional stochastic path
```

The most likely defensible paper claim is not that PUF magically makes matrices secret. The claim should be that edge KV-cache can be made non-migratable by binding its attention-equivalent representation to a physical root, and that session/layout choices are necessary to resist profiling.
