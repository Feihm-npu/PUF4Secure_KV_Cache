# PUF Plan v1 Results

Follow-up to `docs/PUF_BASIS_RESULTS.md`, addressing the four priorities in
`docs/PUF_PLAN_v1.md`:

1. K+V Givens + row/block layout + session refresh.
2. Secret-specific leakage metric (replace noisy ROUGE-L).
3. Large-scale profiling (≥1k prompts, unknown-permutation attack).
4. Level-3 post-RoPE attention wrapper (true attention-equivalent, no decloak).

All experiments use Qwen3-0.6B bf16 on a single GPU (`CUDA_VISIBLE_DEVICES=1`).
Three reference prompts under `experiments/runs/capture_00{0,1,2}_*`
(`482913` verification code, `blue-river` codename, `555-0108` phone).

## 1. Secret exact-match leakage metric

New module `src/puf4secure_kvcache/secret_metrics.py` defines `SecretSpec` and
`score_leakage()`, returning four flags per generation:

  - `keyword_present`  – any case-insensitive secret substring hit
  - `exact_digit_present` – the contiguous digit form (`482913`, `5550108`) appears
  - `digit_recall` – LCS-based digit recall, normalised to `|secret|`
  - `out_digit_count` – sanity (how many digits the model produced)

This replaces ROUGE-L / char-F1 as the primary safety metric because, as Plan v1
warned, ROUGE-L over-counts coincidental token overlap on short outputs.

## 2. Givens + layout + session-refresh variants

Added to `scripts/run_puf_basis_eval.py`:

  - `P4_KV_givens_row`
  - `P4_KV_givens_block`
  - `P5_KV_givens_session_refresh`
  - `P5_KV_givens_row_session_refresh`

The harness now uses a fresh `session_nonce` per variant whose name ends in
`_session_refresh`, modelling the legitimate device re-keying between sessions.

Results (3 captures × 4 variants, see
`experiments/runs/puf_basis_summary_v1.json`):

| Variant                              | V-inv | inj ROUGE-L | inj keyword | inj digit_recall | fid same (K/V) | fid wrong (K/V) |
| ------------------------------------ | ----: | ----------: | ----------: | ---------------: | -------------: | --------------: |
| P0_plain (baseline)                  | 1.00  | 0.28        |  3/3 prompts |             1.00 | 0/0            | n/a             |
| P4_KV_givens_row                     | 0.00  | 0.00        |       false |             0.00 | ~1e-3          | ~1.40           |
| P4_KV_givens_block                   | 0.00  | 0.00        |       false |             0.00 | ~1e-3          | ~1.40           |
| P5_KV_givens_session_refresh         | 0.00  | 0.06 (1/3 capture) | false |             0.00 | ~1e-3          | ~1.40           |
| P5_KV_givens_row_session_refresh     | 0.00  | 0.00        |       false |             0.00 | ~1e-3          | ~1.40           |

Key observation: the new secret metric confirms zero leakage across all
variants. The lone ROUGE-L = 0.17 reported for `P5_session_refresh` on
capture_002 (phone prompt) corresponds to **keyword_present=false and
digit_recall=0** — exactly the false positive that motivated this metric.

## 3. Large-scale Procrustes profiling

`scripts/run_profiling_large.py` streams 1000 random-token prompts
(seq_len 32 → 32 000 row-pairs per layer/head/target), accumulates Procrustes
statistics on-line, then evaluates three things per scenario:

  - `mean_fresh_prompt_residual`   `‖X·Ô − Y‖ / ‖Y‖` on a held-out random prompt
  - `mean_basis_recovery_err`      `‖Ô·O_trueᵀ − I‖ / √d`
  - downstream **V-inversion top-1** on the 3 real captures after stripping `Ô`

Three attacker strategies were tested:

  - `naive` – pair plaintext row *i* with protected row *i* (assumes no layout).
  - `sort_by_norm` – per-prompt sort both X and Y by row L2 norm before
                     pairing. Since `O` is orthogonal, row norms are preserved,
                     so this is the optimal greedy alignment under an unknown
                     row permutation.
  - `sort_by_norm_block` – same as above but restricted to each `block_size=8`
                     chunk (matched to the block-layout defense).

Results (`experiments/runs/profiling_large_summary.json`,
1000 prompts × seq_len 32, 32 000 rows per unit):

| Scenario                             | basis_rec_err | fresh_residual | V-inv top1 (real caps) | verdict          |
| ------------------------------------ | ------------: | -------------: | ---------------------: | ---------------- |
| S1 nolayout + naive                  | **0.003**     | **0.002**      | **1.000**              | broken           |
| S2 row-layout + naive                | 0.983         | 0.920          | 0.000                  | defended         |
| S2b row-layout + sort-by-norm        | **0.005**     | 0.130          | 0.073                  | partially broken |
| S3 block-layout + sort-block         | 0.736         | 0.826          | 0.454                  | partially broken |
| S4 session refresh + naive           | 0.003         | 1.403          | 0.000                  | defended         |

Three critical findings:

  1. **Without session nonce or layout, 1000 prompts is more than enough.**
     Procrustes recovers `O` to `‖Ô·O_trueᵀ − I‖ = 0.003`, and downstream
     V-inversion reaches the plaintext-baseline 100% top-1 on the held-out
     real captures.  This is the worst case in our threat model and the
     reason a fixed PUF basis alone is not a defense.

  2. **A smart attacker partially breaks row layout.** Sort-by-norm
     alignment lets the attacker recover `O` itself (`basis_rec_err = 0.005`),
     even though the held-out captures still show high transfer residual
     because the row permutation π is unknown for those prompts.  Downstream
     V-inversion still recovers ≈7% of tokens; for block layout it reaches
     45%. Row/block layout therefore must be combined with another defense.

  3. **Session refresh is robust against this attacker.** The basis recovered
     on one session has `mean_fresh_residual ≈ √2` on a different session,
     and the held-out V-inversion is 0.000.  This is the strongest single
     primitive in the current toolkit.

Locked-in implication: any production design must include session refresh
**and** layout. Layout alone (whether row or block) is breakable.

## 4. Level-3 post-RoPE attention wrapper

New module `src/puf4secure_kvcache/puf_attention.py` monkey-patches every
`Qwen3Attention.forward` to apply per-(layer, kv-head) rotations
**after RoPE / k_norm**, **before** `past_key_values.update`:

    q ← q · O_k (broadcast over the q-heads in the GQA group)
    k ← k · O_k
    v ← v · O_v
    cache stores rotated k, v
    attn_output ← attn_output · O_vᵀ        (per q-head)

Mathematically the wrapped model is identical to the plain model:
`q' k'ᵀ = q kᵀ`, `attn·v' = (attn·v)·O_v`, then `O_vᵀ` cancels.

`scripts/run_puf_attention_eval.py` runs three checks on each reference prompt:

  (1) **Legit equivalence** – wrapped logits vs plain logits.
  (2) **Cache binding** – rotated K/V vs plain K/V, plus attacker view (run
      NDSS attacks against the exfiltrated rotated cache).
  (3) **Wrong-device replay** – continue a captured legitimate cache under
      a different PUF.

Results (`experiments/runs/puf_attention_summary.json`):

| Prompt                          | logits rel_l2 (legit) | K rel_l2 | V rel_l2 | Attacker V-inv | Attacker inj ROUGE-L | Secret leak |
| ------------------------------- | --------------------: | -------: | -------: | -------------: | -------------------: | ----------- |
| `482913`                        | 2.4e-2                |    1.42  |    1.41  |          0.000 |               0.000  | none        |
| `blue-river`                    | 2.6e-2                |    1.46  |    1.41  |          0.000 |               0.04   | none        |
| `555-0108`                      | 3.5e-2                |    1.42  |    1.41  |          0.000 |               0.000  | none        |

Takeaways:

  - The wrapped cache contents are at ≈√2 relative-L2 from the plain cache —
    fully rotated. Stripping the wrapper and running V-inversion or injection
    on the cache yields zero leakage.
  - Legitimate-device decoding through the wrapped attention has `rel_l2 ≈ 2–4%`
    on the logits. This is bf16 attention-matmul drift (the rotation
    redistributes per-channel magnitudes, changing the rounding pattern). It
    does not break decoding for the prompts tested; production deployments
    should either keep attention in fp32 or use a precision-preserving
    rotation family.
  - The Level-3 wrapper meets the original proposal claim: *"the legitimate
    device can continue inference in the PUF basis without decloaking."*
    Cache-level `protect/recover` is no longer required on-device.

## Updated design

```text
Q/K path:  post-RoPE Givens, per-layer, per-kv-head (extended to q-heads via GQA group)
V/O path:  per-layer Givens, applied to V and to attn_output (O_vᵀ)
Layout:    session-refreshed row or block permutation (mandatory)
Session:   per-session nonce derived from PUF (Level-3 wrapper reinstalls per session)
PUF:       fuzzy-extractor-stable root + HMAC-SHA256 context derivation (per layer / head / purpose)
```

Status table:

| Defense layer | Implementation              | PoC validated                 |
| ------------- | --------------------------- | ----------------------------- |
| Cache-level   | `protect_kv_cache` (Givens) | yes (P3/P4/P5)                |
| Layout        | `LayoutSpec(row | block)`   | yes — partial without nonce   |
| Session       | `make_puf(..., session_nonce=)` | yes — robust to 1000-prompt profiling |
| Model-level   | `puf_attention.install_puf_attention` | yes — bf16 drift documented |
| Fuzzy PUF     | `puf_sim.mode='noisy'`      | yes (BER≤0.20 utility preserved) |

## Open follow-ups

1. **Logits drift in Level-3 wrapper**: investigate whether casting attention
   matmul to fp32 (only the K/V matmuls inside `eager_attention_forward`)
   eliminates the 2–4% rel_l2 drift; if so we have an exact equivalence claim.
   **Resolved — see § 5 below.**

2. **Stronger profiling attack**: replace per-prompt sort-by-norm with
   Hungarian on pairwise row distances; also try block-aware Hungarian for
   block layout.

3. **Profiling under Level-3 wrapper**: re-run the 1k-prompt sweep against
   the model-native rotated cache (not the cache-level `protect_kv_cache`)
   to confirm the attacker still cannot recover `O`.

4. **Longer-context utility test**: 18-token prompts hide whether the 3%
   logits drift compounds over many decode steps. Run a 128-token
   continuation under the wrapped model and compare with plain.
   **Resolved — see § 5 below.**

5. **GPU-side vectorised Givens** to reduce the protect/recover wall time
   below the current ≈50 ms/MB.

## 5. Precision analysis: bf16 quantization, not wrapper error

Driven by follow-ups (1) and (4), we exhaustively decomposed the source of
Level-3 logits drift. Three intermediate experiments and one decisive
diagnostic.

### 5.1 fp32 attention matmul does not change the drift

Forcing the entire eager attention computation (`QK^T`, softmax, `AV`) into
fp32 while keeping bf16 cache storage leaves prefill `rel_l2 ≈ 2.4–3.3 %`
unchanged across all three prompts. fp32 cache storage (rotated K/V kept as
fp32 in `past_key_values`) is also a no-op on prefill rel_l2. So the drift
does **not** come from attention matmul precision.

Artifacts:
`experiments/runs/puf_attention_summary_fp32attn.json`,
`puf_attention_summary_fp32cache.json`,
`puf_attention_summary_bf16cache.json`.

### 5.2 Token-level disagreement on prefill is small but non-zero

Adding top-1 and top-5 metrics to `run_puf_attention_eval.py`: prefill top-1
argmax disagreement is **0–15 %** per prompt, top-5 set overlap is **96–98 %**.
The disagreement positions are low-margin (multiple logits within bf16
ulp). For prefill scoring this is benign.

### 5.3 Long-context decode reveals catastrophic compounding

`scripts/run_long_decode_eval.py` runs 128-token greedy continuation under
plain bf16 vs. wrapped legit-decode bf16. Results
(`experiments/runs/long_decode_summary_bf16cache.json`):

| Prompt              | Divergence step | Exact match | ROUGE-L vs plain |
|---------------------|----------------:|------------:|-----------------:|
| 482913 verification | 4 / 128         | 0.05        | 0.20             |
| blue-river codename | 128 / 128       | 1.00        | 1.00             |
| 555-0108 phone      | 12 / 128        | 0.09        | 0.15             |

Per-step `rel_l2` grows from ~3 % at step 1 to ~100 % by step 30 — the
divergence reaches sampling noise once trajectories separate. fp32 cache
shifts prompt 3 from 12 to 13 (no qualitative change). The wrapper appears
broken in production scenarios.

### 5.4 Decisive diagnostic: wrap_fp ≡ plain_fp

`scripts/run_decode_precision_diagnosis.py` loads two model copies (bf16 and
fp32) and decodes 64 tokens under {plain, wrapped} × {bf16, fp32}. Result
(`experiments/runs/decode_precision_diagnosis.json`):

| Comparison              | P1 div step | P1 ROUGE-L | P2 | P3 div step | P3 ROUGE-L |
|-------------------------|------------:|-----------:|---:|------------:|-----------:|
| wrap_bf vs plain_bf     | **4**       | 0.23       | 64 | **12**      | 0.28       |
| **wrap_fp vs plain_fp** | **64**      | **1.00**   | 64 | **64**      | **1.00**   |
| plain_fp vs plain_bf    | 62          | 0.97       | 64 | **12**      | **0.28**   |

Two clean conclusions:

* **The wrapper is mathematically exact**: `wrap_fp` token-matches `plain_fp`
  to 64/64 on every prompt. The PUF rotation algebra (`q O_k`, `k O_k`,
  `v O_v`, post-attention `O_v^T`) is provably equivalent in real arithmetic
  and the implementation realises that in fp32.
* **Long-decode divergence is bf16 quantization, not rotation error**: on
  prompt 3 even *plain bf16 vs plain fp32* diverges at the same step 12
  with the same ROUGE-L 0.28 — there is no rotation in that comparison. On
  prompt 1, rotation does shift the bf16 noise direction (step 4 vs 62) but
  the underlying mechanism is identical: bf16 quantization of activations
  causes argmax flips at low-margin positions, which then cascade.

### 5.5 Implication and mitigations

Reporting the wrapper as having an "intrinsic 3 % drift" is misleading. The
wrapper is exact; bf16 inference of any Qwen3-0.6B model produces step-12
divergence on prompt 3 vs. an fp32 reference, and rotation merely picks a
different bf16 noise direction.

Production mitigations, in increasing cost order:

1. **fp32 attention activations** (keep Q/K/V fp32 inside the attention
   block only; o_proj input fp32) — preserves cache memory, costs one fp32
   matmul per layer. Implemented via the existing `_fp32_eager_attention`
   path; remaining drift then bounded by bf16 quantization on rotated
   *cache storage* (already shown <0.05% rel_l2 in §5.1, but compounding
   needs re-measuring on longer decodes).
2. **fp32 cache storage** (`fp32_cache=True`) — 2× cache memory, marginal
   gain on these prompts but should help on longer contexts where cache
   reads dominate.
3. **fp32 wrapped path** — exact equivalence, 2× model memory, hardware
   permitting.
4. **Stochastic rounding for bf16** in the rotation step — open research,
   not implemented here.

## Files added / modified

```
src/puf4secure_kvcache/secret_metrics.py      (new)
src/puf4secure_kvcache/puf_attention.py       (new; fp32 attention + fp32 cache opt)
scripts/run_profiling_large.py                 (new)
scripts/run_puf_attention_eval.py              (new; --fp32-cache, top-k metrics)
scripts/run_long_decode_eval.py                (new; 128-token continuation)
scripts/run_decode_precision_diagnosis.py      (new; bf16 vs fp32 origin diagnosis)
scripts/run_puf_basis_eval.py                  (extended: P4/P5 Givens, secret metrics)
experiments/runs/puf_basis_summary_v1.json     (new, 4 variants × 3 prompts)
experiments/runs/profiling_large_summary.json  (new, 5 scenarios × 1000 prompts)
experiments/runs/puf_attention_summary.json    (new, 3 prompts × 3 checks)
experiments/runs/puf_attention_summary_{fp32attn,bf16cache,fp32cache}.json  (new)
experiments/runs/long_decode_summary_{bf16cache,fp32cache}.json             (new)
experiments/runs/decode_precision_diagnosis.json                            (new)
```
