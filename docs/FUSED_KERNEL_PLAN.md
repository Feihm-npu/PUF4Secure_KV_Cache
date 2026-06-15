# Fused Affine De-masking Triton Kernel — Implementation Plan (TODO 3)

Goal: make the affine PUF-Cache path cheap enough for real paged/flash serving by
fusing the additive-mask subtraction into the attention kernel's KV-load stage,
so de-masking piggybacks on a load that already happens instead of a separate
O(n)-per-step pass over the whole cache (the source of the current ~-34% decode
overhead). Target: fused affine decode overhead approaches the orthogonal path
(~-15%), while still never materializing model-native plaintext K/V.

Environment (verified 2026-06-16): Triton 3.5.1 with `tl.randn/tl.rand/tl.randint`
(counter-based Philox); A100 sm_80; current de-mask is the eager pass
`k_attn = k.float() - k_mask` in `puf_attention.py` (an `_sdpa_attention` backend
already exists as the pre-attention de-mask reference).

## Key design decisions

### D1. Regenerate the mask in-kernel via counter-based RNG (no HBM traffic)
Attention decode is memory-bandwidth bound. Storing the mask M in HBM and loading
it alongside K~/V~ would DOUBLE cache memory and load bandwidth — that is not
"near free." Instead regenerate M inside the kernel from a counter-based PRG keyed
by the PUF root. Triton's `tl.randn(seed, offset)` IS a Philox Gaussian generator:
for a loaded block of positions, `M = tl.randn(seed, pos*D + channel) * sigma`
costs a few ALU ops per element and overlaps the memory-bound page load. No extra
HBM reads, no extra cache memory — the property the orthogonal path already has.

### D2. Migrate the mask keystream to counter-based (construction refinement)
Today the mask is `torch.randn(generator=HMAC-seed)` generated sequentially — not
random-access, so it cannot be regenerated per-page inside a paged kernel. Change
BOTH the write path (Protect: C~ = XO + M) and the read path (Attend: subtract M)
to the same counter-based generator so they produce identical masks:

    M[ell,h,pos,j] = tl.randn(seed_{ell,h}, pos*D + j) * sigma
    seed_{ell,h}   = trunc(HMAC(root, nonce, ell, h))     # host-derived from PUF

To guarantee the write-path mask equals the in-kernel `tl.randn` bit-for-bit, use
ONE small Triton `generate_mask` kernel at Protect time and the identical
`tl.randn` call inside the fused attention kernel — same primitive on both paths.

Security note: the keystream is still i.i.d. N(0, sigma^2) per element, so
Theorems B (Gaussian-mechanism advantage) and E (precision window) are unchanged;
the per-write nonce still prevents reuse. The PRF in Theorem B is now instantiated
by Philox rather than HMAC. Philox is statistically excellent (BigCrush) and
counter-based but NOT cryptographic, so document the tradeoff honestly: the
migration attacker has no known plaintext (same-session KPA is already the stated
non-goal), so a cryptanalytic key-recovery on Philox is out of scope; for
defense-in-depth, refresh the Philox seed per (layer, head, nonce, PAGE) from an
HMAC subkey so only block_size*D Gaussian samples are ever drawn per Philox key.

### D3. Keyspace >= 128 bits (security-critical)
`tl.randn`'s seed is a single integer. A 32-bit seed is brute-forceable offline
(2^32 per stream). Ensure >=128-bit effective key by either (a) folding a 64-bit
per-session subkey into the high bits of the counter and using a 64-bit seed, or
(b) XOR-combining two independent Philox streams, or (c) a hand-rolled 10-round
Philox-4x32 with a 128-bit key in Triton. Resolve in Phase 0; verify the chosen
scheme's statistical quality (the mask must stay ~Gaussian).

### D4. fp32 cache stays (Theorem E), the kernel removes only the extra pass
The affine path is fp32-only (catastrophic cancellation, Theorem E). The fused
kernel loads fp32 K~/V~, subtracts fp32 M in SRAM, and does QK/PV in fp32/tf32
accumulation. It does NOT remove the fp32 requirement (2x bandwidth vs fp16 plain);
it removes the SEPARATE full-cache de-mask pass. Hypothesis: since that pass cost
roughly a second cache traversal per step, folding it into the load roughly halves
the affine overhead (-34% -> ~-17%, near orthogonal).

### D5. Orthogonal O stays folded; kernel only handles the additive mask
O_k/O_v fold into W_q,W_k,W_v,W_o at load time (already conceptually done), so the
kernel computes on XO coordinates throughout and never forms model-native plaintext
even in SRAM (in-use materialization stays at the rotated coordinate, Definition 3).

## Kernel algorithm (paged flash-decode, single query token)

    @triton.jit
    def fused_demask_attn_decode(Q, K_cache, V_cache, block_table,
                                 seed_k, seed_v, sigma, scale, D, BLOCK, Out):
      m_i, l_i, acc = -inf, 0.0, 0
      for blk in range(num_seq_blocks):
        phys     = block_table[blk]
        pos      = blk*BLOCK + arange(BLOCK)          # absolute positions
        ctr      = pos[:,None]*D + arange(D)[None,:]  # Philox counters
        Kt = load(K_cache[phys,:,kv_head,:])          # K~ = KO + M_k  (fp32)
        K  = Kt - tl.randn(seed_k, ctr) * sigma       # <-- FUSED de-mask -> KO
        S  = tl.dot(Q, K.trans()) * scale             # scores on protected coords
        # online-softmax update of m_i, l_i, p ...
        Vt = load(V_cache[phys,:,kv_head,:])          # V~ = VO + M_v
        V  = Vt - tl.randn(seed_v, ctr) * sigma       # <-- FUSED de-mask -> VO
        acc += p @ V
      Out = acc / l_i                                  # x O_v^T folded into W_o

The de-mask (`- tl.randn(...)`) lives between the page load and the matmul — the
single change versus a stock paged-attention kernel. GQA: query heads share their
KV head's seeds.

## Phases and milestones

- **Phase 0 — construction + keying (1 day).** Counter-based mask on write+read paths
  (one Triton `generate_mask` used everywhere); resolve D3 keyspace; re-run the
  std-sweep / utility / RQ11 confidentiality to confirm Theorems B/E unchanged and
  numbers match the current HMAC-randn construction within noise.
- **Phase 1 — standalone fused decode kernel (2-3 days).** Start from a vetted
  Triton flash-decode reference; add the in-kernel `tl.randn` de-mask. Validate
  bit/float-exact against eager `(K~ - M)` then SDPA on random tensors (no model).
- **Phase 2 — paged KV layout + HF integration (2 days).** Block table -> absolute
  position -> counter mapping; per-(layer,head,nonce) seeds derived on host from the
  PUF root; register as `_puf_attn_backend="triton_fused"` replacing the eager/sdpa
  de-mask path; GQA + multi-layer.
- **Phase 3 — correctness/numerics (1 day).** Fused vs eager-demask-SDPA: max logit
  diff < 1e-3, exact greedy tokens, fp32, across prefill 64/256/1024, both Qwen3 and
  Llama; reuse the divergence benchmark + PPL/HellaSwag/MMLU on the fused path.
- **Phase 4 — performance (1 day).** Decode tok/s for: plain fp16-SDPA, plain fp32,
  orthogonal fp32 (folded), affine eager fp32 (current ~-34%), **affine FUSED fp32**.
  Sweep prefill; report per-token cost; confirm O(1) extra/token. Compare to the AES
  baselines (RQ9) — fused affine should beat AES decrypt-on-load AND never
  materialize plaintext.
- **Phase 5 — paper (0.5 day).** New RQ12 fused-kernel overhead; update the
  performance table and the "fused kernel remains future work" caveats in
  Sections sec:design / sec:discussion / Open Gaps; update the dichotomy discussion
  (affine now deployable at near-orthogonal cost).

Decode-first (the O(n^2)-per-decode pass is where the overhead is). Prefill fusion
(full flash-attention with de-mask) is a follow-up once decode is validated.

## Validation matrix
- mask-match: Triton write-path mask == in-kernel regen (same `tl.randn`), exact.
- correctness: fused == eager-demask+SDPA (logit diff, greedy tokens), fp32.
- generality: Qwen3-0.6B + Llama-3.2-1B; GQA; prefill 64/256/1024; multi-layer.
- numerics: fp32 exact; fp16/bf16 as the Theorem-E precision study (expected fragile).
- security: re-run RQ11 (advantage vs bound, guessing entropy) on the Philox mask.
- perf: tok/s table above; KV-memory unchanged (no M in HBM).

## Risks & mitigations
- Triton paged-attn complexity -> start from a vetted reference; decode-only first.
- `tl.randn` seed entropy (D3) -> per-page HMAC subkey / 128-bit Philox; verify.
- Matching write mask to in-kernel `tl.randn` -> same primitive both paths; pin Triton.
- fp32 bandwidth inherent -> honest framing (kernel removes the extra pass, not fp32).
- `tl.randn` reproducibility across Triton versions -> pin 3.5.1; snapshot test.

## Effort: ~1.5-2 weeks focused. Highest-risk item is Phase 1 (the kernel); the
## `tl.randn` primitive removes the biggest unknown (in-kernel Gaussian regen).
