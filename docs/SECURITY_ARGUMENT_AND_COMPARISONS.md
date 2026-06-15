# Security Argument and Comparison Notes

This document is the Wave-5 non-empirical hardening artifact. It should be used
to update the paper after the full-scale runs are complete. It intentionally does
not introduce new measured claims.

## Threat Model

The attacker can obtain a KV cache snapshot from device memory, disk-backed cache
offload, an inter-process serving boundary, or a migrated serving state. The
attacker knows the model architecture, tokenizer, weights, RoPE schedule, serving
software, and the defense algorithm. The attacker may own a public copy of the
same model and may execute offline attacks on arbitrary hardware.

We separate four attacker classes:

| Class | Capability | Intended result |
| --- | --- | --- |
| A0 cache-only | Observes one protected cache and public model weights. | Direction-space replay/injection is disrupted, but finite-candidate matching is not fully prevented by the current orthogonal-only design because vector norms are preserved. |
| A1 wrong-device replay | Copies a protected cache to another device with a different PUF root. | Basis mismatch makes the cache uninterpretable. |
| A2 same-session known plaintext | Obtains aligned plaintext/protected cache pairs in the same session. | Can learn the fixed basis; this is a designed failure mode. |
| A3 cross-session profiler | Learns a basis in one session and attacks a later refreshed session. | Learned basis does not transfer because the nonce changes the orthogonal basis. |

The design does not defend a fully live device compromise where the attacker can
query the PUF-derived basis, read the reconstructed root, or run arbitrary code
inside the same protected process during the session. That case is equivalent to
compromising the legitimate inference endpoint.

The current orthogonal-only prototype also does not defend finite-candidate
matching attacks that use only per-vector norms. This is a cache-only attack, not
a same-session profiling attack: the attacker enumerates candidate secrets,
computes each candidate's plaintext KV norm signature with the public model, and
compares that signature to the protected cache. Because orthogonal maps preserve
norms exactly, this attack is basis-invariant.

Wave-10 tests a candidate mitigation for this limitation: expose only
unit-normalized rotated KV vectors in the migratable cache and keep the original
content norms in private in-process sidecar state that is restored inside
attention. This blocks the tested norm-only candidate attack on Qwen3 and Llama,
but it changes the trusted-state model. The sidecar must be treated as protected
serving state, not as part of the exported/migratable KV cache.

Wave-11 tests a no-sidecar alternative: add a PUF-derived affine mask to each
cached K/V row and regenerate/subtract the mask inside attention. This avoids
storing secret content metadata, but it also changes the original claim that the
attention computation can proceed directly in the protected coordinate system.

## Security Argument Sketch

For a layer/head/target, the protected row matrix is

```text
Y = X O_s,
```

where `X` is the model-native K or V row matrix and `O_s` is an orthogonal matrix
derived from the device root and session nonce. If the attacker has aligned
plaintext/protected pairs in the same session, recovering `O_s` is exactly the
orthogonal Procrustes problem. Our KPA experiments confirm this: with 16 known
32-token prompts on Qwen3-0.6B, same-session residual falls to `4.21e-05`.

However, the same equation gives an immediate invariant:

```text
||Y_i||_2 = ||X_i O_s||_2 = ||X_i||_2
```

for every token/head row `i`. Wave-9 verifies that this invariant is sufficient
for finite-candidate secret recovery in the tested setting: norm-only matching
recovers the true secret with top1/MRR `1.0` on Qwen3-0.6B, Llama-3.2-1B, and
Qwen2.5-1.5B across six secret families and 32 candidates. Therefore the
orthogonal-basis argument supports direction hiding and wrong-device replay
failure, but not broad cache confidentiality against candidate-norm matching.

A possible revised invariant-breaking design stores

```text
Y_i = (X_i O_s) / ||X_i O_s||_2
```

in the exposed cache and keeps `||X_i O_s||_2` in trusted sidecar state. The
legitimate attention path multiplies the sidecar norm back before QK/AV matmuls.
Initial Wave-10 artifacts show fp32 sanity equivalence and reduce norm-L2
candidate recovery to top1 `0.000` on Qwen3 and `0.033` on Llama for 32-candidate
six-secret-family runs. This is not yet a final security proof because a stronger
attacker may target sidecar handling, migration semantics, or other
basis-invariant statistics.

A no-sidecar variant stores

```text
Y_i = X_i O_s + M_{s,i},
```

where `M_{s,i}` is a PUF-derived mask for `(session, layer, head, position)`. The
legitimate process regenerates `M_{s,i}` and subtracts it inside attention before
using K/V. Initial Wave-11 artifacts show that bounded non-orthogonal linear
maps are insufficient, but affine masks at std128 reduce 32-candidate norm/raw-L2
recovery near chance on Qwen3 and Llama. This removes the sidecar objection, but
introduces mask-generation overhead and fp32 cancellation risk; it must be
audited before becoming a paper claim.

The intended boundary is therefore not secrecy of a fixed transform. The boundary
is session freshness. For independent session nonce `s'`, the target cache is

```text
Y' = X' O_{s'}.
```

An attacker who learned `O_s` strips `Y'` as `Y' O_s^T = X' O_{s'} O_s^T`. For
independently derived orthogonal bases, `O_{s'} O_s^T` is another unknown
orthogonal transform, so the stripped cache remains in a random basis relative to
the public model. Empirically, the same KPA sweep keeps session-refresh residual
near `1.416`, matching the expected mismatch scale for unrelated normalized
vectors.

This argument relies on three assumptions that should be explicit in the paper:

1. The PUF reconstruction root is unavailable outside the legitimate device.
2. The session nonce is fresh and not reused for caches that the attacker can profile and later attack.
3. The basis generator behaves as a pseudorandom orthogonal-map generator for the attacker's feasible observations.

## Comparison Table

| Defense | Secret placement | Cache usable without decrypt-before-attend? | Non-migratable by construction? | Main cost | Main limitation |
| --- | --- | --- | --- | --- | --- |
| No defense | None | Yes | No | None | Cache inversion, matching, and replay. |
| AES-encrypted KV at rest | Software key or KMS/TEE key | No | Only if key is device-bound | Encrypt/decrypt around cache reads/writes | Plaintext exists at attention boundary; key lifecycle dominates. |
| TEE-protected serving | TEE root and attested runtime | Yes inside enclave/VM | Yes if sealed to device/attestation identity | TEE memory/performance and operational complexity | Large GPU serving and side-channel assumptions are hard to audit. |
| Software random basis | Process memory secret | Yes | No if secret migrates or is dumped | Basis rotation overhead | Secret is not physically bound. |
| PUF-bound session basis (`PUF-Cache`) | Device PUF root + session nonce | Yes | Yes under root/nonce assumptions | Q/K/V rotations and precision-aware attention | Same-session KPA breaks reused bases; orthogonal-only norms leak finite candidates; real PUF validation pending. |
| PUF-bound basis + unit-norm sidecar prototype | Device PUF root + session nonce + private norm sidecar | Yes inside the legitimate process | Yes only if the sidecar is not exported with the cache | Normalize/restore cache norms plus sidecar management | Sidecar threat model, migration semantics, and overhead are not fully audited. |
| PUF-bound affine-mask prototype | Device PUF root + session nonce | Yes after PUF mask subtraction inside attention | Yes under root/nonce assumptions | Mask generation/subtraction and numerical cancellation | No sidecar, but no longer computes directly on masked cache; overhead and stronger attacks pending. |

## Paper Integration Checklist

- Add A0-A3 attacker classes to the threat model.
- State same-session profiling as an explicit non-goal unless the session nonce is reused.
- State the norm-side-channel limitation for the orthogonal-only design.
- If using the unit-norm sidecar mitigation, explicitly define the sidecar as trusted non-exported state and report it as a prototype until fully audited.
- If using the affine-mask mitigation, explicitly state that the legitimate path regenerates and subtracts PUF masks inside attention, and report it as a prototype until overhead/numerical/security audits complete.
- Add the comparison table to Discussion or Related Work after citations are verified.
- Tie KPA and native-Hungarian results to the argument that session refresh, not layout, is the security boundary.
- Keep real PUF as a limitation until hardware traces are available.
