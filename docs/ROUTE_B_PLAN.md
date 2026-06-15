# Route B Plan — From Prototype to a Provable Systems-Security Paper

Status legend: ✅ done · 🟡 in progress · ⏳ TODO (software, I can run) · 🔬 TODO (needs hardware/data placeholder)

## 0. Reposition (the one-sentence pivot)

Move the contribution from "physical non-migratability" (which PUF→key→AES also gives) to:

> **No-plaintext-materialization KV-cache protection.** Attention computes *directly on
> protected coordinates*; the model-native plaintext K/V is **never materialized** anywhere
> in the memory hierarchy (at-rest or in-use). Encrypt-at-rest cannot offer this — it must
> decrypt a page to plaintext before attending. The price of computing on protected
> coordinates is that confidentiality becomes **statistical** (Gaussian-mechanism grade),
> not cryptographic IND. The paper *characterizes this tradeoff* and instantiates it with a
> real PUF.

Working title: *Computing on Protected Coordinates: A Geometry–Confidentiality Tradeoff for
Non-Migratable KV-Cache.*

## 1. New paper skeleton

1. Intro — KV persistence leak → no-materialization goal → 3 theory results + real PUF system.
2. Background, threat model (formal, at-rest/live × on/off-device).
3. **Security model & limits** — Def 0–3, Thm A (impossibility), Thm D (dichotomy), design reqs R1–R3.
4. Construction — folded-O (no materialization) + nonce'd additive PUF keystream + MAC.
5. Security analysis — Thm B (migration), Thm C (cross-session), Thm E (precision window), PUF/FE layer.
6. Evaluation — security/utility/systems/PUF + 5 baselines.
7. Discussion, related work, conclusion.

## 2. Theory (the spine) — STATUS ✅ drafted in `paper_latex/sections/sec_theory.tex`

### 2.1 Threat model: time × space split
- At-rest (persistent, migratable): `C̃`, weights `W`, FE helper `h`, non-secret IV/nonce/MAC metadata.
- Live (transient, in-RAM only during session): root `r_D`, bases `{O}`, masks `{M}`, plaintext `C`, activations.
- Migration adversary `A_mig`: all at-rest state + a *different* device root `r_{D'}`; **no** access to `r_D`/`P_D`.
- Non-migratability = at-rest state useless without the live root; the root cannot be reconstructed off-device (physical PUF). The boundary is **temporal**: cache persists after the session; the root only exists during it.

### 2.2 Definitions
- **Def 0 Correctness** — on-device attention is ε_num-exact vs plaintext attention.
- **Def 1 Migration confidentiality (IND-MIG)** — adversary with at-rest view + `r_{D'}` but not `r_D` cannot distinguish `m_0,m_1`. (Your candidate-matching attack IS the concrete distinguisher.)
- **Def 2 Cross-session resistance** — basis/mask learned in profiled sessions gives no advantage on a fresh-nonce challenge.
- **Def 3 In-use materialization leakage** — a single live snapshot reveals at most `C·O` (rotated), never plaintext `C`.

### 2.3 Theorems
- **Thm A (impossibility, full proof).** Any right-orthogonal scheme `C̃=C·O` leaks *exactly* the Gram matrix `CCᵀ` (a maximal invariant), so it fails Def 1 whenever `C_0C_0ᵀ ≠ C_1C_1ᵀ`. Generalizes the empirical norm/Gram/spectrum top-1=1.000 to ALL orthogonal maps. ⇒ requirement **R1: perturb the Gram matrix** (must be non-isometric/additive).
- **Thm D (dichotomy, sketch).** Compute-on-protected-coordinates ⇒ at most statistical confidentiality (`Adv ≳ ‖ΔC‖/σ`); cryptographic IND ⇒ must decrypt (materialize plaintext). Orthogonal = σ=0 end, AES = σ=∞ end, affine = tunable middle.
- **Thm B (migration confidentiality, hybrid proof).** Nonce'd affine `C̃=C·O+M`, `M=PRF_r(n_s,w,ℓ,h,i)~N(0,σ²)`: `Adv_IND-MIG ≤ ‖ΔC‖_F/(σ√(2π)) + Adv^PRF + Adv^FE`. Gives a *principled* σ lower bound (replaces std128). ⇒ requirement **R3: σ window**.
- **Thm C (cross-session, sketch).** Fresh nonce ⇒ PRF independence ⇒ profiled basis is random on challenge; reduces to Thm B.
- **Thm E (precision feasibility, full proof).** Recovering `C·O = C̃−M` in fp with unit roundoff `u`: feasible (num-fidelity τ AND confidentiality ε) iff `u ≤ ε·τ·√(2π)`. Predicts fp32-only (u≈6e-8 ok for ε=.03,τ=1e-3; fp16 5e-4, bf16 4e-3 fail) — matches the 0%/15.6%/56.25% divergence data.

### 2.4 Requirement R2 (from the differencing attack)
Per-position, session-stable mask is **keystream reuse**: two same-session caches at shared
positions difference out the mask. Fix: **per-write nonce `w`** (fresh per `Protect` call,
stored as non-secret IV) ⇒ no reuse ⇒ differencing leaves fresh `M_a−M_b`. AEAD nonce hygiene.

### 2.5 PUF/FE assumption layer
- P1 entropy `H_∞(R)≥m`; P2 reliability BER≤p, Rep failure δ_FE; P3 unclonability/independence `R_{D'}⊥R_D`.
- FE `(Gen,Rep)`: near-uniform `r`, helper `h` leaks ≤ (n−k) bits ⇒ `r` still a PRF key.
- Chain: P1+FE ⇒ PRF key; P3 ⇒ `r_{D'}` useless; P2 ⇒ Def 0 holds w.p. 1−δ_FE.

## 3. Methodology / systems — STATUS 🟡 (prose ✅ in sec3; code ⏳)
- Folded `O` into `W_q,W_k,W_v,W_o` ⇒ rotation is free + plaintext never materialized.
- Nonce'd additive PUF keystream (R2) + MAC over `(C̃,w,n_s)` for integrity (closes the GCM gap).
- Paged/flash: per-page write-nonce + position-indexed keystream ⇒ CTR-style random access; fuse subtraction into cache-load.
- 5 baselines (shared PUF root): B1 plaintext · B2 PUF-key+AES-CTR · B3 PUF-key+AES-GCM · B4 orthogonal-only · B5 affine (ours).

## 4. Experiment TODO (with acceptance criteria)

**Security**
- ⏳ Large-candidate IND-MIG distinguisher (1e6 space, real distributions); report full-space rank/MRR. Accept: affine Adv ≤ theory ε.
- ⏳ Differencing-attack PoC: per-position mask broken; nonce'd mask immune. (`scripts/run_differencing_attack.py`.)
- ⏳ KPA / adaptive attack against the affine mask (so far only orthogonal was attacked).
- 🔬 Helper-data-leakage-aware confidentiality regression.

**Utility**
- ⏳ Re-run PPL/MMLU/long-decode under the nonce'd construction.

**Systems**
- ⏳ 5-baseline throughput/latency under a real paged-attention stack (fp32/fp16/bf16).
- ⏳ Fused cache-load subtraction kernel; KV-mem + IV/MAC metadata overhead.

**PUF hardware (placeholder)**
- 🔬 FPGA PUF stream + FE calibration: intra/inter Hamming, reliability vs temp/voltage, min-entropy (NIST SP 800-90B), helper-data leakage, Rep failure vs budget.
- 🔬 Real cross-device `r_{D'}` migration confidentiality.

**Real privacy**
- 🔬 Secrets mined from genuine records (not seeded); LongBench/Enron true leak rates.

## 5. Attacks to defend/falsify in-paper
Differencing (R2 fix), affine-KPA, large-candidate, helper-data leakage, live snapshot (Def 3),
σ-outside-window numerical failure, MAC-absent injection forgery.

## 6. Milestones (theory-first)
1. ✅ Theory draft (Thm A/D/B/C/E + Def 0–3) — done, compiles.
2. ⏳ Nonce+MAC construction in code + differencing PoC (validates R2).
3. ⏳ Large-candidate + affine-KPA security runs.
4. ⏳ Paged real-stack 5-baseline systems comparison (decisive table).
5. 🔬 FPGA PUF calibration.
6. 🔬 Real private corpora.
