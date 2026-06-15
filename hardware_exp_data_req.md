# Real PUF Hardware: Experiment & Data Requirements

This document specifies what physical-PUF hardware experiments and datasets are
needed to replace the software simulator (`src/puf4secure_kvcache/puf_sim.py`,
class `PUFSim`) with a real silicon PUF, and to upgrade every "simulated PUF"
claim in the paper to a hardware-validated one. It is written so a hardware/FPGA
collaborator can execute it without reading the rest of the codebase.

Everything below is currently **stubbed in software**. No physical PUF was used
in any result to date; the simulator models stable regeneration, wrong-device
roots, and bit-error correction, but not physical reliability, uniqueness, or
helper-data leakage. Those three are exactly what the hardware campaign must
measure.

---

## 1. What the software currently fakes (and must be replaced)

`PUFSim` is the single trust anchor of the whole defense. The protected KV-cache
bases and affine masks are all derived from it. The real hardware must be a
**drop-in replacement** that satisfies the same interface:

```
PUFSim(device_id, session_nonce, mode, ber, correction_capacity, ...)
  ._reconstructed_root() -> 32 bytes        # the stable, device-bound secret
  .derive_bytes(layer, group, block, purpose, n_bytes) -> bytes
  .derive_seed(layer, group, block, purpose) -> int (63-bit)
```

Key facts the replacement must preserve:

| Property in code | Current (simulated) value | What hardware must provide |
| --- | --- | --- |
| Root length | 32 bytes (256-bit) | >= 256 bits of full-entropy key after fuzzy extraction |
| Correction budget | `correction_capacity = 64` bits | ECC that corrects the measured intra-device Hamming distance with margin |
| Context derivation | HMAC-SHA256(root, session_nonce \|\| layer \|\| group \|\| block \|\| purpose) | unchanged — keep the HMAC; only the `root` source changes |
| `mode="stable"` | perfect regeneration | same device + same helper data regenerates the identical root across power cycles / temperature / voltage / aging |
| `mode="wrong_device"` | a different device's root | a *different physical device* yields an unrelated root even with the victim's helper data |
| `mode="noisy"`, BER sweep | bit flips then fuzzy-correct | the real raw-response BER must fall inside the ECC budget under the operating envelope |

The integration boundary is small: only `_reconstructed_root()` changes. The
HMAC context-derivation, the orthogonal-basis / affine-mask generators, and the
attention wrapper stay as-is. So the hardware deliverable is precisely **a stable
256-bit root + its helper data, per device**, plus the measurements that justify
the ECC budget and the uniqueness/leakage claims.

---

## 2. Hardware required

Pick one PUF primitive (in rough order of integration ease on FPGA):

- **SRAM PUF** — power-on SRAM state. Needs uninitialized SRAM read at boot;
  cleanest entropy, well-studied fuzzy extractors. Preferred if the board exposes
  raw SRAM power-up values.
- **RO (ring-oscillator) PUF** — frequency comparisons of RO pairs. Easy to
  instantiate in FPGA fabric; sensitive to temperature, so the reliability sweep
  matters most here.
- **Arbiter PUF / XOR-Arbiter PUF** — delay-based, large CRP space; if used,
  include a modeling-attack resistance test (see §4.6).

Minimum kit:
- >= 10 nominally-identical FPGA boards (for inter-device uniqueness statistics).
  20+ preferred for tight confidence intervals.
- A thermal chamber or at least a controllable heat source covering the target
  operating range (e.g. 0 C to 70 C, or the deployment envelope).
- A programmable supply for +/-10% Vdd variation.
- A host link (UART/JTAG/PCIe) to stream raw responses and helper data off-device.

---

## 3. Data to collect (datasets and formats)

All artifacts should land under `experiments/puf_hardware/` mirroring the
existing `experiments/runs/` JSON convention.

### 3.1 Raw response / CRP dataset
- Per device, per challenge: the raw PUF response bits.
- SRAM PUF: dump the same SRAM region; >= 4096 bits per device (we need 256
  post-extraction, so collect generously for ECC overhead).
- RO/Arbiter PUF: >= 100,000 CRPs per device (large enough for both key
  derivation and the modeling-attack test).
- Format: `{device_id, challenge_hex, response_bits, temp_C, vdd_mV, timestamp, power_cycle_idx}` per row (JSONL or parquet).

### 3.2 Repeated-measurement dataset (reliability)
- For each device, re-measure the **same** challenges:
  - >= 100 power cycles at nominal temp/voltage,
  - a temperature sweep (>= 5 points across the envelope), >= 20 reads each,
  - a voltage sweep (>= 3 points), >= 20 reads each,
  - if feasible, an accelerated-aging point (post burn-in).
- This dataset yields intra-device (within-device) Hamming distance, i.e. the raw
  BER the ECC must absorb.

### 3.3 Enrollment / helper-data dataset
- For each device: the fuzzy-extractor **helper data** (e.g. code-offset syndrome)
  produced at enrollment, plus the resulting stable 256-bit root commitment
  (store a hash of the root, never the root itself).

---

## 4. Experiments and the paper claim each validates

Each experiment replaces a specific simulated assumption. Target numbers are
starting points; report measured values with confidence intervals.

### 4.1 Reliability / intra-device BER  -> validates the correction budget
- **Measure:** mean and worst-case intra-device fractional Hamming distance of
  the raw response vs the enrolled reference, across the full temp/voltage/aging
  envelope (dataset §3.2).
- **Paper claim it backs:** the `correction_capacity = 64` bits over a 256-bit
  root, and the "BER 0.00-0.20 preserves utility, >= 0.30 collapses" result
  (currently simulated in `run_noise_eval` / the PUF-noise table).
- **Acceptance:** worst-case raw BER + safety margin must be < the ECC budget
  chosen in §4.3, with failure-to-reconstruct probability target <= 1e-6
  (or the deployment's key-failure spec).

### 4.2 Uniqueness / inter-device Hamming distance  -> validates wrong-device binding
- **Measure:** pairwise inter-device fractional Hamming distance over all device
  pairs (dataset §3.1). Ideal ~ 0.50.
- **Paper claim it backs:** `mode="wrong_device"` and every non-migratability
  result (the migration head-to-head, V relative-L2 ~ sqrt(2), wrong-device
  V-inversion 0.000). These currently assume an unrelated root on a different
  device; uniqueness near 0.50 is what makes that real.
- **Acceptance:** mean inter-device HD in [0.45, 0.55]; no device pair below a
  safety threshold (e.g. > 0.35).

### 4.3 Fuzzy-extractor calibration  -> validates stable 256-bit root
- **Do:** choose an ECC (e.g. BCH / Reed-Muller / concatenated) sized to the §4.1
  worst-case BER; generate helper data; verify regeneration of the identical root
  across all conditions in §3.2.
- **Paper claim it backs:** `mode="stable"` perfect regeneration; the assumption
  that the legitimate device always recovers the same basis (RQ3/RQ4 legitimate
  equivalence; legit-device V-inversion 1.000).
- **Acceptance:** 0 reconstruction failures across the reliability dataset at the
  chosen budget; extracted key passes NIST SP 800-90B min-entropy estimation for
  >= 256 bits.

### 4.4 Helper-data leakage  -> NEW claim, currently unmeasured
- **Measure:** residual min-entropy of the root conditioned on public helper data;
  confirm the fuzzy extractor is information-theoretically or computationally
  secure for the chosen construction.
- **Paper claim it backs:** the threat-model assumption that the attacker may hold
  "public helper data" yet cannot reconstruct the root. The simulator simply
  asserts this; hardware must show the helper data does not collapse entropy.
- **Acceptance:** conditional min-entropy >= 128 bits (or the target security
  level) after helper-data exposure.

### 4.5 Throughput / latency of root reconstruction  -> deployment overhead
- **Measure:** wall-clock to reconstruct the root + derive the per-(layer,head)
  seeds at session start, and whether it must be amortized once per session.
- **Paper claim it backs:** the performance discussion — current decode-overhead
  numbers (orthogonal -15%, affine -32%) assume root derivation is a one-time
  per-session cost. Confirm that holds on hardware.
- **Acceptance:** root reconstruction is a per-session constant (not per-token),
  with measured latency reported.

### 4.6 (If Arbiter/RO PUF) modeling-attack resistance
- **Measure:** train ML models (logistic regression, MLP, evolutionary strategies)
  on the §3.1 CRPs; report prediction accuracy vs CRPs used.
- **Paper claim it backs:** the unclonability assumption (A1/A2/A3 attacker
  classes) — a strong-PUF whose CRPs are ML-predictable is not unclonable.
- **Acceptance:** prediction accuracy stays near 50% within the CRP budget an
  attacker could realistically observe; otherwise switch to a controlled / weak
  PUF used only for key storage (SRAM-style), which sidesteps this.

---

## 5. Integration steps once data exists

1. Implement `HardwarePUF` with the same interface as `PUFSim`, where
   `_reconstructed_root()` runs the fuzzy-extractor `Rep(raw_response, helper)`
   on-device and returns the 256-bit root. Keep `derive_bytes` / `derive_seed`
   byte-for-byte identical (same HMAC-SHA256 context) so all downstream bases and
   masks are unchanged.
2. Add `make_puf(..., backend="hardware")` to select it; default stays the
   simulator so existing experiments reproduce.
3. Re-run the wrapper sanity (`run_wrapper_sanity.py`) with the hardware root on
   one device to confirm fp32 equivalence still holds (max logit diff ~ 1e-5).
4. Re-run the migration / wrong-device test (`run_migration_comparison.py`) across
   two *physical* devices to replace the simulated wrong-device root with a real
   one.
5. Replace the simulated PUF-noise table with the §4.1 measured BER envelope and
   the §4.3 reconstruction-failure rate.

---

## 6. Minimal viable campaign (if resources are tight)

If a full multi-device thermal campaign is infeasible, the smallest result that
still upgrades the paper from "simulated" to "hardware-validated" is:

- **2 physical devices**, SRAM PUF, nominal conditions only:
  - enroll device A, show stable root regeneration over >= 100 power cycles (§4.3),
  - show device B yields an unrelated root with A's helper data (§4.2 minimal),
  - run `run_migration_comparison.py` across A and B for a real wrong-device row.
- This converts the single most load-bearing assumption (physical
  non-migratability) into a measured result, even without the full
  reliability/uniqueness/entropy statistics. The temperature/voltage/aging and
  helper-data-leakage experiments can then be framed as the next hardware step.

---

## 7. Summary table: simulated -> required measurement

| Simulated assumption (today) | Hardware experiment (§) | Acceptance target |
| --- | --- | --- |
| Stable root regeneration | Reliability + fuzzy-extractor (4.1, 4.3) | 0 reconstruction failures; fail prob <= 1e-6 |
| Wrong-device = unrelated root | Uniqueness (4.2) | inter-device HD in [0.45, 0.55] |
| 256-bit full-entropy root | Min-entropy estimation (4.3) | >= 256-bit min-entropy (SP 800-90B) |
| Helper data leaks nothing usable | Helper-data leakage (4.4) | conditional min-entropy >= 128 bits |
| BER <= 0.20 utility / >= 0.30 collapse | Reliability envelope (4.1) | worst-case BER + margin < ECC budget |
| Per-session (not per-token) cost | Throughput (4.5) | one-time per-session reconstruction |
| PUF is unclonable | Modeling-attack (4.6, strong PUFs) | ML prediction ~ 50% within attacker CRP budget |
