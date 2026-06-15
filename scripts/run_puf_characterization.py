"""Simulated PUF / fuzzy-extractor characterization (paper RQ10, Table puf-hw).

Measures the primitive statistics that the security reductions of Section
"Security Model and Theoretical Limits" assume (Assumptions P1-P3 + fuzzy
extractor), using the software PUF simulator. These are SIMULATED values: the
root is a SHA256-derived 256-bit pseudorandom string keyed by device_id, and the
noise/fuzzy-extractor model corrects up to a 64-bit budget. Real silicon numbers
require FPGA traces (left as the hardware TODO); this run discharges the
assumptions in simulation and produces the same table shape.

  Quantity                       Assumption   Simulated
  intra-device BER p             P2           input sweep 0.00-0.40
  inter-device frac. Hamming     P3           ~0.5
  min-entropy H_inf(R|h)         P1,FE        256 bits (sim PRG root)
  Rep failure delta_FE           P2           tail of Binomial(256, p) > 64
  helper-data leakage            FE           N/A (sim has no syndrome helper)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct

from puf4secure_kvcache.puf_sim import _device_root, PUFSim


def _frac_hamming(a: bytes, b: bytes) -> float:
    bits = len(a) * 8
    d = sum(bin(x ^ y).count("1") for x, y in zip(a, b))
    return d / bits


def inter_device_hamming(n_pairs: int) -> dict:
    """Fractional Hamming between roots of distinct simulated devices (P3)."""
    vals = []
    for i in range(n_pairs):
        a = _device_root(f"device_{2*i}")
        b = _device_root(f"device_{2*i+1}")
        vals.append(_frac_hamming(a, b))
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {"pairs": n_pairs, "mean": mean, "std": var ** 0.5,
            "min": min(vals), "max": max(vals)}


def _noisy_root_hamming(device_id: str, ber: float, noise_seed: int) -> int:
    """Raw Hamming distance of a noisy response vs the true root (pre-correction)."""
    true_root = _device_root(device_id)
    seed_block = hashlib.sha256(
        b"puf_noise|" + device_id.encode() + b"|" + (b"\x00" * 16)
        + struct.pack(">q", noise_seed)
    ).digest()
    n_bits = len(true_root) * 8
    blocks, ctr = [], 0
    while sum(len(b) for b in blocks) < n_bits:
        blocks.append(hashlib.sha256(seed_block + struct.pack(">I", ctr)).digest())
        ctr += 1
    noise_buf = b"".join(blocks)
    flips = 0
    for i in range(n_bits):
        if noise_buf[i] < ber * 256:
            flips += 1
    return flips


def reliability_curve(bers: list[float], trials: int, budget: int) -> list[dict]:
    """Rep failure delta_FE vs BER (P2): fraction of noisy responses whose raw
    Hamming exceeds the correction budget, so the fuzzy extractor cannot recover."""
    rows = []
    n_bits = 256
    for ber in bers:
        raw_hd, fails, ok = [], 0, 0
        for t in range(trials):
            hd = _noisy_root_hamming("device_A", ber, noise_seed=t)
            raw_hd.append(hd)
            # PUFSim corrects iff hamming <= budget (matches _reconstructed_root).
            puf = PUFSim(device_id="device_A", mode="noisy", ber=ber,
                         correction_capacity=budget, noise_seed=t)
            recovered = puf._reconstructed_root()
            if recovered == _device_root("device_A"):
                ok += 1
            else:
                fails += 1
        rows.append({
            "ber": ber,
            "mean_raw_frac_hamming": (sum(raw_hd) / len(raw_hd)) / n_bits,
            "rep_success": ok / trials,
            "rep_failure_delta_fe": fails / trials,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--out", default="experiments/runs/puf_characterization_sim.json")
    args = ap.parse_args()

    bers = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    inter = inter_device_hamming(args.pairs)
    curve = reliability_curve(bers, args.trials, args.budget)
    summary = {
        "kind": "SIMULATED (software PUF, SHA256 root, no physical entropy)",
        "root_bits": 256,
        "correction_budget_bits": args.budget,
        "inter_device_frac_hamming": inter,
        "min_entropy_bits_sim": 256,
        "min_entropy_note": "simulated PRG root; real H_inf needs FPGA traces",
        "helper_data_leakage": "N/A (simulator has no syndrome helper data)",
        "reliability_curve": curve,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"inter-device frac Hamming: mean={inter['mean']:.4f} std={inter['std']:.4f} "
          f"(expect ~0.5)")
    print(f"{'BER':>6} {'raw_fracHD':>12} {'Rep success':>12} {'delta_FE':>10}")
    for r in curve:
        print(f"{r['ber']:>6.2f} {r['mean_raw_frac_hamming']:>12.4f} "
              f"{r['rep_success']:>12.4f} {r['rep_failure_delta_fe']:>10.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
