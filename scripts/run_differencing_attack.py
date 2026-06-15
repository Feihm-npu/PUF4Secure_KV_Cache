"""Differencing attack on same-session affine-masked KV caches (paper RQ8 / R2).

Threat: a migration adversary holds two protected caches produced in the *same*
session that share a position (e.g. two prefills of a multi-turn conversation, or
a chosen-message probe). The affine scheme stores ``X_i O + M_i``. If the additive
mask ``M_i`` is a function of position only, it is identical in both caches, so
differencing them at the shared position cancels the mask:

    C~^a_i - C~^b_i = (X^a_i - X^b_i) O           # mask-free, back to Theorem A

and the per-token L2 norm (a right-orthogonal invariant, ||(.)O|| = ||.||) leaks
the secret to a finite-candidate matcher. The fix (requirement R2) is a fresh
per-write nonce so the two keystreams are independent and the difference retains a
fresh mask.

This PoC demonstrates the mechanism and the fix at the cache-tensor level with
synthetic K/V-scale vectors so it runs in seconds with no model or GPU. It
produces the four cells of paper Table~\\ref{tab:differencing}:

    mask derivation        single-cache top-1     differenced top-1
    position-only M_{s,i}   ~ 1/N (chance)         ~ 1.000   (BROKEN)
    per-write-nonce M_{w,i} ~ 1/N (chance)         ~ 1/N     (immune)

TODO(real-model): swap the synthetic vectors for captured Qwen3-0.6B / Llama-3.2-1B
layer-0 K/V rows at a secret position (reuse capture_kv.py) to report the
real-cache numbers for the paper. The synthetic run already validates R2 and the
Theorem-A norm-invariant restoration.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

from puf4secure_kvcache.puf_sim import make_puf
from puf4secure_kvcache.puf_attention import _seed_generator


def _orthogonal_basis(puf, d: int, layer: int, head: int) -> torch.Tensor:
    """PUF-seeded orthogonal matrix O in O(d) via QR (det fixed to +1)."""
    seed = puf.derive_seed(layer=layer, group=head, block=-1, purpose="K_attn")
    g = _seed_generator(seed, torch.device("cpu"))
    a = torch.randn(d, d, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # make QR deterministic
    return q


def _mask_rows(puf, d: int, layer: int, position: int, std: float,
               write_nonce: str | None) -> torch.Tensor:
    """One PUF-derived Gaussian mask row N(0, std^2 I_d) at a position.

    When ``write_nonce`` is None the mask is a function of position only (legacy
    behaviour); otherwise it is fresh per write (R2)."""
    purpose = "V_affine_mask" if write_nonce is None else f"V_affine_mask|wn={write_nonce}"
    seed = puf.derive_seed(layer=layer, group=-1, block=position, purpose=purpose)
    g = _seed_generator(seed, torch.device("cpu"))
    return torch.randn(d, generator=g, dtype=torch.float64) * float(std)


def _norm_match_top1(diff_vec: torch.Tensor, candidates: torch.Tensor,
                     reference: torch.Tensor, true_idx: int) -> int:
    """Finite-candidate norm-L2 matcher (Theorem A invariant).

    The adversary knows the reference token r0 in the other cache and the
    candidate set. It predicts argmin_c | ||diff|| - ||c - r0|| |. Returns 1 if
    the true candidate is ranked first, else 0."""
    target = diff_vec.norm()
    cand_norms = (candidates - reference).norm(dim=1)
    pred = int(torch.argmin((cand_norms - target).abs()).item())
    return int(pred == true_idx)


def run(n_candidates: int, dim: int, trials: int, mask_std: float,
        device_id: str, seed0: int) -> dict:
    results = {}
    for write_nonce_mode in ("position_only", "per_write_nonce"):
        single_hits = 0
        diff_hits = 0
        for t in range(trials):
            puf = make_puf(device_id, mode="stable")
            # Per-trial RNG for the (non-PUF) plaintext content.
            cg = torch.Generator().manual_seed(seed0 + t)
            candidates = torch.randn(n_candidates, dim, generator=cg, dtype=torch.float64)
            true_idx = int(torch.randint(0, n_candidates, (1,), generator=cg).item())
            secret = candidates[true_idx]
            reference = torch.randn(dim, generator=cg, dtype=torch.float64)  # turn-B token r0

            O = _orthogonal_basis(puf, dim, layer=0, head=0)
            pos = 7  # shared secret slot

            # Two same-session writes. write_nonce distinguishes the two regimes.
            wn_a = None if write_nonce_mode == "position_only" else "writeA"
            wn_b = None if write_nonce_mode == "position_only" else "writeB"
            Ma = _mask_rows(puf, dim, 0, pos, mask_std, wn_a)
            Mb = _mask_rows(puf, dim, 0, pos, mask_std, wn_b)

            c_a = secret @ O + Ma        # cache A, secret slot
            c_b = reference @ O + Mb     # cache B, known reference slot

            # (1) Single-cache attack: norm-match on c_a directly. The mask is
            #     present, so this should be at chance for both regimes.
            single_hits += _norm_match_top1(c_a, candidates, torch.zeros(dim), true_idx)

            # (2) Differencing attack: norm-match on (c_a - c_b). For position-only
            #     masks Ma == Mb so the mask cancels and the true secret leaks.
            diff_hits += _norm_match_top1(c_a - c_b, candidates, reference, true_idx)

        results[write_nonce_mode] = {
            "single_cache_top1": single_hits / trials,
            "differenced_top1": diff_hits / trials,
        }
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--mask-std", type=float, default=128.0)
    ap.add_argument("--device-id", type=str, default="device-A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default="experiments/runs/differencing_attack_synthetic.json")
    args = ap.parse_args()

    res = run(args.candidates, args.dim, args.trials, args.mask_std,
              args.device_id, args.seed)
    chance = 1.0 / args.candidates
    summary = {
        "config": vars(args),
        "chance_level": chance,
        "results": res,
        "note": "synthetic K/V-scale vectors; TODO swap in real captured caches.",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"N={args.candidates} candidates, chance={chance:.4f}, std={args.mask_std}, "
          f"dim={args.dim}, trials={args.trials}")
    print(f"{'mask derivation':22s} {'single-cache top-1':>20s} {'differenced top-1':>20s}")
    for mode, r in res.items():
        flag = "  <- BROKEN" if r["differenced_top1"] > 5 * chance else ""
        print(f"{mode:22s} {r['single_cache_top1']:>20.4f} {r['differenced_top1']:>20.4f}{flag}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
