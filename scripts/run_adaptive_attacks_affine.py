"""Adaptive attack suite against the affine PUF mask (defines the threat boundary).

The migration attacker (one cache, no known plaintext, no basis) is at chance
(see the confidentiality analysis). This script measures two *stronger* attackers
to state the scope precisely:

  A1 same-session KPA: the attacker holds aligned (plaintext, protected) pairs from
     the SAME session. Differencing across known prompts cancels the additive mask,
     so (X^k - X^1) O is observable and O is recovered by orthogonal Procrustes;
     the mask M is then recovered from one known pair. Stripping O and M from a
     held-out cache returns plaintext. => affine, like orthogonal, is broken under
     same-session KPA; its protection is only for the migration attacker. Under a
     refreshed session nonce the recovered (O, M) do not transfer.

  A2 averaging: the attacker observes the SAME secret cached n times under
     independent per-write masks (R2 freshness). Averaging cancels the zero-mean
     mask, recovering X O after ~ (sigma/Delta)^2 observations, after which the
     Theorem-A invariant leak applies. We measure top-1 vs n. This requires the
     unusual capability of hundreds of re-cachings of identical content.

Together with the differencing attack (run_differencing_native.py) these bound
exactly which adversaries the affine mask does and does not stop.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from puf4secure_kvcache.model_utils import resolve_snapshot_path
from puf4secure_kvcache.puf_sim import make_puf
from puf4secure_kvcache.puf_attention import _seed_generator

_cc_path = pathlib.Path(__file__).parent / "run_candidate_secret_collision.py"
_spec = importlib.util.spec_from_file_location("cc", _cc_path)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def _orth(puf, d, layer, head):
    seed = puf.derive_seed(layer=layer, group=head, block=-1, purpose="K_attn")
    g = _seed_generator(seed, torch.device("cpu"))
    a = torch.randn(d, d, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def _gen_mask(shape, gen, sigma):
    return torch.randn(shape, generator=gen, dtype=torch.float64) * sigma


def kpa_affine(model, tok, secret_type, known_secrets, target_secret, layer, sigma, device_id, seed0):
    """Recover (O, M) from len(known_secrets) same-session pairs; report strip
    residual on a held-out cache for same-session vs refreshed-nonce targets."""
    puf = make_puf(device_id)
    keep = {layer}
    n_known = len(known_secrets)
    Xk = [cc._capture_kv(model, tok, cc._prompt(c, secret_type), keep)[layer][1].double()
          for c in known_secrets]                           # each [1,H,S,D]
    H, S, d = Xk[0].shape[1], Xk[0].shape[2], Xk[0].shape[3]
    O = torch.stack([_orth(puf, d, layer, h) for h in range(H)])
    rot = lambda V: torch.einsum("bhsd,hde->bhse", V, O)
    g = torch.Generator().manual_seed(seed0)
    # one shared session mask M (same nonce/position) across the known set
    M = _gen_mask(Xk[0].shape, g, sigma)
    Ck = [rot(X) + M for X in Xk]                            # protected known caches

    # recover O per head via Procrustes on differences (mask cancels)
    Ohat = torch.empty(H, d, d, dtype=torch.float64)
    for h in range(H):
        Yrows, Drows = [], []
        for k in range(1, n_known):
            Yrows.append((Xk[k][0, h] - Xk[0][0, h]))       # [S,d] plaintext diff
            Drows.append((Ck[k][0, h] - Ck[0][0, h]))       # [S,d] protected diff
        Y = torch.cat(Yrows, 0); D = torch.cat(Drows, 0)     # [(k-1)S, d]
        U, _, Vt = torch.linalg.svd(Y.T @ D)                 # Procrustes Y O ~ D
        Ohat[h] = U @ Vt
    roth = lambda V: torch.einsum("bhsd,hde->bhse", V, Ohat)
    Mhat = Ck[0] - roth(Xk[0])                               # recover mask

    # held-out target: a fresh secret, same session (same O, same M)
    Xt = cc._capture_kv(model, tok, cc._prompt(target_secret, secret_type), keep)[layer][1].double()
    Ct_same = rot(Xt) + M
    # refreshed nonce: independent mask M'
    Mp = _gen_mask(Xt.shape, torch.Generator().manual_seed(seed0 + 777), sigma)
    Ct_refresh = rot(Xt) + Mp

    def strip_resid(Ct):
        Xhat = torch.einsum("bhsd,hed->bhse", (Ct - Mhat), Ohat)   # (Ct-Mhat) Ohat^T
        return float((Xhat - Xt).norm() / (Xt.norm() + 1e-12))

    return {"n_known": n_known,
            "same_session_residual": strip_resid(Ct_same),
            "refresh_residual": strip_resid(Ct_refresh)}


def averaging(model, tok, records, layer, sigma, device_id, n_obs_list, seed0):
    """Average n independent-mask observations of the same secret, then invariant-
    match against candidates. top-1 vs n. The mean of n iid N(0,sigma^2) masks is
    N(0, sigma^2/n), so we sample the reduced-variance noise directly (exact in
    distribution) and can sweep n up to ~1e6 cheaply."""
    puf = make_puf(device_id); keep = {layer}
    rng = torch.Generator().manual_seed(seed0)
    per_n = {n: [] for n in n_obs_list}
    for rec in records:
        cands = [cc._capture_kv(model, tok, cc._prompt(c, rec["secret_type"]), keep)[layer][1].double()
                 for c in rec["candidates"]]
        true_idx = rec["true_candidate_index"]
        H, d = cands[0].shape[1], cands[0].shape[3]
        O = torch.stack([_orth(puf, d, layer, h) for h in range(H)])
        Rtrue = torch.einsum("bhsd,hde->bhse", cands[true_idx], O)
        cand_norm = [c.squeeze(0).norm(dim=-1) for c in cands]    # invariant sigs
        for n in n_obs_list:
            eff = sigma / (n ** 0.5)                      # std of the averaged mask
            avg = Rtrue + torch.randn(Rtrue.shape, generator=rng, dtype=torch.float64) * eff
            an = avg.squeeze(0).norm(dim=-1)
            scores = [float((cn - an).norm()) for cn in cand_norm]
            pred = min(range(len(scores)), key=lambda j: scores[j])
            per_n[n].append(int(pred == true_idx))
    return {str(n): sum(v) / len(v) for n, v in per_n.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", default="/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--secret-types", default="verification_code")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=128.0)
    ap.add_argument("--device-id", default="device_A")
    ap.add_argument("--tag", default="qwen3")
    args = ap.parse_args()

    p = pathlib.Path(args.model_cache_dir)
    mp = p if (p / "config.json").exists() else resolve_snapshot_path(p, snapshot=args.snapshot)
    tok = AutoTokenizer.from_pretrained(mp, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mp, local_files_only=True, trust_remote_code=True,
                                                 dtype="auto", device_map="auto")
    model.eval(); model.config._attn_implementation = "eager"; model.to(torch.float32)
    secret_types = cc._parse_secret_types(args.secret_types)
    records = cc._make_records(tok, args.samples, args.candidates, args.seed, secret_types)

    # KPA pool: 65 distinct same-length prompts (up to 64 known + 1 held-out target).
    kpa_pool = cc._make_records(tok, 1, 65, args.seed + 1, secret_types)[0]
    kpa_secrets, kpa_type = kpa_pool["candidates"], kpa_pool["secret_type"]
    try:
        kpa = [kpa_affine(model, tok, kpa_type, kpa_secrets[:k], kpa_secrets[64],
                          args.layer, args.sigma, args.device_id, args.seed)
               for k in (4, 16, 64)]
        avg = averaging(model, tok, records, args.layer, args.sigma, args.device_id,
                        [1, 256, 4096, 65536, 262144, 1048576], args.seed)
    finally:
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    res = {"config": vars(args), "kpa_affine": kpa, "averaging_top1_by_nobs": avg,
           "chance": 1.0 / args.candidates}
    out = f"experiments/runs/adaptive_attacks_affine_{args.tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, default=str)

    print(f"=== Adaptive attacks vs affine ({args.tag}, sigma={args.sigma}) ===")
    print("A1 same-session KPA (recover O+M, strip residual; ~0 = broken):")
    for k in kpa:
        print(f"  n_known={k['n_known']:>3}: same-session resid={k['same_session_residual']:.4f}  "
              f"refresh resid={k['refresh_residual']:.4f}")
    print(f"A2 averaging attack top-1 vs #observations (chance={1.0/args.candidates:.3f}):")
    for n, t in avg.items():
        print(f"  n_obs={n:>4}: top-1={t:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
