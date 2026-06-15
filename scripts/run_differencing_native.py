"""Real-model differencing attack on same-session affine-masked caches (RQ8/R2).

Companion to run_differencing_attack.py (synthetic), but on captured Qwen3/Llama
layer-0 K/V. Two protected caches are produced in the same session from prompts
that are identical except for the secret slot:

    cache A : prompt(true secret s*)        -> X_A O + M_A
    cache B : prompt(known reference r0)     -> X_B O + M_B   (r0 known to attacker)

The migration adversary differences the two leaked caches. If the additive mask
is a function of position only, M_A == M_B and the mask cancels, leaving
(X_A - X_B) O whose per-token L2 norms (and Gram) are right-orthogonal invariants
(Theorem A) and match the public-model candidate differences X_c - X_B. A fresh
per-write nonce makes M_A, M_B independent, so the difference keeps a fresh mask
and the match collapses to chance (requirement R2).

The protection (orthogonal O + Gaussian mask) is applied here as explicit tensor
ops so the per-write nonce is controlled exactly; O is PUF-seeded per (layer,head).
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

# Reuse the collision-attack helpers (records, prompts, capture, distances).
_cc_path = pathlib.Path(__file__).parent / "run_candidate_secret_collision.py"
_spec = importlib.util.spec_from_file_location("cc", _cc_path)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def _orth(puf, d, layer, head):
    seed = puf.derive_seed(layer=layer, group=head, block=-1, purpose="K_attn")
    g = _seed_generator(seed, torch.device("cpu"))
    a = torch.randn(d, d, generator=g, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def _mask(puf, H, S, d, layer, std, nonce):
    purpose = "affine" if nonce is None else f"affine|wn={nonce}"
    out = torch.empty(H, S, d, dtype=torch.float32)
    for h in range(H):
        for p in range(S):
            seed = puf.derive_seed(layer=layer, group=h, block=p, purpose=purpose)
            g = _seed_generator(seed, torch.device("cpu"))
            out[h, p] = torch.randn(d, generator=g, dtype=torch.float32) * std
    return out


def _protect(kv_layer, puf, layer, std, nonce):
    """Apply X O + M to a captured (k, v) pair at one layer. Returns (k', v')."""
    res = []
    for x in kv_layer:                      # k then v, shape [1, H, S, d]
        x = x.float()
        _, H, S, d = x.shape
        prot = x.clone()
        M = _mask(puf, H, S, d, layer, std, nonce)
        for h in range(H):
            O = _orth(puf, d, layer, h)
            prot[0, h] = x[0, h] @ O + M[h]
        res.append(prot)
    return tuple(res)


@torch.inference_mode()
def run(model, tokenizer, records, layers, std, include_k, include_v, device_id):
    puf = make_puf(device_id)
    keep = set(layers)
    out = {"position_only": [], "per_write_nonce": []}
    for mode, (na, nb) in [("position_only", (None, None)),
                           ("per_write_nonce", ("writeA", "writeB"))]:
        rows = []
        for rec in records:
            ref = rec["candidates"][rec["true_candidate_index"] - 1]  # a known non-secret reference r0
            xa = cc._capture_kv(model, tokenizer, rec["prompt"], keep)                    # plaintext A (s*)
            xb = cc._capture_kv(model, tokenizer, cc._prompt(ref, rec["secret_type"]), keep)  # plaintext B (r0)
            # Two same-session protected caches, then the attacker's difference.
            diff = {}
            for L in layers:
                pa = _protect(xa[L], puf, L, std, na)
                pb = _protect(xb[L], puf, L, std, nb)
                diff[L] = tuple((pa[i] - pb[i]) for i in range(2))
            # Candidate-side plaintext differences X_c - X_B (public model, known r0).
            cand_diffs = []
            for c in rec["candidates"]:
                xc = cc._capture_kv(model, tokenizer, cc._prompt(c, rec["secret_type"]), keep)
                cand_diffs.append({L: tuple((xc[L][i].float() - xb[L][i].float()) for i in range(2))
                                   for L in layers})
            for dm in ("norm_l2", "gram_l2"):
                rank = cc._rank_record(cand_diffs, diff, rec["true_candidate_index"],
                                       layers, include_k, include_v, dm)
                rows.append({"index": rec["index"], "dm": dm, **rank})
        out[mode] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", default="/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--secret-types", default="verification_code")
    ap.add_argument("--mask-std", type=float, default=128.0)
    ap.add_argument("--layers", default="0")
    ap.add_argument("--device-id", default="device_A")
    ap.add_argument("--out", default="experiments/runs/differencing_native_qwen3.json")
    args = ap.parse_args()

    p = pathlib.Path(args.model_cache_dir)
    model_path = p if (p / "config.json").exists() else resolve_snapshot_path(p, snapshot=args.snapshot)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                 trust_remote_code=True, dtype="auto", device_map="auto")
    model.eval()
    model.config._attn_implementation = "eager"
    model.to(torch.float32)

    secret_types = cc._parse_secret_types(args.secret_types)
    records = cc._make_records(tokenizer, args.samples, args.candidates, args.seed, secret_types)
    layers = [0 if t.strip() == "0" else int(t) for t in args.layers.split(",")]

    res = None
    try:
        res = run(model, tokenizer, records, layers, args.mask_std, False, True, args.device_id)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {"config": vars(args), "chance": 1.0 / args.candidates, "modes": {}}
    for mode, rows in res.items():
        summary["modes"][mode] = {}
        for dm in ("norm_l2", "gram_l2"):
            sub = [r for r in rows if r["dm"] == dm]
            summary["modes"][mode][dm] = {
                "samples": len(sub),
                "top1": sum(r["top1"] for r in sub) / max(len(sub), 1),
                "mrr": sum(r["mrr"] for r in sub) / max(len(sub), 1),
            }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"N={args.candidates} chance={summary['chance']:.4f} std={args.mask_std} samples={args.samples}")
    print(f"{'mode':18s} {'distance':10s} {'differenced top-1':>18s} {'mrr':>8s}")
    for mode in ("position_only", "per_write_nonce"):
        for dm in ("norm_l2", "gram_l2"):
            s = summary["modes"][mode][dm]
            flag = "  <- BROKEN" if s["top1"] > 5 * summary["chance"] else ""
            print(f"{mode:18s} {dm:10s} {s['top1']:>18.4f} {s['mrr']:>8.3f}{flag}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
