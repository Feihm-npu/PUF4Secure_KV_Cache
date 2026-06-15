"""Rigorous confidentiality analysis: theory-matched advantage, guessing entropy.

Turns the confidentiality claim from "top-1 on 32 candidates" into (a) a measured
IND-MIG advantage that tracks the proven Gaussian-mechanism bound of Theorem B
across the mask scale sigma, and (b) an information-theoretic guessing-entropy
metric. Captures plaintext caches once, then evaluates every sigma and attacker
tier at the tensor level.

Three attacker tiers (per sigma), all on the same protected layer-0 V-cache
C~ = C O + M, M ~ N(0, sigma^2):
  T0 migration (realizable): has only invariants (norm / Gram); no basis O.
       This is the post-compromise migration adversary of the threat model.
  T1 basis-equipped (optimal): additionally knows O (models a same-session-KPA
       adversary that learned the basis, then migrated). Its likelihood-optimal
       distinguisher ||C~ - c O|| realizes the worst case the bound assumes.
  BOUND: analytic Gaussian-mechanism advantage. For two candidates the exact
       optimal advantage is the total variation 2*Phi(Delta/2sigma)-1, upper-
       bounded by Delta/(sigma*sqrt(2pi)) (Theorem B), Delta=||C_0-C_1||_F.

Metrics: 2-candidate IND advantage |2 acc - 1| (matches Theorem B directly),
N-candidate top-1, and guessing entropy GE = mean rank of the true secret
(1 = fully leaked; (N+1)/2 = chance). Also the fp32 de-mask round-trip error
(scales ~ sigma*u, Theorem E).
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
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


def _norm_sig(x):           # per-token L2 norms, [H, S]
    return x.squeeze(0).norm(dim=-1)


def _gram_sig(x):           # per-head Gram XX^T flattened, [H, S*S]
    xh = x.squeeze(0)
    g = xh @ xh.transpose(-1, -2)
    return g.reshape(g.shape[0], -1)


def _rank_of_true(scores, true_idx):
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    return order.index(true_idx) + 1


@torch.inference_mode()
def analyze(model, tokenizer, records, layer, sigmas, device_id, seed0):
    puf = make_puf(device_id)
    keep = {layer}
    # Plaintext layer-0 V per candidate per record (captured once).
    bank = []   # bank[r] = list over candidates of V tensor [1,H,S,D]
    for rec in records:
        cands = []
        for c in rec["candidates"]:
            kv = cc._capture_kv(model, tokenizer, cc._prompt(c, rec["secret_type"]), keep)
            cands.append(kv[layer][1].double())   # V, [1,H,S,D]
        bank.append(cands)
    d = bank[0][0].shape[-1]
    H = bank[0][0].shape[1]

    # Fixed per-head orthogonal basis (session basis).
    O = torch.stack([_orth(puf, d, layer, h) for h in range(H)])   # [H,d,d]

    def rotate(V):                       # apply per-head O: [1,H,S,d]
        return torch.einsum("bhsd,hde->bhse", V, O)

    # Precompute candidate signatures in plaintext space (T0) and rotated (T1).
    out = {"sigmas": [], "by_sigma": {}}
    g_noise = torch.Generator().manual_seed(seed0)
    for sigma in sigmas:
        t0_norm_rank, t0_gram_rank, t1_rank = [], [], []
        two_cand_correct, two_cand_n = 0, 0
        bound_terms, tv_terms, err_terms = [], [], []
        for r, rec in enumerate(records):
            cands = bank[r]
            true_idx = rec["true_candidate_index"]
            Vtrue = cands[true_idx]
            Rtrue = rotate(Vtrue)
            if sigma == 0.0:
                M = torch.zeros_like(Rtrue)
            else:
                M = torch.randn(Rtrue.shape, generator=g_noise, dtype=torch.float64) * sigma
            leaked = Rtrue + M                              # C~ = C O + M
            # fp32 de-mask round-trip error (Theorem E): store fp32, subtract.
            recov = (leaked.float() - M.float()).double()
            err_terms.append(float((recov - Rtrue).abs().max() / (Rtrue.abs().max() + 1e-12)))

            # T0 migration: invariant matchers (no O).
            ln = _norm_sig(leaked)
            lg = _gram_sig(leaked)
            sc_norm = [float((_norm_sig(c) - ln).norm()) for c in cands]
            sc_gram = [float((_gram_sig(c) - lg).norm()) for c in cands]
            t0_norm_rank.append(_rank_of_true(sc_norm, true_idx))
            t0_gram_rank.append(_rank_of_true(sc_gram, true_idx))
            # T1 basis-equipped: ||C~ - c O|| (optimal Gaussian distinguisher).
            sc_t1 = [float((leaked - rotate(c)).norm()) for c in cands]
            t1_rank.append(_rank_of_true(sc_t1, true_idx))

            # 2-candidate IND advantage vs analytic bound, over ALL decoy pairs
            # (true vs each decoy) so the empirical advantage is low-variance and
            # can be compared to the Gaussian-mechanism bound tightly.
            Rdec = {j: rotate(cands[j]) for j in range(len(cands))}
            for decoy in range(len(cands)):
                if decoy == true_idx:
                    continue
                Delta = float((Vtrue - cands[decoy]).norm())  # ||C0-C1||_F (O preserves norm)
                bound_terms.append(Delta / (sigma * math.sqrt(2 * math.pi)) if sigma > 0 else 1.0)
                tv_terms.append((2 * 0.5 * (1 + math.erf(Delta / (2 * sigma * math.sqrt(2)))) - 1)
                                if sigma > 0 else 1.0)
                for b, true_c in ((0, true_idx), (1, decoy)):
                    Rb = Rdec[true_c]
                    Mb = (torch.randn(Rb.shape, generator=g_noise, dtype=torch.float64) * sigma
                          if sigma > 0 else torch.zeros_like(Rb))
                    cb = Rb + Mb
                    d_true = float((cb - Rdec[true_idx]).norm())
                    d_decoy = float((cb - Rdec[decoy]).norm())
                    pred = true_idx if d_true <= d_decoy else decoy
                    two_cand_correct += int(pred == true_c)
                    two_cand_n += 1

        N = len(records[0]["candidates"])
        def ge(ranks): return sum(ranks) / len(ranks)
        out["sigmas"].append(sigma)
        out["by_sigma"][str(sigma)] = {
            "N": N,
            "t0_norm_top1": sum(1 for x in t0_norm_rank if x == 1) / len(t0_norm_rank),
            "t0_gram_top1": sum(1 for x in t0_gram_rank if x == 1) / len(t0_gram_rank),
            "t0_norm_GE": ge(t0_norm_rank), "t0_gram_GE": ge(t0_gram_rank),
            "t1_top1": sum(1 for x in t1_rank if x == 1) / len(t1_rank),
            "t1_GE": ge(t1_rank),
            "two_cand_adv": abs(2 * two_cand_correct / two_cand_n - 1),
            "analytic_bound": min(1.0, sum(bound_terms) / len(bound_terms)),
            "exact_TV": sum(tv_terms) / len(tv_terms),
            "fp32_roundtrip_relerr": sum(err_terms) / len(err_terms),
            "chance_top1": 1.0 / N, "chance_GE": (N + 1) / 2,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", default="/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--secret-types", default="verification_code")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--sigmas", default="0,8,16,32,64,128,256,512")
    ap.add_argument("--device-id", default="device_A")
    ap.add_argument("--tag", default="qwen3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = pathlib.Path(args.model_cache_dir)
    model_path = p if (p / "config.json").exists() else resolve_snapshot_path(p, snapshot=args.snapshot)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                 trust_remote_code=True, dtype="auto", device_map="auto")
    model.eval(); model.config._attn_implementation = "eager"; model.to(torch.float32)

    secret_types = cc._parse_secret_types(args.secret_types)
    records = cc._make_records(tokenizer, args.samples, args.candidates, args.seed, secret_types)
    sigmas = [float(s) for s in args.sigmas.split(",")]
    try:
        res = analyze(model, tokenizer, records, args.layer, sigmas, args.device_id, args.seed)
    finally:
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    res["config"] = vars(args)
    out = args.out or f"experiments/runs/confidentiality_analysis_{args.tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)

    print(f"=== Confidentiality analysis ({args.tag}, N={args.candidates}, samples={args.samples}) ===")
    print(f"{'sigma':>6} {'T0normGE':>9} {'T0gramGE':>9} {'T1 GE':>7} {'2candAdv':>9} {'bound':>7} {'TV':>6} {'fp32err':>9}")
    for s in sigmas:
        r = res["by_sigma"][str(s)]
        print(f"{s:>6.0f} {r['t0_norm_GE']:>9.2f} {r['t0_gram_GE']:>9.2f} {r['t1_GE']:>7.2f} "
              f"{r['two_cand_adv']:>9.3f} {r['analytic_bound']:>7.3f} {r['exact_TV']:>6.3f} {r['fp32_roundtrip_relerr']:>9.1e}")
    print(f"chance: GE={(args.candidates+1)/2:.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
