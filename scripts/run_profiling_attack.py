"""Procrustes profiling attack against PUF-basis protected KV caches.

Setup:
  - Three captured prompts already exist under experiments/runs/capture_*.
  - We use them as the attacker's "chosen-input" pairs: for each capture,
    we have the plaintext KV cache and we can simulate the matching protected
    cache under a known PUF/spec.
  - We fit O_hat per (layer, head, target) via orthogonal Procrustes on the
    aggregated rows, then evaluate
       * basis recovery error  || O_hat O_true^T - I || / sqrt(d)
       * in-sample residual    || X O_hat - Y ||_F / ||Y||_F
       * out-of-sample transfer to a held-out session (different nonce)
  - The defenses we sweep are:
       * P3_KV_givens                (fixed-basis, no layout)
       * P3_KV_givens + row layout   (Exp 5 / Variant P4)
       * P3_KV_givens + session refresh  (Variant P5)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from puf4secure_kvcache.kv_io import load_capture
from puf4secure_kvcache.model_utils import load_model_and_tokenizer, resolve_snapshot_path
from puf4secure_kvcache.profiling import _solve_procrustes
from puf4secure_kvcache.puf_basis import (
    LayoutSpec, ProtectionSpec, _basis_for, protect_kv_cache,
)
from puf4secure_kvcache.puf_sim import make_puf


def _aggregate_pairs(plain_caches, prot_caches, layer, head, sel):
    Xs, Ys = [], []
    for plain, prot in zip(plain_caches, prot_caches):
        p_t = plain[layer][sel][0, head].to(torch.float32)
        q_t = prot[layer][sel][0, head].to(torch.float32)
        n = min(p_t.shape[0], q_t.shape[0])
        Xs.append(p_t[:n].cpu())
        Ys.append(q_t[:n].cpu())
    return torch.cat(Xs, dim=0), torch.cat(Ys, dim=0)


def _basis_truth(puf, kind, head_dim, layer, head, purpose, device):
    return _basis_for(puf, kind, head_dim, layer, head, purpose, device, torch.float32)


def evaluate(scenario_name, spec, puf_fit, puf_target, layer, head, target,
             plain_caches, prot_caches_fit, prot_caches_target):
    """Fit O_hat on prot_caches_fit and test against prot_caches_target."""
    sel = 1 if target == "V" else 0
    X, Y = _aggregate_pairs(plain_caches, prot_caches_fit, layer, head, sel)
    if X.shape[0] < X.shape[1]:
        # underdetermined — pad with zeros so we still get something
        pass
    O_hat = _solve_procrustes(X, Y)

    # truth basis (only meaningful when layout == "none" and same session)
    kind = spec.kind_v if target == "V" else spec.kind_k
    O_true = _basis_truth(puf_fit, kind, X.shape[1], layer, head,
                          target, torch.device("cpu"))
    rec_err = (O_hat @ O_true.T - torch.eye(O_hat.shape[0])).norm().item() / math.sqrt(O_hat.shape[0])

    # in-sample residual
    in_res = (X @ O_hat - Y).norm().item() / max(Y.norm().item(), 1e-12)

    # out-of-sample residual (apply same O_hat to the target session's protected cache)
    X2, Y2 = _aggregate_pairs(plain_caches, prot_caches_target, layer, head, sel)
    out_res = (X2 @ O_hat - Y2).norm().item() / max(Y2.norm().item(), 1e-12)

    return {
        "scenario": scenario_name,
        "layer": layer, "head": head, "target": target,
        "n_rows": int(X.shape[0]),
        "basis_recovery_err": rec_err,
        "in_sample_residual": in_res,
        "transfer_residual": out_res,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("experiments/runs"))
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/profiling_summary.json"))
    args = ap.parse_args()

    capture_dirs = sorted(p for p in args.runs_dir.iterdir()
                          if p.is_dir() and (p / "meta.json").exists())

    # Load plain caches on CPU (Procrustes doesn't need GPU).
    plain_caches = []
    for c in capture_dirs:
        _, kv = load_capture(c)
        plain_caches.append(kv)
    print(f"Loaded {len(plain_caches)} plaintext caches.")

    # ---- scenarios ----
    base_spec_no_layout = ProtectionSpec(
        kind="givens", kind_k="givens", kind_v="hadamard",
        include_k=True, include_v=True,
        layout=LayoutSpec(kind="none"),
    )
    base_spec_row = ProtectionSpec(
        kind="givens", kind_k="givens", kind_v="hadamard",
        include_k=True, include_v=True,
        layout=LayoutSpec(kind="row"),
    )

    # Session A (fit), Session A (target, same)  -- fixed basis, no layout
    puf_A = make_puf("device_A")
    # Session B (different nonce, same device) -- session-refreshed basis
    puf_A_sess2 = make_puf("device_A", session_nonce=b"\x99" * 16)

    scenarios = []

    # (1) fixed basis, no layout, same session -> EXPECTED EASY
    prot_A = [protect_kv_cache(kv, puf_A, base_spec_no_layout) for kv in plain_caches]
    scenarios.append(("S1_fixed_nolayout_same_session", base_spec_no_layout,
                       puf_A, puf_A, prot_A, prot_A))

    # (2) fixed basis, row layout, same session -> tests if row layout breaks Procrustes
    prot_A_row = [protect_kv_cache(kv, puf_A, base_spec_row) for kv in plain_caches]
    scenarios.append(("S2_fixed_rowlayout_same_session", base_spec_row,
                       puf_A, puf_A, prot_A_row, prot_A_row))

    # (3) session refreshed, no layout, fit on A, test on A_sess2 -> tests session nonce
    prot_A2 = [protect_kv_cache(kv, puf_A_sess2, base_spec_no_layout) for kv in plain_caches]
    scenarios.append(("S3_fixed_nolayout_session_refresh", base_spec_no_layout,
                       puf_A, puf_A_sess2, prot_A, prot_A2))

    # ---- per-(layer, head, target) profiling ----
    out = []
    layers_to_probe = [0, 13, 27]
    heads_to_probe = [0, 4]
    targets = ["K", "V"]

    for name, spec, puf_fit, puf_target, prot_fit, prot_target in scenarios:
        for layer in layers_to_probe:
            for head in heads_to_probe:
                for target in targets:
                    res = evaluate(name, spec, puf_fit, puf_target,
                                   layer, head, target, plain_caches,
                                   prot_fit, prot_target)
                    out.append(res)
                    print(f"{name:38s} L{layer:>2} h{head} {target} "
                          f"n={res['n_rows']:3d} rec_err={res['basis_recovery_err']:.3f} "
                          f"in_res={res['in_sample_residual']:.3f} "
                          f"transfer={res['transfer_residual']:.3f}")

    # Aggregate
    agg = {}
    for r in out:
        key = r["scenario"]
        agg.setdefault(key, []).append(r)
    summary = {}
    for k, rs in agg.items():
        rec = [r["basis_recovery_err"] for r in rs]
        ins = [r["in_sample_residual"] for r in rs]
        outr = [r["transfer_residual"] for r in rs]
        summary[k] = {
            "mean_basis_recovery_err": sum(rec) / len(rec),
            "mean_in_sample_residual": sum(ins) / len(ins),
            "mean_transfer_residual": sum(outr) / len(outr),
            "n_units": len(rs),
        }

    print("\n=== Aggregate ===")
    print(json.dumps(summary, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"per_unit": out, "summary": summary}, fh, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
