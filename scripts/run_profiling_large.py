"""Large-scale Procrustes profiling attack (Plan v1, task 3).

Goal: stress-test the row/block layout and session-refresh defenses with
profiling sample sizes far exceeding head_dim (128 for Qwen3-0.6B).

For each scenario we generate N random-token prompts, run a forward pass to
obtain plaintext KV per prompt, apply the protection under a fixed PUF, and
accumulate Procrustes statistics M = sum X^T Y per (layer, head, target).
Then we recover O_hat = U V^T from svd(M).

Three attacker strategies are evaluated:

  (a) naive             - pair row i of plaintext with row i of protected.
  (b) sort-by-norm      - per prompt, sort rows of X and Y by L2 norm before
                          pairing. Because O is orthogonal, row norms are
                          preserved, so this is the optimal greedy alignment
                          under an unknown but PUF-fixed row permutation.
  (c) sort-by-norm-block - same as (b) but restricted within each
                          block_size chunk (for layout=block).

In addition to residuals, we run a downstream V-inversion attack on the
held-out (real-prompt) captures after stripping O_hat from their protected
cache. This is the operational metric: "did the recovered basis actually
let the attacker re-run the original NDSS attack?".

Output: experiments/runs/profiling_large_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from puf4secure_kvcache.attacks import invert_v_layer0
from puf4secure_kvcache.kv_io import load_capture
from puf4secure_kvcache.metrics import token_accuracy
from puf4secure_kvcache.model_utils import (
    forward_with_kv,
    kv_to_list,
    load_model_and_tokenizer,
    resolve_snapshot_path,
)
from puf4secure_kvcache.profiling import _solve_procrustes
from puf4secure_kvcache.puf_basis import (
    LayoutSpec,
    ProtectionSpec,
    _basis_for,
    protect_kv_cache,
)
from puf4secure_kvcache.puf_sim import make_puf


# ---------------------------------------------------------------------------
# Prompt source: random token IDs of fixed length. We avoid tokenizer decode
# because we only care about the model's internal KV; the attacker controls
# the model input regardless of human-readable text.
# ---------------------------------------------------------------------------

def random_prompt_ids(tokenizer, n_prompts: int, seq_len: int, seed: int = 0):
    """Yield (input_ids[1, S]) random sequences inside the safe vocab range."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Avoid special tokens by sampling from the lower 80% of the vocab.
    vocab_size = max(tokenizer.vocab_size, 32000)
    cap = int(vocab_size * 0.8)
    for _ in range(n_prompts):
        ids = torch.randint(low=10, high=cap, size=(1, seq_len), generator=g)
        yield ids


# ---------------------------------------------------------------------------
# Profiling accumulator
# ---------------------------------------------------------------------------

class ProcrustesAccumulator:
    """Online accumulation of X^T Y for every (layer, head, target).

    Memory: 28 layers x 8 kv_heads x 2 targets x 128 x 128 fp32 ~= 29 MB.
    """

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 alignment: str = "naive", block_size: int = 8,
                 device: torch.device = torch.device("cpu")):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.alignment = alignment
        self.block_size = block_size
        self.device = device
        self.M = {}  # (layer, head, target) -> [d, d] fp32
        self.n_rows = {}

    def _key(self, layer, head, target):
        return (layer, head, target)

    def _align(self, X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return permuted (X', Y') so that row i of X' is paired with row i of Y'.

        - naive            : identity (X, Y).
        - sort_by_norm     : sort each by L2 row-norm.
        - sort_by_norm_block: sort within each block of size self.block_size.
        """
        if self.alignment == "naive":
            return X, Y
        if self.alignment == "sort_by_norm":
            nx = X.norm(dim=1)
            ny = Y.norm(dim=1)
            return X[nx.argsort()], Y[ny.argsort()]
        if self.alignment == "sort_by_norm_block":
            S, d = X.shape
            bs = self.block_size
            X_out = X.clone()
            Y_out = Y.clone()
            for s in range(0, S, bs):
                e = min(s + bs, S)
                xb = X[s:e]
                yb = Y[s:e]
                X_out[s:e] = xb[xb.norm(dim=1).argsort()]
                Y_out[s:e] = yb[yb.norm(dim=1).argsort()]
            return X_out, Y_out
        raise ValueError(f"Unknown alignment: {self.alignment!r}")

    def add(self, plain_kv, protected_kv):
        for layer in range(self.num_layers):
            for tgt_idx, target in [(0, "K"), (1, "V")]:
                p = plain_kv[layer][tgt_idx]       # [1, H, S, d]
                q = protected_kv[layer][tgt_idx]
                for h in range(p.shape[1]):
                    X = p[0, h].to(torch.float32).to(self.device)
                    Y = q[0, h].to(torch.float32).to(self.device)
                    X, Y = self._align(X, Y)
                    key = self._key(layer, h, target)
                    if key not in self.M:
                        self.M[key] = torch.zeros(X.shape[1], X.shape[1],
                                                  dtype=torch.float32,
                                                  device=self.device)
                        self.n_rows[key] = 0
                    self.M[key] += X.T @ Y
                    self.n_rows[key] += X.shape[0]

    def solve(self) -> dict:
        out = {}
        for key, M in self.M.items():
            U, _, Vh = torch.linalg.svd(M, full_matrices=False)
            out[key] = (U @ Vh).contiguous()
        return out


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def residual_on_pair(X: torch.Tensor, Y: torch.Tensor, O_hat: torch.Tensor,
                     accumulator: ProcrustesAccumulator) -> float:
    X, Y = accumulator._align(X, Y)
    diff = (X @ O_hat - Y).norm().item()
    return diff / max(Y.norm().item(), 1e-12)


def evaluate_against_real_captures(O_hats: dict, real_captures, spec, puf, model,
                                   alignment: str = "naive", block_size: int = 8):
    """For each real capture: protect, strip O_hat from each head, run V-inversion.

    Returns {capture_idx: {transfer_residual_mean, v_inversion_top1}}.
    """
    device = next(model.parameters()).device
    out = []
    for cap_idx, (meta, kv) in enumerate(real_captures):
        kv_dev = [(k.to(device), v.to(device)) for k, v in kv]
        protected = protect_kv_cache(kv_dev, puf, spec)
        # Strip per-head O_hat.T  (attacker recovers X = Y @ O_hat^T)
        stripped = []
        all_res = []
        for layer in range(len(protected)):
            k_p, v_p = protected[layer]
            k_s = torch.empty_like(k_p)
            v_s = torch.empty_like(v_p)
            for h in range(k_p.shape[1]):
                for target, src, dst in [("K", k_p, k_s), ("V", v_p, v_s)]:
                    Y = src[0, h].to(torch.float32)
                    O_hat = O_hats.get((layer, h, target))
                    if O_hat is None:
                        rec = Y
                    else:
                        rec = Y @ O_hat.T.to(Y.device)
                    dst[0, h] = rec.to(src.dtype)
                    # transfer residual: compare rec to plaintext at same position
                    if O_hat is not None:
                        X = kv_dev[layer][0 if target == "K" else 1][0, h].to(torch.float32)
                        # Account for alignment used during fit when comparing
                        if alignment != "naive":
                            X_a, _ = ProcrustesAccumulator(1, 1, 1, alignment=alignment,
                                                          block_size=block_size).\
                                _align(X, X)
                            res = (rec - X_a).norm().item() / max(X.norm().item(), 1e-12)
                        else:
                            res = (rec - X).norm().item() / max(X.norm().item(), 1e-12)
                        all_res.append(res)
            stripped.append((k_s, v_s))
        # Downstream V-inversion on stripped cache (uses layer 0, V).
        target_ids = torch.tensor(meta["input_ids"], dtype=torch.long)
        inv = invert_v_layer0(model, stripped[0][1], target_ids)
        out.append({
            "capture": cap_idx,
            "transfer_residual_mean": sum(all_res) / max(len(all_res), 1),
            "v_inversion_top1": inv.token_accuracy_top1,
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--runs-dir", type=Path, default=Path("experiments/runs"))
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/profiling_large_summary.json"))
    args = ap.parse_args()

    model, tokenizer = load_model_and_tokenizer(resolve_snapshot_path())
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)
    print(f"Model: layers={num_layers}, kv_heads={num_kv_heads}, head_dim={head_dim}")

    # Load 3 real captures (held-out evaluation set)
    real_capture_dirs = sorted(p for p in args.runs_dir.iterdir()
                                if p.is_dir() and (p / "meta.json").exists()
                                and p.name.startswith("capture_"))
    real_captures = [load_capture(p) for p in real_capture_dirs]
    print(f"Loaded {len(real_captures)} held-out real captures.")

    # ---- specs ----
    spec_nolayout = ProtectionSpec(kind="givens", include_k=True, include_v=True,
                                   layout=LayoutSpec(kind="none"))
    spec_row = ProtectionSpec(kind="givens", include_k=True, include_v=True,
                              layout=LayoutSpec(kind="row"))
    spec_block = ProtectionSpec(kind="givens", include_k=True, include_v=True,
                                layout=LayoutSpec(kind="block", block_size=8))

    puf_A = make_puf("device_A")
    puf_A_sess2 = make_puf("device_A", session_nonce=b"\x99" * 16)

    scenarios = [
        # (name, spec_for_fit, puf_fit, puf_target, alignment)
        ("S1_large_nolayout_naive",
         spec_nolayout, puf_A, puf_A, "naive"),
        ("S2_large_rowlayout_naive",
         spec_row, puf_A, puf_A, "naive"),
        ("S2b_large_rowlayout_sortnorm",
         spec_row, puf_A, puf_A, "sort_by_norm"),
        ("S3_large_blocklayout_sortnormblock",
         spec_block, puf_A, puf_A, "sort_by_norm_block"),
        ("S4_large_session_refresh_naive",
         spec_nolayout, puf_A, puf_A_sess2, "naive"),
    ]

    summary = {"settings": {"n_prompts": args.n_prompts, "seq_len": args.seq_len,
                            "head_dim": head_dim}, "scenarios": {}}

    for name, spec_fit, puf_fit, puf_target, alignment in scenarios:
        print(f"\n=== {name} (alignment={alignment}) ===", flush=True)
        t0 = time.perf_counter()
        accum = ProcrustesAccumulator(num_layers, num_kv_heads, head_dim,
                                      alignment=alignment, block_size=8,
                                      device=torch.device("cpu"))

        # Stream: generate prompts, forward, protect, accumulate, discard.
        torch.manual_seed(args.seed)
        for i, ids in enumerate(random_prompt_ids(tokenizer, args.n_prompts,
                                                  args.seq_len, seed=args.seed)):
            ids = ids.to(device)
            with torch.inference_mode():
                out = model(input_ids=ids, use_cache=True)
            kv = kv_to_list(out.past_key_values)
            # Move to CPU before protect/accumulate to save GPU memory.
            kv_cpu = [(k.cpu(), v.cpu()) for k, v in kv]
            del out, kv
            prot = protect_kv_cache(kv_cpu, puf_fit, spec_fit)
            accum.add(kv_cpu, prot)
            del kv_cpu, prot
            if (i + 1) % 100 == 0:
                print(f"  ...{i+1}/{args.n_prompts} prompts processed "
                      f"({time.perf_counter()-t0:.1f}s)", flush=True)

        # Solve
        O_hats = accum.solve()
        t_fit = time.perf_counter() - t0
        print(f"  fit time: {t_fit:.1f}s, rows per unit: "
              f"{next(iter(accum.n_rows.values()))}")

        # Compare against true basis when meaningful (no session refresh, no layout)
        # Single fresh prompt shared across all (layer, head, target) units.
        ids = next(random_prompt_ids(tokenizer, 1, args.seq_len,
                                      seed=args.seed + 99999))
        ids = ids.to(device)
        with torch.inference_mode():
            test_out = model(input_ids=ids, use_cache=True)
        test_kv_cpu = [(k.cpu(), v.cpu()) for k, v in kv_to_list(test_out.past_key_values)]
        test_prot = protect_kv_cache(test_kv_cpu, puf_target, spec_fit)
        del test_out

        residual_stats = []
        rec_err_stats = []
        for (layer, h, target), O_hat in O_hats.items():
            sel = 1 if target == "V" else 0
            X = test_kv_cpu[layer][sel][0, h].to(torch.float32)
            Y = test_prot[layer][sel][0, h].to(torch.float32)
            X_a, Y_a = accum._align(X, Y)
            res = (X_a @ O_hat - Y_a).norm().item() / max(Y_a.norm().item(), 1e-12)
            residual_stats.append(res)
            kind = spec_fit.kind_v if target == "V" else spec_fit.kind_k
            O_true = _basis_for(puf_fit, kind, X.shape[1], layer, h, target,
                                torch.device("cpu"), torch.float32)
            rec = (O_hat @ O_true.T - torch.eye(O_hat.shape[0])).norm().item() / math.sqrt(O_hat.shape[0])
            rec_err_stats.append(rec)

        # Downstream V-inversion on real captures
        downstream = evaluate_against_real_captures(O_hats, real_captures, spec_fit,
                                                    puf_target, model,
                                                    alignment=alignment)
        v_inv_top1 = sum(d["v_inversion_top1"] for d in downstream) / max(len(downstream), 1)
        transfer_res = sum(d["transfer_residual_mean"] for d in downstream) / max(len(downstream), 1)

        summary["scenarios"][name] = {
            "alignment": alignment,
            "spec": {
                "kind": spec_fit.kind_k, "layout": spec_fit.layout.kind,
            },
            "puf_fit": puf_fit.session_nonce.hex(),
            "puf_target": puf_target.session_nonce.hex(),
            "mean_fresh_prompt_residual": sum(residual_stats) / len(residual_stats),
            "mean_basis_recovery_err": sum(rec_err_stats) / len(rec_err_stats),
            "downstream_v_inversion_top1_mean": v_inv_top1,
            "downstream_transfer_residual_mean": transfer_res,
            "n_units": len(O_hats),
            "rows_per_unit": next(iter(accum.n_rows.values())),
            "fit_time_s": t_fit,
        }
        print(f"  mean_fresh_residual={summary['scenarios'][name]['mean_fresh_prompt_residual']:.3f}")
        print(f"  mean_basis_rec_err={summary['scenarios'][name]['mean_basis_recovery_err']:.3f}")
        print(f"  downstream V-inv top1 (real captures, mean)={v_inv_top1:.3f}")
        print(f"  downstream transfer residual (real captures, mean)={transfer_res:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
