"""Profiling attack against Level-3 native PUF attention caches.

Unlike `run_profiling_large.py`, this script does not call `protect_kv_cache`.
It dumps the K/V cache produced by `puf_attention.install_puf_attention`, then
fits an orthogonal Procrustes map from plaintext cache rows to the native rotated
cache rows.

Current scope: Level-3 wrapper with no cache layout. This tests the two most
important native-cache facts across supported attention families:

  * fixed same-session basis is profileable;
  * a basis learned in one session does not transfer to another session.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForCausalLM, AutoTokenizer

from puf4secure_kvcache.attacks import invert_v_layer0
from puf4secure_kvcache.model_utils import kv_to_list, resolve_snapshot_path
from puf4secure_kvcache.puf_attention import install_puf_attention, uninstall_puf_attention
from puf4secure_kvcache.puf_basis import _basis_for
from puf4secure_kvcache.puf_sim import make_puf


DEFAULT_MODEL_CACHE = Path("/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")
DEFAULT_PROMPTS_FILE = Path("experiments/prompts/synthetic_privacy_prompts.txt")


def _model_path(path: str | Path, snapshot: str | None) -> Path:
    path = Path(path)
    if (path / "config.json").exists():
        return path
    return resolve_snapshot_path(path, snapshot=snapshot)


def _cuda_summary() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    devices = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        devices.append({
            "logical_index": idx,
            "name": props.name,
            "total_mem_mib": props.total_memory / (1024 ** 2),
            "allocated_mib": torch.cuda.memory_allocated(idx) / (1024 ** 2),
            "reserved_mib": torch.cuda.memory_reserved(idx) / (1024 ** 2),
        })
    return {
        "available": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "devices": devices,
    }


def random_prompt_ids(tokenizer, n_prompts: int, seq_len: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    vocab_size = max(tokenizer.vocab_size, 32000)
    cap = int(vocab_size * 0.8)
    for _ in range(n_prompts):
        yield torch.randint(low=10, high=cap, size=(1, seq_len), generator=g)


class NativeAccumulator:
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 alignment: str = "naive"):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.alignment = alignment
        self.M: dict[tuple[int, int, str], torch.Tensor] = {}
        self.n_rows: dict[tuple[int, int, str], int] = {}

    def _hungarian_profile(self, X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if X.shape[0] <= 1:
            return X, Y
        Dx = torch.cdist(X.float().cpu(), X.float().cpu())
        Dy = torch.cdist(Y.float().cpu(), Y.float().cpu())
        Px = Dx.sort(dim=1).values
        Py = Dy.sort(dim=1).values
        cost = torch.cdist(Px, Py).numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        perm_y = torch.empty(X.shape[0], dtype=torch.long)
        perm_y[torch.as_tensor(row_ind, dtype=torch.long)] = torch.as_tensor(col_ind, dtype=torch.long)
        return X, Y[perm_y.to(Y.device)]

    def _align(self, X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.alignment == "naive":
            return X, Y
        if self.alignment == "hungarian":
            return self._hungarian_profile(X, Y)
        raise ValueError(f"unknown alignment: {self.alignment!r}")

    def add(self, plain_kv, wrapped_kv) -> None:
        for layer in range(self.num_layers):
            for target_idx, target in [(0, "K"), (1, "V")]:
                p = plain_kv[layer][target_idx]
                w = wrapped_kv[layer][target_idx]
                for h in range(p.shape[1]):
                    X = p[0, h].to(torch.float32).cpu()
                    Y = w[0, h].to(torch.float32).cpu()
                    X, Y = self._align(X, Y)
                    key = (layer, h, target)
                    if key not in self.M:
                        self.M[key] = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float32)
                        self.n_rows[key] = 0
                    self.M[key] += X.T @ Y
                    self.n_rows[key] += X.shape[0]

@torch.inference_mode()
def forward_kv(model, input_ids: torch.Tensor):
    device = next(model.parameters()).device
    out = model(input_ids=input_ids.to(device), use_cache=True)
    kv = [(k.detach().cpu(), v.detach().cpu()) for k, v in kv_to_list(out.past_key_values)]
    del out
    return kv


def solve_from_cross_cov(M: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return (U @ Vh).contiguous()


def solve_all(accum: NativeAccumulator) -> dict[tuple[int, int, str], torch.Tensor]:
    return {key: solve_from_cross_cov(M) for key, M in accum.M.items()}


def residual_for_pair(plain_kv, wrapped_kv, O_hats: dict[tuple[int, int, str], torch.Tensor]) -> tuple[float, float]:
    residuals = []
    transfer = []
    for layer in range(len(plain_kv)):
        for target_idx, target in [(0, "K"), (1, "V")]:
            X_all = plain_kv[layer][target_idx]
            Y_all = wrapped_kv[layer][target_idx]
            for h in range(X_all.shape[1]):
                O_hat = O_hats[(layer, h, target)]
                X = X_all[0, h].to(torch.float32)
                Y = Y_all[0, h].to(torch.float32)
                pred = X @ O_hat
                residuals.append((pred - Y).norm().item() / max(Y.norm().item(), 1e-12))
                rec = Y @ O_hat.T
                transfer.append((rec - X).norm().item() / max(X.norm().item(), 1e-12))
    return sum(residuals) / len(residuals), sum(transfer) / len(transfer)


def basis_recovery_error(O_hats: dict[tuple[int, int, str], torch.Tensor], puf, head_dim: int) -> float:
    errs = []
    for (layer, h, target), O_hat in O_hats.items():
        purpose = "K_attn" if target == "K" else "V_attn"
        O_true = _basis_for(puf, "givens", head_dim, layer, h, purpose,
                            torch.device("cpu"), torch.float32)
        I = torch.eye(head_dim, dtype=torch.float32)
        errs.append((O_hat @ O_true.T - I).norm().item() / math.sqrt(head_dim))
    return sum(errs) / len(errs)


def strip_cache(wrapped_kv, O_hats: dict[tuple[int, int, str], torch.Tensor]):
    stripped = []
    for layer, (k, v) in enumerate(wrapped_kv):
        k_out = torch.empty_like(k)
        v_out = torch.empty_like(v)
        for h in range(k.shape[1]):
            O_k = O_hats[(layer, h, "K")]
            O_v = O_hats[(layer, h, "V")]
            k_out[0, h] = (k[0, h].to(torch.float32) @ O_k.T).to(k.dtype)
            v_out[0, h] = (v[0, h].to(torch.float32) @ O_v.T).to(v.dtype)
        stripped.append((k_out, v_out))
    return stripped


@torch.inference_mode()
def evaluate_downstream(plain_model, wrapped_model, tokenizer, prompts: list[str],
                        O_hats: dict[tuple[int, int, str], torch.Tensor]) -> dict:
    device = next(plain_model.parameters()).device
    rows = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        ids = inputs["input_ids"]
        plain_kv = forward_kv(plain_model, ids)
        wrapped_kv = forward_kv(wrapped_model, ids)
        _, transfer_residual = residual_for_pair(plain_kv, wrapped_kv, O_hats)
        stripped = strip_cache(wrapped_kv, O_hats)
        inv = invert_v_layer0(plain_model, stripped[0][1].to(device), ids[0].to(device))
        rows.append({
            "prompt": prompt,
            "transfer_residual": transfer_residual,
            "v_inversion_top1": inv.token_accuracy_top1,
        })
    return {
        "rows": rows,
        "mean_transfer_residual": sum(r["transfer_residual"] for r in rows) / max(len(rows), 1),
        "mean_v_inversion_top1": sum(r["v_inversion_top1"] for r in rows) / max(len(rows), 1),
    }


def load_model(model_path: Path, *, force_fp32: bool):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    model.config._attn_implementation = "eager"
    if force_fp32:
        model.to(torch.float32)
    return model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    ap.add_argument("--snapshot", type=str, default=None)
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/profiling_native_summary.json"))
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS_FILE)
    ap.add_argument("--force-fp32", action="store_true")
    ap.add_argument("--fp32-cache", action="store_true")
    ap.add_argument("--alignment", choices=["naive", "hungarian"], default="naive")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    model_path = _model_path(args.model_cache_dir, args.snapshot)
    out = {
        "settings": vars(args) | {"model_path": str(model_path)},
        "resource_before": _cuda_summary(),
        "scenarios": {},
    }

    plain_model = None
    wrapped_model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        plain_model = load_model(model_path, force_fp32=args.force_fp32)
        wrapped_model = load_model(model_path, force_fp32=args.force_fp32)
        num_layers = plain_model.config.num_hidden_layers
        num_kv_heads = plain_model.config.num_key_value_heads
        head_dim = getattr(plain_model.config, "head_dim",
                           plain_model.config.hidden_size // plain_model.config.num_attention_heads)
        prompts = [p.strip() for p in args.prompts_file.read_text(encoding="utf-8").splitlines() if p.strip()]

        puf_fit = make_puf("device_A")
        scenarios = [
            ("S1_native_same_session", puf_fit),
            ("S4_native_session_refresh", make_puf("device_A", session_nonce=b"\x99" * 16)),
        ]

        print(f"Profiling native cache: prompts={args.n_prompts}, seq_len={args.seq_len}", flush=True)
        install_puf_attention(wrapped_model, puf_fit, kind_k="givens", kind_v="givens",
                              fp32_cache=args.fp32_cache)
        accum = NativeAccumulator(num_layers, num_kv_heads, head_dim, alignment=args.alignment)
        t0 = time.perf_counter()
        for i, ids in enumerate(random_prompt_ids(tokenizer, args.n_prompts, args.seq_len, args.seed)):
            plain_kv = forward_kv(plain_model, ids)
            wrapped_kv = forward_kv(wrapped_model, ids)
            accum.add(plain_kv, wrapped_kv)
            del plain_kv, wrapped_kv
            if torch.cuda.is_available() and (i + 1) % 25 == 0:
                torch.cuda.empty_cache()
            if (i + 1) % 100 == 0:
                print(f"  ...{i+1}/{args.n_prompts} prompts ({time.perf_counter() - t0:.1f}s)", flush=True)
        O_hats = solve_all(accum)
        fit_time = time.perf_counter() - t0
        rows_per_unit = next(iter(accum.n_rows.values()))
        print(f"Solved native Procrustes: rows_per_unit={rows_per_unit}, fit_time={fit_time:.1f}s", flush=True)

        # Fresh prompt under each target session and downstream inversion on real prompts.
        uninstall_puf_attention(wrapped_model)
        for name, puf_target in scenarios:
            install_puf_attention(wrapped_model, puf_target, kind_k="givens", kind_v="givens",
                                  fp32_cache=args.fp32_cache)
            try:
                fresh_ids = next(random_prompt_ids(tokenizer, 1, args.seq_len, args.seed + 99999))
                fresh_plain = forward_kv(plain_model, fresh_ids)
                fresh_wrapped = forward_kv(wrapped_model, fresh_ids)
                fresh_residual, fresh_transfer = residual_for_pair(fresh_plain, fresh_wrapped, O_hats)
                downstream = evaluate_downstream(plain_model, wrapped_model, tokenizer, prompts, O_hats)
            finally:
                uninstall_puf_attention(wrapped_model)

            rec_err = basis_recovery_error(O_hats, puf_fit, head_dim)
            out["scenarios"][name] = {
                "puf_fit": puf_fit.session_nonce.hex(),
                "puf_target": puf_target.session_nonce.hex(),
                "rows_per_unit": rows_per_unit,
                "n_units": len(O_hats),
                "fit_time_s": fit_time,
                "mean_basis_recovery_err_vs_fit": rec_err,
                "fresh_prompt_residual": fresh_residual,
                "fresh_prompt_transfer_residual": fresh_transfer,
                "downstream": downstream,
            }
            print(f"{name}: fresh_residual={fresh_residual:.3f}, "
                  f"downstream_v_inv={downstream['mean_v_inversion_top1']:.3f}", flush=True)
    finally:
        models = [wrapped_model, plain_model]
        wrapped_model = None
        plain_model = None
        for model in models:
            if model is not None:
                try:
                    uninstall_puf_attention(model)
                except Exception:
                    pass
                del model
        del models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        out["resource_after_cleanup"] = _cuda_summary()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
