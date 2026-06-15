"""Known-plaintext profiling attack against native Level-3 caches.

This Wave-3 script makes the reviewer-facing KPA explicit: the attacker obtains
N chosen or known plaintext/cache pairs in one session, solves the per-head
orthogonal Procrustes map, and tests whether the learned map transfers to held-
out same-session and refreshed-session targets.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from puf4secure_kvcache.puf_attention import install_puf_attention, uninstall_puf_attention
from puf4secure_kvcache.puf_sim import make_puf

from run_profiling_native import (  # noqa: E402
    NativeAccumulator,
    _cuda_summary,
    _model_path,
    evaluate_downstream,
    forward_kv,
    load_model,
    random_prompt_ids,
    residual_for_pair,
    solve_all,
)


DEFAULT_MODEL_CACHE = Path("/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")
DEFAULT_PROMPTS_FILE = Path("experiments/prompts/synthetic_privacy_prompts.txt")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    ap.add_argument("--snapshot", type=str, default=None)
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/kpa_native_summary.json"))
    ap.add_argument("--known-prompts", type=int, nargs="+", default=[4, 16, 64, 256])
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=314159)
    ap.add_argument("--alignment", choices=["naive", "hungarian"], default="naive")
    ap.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS_FILE)
    ap.add_argument("--skip-downstream", action="store_true")
    ap.add_argument("--force-fp32", action="store_true")
    ap.add_argument("--fp32-cache", action="store_true")
    return ap.parse_args()


def main() -> None:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    args = parse_args()
    model_path = _model_path(args.model_cache_dir, args.snapshot)
    cutoffs = sorted(set(args.known_prompts))
    max_known = max(cutoffs)
    out = {
        "settings": vars(args) | {"model_path": str(model_path)},
        "resource_before": _cuda_summary(),
        "cutoffs": [],
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
        puf_refresh = make_puf("device_A", session_nonce=b"\x77" * 16)
        accum = NativeAccumulator(num_layers, num_kv_heads, head_dim, alignment=args.alignment)
        install_puf_attention(wrapped_model, puf_fit, kind_k="givens", kind_v="givens", fp32_cache=args.fp32_cache)
        installed = True
        t0 = time.perf_counter()
        try:
            prompt_iter = random_prompt_ids(tokenizer, max_known, args.seq_len, args.seed)
            for idx, ids in enumerate(prompt_iter, 1):
                plain_kv = forward_kv(plain_model, ids)
                wrapped_kv = forward_kv(wrapped_model, ids)
                accum.add(plain_kv, wrapped_kv)
                del plain_kv, wrapped_kv
                if idx in cutoffs:
                    O_hats = solve_all(accum)
                    rows_per_unit = next(iter(accum.n_rows.values()))
                    cutoff_rec = {
                        "known_prompts": idx,
                        "rows_per_unit": rows_per_unit,
                        "n_units": len(O_hats),
                        "fit_elapsed_s": time.perf_counter() - t0,
                        "targets": {},
                    }
                    for target_name, target_puf in [("same_session", puf_fit), ("session_refresh", puf_refresh)]:
                        uninstall_puf_attention(wrapped_model)
                        installed = False
                        install_puf_attention(wrapped_model, target_puf, kind_k="givens", kind_v="givens",
                                              fp32_cache=args.fp32_cache)
                        installed = True
                        fresh_ids = next(random_prompt_ids(tokenizer, 1, args.seq_len, args.seed + 100000 + idx))
                        fresh_plain = forward_kv(plain_model, fresh_ids)
                        fresh_wrapped = forward_kv(wrapped_model, fresh_ids)
                        residual, transfer = residual_for_pair(fresh_plain, fresh_wrapped, O_hats)
                        rec = {
                            "fresh_prompt_residual": residual,
                            "fresh_prompt_transfer_residual": transfer,
                        }
                        if not args.skip_downstream:
                            rec["downstream"] = evaluate_downstream(plain_model, wrapped_model, tokenizer, prompts, O_hats)
                        cutoff_rec["targets"][target_name] = rec
                    uninstall_puf_attention(wrapped_model)
                    installed = False
                    install_puf_attention(wrapped_model, puf_fit, kind_k="givens", kind_v="givens",
                                          fp32_cache=args.fp32_cache)
                    installed = True
                    out["cutoffs"].append(cutoff_rec)
                    print(json.dumps(cutoff_rec, indent=2), flush=True)
        finally:
            if installed:
                uninstall_puf_attention(wrapped_model)
    finally:
        for model in (wrapped_model, plain_model):
            if model is not None:
                try:
                    uninstall_puf_attention(model)
                except Exception:
                    pass
                del model
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
