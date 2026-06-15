"""End-to-end serving performance benchmark for the PUF attention wrapper.

Measures prefill latency, decode latency, decode tokens/s, and KV-cache memory
for plain and Level-3 wrapped paths. Inputs are deterministic token sequences so
that the benchmark focuses on serving cost rather than dataset preprocessing.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from puf4secure_kvcache.model_utils import kv_to_list, resolve_snapshot_path
from puf4secure_kvcache.puf_attention import install_puf_attention, uninstall_puf_attention
from puf4secure_kvcache.puf_sim import make_puf


DEFAULT_MODEL_CACHE = Path("/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")


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


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _kv_bytes(past_key_values) -> int:
    total = 0
    for k, v in kv_to_list(past_key_values):
        total += k.numel() * k.element_size()
        total += v.numel() * v.element_size()
    return total


def _make_input(vocab_size: int, length: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + length)
    # Avoid the lowest ids because many tokenizers reserve them for specials.
    ids = torch.randint(100, max(101, vocab_size - 1), (1, length), generator=g)
    return ids.to(device)


@contextmanager
def _mode(model, mode: str, *, fp32_cache: bool, device_id: str,
          affine_mask: bool = False, mask_std: float = 4.0):
    installed = False
    if mode in {"wrapped", "wrapped_fast"}:
        install_puf_attention(
            model,
            make_puf(device_id),
            fp32_cache=fp32_cache,
            fast_givens=(mode == "wrapped_fast"),
            affine_mask=affine_mask,
            mask_std=mask_std,
        )
        installed = True
    elif mode != "plain":
        raise ValueError(f"unknown mode: {mode}")
    try:
        yield
    finally:
        if installed:
            uninstall_puf_attention(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.inference_mode()
def run_once(model, input_ids: torch.Tensor, decode_tokens: int) -> dict:
    attn_mask = torch.ones_like(input_ids)
    _sync()
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
    _sync()
    prefill_s = time.perf_counter() - t0
    cache = out.past_key_values
    next_logits = out.logits[:, -1, :]
    generated = 0
    _sync()
    t0 = time.perf_counter()
    for _ in range(decode_tokens):
        next_tok = next_logits.argmax(dim=-1, keepdim=True)
        cur_len = cache.get_seq_length() if hasattr(cache, "get_seq_length") else cache[0][0].shape[2]
        step_mask = torch.ones(1, cur_len + 1, device=input_ids.device, dtype=torch.long)
        cache_position = torch.tensor([cur_len], device=input_ids.device, dtype=torch.long)
        out = model(
            input_ids=next_tok,
            past_key_values=cache,
            attention_mask=step_mask,
            cache_position=cache_position,
            use_cache=True,
        )
        cache = out.past_key_values
        next_logits = out.logits[:, -1, :]
        generated += 1
    _sync()
    decode_s = time.perf_counter() - t0
    return {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tokens": generated,
        "decode_tokens_per_s": generated / max(decode_s, 1e-12),
        "kv_cache_bytes_after_decode": _kv_bytes(cache),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    ap.add_argument("--snapshot", type=str, default=None)
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/performance_eval.json"))
    ap.add_argument("--modes", nargs="+", choices=["plain", "wrapped", "wrapped_fast"], default=["plain", "wrapped"])
    ap.add_argument("--prompt-lengths", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument("--force-fp32", action="store_true")
    ap.add_argument("--fp32-cache", action="store_true")
    ap.add_argument("--affine-mask", action="store_true", help="Use the no-sidecar PUF-derived affine-mask wrapped path.")
    ap.add_argument("--mask-std", type=float, default=4.0, help="Standard deviation of the affine mask (used when --affine-mask).")
    ap.add_argument("--device-id", default="device_A")
    return ap.parse_args()


def main() -> None:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    args = parse_args()
    model_path = _model_path(args.model_cache_dir, args.snapshot)
    out = {
        "settings": vars(args) | {"model_path": str(model_path)},
        "resource_before": _cuda_summary(),
        "results": [],
    }
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
        )
        model.eval()
        model.config._attn_implementation = "eager"
        if args.force_fp32:
            model.to(torch.float32)
        device = next(model.parameters()).device
        vocab_size = len(tokenizer)
        out["model"] = {
            "model_type": getattr(model.config, "model_type", None),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
            "num_attention_heads": getattr(model.config, "num_attention_heads", None),
            "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
            "dtype": str(next(model.parameters()).dtype),
        }

        for mode in args.modes:
            with _mode(model, mode, fp32_cache=args.fp32_cache, device_id=args.device_id,
                       affine_mask=args.affine_mask, mask_std=args.mask_std):
                for prompt_len in args.prompt_lengths:
                    input_ids = _make_input(vocab_size, prompt_len, device, args.seed)
                    for _ in range(args.warmup):
                        run_once(model, input_ids, args.decode_tokens)
                    runs = [run_once(model, input_ids, args.decode_tokens) for _ in range(args.repeats)]
                    rec = {
                        "mode": mode,
                        "prompt_length": prompt_len,
                        "decode_tokens": args.decode_tokens,
                        "repeats": args.repeats,
                        "prefill_s": _stats([r["prefill_s"] for r in runs]),
                        "decode_s": _stats([r["decode_s"] for r in runs]),
                        "decode_tokens_per_s": _stats([r["decode_tokens_per_s"] for r in runs]),
                        "kv_cache_bytes_after_decode": runs[-1]["kv_cache_bytes_after_decode"],
                    }
                    out["results"].append(rec)
                    print(json.dumps(rec, indent=2), flush=True)
    finally:
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
