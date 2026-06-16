"""Long-decode divergence benchmark for plain vs PUF-wrapped inference.

The existing utility script reports a few prompt-level greedy continuations. This
Wave-2 benchmark scales that diagnostic to many prompts and emits a divergence
curve: at each decode step, the fraction of prompts whose wrapped continuation
has already diverged from the plain continuation.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from puf4secure_kvcache.model_utils import resolve_snapshot_path
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


def load_prompts(dataset: str, split: str, text_field: str, samples: int, max_chars: int) -> list[str]:
    ds = load_dataset(dataset, split=split)
    prompts: list[str] = []
    for rec in ds:
        text = str(rec[text_field]).strip()
        if not text:
            continue
        prompts.append(text[:max_chars])
        if len(prompts) >= samples:
            break
    return prompts


@torch.inference_mode()
def greedy_tokens(model, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = model(**inputs, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]
    tokens: list[int] = []
    for _ in range(max_new_tokens):
        next_tok = int(logits.argmax(dim=-1).item())
        tokens.append(next_tok)
        if tokenizer.eos_token_id is not None and next_tok == tokenizer.eos_token_id:
            break
        nxt = torch.tensor([[next_tok]], device=device)
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    return tokens


def divergence_step(a: list[int], b: list[int]) -> int:
    for idx, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return idx
    return min(len(a), len(b))


def summarize(records: list[dict], max_new_tokens: int) -> dict:
    n = len(records)
    diverged = [r for r in records if r["divergence_step"] < max_new_tokens]
    curve = []
    for step in range(max_new_tokens + 1):
        curve.append({
            "step": step,
            "diverged_fraction": sum(int(r["divergence_step"] <= step) for r in diverged) / max(n, 1),
        })
    return {
        "samples": n,
        "max_new_tokens": max_new_tokens,
        "full_match_rate": 1.0 - len(diverged) / max(n, 1),
        "divergence_rate": len(diverged) / max(n, 1),
        "mean_divergence_step_diverged_only": (
            sum(int(r["divergence_step"]) for r in diverged) / max(len(diverged), 1)
        ),
        "divergence_curve": curve,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    ap.add_argument("--snapshot", type=str, default=None)
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/decode_divergence.json"))
    ap.add_argument("--dataset", default="ag_news")
    ap.add_argument("--split", default="test")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--max-prompt-chars", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--force-fp32", action="store_true")
    ap.add_argument("--cast-dtype", choices=["none", "fp16", "bf16"], default="none",
                    help="Cast the model to this dtype (ignored if --force-fp32).")
    ap.add_argument("--fp32-cache", action="store_true")
    ap.add_argument("--affine-mask", action="store_true", help="Use the no-sidecar PUF-derived affine-mask wrapped path.")
    ap.add_argument("--mask-std", type=float, default=4.0, help="Standard deviation of the affine mask (used when --affine-mask).")
    ap.add_argument("--attn-backend", default="eager", help="eager | sdpa | triton_fused (fused regenerates+subtracts the mask inside the decode kernel).")
    ap.add_argument("--device-id", default="device_A")
    return ap.parse_args()


def main() -> None:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    args = parse_args()
    model_path = _model_path(args.model_cache_dir, args.snapshot)
    prompts = load_prompts(args.dataset, args.split, args.text_field, args.samples, args.max_prompt_chars)
    out = {
        "settings": vars(args) | {"model_path": str(model_path)},
        "resource_before": _cuda_summary(),
        "prompt_count": len(prompts),
        "records": [],
    }
    model = None
    installed = False
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
        elif args.cast_dtype == "fp16":
            model.to(torch.float16)
        elif args.cast_dtype == "bf16":
            model.to(torch.bfloat16)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        plain = [greedy_tokens(model, tokenizer, prompt, args.max_new_tokens) for prompt in prompts]
        install_puf_attention(model, make_puf(args.device_id), fp32_cache=args.fp32_cache,
                               affine_mask=args.affine_mask, mask_std=args.mask_std,
                               attn_backend=args.attn_backend)
        installed = True
        wrapped = [greedy_tokens(model, tokenizer, prompt, args.max_new_tokens) for prompt in prompts]
        uninstall_puf_attention(model)
        installed = False

        for idx, prompt in enumerate(prompts):
            step = divergence_step(plain[idx], wrapped[idx])
            denom = max(len(plain[idx]), len(wrapped[idx]), 1)
            matches = sum(1 for a, b in zip(plain[idx], wrapped[idx]) if a == b)
            out["records"].append({
                "index": idx,
                "prompt": prompt,
                "plain_tokens": plain[idx],
                "wrapped_tokens": wrapped[idx],
                "divergence_step": step,
                "token_match_rate": matches / denom,
            })
        out["summary"] = summarize(out["records"], args.max_new_tokens)
        print(json.dumps(out["summary"], indent=2), flush=True)
    finally:
        if model is not None and installed:
            uninstall_puf_attention(model)
        if model is not None:
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
