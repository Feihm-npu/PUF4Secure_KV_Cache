"""Wave-0 sanity check for the Level-3 PUF attention wrapper.

For one local Hugging Face model, compare plain and wrapped inference on a small
prompt set. The script is intentionally lightweight: it checks first-forward
logit differences and short greedy-decode agreement before larger Wave-1 utility
or attack experiments are launched.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from puf4secure_kvcache.model_utils import resolve_snapshot_path
from puf4secure_kvcache.puf_attention import install_puf_attention, uninstall_puf_attention
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


@torch.inference_mode()
def _forward_logits(model, tokenizer, prompt: str, max_length: int) -> torch.Tensor:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    out = model(**enc, use_cache=True)
    return out.logits.detach().float().cpu()


@torch.inference_mode()
def _greedy_tokens(model, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
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


def _divergence_step(a: list[int], b: list[int]) -> int:
    for idx, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return idx
    return min(len(a), len(b))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    ap.add_argument("--snapshot", type=str, default=None)
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/wave0_wrapper_sanity.json"))
    ap.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS_FILE)
    ap.add_argument("--prompt-count", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--force-fp32", action="store_true")
    ap.add_argument("--fp32-cache", action="store_true")
    ap.add_argument("--fast-givens", action="store_true")
    ap.add_argument("--norm-blind", action="store_true")
    ap.add_argument("--norm-log-range", type=float, default=1.0)
    ap.add_argument("--unit-norm-cache", action="store_true")
    ap.add_argument("--nonorth-log-range", type=float, default=0.0)
    ap.add_argument("--affine-mask", action="store_true")
    ap.add_argument("--mask-std", type=float, default=4.0)
    ap.add_argument("--attn-backend", choices=["eager", "sdpa"], default="eager",
                    help="Attention inner loop: fp32 eager matmul or fused SDPA API.")
    ap.add_argument("--device-id", default="device_A")
    return ap.parse_args()


def main() -> None:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    args = parse_args()
    model_path = _model_path(args.model_cache_dir, args.snapshot)
    prompts = [p.strip() for p in args.prompts_file.read_text(encoding="utf-8").splitlines() if p.strip()]
    prompts = prompts[: args.prompt_count]
    out = {
        "settings": vars(args) | {"model_path": str(model_path)},
        "resource_before": _cuda_summary(),
        "results": [],
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
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        out["model"] = {
            "model_type": getattr(model.config, "model_type", None),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
            "num_attention_heads": getattr(model.config, "num_attention_heads", None),
            "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
            "dtype": str(next(model.parameters()).dtype),
        }

        plain_logits = []
        plain_tokens = []
        for prompt in prompts:
            plain_logits.append(_forward_logits(model, tokenizer, prompt, args.max_length))
            plain_tokens.append(_greedy_tokens(model, tokenizer, prompt, args.max_new_tokens))

        install_puf_attention(
            model,
            make_puf(args.device_id),
            kind_k="givens",
            kind_v="givens",
            fp32_cache=args.fp32_cache,
            fast_givens=args.fast_givens,
            norm_blind=args.norm_blind,
            norm_log_range=args.norm_log_range,
            unit_norm_cache=args.unit_norm_cache,
            nonorth_log_range=args.nonorth_log_range,
            affine_mask=args.affine_mask,
            mask_std=args.mask_std,
            attn_backend=args.attn_backend,
        )
        installed = True

        for idx, prompt in enumerate(prompts):
            wrapped_logits = _forward_logits(model, tokenizer, prompt, args.max_length)
            wrapped_tokens = _greedy_tokens(model, tokenizer, prompt, args.max_new_tokens)
            diff = (wrapped_logits - plain_logits[idx]).abs()
            denom = max(len(plain_tokens[idx]), len(wrapped_tokens), 1)
            same = sum(1 for a, b in zip(plain_tokens[idx], wrapped_tokens) if a == b)
            out["results"].append({
                "prompt_index": idx,
                "prompt": prompt,
                "max_abs_logit_diff": float(diff.max().item()),
                "mean_abs_logit_diff": float(diff.mean().item()),
                "plain_tokens": plain_tokens[idx],
                "wrapped_tokens": wrapped_tokens,
                "token_match_rate": same / denom,
                "divergence_step": _divergence_step(plain_tokens[idx], wrapped_tokens),
            })
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
