"""Utility benchmark for plain vs. PUF-wrapped inference.

This script is the first P0 experiment block for the paper evidence chain. It
keeps the benchmark lightweight enough for repeated sanity runs while producing
standard utility signals:

  * perplexity on a cached text dataset (default: ag_news/test)
  * multiple-choice accuracy by conditional log-likelihood (default: hellaswag)
    * long greedy decode agreement against the plain baseline

The Level-3 wrapper currently supports Qwen3, Qwen2, and Llama-family RoPE
decoder models.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from puf4secure_kvcache.metrics import char_overlap, rouge_l_f1
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


def _select_first(dataset, n: int):
    if n <= 0:
        return []
    n = min(n, len(dataset))
    return dataset.select(range(n))


@contextmanager
def _mode_context(model, mode: str, *, fp32_cache: bool, device_id: str,
                  fast_givens: bool, affine_mask: bool = False, mask_std: float = 4.0):
    installed = False
    if mode == "wrapped":
        puf = make_puf(device_id)
        install_puf_attention(
            model,
            puf,
            kind_k="givens",
            kind_v="givens",
            fp32_cache=fp32_cache,
            fast_givens=fast_givens,
            affine_mask=affine_mask,
            mask_std=mask_std,
        )
        installed = True
    elif mode != "plain":
        raise ValueError(f"unknown mode: {mode!r}")
    try:
        yield
    finally:
        if installed:
            uninstall_puf_attention(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.inference_mode()
def perplexity_eval(model, tokenizer, *, dataset_name: str, split: str,
                    text_field: str, samples: int, max_length: int) -> dict:
    ds = _select_first(load_dataset(dataset_name, split=split), samples)
    total_nll = 0.0
    total_tokens = 0
    used = 0
    device = next(model.parameters()).device

    for rec in ds:
        text = str(rec[text_field]).strip()
        if not text:
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        if enc["input_ids"].shape[1] < 2:
            continue
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, labels=enc["input_ids"], use_cache=False)
        n_tokens = int(enc["attention_mask"].sum().item()) - 1
        total_nll += float(out.loss.item()) * n_tokens
        total_tokens += n_tokens
        used += 1

    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "dataset": dataset_name,
        "split": split,
        "text_field": text_field,
        "samples_requested": samples,
        "samples_used": used,
        "tokens": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 80.0)),
    }


def _conditional_loglik(model, tokenizer, context: str, completion: str,
                        max_length: int) -> float | None:
    device = next(model.parameters()).device
    ctx_ids = tokenizer(context, add_special_tokens=True)["input_ids"]
    end_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not end_ids:
        return None
    if len(ctx_ids) + len(end_ids) > max_length:
        keep_ctx = max_length - len(end_ids)
        if keep_ctx < 1:
            return None
        ctx_ids = ctx_ids[-keep_ctx:]
    input_ids = torch.tensor([ctx_ids + end_ids], device=device)
    labels = torch.tensor([[-100] * len(ctx_ids) + end_ids], device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
    return -float(out.loss.item()) * len(end_ids)


@torch.inference_mode()
def hellaswag_eval(model, tokenizer, *, dataset_name: str, split: str,
                   samples: int, max_length: int) -> dict:
    ds = _select_first(load_dataset(dataset_name, split=split), samples)
    correct = 0
    used = 0
    skipped = 0
    records = []

    for idx, rec in enumerate(ds):
        endings = list(rec["endings"])
        label_raw = rec["label"]
        if label_raw == "":
            skipped += 1
            continue
        label = int(label_raw)
        context = rec.get("ctx") or (str(rec.get("ctx_a", "")) + " " + str(rec.get("ctx_b", ""))).strip()
        scores = []
        for ending in endings:
            score = _conditional_loglik(model, tokenizer, context, " " + str(ending), max_length)
            scores.append(float("-inf") if score is None else score)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct += int(pred == label)
        used += 1
        records.append({"index": idx, "label": label, "pred": pred, "scores": scores})

    return {
        "dataset": dataset_name,
        "split": split,
        "samples_requested": samples,
        "samples_used": used,
        "skipped": skipped,
        "accuracy": correct / max(used, 1),
        "records": records,
    }


@torch.inference_mode()
def mmlu_eval(model, tokenizer, *, dataset_name: str, subset: str, split: str,
              samples: int, max_length: int, seed: int = 20260608) -> dict:
    """Standard 4-choice MMLU accuracy via per-letter conditional log-likelihood.

    Uses the lm-eval-harness style letter prompt (``Answer: A/B/C/D``) and scores
    each option with the same conditional-loglik mechanism as HellaSwag, so the
    plain-vs-wrapped delta isolates the wrapper's effect on a standard benchmark.
    """
    ds = load_dataset(dataset_name, subset, split=split)
    ds = ds.shuffle(seed=seed)
    ds = _select_first(ds, samples)
    letters = ["A", "B", "C", "D"]
    correct = 0
    used = 0
    skipped = 0
    records = []

    for idx, rec in enumerate(ds):
        choices = list(rec["choices"])
        answer = int(rec["answer"])
        if len(choices) != 4 or answer not in range(4):
            skipped += 1
            continue
        lines = [rec["question"].strip()]
        for letter, choice in zip(letters, choices):
            lines.append(f"{letter}. {choice}")
        lines.append("Answer:")
        context = "\n".join(lines)
        scores = []
        for letter in letters:
            score = _conditional_loglik(model, tokenizer, context, " " + letter, max_length)
            scores.append(float("-inf") if score is None else score)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct += int(pred == answer)
        used += 1
        records.append({"index": idx, "answer": answer, "pred": pred, "scores": scores})

    return {
        "dataset": dataset_name,
        "subset": subset,
        "split": split,
        "samples_requested": samples,
        "samples_used": used,
        "skipped": skipped,
        "accuracy": correct / max(used, 1),
        "records": records,
    }


@torch.inference_mode()
def greedy_decode(model, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
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


def long_decode_eval(model, tokenizer, *, prompts: Iterable[str], max_new_tokens: int) -> list[dict]:
    out = []
    for prompt in prompts:
        toks = greedy_decode(model, tokenizer, prompt, max_new_tokens)
        out.append({
            "prompt": prompt,
            "tokens": toks,
            "text": tokenizer.decode(toks, skip_special_tokens=True),
            "length": len(toks),
        })
    return out


def compare_long_decode(mode_results: dict[str, list[dict]]) -> list[dict]:
    if "plain" not in mode_results:
        return []
    plain = mode_results["plain"]
    comparisons = []
    for mode, records in mode_results.items():
        if mode == "plain":
            continue
        for idx, (base, other) in enumerate(zip(plain, records)):
            denom = max(len(base["tokens"]), len(other["tokens"]), 1)
            eq = sum(1 for a, b in zip(base["tokens"], other["tokens"]) if a == b)
            comparisons.append({
                "prompt_index": idx,
                "mode": mode,
                "divergence_step": _divergence_step(base["tokens"], other["tokens"]),
                "exact_match_rate": eq / denom,
                "rouge_l_vs_plain": rouge_l_f1(other["text"], base["text"]),
                "char_overlap_vs_plain": char_overlap(other["text"], base["text"]),
                "plain_text": base["text"],
                "mode_text": other["text"],
            })
    return comparisons


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    ap.add_argument("--snapshot", type=str, default=None)
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/utility_summary.json"))
    ap.add_argument("--modes", nargs="+", choices=["plain", "wrapped"], default=["plain", "wrapped"])
    ap.add_argument("--force-fp32", action="store_true", help="Cast the loaded model to fp32 before benchmarking.")
    ap.add_argument("--fp32-cache", action="store_true", help="Store wrapped K/V cache in fp32.")
    ap.add_argument("--fast-givens", action="store_true", help="Use vectorized pairwise Givens rotations instead of dense matrices.")
    ap.add_argument("--affine-mask", action="store_true", help="Use the no-sidecar PUF-derived affine-mask wrapped path.")
    ap.add_argument("--mask-std", type=float, default=4.0, help="Standard deviation of the affine mask (used when --affine-mask).")
    ap.add_argument("--device-id", default="device_A")

    ap.add_argument("--ppl-dataset", default="ag_news")
    ap.add_argument("--ppl-split", default="test")
    ap.add_argument("--ppl-text-field", default="text")
    ap.add_argument("--ppl-samples", type=int, default=32)
    ap.add_argument("--ppl-max-length", type=int, default=256)

    ap.add_argument("--mc-dataset", default="hellaswag")
    ap.add_argument("--mc-split", default="validation")
    ap.add_argument("--mc-samples", type=int, default=32)
    ap.add_argument("--mc-max-length", type=int, default=256)

    ap.add_argument("--mmlu-dataset", default="cais/mmlu")
    ap.add_argument("--mmlu-subset", default="all")
    ap.add_argument("--mmlu-split", default="test")
    ap.add_argument("--mmlu-samples", type=int, default=0, help="MMLU samples to score (0 disables MMLU).")
    ap.add_argument("--mmlu-max-length", type=int, default=512)

    ap.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS_FILE)
    ap.add_argument("--long-decode-prompts", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    return ap.parse_args()


def main() -> None:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    args = parse_args()
    model_path = _model_path(args.model_cache_dir, args.snapshot)
    out = {
        "settings": vars(args) | {"model_path": str(model_path)},
        "resource_before": _cuda_summary(),
        "modes": {},
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
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        else:
            setattr(model.config, "_attn_implementation", "eager")
        if args.force_fp32:
            model.to(torch.float32)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        prompts = [p.strip() for p in args.prompts_file.read_text(encoding="utf-8").splitlines() if p.strip()]
        prompts = prompts[: args.long_decode_prompts]
        long_results: dict[str, list[dict]] = {}

        for mode in args.modes:
            print(f"=== utility mode: {mode} ===", flush=True)
            with _mode_context(
                model,
                mode,
                fp32_cache=args.fp32_cache,
                device_id=args.device_id,
                fast_givens=args.fast_givens,
                affine_mask=args.affine_mask,
                mask_std=args.mask_std,
            ):
                rec = {}
                if args.ppl_samples > 0:
                    rec["perplexity"] = perplexity_eval(
                        model, tokenizer,
                        dataset_name=args.ppl_dataset,
                        split=args.ppl_split,
                        text_field=args.ppl_text_field,
                        samples=args.ppl_samples,
                        max_length=args.ppl_max_length,
                    )
                    print(f"  ppl={rec['perplexity']['perplexity']:.4f}", flush=True)
                if args.mc_samples > 0:
                    rec["hellaswag"] = hellaswag_eval(
                        model, tokenizer,
                        dataset_name=args.mc_dataset,
                        split=args.mc_split,
                        samples=args.mc_samples,
                        max_length=args.mc_max_length,
                    )
                    print(f"  hellaswag_acc={rec['hellaswag']['accuracy']:.4f}", flush=True)
                if args.mmlu_samples > 0:
                    rec["mmlu"] = mmlu_eval(
                        model, tokenizer,
                        dataset_name=args.mmlu_dataset,
                        subset=args.mmlu_subset,
                        split=args.mmlu_split,
                        samples=args.mmlu_samples,
                        max_length=args.mmlu_max_length,
                    )
                    print(f"  mmlu_acc={rec['mmlu']['accuracy']:.4f} (n={rec['mmlu']['samples_used']})", flush=True)
                if prompts and args.max_new_tokens > 0:
                    rec["long_decode"] = long_decode_eval(
                        model, tokenizer,
                        prompts=prompts,
                        max_new_tokens=args.max_new_tokens,
                    )
                    long_results[mode] = rec["long_decode"]
                    print(f"  decoded_prompts={len(rec['long_decode'])}", flush=True)
                out["modes"][mode] = rec

        out["long_decode_comparisons"] = compare_long_decode(long_results)
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
