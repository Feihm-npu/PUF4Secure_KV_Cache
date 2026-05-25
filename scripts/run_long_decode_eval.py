"""Long-context generation utility test for the Level-3 PUF attention wrapper.

Question: does the ~2-3% bf16 logits drift between plain and wrapped legitimate
decoding compound across many autoregressive decode steps and derail output
quality?

Method: greedy-decode N=128 tokens on each prompt under two configurations:
  (A) plain bf16 model (no wrapper).
  (B) wrapped model on the matching PUF (legit decoding).

Metrics:
  * Tokens-equal prefix length (first divergence step).
  * Total tokens equal / total tokens (exact-match rate).
  * Character overlap and ROUGE-L between the two generations.
  * Per-step max-abs and rel_l2 logits gap (sample-averaged).

This is the closest analogue to "does my user notice anything different".
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from puf4secure_kvcache.metrics import char_overlap, rouge_l_f1
from puf4secure_kvcache.model_utils import load_model_and_tokenizer, resolve_snapshot_path
from puf4secure_kvcache.puf_attention import install_puf_attention, uninstall_puf_attention
from puf4secure_kvcache.puf_sim import make_puf


PROMPTS_FILE = Path("experiments/prompts/synthetic_privacy_prompts.txt")


@torch.inference_mode()
def greedy_decode(model, tokenizer, prompt: str, max_new_tokens: int,
                  return_logits: bool = False):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**inputs, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]
    tokens: list[int] = []
    step_logits = [] if return_logits else None
    for _ in range(max_new_tokens):
        if return_logits:
            step_logits.append(logits.detach().float().cpu())
        next_tok = int(logits.argmax(dim=-1).item())
        tokens.append(next_tok)
        if next_tok == tokenizer.eos_token_id:
            break
        nxt = torch.tensor([[next_tok]], device=device)
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    return tokens, step_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/long_decode_summary.json"))
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--fp32-cache", action="store_true")
    args = ap.parse_args()

    prompts = [p.strip() for p in PROMPTS_FILE.read_text().splitlines() if p.strip()]
    model, tokenizer = load_model_and_tokenizer(resolve_snapshot_path())
    model.config._attn_implementation = "eager"

    out = {"max_new_tokens": args.max_new_tokens, "fp32_cache": args.fp32_cache, "prompts": []}

    for prompt in prompts:
        rec = {"prompt": prompt}

        # --- plain bf16 baseline ---
        plain_toks, plain_logits = greedy_decode(
            model, tokenizer, prompt, args.max_new_tokens, return_logits=True)
        plain_text = tokenizer.decode(plain_toks, skip_special_tokens=True)

        # --- wrapped legit decode ---
        puf = make_puf("device_A")
        install_puf_attention(model, puf, kind_k="givens", kind_v="givens",
                              fp32_cache=args.fp32_cache)
        try:
            wrap_toks, wrap_logits = greedy_decode(
                model, tokenizer, prompt, args.max_new_tokens, return_logits=True)
        finally:
            uninstall_puf_attention(model)
        wrap_text = tokenizer.decode(wrap_toks, skip_special_tokens=True)

        # divergence step
        div = len(plain_toks)
        for i, (a, b) in enumerate(zip(plain_toks, wrap_toks)):
            if a != b:
                div = i
                break
        eq = sum(1 for a, b in zip(plain_toks, wrap_toks) if a == b)
        denom = max(len(plain_toks), len(wrap_toks))

        # logits gap per step (limited to min length)
        max_abs_per_step, rel_l2_per_step = [], []
        for pl, wl in zip(plain_logits, wrap_logits):
            max_abs_per_step.append((pl - wl).abs().max().item())
            denom_l = max(pl.norm().item(), 1e-12)
            rel_l2_per_step.append((pl - wl).norm().item() / denom_l)

        rec["plain_text"] = plain_text
        rec["wrapped_text"] = wrap_text
        rec["plain_len"] = len(plain_toks)
        rec["wrapped_len"] = len(wrap_toks)
        rec["divergence_step"] = div
        rec["exact_match_rate"] = eq / max(denom, 1)
        rec["char_overlap"] = char_overlap(wrap_text, plain_text)
        rec["rouge_l_vs_plain"] = rouge_l_f1(wrap_text, plain_text)
        rec["logits_max_abs_per_step_mean"] = sum(max_abs_per_step) / max(len(max_abs_per_step), 1)
        rec["logits_max_abs_per_step_max"] = max(max_abs_per_step) if max_abs_per_step else 0.0
        rec["logits_rel_l2_per_step_mean"] = sum(rel_l2_per_step) / max(len(rel_l2_per_step), 1)
        rec["logits_rel_l2_per_step_max"] = max(rel_l2_per_step) if rel_l2_per_step else 0.0
        rec["logits_rel_l2_first_step"] = rel_l2_per_step[0] if rel_l2_per_step else 0.0
        rec["logits_rel_l2_last_step"] = rel_l2_per_step[-1] if rel_l2_per_step else 0.0

        out["prompts"].append(rec)
        print(f"--- {prompt[:50]}...")
        print(f"  divergence step: {div} / {len(plain_toks)} tokens")
        print(f"  exact-match rate: {rec['exact_match_rate']:.3f}")
        print(f"  ROUGE-L (wrap vs plain): {rec['rouge_l_vs_plain']:.3f}")
        print(f"  logits rel_l2 step mean/max/last: "
              f"{rec['logits_rel_l2_per_step_mean']:.3e} / "
              f"{rec['logits_rel_l2_per_step_max']:.3e} / "
              f"{rec['logits_rel_l2_last_step']:.3e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
