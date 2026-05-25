"""Diagnostic: does the Level-3 long-decode divergence come from bf16 noise on
rotated coordinates, or from a wrapper correctness bug?

We compare three decode paths starting from the same prompt with greedy decode:
  (A) plain model in bf16
  (B) plain model cast to fp32
  (C) wrapped model in bf16 (with fp32 attention matmul + optional fp32 cache)
  (D) wrapped model cast to fp32 (no rotated-bf16 quantization anywhere)

If (D) matches (B) exactly while (C) diverges from (A), then the wrapper math is
correct and the divergence is bf16 noise on rotated coordinates compounding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from puf4secure_kvcache.metrics import rouge_l_f1
from puf4secure_kvcache.model_utils import load_model_and_tokenizer, resolve_snapshot_path
from puf4secure_kvcache.puf_attention import install_puf_attention, uninstall_puf_attention
from puf4secure_kvcache.puf_sim import make_puf


PROMPTS_FILE = Path("experiments/prompts/synthetic_privacy_prompts.txt")


@torch.inference_mode()
def greedy_decode(model, tokenizer, prompt: str, max_new_tokens: int):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**inputs, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]
    tokens: list[int] = []
    for _ in range(max_new_tokens):
        next_tok = int(logits.argmax(dim=-1).item())
        tokens.append(next_tok)
        if next_tok == tokenizer.eos_token_id:
            break
        nxt = torch.tensor([[next_tok]], device=device)
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    return tokens


def divergence_step(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/decode_precision_diagnosis.json"))
    args = ap.parse_args()

    prompts = [p.strip() for p in PROMPTS_FILE.read_text().splitlines() if p.strip()]

    # Two model instances: bf16 and fp32 — load fresh each.
    model_bf, tokenizer = load_model_and_tokenizer(resolve_snapshot_path())
    model_bf.config._attn_implementation = "eager"
    print("Casting copy to fp32 ...")
    model_fp = load_model_and_tokenizer(resolve_snapshot_path())[0]
    model_fp.config._attn_implementation = "eager"
    model_fp.to(torch.float32)

    out = {"max_new_tokens": args.max_new_tokens, "prompts": []}

    for prompt in prompts:
        rec = {"prompt": prompt}

        plain_bf = greedy_decode(model_bf, tokenizer, prompt, args.max_new_tokens)
        plain_fp = greedy_decode(model_fp, tokenizer, prompt, args.max_new_tokens)

        puf = make_puf("device_A")

        install_puf_attention(model_bf, puf, kind_k="givens", kind_v="givens", fp32_cache=False)
        try:
            wrap_bf = greedy_decode(model_bf, tokenizer, prompt, args.max_new_tokens)
        finally:
            uninstall_puf_attention(model_bf)

        install_puf_attention(model_fp, puf, kind_k="givens", kind_v="givens", fp32_cache=False)
        try:
            wrap_fp = greedy_decode(model_fp, tokenizer, prompt, args.max_new_tokens)
        finally:
            uninstall_puf_attention(model_fp)

        rec["plain_bf_text"] = tokenizer.decode(plain_bf, skip_special_tokens=True)
        rec["plain_fp_text"] = tokenizer.decode(plain_fp, skip_special_tokens=True)
        rec["wrap_bf_text"] = tokenizer.decode(wrap_bf, skip_special_tokens=True)
        rec["wrap_fp_text"] = tokenizer.decode(wrap_fp, skip_special_tokens=True)

        rec["divergence"] = {
            "wrap_bf_vs_plain_bf": divergence_step(wrap_bf, plain_bf),
            "wrap_fp_vs_plain_fp": divergence_step(wrap_fp, plain_fp),
            "plain_fp_vs_plain_bf": divergence_step(plain_fp, plain_bf),
            "wrap_fp_vs_plain_bf": divergence_step(wrap_fp, plain_bf),
        }
        rec["rouge_l"] = {
            "wrap_bf_vs_plain_bf": rouge_l_f1(rec["wrap_bf_text"], rec["plain_bf_text"]),
            "wrap_fp_vs_plain_fp": rouge_l_f1(rec["wrap_fp_text"], rec["plain_fp_text"]),
            "plain_fp_vs_plain_bf": rouge_l_f1(rec["plain_fp_text"], rec["plain_bf_text"]),
        }
        out["prompts"].append(rec)
        print(f"--- {prompt[:50]}...")
        print(f"  divergence wrap_bf vs plain_bf : step {rec['divergence']['wrap_bf_vs_plain_bf']:>3}, "
              f"ROUGE-L {rec['rouge_l']['wrap_bf_vs_plain_bf']:.3f}")
        print(f"  divergence wrap_fp vs plain_fp : step {rec['divergence']['wrap_fp_vs_plain_fp']:>3}, "
              f"ROUGE-L {rec['rouge_l']['wrap_fp_vs_plain_fp']:.3f}")
        print(f"  divergence plain_fp vs plain_bf: step {rec['divergence']['plain_fp_vs_plain_bf']:>3}, "
              f"ROUGE-L {rec['rouge_l']['plain_fp_vs_plain_bf']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
