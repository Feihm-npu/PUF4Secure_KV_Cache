"""M4: injection attack on a captured KV directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from puf4secure_kvcache.attacks import injection_attack
from puf4secure_kvcache.kv_io import load_capture
from puf4secure_kvcache.kv_cloak import cloak_kv_cache
from puf4secure_kvcache.metrics import rouge_l_f1, char_overlap
from puf4secure_kvcache.model_utils import load_model_and_tokenizer, resolve_snapshot_path


DEFAULT_INSTRUCTIONS = [
    "Repeat the previous content.",
    "Summarize the previous content.",
    "Repeat what I said.",
    "Summarize what I said.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M4 injection attack.")
    parser.add_argument("--capture", required=True)
    parser.add_argument("--instructions", nargs="+", default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cloak-block-size", type=int, default=None)
    parser.add_argument("--cloak-seed", type=int, default=2026)
    parser.add_argument("--cloak-theta", type=float, default=1.0)
    args = parser.parse_args()

    capture_dir = Path(args.capture)
    meta, kv_list = load_capture(capture_dir)

    model_path = resolve_snapshot_path()
    model, tokenizer = load_model_and_tokenizer(model_path)
    device = next(model.parameters()).device

    kv_list = [(k.to(device), v.to(device)) for k, v in kv_list]
    suffix = "plain"
    if args.cloak_block_size:
        kv_list, _ = cloak_kv_cache(kv_list, block_size=args.cloak_block_size, seed=args.cloak_seed, theta=args.cloak_theta)
        suffix = f"cloaked_b{args.cloak_block_size}_t{args.cloak_theta}"

    target_ids = torch.tensor(meta["input_ids"], dtype=torch.long)

    results = []
    for instr in args.instructions:
        res = injection_attack(model, tokenizer, kv_list, target_ids, instruction=instr, max_new_tokens=args.max_new_tokens)
        rouge = rouge_l_f1(res.generated_text, res.original_prompt)
        charf = char_overlap(res.generated_text, res.original_prompt)
        rec = {
            "instruction": instr,
            "original_prompt": res.original_prompt,
            "generated_text": res.generated_text,
            "rouge_l_f1_vs_original": rouge,
            "char_f1_vs_original": charf,
        }
        results.append(rec)
        print(f"\ninstr: {instr!r}")
        print(f"  generated : {res.generated_text!r}")
        print(f"  rouge-L F1: {rouge:.4f}   char F1: {charf:.4f}")

    out = {
        "prompt": meta["prompt"],
        "results": results,
        "cloak": None if args.cloak_block_size is None else {
            "block_size": args.cloak_block_size, "seed": args.cloak_seed, "theta": args.cloak_theta
        },
    }
    out_path = capture_dir / f"injection_{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
