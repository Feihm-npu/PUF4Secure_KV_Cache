"""Capture plaintext KV-cache for one or many prompts and persist to disk."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from puf4secure_kvcache.attacks import describe_kv_shapes
from puf4secure_kvcache.kv_io import CaptureMeta, save_capture
from puf4secure_kvcache.model_utils import (
    forward_with_kv,
    kv_to_list,
    load_model_and_tokenizer,
    resolve_snapshot_path,
)


def _safe_name(text: str, max_len: int = 48) -> str:
    keep = []
    for ch in text:
        if ch.isalnum():
            keep.append(ch)
        elif ch in " _-":
            keep.append("_")
        if len(keep) >= max_len:
            break
    return "".join(keep) or "prompt"


def capture_one(model, tokenizer, prompt: str, out_dir: Path, model_path: Path):
    inputs, past_kv, _ = forward_with_kv(model, tokenizer, prompt)
    kv_list = kv_to_list(past_kv)
    meta = CaptureMeta(
        prompt=prompt,
        model_path=str(model_path),
        model_type=model.config.model_type,
        num_layers=model.config.num_hidden_layers,
        num_attention_heads=model.config.num_attention_heads,
        num_key_value_heads=model.config.num_key_value_heads,
        head_dim=getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads),
        hidden_size=model.config.hidden_size,
        seq_len=int(inputs["input_ids"].shape[1]),
        input_ids=[int(x) for x in inputs["input_ids"][0].tolist()],
        dtype=str(next(model.parameters()).dtype),
        device=str(next(model.parameters()).device),
    )
    save_capture(out_dir, meta, kv_list)
    return meta, kv_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture KV-cache shapes and persist to disk.")
    parser.add_argument("--prompt", default=None, help="Single prompt text.")
    parser.add_argument("--prompts-file", default=None, help="File with one prompt per non-empty line.")
    parser.add_argument("--out-root", default="experiments/runs", help="Output directory root.")
    parser.add_argument("--seed", type=int, default=0, help="Torch manual seed (for any nondeterminism).")
    parser.add_argument("--print-shapes", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.prompt is None and args.prompts_file is None:
        parser.error("Must supply --prompt or --prompts-file.")

    prompts: list[str] = []
    if args.prompt is not None:
        prompts.append(args.prompt)
    if args.prompts_file is not None:
        for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                prompts.append(line)

    model_path = resolve_snapshot_path()
    model, tokenizer = load_model_and_tokenizer(model_path)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for idx, prompt in enumerate(prompts):
        sub = out_root / f"capture_{idx:03d}_{_safe_name(prompt)}"
        meta, kv_list = capture_one(model, tokenizer, prompt, sub, model_path)
        print(f"[{idx}] prompt={prompt!r}")
        print(f"    saved to {sub}")
        print(f"    seq_len={meta.seq_len}  layers={meta.num_layers}  "
              f"kv_heads={meta.num_key_value_heads}  head_dim={meta.head_dim}")
        if args.print_shapes:
            for item in describe_kv_shapes(kv_list):
                print(f"    {item}")


if __name__ == "__main__":
    main()
