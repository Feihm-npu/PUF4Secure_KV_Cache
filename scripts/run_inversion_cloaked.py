"""M5: run inversion attack against cloaked KV to confirm defense."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from puf4secure_kvcache.attacks import invert_v_layer0, invert_k_layer0
from puf4secure_kvcache.kv_cloak import cloak_kv_cache
from puf4secure_kvcache.kv_io import load_capture
from puf4secure_kvcache.model_utils import load_model_and_tokenizer, resolve_snapshot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inversion attack against cloaked KV (M5 sanity check).")
    parser.add_argument("--capture", required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    capture_dir = Path(args.capture)
    meta, kv_list = load_capture(capture_dir)
    target_ids = torch.tensor(meta["input_ids"], dtype=torch.long)

    model, tokenizer = load_model_and_tokenizer(resolve_snapshot_path())
    device = next(model.parameters()).device
    kv_list = [(k.to(device), v.to(device)) for k, v in kv_list]
    cloaked, _ = cloak_kv_cache(kv_list, block_size=args.block_size, seed=args.seed, theta=args.theta)

    v_plain = invert_v_layer0(model, kv_list[0][1], target_ids)
    v_cloak = invert_v_layer0(model, cloaked[0][1], target_ids)
    k_plain = invert_k_layer0(model, kv_list[0][0], target_ids)
    k_cloak = invert_k_layer0(model, cloaked[0][0], target_ids)

    out = {
        "prompt": meta["prompt"],
        "V_plain_top1": v_plain.token_accuracy_top1,
        "V_cloaked_top1": v_cloak.token_accuracy_top1,
        "K_plain_top1": k_plain.token_accuracy_top1,
        "K_cloaked_top1": k_cloak.token_accuracy_top1,
        "V_plain_text": tokenizer.decode(v_plain.recovered_ids[:, 0].tolist(), skip_special_tokens=True),
        "V_cloaked_text": tokenizer.decode(v_cloak.recovered_ids[:, 0].tolist(), skip_special_tokens=True),
    }
    with open(capture_dir / f"inversion_cloaked_b{args.block_size}_t{args.theta}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
