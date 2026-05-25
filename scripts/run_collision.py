"""M3: collision attack on a captured KV directory."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from puf4secure_kvcache.attacks import collision_attack
from puf4secure_kvcache.kv_io import load_capture
from puf4secure_kvcache.kv_cloak import cloak_kv_cache
from puf4secure_kvcache.metrics import rouge_l_f1, char_overlap
from puf4secure_kvcache.model_utils import load_model_and_tokenizer, resolve_snapshot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M3 collision attack.")
    parser.add_argument("--capture", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--use", choices=["K", "V"], default="V")
    parser.add_argument("--top-k-fraction", type=float, default=1.0 / 8.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--fixed-prefix", type=int, default=1, help="Treat first N tokens as known (e.g. BOS).")
    parser.add_argument("--cloak-block-size", type=int, default=None, help="If set, apply KV-Cloak before attack.")
    parser.add_argument("--cloak-seed", type=int, default=2026)
    parser.add_argument("--cloak-theta", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    capture_dir = Path(args.capture)
    meta, kv_list = load_capture(capture_dir)
    target_ids = torch.tensor(meta["input_ids"], dtype=torch.long)

    model_path = resolve_snapshot_path()
    model, tokenizer = load_model_and_tokenizer(model_path)
    device = next(model.parameters()).device
    kv_list = [(k.to(device), v.to(device)) for k, v in kv_list]

    suffix = "plain"
    if args.cloak_block_size:
        kv_list, _ = cloak_kv_cache(kv_list, block_size=args.cloak_block_size, seed=args.cloak_seed, theta=args.cloak_theta)
        suffix = f"cloaked_b{args.cloak_block_size}_t{args.cloak_theta}"

    result = collision_attack(
        model,
        tokenizer,
        kv_list,
        target_ids,
        layer_idx=args.layer,
        use_v=(args.use == "V"),
        top_k_fraction=args.top_k_fraction,
        batch_size=args.batch_size,
        sigma=args.sigma,
        fixed_prefix_len=args.fixed_prefix,
        verbose=args.verbose,
    )

    recovered_text = tokenizer.decode(result.recovered_ids, skip_special_tokens=True)
    target_text = tokenizer.decode(result.target_ids, skip_special_tokens=True)
    rouge = rouge_l_f1(recovered_text, target_text)
    charf = char_overlap(recovered_text, target_text)

    out = {
        "prompt": meta["prompt"],
        "layer": result.layer,
        "use": "V" if result.used_v else "K",
        "token_accuracy": result.token_accuracy,
        "rouge_l_f1": rouge,
        "char_f1": charf,
        "recovered_text": recovered_text,
        "target_text": target_text,
        "n_sigma_hits": sum(1 for p in result.positions if p.sigma_hit),
        "n_positions_attacked": len(result.positions),
        "positions": [asdict(p) for p in result.positions],
        "top_k_fraction": args.top_k_fraction,
        "sigma": args.sigma,
        "batch_size": args.batch_size,
        "fixed_prefix": args.fixed_prefix,
        "cloak": None if args.cloak_block_size is None else {
            "block_size": args.cloak_block_size, "seed": args.cloak_seed, "theta": args.cloak_theta
        },
    }
    out_path = capture_dir / f"collision_{suffix}_L{args.layer}_{args.use}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "positions"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
