"""M2 baseline: algebraic inversion of first-layer K / V."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from puf4secure_kvcache.attacks import invert_k_layer0, invert_v_layer0
from puf4secure_kvcache.kv_io import load_capture
from puf4secure_kvcache.model_utils import load_model_and_tokenizer, resolve_snapshot_path


def run(capture_dir: Path, top_k: int = 5) -> dict:
    meta, kv_list = load_capture(capture_dir)
    target_ids = torch.tensor(meta["input_ids"], dtype=torch.long)

    model_path = resolve_snapshot_path()
    model, tokenizer = load_model_and_tokenizer(model_path)
    device = next(model.parameters()).device

    k0 = kv_list[0][0].to(device)
    v0 = kv_list[0][1].to(device)

    v_res = invert_v_layer0(model, v0, target_ids, top_k=top_k)
    k_res = invert_k_layer0(model, k0, target_ids, top_k=top_k)

    decoded_v_top1 = tokenizer.decode(v_res.recovered_ids[:, 0].tolist(), skip_special_tokens=True)
    decoded_k_top1 = tokenizer.decode(k_res.recovered_ids[:, 0].tolist(), skip_special_tokens=True)
    target_text = tokenizer.decode(target_ids.tolist(), skip_special_tokens=True)

    result = {
        "prompt": meta["prompt"],
        "target_text": target_text,
        "V_top1_text": decoded_v_top1,
        "K_top1_text": decoded_k_top1,
        "V_acc_top1": v_res.token_accuracy_top1,
        "V_acc_topk": v_res.token_accuracy_topk,
        "K_acc_top1": k_res.token_accuracy_top1,
        "K_acc_topk": k_res.token_accuracy_topk,
        "top_k": top_k,
        "V_recovered_per_pos": [
            [tokenizer.decode([int(tok)]) for tok in row] for row in v_res.recovered_ids.tolist()
        ],
        "K_recovered_per_pos": [
            [tokenizer.decode([int(tok)]) for tok in row] for row in k_res.recovered_ids.tolist()
        ],
        "target_per_pos": [tokenizer.decode([int(tok)]) for tok in target_ids.tolist()],
    }

    with open(capture_dir / "inversion_result.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M2 inversion attack on a captured KV directory.")
    parser.add_argument("--capture", required=True, help="Capture directory (containing kv.pt and meta.json).")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    res = run(Path(args.capture), top_k=args.top_k)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_per_pos")}, indent=2, ensure_ascii=False))
    print("target per pos:", res["target_per_pos"])
    print("V top1   :", [r[0] for r in res["V_recovered_per_pos"]])
    print("K top1   :", [r[0] for r in res["K_recovered_per_pos"]])


if __name__ == "__main__":
    main()
