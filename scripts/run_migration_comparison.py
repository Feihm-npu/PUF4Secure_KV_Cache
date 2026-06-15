"""E3: KV-Cloak vs PUF-Cache under cache+software migration.

Threat model contrast. KV-Cloak's secret is a software key (a seed-derived set
of matrices/permutations/masks). PUF-Cache's secret is a device PUF root. We
measure layer-0 V-inversion top-1 token accuracy under four recovery conditions:

  * KV-Cloak, key withheld   -- attacker has the cloaked cache but not the key.
  * KV-Cloak, key migrated   -- attacker dumped the cache AND the software key
                                (e.g. from the same process memory) and decloaks.
  * PUF-Cache, wrong device  -- attacker has the cache + full software + a PUF on
                                a DIFFERENT device (different root) and recovers.
  * PUF-Cache, legit device  -- the originating device recovers normally.

The point: KV-Cloak confidentiality reduces to "keep the software key secret",
which fails when the key migrates with the cache; PUF-Cache stays bound even when
the attacker has the cache and the complete software stack, because the root is
physical and does not migrate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from puf4secure_kvcache.model_utils import load_model_and_tokenizer, kv_to_list, resolve_snapshot_path
from puf4secure_kvcache.attacks import invert_v_layer0
from puf4secure_kvcache.kv_cloak import cloak_kv_cache, decloak_kv_cache
from puf4secure_kvcache.puf_basis import protect_kv_cache, recover_kv_cache, ProtectionSpec
from puf4secure_kvcache.puf_sim import make_puf


@torch.inference_mode()
def _capture(model, tokenizer, prompt: str):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc, use_cache=True)
    kv = kv_to_list(out.past_key_values)
    return kv, enc["input_ids"][0]


def _v_top1(model, v0, target_ids) -> float:
    return float(invert_v_layer0(model, v0, target_ids).token_accuracy_top1)


def _v_rel_l2(a, b) -> float:
    da = (a.float() - b.float()).norm().item()
    return da / max(b.float().norm().item(), 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", type=Path,
                    default=Path("/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"))
    ap.add_argument("--snapshot", type=str, default=None)
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/migration_comparison.json"))
    ap.add_argument("--prompts-file", type=Path,
                    default=Path("experiments/prompts/synthetic_privacy_prompts.txt"))
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--cloak-seed", type=int, default=2026)
    ap.add_argument("--force-fp32", action="store_true")
    args = ap.parse_args()

    model_path = resolve_snapshot_path(args.model_cache_dir, args.snapshot)
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()
    if args.force_fp32:
        model.to(torch.float32)

    spec = ProtectionSpec(kind="givens")
    puf_owner = make_puf("device_owner")
    puf_attacker = make_puf("device_attacker")  # different root, same software

    prompts = [p.strip() for p in args.prompts_file.read_text(encoding="utf-8").splitlines() if p.strip()]
    rows = []
    for idx, prompt in enumerate(prompts):
        kv, target_ids = _capture(model, tokenizer, prompt)
        v_plain = kv[0][1]

        # --- KV-Cloak ---
        cloaked, keys = cloak_kv_cache(kv, block_size=args.block_size, seed=args.cloak_seed, theta=args.theta)
        decloaked = decloak_kv_cache(cloaked, keys, block_size=args.block_size)

        # --- PUF-Cache (cache-level orthogonal) ---
        protected = protect_kv_cache(kv, puf_owner, spec)
        wrong_dev = recover_kv_cache(protected, puf_attacker, spec)
        legit = recover_kv_cache(protected, puf_owner, spec)

        row = {
            "index": idx,
            "seq_len": int(target_ids.numel()),
            "plain_top1": _v_top1(model, v_plain, target_ids),
            "cloak_key_withheld_top1": _v_top1(model, cloaked[0][1], target_ids),
            "cloak_key_migrated_top1": _v_top1(model, decloaked[0][1], target_ids),
            "puf_attacker_no_recover_top1": _v_top1(model, protected[0][1], target_ids),
            "puf_wrong_device_top1": _v_top1(model, wrong_dev[0][1], target_ids),
            "puf_wrong_device_v_rel_l2": _v_rel_l2(wrong_dev[0][1], v_plain),
            "puf_legit_device_top1": _v_top1(model, legit[0][1], target_ids),
        }
        rows.append(row)
        print(f"[{idx}] plain={row['plain_top1']:.3f} "
              f"cloak(withheld)={row['cloak_key_withheld_top1']:.3f} "
              f"cloak(migrated)={row['cloak_key_migrated_top1']:.3f} "
              f"puf(wrongdev)={row['puf_wrong_device_top1']:.3f} "
              f"puf(legit)={row['puf_legit_device_top1']:.3f}", flush=True)

    keys_to_avg = [k for k in rows[0] if k not in ("index", "seq_len")]
    summary = {k: sum(r[k] for r in rows) / len(rows) for k in keys_to_avg}
    out = {
        "settings": vars(args) | {"model_path": str(model_path)},
        "summary": summary,
        "rows": rows,
    }
    args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
