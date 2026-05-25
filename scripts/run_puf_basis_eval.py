"""PUF-basis defense evaluation.

Sweeps variants P0-P5 from docs/PUF_BASIS_TECHNICAL_PLAN.md and reports
inversion / collision / injection / fidelity / cross-device metrics.

Output: experiments/runs/puf_basis_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from puf4secure_kvcache.attacks import (
    collision_attack,
    injection_attack,
    invert_v_layer0,
)
from puf4secure_kvcache.kv_io import load_capture
from puf4secure_kvcache.metrics import char_overlap, rouge_l_f1
from puf4secure_kvcache.model_utils import load_model_and_tokenizer, resolve_snapshot_path
from puf4secure_kvcache.puf_basis import (
    LayoutSpec,
    ProtectionSpec,
    protect_kv_cache,
    recover_kv_cache,
    relative_l2_error,
)
from puf4secure_kvcache.puf_sim import make_puf
from puf4secure_kvcache.secret_metrics import score_leakage, secret_for_prompt


# ---- variants ------------------------------------------------------------

def variant_specs() -> dict[str, ProtectionSpec]:
    """Return the named variants we run.

    P2_v_signed/givens/hadamard/qr : V-only with various matrix kinds.
    P3_kv_* : K+V with matched kind on both paths.
    P4_kv_givens_hadamard_blocklayout : K+V with block-row layout (the working
        hypothesis: Givens for K, Hadamard for V).
    """
    out = {}
    out["P2_V_signed_perm"] = ProtectionSpec(kind="signed_perm", include_k=False, include_v=True)
    out["P2_V_givens"]      = ProtectionSpec(kind="givens",      include_k=False, include_v=True)
    out["P2_V_hadamard"]    = ProtectionSpec(kind="hadamard",    include_k=False, include_v=True)
    out["P2_V_qr"]          = ProtectionSpec(kind="qr",          include_k=False, include_v=True)

    out["P3_KV_givens"]    = ProtectionSpec(kind="givens",   include_k=True, include_v=True)
    out["P3_KV_hadamard"]  = ProtectionSpec(kind="hadamard", include_k=True, include_v=True)
    out["P3_KV_givens_hadamard"] = ProtectionSpec(
        kind="givens", kind_k="givens", kind_v="hadamard",
        include_k=True, include_v=True,
    )

    out["P4_KV_givens_hadamard_block"] = ProtectionSpec(
        kind="givens", kind_k="givens", kind_v="hadamard",
        include_k=True, include_v=True,
        layout=LayoutSpec(kind="block", block_size=8),
    )
    out["P4_KV_givens_hadamard_row"] = ProtectionSpec(
        kind="givens", kind_k="givens", kind_v="hadamard",
        include_k=True, include_v=True,
        layout=LayoutSpec(kind="row"),
    )
    # Plan v1: prefer pure Givens (fastest) with layout / session refresh.
    out["P4_KV_givens_row"] = ProtectionSpec(
        kind="givens", include_k=True, include_v=True,
        layout=LayoutSpec(kind="row"),
    )
    out["P4_KV_givens_block"] = ProtectionSpec(
        kind="givens", include_k=True, include_v=True,
        layout=LayoutSpec(kind="block", block_size=8),
    )
    # P5 specs are identical to P3/P4 but the eval harness will swap session_nonce.
    out["P5_KV_givens_session_refresh"] = ProtectionSpec(
        kind="givens", include_k=True, include_v=True,
        layout=LayoutSpec(kind="none"),
    )
    out["P5_KV_givens_row_session_refresh"] = ProtectionSpec(
        kind="givens", include_k=True, include_v=True,
        layout=LayoutSpec(kind="row"),
    )
    return out


# ---- helpers -------------------------------------------------------------

def _to_device(kv, device):
    return [(k.to(device), v.to(device)) for k, v in kv]


def _attack_suite(model, tokenizer, kv_list, target_ids, target_text, *,
                  do_inversion=True, do_collision=True, do_injection=True,
                  injection_instruction="Repeat the previous content."):
    res = {}
    if do_inversion:
        inv = invert_v_layer0(model, kv_list[0][1], target_ids)
        res["inversion_V_L0_top1"] = inv.token_accuracy_top1
    if do_collision:
        col = collision_attack(model, tokenizer, kv_list, target_ids,
                               layer_idx=0, use_v=True, top_k_fraction=0.01,
                               batch_size=256, sigma=3.0, fixed_prefix_len=0)
        col_text = tokenizer.decode(col.recovered_ids, skip_special_tokens=True)
        res["collision_V_L0_token_acc"] = col.token_accuracy
        res["collision_V_L0_rouge_l"] = rouge_l_f1(col_text, target_text)
        res["collision_V_L0_text"] = col_text
    if do_injection:
        inj = injection_attack(model, tokenizer, kv_list, target_ids,
                               instruction=injection_instruction, max_new_tokens=48)
        res["injection_rouge_l"] = rouge_l_f1(inj.generated_text, target_text)
        res["injection_char_f1"] = char_overlap(inj.generated_text, target_text)
        res["injection_text"] = inj.generated_text
    return res


def evaluate_capture(model, tokenizer, capture_dir: Path,
                     variants: dict[str, ProtectionSpec],
                     *, do_attacks: bool = True):
    meta, kv_list = load_capture(capture_dir)
    device = next(model.parameters()).device
    kv_list = _to_device(kv_list, device)
    target_ids = torch.tensor(meta["input_ids"], dtype=torch.long)
    target_text = tokenizer.decode(meta["input_ids"], skip_special_tokens=True)
    secret = secret_for_prompt(meta["prompt"])

    puf_A = make_puf("device_A")
    puf_B = make_puf("device_B")  # wrong-device

    # ---- P0 plaintext baseline ----
    record = {"prompt": meta["prompt"], "seq_len": meta["seq_len"], "variants": {}}
    if do_attacks:
        p0 = _attack_suite(model, tokenizer, kv_list, target_ids, target_text)
        if secret is not None and "injection_text" in p0:
            p0["injection_leakage"] = score_leakage(p0["injection_text"], secret)
        record["variants"]["P0_plain"] = p0

    # ---- protected variants ----
    for name, spec in variants.items():
        # Session refresh: legitimate device uses a fresh per-variant nonce.
        if name.endswith("_session_refresh"):
            fresh_nonce = os.urandom(16)
            local_puf_A = make_puf("device_A", session_nonce=fresh_nonce)
            local_puf_B = make_puf("device_B", session_nonce=fresh_nonce)
        else:
            local_puf_A, local_puf_B = puf_A, puf_B

        t0 = time.perf_counter()
        protected = protect_kv_cache(kv_list, local_puf_A, spec)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_protect = time.perf_counter() - t0

        # Fidelity: same-device recovery
        t0 = time.perf_counter()
        recovered = recover_kv_cache(protected, local_puf_A, spec)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_recover = time.perf_counter() - t0
        fidelity = relative_l2_error(kv_list, recovered)

        # Wrong-device recovery
        wrong_recovered = recover_kv_cache(protected, local_puf_B, spec)
        wrong_fidelity = relative_l2_error(kv_list, wrong_recovered)

        rec = {
            "spec": {
                "kind_k": spec.kind_k, "kind_v": spec.kind_v,
                "include_k": spec.include_k, "include_v": spec.include_v,
                "layout": spec.layout.kind, "block_size": spec.layout.block_size,
                "session_refresh": name.endswith("_session_refresh"),
            },
            "protect_s": t_protect,
            "recover_s": t_recover,
            "fidelity_same_device": fidelity,
            "fidelity_wrong_device": wrong_fidelity,
        }

        if do_attacks:
            # Attack the protected cache directly (attacker view)
            atk_prot = _attack_suite(model, tokenizer, protected, target_ids, target_text)
            if secret is not None and "injection_text" in atk_prot:
                atk_prot["injection_leakage"] = score_leakage(atk_prot["injection_text"], secret)
            rec["attacks_on_protected"] = atk_prot

            # Wrong-device replay
            atk_wrong = _attack_suite(model, tokenizer, wrong_recovered, target_ids, target_text,
                                      do_collision=False)
            if secret is not None and "injection_text" in atk_wrong:
                atk_wrong["injection_leakage"] = score_leakage(atk_wrong["injection_text"], secret)
            rec["attacks_on_wrong_device_recovered"] = atk_wrong

            # Legitimate same-device recovery (utility check)
            inj_same = injection_attack(model, tokenizer, recovered, target_ids,
                                        instruction="Repeat the previous content.",
                                        max_new_tokens=48)
            rec["same_device_decloak_injection_text"] = inj_same.generated_text
            rec["same_device_decloak_injection_rouge_l"] = rouge_l_f1(inj_same.generated_text, target_text)
            if secret is not None:
                rec["same_device_decloak_injection_leakage"] = score_leakage(inj_same.generated_text, secret)

        record["variants"][name] = rec

    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("experiments/runs"))
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/puf_basis_summary.json"))
    ap.add_argument("--no-attacks", action="store_true",
                    help="Only run fidelity (faster, useful when iterating).")
    ap.add_argument("--variants", type=str, default="",
                    help="Comma-separated subset of variant names to run.")
    args = ap.parse_args()

    capture_dirs = sorted(p for p in args.runs_dir.iterdir()
                          if p.is_dir() and (p / "meta.json").exists())

    model, tokenizer = load_model_and_tokenizer(resolve_snapshot_path())
    variants = variant_specs()
    if args.variants:
        keep = set(args.variants.split(","))
        variants = {k: v for k, v in variants.items() if k in keep}

    out = []
    for c in capture_dirs:
        print(f"=== {c.name} ===", flush=True)
        rec = evaluate_capture(model, tokenizer, c, variants, do_attacks=not args.no_attacks)
        out.append(rec)
        # Print compact view
        print(json.dumps({
            "prompt": rec["prompt"],
            "variants": {
                k: {
                    "fid_same": v["fidelity_same_device"],
                    "fid_wrong": v["fidelity_wrong_device"],
                    **({kk: vv for kk, vv in v.get("attacks_on_protected", {}).items()
                        if not kk.endswith("_text")} if "attacks_on_protected" in v else {}),
                }
                for k, v in rec["variants"].items() if isinstance(v, dict) and "spec" in v
            },
        }, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
