"""Architecture-independent candidate-secret collision attack.

The attacker knows the prompt context and a finite candidate set for a secret
(e.g. a 6-digit verification code). For each candidate, the attacker runs the
public model locally and compares the candidate KV cache to the leaked cache.
This tests cache matching without relying on square value projections.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from puf4secure_kvcache.model_utils import kv_to_list, resolve_snapshot_path
from puf4secure_kvcache.puf_attention import install_puf_attention, uninstall_puf_attention
from puf4secure_kvcache.puf_sim import make_puf


DEFAULT_MODEL_CACHE = Path("/home/feihm/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B")
SECRET_TYPES = ["verification_code", "phone", "ssn", "email", "api_key", "codename"]


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
    return {"available": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "devices": devices}


def _rand_digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def _rand_word(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))


def _rand_secret(kind: str, rng: random.Random) -> str:
    if kind == "verification_code":
        return _rand_digits(rng, 6)
    if kind == "phone":
        return f"{_rand_digits(rng, 3)}-{_rand_digits(rng, 3)}-{_rand_digits(rng, 4)}"
    if kind == "ssn":
        return f"{_rand_digits(rng, 3)}-{_rand_digits(rng, 2)}-{_rand_digits(rng, 4)}"
    if kind == "email":
        return f"{_rand_word(rng, 5)}.{_rand_word(rng, 4)}{rng.randrange(100):02d}@{_rand_word(rng, 6)}.example"
    if kind == "api_key":
        return "sk-" + _rand_word(rng, 8) + _rand_digits(rng, 8)
    if kind == "codename":
        return f"{_rand_word(rng, 5)}-{_rand_word(rng, 6)}"
    raise ValueError(f"unknown secret type: {kind}")


def _prompt(secret: str, kind: str = "verification_code") -> str:
    if kind == "verification_code":
        return f"For this account recovery request, the temporary verification code is {secret}. Keep it private."
    if kind == "phone":
        return f"For this account recovery request, the callback phone number is {secret}. Keep it private."
    if kind == "ssn":
        return f"For this account recovery request, the synthetic account identifier is {secret}. Keep it private."
    if kind == "email":
        return f"For this account recovery request, the private contact email is {secret}. Keep it private."
    if kind == "api_key":
        return f"For this account recovery request, the internal API key is {secret}. Keep it private."
    if kind == "codename":
        return f"For this account recovery request, the confidential codename is {secret}. Keep it private."
    raise ValueError(f"unknown secret type: {kind}")


def _parse_secret_types(raw: str) -> list[str]:
    if raw.strip() == "all":
        return list(SECRET_TYPES)
    kinds = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(kinds) - set(SECRET_TYPES))
    if unknown:
        raise ValueError(f"unknown secret types: {unknown}; valid values are {SECRET_TYPES} or 'all'")
    return kinds


def _make_records(tokenizer, samples: int, candidates: int, seed: int, secret_types: list[str]) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for idx in range(samples):
        kind = secret_types[idx % len(secret_types)]
        true_secret = _rand_secret(kind, rng)
        true_len = len(tokenizer(_prompt(true_secret, kind), add_special_tokens=True)["input_ids"])
        cand = [true_secret]
        seen = {true_secret}
        attempts = 0
        while len(cand) < candidates and attempts < candidates * 1000:
            attempts += 1
            secret = _rand_secret(kind, rng)
            if secret in seen:
                continue
            if len(tokenizer(_prompt(secret, kind), add_special_tokens=True)["input_ids"]) != true_len:
                continue
            seen.add(secret)
            cand.append(secret)
        if len(cand) < candidates:
            raise RuntimeError(f"only built {len(cand)} same-length candidates for record {idx} ({kind})")
        rng.shuffle(cand)
        records.append({
            "index": idx,
            "secret_type": kind,
            "true_secret": true_secret,
            "true_code": true_secret,
            "candidates": cand,
            "true_candidate_index": cand.index(true_secret),
            "prompt": _prompt(true_secret, kind),
        })
    return records


@torch.inference_mode()
def _capture_kv(model, tokenizer, prompt: str, keep_layers=None):
    """Capture KV cache. When ``keep_layers`` is given, only those layer indices
    are retained (as a dict), which avoids holding all layers of thousands of
    candidate caches in CPU RAM during large-candidate sweeps."""
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc, use_cache=True)
    pairs = kv_to_list(out.past_key_values)
    if keep_layers is None:
        return [(k.detach().cpu(), v.detach().cpu()) for k, v in pairs]
    keep = set(keep_layers)
    return {i: (k.detach().cpu(), v.detach().cpu()) for i, (k, v) in enumerate(pairs) if i in keep}


def _distance(a, b, layers: list[int], include_k: bool, include_v: bool, distance_mode: str) -> float:
    vals = []
    for layer in layers:
        for idx, include in [(0, include_k), (1, include_v)]:
            if not include:
                continue
            x = a[layer][idx].float()
            y = b[layer][idx].float()
            n = min(x.shape[2], y.shape[2])
            xb = x[:, :, :n]
            yb = y[:, :, :n]
            if distance_mode == "l2":
                diff = xb - yb
            elif distance_mode == "norm_l2":
                # per-token L2 norm: diag of the Gram matrix; orthogonal-invariant.
                diff = xb.norm(dim=-1) - yb.norm(dim=-1)
            elif distance_mode == "gram_l2":
                # full pairwise inner-product (Gram) matrix X X^T per head;
                # invariant under X -> X O for orthogonal O. Strictly stronger
                # than norm_l2 (which is only its diagonal).
                gx = xb @ xb.transpose(-1, -2)
                gy = yb @ yb.transpose(-1, -2)
                diff = gx - gy
            elif distance_mode == "svd_l2":
                # singular-value spectrum of the [n, D] per-head matrix;
                # sigma(X O) = sigma(X) for orthogonal O, so also invariant.
                sx = torch.linalg.svdvals(xb.float())
                sy = torch.linalg.svdvals(yb.float())
                diff = sx - sy
            else:
                raise ValueError(f"unknown distance mode: {distance_mode}")
            vals.append(float(diff.norm().item()))
    return sum(vals) / max(len(vals), 1)


def _rank_record(candidate_kvs, leaked_kv, true_idx: int, layers: list[int], include_k: bool,
                 include_v: bool, distance_mode: str) -> dict:
    scores = [_distance(kv, leaked_kv, layers, include_k, include_v, distance_mode) for kv in candidate_kvs]
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank = order.index(true_idx) + 1
    return {
        "pred_candidate_index": order[0],
        "true_rank": rank,
        "top1": int(order[0] == true_idx),
        "mrr": 1.0 / rank,
        "true_score": scores[true_idx],
        "best_score": scores[order[0]],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    ap.add_argument("--snapshot", type=str, default=None)
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/candidate_secret_collision.json"))
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument("--secret-types", default="verification_code", help="Comma-separated secret types or 'all'.")
    ap.add_argument("--modes", nargs="+", choices=["plain", "protected_native"], default=["plain", "protected_native"])
    ap.add_argument("--layers", default="0", help="Comma-separated layer indices, or '0,mid,last'.")
    ap.add_argument("--distance-mode", choices=["l2", "norm_l2", "gram_l2", "svd_l2"], default="l2")
    ap.add_argument("--distance-modes", default=None,
                    help="Comma-separated distance modes computed in one capture pass "
                         "(reuses candidate caches). Overrides --distance-mode when set.")
    ap.add_argument("--include-k", action="store_true")
    ap.add_argument("--include-v", action="store_true")
    ap.add_argument("--force-fp32", action="store_true")
    ap.add_argument("--fp32-cache", action="store_true")
    ap.add_argument("--norm-blind", action="store_true")
    ap.add_argument("--norm-log-range", type=float, default=1.0)
    ap.add_argument("--unit-norm-cache", action="store_true")
    ap.add_argument("--nonorth-log-range", type=float, default=0.0)
    ap.add_argument("--affine-mask", action="store_true")
    ap.add_argument("--mask-std", type=float, default=4.0)
    ap.add_argument("--device-id", default="device_A")
    return ap.parse_args()


def main() -> None:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    args = parse_args()
    if not args.include_k and not args.include_v:
        args.include_k = True
        args.include_v = True
    model_path = _model_path(args.model_cache_dir, args.snapshot)
    out = {"settings": vars(args) | {"model_path": str(model_path)}, "resource_before": _cuda_summary(), "modes": {}}
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
        model.config._attn_implementation = "eager"
        if args.force_fp32:
            model.to(torch.float32)
        secret_types = _parse_secret_types(args.secret_types)
        records = _make_records(tokenizer, args.samples, args.candidates, args.seed, secret_types)
        layer_tokens = args.layers.split(",")
        layers = []
        for tok in layer_tokens:
            tok = tok.strip()
            if tok == "mid":
                layers.append(model.config.num_hidden_layers // 2)
            elif tok == "last":
                layers.append(model.config.num_hidden_layers - 1)
            else:
                layers.append(int(tok))
        out["model"] = {
            "model_type": getattr(model.config, "model_type", None),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
            "num_attention_heads": getattr(model.config, "num_attention_heads", None),
            "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
            "dtype": str(next(model.parameters()).dtype),
        }
        # Candidate caches are public-model computations and are shared by modes.
        keep_layers = set(layers)
        candidate_cache_bank = []
        for rec in records:
            candidate_cache_bank.append([
                _capture_kv(model, tokenizer, _prompt(secret, rec["secret_type"]), keep_layers)
                for secret in rec["candidates"]
            ])

        for mode in args.modes:
            installed = False
            mode_rows = []
            try:
                if mode == "protected_native":
                    install_puf_attention(
                        model,
                        make_puf(args.device_id),
                        fp32_cache=args.fp32_cache,
                        norm_blind=args.norm_blind,
                        norm_log_range=args.norm_log_range,
                        unit_norm_cache=args.unit_norm_cache,
                        nonorth_log_range=args.nonorth_log_range,
                        affine_mask=args.affine_mask,
                        mask_std=args.mask_std,
                    )
                    installed = True
                leaked = [_capture_kv(model, tokenizer, rec["prompt"], keep_layers) for rec in records]
            finally:
                if installed:
                    uninstall_puf_attention(model)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            dist_modes = (
                [m.strip() for m in args.distance_modes.split(",") if m.strip()]
                if args.distance_modes else [args.distance_mode]
            )
            dist_results = {}
            for dm in dist_modes:
                mode_rows = []
                for rec, candidate_kvs, leaked_kv in zip(records, candidate_cache_bank, leaked):
                    rank = _rank_record(
                        candidate_kvs, leaked_kv, rec["true_candidate_index"],
                        layers, args.include_k, args.include_v, dm,
                    )
                    mode_rows.append({k: rec[k] for k in ("index", "secret_type", "true_secret")} | rank)
                top1 = sum(r["top1"] for r in mode_rows) / max(len(mode_rows), 1)
                mrr = sum(r["mrr"] for r in mode_rows) / max(len(mode_rows), 1)
                by_type = {}
                for kind in sorted({r["secret_type"] for r in mode_rows}):
                    subset = [r for r in mode_rows if r["secret_type"] == kind]
                    by_type[kind] = {
                        "samples": len(subset),
                        "top1": sum(r["top1"] for r in subset) / max(len(subset), 1),
                        "mrr": sum(r["mrr"] for r in subset) / max(len(subset), 1),
                    }
                dist_results[dm] = {
                    "summary": {"samples": len(mode_rows), "top1": top1, "mrr": mrr, "by_type": by_type},
                    "records": mode_rows,
                }
                print(f"{mode} [{dm}]", json.dumps(dist_results[dm]["summary"], indent=2), flush=True)
            # Back-compat: when a single distance mode is requested, keep the old
            # schema (out["modes"][mode]["summary"]); otherwise key by distance mode.
            out["modes"][mode] = dist_results[dist_modes[0]] if len(dist_modes) == 1 else dist_results
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
