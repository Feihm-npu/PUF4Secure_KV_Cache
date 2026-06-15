"""Encrypt-at-rest AES baselines for the RQ9 head-to-head (B1/B2/B3).

Measures end-to-end decode throughput for:
  B1 plaintext            : model decode, no protection.
  B2 PUF-key + AES-CTR    : decrypt the KV cache on load each decode step.
  B3 PUF-key + AES-GCM    : same, with authentication (tag).

The crypto is run on a buffer the size of the *actual* KV cache at each decode
step (the cache grows every step), modelling decrypt-on-load over the whole cache
read by attention -- the same whole-cache-per-step cost profile as the affine
mask subtraction, so the comparison to B5 is apples-to-apples. This is what makes
AES "materialize a plaintext page": the cache must be decrypted to plaintext K/V
before attention, whereas orthogonal/affine PUF-Cache never produce plaintext K/V.

Throughput is reported as generated / (model_decode_s + crypto_s), i.e. crypto is
on the decode critical path. Orthogonal (B4) and affine (B5) decode tok/s come
from run_performance_eval.py in the same session for consistency.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from puf4secure_kvcache.model_utils import resolve_snapshot_path
from puf4secure_kvcache.puf_sim import make_puf

DEFAULT_MODEL = Path("/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")


def _kv_bytes(past) -> int:
    total = 0
    layers = past.to_legacy_cache() if hasattr(past, "to_legacy_cache") else past
    for k, v in layers:
        total += k.element_size() * k.nelement() + v.element_size() * v.nelement()
    return total


def _make_input(vocab, length, device, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(5, max(6, vocab - 1), (1, length), generator=g)).to(device)


@torch.inference_mode()
def run_once(model, input_ids, decode_tokens, key):
    ctr_cipher_nonce = os.urandom(16)
    gcm = AESGCM(key)
    gcm_nonce = os.urandom(12)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    out = model(input_ids=input_ids, use_cache=True)
    cache = out.past_key_values
    next_logits = out.logits[:, -1, :]

    model_decode_s = 0.0
    ctr_s = 0.0          # whole-cache-per-step policy (re-decrypt all, no plaintext kept)
    gcm_s = 0.0
    ctr_inc_s = 0.0      # incremental policy (decrypt only the new token; plaintext resident)
    gcm_inc_s = 0.0
    generated = 0
    prev_bytes = _kv_bytes(cache)
    for _ in range(decode_tokens):
        nxt = next_logits.argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        model_decode_s += time.perf_counter() - t0
        cache = out.past_key_values
        next_logits = out.logits[:, -1, :]
        generated += 1

        nbytes = _kv_bytes(cache)
        delta = max(nbytes - prev_bytes, 0)   # bytes of the new token's KV
        prev_bytes = nbytes
        whole = b"\0" * nbytes
        inc = b"\0" * delta
        # Whole-cache-per-step (B2/B3-naive): re-decrypt the entire cache, same
        # O(n)-per-step cost profile as the affine mask subtraction.
        t0 = time.perf_counter()
        Cipher(algorithms.AES(key), modes.CTR(ctr_cipher_nonce)).decryptor().update(whole)
        ctr_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        gcm.encrypt(gcm_nonce, whole, None)
        gcm_s += time.perf_counter() - t0
        # Incremental (B2/B3-resident): decrypt only the new token, keep the rest
        # as resident plaintext -- O(1)/step but materializes the full plaintext.
        t0 = time.perf_counter()
        Cipher(algorithms.AES(key), modes.CTR(ctr_cipher_nonce)).decryptor().update(inc)
        ctr_inc_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        gcm.encrypt(gcm_nonce, inc, None)
        gcm_inc_s += time.perf_counter() - t0

    return {
        "generated": generated,
        "model_decode_s": model_decode_s,
        "ctr_s": ctr_s, "gcm_s": gcm_s, "ctr_inc_s": ctr_inc_s, "gcm_inc_s": gcm_inc_s,
        "b1_plain_tok_s": generated / max(model_decode_s, 1e-12),
        "b2_aes_ctr_tok_s": generated / max(model_decode_s + ctr_s, 1e-12),
        "b3_aes_gcm_tok_s": generated / max(model_decode_s + gcm_s, 1e-12),
        "b2_aes_ctr_inc_tok_s": generated / max(model_decode_s + ctr_inc_s, 1e-12),
        "b3_aes_gcm_inc_tok_s": generated / max(model_decode_s + gcm_inc_s, 1e-12),
        "final_kv_bytes": _kv_bytes(cache),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--prefill-lengths", type=int, nargs="+", default=[256])
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--force-fp32", action="store_true")
    ap.add_argument("--device-id", default="device_A")
    ap.add_argument("--out", default="experiments/runs/aes_baseline_perf_qwen3.json")
    args = ap.parse_args()

    p = args.model_cache_dir
    model_path = p if (p / "config.json").exists() else resolve_snapshot_path(p, snapshot=args.snapshot)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                 trust_remote_code=True, dtype="auto", device_map="auto")
    model.eval()
    model.config._attn_implementation = "eager"
    if args.force_fp32:
        model.to(torch.float32)
    device = next(model.parameters()).device
    vocab = model.config.vocab_size
    # PUF-derived AES-256 key (the "PUF-key" baseline): 32 bytes from the root.
    key = make_puf(args.device_id).derive_bytes(purpose="aes_key", n_bytes=32)

    results = {"config": vars(args) | {"dtype": str(next(model.parameters()).dtype)}, "by_prefill": {}}
    for L in args.prefill_lengths:
        inp = _make_input(vocab, L, device, seed=1234)
        for _ in range(args.warmup):
            run_once(model, inp, args.decode_tokens, key)
        runs = [run_once(model, inp, args.decode_tokens, key) for _ in range(args.repeats)]
        agg = {}
        for k in ("b1_plain_tok_s", "b2_aes_ctr_tok_s", "b3_aes_gcm_tok_s",
                  "b2_aes_ctr_inc_tok_s", "b3_aes_gcm_inc_tok_s",
                  "model_decode_s", "ctr_s", "gcm_s", "ctr_inc_s", "gcm_inc_s"):
            vals = [r[k] for r in runs]
            agg[k] = sum(vals) / len(vals)
        agg["final_kv_bytes"] = runs[-1]["final_kv_bytes"]
        b1 = agg["b1_plain_tok_s"]
        for key, col in [("b2_overhead_pct", "b2_aes_ctr_tok_s"), ("b3_overhead_pct", "b3_aes_gcm_tok_s"),
                         ("b2_inc_overhead_pct", "b2_aes_ctr_inc_tok_s"), ("b3_inc_overhead_pct", "b3_aes_gcm_inc_tok_s")]:
            agg[key] = 100 * (b1 - agg[col]) / b1
        results["by_prefill"][str(L)] = agg
        print(f"prefill={L} kvMB={agg['final_kv_bytes']/2**20:.1f} | B1 {b1:.2f} | "
              f"whole-cache: B2-CTR {agg['b2_aes_ctr_tok_s']:.2f} ({agg['b2_overhead_pct']:+.1f}%) "
              f"B3-GCM {agg['b3_aes_gcm_tok_s']:.2f} ({agg['b3_overhead_pct']:+.1f}%) | "
              f"incremental(resident): B2-CTR {agg['b2_aes_ctr_inc_tok_s']:.2f} ({agg['b2_inc_overhead_pct']:+.1f}%) "
              f"B3-GCM {agg['b3_aes_gcm_inc_tok_s']:.2f} ({agg['b3_inc_overhead_pct']:+.1f}%)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
