"""P2 validation: end-to-end fused decode == eager-demask decode on a real model.

Greedy-decodes the affine-protected model with attn_backend "eager" vs
"triton_fused" and checks the generated tokens are identical. Prefill falls back
to eager de-masking; the fused Triton kernel handles every decode step (S==1),
regenerating and subtracting the PUF mask inside the attention kernel (GQA-aware).

Usage: python scripts/test_fused_e2e.py <model-cache-dir> [n_tokens]
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from puf4secure_kvcache.model_utils import resolve_snapshot_path
from puf4secure_kvcache.puf_attention import install_puf_attention, uninstall_puf_attention
from puf4secure_kvcache.puf_sim import make_puf


@torch.inference_mode()
def generate(model, enc, backend, n):
    install_puf_attention(model, make_puf("device_A"), fp32_cache=True,
                          affine_mask=True, mask_std=128.0, attn_backend=backend)
    out = model.generate(**enc, max_new_tokens=n, min_new_tokens=n, do_sample=False)
    uninstall_puf_attention(model)
    return out[0].tolist()[enc["input_ids"].shape[1]:]


def main():
    mp = resolve_snapshot_path(Path(sys.argv[1]))
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    tok = AutoTokenizer.from_pretrained(mp, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mp, local_files_only=True, trust_remote_code=True,
                                                 dtype="auto", device_map="cuda").eval()
    model.config._attn_implementation = "eager"
    model.to(torch.float32)
    enc = tok("Once upon a time in a small village there lived a curious",
              return_tensors="pt").to("cuda")
    te = generate(model, enc, "eager", n)
    tf = generate(model, enc, "triton_fused", n)
    match = te == tf
    print("eager:", te)
    print("fused:", tf)
    print(f"TOKEN MATCH ({n} tokens): {match}")
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
