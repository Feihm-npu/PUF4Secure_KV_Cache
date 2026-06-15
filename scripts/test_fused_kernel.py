"""P1/P2 validation: fused affine de-masking flash-decode kernel vs eager reference.

Confirms the Triton kernel (regenerating the PUF mask in-register via tl.randn,
same counter scheme as mask_gen) matches eager (K~ - M) + SDPA to fp32 rounding,
recovers the true K O / V O exactly, and handles batch + grouped-query attention.
"""
import torch
import torch.nn.functional as F

from puf4secure_kvcache.mask_gen import counter_seeds, counter_mask_rows
from puf4secure_kvcache.fused_attn import fused_demask_decode
from puf4secure_kvcache.puf_sim import make_puf


def check(puf, B, Hq, Hkv, D, N, sigma=128.0, tol=1e-3):
    dev = torch.device("cuda")
    sk = counter_seeds(puf, 0, "K_affine_mask|wn=wA")
    sv = counter_seeds(puf, 0, "V_affine_mask|wn=wA")
    q = torch.randn(B, Hq, D, device=dev)
    Kclean = torch.randn(B, Hkv, N, D, device=dev)        # = K O (what attention should consume)
    Vclean = torch.randn(B, Hkv, N, D, device=dev)
    Mk = counter_mask_rows(*sk, 0, N, Hkv, D, sigma, dev).permute(1, 0, 2).contiguous()  # [Hkv,N,D]
    Mv = counter_mask_rows(*sv, 0, N, Hkv, D, sigma, dev).permute(1, 0, 2).contiguous()
    Kt, Vt = Kclean + Mk[None], Vclean + Mv[None]         # stored masked cache [B,Hkv,N,D]
    gqa = Hq != Hkv
    out_ref = F.scaled_dot_product_attention(
        q[:, :, None, :], Kclean, Vclean, enable_gqa=gqa).squeeze(2)   # [B,Hq,D]
    out = fused_demask_decode(q, Kt, Vt, sk, sv, sigma)
    d = (out - out_ref).abs().max().item()
    ok = d < tol
    tag = f"B={B} Hq={Hq} Hkv={Hkv} D={D} N={N}"
    print(f"{tag:38s}: max|fused-ref|={d:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    puf = make_puf("device_A")
    torch.manual_seed(0)
    cases = [
        (1, 8, 8, 128, 200),     # no GQA, no batch
        (2, 8, 8, 128, 200),     # batch
        (1, 16, 8, 128, 200),    # Qwen3-0.6B: 16 q / 8 kv, d128, GQA 2:1
        (1, 32, 8, 64, 200),     # Llama-3.2-1B: 32 q / 8 kv, d64, GQA 4:1
        (2, 16, 8, 128, 512),    # batch + GQA + longer N
        (1, 16, 8, 128, 37),     # N < BLOCK_N
    ]
    allok = all(check(puf, *c) for c in cases)
    print("ALL PASS" if allok else "SOME FAILED")


if __name__ == "__main__":
    main()
