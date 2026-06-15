"""P1 validation: fused affine de-masking flash-decode kernel vs eager reference.

Confirms the Triton kernel (regenerating the PUF mask in-register via tl.randn,
same counter scheme as mask_gen) matches eager (K~ - M) + SDPA to fp32 rounding,
and that the in-kernel de-mask recovers the true K O / V O exactly.
"""
import math
import torch
import torch.nn.functional as F

from puf4secure_kvcache.mask_gen import counter_seeds, counter_mask_rows
from puf4secure_kvcache.fused_attn import fused_demask_decode
from puf4secure_kvcache.puf_sim import make_puf


def check(puf, H, D, N, sigma=128.0, tol=1e-3):
    dev = torch.device("cuda")
    sk = counter_seeds(puf, 0, "K_affine_mask|wn=wA")
    sv = counter_seeds(puf, 0, "V_affine_mask|wn=wA")
    q = torch.randn(H, D, device=dev)
    Kclean = torch.randn(H, N, D, device=dev)        # = K O (what attention should consume)
    Vclean = torch.randn(H, N, D, device=dev)
    Mk = counter_mask_rows(*sk, 0, N, H, D, sigma, dev).permute(1, 0, 2).contiguous()
    Mv = counter_mask_rows(*sv, 0, N, H, D, sigma, dev).permute(1, 0, 2).contiguous()
    Kt, Vt = Kclean + Mk, Vclean + Mv                # stored masked cache K~,V~
    out_ref = F.scaled_dot_product_attention(q[:, None, :], Kt - Mk, Vt - Mv).squeeze(1)
    out = fused_demask_decode(q, Kt, Vt, sk, sv, sigma)
    out_clean = F.scaled_dot_product_attention(q[:, None, :], Kclean, Vclean).squeeze(1)
    d_ref = (out - out_ref).abs().max().item()
    d_clean = (out - out_clean).abs().max().item()
    ok = d_ref < tol and d_clean < tol
    print(f"H={H} D={D} N={N}: max|fused-ref|={d_ref:.2e} max|fused-cleanSDPA|={d_clean:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    puf = make_puf("device_A")
    torch.manual_seed(0)
    cases = [(8, 128, 200), (8, 64, 200), (8, 128, 512), (4, 128, 37), (8, 128, 1024)]
    allok = all(check(puf, *c) for c in cases)
    print("ALL PASS" if allok else "SOME FAILED")


if __name__ == "__main__":
    main()
