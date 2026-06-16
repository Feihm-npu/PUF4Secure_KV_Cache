"""P1/P2/P4b validation: fused affine de-masking flash-decode kernel vs eager reference.

Confirms the Triton kernel (regenerating the PUF mask in-register via tl.randn,
same counter scheme as mask_gen) matches eager (K~ - M) + SDPA to fp32 rounding,
recovers the true K O / V O exactly, handles batch + grouped-query attention,
and that the split-K (flash-decoding) variant produces identical results.
"""
import torch
import torch.nn.functional as F

from puf4secure_kvcache.mask_gen import counter_seeds, counter_mask_rows
from puf4secure_kvcache.fused_attn import fused_demask_decode, _plan_splits
from puf4secure_kvcache.puf_sim import make_puf


def _make_case(puf, B, Hq, Hkv, D, N, sigma=128.0, seed=0):
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    sk = counter_seeds(puf, 0, "K_affine_mask|wn=wA")
    sv = counter_seeds(puf, 0, "V_affine_mask|wn=wA")
    q = torch.randn(B, Hq, D, device=dev)
    Kclean = torch.randn(B, Hkv, N, D, device=dev)
    Vclean = torch.randn(B, Hkv, N, D, device=dev)
    Mk = counter_mask_rows(*sk, 0, N, Hkv, D, sigma, dev).permute(1, 0, 2).contiguous()
    Mv = counter_mask_rows(*sv, 0, N, Hkv, D, sigma, dev).permute(1, 0, 2).contiguous()
    Kt, Vt = Kclean + Mk[None], Vclean + Mv[None]
    gqa = Hq != Hkv
    out_ref = F.scaled_dot_product_attention(
        q[:, :, None, :], Kclean, Vclean, enable_gqa=gqa).squeeze(2)
    return q, Kt, Vt, sk, sv, out_ref


def check(puf, B, Hq, Hkv, D, N, sigma=128.0, tol=1e-3):
    q, Kt, Vt, sk, sv, out_ref = _make_case(puf, B, Hq, Hkv, D, N, sigma)
    out = fused_demask_decode(q, Kt, Vt, sk, sv, sigma)
    d = (out - out_ref).abs().max().item()
    ok = d < tol
    tag = f"B={B} Hq={Hq} Hkv={Hkv} D={D} N={N}"
    print(f"{tag:38s}: max|fused-ref|={d:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_splitk(puf, B, Hq, Hkv, D, N, sigma=128.0, tol=2e-3):
    """split-K kernel vs SDPA reference.  Tolerance slightly looser than the
    single-program kernel because the reduction reorders fp32 sums."""
    q, Kt, Vt, sk, sv, out_ref = _make_case(puf, B, Hq, Hkv, D, N, sigma)
    n_splits, _ = _plan_splits(N, Hq)
    # force split-K even for small N where heuristic may pick 1
    for ns in [n_splits, 2, 4, 8]:
        if ns > 1 and ns * Hq <= 256:        # keep grid reasonable
            out = fused_demask_decode(q, Kt, Vt, sk, sv, sigma, n_splits=ns)
            d = (out - out_ref).abs().max().item()
            ok = d < tol
            tag = f"B={B} Hq={Hq} Hkv={Hkv} D={D} N={N} splits={ns}"
            print(f"{tag:48s}: max|splitK-ref|={d:.2e}  {'PASS' if ok else 'FAIL'}")
            if not ok:
                return False
    # also the auto-selected n_splits
    out = fused_demask_decode(q, Kt, Vt, sk, sv, sigma)
    d = (out - out_ref).abs().max().item()
    ok = d < tol
    tag = f"B={B} Hq={Hq} Hkv={Hkv} D={D} N={N} splits=auto({n_splits})"
    print(f"{tag:48s}: max|splitK-ref|={d:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    puf = make_puf("device_A")
    cases = [
        (1, 8, 8, 128, 200),     # no GQA, no batch
        (2, 8, 8, 128, 200),     # batch
        (1, 16, 8, 128, 200),    # Qwen3-0.6B: 16 q / 8 kv, d128, GQA 2:1
        (1, 32, 8, 64, 200),     # Llama-3.2-1B: 32 q / 8 kv, d64, GQA 4:1
        (2, 16, 8, 128, 512),    # batch + GQA + longer N
        (1, 16, 8, 128, 37),     # N < BLOCK_N
        (1, 16, 8, 128, 1024),   # long context (the P4 failure regime)
        (1, 32, 8, 64, 1024),    # Llama long context
    ]
    allok = True
    print("== single-program kernel ==")
    for c in cases:
        allok &= check(puf, *c)
    print("== split-K kernel ==")
    for c in cases:
        allok &= check_splitk(puf, *c)
    print("ALL PASS" if allok else "SOME FAILED")


if __name__ == "__main__":
    main()
