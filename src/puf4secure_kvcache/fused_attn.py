"""Fused affine de-masking flash-decode attention kernel (P1, TODO 3).

A Triton flash-decoding kernel that regenerates the PUF affine mask in-register via
``tl.randn`` and subtracts it between the KV cache load and the QK/PV matmuls, so
de-masking piggybacks on the load that attention already performs (no extra HBM
traffic, no mask stored in HBM). The mask uses the SAME counter scheme as
``mask_gen`` (flat counter ``pos*(H*D)+head*D+ch``, key ``sa,sb``), so the masks
match the write path bit-for-bit.

This is the standalone decode kernel (seq_q = 1, contiguous cache). Paged layout
and HF integration are P2. fp32 only (the affine path is fp32, Theorem E).

    K_clean[j] = K~[j] - sigma*(randn(sa_k, ctr) + randn(sb_k, ctr))/sqrt(2)   # = K O
    out = softmax(q . K_clean^T / sqrt(d)) . V_clean                          # never forms plaintext
"""
from __future__ import annotations

import math

import torch

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _fused_demask_decode(
            Q, Kt, Vt, Out,
            sa_k, sb_k, sa_v, sb_v,
            scale, msd,                       # msd = sigma / sqrt(2)
            N, D, HD, groups,                 # seq len, head dim, Hkv*D (counter stride/pos), GQA groups
            sqb, sqh, sqd,                    # Q strides   [B, Hq, D]
            skb, skh, skn, skd,               # Kt strides  [B, Hkv, N, D]
            svb, svh, svn, svd,
            sob, soh, sod,
            BLOCK_N: tl.constexpr, D_P: tl.constexpr,
        ):
            b = tl.program_id(0)
            h = tl.program_id(1)              # query head
            kvh = h // groups                 # shared KV head (GQA)
            offs_d = tl.arange(0, D_P)
            d_mask = offs_d < D
            q = tl.load(Q + b * sqb + h * sqh + offs_d * sqd, mask=d_mask, other=0.0)

            m_i = -float("inf")
            l_i = 0.0
            acc = tl.zeros([D_P], dtype=tl.float32)
            for start in range(0, N, BLOCK_N):
                offs_n = start + tl.arange(0, BLOCK_N)
                n_mask = offs_n < N
                ctr = offs_n[:, None] * HD + kvh * D + offs_d[None, :]    # counter uses the KV head
                # --- K: load masked page, regenerate mask, de-mask ---
                kp = Kt + b * skb + kvh * skh + offs_n[:, None] * skn + offs_d[None, :] * skd
                Ktile = tl.load(kp, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
                Mk = (tl.randn(sa_k, ctr) + tl.randn(sb_k, ctr)) * msd
                K = Ktile - Mk
                s = tl.sum(q[None, :] * K, axis=1) * scale               # [BLOCK_N]
                s = tl.where(n_mask, s, -float("inf"))
                m_new = tl.maximum(m_i, tl.max(s, axis=0))
                p = tl.exp(s - m_new)
                alpha = tl.exp(m_i - m_new)
                l_i = l_i * alpha + tl.sum(p, axis=0)
                # --- V: load masked page, regenerate mask, de-mask, accumulate ---
                vp = Vt + b * svb + kvh * svh + offs_n[:, None] * svn + offs_d[None, :] * svd
                Vtile = tl.load(vp, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
                Mv = (tl.randn(sa_v, ctr) + tl.randn(sb_v, ctr)) * msd
                V = Vtile - Mv
                acc = acc * alpha + tl.sum(p[:, None] * V, axis=0)        # [D_P]
                m_i = m_new
            out = acc / l_i
            tl.store(Out + b * sob + h * soh + offs_d * sod, out, mask=d_mask)

        _kernel = _fused_demask_decode
    return _kernel


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


@torch.inference_mode()
def fused_demask_decode(q: torch.Tensor, k_tilde: torch.Tensor, v_tilde: torch.Tensor,
                        seeds_k: tuple[int, int], seeds_v: tuple[int, int],
                        sigma: float, scale: float | None = None,
                        block_n: int = 64) -> torch.Tensor:
    """Fused decode attention with in-kernel de-masking.

    q: [B, Hq, D] (single decode query per head); k_tilde/v_tilde: [B, Hkv, N, D]
    storing K O + M, V O + M. Returns [B, Hq, D]. Hq/Hkv may differ (GQA); the
    counter stride uses Hkv*D, matching mask_gen / the write path."""
    import triton
    B, Hq, D = q.shape
    _, Hkv, N, Dk = k_tilde.shape
    assert Dk == D and v_tilde.shape == k_tilde.shape and Hq % Hkv == 0
    groups = Hq // Hkv
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    out = torch.empty((B, Hq, D), device=q.device, dtype=torch.float32)
    sa_k, sb_k = seeds_k
    sa_v, sb_v = seeds_v
    _get_kernel()[(B, Hq)](
        q, k_tilde, v_tilde, out,
        int(sa_k), int(sb_k), int(sa_v), int(sb_v),
        float(scale), float(sigma) / math.sqrt(2.0),
        N, D, Hkv * D, groups,
        q.stride(0), q.stride(1), q.stride(2),
        k_tilde.stride(0), k_tilde.stride(1), k_tilde.stride(2), k_tilde.stride(3),
        v_tilde.stride(0), v_tilde.stride(1), v_tilde.stride(2), v_tilde.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_N=block_n, D_P=_next_pow2(D),
    )
    return out
