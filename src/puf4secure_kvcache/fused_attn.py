"""Fused affine de-masking flash-decode attention kernel (P1 + P4b split-K).

A Triton flash-decoding kernel that regenerates the PUF affine mask in-register via
``tl.randn`` and subtracts it between the KV cache load and the QK/PV matmuls, so
de-masking piggybacks on the load that attention already performs (no extra HBM
traffic, no mask stored in HBM). The mask uses the SAME counter scheme as
``mask_gen`` (flat counter ``pos*(H*D)+head*D+ch``, key ``sa,sb``), so the masks
match the write path bit-for-bit.

The naive single-program-per-head kernel (grid=(B,Hq)) suffers low occupancy on
large SMs and degrades at long context (serial over N). The split-K (flash-
decoding) variant grid=(B,Hq,n_splits) parallelises across the sequence dimension
and merges partial (m_i, l_i, acc) via a second log-sum-exp reduction kernel.

Decode-only (seq_q = 1, contiguous cache). fp32 only (Theorem E).

    K_clean[j] = K~[j] - sigma*(randn(sa_k, ctr) + randn(sb_k, ctr))/sqrt(2)   # = K O
    out = softmax(q . K_clean^T / sqrt(d)) . V_clean                          # never forms plaintext
"""
from __future__ import annotations

import math

import torch

_kernel = None
_splitk_kernel = None
_reduce_kernel = None


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


def _get_splitk_kernel():
    global _splitk_kernel
    if _splitk_kernel is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _fused_demask_decode_splitk(
            Q, Kt, Vt, PartOut, PartM, PartL,
            sa_k, sb_k, sa_v, sb_v,
            scale, msd,
            N, D, HD, groups,
            split_size,                      # tokens per split (runtime)
            sqb, sqh, sqd,
            skb, skh, skn, skd,
            svb, svh, svn, svd,
            pob, poh, pos, pod,             # PartOut [B, Hq, n_splits, D]
            pmb, pmh, pms,                  # PartM   [B, Hq, n_splits]
            plb, plh, pls,                  # PartL   [B, Hq, n_splits]
            BLOCK_N: tl.constexpr, D_P: tl.constexpr,
        ):
            b = tl.program_id(0)
            h = tl.program_id(1)
            sid = tl.program_id(2)          # split index
            kvh = h // groups

            chunk_start = sid * split_size
            chunk_end = tl.minimum(chunk_start + split_size, N)

            offs_d = tl.arange(0, D_P)
            d_mask = offs_d < D
            q = tl.load(Q + b * sqb + h * sqh + offs_d * sqd, mask=d_mask, other=0.0)

            m_i = -float("inf")
            l_i = 0.0
            acc = tl.zeros([D_P], dtype=tl.float32)
            # empty split (chunk_start >= chunk_end) => loop body skipped, stores
            # m=-inf, l=0, acc=0 which the reduction correctly ignores.
            for start_n in range(chunk_start, chunk_end, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                n_mask = offs_n < chunk_end
                ctr = offs_n[:, None] * HD + kvh * D + offs_d[None, :]
                kp = Kt + b * skb + kvh * skh + offs_n[:, None] * skn + offs_d[None, :] * skd
                Ktile = tl.load(kp, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
                Mk = (tl.randn(sa_k, ctr) + tl.randn(sb_k, ctr)) * msd
                K = Ktile - Mk
                s = tl.sum(q[None, :] * K, axis=1) * scale
                s = tl.where(n_mask, s, -float("inf"))
                m_new = tl.maximum(m_i, tl.max(s, axis=0))
                p = tl.exp(s - m_new)
                alpha = tl.exp(m_i - m_new)
                l_i = l_i * alpha + tl.sum(p, axis=0)
                vp = Vt + b * svb + kvh * svh + offs_n[:, None] * svn + offs_d[None, :] * svd
                Vtile = tl.load(vp, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
                Mv = (tl.randn(sa_v, ctr) + tl.randn(sb_v, ctr)) * msd
                V = Vtile - Mv
                acc = acc * alpha + tl.sum(p[:, None] * V, axis=0)
                m_i = m_new

            # store UNNORMALIZED acc (reduction divides by l_f).
            tl.store(PartOut + b * pob + h * poh + sid * pos + offs_d * pod, acc, mask=d_mask)
            tl.store(PartM + b * pmb + h * pmh + sid * pms, m_i)
            tl.store(PartL + b * plb + h * plh + sid * pls, l_i)

        _splitk_kernel = _fused_demask_decode_splitk
    return _splitk_kernel


def _get_reduce_kernel():
    global _reduce_kernel
    if _reduce_kernel is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _reduce(PartOut, PartM, PartL, Out, D,
                    pob, poh, pos, pod,
                    pmb, pmh, pms,
                    plb, plh, pls,
                    ob, oh, od,
                    n_splits,
                    MAX_SPLITS: tl.constexpr, D_P: tl.constexpr):
            b = tl.program_id(0)
            h = tl.program_id(1)
            offs_d = tl.arange(0, D_P)
            d_mask = offs_d < D
            offs_s = tl.arange(0, MAX_SPLITS)
            s_mask = offs_s < n_splits

            ms = tl.load(PartM + b * pmb + h * pmh + offs_s * pms,
                         mask=s_mask, other=-float("inf"))
            ls = tl.load(PartL + b * plb + h * plh + offs_s * pls,
                         mask=s_mask, other=0.0)
            accs = tl.load(PartOut + b * pob + h * poh + offs_s[:, None] * pos + offs_d[None, :] * pod,
                           mask=s_mask[:, None] & d_mask[None, :], other=0.0)

            m_f = tl.max(ms, axis=0)
            alpha = tl.exp(ms - m_f)
            alpha = tl.where(s_mask, alpha, 0.0)
            l_f = tl.sum(ls * alpha, axis=0)
            acc_f = tl.sum(accs * alpha[:, None], axis=0)
            out = acc_f / l_f
            tl.store(Out + b * ob + h * oh + offs_d * od, out, mask=d_mask)

        _reduce_kernel = _reduce
    return _reduce_kernel


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _plan_splits(N: int, Hq: int, block_n: int = 64,
                 max_splits: int = 8, target_sms: int = 108) -> tuple[int, int]:
    """Choose n_splits to lift occupancy toward one SM-wave while keeping each
    split at least one block.  Returns (n_splits, split_size).

    Sweep-tuned on A100: n_splits up to 8 saturates 108 SMs at Hq=16 (128 progs)
    and Hq=32 (256 progs).  block_n=64 minimises per-step launch overhead across
    the 28-layer decode loop where the kernel is <10% of per-step time."""
    max_by_size = max(1, N // block_n)               # >= 1 block per split
    min_for_wave = max(1, (target_sms + Hq - 1) // Hq)
    n_splits = max(1, min(max_splits, max_by_size))
    n_splits = max(n_splits, min(min_for_wave, max_by_size, max_splits))
    split_size = (N + n_splits - 1) // n_splits       # ceil, partial last block masked
    return n_splits, split_size


@torch.inference_mode()
def fused_demask_decode(q: torch.Tensor, k_tilde: torch.Tensor, v_tilde: torch.Tensor,
                        seeds_k: tuple[int, int], seeds_v: tuple[int, int],
                        sigma: float, scale: float | None = None,
                        block_n: int = 64, n_splits: int | None = None) -> torch.Tensor:
    """Fused decode attention with in-kernel de-masking.

    q: [B, Hq, D] (single decode query per head); k_tilde/v_tilde: [B, Hkv, N, D]
    storing K O + M, V O + M. Returns [B, Hq, D]. Hq/Hkv may differ (GQA); the
    counter stride uses Hkv*D, matching mask_gen / the write path.

    When n_splits > 1 the kernel uses the split-K (flash-decoding) grid
    (B, Hq, n_splits) plus a log-sum-exp reduction kernel, which lifts SM
    occupancy at long context.  n_splits=None auto-selects via _plan_splits."""
    B, Hq, D = q.shape
    _, Hkv, N, Dk = k_tilde.shape
    assert Dk == D and v_tilde.shape == k_tilde.shape and Hq % Hkv == 0
    groups = Hq // Hkv
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    out = torch.empty((B, Hq, D), device=q.device, dtype=torch.float32)
    sa_k, sb_k = seeds_k
    sa_v, sb_v = seeds_v
    msd = float(sigma) / math.sqrt(2.0)
    D_P = _next_pow2(D)

    if n_splits is None:
        n_splits, split_size = _plan_splits(N, Hq, block_n)
    else:
        split_size = (N + n_splits - 1) // n_splits if n_splits > 0 else N
        n_splits = max(1, n_splits)

    sq0, sq1, sq2 = q.stride(0), q.stride(1), q.stride(2)
    k0, k1, k2, k3 = k_tilde.stride(0), k_tilde.stride(1), k_tilde.stride(2), k_tilde.stride(3)
    v0, v1, v2, v3 = v_tilde.stride(0), v_tilde.stride(1), v_tilde.stride(2), v_tilde.stride(3)
    so0, so1, so2 = out.stride(0), out.stride(1), out.stride(2)

    if n_splits <= 1:
        _get_kernel()[(B, Hq)](
            q, k_tilde, v_tilde, out,
            int(sa_k), int(sb_k), int(sa_v), int(sb_v),
            float(scale), msd,
            N, D, Hkv * D, groups,
            sq0, sq1, sq2, k0, k1, k2, k3, v0, v1, v2, v3, so0, so1, so2,
            BLOCK_N=block_n, D_P=D_P,
        )
        return out

    # --- split-K flash-decoding ---
    MAX_SPLITS = _next_pow2(n_splits)            # tl.arange needs a power-of-2 constexpr
    part_out = torch.empty((B, Hq, n_splits, D), device=q.device, dtype=torch.float32)
    part_m = torch.empty((B, Hq, n_splits), device=q.device, dtype=torch.float32)
    part_l = torch.empty((B, Hq, n_splits), device=q.device, dtype=torch.float32)
    po0, po1, po2, po3 = part_out.stride(0), part_out.stride(1), part_out.stride(2), part_out.stride(3)
    pm0, pm1, pm2 = part_m.stride(0), part_m.stride(1), part_m.stride(2)
    pl0, pl1, pl2 = part_l.stride(0), part_l.stride(1), part_l.stride(2)

    _get_splitk_kernel()[(B, Hq, n_splits)](
        q, k_tilde, v_tilde, part_out, part_m, part_l,
        int(sa_k), int(sb_k), int(sa_v), int(sb_v),
        float(scale), msd,
        N, D, Hkv * D, groups, split_size,
        sq0, sq1, sq2, k0, k1, k2, k3, v0, v1, v2, v3,
        po0, po1, po2, po3, pm0, pm1, pm2, pl0, pl1, pl2,
        BLOCK_N=block_n, D_P=D_P,
    )
    _get_reduce_kernel()[(B, Hq)](
        part_out, part_m, part_l, out, D,
        po0, po1, po2, po3, pm0, pm1, pm2, pl0, pl1, pl2, so0, so1, so2,
        n_splits, MAX_SPLITS=MAX_SPLITS, D_P=D_P,
    )
    return out
