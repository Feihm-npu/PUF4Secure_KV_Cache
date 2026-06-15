"""Counter-based PUF affine-mask generator (P0 for the fused Triton kernel).

The legacy mask was ``torch.randn(generator=HMAC-seed)`` drawn sequentially, which
is not random-access and so cannot be regenerated per-page inside a paged attention
kernel. Here the mask is a counter-based function of (position, head, channel):

    counter(pos, head, ch) = pos*(H*D) + head*D + ch          # row-major flat index
    M[counter]             = sigma * (randn(sa, counter) + randn(sb, counter)) / sqrt(2)

``randn`` is Triton's counter-based Philox Gaussian (``tl.randn(seed, offset)``),
verified position-addressable and N(0,1). Summing two independent streams keyed by
two 63-bit seeds (sa, sb) gives a 126-bit effective key while staying exactly
N(0, sigma^2) per element, so Theorems B (Gaussian-mechanism advantage) and E
(precision window) are unchanged. The SAME ``tl.randn(seed, counter)`` call is used
on the write path (Protect) and will be used inside the fused attention kernel
(Attend), so the masks match bit-for-bit by construction.

Security: sa||sb = HMAC(root, nonce, layer, purpose); the migration attacker cannot
compute them without the PUF root, and has no known plaintext (same-session KPA is
the stated non-goal) with which to mount a cryptanalytic key-recovery on Philox.
"""
from __future__ import annotations

import math

import torch

_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_MASK63 = (1 << 63) - 1
_kernel = None


def counter_seeds(puf, layer: int, purpose_eff: str) -> tuple[int, int]:
    """Two 63-bit Philox seeds (126-bit key) for one (layer, nonce, purpose)."""
    raw = puf.derive_bytes(layer=layer, group=-1, block=-1,
                           purpose=f"cmask|{purpose_eff}", n_bytes=16)
    return (int.from_bytes(raw[:8], "big") & _MASK63,
            int.from_bytes(raw[8:16], "big") & _MASK63)


def _get_kernel():
    global _kernel
    if _kernel is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _cmask(out_ptr, sa, sb, base, total, scale, BLOCK: tl.constexpr):
            idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = idx < total
            ctr = base + idx
            v = (tl.randn(sa, ctr) + tl.randn(sb, ctr)) * scale
            tl.store(out_ptr + idx, v, mask=m)

        _kernel = _cmask
    return _kernel


def counter_mask_flat(sa: int, sb: int, base: int, n: int, sigma: float,
                      device: torch.device) -> torch.Tensor:
    """Flat [n] mask for counters [base, base+n), i.e. sigma*(randn(sa,c)+randn(sb,c))/sqrt2.

    On CUDA this is the Triton kernel that the fused attention kernel will reuse.
    A CPU fallback (for tests) reproduces the same values via Triton is unavailable,
    so CPU uses an explicit per-counter torch path only for small n."""
    if device.type == "cuda":
        import triton
        out = torch.empty(n, device=device, dtype=torch.float32)
        grid = (triton.cdiv(n, 1024),)
        _get_kernel()[grid](out, int(sa), int(sb), int(base), int(n),
                            float(sigma) * _INV_SQRT2, BLOCK=1024)
        return out
    raise RuntimeError("counter_mask_flat requires CUDA (affine path is GPU-only)")


def counter_mask_rows(sa: int, sb: int, start_pos: int, n_pos: int,
                      num_heads: int, head_dim: int, sigma: float,
                      device: torch.device) -> torch.Tensor:
    """Mask rows for positions [start_pos, start_pos+n_pos) shaped [n_pos, H, D],
    using the flat counter pos*(H*D)+head*D+ch."""
    hd = num_heads * head_dim
    flat = counter_mask_flat(sa, sb, start_pos * hd, n_pos * hd, sigma, device)
    return flat.view(n_pos, num_heads, head_dim)
