"""PUF-derived orthogonal basis generation and KV-cache protect/recover.

Implements five orthogonal matrix structures from
`docs/PUF_BASIS_TECHNICAL_PLAN.md` section 5:

  - "signed_perm"  : O = P D   (permutation x +/-1 sign diagonal)
  - "givens"       : block-diagonal 2x2 rotations on consecutive pairs
  - "hadamard"     : O = D1 H D2 P_h  (normalized Walsh-Hadamard sandwich)
  - "householder"  : product of k Householder reflectors
  - "qr"           : dense orthogonal from random matrix QR

Each generator takes a seed (int) from the PUF.  All matrices are orthogonal
(O^T O = I) so inversion is exactly O^T.

The public KV interface is:

    protect_kv_cache(kv_list, puf, *, kind, include_k, include_v, layout)
    recover_kv_cache(protected, puf, *, kind, include_k, include_v, layout)

For non-migratability tests, simply call `recover_kv_cache` with a different
PUF (wrong_device mode) and observe that the recovered tensors are wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch

from .puf_sim import PUFSim


# -------------------------------------------------------------------------
# Orthogonal matrix generators
# -------------------------------------------------------------------------

def _seed_generator(seed: int, device: torch.device) -> torch.Generator:
    """Build a torch.Generator on `device` initialized with `seed`.

    Some torch builds only allow Generator on CPU; we therefore generate
    randomness on CPU and move to the target device.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & ((1 << 63) - 1))
    return g


def _signed_perm(d: int, seed: int, device, dtype=torch.float32) -> torch.Tensor:
    g = _seed_generator(seed, device)
    perm = torch.randperm(d, generator=g)
    signs = (torch.randint(0, 2, (d,), generator=g, dtype=torch.int64) * 2 - 1).to(dtype)
    O = torch.zeros(d, d, dtype=dtype)
    O[torch.arange(d), perm] = signs
    return O.to(device)


def _givens(d: int, seed: int, device, dtype=torch.float32) -> torch.Tensor:
    """Block-diagonal 2x2 rotations on pairs (0,1), (2,3), ...

    For odd d, the last coordinate is left untouched.
    """
    g = _seed_generator(seed, device)
    n_pairs = d // 2
    thetas = (torch.rand(n_pairs, generator=g, dtype=torch.float32) * (2 * math.pi))
    O = torch.eye(d, dtype=dtype)
    cos = thetas.cos().to(dtype)
    sin = thetas.sin().to(dtype)
    idx = torch.arange(n_pairs) * 2
    O[idx, idx] = cos
    O[idx + 1, idx + 1] = cos
    O[idx, idx + 1] = -sin
    O[idx + 1, idx] = sin
    return O.to(device)


def _hadamard_matrix(d: int) -> torch.Tensor:
    """Return the unnormalized Hadamard matrix of order d via Sylvester recursion.

    Requires d to be a power of two.
    """
    assert d > 0 and (d & (d - 1)) == 0, "Hadamard requires power-of-two dim"
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H


def _hadamard(d: int, seed: int, device, dtype=torch.float32) -> torch.Tensor:
    """O = D1 H D2 P / sqrt(d).  Random diagonal signs and a permutation."""
    g = _seed_generator(seed, device)
    H = _hadamard_matrix(d) / math.sqrt(d)
    d1 = (torch.randint(0, 2, (d,), generator=g, dtype=torch.int64) * 2 - 1).to(torch.float32)
    d2 = (torch.randint(0, 2, (d,), generator=g, dtype=torch.int64) * 2 - 1).to(torch.float32)
    perm = torch.randperm(d, generator=g)
    P = torch.eye(d, dtype=torch.float32)[perm]
    O = torch.diag(d1) @ H @ torch.diag(d2) @ P
    return O.to(device=device, dtype=dtype)


def _householder(d: int, seed: int, device, dtype=torch.float32, *, k: int = 4) -> torch.Tensor:
    """Product of `k` Householder reflectors built from random unit vectors."""
    g = _seed_generator(seed, device)
    O = torch.eye(d, dtype=torch.float32)
    for _ in range(k):
        v = torch.randn(d, generator=g, dtype=torch.float32)
        v = v / v.norm().clamp_min(1e-12)
        # I - 2 v v^T
        H = torch.eye(d) - 2.0 * torch.outer(v, v)
        O = O @ H
    return O.to(device=device, dtype=dtype)


def _qr(d: int, seed: int, device, dtype=torch.float32) -> torch.Tensor:
    g = _seed_generator(seed, device)
    A = torch.randn(d, d, generator=g, dtype=torch.float32)
    Q, R = torch.linalg.qr(A)
    # Fix sign ambiguity for determinism
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    return Q.to(device=device, dtype=dtype)


_KINDS = {
    "signed_perm": _signed_perm,
    "givens": _givens,
    "hadamard": _hadamard,
    "householder": _householder,
    "qr": _qr,
}


def make_orthogonal(kind: str, d: int, seed: int, device, dtype=torch.float32, **kwargs) -> torch.Tensor:
    if kind not in _KINDS:
        raise ValueError(f"Unknown kind {kind!r}. Choices: {list(_KINDS)}")
    return _KINDS[kind](d, seed, device, dtype, **kwargs)


# -------------------------------------------------------------------------
# Row layout permutations (token-position scrambling)
# -------------------------------------------------------------------------

@dataclass
class LayoutSpec:
    kind: str = "none"            # "none" | "row" | "block"
    block_size: int = 16


def _layout_permutation(seq_len: int, layout: LayoutSpec, puf: PUFSim,
                        layer: int, group: int) -> Optional[torch.Tensor]:
    """Return a length-`seq_len` LongTensor permutation, or None for identity."""
    if layout.kind == "none":
        return None
    seed = puf.derive_seed(layer=layer, group=group, block=-1, purpose=f"layout:{layout.kind}")
    g = _seed_generator(seed, torch.device("cpu"))
    if layout.kind == "row":
        return torch.randperm(seq_len, generator=g)
    if layout.kind == "block":
        n_full = seq_len // layout.block_size
        tail = seq_len - n_full * layout.block_size
        if n_full <= 1:
            return None
        block_perm = torch.randperm(n_full, generator=g)
        out = []
        for b in block_perm.tolist():
            out.extend(range(b * layout.block_size, (b + 1) * layout.block_size))
        if tail > 0:
            out.extend(range(n_full * layout.block_size, seq_len))
        return torch.tensor(out, dtype=torch.long)
    raise ValueError(f"Unknown layout kind: {layout.kind!r}")


# -------------------------------------------------------------------------
# KV-cache protect/recover
# -------------------------------------------------------------------------

@dataclass
class ProtectionSpec:
    kind: str = "givens"                  # matrix kind for O / U
    kind_k: Optional[str] = None          # override for K path
    kind_v: Optional[str] = None          # override for V path
    include_k: bool = True
    include_v: bool = True
    layout: LayoutSpec = None
    householder_k: int = 4                # for kind == "householder"

    def __post_init__(self):
        if self.layout is None:
            self.layout = LayoutSpec()
        if self.kind_k is None:
            self.kind_k = self.kind
        if self.kind_v is None:
            self.kind_v = self.kind


def _basis_for(puf: PUFSim, kind: str, head_dim: int, layer: int, group: int,
               purpose: str, device, dtype, householder_k: int = 4) -> torch.Tensor:
    seed = puf.derive_seed(layer=layer, group=group, block=-1, purpose=purpose)
    if kind == "householder":
        return make_orthogonal(kind, head_dim, seed, device, dtype, k=householder_k)
    return make_orthogonal(kind, head_dim, seed, device, dtype)


@torch.inference_mode()
def protect_kv_cache(
    kv_list: list[tuple[torch.Tensor, torch.Tensor]],
    puf: PUFSim,
    spec: ProtectionSpec,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Apply PUF-derived orthogonal basis (and optional row layout) to KV cache.

    For each layer L and each KV-head h (kv_group_id = h), we compute
      K' [t, :] = (P_layout @ K)[t, :] @ O_{L,h}
      V' [t, :] = (P_layout @ V)[t, :] @ U_{L,h}

    Returns a new list of (K', V') with the same shapes as kv_list.
    """
    out = []
    for layer_idx, (k, v) in enumerate(kv_list):
        # shape: [1, kv_h, seq, head_dim]
        device = k.device
        head_dim = k.shape[-1]
        kv_heads = k.shape[1]
        seq_len = k.shape[2]
        k_dtype = k.dtype
        v_dtype = v.dtype

        layout_perm = _layout_permutation(seq_len, spec.layout, puf,
                                          layer=layer_idx, group=-1)

        k_out = torch.empty_like(k)
        v_out = torch.empty_like(v)

        for h in range(kv_heads):
            k_h = k[0, h].to(torch.float32)
            v_h = v[0, h].to(torch.float32)
            if layout_perm is not None:
                k_h = k_h[layout_perm]
                v_h = v_h[layout_perm]
            if spec.include_k:
                O = _basis_for(puf, spec.kind_k, head_dim, layer_idx, h,
                               "K", device, torch.float32,
                               householder_k=spec.householder_k)
                k_h = k_h @ O
            if spec.include_v:
                U = _basis_for(puf, spec.kind_v, head_dim, layer_idx, h,
                               "V", device, torch.float32,
                               householder_k=spec.householder_k)
                v_h = v_h @ U
            k_out[0, h] = k_h.to(k_dtype)
            v_out[0, h] = v_h.to(v_dtype)

        out.append((k_out, v_out))
    return out


@torch.inference_mode()
def recover_kv_cache(
    protected: list[tuple[torch.Tensor, torch.Tensor]],
    puf: PUFSim,
    spec: ProtectionSpec,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Inverse of protect_kv_cache (using the PUF passed in)."""
    out = []
    for layer_idx, (k, v) in enumerate(protected):
        device = k.device
        head_dim = k.shape[-1]
        kv_heads = k.shape[1]
        seq_len = k.shape[2]
        k_dtype = k.dtype
        v_dtype = v.dtype

        layout_perm = _layout_permutation(seq_len, spec.layout, puf,
                                          layer=layer_idx, group=-1)
        if layout_perm is not None:
            inv_layout = torch.argsort(layout_perm)
        else:
            inv_layout = None

        k_out = torch.empty_like(k)
        v_out = torch.empty_like(v)
        for h in range(kv_heads):
            k_h = k[0, h].to(torch.float32)
            v_h = v[0, h].to(torch.float32)
            if spec.include_k:
                O = _basis_for(puf, spec.kind_k, head_dim, layer_idx, h,
                               "K", device, torch.float32,
                               householder_k=spec.householder_k)
                k_h = k_h @ O.T
            if spec.include_v:
                U = _basis_for(puf, spec.kind_v, head_dim, layer_idx, h,
                               "V", device, torch.float32,
                               householder_k=spec.householder_k)
                v_h = v_h @ U.T
            if inv_layout is not None:
                k_h = k_h[inv_layout]
                v_h = v_h[inv_layout]
            k_out[0, h] = k_h.to(k_dtype)
            v_out[0, h] = v_h.to(v_dtype)
        out.append((k_out, v_out))
    return out


# -------------------------------------------------------------------------
# Diagnostics
# -------------------------------------------------------------------------

def relative_l2_error(a: list[tuple[torch.Tensor, torch.Tensor]],
                      b: list[tuple[torch.Tensor, torch.Tensor]]) -> dict:
    num_k = 0.0
    den_k = 0.0
    num_v = 0.0
    den_v = 0.0
    for (ka, va), (kb, vb) in zip(a, b):
        num_k += (ka.float() - kb.float()).norm().item() ** 2
        den_k += ka.float().norm().item() ** 2
        num_v += (va.float() - vb.float()).norm().item() ** 2
        den_v += va.float().norm().item() ** 2
    return {
        "K_rel_l2": math.sqrt(num_k / max(den_k, 1e-12)),
        "V_rel_l2": math.sqrt(num_v / max(den_v, 1e-12)),
    }
