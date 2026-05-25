"""KV-Cloak: reversible obfuscation of KV-cache tensors.

For each layer, each KV head, and each block of `block_size` consecutive token
positions we apply

    cloaked = S @ P @ (block + A)

where
  S : random orthogonal matrix of shape [block_size, block_size]
  P : random permutation of `block_size` rows
  A : additive beacon mask of shape [block_size, head_dim]

The transformation is exactly reversible given the (S, P, A) triple, which is
the secret key held by the legitimate runtime.

For positions that do not fit a full block at the tail of the sequence we apply
the same transform restricted to the actual number of rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CloakKeys:
    s_table: list[torch.Tensor]      # s_table[n] = orthogonal [n, n], n in [1..block_size]
    s_inv_table: list[torch.Tensor]
    mask: torch.Tensor               # [block_size, head_dim] (fp32)
    permutation: torch.Tensor        # [block_size]


def _orthogonal_matrix(size: int, *, device, generator: torch.Generator) -> torch.Tensor:
    rand = torch.randn(size, size, device=device, dtype=torch.float32, generator=generator)
    q, _ = torch.linalg.qr(rand)
    return q  # keep fp32 for reversibility


def generate_keys(
    block_size: int,
    head_dim: int,
    *,
    device,
    dtype,
    seed: int = 2026,
    theta: float = 1.0,
) -> CloakKeys:
    generator = torch.Generator(device=device if isinstance(device, str) else device.type)
    generator.manual_seed(seed)
    # One orthogonal matrix per possible block size [1..block_size].
    s_table: list[torch.Tensor] = [torch.empty(0)]  # index 0 unused placeholder
    s_inv_table: list[torch.Tensor] = [torch.empty(0)]
    for n in range(1, block_size + 1):
        s_n = _orthogonal_matrix(n, device=device, generator=generator)
        s_table.append(s_n)
        s_inv_table.append(s_n.T.contiguous())
    permutation = torch.randperm(block_size, device=device, generator=generator)
    mask = torch.randn(block_size, head_dim, device=device, dtype=torch.float32, generator=generator) * theta
    return CloakKeys(s_table=s_table, s_inv_table=s_inv_table, mask=mask, permutation=permutation)


def _local_permutation(perm_full: torch.Tensor, n: int) -> torch.Tensor:
    """Derive a permutation of [0..n-1] from the first n entries of a full permutation
    by taking their relative ranks.  When n == perm_full.shape[0], returns perm_full.
    """
    if n == perm_full.shape[0]:
        return perm_full
    head = perm_full[:n]
    return torch.argsort(torch.argsort(head))


def cloak_block(block: torch.Tensor, keys: CloakKeys) -> torch.Tensor:
    """Apply S P (block + A) on a [n, head_dim] block where n <= block_size.

    Computation is performed in fp32 for numerical stability, the result is
    cast back to the block's original dtype.
    """
    n = block.shape[0]
    out_dtype = block.dtype
    block32 = block.to(torch.float32)
    local_perm = _local_permutation(keys.permutation, n)
    masked = block32 + keys.mask[:n]
    permuted = masked[local_perm]
    return (keys.s_table[n] @ permuted).to(out_dtype)


def decloak_block(cloaked: torch.Tensor, keys: CloakKeys) -> torch.Tensor:
    n = cloaked.shape[0]
    out_dtype = cloaked.dtype
    cloaked32 = cloaked.to(torch.float32)
    local_perm = _local_permutation(keys.permutation, n)
    permuted = keys.s_inv_table[n] @ cloaked32
    inv_perm = torch.argsort(local_perm)
    unpermuted = permuted[inv_perm]
    return (unpermuted - keys.mask[:n]).to(out_dtype)


def _apply_to_tensor(tensor: torch.Tensor, keys: CloakKeys, block_size: int, *, encrypt: bool) -> torch.Tensor:
    """Apply cloak/decloak to a [1, num_kv_heads, seq, head_dim] tensor."""
    out = tensor.clone()
    seq = tensor.shape[2]
    fn = cloak_block if encrypt else decloak_block
    for start in range(0, seq, block_size):
        end = min(start + block_size, seq)
        for h in range(tensor.shape[1]):
            block = tensor[0, h, start:end, :]
            out[0, h, start:end, :] = fn(block, keys)
    return out


def cloak_kv_cache(
    kv_list: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    block_size: int,
    seed: int = 2026,
    theta: float = 1.0,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[CloakKeys, CloakKeys]]]:
    """Encrypt every layer's K and V using independent keys.

    Returns (cloaked_kv, keys_per_layer) where keys_per_layer[i] = (k_keys, v_keys).
    """
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    keys_per_layer: list[tuple[CloakKeys, CloakKeys]] = []
    for layer_idx, (k, v) in enumerate(kv_list):
        head_dim = k.shape[-1]
        k_keys = generate_keys(block_size, head_dim, device=k.device, dtype=k.dtype, seed=seed + 2 * layer_idx, theta=theta)
        v_keys = generate_keys(block_size, head_dim, device=v.device, dtype=v.dtype, seed=seed + 2 * layer_idx + 1, theta=theta)
        k_c = _apply_to_tensor(k, k_keys, block_size, encrypt=True)
        v_c = _apply_to_tensor(v, v_keys, block_size, encrypt=True)
        out.append((k_c, v_c))
        keys_per_layer.append((k_keys, v_keys))
    return out, keys_per_layer


def decloak_kv_cache(
    cloaked_kv: list[tuple[torch.Tensor, torch.Tensor]],
    keys_per_layer: list[tuple[CloakKeys, CloakKeys]],
    *,
    block_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for (k, v), (k_keys, v_keys) in zip(cloaked_kv, keys_per_layer):
        k_p = _apply_to_tensor(k, k_keys, block_size, encrypt=False)
        v_p = _apply_to_tensor(v, v_keys, block_size, encrypt=False)
        out.append((k_p, v_p))
    return out
