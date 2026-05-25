"""Procrustes profiling attack against PUF-basis protected KV cache.

Threat model:
  - The attacker has many (plaintext, protected) cache pairs for a fixed PUF
    session, where the attacker chooses the plaintexts.
  - The attacker tries to recover the per-(layer, head) orthogonal basis O
    such that  K_plain @ O ~= K_protected, then strips the basis from a
    target protected cache.

For each (layer L, kv-head h), aggregate row vectors across all profiling
prompts and solve the orthogonal Procrustes problem:

      O_hat = argmin_O || X @ O - Y ||_F     s.t. O^T O = I
      O_hat = U V^T   where  U S V^T = svd(X^T Y)

Two settings are evaluated:
  - aligned    : rows of X and Y correspond position-by-position
  - unaligned  : Y rows are unknown permutation of X rows; we Procrustes-match
                 the *row distributions* (e.g. by sorting on the first PC).
                 The aligned setting is the strongest attacker, the unaligned
                 setting models the layout=row defense.
"""
from __future__ import annotations

from dataclasses import dataclass

import math
import torch


def _solve_procrustes(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Closed-form orthogonal Procrustes for X @ O ~= Y. Inputs [N, d]."""
    M = X.T @ Y
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return (U @ Vh).contiguous()


@dataclass
class ProfilingResult:
    layer: int
    head: int
    target: str               # "K" or "V"
    n_rows: int
    O_recovery_error: float   # || O_hat O_true^T - I ||_F / sqrt(d)
    residual: float           # || X O_hat - Y ||_F / || Y ||_F
    transfer_residual: float  # same on held-out session/prompts


def fit_basis(
    plain_pairs: list[tuple[torch.Tensor, torch.Tensor]],     # list of (K, V) plaintext caches per prompt
    protected_pairs: list[tuple[torch.Tensor, torch.Tensor]], # same prompts, protected
    *,
    layer: int,
    head: int,
    target: str = "V",
) -> torch.Tensor:
    """Fit O_hat from a batch of (plain, protected) caches for a given (layer, head, target).

    Each pair item is a per-prompt list[(K, V)]; we use only [layer][use_v]
    and head-index `head`.  Returns O_hat [d, d].
    """
    Xs = []
    Ys = []
    for plain, prot in zip(plain_pairs, protected_pairs):
        sel = 1 if target == "V" else 0
        plain_t = plain[layer][sel][0, head].to(torch.float32)        # [seq, d]
        prot_t = prot[layer][sel][0, head].to(torch.float32)
        # If shapes mismatch, truncate to common length.
        n = min(plain_t.shape[0], prot_t.shape[0])
        Xs.append(plain_t[:n])
        Ys.append(prot_t[:n])
    X = torch.cat(Xs, dim=0)
    Y = torch.cat(Ys, dim=0)
    return _solve_procrustes(X, Y)


def evaluate_recovery(
    O_hat: torch.Tensor,
    O_true: torch.Tensor,
) -> float:
    d = O_hat.shape[0]
    I = torch.eye(d, dtype=O_hat.dtype, device=O_hat.device)
    err = (O_hat @ O_true.T - I).norm().item() / math.sqrt(d)
    return err


def residual_fit(plain: torch.Tensor, protected: torch.Tensor, O_hat: torch.Tensor) -> float:
    """Relative residual || plain @ O_hat - protected ||_F / ||protected||_F."""
    diff = (plain.to(torch.float32) @ O_hat - protected.to(torch.float32)).norm().item()
    denom = protected.to(torch.float32).norm().item()
    return diff / max(denom, 1e-12)


def strip_basis_from_cache(
    protected: list[tuple[torch.Tensor, torch.Tensor]],
    O_hats_per_head: dict[tuple[int, int, str], torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Apply O_hat^T to each (layer, head, target) of a protected cache."""
    out = []
    for layer_idx, (k, v) in enumerate(protected):
        k_out = torch.empty_like(k)
        v_out = torch.empty_like(v)
        for h in range(k.shape[1]):
            kh = k[0, h].to(torch.float32)
            vh = v[0, h].to(torch.float32)
            O_k = O_hats_per_head.get((layer_idx, h, "K"))
            O_v = O_hats_per_head.get((layer_idx, h, "V"))
            if O_k is not None:
                kh = kh @ O_k.T
            if O_v is not None:
                vh = vh @ O_v.T
            k_out[0, h] = kh.to(k.dtype)
            v_out[0, h] = vh.to(v.dtype)
        out.append((k_out, v_out))
    return out
