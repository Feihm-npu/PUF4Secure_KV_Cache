"""Level-3 PUF-bound attention wrapper for RoPE decoder-only models.

This monkey-patches every attention module to apply per-(layer, kv-head)
orthogonal rotations derived from a `PUFSim` to Q, K, V *inside* the attention
block, so the contents stored in the past-KV cache are never in the model-native
basis. Mathematically the wrapped attention is identical to the plain one:

    q' = q O_k      k' = k O_k      v' = v O_v
    softmax(q' k'^T / sqrt d) v' = softmax(q k^T / sqrt d) v O_v
    apply O_v^T to attn_output  =>  identical to plain output

So legitimate same-device decoding does not require an explicit decloak: the
cache is already device-bound. An attacker exfiltrating the cache without the
PUF cannot invert K nor V (no O_k, O_v available).

Constraints / scope:
  * Eager attention only (we wrap eager_attention_forward semantics).
  * Supported model families: Qwen3, Qwen2, and Llama.
  * Single-PUF, single-session install: the rotations are fixed when
    `install_puf_attention(...)` is called. Switching session requires
    `uninstall_puf_attention(...)` + reinstall.
  * GQA layout is handled: each q-head reuses the O_k/O_v of its kv-head.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional performance dependency
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

from .puf_basis import _basis_for, _seed_generator
from .puf_sim import PUFSim


_ORIG_FORWARDS: dict[int, Callable] = {}


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to query-head count.

    Mirrors the helper used by Llama/Qwen attention implementations. Keeping it
    local avoids family-specific imports in the wrapped fp32 attention path.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                          cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _fp32_eager_attention(module, q, k, v, attention_mask, scaling):
    """Eager attention computed entirely in fp32 (cast back at the end).

    Mirrors the Llama/Qwen eager attention routine but forces fp32 throughout
    the QK^T, softmax, and AV matmuls. The rotated
    PUF basis makes bf16 rounding error asymmetric vs. the plain path, so we
    keep the entire attention inner loop in fp32.
    """
    out_dtype = q.dtype
    qf = q.float()
    kf = _repeat_kv(k, module.num_key_value_groups).float()
    vf = _repeat_kv(v, module.num_key_value_groups).float()

    attn_weights = torch.matmul(qf, kf.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : kf.shape[-2]].float()
        attn_weights = attn_weights + causal_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(attn_weights, vf)              # fp32
    attn_output = attn_output.transpose(1, 2).contiguous()    # [B, S, Hq, d]
    return attn_output, attn_weights.to(out_dtype)


def _sdpa_attention(module, q, k, v, attention_mask, scaling):
    """Attention via ``F.scaled_dot_product_attention`` (the fused-kernel API).

    Demonstrates that the orthogonal PUF rotations compose with the production
    attention entry point used by FlashAttention / memory-efficient backends:
    the rotations are applied to Q/K/V *outside* this call, so the fused kernel
    operates on the rotated tensors unchanged. We keep fp32 here for the same
    rounding-symmetry reason as the eager path; in fp32 SDPA dispatches to the
    math/memory-efficient backend rather than fp16 FlashAttention, but the
    integration point is identical.
    """
    out_dtype = q.dtype
    qf = q.float()
    kf = _repeat_kv(k, module.num_key_value_groups).float()
    vf = _repeat_kv(v, module.num_key_value_groups).float()
    attn_mask = None
    if attention_mask is not None:
        attn_mask = attention_mask[:, :, :, : kf.shape[-2]].float()
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        qf, kf, vf, attn_mask=attn_mask, scale=scaling,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()    # [B, S, Hq, d]
    return attn_output, None


def _givens_cos_sin(puf: PUFSim, head_dim: int, layer: int, group: int,
                    purpose: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    seed = puf.derive_seed(layer=layer, group=group, block=-1, purpose=purpose)
    g = _seed_generator(seed, torch.device("cpu"))
    thetas = torch.rand(head_dim // 2, generator=g, dtype=torch.float32) * (2 * math.pi)
    return thetas.cos().to(device), thetas.sin().to(device)


def _head_diag_scale(puf: PUFSim, head_dim: int, layer: int, group: int,
                     purpose: str, device, log_range: float) -> torch.Tensor:
    seed = puf.derive_seed(layer=layer, group=group, block=-1, purpose=purpose)
    g = _seed_generator(seed, torch.device("cpu"))
    u = torch.rand(head_dim, generator=g, dtype=torch.float32)
    return torch.exp((u * 2.0 - 1.0) * float(log_range)).to(device)


def _cache_positions(seq_len: int, cache_position, *, device) -> torch.Tensor:
    if cache_position is not None:
        pos = cache_position.detach().flatten().to(torch.long)
        if pos.numel() == seq_len:
            return pos.cpu()
    return torch.arange(seq_len, dtype=torch.long, device="cpu")


def _norm_scales_bhsd(puf: PUFSim, num_heads: int, positions: torch.Tensor,
                      head_dim: int, layer: int, purpose: str, device,
                      log_range: float) -> torch.Tensor:
    """PUF-derived positive cache scales with shape [1, H, S, D]."""
    pos_list = [int(p) for p in positions.cpu().tolist()]
    scales = torch.empty((num_heads, len(pos_list), head_dim), dtype=torch.float32)
    for h in range(num_heads):
        for j, pos in enumerate(pos_list):
            seed = puf.derive_seed(layer=layer, group=h, block=pos, purpose=purpose)
            g = _seed_generator(seed, torch.device("cpu"))
            u = torch.rand(head_dim, generator=g, dtype=torch.float32)
            scales[h, j] = torch.exp((u * 2.0 - 1.0) * float(log_range))
    return scales.to(device=device)[None]


def _affine_mask_bhsd(puf: PUFSim, num_heads: int, positions: torch.Tensor,
                      head_dim: int, layer: int, purpose: str, device,
                      mask_std: float) -> torch.Tensor:
    """PUF-derived additive cache mask with shape [1, H, S, D]."""
    pos_list = [int(p) for p in positions.cpu().tolist()]
    mask = torch.empty((num_heads, len(pos_list), head_dim), dtype=torch.float32)
    for h in range(num_heads):
        for j, pos in enumerate(pos_list):
            seed = puf.derive_seed(layer=layer, group=h, block=pos, purpose=purpose)
            g = _seed_generator(seed, torch.device("cpu"))
            mask[h, j] = torch.randn(head_dim, generator=g, dtype=torch.float32) * float(mask_std)
    return mask.to(device=device)[None]


def _affine_mask_cached(attn, slot: str, positions: torch.Tensor, num_heads: int,
                        head_dim: int, layer: int, purpose: str, device,
                        mask_std: float) -> torch.Tensor:
    """Incrementally-cached PUF affine mask, shape ``[1, H, len(positions), D]``.

    The mask ``M_{w,i}`` depends only on ``(session, write_nonce, layer, position)``
    and never on content, so each position's row is generated exactly once per
    ``(session, write_nonce)`` via a continued per-``(layer, purpose)`` generator
    and cached on the module (the write nonce is constant for a cache lifetime,
    so freshness adds no regeneration cost). This
    makes decode O(1) amortized per step; the previous per-position re-derivation
    regenerated every cached position on every step, which is O(n^2) over a
    decode and dominated wall-clock at long sequence lengths.
    """
    pos = positions.to(torch.long).flatten()
    if pos.numel() == 0:
        return torch.zeros((1, num_heads, 0, head_dim), dtype=torch.float32, device=device)
    need = int(pos.max().item()) + 1
    cache = attn._puf_mask_cache.get(slot)
    have = 0 if cache is None else cache.shape[0]
    if need > have:
        g = attn._puf_mask_gen.get(slot)
        if g is None:
            # Fold the per-write nonce into the keystream context (R2). When the
            # nonce is unset, purpose_eff == purpose, so the seed and hence the
            # whole keystream are byte-for-byte identical to the legacy path.
            wn = getattr(attn, "_puf_write_nonce", None)
            purpose_eff = purpose if wn is None else f"{purpose}|wn={wn}"
            seed = attn._puf_root.derive_seed(layer=layer, group=-1, block=-1, purpose=purpose_eff)
            g = _seed_generator(seed, torch.device("cpu"))
            attn._puf_mask_gen[slot] = g
        delta = need - have
        new = (torch.randn((delta, num_heads, head_dim), generator=g, dtype=torch.float32)
               * float(mask_std)).to(device)
        cache = new if cache is None else torch.cat([cache, new.to(cache.device)], dim=0)
        attn._puf_mask_cache[slot] = cache
    rows = cache.index_select(0, pos.to(cache.device))    # [len, H, D]
    return rows.permute(1, 0, 2).unsqueeze(0).to(device)  # [1, H, len, D]


def _update_norm_sidecar(attn, name: str, current: torch.Tensor, full_seq_len: int) -> torch.Tensor:
    prev = getattr(attn, name, None)
    if (
        prev is None
        or prev.shape[:2] != current.shape[:2]
        or full_seq_len == current.shape[2]
        or prev.shape[2] + current.shape[2] != full_seq_len
    ):
        full = current
    else:
        full = torch.cat([prev.to(current.device), current], dim=2)
    if full.shape[2] > full_seq_len:
        full = full[:, :, -full_seq_len:]
    setattr(attn, name, full.detach())
    return full


def _apply_givens_bhsd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       *, transpose: bool = False) -> torch.Tensor:
    """Apply row-vector Givens rotations to [B, H, S, D] tensors."""
    x = x.float()
    out = torch.empty_like(x)
    c = cos.to(x.device)[None, :, None, :]
    s = sin.to(x.device)[None, :, None, :]
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    if transpose:
        out[..., 0::2] = x0 * c - x1 * s
        out[..., 1::2] = x0 * s + x1 * c
    else:
        out[..., 0::2] = x0 * c + x1 * s
        out[..., 1::2] = -x0 * s + x1 * c
    if x.shape[-1] % 2:
        out[..., -1] = x[..., -1]
    return out


def _apply_givens_bshd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       *, transpose: bool = False) -> torch.Tensor:
    """Apply row-vector Givens rotations to [B, S, H, D] tensors."""
    x = x.float()
    out = torch.empty_like(x)
    c = cos.to(x.device)[None, None, :, :]
    s = sin.to(x.device)[None, None, :, :]
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    if transpose:
        out[..., 0::2] = x0 * c - x1 * s
        out[..., 1::2] = x0 * s + x1 * c
    else:
        out[..., 0::2] = x0 * c + x1 * s
        out[..., 1::2] = -x0 * s + x1 * c
    if x.shape[-1] % 2:
        out[..., -1] = x[..., -1]
    return out


if _TRITON_AVAILABLE:
    @triton.jit
    def _givens_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                       total_pairs: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
                       D: tl.constexpr, layout: tl.constexpr, transpose: tl.constexpr,
                       BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_pairs
        pair = offsets % (D // 2)
        tmp = offsets // (D // 2)
        if layout == 0:  # [B, H, S, D]
            s_idx = tmp % S
            h_idx = (tmp // S) % H
            b_idx = tmp // (S * H)
            base = ((b_idx * H + h_idx) * S + s_idx) * D + pair * 2
        else:  # [B, S, H, D]
            h_idx = tmp % H
            s_idx = (tmp // H) % S
            b_idx = tmp // (S * H)
            base = ((b_idx * S + s_idx) * H + h_idx) * D + pair * 2
        c = tl.load(cos_ptr + h_idx * (D // 2) + pair, mask=mask, other=1.0)
        s = tl.load(sin_ptr + h_idx * (D // 2) + pair, mask=mask, other=0.0)
        x0 = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        if transpose:
            y0 = x0 * c - x1 * s
            y1 = x0 * s + x1 * c
        else:
            y0 = x0 * c + x1 * s
            y1 = -x0 * s + x1 * c
        tl.store(out_ptr + base, y0, mask=mask)
        tl.store(out_ptr + base + 1, y1, mask=mask)


def _apply_givens_triton(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         *, layout: str, transpose: bool = False) -> torch.Tensor:
    if not (_TRITON_AVAILABLE and x.is_cuda and x.shape[-1] % 2 == 0):
        if layout == "bhsd":
            return _apply_givens_bhsd(x, cos, sin, transpose=transpose)
        return _apply_givens_bshd(x, cos, sin, transpose=transpose)
    x_contig = x.float().contiguous()
    cos_contig = cos.to(device=x.device, dtype=torch.float32).contiguous()
    sin_contig = sin.to(device=x.device, dtype=torch.float32).contiguous()
    out = torch.empty_like(x_contig)
    if layout == "bhsd":
        _, H, S, D = x_contig.shape
        layout_id = 0
    else:
        _, S, H, D = x_contig.shape
        layout_id = 1
    total_pairs = x_contig.numel() // 2
    block = 256
    grid = (triton.cdiv(total_pairs, block),)
    _givens_kernel[grid](x_contig, cos_contig, sin_contig, out,
                         total_pairs, H, S, D, layout_id, transpose, BLOCK=block)
    return out


def _wrapped_forward(self, hidden_states, position_embeddings, attention_mask,
                     past_key_values=None, cache_position=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q = self.q_proj(hidden_states).view(hidden_shape)
    k = self.k_proj(hidden_states).view(hidden_shape)
    if hasattr(self, "q_norm"):
        q = self.q_norm(q)
    if hasattr(self, "k_norm"):
        k = self.k_norm(k)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = _apply_rotary_pos_emb(q, k, cos, sin)

    # --- PUF rotation (post-RoPE) in fp32; optionally keep K/V fp32 in cache ---
    q_dt, k_dt, v_dt = q.dtype, k.dtype, v.dtype
    store_dtype = torch.float32 if getattr(self, "_puf_fp32_cache", False) else k_dt
    if getattr(self, "_puf_fast_givens", False):
        q = _apply_givens_triton(q, self._puf_Ok_q_cos, self._puf_Ok_q_sin, layout="bhsd").to(q_dt)
        k = _apply_givens_triton(k, self._puf_Ok_cos, self._puf_Ok_sin, layout="bhsd").to(store_dtype)
        v = _apply_givens_triton(v, self._puf_Ov_cos, self._puf_Ov_sin, layout="bhsd").to(store_dtype)
    else:
        Ok = self._puf_Ok                  # fp32 [Hkv, d, d]
        Ov = self._puf_Ov
        Ok_q = self._puf_Ok_q              # fp32 [Hq, d, d]
        q = torch.einsum("bhsd,hde->bhse", q.float(), Ok_q.to(q.device)).to(q_dt)
        k = torch.einsum("bhsd,hde->bhse", k.float(), Ok.to(k.device)).to(store_dtype)
        v = torch.einsum("bhsd,hde->bhse", v.float(), Ov.to(v.device)).to(store_dtype)

    if getattr(self, "_puf_nonorth_log_range", 0.0) > 0.0:
        q = (q.float() * self._puf_K_q_inv_scale.to(q.device)[None, :, None, :]).to(q_dt)
        k = (k.float() * self._puf_K_scale.to(k.device)[None, :, None, :]).to(store_dtype)
        v = (v.float() * self._puf_V_scale.to(v.device)[None, :, None, :]).to(store_dtype)

    if getattr(self, "_puf_affine_mask", False):
        cur_pos = _cache_positions(k.shape[2], cache_position, device=k.device)
        k_mask = _affine_mask_cached(
            self, "K", cur_pos, k.shape[1], k.shape[-1], self.layer_idx,
            "K_affine_mask", k.device, self._puf_mask_std,
        )
        v_mask = _affine_mask_cached(
            self, "V", cur_pos, v.shape[1], v.shape[-1], self.layer_idx,
            "V_affine_mask", v.device, self._puf_mask_std,
        )
        k = (k.float() + k_mask).to(store_dtype)
        v = (v.float() + v_mask).to(store_dtype)

    if getattr(self, "_puf_unit_norm_cache", False):
        k_current_norm = k.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v_current_norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k = (k.float() / k_current_norm).to(store_dtype)
        v = (v.float() / v_current_norm).to(store_dtype)
    elif getattr(self, "_puf_norm_blind", False):
        cur_pos = _cache_positions(k.shape[2], cache_position, device=k.device)
        k_scale = _norm_scales_bhsd(
            self._puf_root, k.shape[1], cur_pos, k.shape[-1], self.layer_idx,
            "K_norm", k.device, self._puf_norm_log_range,
        )
        v_scale = _norm_scales_bhsd(
            self._puf_root, v.shape[1], cur_pos, v.shape[-1], self.layer_idx,
            "V_norm", v.device, self._puf_norm_log_range,
        )
        k = (k.float() * k_scale).to(store_dtype)
        v = (v.float() * v_scale).to(store_dtype)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

    if getattr(self, "_puf_unit_norm_cache", False):
        k_full_norm = _update_norm_sidecar(self, "_puf_k_norm_sidecar", k_current_norm, k.shape[2])
        v_full_norm = _update_norm_sidecar(self, "_puf_v_norm_sidecar", v_current_norm, v.shape[2])
        k_attn = k.float() * k_full_norm.to(k.device)
        v_attn = v.float() * v_full_norm.to(v.device)
    elif getattr(self, "_puf_affine_mask", False):
        full_pos = _cache_positions(k.shape[2], cache_position, device=k.device)
        if full_pos.numel() != k.shape[2]:
            full_pos = torch.arange(k.shape[2], dtype=torch.long, device=k.device)
        k_mask = _affine_mask_cached(
            self, "K", full_pos, k.shape[1], k.shape[-1], self.layer_idx,
            "K_affine_mask", k.device, self._puf_mask_std,
        )
        v_mask = _affine_mask_cached(
            self, "V", full_pos, v.shape[1], v.shape[-1], self.layer_idx,
            "V_affine_mask", v.device, self._puf_mask_std,
        )
        k_attn = k.float() - k_mask
        v_attn = v.float() - v_mask
    elif getattr(self, "_puf_norm_blind", False):
        full_pos = _cache_positions(k.shape[2], cache_position, device=k.device)
        if full_pos.numel() != k.shape[2]:
            full_pos = torch.arange(k.shape[2], dtype=torch.long, device="cpu")
        k_unscale = _norm_scales_bhsd(
            self._puf_root, k.shape[1], full_pos, k.shape[-1], self.layer_idx,
            "K_norm", k.device, self._puf_norm_log_range,
        )
        v_unscale = _norm_scales_bhsd(
            self._puf_root, v.shape[1], full_pos, v.shape[-1], self.layer_idx,
            "V_norm", v.device, self._puf_norm_log_range,
        )
        k_attn = k.float() / k_unscale
        v_attn = v.float() / v_unscale
    else:
        k_attn = k
        v_attn = v

    # fp32 attention matmul (rotation-aware: keeps bf16 rounding from
    # accumulating asymmetrically vs. the unrotated baseline path). The SDPA
    # backend routes the same rotated tensors through the fused-kernel API.
    if getattr(self, "_puf_attn_backend", "eager") == "sdpa":
        attn_output, attn_weights = _sdpa_attention(
            self, q, k_attn, v_attn, attention_mask, scaling=self.scaling,
        )
    else:
        attn_output, attn_weights = _fp32_eager_attention(
            self, q, k_attn, v_attn, attention_mask, scaling=self.scaling,
        )
    # attn_output is fp32 [B, S, Hq, d]; apply Ov_q^T then cast back.
    if getattr(self, "_puf_nonorth_log_range", 0.0) > 0.0:
        attn_output = attn_output * self._puf_V_q_inv_scale.to(attn_output.device)[None, None, :, :]
    if getattr(self, "_puf_fast_givens", False):
        attn_output = _apply_givens_triton(
            attn_output, self._puf_Ov_q_cos, self._puf_Ov_q_sin, layout="bshd", transpose=True,
        ).to(q_dt)
    else:
        Ov_q = self._puf_Ov_q
        attn_output = torch.einsum("bshd,hed->bshe", attn_output,
                                    Ov_q.to(attn_output.device)).to(q_dt)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def install_puf_attention(model, puf: PUFSim, *,
                          kind_k: str = "givens",
                          kind_v: str = "givens",
                          fp32_cache: bool = False,
                          fast_givens: bool = False,
                          norm_blind: bool = False,
                          norm_log_range: float = 1.0,
                          unit_norm_cache: bool = False,
                          nonorth_log_range: float = 0.0,
                          affine_mask: bool = False,
                          mask_std: float = 4.0,
                          write_nonce: str | None = None,
                          attn_backend: str = "eager"):
    """Install the PUF rotation patch on supported decoder attention modules.

    The K/V baskets stored in past_key_values from now on are in the PUF-rotated
    basis. With norm_blind=True, they are also multiplied by PUF-derived
    per-position diagonal scales and unscaled inside attention before use.
    With unit_norm_cache=True, exposed cache vectors are unit-normalized and
    their original norms are kept in a private in-process sidecar.
    With affine_mask=True, PUF-derived additive masks are stored in the exported
    cache and regenerated/subtracted inside attention.
    """
    supported = {"qwen3", "qwen2", "llama"}
    model_type = getattr(model.config, "model_type", None)
    if model_type not in supported:
        raise ValueError(f"install_puf_attention supports {sorted(supported)} (got {model_type})")
    if fast_givens and (kind_k != "givens" or kind_v != "givens"):
        raise ValueError("fast_givens requires kind_k=kind_v='givens'")
    if nonorth_log_range > 0.0 and (norm_blind or unit_norm_cache):
        raise ValueError("nonorth_log_range is mutually exclusive with norm_blind/unit_norm_cache")
    if affine_mask and (norm_blind or unit_norm_cache):
        raise ValueError("affine_mask is mutually exclusive with norm_blind/unit_norm_cache")
    model.config._attn_implementation = "eager"

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        head_dim = attn.head_dim
        Hkv = attn.k_proj.out_features // head_dim
        device = next(attn.parameters()).device
        dtype = next(attn.parameters()).dtype

        if fast_givens:
            Ok_cos_sin = [_givens_cos_sin(puf, head_dim, layer_idx, h, "K_attn", device)
                          for h in range(Hkv)]
            Ov_cos_sin = [_givens_cos_sin(puf, head_dim, layer_idx, h, "V_attn", device)
                          for h in range(Hkv)]
            Ok_cos = torch.stack([cs[0] for cs in Ok_cos_sin]).float()
            Ok_sin = torch.stack([cs[1] for cs in Ok_cos_sin]).float()
            Ov_cos = torch.stack([cs[0] for cs in Ov_cos_sin]).float()
            Ov_sin = torch.stack([cs[1] for cs in Ov_cos_sin]).float()
            attn._puf_Ok_cos = Ok_cos
            attn._puf_Ok_sin = Ok_sin
            attn._puf_Ov_cos = Ov_cos
            attn._puf_Ov_sin = Ov_sin
            attn._puf_Ok_q_cos = Ok_cos.repeat_interleave(attn.num_key_value_groups, dim=0).float()
            attn._puf_Ok_q_sin = Ok_sin.repeat_interleave(attn.num_key_value_groups, dim=0).float()
            attn._puf_Ov_q_cos = Ov_cos.repeat_interleave(attn.num_key_value_groups, dim=0).float()
            attn._puf_Ov_q_sin = Ov_sin.repeat_interleave(attn.num_key_value_groups, dim=0).float()
        else:
            Ok = torch.stack([
                _basis_for(puf, kind_k, head_dim, layer_idx, h, "K_attn",
                           device, torch.float32)
                for h in range(Hkv)
            ])
            Ov = torch.stack([
                _basis_for(puf, kind_v, head_dim, layer_idx, h, "V_attn",
                           device, torch.float32)
                for h in range(Hkv)
            ])
            attn._puf_Ok = Ok.float()
            attn._puf_Ov = Ov.float()
            attn._puf_Ok_q = Ok.repeat_interleave(attn.num_key_value_groups, dim=0).float()
            attn._puf_Ov_q = Ov.repeat_interleave(attn.num_key_value_groups, dim=0).float()
        if nonorth_log_range > 0.0:
            K_scale = torch.stack([
                _head_diag_scale(puf, head_dim, layer_idx, h, "K_nonorth_scale", device, nonorth_log_range)
                for h in range(Hkv)
            ]).float()
            V_scale = torch.stack([
                _head_diag_scale(puf, head_dim, layer_idx, h, "V_nonorth_scale", device, nonorth_log_range)
                for h in range(Hkv)
            ]).float()
        else:
            K_scale = torch.ones((Hkv, head_dim), device=device, dtype=torch.float32)
            V_scale = torch.ones((Hkv, head_dim), device=device, dtype=torch.float32)
        attn._puf_K_scale = K_scale
        attn._puf_V_scale = V_scale
        attn._puf_K_q_inv_scale = K_scale.reciprocal().repeat_interleave(attn.num_key_value_groups, dim=0).float()
        attn._puf_V_q_inv_scale = V_scale.reciprocal().repeat_interleave(attn.num_key_value_groups, dim=0).float()
        attn._puf_fp32_cache = fp32_cache
        attn._puf_fast_givens = fast_givens
        attn._puf_norm_blind = norm_blind
        attn._puf_norm_log_range = norm_log_range
        attn._puf_nonorth_log_range = nonorth_log_range
        attn._puf_affine_mask = affine_mask
        attn._puf_mask_std = mask_std
        # Per-write nonce (R2, keystream freshness). Default None reproduces the
        # legacy position-only keystream exactly; setting it derives an
        # independent keystream per Protect() call so that differencing two
        # same-session caches no longer cancels the additive mask.
        attn._puf_write_nonce = write_nonce
        attn._puf_mask_cache = {}
        attn._puf_mask_gen = {}
        attn._puf_attn_backend = attn_backend
        attn._puf_unit_norm_cache = unit_norm_cache
        attn._puf_k_norm_sidecar = None
        attn._puf_v_norm_sidecar = None
        attn._puf_root = puf

        # Save original forward once
        if id(attn) not in _ORIG_FORWARDS:
            _ORIG_FORWARDS[id(attn)] = attn.forward
        attn.forward = _wrapped_forward.__get__(attn, type(attn))


def uninstall_puf_attention(model) -> None:
    """Restore the original attention forwards."""
    for layer in model.model.layers:
        attn = layer.self_attn
        if id(attn) in _ORIG_FORWARDS:
            attn.forward = _ORIG_FORWARDS.pop(id(attn))
        for name in (
            "_puf_Ok", "_puf_Ov", "_puf_Ok_q", "_puf_Ov_q",
            "_puf_Ok_cos", "_puf_Ok_sin", "_puf_Ov_cos", "_puf_Ov_sin",
            "_puf_Ok_q_cos", "_puf_Ok_q_sin", "_puf_Ov_q_cos", "_puf_Ov_q_sin",
            "_puf_K_scale", "_puf_V_scale", "_puf_K_q_inv_scale", "_puf_V_q_inv_scale",
            "_puf_fp32_cache", "_puf_fast_givens", "_puf_norm_blind",
            "_puf_norm_log_range", "_puf_nonorth_log_range", "_puf_affine_mask",
            "_puf_mask_std", "_puf_write_nonce", "_puf_mask_cache", "_puf_mask_gen", "_puf_attn_backend",
            "_puf_unit_norm_cache", "_puf_k_norm_sidecar",
            "_puf_v_norm_sidecar", "_puf_root",
        ):
            if hasattr(attn, name):
                delattr(attn, name)
