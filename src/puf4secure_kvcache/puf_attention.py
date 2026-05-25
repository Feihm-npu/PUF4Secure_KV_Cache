"""Level-3 PUF-bound attention wrapper for Qwen3 models.

This monkey-patches every `Qwen3Attention.forward` to apply per-(layer, kv-head)
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
  * Single-PUF, single-session install: the rotations are fixed when
    `install_puf_attention(...)` is called. Switching session requires
    `uninstall_puf_attention(...)` + reinstall.
  * GQA layout is handled: each q-head reuses the O_k/O_v of its kv-head.
"""
from __future__ import annotations

from typing import Callable

import torch

from .puf_basis import _basis_for
from .puf_sim import PUFSim


_ORIG_FORWARDS: dict[int, Callable] = {}


def _fp32_eager_attention(module, q, k, v, attention_mask, scaling):
    """Eager attention computed entirely in fp32 (cast back at the end).

    Mirrors `transformers.models.qwen3.modeling_qwen3.eager_attention_forward`
    but forces fp32 throughout the QK^T, softmax, and AV matmuls. The rotated
    PUF basis makes bf16 rounding error asymmetric vs. the plain path, so we
    keep the entire attention inner loop in fp32.
    """
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    out_dtype = q.dtype
    qf = q.float()
    kf = repeat_kv(k, module.num_key_value_groups).float()
    vf = repeat_kv(v, module.num_key_value_groups).float()

    attn_weights = torch.matmul(qf, kf.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : kf.shape[-2]].float()
        attn_weights = attn_weights + causal_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(attn_weights, vf)              # fp32
    attn_output = attn_output.transpose(1, 2).contiguous()    # [B, S, Hq, d]
    return attn_output, attn_weights.to(out_dtype)


def _wrapped_forward(self, hidden_states, position_embeddings, attention_mask,
                     past_key_values=None, cache_position=None, **kwargs):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # --- PUF rotation (post-RoPE) in fp32; optionally keep K/V fp32 in cache ---
    Ok = self._puf_Ok                  # fp32 [Hkv, d, d]
    Ov = self._puf_Ov
    Ok_q = self._puf_Ok_q              # fp32 [Hq, d, d]
    q_dt, k_dt, v_dt = q.dtype, k.dtype, v.dtype
    store_dtype = torch.float32 if getattr(self, "_puf_fp32_cache", False) else k_dt
    q = torch.einsum("bhsd,hde->bhse", q.float(), Ok_q.to(q.device)).to(q_dt)
    k = torch.einsum("bhsd,hde->bhse", k.float(), Ok.to(k.device)).to(store_dtype)
    v = torch.einsum("bhsd,hde->bhse", v.float(), Ov.to(v.device)).to(store_dtype)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

    # fp32 attention matmul (rotation-aware: keeps bf16 rounding from
    # accumulating asymmetrically vs. the unrotated baseline path).
    attn_output, attn_weights = _fp32_eager_attention(
        self, q, k, v, attention_mask, scaling=self.scaling,
    )
    # attn_output is fp32 [B, S, Hq, d]; apply Ov_q^T then cast back.
    Ov_q = self._puf_Ov_q
    attn_output = torch.einsum("bshd,hed->bshe", attn_output,
                                Ov_q.to(attn_output.device)).to(q_dt)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def install_puf_attention(model, puf: PUFSim, *,
                          kind_k: str = "givens",
                          kind_v: str = "givens",
                          fp32_cache: bool = False):
    """Install the PUF rotation patch on every Qwen3Attention module.

    The K/V baskets stored in past_key_values from now on are in the PUF-rotated
    basis. Legitimate decoding works without changes; an attacker dumping the
    cache cannot map back to plain K/V without the device PUF.
    """
    if model.config.model_type != "qwen3":
        raise ValueError(f"install_puf_attention only supports Qwen3 (got {model.config.model_type})")
    model.config._attn_implementation = "eager"

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        head_dim = attn.head_dim
        Hkv = attn.k_proj.out_features // head_dim
        device = next(attn.parameters()).device
        dtype = next(attn.parameters()).dtype

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
        attn._puf_fp32_cache = fp32_cache

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
        for name in ("_puf_Ok", "_puf_Ov", "_puf_Ok_q", "_puf_Ov_q", "_puf_fp32_cache"):
            if hasattr(attn, name):
                delattr(attn, name)
