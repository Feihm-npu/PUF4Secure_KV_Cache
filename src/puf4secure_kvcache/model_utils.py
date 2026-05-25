from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_CACHE = Path("/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")


def resolve_snapshot_path(repo_cache_dir: str | Path = DEFAULT_MODEL_CACHE, snapshot: str | None = None) -> Path:
    repo_cache_dir = Path(repo_cache_dir)
    if snapshot is None:
        ref_path = repo_cache_dir / "refs" / "main"
        snapshot = ref_path.read_text(encoding="utf-8").strip()

    model_path = repo_cache_dir / "snapshots" / snapshot
    if not model_path.exists():
        raise FileNotFoundError(f"Model snapshot not found: {model_path}")
    return model_path


def load_model_and_tokenizer(model_path: str | Path | None = None, device: str = "auto"):
    if model_path is None:
        model_path = resolve_snapshot_path()

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype="auto",
        device_map=device,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def forward_with_kv(model, tokenizer, prompt: str, add_special_tokens: bool = True):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs, use_cache=True, output_hidden_states=True)
    return inputs, outputs.past_key_values, outputs


def get_attention_module(model, layer_idx: int):
    return model.model.layers[layer_idx].self_attn


def get_embedding_matrix(model) -> torch.Tensor:
    return model.get_input_embeddings().weight


def kv_to_list(past_key_values) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Normalize HF DynamicCache / legacy tuple to list[(K, V)] (detached, on cpu copy)."""
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in past_key_values:
        k, v = layer[0], layer[1]
        out.append((k.detach(), v.detach()))
    return out


def kv_to_legacy_tuple(kv_list: list[tuple[torch.Tensor, torch.Tensor]]):
    """Convert list[(K,V)] back to a HF-compatible cache object."""
    try:
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(kv_list):
            cache.update(k, v, layer_idx)
        return cache
    except Exception:
        return tuple((k, v) for k, v in kv_list)
