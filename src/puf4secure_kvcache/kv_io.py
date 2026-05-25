"""Persistence of KV-cache and run metadata for reproducibility."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class CaptureMeta:
    prompt: str
    model_path: str
    model_type: str
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    seq_len: int
    input_ids: list[int]
    dtype: str
    device: str


def save_capture(out_dir: Path, meta: CaptureMeta, kv_list: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(meta), fh, indent=2, ensure_ascii=False)
    tensors = {}
    for layer_idx, (k, v) in enumerate(kv_list):
        tensors[f"layer{layer_idx}_K"] = k.contiguous().cpu()
        tensors[f"layer{layer_idx}_V"] = v.contiguous().cpu()
    torch.save(tensors, out_dir / "kv.pt")


def load_capture(out_dir: Path):
    out_dir = Path(out_dir)
    with open(out_dir / "meta.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    tensors = torch.load(out_dir / "kv.pt", map_location="cpu", weights_only=True)
    n = meta["num_layers"]
    kv_list = [(tensors[f"layer{i}_K"], tensors[f"layer{i}_V"]) for i in range(n)]
    return meta, kv_list
