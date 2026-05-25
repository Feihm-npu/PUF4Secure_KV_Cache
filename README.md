# PUF4Secure KV-cache Reproduction

This project reproduces and validates the attack and defense ideas from `Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference` with a local Hugging Face model.

Default local model:

```text
/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B
```

## Scope

- Capture plaintext KV-cache from Qwen3-0.6B during inference.
- Validate three attack directions: inversion, collision, and injection.
- Prototype KV-Cloak-style reversible obfuscation for cached K/V tensors.
- Compare attack quality before and after defense.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_model.py
python scripts/capture_kv.py --prompt "Repeat the word privacy twice."
```

## Project Layout

```text
configs/                 Experiment defaults
docs/                    Reproduction notes and plan
scripts/                 CLI entry points
src/puf4secure_kvcache/  Reproduction library code
experiments/             Generated experiment outputs
```

## Ethics

Use only public, synthetic, or self-authored prompts. The attack code in this repository is for defensive validation of KV-cache privacy risk.
