from puf4secure_kvcache.model_utils import load_model_and_tokenizer, resolve_snapshot_path


def main() -> None:
    model_path = resolve_snapshot_path()
    model, tokenizer = load_model_and_tokenizer(model_path)
    print(f"model_path={model_path}")
    print(f"model_type={model.config.model_type}")
    print(f"layers={model.config.num_hidden_layers}")
    print(f"hidden_size={model.config.hidden_size}")
    print(f"vocab_size={len(tokenizer)}")


if __name__ == "__main__":
    main()
