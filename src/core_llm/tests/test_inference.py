from src.core_llm.inference.generate import (
    load_model,
    generate_text,
)


def test_model_loading():
    model, tokenizer, device = load_model()

    assert model is not None
    assert tokenizer.vocab_size == 256
    assert device is not None


def test_text_generation():
    model, tokenizer, device = load_model()

    generated = generate_text(
        model,
        tokenizer,
        device,
        "Deep Learning",
        max_new_tokens=5,
    )

    assert isinstance(generated, str)
    assert len(generated) > 0