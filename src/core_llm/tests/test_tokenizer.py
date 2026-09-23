from pathlib import Path

import sentencepiece as spm

from src.core_llm.tokenizer.bpe_tokenizer import BPETokenizer


def test_bpe_tokenizer_encode_decode(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"

    corpus.write_text(
        "hello world\n"
        "this is a test\n"
        "नमस्ते दुनिया\n"
        "hello दुनिया\n",
        encoding="utf-8",
    )

    model_prefix = str(tmp_path / "tokenizer")

    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=model_prefix,
        vocab_size=64,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
    )

    tokenizer = BPETokenizer(
        model_prefix + ".model"
    )

    text = "hello world"

    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
    assert decoded == text