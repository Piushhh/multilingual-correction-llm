# Member 1 -- LLM Core

A from-scratch, byte-level-BPE-tokenized, decoder-only (GPT-style) Transformer
language model, plus the training and inference code around it. This is the
whole "build the LLM" workstream from the roadmap: tokenizer -> model ->
training -> domain adaptation -> inference -> (optionally) an API.

## Layout

```
core_llm/
├── tokenizer/
│   └── train_tokenizer.py   # trains a byte-level BPE tokenizer -> tokenizer.json
├── model/
│   ├── embeddings.py        # token + learned positional embeddings
│   ├── attention.py         # causal multi-head self-attention
│   ├── transformer_block.py # pre-norm attention + feed-forward block
│   └── transformer.py       # stacks blocks into CausalTransformerLM
├── training/
│   ├── dataset.py           # tokenizes a text file into (input, target) chunks
│   ├── train_utils.py       # seed, checkpoint save/load
│   └── pretrain.py          # the training loop (base AND domain-adaptive)
└── inference/
    └── generate.py          # LLM class: load a checkpoint, generate text
```

## Why these design choices

- **Byte-level BPE tokenizer**: handles English, Hindi (Devanagari), and
  code-mixed text with no `<unk>` fallback, since every byte sequence is
  representable. Trained jointly over `data/raw/train.txt` and
  `data/domain/train.txt` so the vocabulary covers both general and domain
  vocabulary from the start.
- **Pre-norm Transformer blocks** (`LayerNorm` before attention/FFN, not
  after): more stable to train, especially at small scale/short training
  runs like this prototype.
- **Weight tying** between the input token embedding and the output
  (`lm_head`) projection: fewer parameters, and a well-known regularizer for
  small language models.
- **Causal mask** in attention: the model can only attend to earlier
  positions, which is what makes this a left-to-right ("causal") language
  model suitable for autoregressive generation.

## Running the pipeline end to end

```bash
# 1. Train the tokenizer (writes data/tokenizer/tokenizer.json)
python -m src.core_llm.tokenizer.train_tokenizer --vocab-size 8000

# 2. Train the base multilingual model
python -m src.core_llm.training.pretrain \
    --train-file data/raw/train.txt \
    --val-file data/raw/val.txt \
    --tokenizer data/tokenizer/tokenizer.json \
    --checkpoint-dir checkpoints/base \
    --seq-len 128 --batch-size 16 --d-model 256 --num-heads 4 \
    --num-layers 4 --d-ff 1024 --epochs 10

# 3. Domain-adaptive pretraining: continue training the base checkpoint
#    on the deep-learning domain corpus (must reuse the SAME tokenizer)
python -m src.core_llm.training.pretrain \
    --train-file data/domain/train.txt \
    --val-file data/domain/val.txt \
    --tokenizer data/tokenizer/tokenizer.json \
    --checkpoint-dir checkpoints/domain \
    --init-from checkpoints/base/best.pt \
    --seq-len 128 --batch-size 16 --d-model 256 --num-heads 4 \
    --num-layers 4 --d-ff 1024 --epochs 5

# 4. Generate text from a trained checkpoint
python -m src.core_llm.inference.generate \
    --checkpoint checkpoints/domain/best.pt \
    --tokenizer data/tokenizer/tokenizer.json \
    --prompt "Deep learning is" \
    --max-new-tokens 50 --temperature 0.8 --top-k 40
```

Or from Python:

```python
from src.core_llm.inference.generate import LLM

llm = LLM.from_checkpoint(
    checkpoint_path="checkpoints/domain/best.pt",
    tokenizer_path="data/tokenizer/tokenizer.json",
)
print(llm.generate("Deep learning is", max_new_tokens=50, temperature=0.8, top_k=40))
```

### Serving it behind your own API

`app/api/main.py` exposes the trained model at `POST /generate` (in addition
to the existing `/health` check). This is *your own* model behind *your own*
endpoint -- there's no external API key involved, since nothing is calling
out to a third-party LLM provider. Run it with:

```bash
uvicorn app.api.main:app --reload
```

then:

```bash
curl -X POST http://127.0.0.1:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Deep learning is", "max_new_tokens": 40}'
```

This is meant as a minimal starting point for Member 3's correction
pipeline/API integration -- the same `LLM` class in `inference/generate.py`
is what a correction pipeline would call.

## Scaling up from this prototype

`data/raw/train.txt` and `data/domain/train.txt` are small illustrative
corpora (a few hundred lines) meant to prove the pipeline works end to end
-- the CLI flags above (`--seq-len`, `--d-model`, `--num-layers`, etc.)
default to a small model on purpose. To get a model that's actually useful:

- Grow the corpora substantially (real corpus sizes for even a small LLM are
  measured in millions of tokens, not hundreds of lines).
- Increase `--d-model`, `--num-layers`, and `--num-heads` once you have data
  to justify a larger model (the current defaults -- 256/4/4 -- are enough
  for a signal that the pipeline works, not a capable model).
- Train for more epochs / more steps, and watch `val_loss` -- if it starts
  rising while `train_loss` keeps falling, you're overfitting and need more
  data or a smaller model, not more epochs.
- Retrain the tokenizer once the corpus is much larger so the vocabulary is
  representative (`--vocab-size` can also grow well past 8000 at that
  point).

## What this workstream deliberately excludes

Per the team split: no OCR, no domain classification, no correction-specific
data or training, and no frontend. Those belong to Member 2
(`src/document_ai/`) and Member 3 (`src/correction/`, most of `app/`).
