\# Member 1 — LLM Core and Model Training



\## 1. Responsibility



Member 1 is responsible for the core language-model pipeline:



\* BPE tokenizer

\* Transformer architecture

\* Model embeddings

\* Causal self-attention

\* Transformer blocks

\* Language-model head

\* Base pretraining

\* Domain adaptation

\* Text generation and inference

\* Correction fine-tuning

\* Correction inference

\* Model testing and checkpoints



Member 1 does not own OCR, complete dataset collection, frontend/backend development, or the complete document-processing pipeline.



\---



\## 2. Model Architecture



The project uses a decoder-only causal Transformer language model.



\### Architecture Configuration



\* Vocabulary size: 256

\* Embedding dimension: 256

\* Transformer layers: 6

\* Attention heads: 8

\* Base/domain context length: 32 tokens

\* Correction context length: 64 tokens

\* Dropout: 0.1

\* Training device: CPU



\### Architecture Flow



Input Text

↓

BPE Tokenizer

↓

Token IDs

↓

Token Embeddings + Positional Embeddings

↓

6 × Transformer Blocks

↓

Causal Multi-Head Self-Attention

↓

Feed-Forward Network

↓

Residual Connections + Normalization

↓

Language Model Head

↓

Next-Token Logits

↓

Token Generation

↓

Decoded Text



\---



\## 3. Tokenization



A BPE tokenizer is used to convert English and Hindi text into token IDs.



The tokenizer was tested with multilingual input and successfully encoded and decoded Hindi text.



Example:



Input:



डीप लर्निंग मशीन लर्निंग



The tokenizer successfully converted the text into token IDs and reconstructed the original Hindi text after decoding.



\---



\## 4. Base Pretraining



The initial language model was trained using causal language modeling.



The training process follows:



Input Sequence

↓

Predict Next Token

↓

Calculate Cross-Entropy Loss

↓

Backpropagation

↓

Parameter Update



A training/validation split was used and the best validation checkpoint was retained.



\### Base Model Result



\* Best validation loss: 0.1818

\* Best validation perplexity: 1.1993



Checkpoint:



src/core\_llm/checkpoints/tiny\_model.pt



Because the available corpus is very small, these metrics should not be interpreted as evidence of general-purpose language-model performance.



\---



\## 5. Domain Adaptation



The base model was further trained on a cleaned domain-specific corpus containing deep-learning-related English and Hindi text.



The purpose of domain adaptation is to increase the model's familiarity with terminology and language patterns associated with the target domain.



Pipeline:



Base Model

↓

Clean Domain Corpus

↓

Domain Training

↓

Validation

↓

Best Domain-Adapted Model



\### Domain Adaptation Result



\* Best epoch: 1

\* Best validation loss: 3.0519

\* Best validation perplexity: 21.1565

\* Early stopping was applied because later epochs increased validation loss.



Checkpoint:



src/core\_llm/checkpoints/domain\_adapted\_model.pt



The best checkpoint was retained because the later epochs showed overfitting on the small domain corpus.



\---



\## 6. Multilingual Generation



The domain-adapted model was tested with both English and Hindi prompts.



Example English prompt:



Deep Learning



Example Hindi prompt:



डीप लर्निंग



The model generated domain-related text for both languages.



This demonstrates that the implemented pipeline can process English and Hindi text, although generation quality remains limited by the small corpus and model size.



\---



\## 7. Correction Fine-Tuning



A separate correction dataset was created containing:



\* 18 correction examples

\* 10 English examples

\* 8 Hindi examples



Each example contains:



Incorrect Text

↓

Correct Text



Example:



Incorrect:



Deep learnig is a subfeld of machne learning.



Correct:



Deep learning is a subfield of machine learning.



The correction model uses the following prompt structure:



Correct: <incorrect text>

Answer:



The training loss is calculated on the correction target.



\---



\## 8. Correction Fine-Tuning Result



The correction model was initialized from the domain-adapted model and fine-tuned using the correction dataset.



Configuration:



\* Context length: 64

\* Learning rate: 1e-5

\* Optimizer: AdamW

\* Weight decay: 0.01

\* Gradient clipping: enabled

\* Maximum epochs: 15

\* Early stopping: enabled



\### Best Result



\* Best epoch: 8

\* Validation loss: 1.9195

\* Validation perplexity: 6.8178



Checkpoint:



src/core\_llm/checkpoints/correction\_model.pt



\---



\## 9. Inference



The correction inference pipeline loads the correction checkpoint and accepts an incorrect English or Hindi sentence.



Pipeline:



Incorrect Text

↓

Correction Prompt

↓

Tokenizer

↓

Correction Model

↓

Autoregressive Generation

↓

Tokenizer Decode

↓

Corrected Text



\---



\## 10. Testing



The complete Member 1 test suite contains six automated tests.



Command:



python -m pytest src/core\_llm/tests -v



Final result:



6 passed in 2.94s



Tests cover:



\* Correction model loading

\* General model loading

\* Text generation

\* Transformer forward pass

\* Vocabulary size

\* BPE tokenizer encode/decode



All six tests passed successfully.



\---



\## 11. Limitations



The current implementation is a research prototype.



Main limitations:



1\. The available training corpus is very small.

2\. The correction dataset contains only 18 examples.

3\. The model is relatively small compared with modern LLMs.

4\. Training was performed on CPU.

5\. Correction generation is currently not sufficiently accurate for production use.

6\. Larger multilingual and domain-specific datasets are required for meaningful evaluation.

7\. Formal correction metrics should be evaluated on a larger held-out test set.



Therefore, the current system demonstrates the architecture and complete training/inference pipeline rather than production-level correction quality.



\---



\## 12. Reproducibility



From the project root:



\### Run all tests



python -m pytest src/core\_llm/tests -v



\### Run text generation



python -m src.core\_llm.inference.generate



\### Run correction inference



python -m src.core\_llm.inference.correct



\### Important Checkpoints



src/core\_llm/checkpoints/tiny\_model.pt



src/core\_llm/checkpoints/domain\_adapted\_model.pt



src/core\_llm/checkpoints/correction\_model.pt



\---



\## 13. Member 1 Completion Status



BPE Tokenizer                  ✓

Transformer Architecture       ✓

Base Pretraining               ✓

Domain Adaptation              ✓

Multilingual Inference         ✓

Correction Fine-Tuning         ✓

Correction Inference           ✓

Model Checkpoints              ✓

Automated Testing              ✓

Documentation                  ✓



Member 1 provides the core language-model and training component that can be integrated with the document/OCR pipeline developed by the other team members.
