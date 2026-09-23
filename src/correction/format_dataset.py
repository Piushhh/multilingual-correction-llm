import json
try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def format_for_training(dataset, formatter, tokenizer, max_length=512):
    """
    Formats the dataset into model inputs and labels.
    """
    def tokenize_function(examples):
        # Build prompt using the formatter for each example
        prompts = [
            formatter.build_prompt(text, lang, domain)
            for text, lang, domain in zip(examples["incorrect_text"], examples["language"], examples.get("domain", ["general"] * len(examples["incorrect_text"])))
        ]
        
        targets = examples["correct_text"]
        
        # Combine prompt and target for causal LM training
        full_texts = [p + t + tokenizer.eos_token for p, t in zip(prompts, targets)]
        
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        
        # In a real Causal LM training setup, you mask the prompt tokens in the labels
        # For this baseline, we use the standard DataCollatorForLanguageModeling
        # which will just shift the input IDs for next token prediction.
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset

def prepare_hf_dataset(file_path, formatter, tokenizer, max_length):
    raw_data = load_jsonl(file_path)
    if not raw_data:
        return None
    if not DATASETS_AVAILABLE:
        print("Warning: datasets library not found. Returning raw data instead of Dataset object.")
        return raw_data
    dataset = Dataset.from_list(raw_data)
    return format_for_training(dataset, formatter, tokenizer, max_length)
