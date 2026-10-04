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
    Uses Causal LM instruction fine-tuning approach:
    - Combines prompt + answer
    - Labels are set to -100 for the prompt tokens so loss is only calculated on the answer tokens.
    """
    def tokenize_function(examples):
        prompts = [
            formatter.build_prompt(text, lang, domain)
            for text, lang, domain in zip(examples["incorrect_text"], examples["language"], examples.get("domain", ["general"] * len(examples["incorrect_text"])))
        ]
        
        targets = examples["correct_text"]
        full_texts = [p + t + tokenizer.eos_token for p, t in zip(prompts, targets)]
        
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        
        # We need to mask the prompt tokens in the labels
        labels = []
        for i in range(len(prompts)):
            # Tokenize prompt to find its length
            prompt_encoded = tokenizer(prompts[i], truncation=True, max_length=max_length)
            prompt_len = len(prompt_encoded["input_ids"])
            
            # Start with all -100 (ignore index)
            label = [-100] * max_length
            
            # The actual sequence length (before padding)
            seq_len = sum(tokenized["attention_mask"][i])
            
            # If the prompt itself exceeds max_length, there's no target left to train on
            if prompt_len < seq_len:
                # Copy the input_ids for the target portion
                label[prompt_len:seq_len] = tokenized["input_ids"][i][prompt_len:seq_len]
                
            labels.append(label)
            
        tokenized["labels"] = labels
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
