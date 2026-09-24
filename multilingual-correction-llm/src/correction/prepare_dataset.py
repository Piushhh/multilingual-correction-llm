import argparse
import json
import os
import random
from pathlib import Path
from collections import Counter

def load_data(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")
    return data

def validate_and_clean(data):
    valid_data = []
    seen_ids = set()
    seen_texts = set()
    
    stats = {
        'total': len(data),
        'missing_fields': 0,
        'duplicate_ids': 0,
        'duplicate_texts': 0,
        'empty_strings': 0,
        'invalid_language': 0,
        'valid': 0,
        'language_distribution': Counter(),
        'domain_distribution': Counter(),
        'error_category_distribution': Counter(),
        'verified_distribution': Counter(),
        'source_type_distribution': Counter()
    }
    
    valid_languages = {'en', 'hi', 'code-mixed'}
    
    for item in data:
        # Check required fields
        if not all(k in item for k in ['id', 'language', 'incorrect_text', 'correct_text']):
            stats['missing_fields'] += 1
            continue
            
        # Check empty strings
        if not item['incorrect_text'].strip() or not item['correct_text'].strip():
            stats['empty_strings'] += 1
            continue
            
        # Check valid language
        if item['language'] not in valid_languages:
            stats['invalid_language'] += 1
            continue
            
        # Check duplicates
        if item['id'] in seen_ids:
            stats['duplicate_ids'] += 1
            continue
            
        text_pair = f"{item['incorrect_text']}|||{item['correct_text']}"
        if text_pair in seen_texts:
            stats['duplicate_texts'] += 1
            continue
            
        seen_ids.add(item['id'])
        seen_texts.add(text_pair)
        valid_data.append(item)
        stats['valid'] += 1
        
        # Distributions
        stats['language_distribution'][item.get('language', 'unknown')] += 1
        stats['domain_distribution'][item.get('domain', 'unknown')] += 1
        stats['verified_distribution'][str(item.get('verified', False))] += 1
        stats['source_type_distribution'][item.get('source_type', 'unknown')] += 1
        for err in item.get('error_categories', []):
            stats['error_category_distribution'][err] += 1
        
    # Convert counters to dicts for clean printing/saving
    for k in ['language_distribution', 'domain_distribution', 'error_category_distribution', 'verified_distribution', 'source_type_distribution']:
        stats[k] = dict(stats[k])
        
    return valid_data, stats

def create_splits(data, seed, train_ratio=0.8, val_ratio=0.1):
    random.seed(seed)
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    for item in data[:train_end]: item['split'] = 'train'
    for item in data[train_end:val_end]: item['split'] = 'validation'
    for item in data[val_end:]: item['split'] = 'test'
    
    return data

def save_data(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    splits = {'train': [], 'validation': [], 'test': []}
    for item in data:
        splits[item.get('split', 'train')].append(item)
        
    for split_name, split_data in splits.items():
        if split_data:
            out_path = Path(output_dir) / f"{split_name}.jsonl"
            with open(out_path, 'w', encoding='utf-8') as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
def main():
    parser = argparse.ArgumentParser(description="Prepare Correction Dataset")
    parser.add_argument('--input', type=str, required=True, help="Input JSONL file")
    parser.add_argument('--output-dir', type=str, required=True, help="Output directory")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for splitting")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    data = load_data(args.input)
    
    print("Validating and cleaning data...")
    valid_data, stats = validate_and_clean(data)
    
    print("\nStatistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
        
    if valid_data:
        print(f"\nCreating splits with seed {args.seed}...")
        split_data = create_splits(valid_data, args.seed)
        
        print(f"Saving to {args.output_dir}...")
        save_data(split_data, args.output_dir)
        print("Done!")
    else:
        print("No valid data found to process.")

if __name__ == "__main__":
    main()
