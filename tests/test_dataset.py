import pytest
from src.correction.prepare_dataset import validate_and_clean, create_splits

def test_validate_and_clean_valid_record():
    data = [{
        "id": "1",
        "language": "en",
        "incorrect_text": "teh",
        "correct_text": "the",
        "domain": "general"
    }]
    valid_data, stats = validate_and_clean(data)
    assert len(valid_data) == 1
    assert stats['valid'] == 1

def test_validate_and_clean_missing_field():
    data = [{
        "id": "2",
        "language": "en",
        "incorrect_text": "teh"
        # missing correct_text
    }]
    valid_data, stats = validate_and_clean(data)
    assert len(valid_data) == 0
    assert stats['missing_fields'] == 1

def test_validate_and_clean_duplicate_record():
    data = [
        {"id": "3", "language": "en", "incorrect_text": "a", "correct_text": "b"},
        {"id": "3", "language": "en", "incorrect_text": "a", "correct_text": "b"}
    ]
    valid_data, stats = validate_and_clean(data)
    assert len(valid_data) == 1
    assert stats['duplicate_ids'] == 1

def test_validate_and_clean_invalid_language():
    data = [{
        "id": "4",
        "language": "fr",
        "incorrect_text": "bonjour",
        "correct_text": "bonjour"
    }]
    valid_data, stats = validate_and_clean(data)
    assert len(valid_data) == 0
    assert stats['invalid_language'] == 1

def test_deterministic_splitting():
    data = [{"id": str(i), "language": "en", "incorrect_text": "a", "correct_text": "b"} for i in range(100)]
    splits1 = create_splits(list(data), seed=42)
    splits2 = create_splits(list(data), seed=42)
    
    assert [d['id'] for d in splits1] == [d['id'] for d in splits2]
    
    # Check ratios roughly
    train_count = sum(1 for d in splits1 if d['split'] == 'train')
    val_count = sum(1 for d in splits1 if d['split'] == 'validation')
    test_count = sum(1 for d in splits1 if d['split'] == 'test')
    
    assert train_count == 80
    assert val_count == 10
    assert test_count == 10
