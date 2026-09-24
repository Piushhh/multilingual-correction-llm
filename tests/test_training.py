import pytest
from src.correction.format_dataset import load_jsonl

def test_load_jsonl_empty_or_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jsonl("nonexistent_file.jsonl")
    
    test_file = tmp_path / "test.jsonl"
    test_file.write_text('{"id": "1", "language": "en"}\n{"id": "2", "language": "hi"}')
    
    data = load_jsonl(test_file)
    assert len(data) == 2
    assert data[0]['language'] == 'en'
