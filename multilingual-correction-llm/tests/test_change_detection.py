import pytest
from src.correction.change_detection import detect_changes

def test_change_detection_insertion():
    original = "The network is fast."
    corrected = "The neural network is fast."
    changes = detect_changes(original, corrected)
    
    # "neural" is inserted
    assert any(c['category'] == 'insertion' and c['corrected'] == 'neural' for c in changes)

def test_change_detection_deletion():
    original = "The the network is fast."
    corrected = "The network is fast."
    changes = detect_changes(original, corrected)
    
    assert any(c['category'] == 'deletion' and 'the' in c['original'].lower() for c in changes)

def test_change_detection_replacement():
    original = "teh cat"
    corrected = "the cat"
    changes = detect_changes(original, corrected)
    
    assert len(changes) == 1
    assert changes[0]['category'] == 'replacement'
    assert changes[0]['original'] == 'teh'
    assert changes[0]['corrected'] == 'the'

def test_change_detection_unchanged():
    original = "The network is fast."
    corrected = "The network is fast."
    changes = detect_changes(original, corrected)
    
    assert len(changes) == 0
