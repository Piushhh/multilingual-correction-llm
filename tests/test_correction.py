import pytest
from src.correction.prompts import PromptFormatter

def test_prompt_formatter_default():
    formatter = PromptFormatter()
    prompt = formatter.build_prompt("teh cat", language="en", domain="general")
    
    assert "LANGUAGE:\nen" in prompt
    assert "DOMAIN:\ngeneral" in prompt
    assert "INPUT:\nteh cat" in prompt
    assert "Correct spelling and grammar." in prompt

def test_prompt_formatter_chat():
    formatter = PromptFormatter()
    messages = formatter.build_chat_messages("teh cat", language="en", domain="general")
    
    assert len(messages) == 2
    assert messages[0]['role'] == 'system'
    assert "LANGUAGE: en" in messages[0]['content']
    assert "DOMAIN: general" in messages[0]['content']
    assert messages[1]['role'] == 'user'
    assert messages[1]['content'] == "teh cat"

def test_prompt_formatter_custom_rules():
    formatter = PromptFormatter(rules="Only correct spelling.")
    prompt = formatter.build_prompt("teh cat", language="en", domain="general")
    
    assert "RULES:\nOnly correct spelling." in prompt
    assert "Correct OCR-induced corruption." not in prompt
