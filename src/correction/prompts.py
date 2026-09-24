import json
from typing import Any, Dict, List, Optional


class PromptFormatter:
    """Formats correction prompts dynamically based on language, domain, and specific rules."""

    DEFAULT_RULES = (
        "Correct spelling and grammar.\n"
        "Correct OCR-induced corruption.\n"
        "Preserve intended meaning.\n"
        "Preserve technical terminology.\n"
        "Do not introduce unsupported facts.\n"
        "Do not rewrite stylistically unless required for correctness.\n"
        "Preserve mathematical/code tokens where applicable."
    )

    def __init__(self, template: Optional[str] = None, rules: Optional[str] = None):
        self.template = template or (
            "TASK:\nCorrect the input text.\n\n"
            "LANGUAGE:\n{language}\n\n"
            "DOMAIN:\n{domain}\n\n"
            "RULES:\n{rules}\n\n"
            "INPUT:\n{text}\n\n"
            "OUTPUT:\n"
        )
        self.rules = rules or self.DEFAULT_RULES

    def build_prompt(self, text: str, language: str = "en", domain: str = "general", **kwargs: Any) -> str:
        rules = kwargs.get("rules", self.rules)
        return self.template.format(
            text=text,
            language=language,
            domain=domain,
            rules=rules,
            **kwargs
        )

    def build_chat_messages(self, text: str, language: str = "en", domain: str = "general") -> List[Dict[str, str]]:
        system_content = (
            f"You are a highly accurate multilingual text correction assistant.\n"
            f"Your task is to correct the provided text.\n\n"
            f"LANGUAGE: {language}\n"
            f"DOMAIN: {domain}\n\n"
            f"RULES:\n{self.rules}"
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text}
        ]


default_formatter = PromptFormatter()
