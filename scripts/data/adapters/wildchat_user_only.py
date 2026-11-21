"""Extract every WildChat user turn, preserving its original content."""

from collections.abc import Mapping
from typing import Any


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    conversation = row["conversation"]
    if not isinstance(conversation, list):
        raise ValueError("conversation must be a list")
    messages = []
    for index, message in enumerate(conversation):
        if not isinstance(message, Mapping):
            raise ValueError(f"message {index} must be an object")
        role = message["role"]
        if not isinstance(role, str):
            raise ValueError(f"message {index} role must be text")
        if role.strip().lower() == "user":
            text = message["content"]
            if not isinstance(text, str):
                raise ValueError(f"message {index} user content must be text")
            messages.append({"role": "user", "content": text})
    return {"messages": messages, "source": "allenai/WildChat", "category": row.get("language")}
