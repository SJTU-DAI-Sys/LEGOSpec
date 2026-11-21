"""Extract all user turns from role/content or from/value conversations."""

from collections.abc import Mapping
from typing import Any


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    conversations = row["conversations"]
    if not isinstance(conversations, list):
        raise ValueError("conversations must be a list")
    messages = []
    for index, message in enumerate(conversations):
        if not isinstance(message, Mapping):
            raise ValueError(f"message {index} must be an object")
        role_key, text_key = ("role", "content") if "role" in message else ("from", "value")
        role = message[role_key]
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"message {index} has an invalid role")
        if role.strip().lower() in {"user", "human"}:
            messages.append({"role": "user", "content": message[text_key]})
    return {"messages": messages, "source": row.get("source"), "category": row.get("category")}
