"""Extract original user turns from Nemotron Post-Training Dataset v1."""

from collections.abc import Mapping
from typing import Any


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = row["messages"]
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    users = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"message {index} must be an object")
        role = message["role"]
        if not isinstance(role, str):
            raise ValueError(f"message {index} role must be text")
        if role.strip().lower() == "user":
            content = message["content"]
            if not isinstance(content, str):
                raise ValueError(f"message {index} user content must be text")
            users.append({"role": "user", "content": content})
    return {
        "messages": users,
        "source": "nv-community/Nemotron-Post-Training-Dataset-v1",
        "category": row.get("category"),
    }
