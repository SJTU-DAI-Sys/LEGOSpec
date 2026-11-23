"""Extract only the original first user turn from Nemotron Chat v3."""

from collections.abc import Mapping
from typing import Any


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")
    first_user = next(
        (
            message
            for message in messages
            if isinstance(message, Mapping) and message.get("role") == "user"
        ),
        None,
    )
    content = first_user.get("content") if first_user else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("original first user content must be nonempty text")
    return {
        "messages": [{"role": "user", "content": content}],
        "source": "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3",
        "category": None,
    }
