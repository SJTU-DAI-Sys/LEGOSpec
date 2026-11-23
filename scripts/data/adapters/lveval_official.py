"""LVEval context + question prompts, with answer-only instructions removed."""

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

TEMPLATES = json.loads(
    Path(__file__).with_name("lveval_official_templates.json").read_text()
)


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    task = row["dataset"].rsplit("_", 1)[0]
    if task not in TEMPLATES:
        raise ValueError(f"Unsupported LVEval task: {task!r}")
    if not isinstance(row["context"], str) or not row["context"].strip():
        raise ValueError("LVEval context must be nonempty text")
    if not isinstance(row["input"], str) or not row["input"].strip():
        raise ValueError("LVEval input must be nonempty text")
    return {
        "prompt": TEMPLATES[task].format(context=row["context"], input=row["input"]),
        "source": "Infinigence/LVEval",
        "category": task,
    }
