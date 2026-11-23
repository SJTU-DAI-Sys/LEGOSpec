"""Extract model-independent prompts from the Qwen3-4B training mixture."""


def extract(row):
    return {
        "messages": row["conversations"],
        "source": row.get("source") or "legospec_mix1712773",
        # The shared converter assigns sequential output IDs. Keep the original
        # identity for matching prompts back to the completed training data.
        "category": {
            "original_id": row["id"],
            "original_category": row.get("category"),
        },
    }
