import json
import tempfile
import unittest
from pathlib import Path

from legospec.data.dataset import load_training_json


def write_records(path, records):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n"
    )


def check_projection_ignores_arbitrary_metadata_and_keeps_all_turns(tmp_path):
    path = tmp_path / "answers.jsonl"
    rows = [
        {
            "conversations": [
                {"role": "system", "content": "系统"},
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "follow-up"},
                {"role": "assistant", "content": "second answer"},
            ],
            "category": category,
            "generation": {"arbitrary": category},
        }
        for i, category in enumerate([None, "chat", 7, ["math"], {"nested": True}])
    ]
    write_records(path, rows)
    dataset = load_training_json(
        path, columns=["conversations"], cache_dir=str(tmp_path / "cache")
    )
    assert dataset.column_names == ["conversations"]
    assert list(dataset["conversations"]) == [row["conversations"] for row in rows]
    # Modifying the same path must invalidate the raw cache.
    write_records(path, rows + [rows[0]])
    assert (
        len(
            load_training_json(
                path, columns=["conversations"], cache_dir=str(tmp_path / "cache")
            )
        )
        == 6
    )


def check_projection_preserves_explicit_auxiliary_columns(tmp_path):
    path = tmp_path / "text.jsonl"
    write_records(path, [{"text": "ready text", "tools": "[]", "category": {"any": 1}}])
    dataset = load_training_json(path, columns=["text", "tools"])
    assert dataset[0] == {"text": "ready text", "tools": "[]"}


def check_standard_loader_preserves_columns(tmp_path):
    path = tmp_path / "plain.jsonl"
    write_records(path, [{"text": "first", "category": "chat"}])
    assert load_training_json(path)[0] == {"text": "first", "category": "chat"}


class TrainingJsonProjectionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def test_projection_ignores_arbitrary_metadata_and_keeps_all_turns(self):
        check_projection_ignores_arbitrary_metadata_and_keeps_all_turns(self.path)

    def test_projection_preserves_explicit_auxiliary_columns(self):
        check_projection_preserves_explicit_auxiliary_columns(self.path)

    def test_standard_loader_preserves_columns(self):
        check_standard_loader_preserves_columns(self.path)

    def test_projection_rejects_missing_payload_instead_of_skipping(self):
        path = self.path / "invalid.jsonl"
        write_records(path, [{"category": "chat"}])
        with self.assertRaisesRegex(Exception, "generating the dataset") as error:
            load_training_json(path, columns=["conversations"])
        self.assertIn("Missing training columns", str(error.exception.__cause__))


if __name__ == "__main__":
    unittest.main()
