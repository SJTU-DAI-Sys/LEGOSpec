"""Select training payloads before Arrow infers unrelated metadata types."""

from pathlib import Path

from datasets import Dataset, load_dataset


def _project_jsonl(path, columns, source_identity):
    # source_identity participates in the generator cache fingerprint. A source
    # edited in place must not silently reuse an older Arrow dataset.
    import json

    try:
        import orjson
        decode = orjson.loads
    except ImportError:
        decode = json.loads
    with open(path, "rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = decode(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            missing = set(columns) - record.keys()
            if missing:
                raise ValueError(f"Missing training columns {sorted(missing)} at {path}:{line_number}")
            yield {column: record[column] for column in columns}


def load_training_json(data_files, *, columns=None, cache_dir=None):
    """Keep the standard loader unless explicit JSONL projection is requested.

    Projection preserves selected values and source order, including all turns.
    Auxiliary inputs such as tools can be selected alongside conversations.
    """
    if columns is None:
        if isinstance(data_files, Path):
            data_files = str(data_files)
        return load_dataset("json", data_files=data_files)["train"]
    # HF treats list-valued gen_kwargs as input shards. Column names are one
    # atomic projection, rather than independent shards to split or iterate.
    columns = tuple(dict.fromkeys(columns))
    if not columns:
        raise ValueError("At least one training data column must be selected")
    path = Path(data_files).expanduser().resolve(strict=True)
    stat = path.stat()
    return Dataset.from_generator(
        _project_jsonl,
        gen_kwargs={
            "path": str(path),
            "columns": columns,
            "source_identity": (stat.st_size, stat.st_mtime_ns),
        },
        cache_dir=cache_dir,
    )
