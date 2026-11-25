# Preparing training data

Prepare conversations for online training, or capture target features for offline training:

```text
arbitrary dataset
  -> convert_dataset.py
  -> standard prompt JSONL
  -> regenerate_train_data.py
  -> regen_token_only (generated-answer JSONL)
  -> capture_hidden_states.py
  -> regen_full (token/mask/hidden-state tar shards)
```

## 1. Convert a source dataset

Inspect a few records, create a dataset adapter under `adapters/`, then convert
the entire dataset:

```bash
python scripts/data/convert_dataset.py inspect \
  --input /path/to/source.parquet --rows 5

python scripts/data/convert_dataset.py convert \
  --input /path/to/source.parquet \
  --adapter scripts/data/adapters/my_dataset.json \
  --output /path/to/prompts.jsonl \
  --turn-mode multi
```

Hugging Face Parquet inputs can be filtered while they are streamed. Repeating
`--hf-filter` combines predicates with AND and lets the Parquet reader skip
files or row groups when their metadata permits it:

```bash
python scripts/data/convert_dataset.py convert \
  --input allenai/WildChat-4.8M \
  --input-format hf \
  --split train \
  --hf-filter language '==' Chinese \
  --hf-filter model '==' gpt-4-0314 \
  --adapter scripts/data/adapters/wildchat_4_8m.json \
  --output cache/data/prompts/wildchat_chinese_gpt-4-0314.jsonl \
  --turn-mode first \
  --skip-invalid
```

This does not materialize the complete Hub dataset locally. Filtering is pushed
into the Hugging Face Parquet loader; row groups that cannot be excluded from
Parquet statistics may still be transferred and scanned. The WildChat adapter
stores the conversation-level language in `category`. `--turn-mode first` keeps
the first user request; use `multi` only when all user turns should be retained
after the original assistant turns are removed.
Empty messages inside a conversation are ignored. `--skip-invalid` skips the
rare conversation that has no usable user message at all.

The converter emits exactly this record contract:

```json
{"id": 0, "conversations": [{"role": "user", "content": "..."}], "source": "my-dataset", "category": null}
```

Only `system` and `user` messages are retained at this stage. The converter is
the only component that knows the source dataset's original fields.

## 2. Generate assistant responses

```bash
python scripts/data/regenerate_train_data.py \
  --model /path/to/Qwen3-4B \
  --input-file-path /path/to/prompts.jsonl \
  --server-address localhost:30000
```

Successful records keep the same four-field contract and gain generated
`assistant` messages. Failed records are written to a separate error JSONL and
are not included in the successful output. Use `--resume` to continue by input
`id`. This is the `regen_token_only` save mode: generated answer tokens are
stored as decoded text without target-model hidden states. Its default location
is `./cache/data/regen_token_only/`; `--output-file-path` overrides it.

For regeneration input, `category` is optional. When it is absent,
`regenerate_train_data.py` normalizes it to `null` in successful and error
outputs so the downstream four-field contract remains stable.

## 3. Capture raw-tar hidden states

```bash
python -m scripts.data.capture_hidden_states \
  --target-model-path /path/to/target \
  --target-model-backend sglang \
  --data-path /path/to/regenerated.jsonl \
  --output-path /path/to/raw-cache \
  --target-layer-ids 1,9,17,25,33 \
  --max-length 4096 --batch-size 1 \
  --tar-shard-bytes 2147483648
```

Capture produces raw-tar shards and a manifest. Set the layer IDs to match your draft recipe and keep the target checkpoint available for training. Capture needs a configured target backend and sufficient GPU memory.

For offline training, set `launcher.train_hidden_states` to the completed cache directory instead of `launcher.train_data`. Training consumes a complete manifest. The cache stores selected raw decoder features separately from the normalized teacher feature used for vocabulary supervision.

Use `--resume` to continue an interrupted capture. It checks committed shards and captures missing records. `--tar-shard-bytes` controls shard size. For Hugging Face capture, select `--target-model-backend hf --no-packed-requests`.

Inspect all capture options with:

```bash
python -m scripts.data.capture_hidden_states --help
```

For response regeneration, start a compatible SGLang server before running step 2 and pass its address through `--server-address`. Choose generation temperature and token limits for your dataset. Existing answer conversations can be used directly for training without repeating regeneration.
