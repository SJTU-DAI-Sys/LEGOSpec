# Data

This package prepares conversations, captures target hidden states, and reads and writes feature caches. It contains six implementation modules and one public entrypoint.

## Conversation datasets

- `template.py`: chat-template definitions and registration.
- `parse.py`: General, Harmony, and Thinking parsers.
- `dataset.py`: JSONL field projection, text/vision preprocessing, tokenization, loss masks, HF Dataset construction, and text/vision collators.
- `__init__.py`: public dataset and hidden-cache interfaces.

Collators receive sequence-parallel dimensions explicitly. Training samplers and DataLoader construction belong to `training/data/loader.py`.

## Hidden capture

`hidden_capture.py` owns task assignment, batch preparation, target calls, reusable pinned buffers, asynchronous GPU-to-CPU transfer, bounded writer scheduling, and capture metrics.

The CLI is parsed in `scripts/data/capture_hidden_states.py`. Target execution, model scaling, and final normalization remain in `modeling/target`.

## Hidden cache

`hidden_cache.py` owns RawSample encoding and selective decoding, streaming tar writes and checksums, shard recovery, asynchronous durable commits, manifest publication and validation, sample-index validation, cache errors, atomic JSON writes, and persistence operations.

Atomic JSON replacement and durable commits retain their distinct synchronization behavior. Storage does not import capture orchestration or dataset readers.

## Hidden dataset

`hidden_dataset.py` provides RegenFullDataset, selective reads, bounded open-file lifetime, cached sample metadata, RegenFullCollator, metadata-based anchor eligibility, complete finite-value audits, and audit-receipt validation with spawned CPU workers.

Audit and normal reading share the same reader. Optional final-norm reconstruction calls the target adapter.

## Dependency flow

Conversation loading follows `dataset → parse → template`. Capture consumes datasets, target interfaces, and hidden-cache storage. Hidden datasets consume storage; storage does not import either consumer.

Caches store raw decoder features separately from normalized teacher features. Raw-tar bytes, tensor ownership, commit ordering, and sample ordering are preserved across module organization changes.
