# Qwen3.8-27B generation resumed on H800_2

Transferred from QZ on 2026-11-23. All nine input/output files were verified against source SHA256 hashes; see migration_verified.json and source_transfer_manifest.json. Preserved successes: part0=278074, part1=276575, total=554649. Original failed records remain.

Uses all eight physical GPUs (TP2 groups 0/1,4/5,6/7,2/3), four TP2 services, 256 concurrent conversations per service. The other environment tests finished; the user authorized expansion to all eight GPUs. No existing main-project environment was modified. activate.sh selects the existing SGLang 0.5.21 generation environment and a job-local SGLang source snapshot.

Sampling: T=1, top-p=0.95, top-k=20, min-p=0, presence-penalty=0, repetition-penalty=1; medium reasoning, strict thinking budget 2048, max-new-tokens 5120 per turn, native multi-turn template and preserved reasoning. See config.json.

The generator processes both imported shards sequentially using all four services. Existing successful IDs are skipped. Unprocessed prompts run before historical failures are retried. Results is a symlink to the original imported output directory: results/part0.jsonl and results/part1.jsonl. Both files append in place.

Main log: logs/latest.log. Server logs: logs/server.*.replica*.log. Controller state: logs/launch.*.json. Full real-GPU smoke passed; receipt: results/smoke.parts0-1.json. CPU request/budget tests and eleven regression tests passed.

Restart only after checking that this job is stopped and these GPUs are free: bash start.sh. It preserves outputs and repeats the existing verified receipt checks. Do not launch a second controller while one is running.

8-GPU expansion: config.6gpu.json, smoke.6gpu.json and part*.6gpu_identity.json retain provenance. Only replicas changed from 3 to 4; all generation parameters and input files are unchanged. verify_8gpu.py checks the added replica before publishing the expanded receipt.

2026-11-23 recovery: the 8-GPU client hit RLIMIT_NOFILE=1024. start.sh now sets 65536 and a 2048-descriptor check passed. User chose six free GPUs while other users occupy 6/7. Current TP2 groups: 0/1,4/5,2/3; concurrency 256/group. All 555612 successful conversations were preserved. config.8gpu.json and smoke.8gpu.json retain prior topology provenance; sampling/template/model/client are unchanged. See nofile_recovery.json.

Current update 2026-11-23 13:39 Asia/Shanghai: user requested 160 concurrent conversations per TP2 group (480 total, same six GPUs). Existing 556086 successes were retained. All sampling/model/template parameters and mem_fraction_static=0.90 are unchanged. max_mamba_cache_size=160; KV pool grew to 1072502 tokens/group. Runtime checks on all endpoints and eleven regression tests passed.

Benchmark: after 180-second warmup, a 300-second production window observed 6999 decode tokens/s, 0 retractions, and 24 committed conversations (0.080/s). Previous 256/group baseline was 7148 decode tokens/s, 110 retractions/300s and 0.601 committed conversations/s from progress records. These are different live-production windows, not identical-request A/B tests; interrupted in-flight multi-turn work restarts, so startup completion rates are not long-run estimates. No throughput improvement was demonstrated in this initial window; concurrency remains at the requested 160. Details: throughput_256_before160.json, throughput_160_after_restart.json and concurrency160_migration.json.
