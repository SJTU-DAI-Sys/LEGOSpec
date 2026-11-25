# Pinned test references

These are test-only numerical/export oracles, not runtime dependencies.
Provenance JSON files preserve original paths, revisions and SHA256 values.
Keep upstream notices and licenses; do not edit reference math to make a test pass.

The current tests use the D-PAL/EAGLE3 references, frozen verifier, and active
loss-regression fixtures. Deferred survival fixtures moved with their code to
development/objectives/tests/fixtures/.

Generated output belongs in tmp_exp/artifacts/. Experiment records, debugging
scripts and historical snapshots belong in tmp_exp/, outside Git and releases.
