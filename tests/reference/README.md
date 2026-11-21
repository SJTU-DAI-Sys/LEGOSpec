# Pinned test references

These files are test-only numerical/export oracles, not runtime dependencies.
`provenance.json` preserves original paths and SHA256 values; upstream headers
and existing provenance files remain intact. `dpal_before` intentionally keeps
an old training wrapper to test mathematical compatibility. `eagle3_benchmark`
is imported by the native EAGLE3 integration test. Do not edit these references
to make the current implementation pass. Generated outputs belong in
`tests/artifacts/`, while completed experiment records belong in `tests/archive/`.
