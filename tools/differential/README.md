# Differential tooling

This standalone, unpublished package compares the exact internal oracle with
development-only external engines. It is deliberately outside the shipped
`hcp-dp` dependency graph.

Rust-Bio `4.0.1` is pinned in `Cargo.lock`. WFA2-lib is built separately at
tag `v2.3.6`, commit `bcf473a6561fa297934a80eaa7c04b4ee412360c`, using
`scripts/build-wfa2-adapter.sh`; no build script downloads source code.

For semiglobal maximize-score problems, the adapter enumerates target spans
and uses the persistent WFA2 end-to-end kernel for each span. This is slower
than a single ends-free call, but it is exact: the score-to-cost baseline
depends on aligned target length, so a symmetric ends-free cost alone can pick
the wrong span.

Reduce a disagreement artifact:

```console
cargo run --manifest-path tools/differential/Cargo.toml -- reduce \
  --input disagreement.json \
  --left internal:auto \
  --right rust-bio \
  --preserve category-set \
  --output minimized.json
```

Backend specifications are declarative. Artifact contents are never executed.

Run the persistent WFA2 adapter across every supported mode and every binary
sequence pair through length three:

```console
cargo run --manifest-path tools/differential/Cargo.toml -- contract \
  --left internal:auto --right wfa2:/absolute/path/to/wfa2-adapter \
  --max-length 3 --output differential-report.json
```
