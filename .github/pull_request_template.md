## Summary

- 

## Correctness Evidence

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
- [ ] `cargo test --workspace`
- [ ] `bash scripts/check.sh`

## Contract Impact

- [ ] No exported problem contract changed.
- [ ] Summary laws remain covered by tests.
- [ ] Returned paths remain independently scored.
- [ ] Full-table or external baselines were updated where applicable.

## Public Surface

- [ ] README, CLI docs, capability matrix, schema docs, and changelog were
      updated where user-visible behavior changed.
- [ ] Performance or memory claims are backed by a reproducible command or
      report artifact.

## Notes

Mention any intentionally deferred follow-up work, performance caveats, or
breaking CLI/API changes.
