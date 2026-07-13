## Summary

- 

## Correctness Evidence

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
- [ ] `cargo test --workspace`
- [ ] `bash scripts/check.sh`

## Oracle Contract Impact

- [ ] Canonical CIGAR, coordinate, scoring, and gap semantics remain explicit.
- [ ] Returned paths are independently rescored.
- [ ] Reference optimum and multiple-optimum behavior remain covered by tests.
- [ ] JSON/PAF/SAM fixtures and checked baselines were updated where applicable.

## Public Surface

- [ ] README, CLI docs, capability matrix, schema docs, and changelog were
      updated where user-visible behavior changed.
- [ ] Performance or memory claims are backed by a reproducible command or
      report artifact.

## Notes

Mention any intentionally deferred follow-up work, performance caveats, or
breaking CLI/API changes.
