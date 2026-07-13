# v0.2 Release Checklist

## Hosted configuration completed

- The `main` branch disallows force pushes and deletion and requires the
  repository's release checks.
- The protected `release` and `pypi` GitHub environments require approval and
  accept only `v0.2.*` tags.
- PyPI has a pending trusted publisher for `hcp-align` bound to
  `logannye/hcp-dp`, `release.yml`, and the `pypi` environment.
- crates.io trusted publishing is bound to `logannye/hcp-dp`, `release.yml`,
  and the `release` environment after the manual bootstrap publication.

## Every candidate

- README, CLI help, schemas, Cargo metadata, Python metadata, and tag agree.
- Formatting, Clippy, tests, rustdoc, MSRV, feature, Python, adapter, and schema
  checks pass.
- `cargo publish --dry-run --locked` contains only the intended crate surface.
- Every archive and wheel installs on its target platform and passes the same
  exact edit-distance smoke corpus.
- JSON v2 error records and differential artifacts validate against their
  distinct schemas.
- Every adapter conversion is independently verified; discrepancies are either
  fixed or have a versioned signature, reason, and issue URL in the allowlist.
- The release tag is annotated, cryptographically verified by GitHub, and
  targets the exact checked-out release commit.
- A pushed release tag is immutable. Transient jobs may be rerun, but any code
  or artifact correction requires a new version and tag.

## `0.2.0-alpha.2`

Completed on 2026-07-13. The immutable signed tag, GitHub prerelease, PyPI
wheels, crates.io crate, docs.rs build, checksums, Sigstore bundle, and GitHub
attestations were verified through fresh public installs.

- The Linux 1,000/100,000 solve RSS delta is no more than 16 MiB.
- Solve and arbitrary-order verify peaks are no more than 128 MiB and no more
  than 15% above the checked baseline.
- Ordered output, count/base backpressure, slow-first-job behavior, and
  recoverable continuation tests pass.
- Create a clean signed tag, run the tag workflow, then manually perform the
  first crates.io publication from that exact tag.
- Configure crates.io trusted publishing only after the first crate exists.
- Verify clean installations from the public GitHub, PyPI, and crates.io
  surfaces after publication; workflow-local artifacts are insufficient.

## `0.2.0-rc.1` and `0.2.0`

- `release` and `pypi` GitHub environments require approval.
- `main` disallows force push/deletion and has all required checks.
- Four consecutive scheduled RSS/performance runs pass.
- The GitHub release contains five archives, five platform wheels, SPDX SBOMs,
  differential/performance reports, `SHA256SUMS`, its Sigstore bundle, and
  GitHub attestations from one tagged commit.
- PyPI uses its pending trusted publisher; crates.io uses OIDC after alpha.2.
- Three-user external-adoption evidence is linked before creating `v0.2.0`.
  Record each pilot in `release/adoption.json`; the stable tag workflow refuses
  fewer than three distinct pilots or missing public evidence links.
- `v0.2.0-rc.1` has been public for at least seven days and no open issue has
  the `release-blocker` label; the stable tag workflow enforces both conditions.
