# v0.2 Alpha Release Checklist

- README, CLI help, schema, and binary describe the same commands.
- Formatting, Clippy, tests, rustdoc, feature checks, and package build pass.
- JSON v2 output validates against the checked-in schema.
- Every mode round-trips through JSON, PAF, and SAM where traceback exists.
- Mutation tests identify named failed checks.
- Verification exit codes 0, 1, 2, and 3 are covered.
- Scheduled performance workflow produces every expected Criterion estimate.
- Release binaries are built from the tagged commit and include checksums,
  provenance, license, and migration guide.
