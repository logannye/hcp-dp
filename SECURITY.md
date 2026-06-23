# Security Policy

HCP-DP is a local Rust library and CLI. It does not run a hosted service and does
not process network traffic by itself. Security reports are still welcome,
especially for crashes, memory exhaustion, unsafe archive handling, or behavior
that could affect automated batch pipelines.

## Supported Versions

| Version | Status |
|---|---|
| `main` | Supported for security fixes. |
| `0.1.0-alpha.*` | Supported for security fixes. |

Older commits and experimental branches are not supported.

## Reporting A Vulnerability

Use GitHub private vulnerability reporting if it is available for this
repository. If it is not available, open a minimal public issue asking for a
security contact path and do not include exploit details in the issue body.

Please include:

- affected command or API,
- input shape needed to reproduce the issue,
- observed impact,
- platform and Rust version,
- whether the issue affects default builds or optional features.

The maintainer will acknowledge valid reports, work toward a fix on `main`, and
include the fix in the next alpha artifact.
