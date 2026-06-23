# Security Policy

HCP-DP is an alpha-stage Rust project. The current public surface is a local
library and CLI; it does not run a hosted service and does not process untrusted
network traffic by itself.

## Supported Versions

| Version | Status |
|---|---|
| `main` | Supported for security fixes. |
| `0.1.0-alpha.*` | Supported once alpha prereleases are published. |

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
include the fix in the next alpha artifact when release artifacts are available.
