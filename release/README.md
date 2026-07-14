# Stable-release evidence

`adoption.json` is deliberately empty until external pilots use the oracle on
real records. Before creating `v0.2.0`, add at least three distinct entries:

```json
{
  "pilot": "organization-or-maintainer",
  "job": "validated_records",
  "summary": "What records were checked and what practical result was obtained.",
  "evidence_url": "https://github.com/logannye/hcp-dp/issues/123",
  "validated_at": "2026-08-01"
}
```

`job` is either `validated_records` or `reproduced_alignment_bug`. The tag
workflow validates this evidence and refuses the stable release until three
distinct pilots have public HTTPS evidence.

Use [`docs/pilot-guide.md`](../docs/pilot-guide.md) for the execution and
privacy contract. The approved outreach ledger, public invitation links, and
original drafts live in [`pilot-outreach.md`](pilot-outreach.md). Invitations
are not release evidence until the corresponding pilot meets the documented
threshold.
