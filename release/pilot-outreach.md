# Pilot outreach ledger and drafts

Prepared on 2026-07-13 and approved by the repository owner on 2026-07-14.
Five invitations were sent through public issues in this repository so the
request, response, and any resulting evidence remain auditable. An invitation
does not count toward the adoption gate until its pilot satisfies
[`docs/pilot-guide.md`](../docs/pilot-guide.md).

## Outreach status

| Target | Contact | Invitation | Status |
| --- | --- | --- | --- |
| WFA2-lib | `@smarco` | [#5](https://github.com/logannye/hcp-dp/issues/5) | Awaiting response |
| Rust-Bio | `@johanneskoester` | [#6](https://github.com/logannye/hcp-dp/issues/6) | Awaiting response |
| Edlib | `@Martinsos` | [#7](https://github.com/logannye/hcp-dp/issues/7) | Awaiting response |
| Parasail | `@jeffdaily` | [#8](https://github.com/logannye/hcp-dp/issues/8) | Awaiting response |
| nf-core Bowtie2 | `@joseespinosa`, `@drpatelh` | [#9](https://github.com/logannye/hcp-dp/issues/9) | Awaiting response |

## Shortlist

1. **Santiago Marco-Sola / WFA2-lib** (`smarco/WFA2-lib`) — validates exact
   affine/ends-free scoring translations and the WFA2 differential adapter.
2. **Rust-Bio maintainers** (`rust-bio/rust-bio`) — validates Rust API
   ergonomics, pairwise gap conventions, and the Rust-Bio 4 adapter.
3. **Martin Šošić / Edlib** (`Martinsos/edlib`) — validates edit-distance path,
   coordinate, and CIGAR equivalence against an established exact library.
4. **Jeff Daily / Parasail** (`jeffdaily/parasail`) — validates linear/affine
   score translation and SIMD-backend disagreement reporting.
5. **nf-core maintainers** (`nf-core/modules` or an alignment-heavy pipeline)
   — tests the secondary pipeline-QA job on real workflow records and
   operational batch behavior.

GitHub issues or project-listed maintainer channels are preferred over personal
email. The upstream projects' current issue/contact policies were checked
immediately before these invitations were opened.

## Shared pilot offer

The ask is one 30–60 minute pilot using the signed prerelease. We will help
translate scoring conventions, accept sanitized/digested evidence, minimize any
reproducer, and convert confirmed defects into regression fixtures. No product
endorsement is requested.

## Draft: WFA2-lib

> We built `hcp-align`, a pure-Rust exact alignment oracle that independently
> verifies objective, spans, and canonical CIGAR, with a pinned WFA2-lib 2.3.6
> differential adapter. Would you be willing to try one small affine or
> ends-free corpus? We are specifically looking for translation mistakes or
> cases where two valid optimal paths should not be reported as bugs. A
> sanitized/minimized fixture is sufficient, and we will document the result
> publicly without asking for an endorsement.

## Draft: Rust-Bio

> We are piloting `hcp-align`, an exact solve/verify/compare tool with a
> Rust-Bio 4.0.1 adapter and explicit first-gap affine translation. Could a
> maintainer try a small pairwise-alignment corpus or review the canonical
> `=XID` conversion? The useful outcome is either 100 verified real records, a
> complete smaller failure corpus, or one portable disagreement artifact. We
> will handle reduction and regression coverage.

## Draft: Edlib

> We built an exact alignment verification tool and use Edlib as an
> informational edit-distance reference. Would you be open to checking a small
> corpus for objective/path/coordinate agreement? Different optimal CIGARs are
> accepted; we are trying to catch only invalid paths or objective differences.
> We can minimize and sanitize any failing pair before documenting it.

## Draft: Parasail

> We are validating `hcp-align` against pinned Parasail, including explicit
> linear and affine gap-convention calibration. Would you be willing to try or
> review a small real corpus? We are most interested in score-translation and
> clipping/coordinate mistakes, not throughput comparisons. Any confirmed
> discrepancy will be reduced and independently rescored before it is reported.

## Draft: nf-core

> We are looking for one pipeline-QA pilot of `hcp-align`, an exact verifier for
> JSON/PAF/forward-strand SAM records. The test can use 100 records from an
> alignment module (or a complete smaller failure corpus), and sequence text
> can remain private; aggregate statuses and digests are enough. We want to
> learn whether exact structure/score/optimality checks expose a real workflow
> bottleneck or merely duplicate existing QA.
