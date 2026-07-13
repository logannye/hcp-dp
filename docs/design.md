# v0.2 Technical Design

The stable architecture has four layers:

1. Canonical alignment and scoring types.
2. Exact path-producing solvers.
3. Independent structure, realized-score, and rolling-frontier optimality checks.
4. JSON/PAF/SAM format adapters and the thin CLI.

Every public alignment is normalized to query/target spans and SAM-oriented
`=XID` CIGAR. Format-specific coordinates and clipping are converted only at
the adapter boundary.

The verifier deliberately separates three claims:

- the record is structurally coherent;
- the CIGAR realizes the claimed objective on the supplied sequences;
- the objective equals an independently computed exact optimum.

An optimal request that exceeds its work budget is `inconclusive`, never
`verified` at a weaker level.

The original height-compressed engine remains an exact path-producing backend.
Its traits and problem-author surface are feature-gated under
`experimental::hcp`; they are not the v0.2 stability boundary.
