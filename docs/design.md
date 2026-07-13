# v0.2 Technical Design

The stable architecture has six layers:

1. Canonical alignment and scoring types.
2. Exact path-producing solvers.
3. Independent structure, realized-score, and rolling-frontier optimality checks.
4. JSON/PAF/SAM format adapters and the thin CLI.
5. Incremental input, transient ID storage, bounded ordered execution, and
   streaming writers.
6. Development-only external adapters and reducers outside the runtime crate.

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

Batch memory is bounded by the largest active records and configured worker
window, not total input count. Arbitrary-ID source data lives in a temporary
redb database with a fixed cache. SAM bodies and reference metadata are spooled
to disk so header requirements do not force record accumulation.

Record boundaries define recovery. JSONL, PAF, SAM, FASTA, and FASTQ can name a
failed record. A malformed whole JSON container, corrupt compression stream,
duplicate source ID, or store failure invalidates downstream assumptions and is
fatal.
