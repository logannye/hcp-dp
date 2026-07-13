# v0.2 Capability Boundary

| Capability | Status | Boundary |
|---|---|---|
| Exact edit-distance score and traceback | supported | Unit-cost Levenshtein. |
| Exact global linear alignment | supported | Match/mismatch or BLOSUM62. |
| Exact global affine alignment | supported | HCP or exact diagonal-band/fallback policy. |
| Exact local linear alignment | supported | Best Smith-Waterman trace. |
| Exact semi-global linear alignment | supported | Full query against any target interval. |
| JSON/JSONL solve and verify | supported | `hcp-align.v2`. |
| PAF/SAM solve and verify | supported | Primary forward-strand records. |
| Exact optimality verification | supported | Explicit work budget; no silent downgrade. |
| Backend comparison | supported | Objective equivalence, tie-aware traces. |
| Integrity fingerprints | supported | Mutation detection, not authenticity. |
| Generic HCP authoring | experimental | `experimental-hcp` feature only. |
| Mapping/indexing/MAPQ | not supported | Intentional non-goal. |
| BAM/CRAM/reverse strand | not supported | Deferred beyond v0.2. |
| Seeded extension, DAG, DTW, Viterbi | removed | Not part of the oracle product. |
