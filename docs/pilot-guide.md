# hcp-align v0.2 pilot guide

This pilot is for aligner/library maintainers and bioinformatics infrastructure
engineers who want an independent exact answer for a real alignment record or
backend disagreement. A useful pilot should take less than an hour and does not
require publishing proprietary sequence data.

## Choose one pilot job

### Validate real records

Use at least 100 records from a real JSON, PAF, or forward-strand SAM workflow,
or use an entire smaller corpus that already represents a production failure.

For native v2 JSON, the scoring model is embedded:

```bash
hcp-align verify \
  --alignment records.jsonl \
  --format jsonl \
  --query-file queries.fa.gz \
  --target-file references.fa.gz \
  --output-format jsonl \
  --output verification.jsonl \
  --errors verification-errors.jsonl
```

PAF and SAM require the exact mode and scoring convention. For example:

```bash
hcp-align verify \
  --alignment records.sam \
  --format sam \
  --query-file queries.fa.gz \
  --target-file references.fa.gz \
  --mode global-affine \
  --match 2 \
  --mismatch-penalty 1 \
  --gap-open -6 \
  --gap-extend -1 \
  --output-format jsonl \
  --output verification.jsonl \
  --errors verification-errors.jsonl
```

Record the total count and the number of `verified`, `invalid`,
`inconclusive`, and `error` results. An `inconclusive` result is not silently
downgraded verification; record whether the cell budget was intentionally
raised or the pilot switched to `--level score`.

### Reproduce a backend disagreement

Generate or import both candidate records, compare them, and distinguish an
objective/validity defect from a harmless optimal-path tie:

```bash
hcp-align compare \
  --left backend-a.jsonl --left-format jsonl \
  --right backend-b.jsonl --right-format jsonl \
  --query-file queries.fa.gz \
  --target-file references.fa.gz \
  --output-format jsonl \
  --output comparison.jsonl
```

Development-tool users can minimize a differential artifact with explicit
backends; artifacts never execute commands embedded in their input:

```bash
cargo run --manifest-path tools/differential/Cargo.toml -- reduce \
  --input disagreement.json \
  --left internal:auto \
  --right rust-bio \
  --preserve category-set \
  --output minimized.json
```

## Report useful evidence

Open the **Pilot evidence** issue template and include:

- pilot identity and engineering context;
- `hcp-align --version` and upstream backend versions;
- input format, mode, scoring, affine-gap, clipping, and coordinate conventions;
- record count and verification status counts;
- the practical outcome: confidence gained, malformed record found, convention
  mismatch isolated, or backend bug reproduced;
- a sanitized or minimized artifact when licensing permits.

Private sequences may be replaced by SHA-256 digests, aggregate counts, or a
minimized synthetic reproducer. Do not publish regulated or licensed sequence
text. A pilot counts for the stable release only when the public evidence issue
documents either at least 100 real records (or a complete smaller failure
corpus) or one reproducible backend defect.
