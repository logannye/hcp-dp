# Example Data

Small FASTA and FASTQ files for trying `hcp-align` batch mode.

```bash
hcp-align edit-distance \
  --query-file examples/data/reads.fa \
  --target-file examples/data/references.fa \
  --verify --format jsonl
```

```bash
hcp-align global-linear \
  --query-file examples/data/reads.fq \
  --target-file examples/data/references.fq \
  --match 1 --mismatch-penalty 1 --gap -1 \
  --format cigar
```
