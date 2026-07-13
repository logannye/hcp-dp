# Python package

`hcp-align` is a streaming subprocess client for the v0.2 JSONL contract.
Platform wheels bundle the exactly matching Rust binary; source-tree users may
also provide an explicit path. Resolution order is constructor path, bundled
binary, `HCP_ALIGN_BIN`, then `PATH`, with a strict version check.

```python
from hcp_align import Client

client = Client()
for record in client.solve_batch(
    "edit-distance",
    query_file="queries.fa.gz",
    target_file="targets.fa.gz",
):
    print(record["record_type"])
```

`solve_batch`, `verify_batch`, and `compare_batch` are generators: records are
decoded as the binary emits JSONL and are not accumulated by the client.
