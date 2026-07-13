# Python subprocess client

This client targets `hcp-align.v2` and resolves the executable from an explicit
constructor argument, `HCP_ALIGN_BIN`, or `PATH`.

```python
from hcp_align import HcpAlign

client = HcpAlign()
record = client.solve_edit_distance("kitten", "sitting")
print(record["alignment"]["objective"]["value"])
```

`verify_json` returns structured invalid and inconclusive reports for exit codes
1 and 3; command/configuration exit code 2 raises `HcpAlignError`.
