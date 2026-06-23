# hcp-align Python Binding

This is a lightweight subprocess binding over the stable `hcp-align --format
json` contract. It intentionally avoids a native extension build.

```python
from hcp_align import HcpAlign

client = HcpAlign()  # or HcpAlign("/path/to/hcp-align")
result = client.edit_distance("kitten", "sitting", verify=True)
print(result["distance"], result["verification_status"])
```

Resolution order:

1. Explicit `HcpAlign(executable=...)`
2. `HCP_ALIGN_BIN`
3. `hcp-align` on `PATH`

The wrapper raises `HcpAlignError` on nonzero CLI exits and returns parsed JSON
objects on success.
