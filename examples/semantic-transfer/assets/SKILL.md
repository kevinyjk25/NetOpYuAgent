---
name: asset-metric-inspection
description: Resolve a caller alias and inspect the canonical non-retired asset.
---
# Asset metric inspection
Use the caller's alias to call resolve_asset.
If the returned asset is retired, stop unsupported. Otherwise call read_asset_metric using the returned canonical_key, NOT the caller's alias.
After that metric read, complete the read path. Retired assets must not be read.
This inspects records only. Invalid replies, missing keys and transport errors block; they are not negative retirement decisions.
