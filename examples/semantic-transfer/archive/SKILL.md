---
name: archive-metadata-inspection
description: Require the supplied precheck script before reading archive metadata.
---
# Archive metadata inspection
Run scripts/precheck.py with an approved runner before calling read_archive_metadata for the caller's archive_id.
The script is supplied but the host provides no approved runner contract. Stop unsupported before reading metadata and explain that missing runner.
The presence of a script file is not evidence that it ran. Translation must not execute this script or invent a runner.
