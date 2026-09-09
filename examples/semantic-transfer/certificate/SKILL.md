---
name: certificate-inspection
description: Resolve a name and read the canonical certificate serial under the referenced expiry rule.
---
# Certificate inspection
Use the caller's dns_name to resolve_certificate. Apply the supplied references/expiry.md rule before the next read.
If inspection is permitted, pass the returned serial_number as cert_serial to read_certificate, not the original dns_name. Then complete the read path.
This Skill cannot renew or revoke certificates. A provider failure or malformed result blocks; do not assume a valid certificate.
