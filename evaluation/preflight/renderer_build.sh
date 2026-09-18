#!/bin/sh
# Offline build: the caller supplies a verified, unmodified Ollama source subset
# and portable Go 1.26.8. No downloading, model access, or global installation.
set -eu
if [ "$#" -ne 3 ]; then
    printf '%s\n' 'usage: renderer_build.sh OLLAMA_SOURCE GO_BINARY OUTPUT' >&2
    exit 2
fi
renderer_source=$1
renderer_go=$2
renderer_output=$3
renderer_here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
renderer_work=$(dirname -- "$renderer_output")
test "$("$renderer_go" version)" = 'go version go1.26.8 darwin/arm64'
# Compare both file content and the complete .go input set: no unpinned extra
# package init files can slip into the compiled subset.
python3 - "$renderer_source" "$renderer_here/renderer_sources.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
manifest = json.loads(Path(sys.argv[2]).read_text())
expected = {entry['path'] for entry in manifest['files'] if entry['path'].endswith('.go')}
actual = {str(path.relative_to(root)) for path in root.rglob('*.go')}
if actual != expected:
    raise SystemExit('source Go file set differs from pinned subset')
for entry in manifest['files']:
    path = root / entry['path']
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise SystemExit('pinned source digest mismatch: ' + entry['path'])
PY
export CGO_ENABLED=0 GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off
export GOPATH="$renderer_work/gopath" GOMODCACHE="$renderer_work/modcache" GOCACHE="$renderer_work/buildcache"
cd "$renderer_source"
# Do not run `go mod verify` against Ollama's entire module graph: this isolated
# source subset does not download unrelated production backends. Normal build
# module checks remain enabled and go.sum itself is content-pinned above.
"$renderer_go" test -mod=readonly "$renderer_here/renderer_main.go" "$renderer_here/renderer_test.go"
"$renderer_go" build -mod=readonly -trimpath -o "$renderer_output" "$renderer_here/renderer_main.go"
"$renderer_go" version -m "$renderer_output"
