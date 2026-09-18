#!/bin/sh
# Offline only. No downloading, model access, global install or runner startup.
set -eu
if [ "$#" -ne 4 ]; then
    printf '%s\n' 'usage: build.sh PINNED_OLLAMA_SOURCE GO_BINARY CACHED_MODULES OUTPUT' >&2
    exit 2
fi
codec_source=$1
codec_go=$2
codec_modules=$3
codec_output=$4
codec_here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
codec_work=$(dirname -- "$codec_output")
test "$("$codec_go" version)" = 'go version go1.26.8 darwin/arm64'
python3 - "$codec_source" "$codec_here" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root, here = map(Path, sys.argv[1:])
manifest = json.loads((here / 'sources.json').read_text())
expected = {entry['path'] for entry in manifest['files'] if entry['path'].endswith('.go')}
actual = {str(path.relative_to(root)) for path in root.rglob('*.go')}
if actual != expected:
    raise SystemExit('source Go file set differs from pinned subset')
for entry in manifest['files']:
    path = root / entry['path']
    raw = path.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != entry['sha256'] or blob != entry['git_blob_sha1']:
        raise SystemExit('pinned source digest mismatch: ' + entry['path'])
for entry in manifest['helpers']:
    if hashlib.sha256((here / entry['path']).read_bytes()).hexdigest() != entry['sha256']:
        raise SystemExit('helper source digest mismatch: ' + entry['path'])
PY
export CGO_ENABLED=0 GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off
export GOPATH="$codec_work/gopath" GOMODCACHE="$codec_modules" GOCACHE="$codec_work/buildcache"
cd "$codec_source"
"$codec_go" test -mod=readonly "$codec_here/parser_main.go" "$codec_here/parser_test.go"
"$codec_go" build -mod=readonly -trimpath -o "$codec_output" "$codec_here/parser_main.go"
"$codec_go" version -m "$codec_output"
