#!/usr/bin/env bash
# scripts/precheck.sh — single entry point for CI / pre-merge checks.
#
# Runs every static audit + the retrieval eval bench with hard thresholds.
# Exits non-zero if ANY check fails. CI calls this; humans can run it
# locally before pushing.
#
# Usage:
#   ./scripts/precheck.sh              # full check (audits + eval)
#   ./scripts/precheck.sh --audits     # audits only (fast, no LLM/embed)
#   ./scripts/precheck.sh --eval       # eval only
#
# Eval thresholds — these are the floors below which we believe the
# system has regressed. Tune as your golden_set + backend mature.
#   recall@3 ≥ 0.65  — right answer in top-3 for 65% of queries
#   MRR      ≥ 0.55  — average reciprocal rank
# Bump these as quality improves; never lower without team discussion.

set -euo pipefail

# Resolve repo root regardless of where the script is called from.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Flags
RUN_AUDITS=1
RUN_EVAL=1
case "${1:-}" in
    --audits) RUN_EVAL=0 ;;
    --eval)   RUN_AUDITS=0 ;;
    --help|-h)
        sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
esac

# ── ANSI helpers (ignored when stdout is not a tty) ─────────────────────
if [ -t 1 ]; then
    RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; RST=$'\033[0m'
else
    RED=""; GRN=""; YEL=""; RST=""
fi

FAILED=0
section() { printf '\n%s── %s ──%s\n' "$YEL" "$1" "$RST"; }
ok()      { printf '%s✓%s %s\n' "$GRN" "$RST" "$1"; }
fail()    { printf '%s✗%s %s\n' "$RED" "$RST" "$1"; FAILED=1; }

# ── 1. Static audits ────────────────────────────────────────────────────
if [ "$RUN_AUDITS" = 1 ]; then
    section "Static audits"

    # Syntax sweep — every .py parses. Uses the same exclusion list as
    # the other audits (see scripts/_audit_common.py) so .venv /
    # site-packages / build dirs don't poison the result with their own
    # syntax errors (some libs ship test fixtures with deliberately bad
    # Python).
    if python3 -c "
import sys, pathlib
sys.path.insert(0, 'scripts')
from _audit_common import iter_repo_python_files
import ast
errs = []
for f in iter_repo_python_files(pathlib.Path('.')):
    try: ast.parse(open(f).read())
    except SyntaxError as e: errs.append(f'{f}:{e.lineno}: {e.msg}')
for e in errs: print('  ✗', e, file=sys.stderr)
sys.exit(1 if errs else 0)
"; then
        ok "syntax sweep"
    else
        fail "syntax sweep"
    fi

    for audit in audit_module_independence audit_imports audit_prompt_templates \
                 audit_directive_parsing audit_wiring; do
        if python3 "scripts/${audit}.py" > /tmp/_${audit}.out 2>&1; then
            ok "$audit"
        else
            fail "$audit"
            cat /tmp/_${audit}.out
        fi
    done
fi

# ── 2. Retrieval eval gate ──────────────────────────────────────────────
if [ "$RUN_EVAL" = 1 ]; then
    section "Retrieval eval (golden set)"

    # MIN_RECALL_3 / MIN_MRR can be overridden by env so different CI
    # branches (e.g. nightly with higher bar) can use different floors
    # without editing this file.
    MIN_RECALL_3="${MIN_RECALL_3:-0.65}"
    MIN_MRR="${MIN_MRR:-0.55}"

    if python3 -m evaluation.cli \
        --golden data/golden_set.jsonl \
        --backend hybrid \
        --top-k 5 \
        --quiet \
        --fail-below-recall-3 "$MIN_RECALL_3" \
        --fail-below-mrr      "$MIN_MRR"; then
        ok "retrieval eval (recall@3 ≥ $MIN_RECALL_3, MRR ≥ $MIN_MRR)"
    else
        fail "retrieval eval below thresholds (recall@3 < $MIN_RECALL_3 or MRR < $MIN_MRR)"
    fi
fi

# ── Final result ────────────────────────────────────────────────────────
echo
if [ "$FAILED" = 1 ]; then
    printf '%sprecheck: FAIL%s\n' "$RED" "$RST"
    exit 1
fi
printf '%sprecheck: PASS%s\n' "$GRN" "$RST"
