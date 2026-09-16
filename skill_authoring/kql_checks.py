"""Inert, deliberately partial KQL verification. Never a query executor.

Recognize literal time conjunctions and a closed set of post-aggregation column
operations. Unknown syntax/functions lose proof, not permission. No Skill names,
business column names, catalog prose interpretation or generated repairs.
"""
import re
from datetime import datetime

STAMP = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
IDENT = r"[A-Za-z_]\w*"
LITERAL = rf"datetime\(\s*(?:{STAMP}|'(?:{STAMP})'|\"(?:{STAMP})\")\s*\)"


def split_code(code, delimiter="|"):
    """Split only outside strings/parentheses; comments cannot create operators."""
    parts, buf, quote, depth, i = [], [], None, 0, 0
    while i < len(code):
        char = code[i]
        if quote:
            buf.append(char)
            if char == "\\" and i + 1 < len(code):
                i += 1
                buf.append(code[i])
            elif char == quote:
                if i + 1 < len(code) and code[i + 1] == quote:
                    i += 1
                    buf.append(code[i])
                else:
                    quote = None
        elif code[i:i + 2] == "//":
            end = code.find("\n", i)
            i = len(code) if end < 0 else end
            buf.append(" ")
            continue
        elif char in "'\"":
            quote = char
            buf.append(char)
        elif char in "[];" or code[i:i + 2] == "/*":
            return None  # Dynamic values, identifiers, multiple statements, block comments.
        elif char == "(":
            depth += 1
            buf.append(char)
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
            buf.append(char)
        elif char == delimiter and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(char)
        i += 1
    return None if quote or depth else [*parts, "".join(buf).strip()]


def interval(code, windows):
    """Judge just the literal time-filter component, not whole-query semantics."""
    unknown = ("unverified", "Outside the single-pipeline literal time-conjunction subset; no equivalent-window claim.")
    parts = split_code(code)
    if len(windows) != 1 or not parts or not re.fullmatch(rf"{IDENT}(?:\(\s*\))?", parts[0]):
        return unknown
    (lower, upper), _ = next(iter(windows.items()))
    try:
        if datetime.fromisoformat(lower) >= datetime.fromisoformat(upper):
            return unknown
    except ValueError:
        return unknown
    terms, other_filters = [], []
    for index, part in enumerate(parts[1:], start=1):
        match = re.fullmatch(r"where\s+(.+)", part, re.S | re.I)
        if not match:
            # No proof across nested/union/derived pipelines or post-aggregation filters.
            if any(re.match(r"where\s", later, re.I) for later in parts[index + 1:]):
                return unknown
            break
        expr = match[1]
        if not re.search(r"\bdatetime\s*\(|\bbetween\b", expr, re.I):
            other_filters.append(expr)
            continue
        for term in re.split(r"\s+and\s+", expr, flags=re.I):
            compare = re.fullmatch(rf"({IDENT})\s*(>=|<=|>|<)\s*({LITERAL})", term.strip(), re.I)
            between = re.fullmatch(rf"({IDENT})\s+between\s*\(\s*({LITERAL})\s*\.\.\s*({LITERAL})\s*\)", term.strip(), re.I)
            if compare:
                name, op, value = compare.groups()
                terms.append((name, op, re.search(STAMP, value)[0]))
            elif between:
                name, lo, hi = between.groups()
                terms.extend([(name, ">=", re.search(STAMP, lo)[0]), (name, "<=", re.search(STAMP, hi)[0])])
            else:
                return unknown
    if not terms or len({t[0] for t in terms}) != 1:
        return unknown
    field = terms[0][0]
    # A nonliteral predicate on the same field could change the interval.
    if any(re.search(rf"\b{re.escape(field)}\b", expression) for expression in other_filters):
        return unknown
    lows = [(value, op == ">") for _, op, value in terms if op in {">", ">="}]
    highs = [(value, op == "<") for _, op, value in terms if op in {"<", "<="}]
    if not lows or not highs:
        return unknown
    try:
        for _, _, value in terms:
            datetime.fromisoformat(value)
    except ValueError:
        return unknown
    lo = max(lows)
    hi = min(highs, key=lambda item: (item[0], not item[1]))
    good = lo == (lower, False) and hi == (upper, True)
    return ("pass" if good else "fail", "The recognized literal time predicates "
            + ("preserve" if good else "do not preserve")
            + " the supplied [start,end) endpoints. between includes both bounds; field selection, other filters and full query behavior are not certified.")


def column_checks(code):
    """Known column sets only after simple summarize/project; unknown invoke abstains."""
    parts = split_code(code)
    if not parts or not re.fullmatch(rf"{IDENT}(?:\(\s*\))?", parts[0]):
        return []
    columns, rows, had_aggregate = None, [], False
    for part in parts[1:]:
        if re.match(r"summarize\s", part, re.I):
            had_aggregate = True
            body = re.split(r"\s+by\s+", part[len("summarize"):].strip(), flags=re.I)
            aggs = split_code(body[0], ",")
            groups = split_code(body[1], ",") if len(body) == 2 else []
            parsed = [re.fullmatch(rf"({IDENT})\s*=\s*(?:count\(\s*\)|(?:sum|min|max|avg|dcount)\(\s*{IDENT}\s*\))", a, re.I) for a in aggs or []]
            columns = ({m[1] for m in parsed} | set(groups)
                       if aggs and all(parsed) and len(body) <= 2 and groups is not None
                       and all(re.fullmatch(IDENT, g) for g in groups) else None)
        elif re.match(r"project\s", part, re.I) and had_aggregate:
            selected = split_code(part[len("project"):].strip(), ",")
            if not selected or not all(re.fullmatch(IDENT, name) for name in selected):
                columns = None
                rows.append(("unverified", part, "Projection expression outside the plain-column subset."))
                continue
            missing = set(selected) - columns if columns is not None else None
            state = "unverified" if missing is None else "fail" if missing else "pass"
            detail = ("An intervening unsupported operator/function or aggregate prevents column proof. Check projection against its actual declared output; no catalog semantics inferred from prose."
                      if missing is None else "Projection references columns removed by aggregation: " + ", ".join(sorted(missing))
                      if missing else "Projected columns exist in the recognized aggregate output; input types and function behavior are not certified.")
            rows.append((state, part, detail))
            columns = set(selected) if missing == set() else None
        elif re.match(r"(?:where|order\s+by|sort\s+by|take|limit)\s", part, re.I):
            continue
        else:
            columns = None
    return rows
