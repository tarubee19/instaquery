"""Guardrails for running model-generated SQL against a live database (used by the demo app).

Layers: (1) parse with sqlglot and allow a single read-only SELECT/WITH statement,
(2) open the database in read-only mode, (3) hard timeout and row cap.
"""

from __future__ import annotations

import re

_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b", re.I)


def check_sql(sql: str) -> tuple[bool, str]:
    s = (sql or "").strip().rstrip(";")
    if not s:
        return False, "empty query"
    if ";" in s:
        try:
            import sqlglot
            if len([x for x in sqlglot.parse(s, read="sqlite") if x is not None]) > 1:
                return False, "only one statement is allowed"
        except Exception:
            return False, "only one statement is allowed"
    try:
        import sqlglot
        from sqlglot import exp
        tree = sqlglot.parse_one(s, read="sqlite")
        if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.With)) and \
                not (isinstance(tree, exp.Subquery)):
            return False, f"only read-only SELECT queries are allowed (got {type(tree).__name__})"
        for node in tree.walk():
            n = node[0] if isinstance(node, tuple) else node
            if isinstance(n, (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter)):
                return False, "data-modifying statements are not allowed"
    except ImportError:
        if not re.match(r"^\s*(with|select)\b", s, re.I) or _FORBIDDEN.search(s):
            return False, "only read-only SELECT queries are allowed"
    except Exception as e:
        return False, f"could not parse SQL: {e}"
    return True, "ok"
