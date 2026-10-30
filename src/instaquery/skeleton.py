"""SQL skeletons for schema-aware retrieval-augmented prompting.

Finding 3 of the original study: naive RAG (three similar questions from *other*
databases, shown with their real SQL) cut execution accuracy by 14 points, because
the 1.5B model copied foreign table and column names ("unknown table" errors 4 -> 92).

The fix tested here (in the spirit of DAIL-SQL, Gao et al. 2023) keeps the useful part
of an example - the *shape* of the query (JOIN / GROUP BY / HAVING / ORDER BY ... LIMIT)
- and removes the part that hurts - identifiers that do not exist in the target schema:

    SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.id = T2.singer_id WHERE T2.year > 2014
 -> SELECT t1.col1 FROM table1 AS t1 JOIN table2 AS t2 ON t1.col2 = t2.col3 WHERE t2.col4 > <num>
"""

from __future__ import annotations

import re

_STR = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_KEYWORDS = {
    "select", "from", "where", "and", "or", "not", "in", "like", "between", "is", "null", "as", "on", "join",
    "inner", "left", "right", "outer", "cross", "group", "by", "order", "having", "limit", "offset", "asc", "desc",
    "distinct", "union", "intersect", "except", "all", "count", "sum", "avg", "min", "max", "case", "when",
    "then", "else", "end", "exists", "cast", "with", "abs", "length", "lower", "upper", "substr", "strftime",
    "round", "iif", "coalesce", "ifnull", "real", "integer", "text", "date", "julianday", "instr", "replace",
    "natural", "using", "glob", "escape", "nullif", "total", "group_concat", "over", "partition", "rank",
    "row_number", "dense_rank", "true", "false", "values", "float", "trim",
}


def sql_skeleton(sql: str) -> str:
    """Mask table names, column names and literals; keep keywords, operators and structure."""
    sql = " ".join(sql.split()).rstrip(";")
    sql = _STR.sub(" STRVAL ", sql)
    sql = _NUM.sub(" NUMVAL ", sql)
    tables: dict[str, str] = {}
    cols: dict[str, str] = {}
    aliases: dict[str, str] = {}
    tokens = re.split(r"(\s+|[(),=<>!*+\-/]|\.)", sql)
    out: list[str] = []
    prev_kw = ""
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if not tok or tok.isspace() or not re.match(r"^[A-Za-z_]\w*$", tok) or low in _KEYWORDS \
                or tok in ("STRVAL", "NUMVAL"):
            out.append(tok)
            if low in _KEYWORDS:
                prev_kw = low
            continue
        nxt = next((t for t in tokens[i + 1:] if t and not t.isspace()), "")
        if prev_kw in ("from", "join"):            # a table name
            tables.setdefault(low, f"table{len(tables) + 1}")
            out.append(tables[low])
            prev_kw = "__table"
        elif prev_kw == "as" or prev_kw == "__table":  # an alias right after a table
            aliases.setdefault(low, low if re.match(r"^t\d+$", low) else f"t{len(aliases) + 1}")
            out.append(aliases[low])
            prev_kw = ""
        elif nxt == ".":                            # alias.column -> keep alias mapping
            aliases.setdefault(low, low if re.match(r"^t\d+$", low) else f"t{len(aliases) + 1}")
            out.append(aliases[low])
        else:
            cols.setdefault(low, f"col{len(cols) + 1}")
            out.append(cols[low])
    text = " ".join("".join(out).split())
    return text.replace("STRVAL", "<str>").replace("NUMVAL", "<num>").replace("( ", "(").replace(" )", ")")


def format_skeleton_examples(examples: list[dict] | None) -> str:
    if not examples:
        return ""
    parts = ["Here are similar questions from OTHER databases with the STRUCTURE of their SQL. "
             "Table and column names are masked (table1, col1, ...): use only names from the schema above."]
    for ex in examples:
        parts.append(f"Q: {ex['question']}\nSQL structure: {sql_skeleton(ex['query'])}")
    return "\n\n".join(parts) + "\n\n"
