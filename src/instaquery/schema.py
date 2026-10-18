"""Turn a SQLite database into a compact, LLM-friendly schema description.

Format (one CREATE TABLE per table, with a few example values for text columns
so the model can match literals such as 'France' vs 'france'):

    CREATE TABLE singer (
      singer_id INTEGER PRIMARY KEY,
      name TEXT, -- e.g. 'Joe Sharp', 'Timbaland'
      country TEXT, -- e.g. 'Netherlands', 'United States'
      FOREIGN KEY (x) REFERENCES other(y)
    );
"""

from __future__ import annotations

import re
import sqlite3
from functools import lru_cache
from pathlib import Path

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SQL_KEYWORDS = {"order", "group", "select", "from", "where", "table", "index", "limit", "join",
                 "union", "values", "by", "as", "on", "in", "is", "not", "and", "or", "having",
                 "default", "check", "primary", "key", "foreign", "references", "unique", "match"}
_TEXTY = ("char", "text", "clob", "varchar", "string")


def quote_ident(name: str) -> str:
    if _SAFE_IDENT.match(name) and name.lower() not in _SQL_KEYWORDS:
        return name
    return '"' + name.replace('"', '""') + '"'


def connect_readonly(db_file: str | Path, timeout: float = 30.0) -> sqlite3.Connection:
    uri = f"file:{Path(db_file).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout, check_same_thread=False)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    return conn


def _sample_values(cur, table: str, col: str, k: int) -> list[str]:
    try:
        cur.execute(
            f"SELECT DISTINCT {quote_ident(col)} FROM {quote_ident(table)} "
            f"WHERE {quote_ident(col)} IS NOT NULL AND TRIM(CAST({quote_ident(col)} AS TEXT)) != '' "
            f"LIMIT {k}")
        vals = []
        for (v,) in cur.fetchall():
            s = str(v).replace("\n", " ").strip()
            if len(s) > 40:
                s = s[:37] + "..."
            vals.append("'" + s.replace("'", "''") + "'")
        return vals
    except sqlite3.Error:
        return []


@lru_cache(maxsize=512)
def serialize_schema(db_file: str, sample_values: int = 3) -> str:
    conn = connect_readonly(db_file)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")
    tables = [r[0] for r in cur.fetchall()]
    blocks = []
    for t in tables:
        cur.execute(f"PRAGMA table_info({quote_ident(t)})")
        cols = cur.fetchall()  # cid, name, type, notnull, dflt, pk
        pk_cols = [c[1] for c in cols if c[5]]
        items: list[tuple[str, str]] = []  # (definition, trailing comment)
        for _, name, ctype, _nn, _d, pk in cols:
            ctype = (ctype or "").strip() or "TEXT"
            definition = f"{quote_ident(name)} {ctype}"
            if pk and len(pk_cols) == 1:
                definition += " PRIMARY KEY"
            comment = ""
            if sample_values and any(x in ctype.lower() for x in _TEXTY):
                vals = _sample_values(cur, t, name, sample_values)
                if vals:
                    comment = " -- e.g. " + ", ".join(vals)
            items.append((definition, comment))
        if len(pk_cols) > 1:
            items.append((f"PRIMARY KEY ({', '.join(quote_ident(c) for c in pk_cols)})", ""))
        try:
            cur.execute(f"PRAGMA foreign_key_list({quote_ident(t)})")
            for fk in cur.fetchall():  # id, seq, table, from, to, ...
                ref_t, frm, to = fk[2], fk[3], fk[4]
                to_txt = f"({quote_ident(to)})" if to else ""
                items.append((f"FOREIGN KEY ({quote_ident(frm)}) REFERENCES {quote_ident(ref_t)}{to_txt}", ""))
        except sqlite3.Error:
            pass
        lines = [f"  {d}{',' if i < len(items) - 1 else ''}{c}" for i, (d, c) in enumerate(items)]
        blocks.append(f"CREATE TABLE {quote_ident(t)} (\n" + "\n".join(lines) + "\n);")
    conn.close()
    return "\n\n".join(blocks)


def table_names(db_file: str | Path) -> list[str]:
    conn = connect_readonly(db_file)
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    conn.close()
    return [r[0] for r in rows]
