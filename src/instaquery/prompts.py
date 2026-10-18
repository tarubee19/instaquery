"""Prompt construction and SQL post-processing (shared by training, eval and serving)."""

from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are an expert data analyst who writes correct SQLite queries. "
    "Given a database schema and a question, reply with exactly one SQLite query that answers it. "
    "Return only the SQL, with no explanation and no markdown."
)


def format_examples(examples: list[dict] | None) -> str:
    if not examples:
        return ""
    parts = ["Here are example questions with their correct SQL from other databases "
             "(use them only as a guide to SQL patterns):"]
    for ex in examples:
        parts.append(f"Q: {ex['question']}\nSQL: {normalize_sql(ex['query'])}")
    return "\n\n".join(parts) + "\n\n"


def build_user_prompt(schema: str, question: str, examples: list[dict] | None = None) -> str:
    return (f"Database schema:\n{schema}\n\n"
            f"{format_examples(examples)}"
            f"Question: {question}\n\n"
            "Write one SQLite query that answers the question. Return only the SQL.")


def build_messages(schema: str, question: str, examples: list[dict] | None = None,
                   answer: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(schema, question, examples)}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": normalize_sql(answer)})
    return msgs


# --------------------------------------------------------------------------
_QUOTED = re.compile(r"('(?:[^']|'')*'|\"(?:[^\"]|\"\")*\")")


def normalize_sql(sql: str) -> str:
    """Collapse whitespace outside string literals and drop the trailing semicolon."""
    sql = sql.strip().rstrip(";").strip()
    out = []
    for i, part in enumerate(_QUOTED.split(sql)):
        out.append(part if i % 2 == 1 else re.sub(r"\s+", " ", part))
    return "".join(out).strip()


_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.S | re.I)
_START = re.compile(r"\b(WITH|SELECT)\b", re.I)


def _cut_at_semicolon(sql: str) -> str:
    in_s = in_d = False
    for i, ch in enumerate(sql):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == ";" and not in_s and not in_d:
            return sql[:i]
    return sql


def extract_sql(text: str) -> str:
    """Pull the first SQL statement out of a model response."""
    if text is None:
        return ""
    t = text.strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    elif t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith(("sql", "sqlite")):
            t = t.split("\n", 1)[1] if "\n" in t else ""
    t = re.sub(r"^\s*(SQL|SQLite|Answer|Query)\s*:\s*", "", t, flags=re.I)
    m = _START.search(t)
    if m and m.start() > 0 and not t[: m.start()].strip().lower().startswith(("with", "select")):
        t = t[m.start():]
    t = _cut_at_semicolon(t)
    # stop at an obvious explanation paragraph
    t = re.split(r"\n\s*\n(?=[A-Z][a-z])", t)[0]
    return normalize_sql(t)
