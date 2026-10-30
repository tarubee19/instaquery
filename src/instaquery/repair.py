"""Execution-guided self-correction (one round).

If a generated query fails to execute (unknown column, syntax error, ...), the
SQLite error message is sent back to the model with its own query and it gets one
chance to fix it. Only queries that *fail to run* are retried, so the gold answer
is never consulted: this is a legitimate inference-time technique, not test leakage.
"""

from __future__ import annotations

from .prompts import extract_sql
from .sql_eval import execute_sql

REPAIR_PROMPT = ("That query failed when executed on the database with this SQLite error:\n{error}\n\n"
                 "Fix the query. Use only tables and columns that exist in the schema. Return only the SQL.")


def self_correct(gen, messages: list[list[dict]], preds: list[str], db_files: list, timeout_s: float = 15.0,
                 extra_body: dict | None = None):
    """Returns (new_preds, info). info[i] = {'retried': bool, 'first_error': str}."""
    retry_idx, retry_msgs, info = [], [], [{"retried": False, "first_error": ""} for _ in preds]
    for i, (msgs, pred, db) in enumerate(zip(messages, preds, db_files)):
        flag, res = execute_sql(db, pred, timeout_s)
        if flag == "ok":
            continue
        err = str(res)[:300]
        info[i] = {"retried": True, "first_error": err}
        retry_idx.append(i)
        retry_msgs.append(msgs + [{"role": "assistant", "content": pred or "(empty)"},
                                  {"role": "user", "content": REPAIR_PROMPT.format(error=err)}])
    new_preds = list(preds)
    if retry_msgs:
        outs = gen.generate(retry_msgs, desc="self-correction")
        for i, o in zip(retry_idx, outs):
            fixed = extract_sql(o.text)
            if fixed:
                new_preds[i] = fixed
    return new_preds, info
