"""Execution accuracy (EX) for Text-to-SQL.

A prediction is correct if, executed on the database, it returns the same
result as the gold query. The comparison (`result_eq`) follows the official
Spider test-suite evaluator (taoyds/test-suite-sql-eval, Apache-2.0):
bag semantics, row order only matters when the gold query has ORDER BY, and
columns may be permuted. DISTINCT is removed from both queries before
execution, as the official script does by default.

Unlike the official script, queries here have a hard timeout (via SQLite's
progress handler) so a runaway cartesian join cannot hang the evaluation.
"""

from __future__ import annotations

import random
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

from .schema import connect_readonly

MAX_ROWS = 100_000


# ---------------------------------------------------------------- execution
def execute_sql(db_file: str | Path, sql: str, timeout_s: float = 15.0):
    """Returns ("ok", rows) or ("error", message)."""
    if not sql or not sql.strip():
        return "error", "empty query"
    conn = connect_readonly(db_file)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(MAX_ROWS + 1)
        if len(rows) > MAX_ROWS:
            return "error", f"result too large (> {MAX_ROWS} rows)"
        return "ok", rows
    except sqlite3.OperationalError as e:
        msg = str(e)
        return "error", ("timeout" if "interrupted" in msg else msg)
    except Exception as e:  # sqlite3.Warning for multiple statements, etc.
        return "error", f"{type(e).__name__}: {e}"
    finally:
        conn.close()


# ---------------------------------------------------- official comparison logic
def _unorder_row(row):
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def _quick_rej(r1, r2, order_matters):
    s1 = [_unorder_row(r) for r in r1]
    s2 = [_unorder_row(r) for r in r2]
    return s1 == s2 if order_matters else set(s1) == set(s2)


def _multiset_eq(l1, l2):
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] += 1
    for e in l2:
        d[e] -= 1
        if d[e] < 0:
            return False
    return True


def _constraint_permutations(tab1_sets_by_columns, result2, rng):
    num_cols = len(result2[0])
    perm_constraints = [set(range(num_cols)) for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)
    for _ in range(20):
        row = rng.choice(result2)
        for c1 in range(num_cols):
            for c2 in set(perm_constraints[c1]):
                if row[c2] not in tab1_sets_by_columns[c1]:
                    perm_constraints[c1].remove(c2)
    return product(*perm_constraints)


def result_eq(result1, result2, order_matters: bool) -> bool:
    if len(result1) == 0 and len(result2) == 0:
        return True
    if len(result1) != len(result2):
        return False
    num_cols = len(result1[0])
    if len(result2[0]) != num_cols:
        return False
    if not _quick_rej(result1, result2, order_matters):
        return False
    tab1_sets = [{row[i] for row in result1} for i in range(num_cols)]
    rng = random.Random(0)
    for perm in _constraint_permutations(tab1_sets, result2, rng):
        if len(perm) != len(set(perm)):
            continue
        r2p = result2 if num_cols == 1 else [tuple(e[i] for i in perm) for e in result2]
        if order_matters:
            if result1 == r2p:
                return True
        elif set(result1) == set(r2p) and _multiset_eq(result1, r2p):
            return True
    return False


_DISTINCT = re.compile(r"\bdistinct\b", re.I)


def remove_distinct(sql: str) -> str:
    try:
        import sqlparse
        toks = [t.value for t in sqlparse.parse(sql)[0].flatten()]
        return "".join(t for t in toks if t.lower() != "distinct")
    except Exception:
        return _DISTINCT.sub("", sql)


def _postprocess(q: str) -> str:
    return q.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")


# ------------------------------------------------------------------ scoring
@dataclass
class ExecResult:
    correct: bool
    pred_executable: bool
    error: str = ""
    gold_rows: int = 0
    pred_rows: int = 0


def exec_match(db_file: str | Path, pred_sql: str, gold_sql: str, timeout_s: float = 15.0,
               keep_distinct: bool = False) -> ExecResult:
    p, g = _postprocess(pred_sql), _postprocess(gold_sql)
    if not keep_distinct:
        p, g = remove_distinct(p), remove_distinct(g)
    order_matters = "order by" in g.lower()
    g_flag, g_res = execute_sql(db_file, g, timeout_s)
    if g_flag != "ok":
        # gold must execute; treat as evaluation-data problem
        return ExecResult(False, False, f"GOLD FAILED: {g_res}")
    p_flag, p_res = execute_sql(db_file, p, timeout_s)
    if p_flag != "ok":
        return ExecResult(False, False, str(p_res), len(g_res), 0)
    ok = result_eq(g_res, p_res, order_matters)
    return ExecResult(ok, True, "" if ok else "wrong result", len(g_res), len(p_res))


# -------------------------------------------------------- error categories
def categorize_error(r: ExecResult) -> str:
    if r.correct:
        return "correct"
    e = (r.error or "").lower()
    if e.startswith("gold failed"):
        return "gold query failed"
    if not r.pred_executable:
        if "no such column" in e:
            return "unknown column"
        if "no such table" in e:
            return "unknown table"
        if "syntax error" in e or "incomplete input" in e or "unrecognized token" in e:
            return "syntax error"
        if "timeout" in e:
            return "timeout"
        if "ambiguous" in e:
            return "ambiguous column"
        if "empty" in e:
            return "empty output"
        return "other execution error"
    if r.pred_rows != r.gold_rows:
        return "wrong result (row count)"
    return "wrong result (values)"


@dataclass
class Scored:
    records: list = field(default_factory=list)

    @property
    def ex(self) -> float:
        return sum(r["correct"] for r in self.records) / max(1, len(self.records))

    @property
    def exec_rate(self) -> float:
        return sum(r["pred_executable"] for r in self.records) / max(1, len(self.records))

    def by(self, key: str) -> dict:
        groups = defaultdict(list)
        for r in self.records:
            groups[r.get(key, "n/a")].append(r["correct"])
        return {k: (sum(v) / len(v), len(v)) for k, v in groups.items()}

    def error_counts(self) -> dict:
        return dict(Counter(r["category"] for r in self.records))


def score(examples: list[dict], preds: list[str], db_lookup, timeout_s: float = 15.0,
          extra: list[dict] | None = None, progress: bool = True, keep_distinct: bool = False) -> Scored:
    """examples: dicts with question/query/db_id (+ optional hardness). db_lookup(db_id)->path."""
    assert len(examples) == len(preds)
    it = range(len(examples))
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="executing", leave=False)
        except ImportError:
            pass
    out = Scored()
    for i in it:
        ex, pred = examples[i], preds[i]
        r = exec_match(db_lookup(ex["db_id"]), pred, ex["query"], timeout_s, keep_distinct)
        rec = {"idx": i, "db_id": ex["db_id"], "question": ex["question"], "gold": ex["query"],
               "pred": pred, "correct": bool(r.correct), "pred_executable": bool(r.pred_executable),
               "error": r.error, "category": categorize_error(r),
               "hardness": ex.get("hardness", ex.get("difficulty", "n/a"))}
        if extra:
            rec.update(extra[i])
        out.records.append(rec)
    return out
