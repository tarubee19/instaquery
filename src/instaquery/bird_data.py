"""BIRD dev (Li et al., NeurIPS 2023): a harder, real-world Text-to-SQL benchmark.

Compared with Spider, BIRD databases are large and messy (95 databases, 33 GB in total,
dirty values, domain knowledge needed), so it is a much better test of whether a model
works on business data. Each question comes with an "evidence" hint written by an
annotator; following the official "with knowledge" setting we append it to the question.

Scoring follows the official BIRD evaluator: a prediction is correct when
set(pred_rows) == set(gold_rows).
"""

from __future__ import annotations

import json
import random
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

from .sql_eval import execute_sql

BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"


def download_bird_dev(dest: Path) -> Path:
    dest = Path(dest)
    root = next((p.parent for p in dest.rglob("dev.json") if (p.parent / "dev_databases").is_dir()), None)
    if root:
        return root
    dest.mkdir(parents=True, exist_ok=True)
    z = dest / "dev.zip"
    if not z.exists():
        subprocess.run(["wget", "-q", "-O", str(z), BIRD_DEV_URL], check=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(dest)
    for inner in dest.rglob("dev_databases.zip"):
        with zipfile.ZipFile(inner) as f:
            f.extractall(inner.parent)
    root = next(p.parent for p in dest.rglob("dev.json") if (p.parent / "dev_databases").is_dir())
    return root


def load_bird_dev(root: Path, n: int | None = 500, seed: int = 42) -> list[dict]:
    """Stratified sample by difficulty (same proportions as the full dev set)."""
    data = json.loads((Path(root) / "dev.json").read_text())
    rows = []
    for i, d in enumerate(data):
        q = d["question"].strip()
        if d.get("evidence"):
            q += f" (Hint: {d['evidence'].strip()})"
        rows.append({"question": q, "query": d["SQL"], "db_id": d["db_id"], "idx": i,
                     "hardness": d.get("difficulty", "n/a"), "split": "bird_dev"})
    if n is None or n >= len(rows):
        return rows
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["hardness"]].append(r)
    out = []
    for k, v in sorted(by.items()):
        take = round(n * len(v) / len(rows))
        out += rng.sample(v, min(take, len(v)))
    return sorted(out, key=lambda r: r["idx"])


def bird_db_path(root: Path, db_id: str) -> Path:
    return Path(root) / "dev_databases" / db_id / f"{db_id}.sqlite"


def bird_match(db_file, pred: str, gold: str, timeout_s: float = 30.0, gold_cache: dict | None = None):
    key = (str(db_file), gold)
    if gold_cache is not None and key in gold_cache:
        g_flag, g_res = gold_cache[key]
    else:
        g_flag, g_res = execute_sql(db_file, gold, timeout_s)
        if gold_cache is not None:
            gold_cache[key] = (g_flag, g_res)
    if g_flag != "ok":
        return None, "GOLD FAILED"
    p_flag, p_res = execute_sql(db_file, pred, timeout_s)
    if p_flag != "ok":
        return False, str(p_res)
    ok = set(map(tuple, p_res)) == set(map(tuple, g_res))
    return ok, "" if ok else "wrong result"
