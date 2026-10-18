"""Thin wrapper around the official Spider evaluator (taoyds/test-suite-sql-eval).

Used for two things:
  * Spider's official difficulty label (easy / medium / hard / extra) per question
  * the official Exact-Set-Match (EM) score, reported next to execution accuracy
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/taoyds/test-suite-sql-eval"


def ensure_repo(dest: Path) -> Path:
    dest = Path(dest)
    if not (dest / "evaluation.py").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, str(dest)], check=True)
    # The official parser calls nltk.word_tokenize, which normally needs the 'punkt' sentence model.
    # A SQL query is a single "sentence", so we patch it to word_tokenize(s, preserve_line=True),
    # which gives the same tokens without downloading punkt (blocked behind some proxies).
    patch = dest / "_nltk_patch.py"
    patch.write_text(
        "import nltk, process_sql\n"
        "process_sql.word_tokenize = lambda s: nltk.word_tokenize(s, preserve_line=True)\n")
    return dest


def _patch_tokenizer():
    import nltk
    import process_sql  # type: ignore
    process_sql.word_tokenize = lambda s: nltk.word_tokenize(s, preserve_line=True)


def add_hardness(examples: list[dict], repo: Path, db_lookup) -> list[dict]:
    """Adds ex['hardness'] using the official parser. Falls back to 'n/a'."""
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        _patch_tokenizer()
        from process_sql import Schema, get_schema, get_sql  # type: ignore
        from evaluation import Evaluator  # type: ignore
    except Exception as e:
        print("[official-eval] could not import official evaluator:", e)
        for ex in examples:
            ex.setdefault("hardness", "n/a")
        return examples
    ev, cache = Evaluator(), {}
    for ex in examples:
        try:
            db = str(db_lookup(ex["db_id"]))
            if db not in cache:
                cache[db] = Schema(get_schema(db))
            ex["hardness"] = ev.eval_hardness(get_sql(cache[db], ex["query"]))
        except Exception:
            ex["hardness"] = "n/a"
    return examples


def exact_match(gold: list[dict], preds: list[str], repo: Path, db_dir: Path, tables_json: Path,
                work: Path) -> dict | None:
    """Runs the official script with --etype match; returns {'easy':..,'all':..} or None."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    gold_f, pred_f = work / "gold.txt", work / "pred.txt"
    gold_f.write_text("\n".join(f"{g['query']}\t{g['db_id']}" for g in gold) + "\n")
    pred_f.write_text("\n".join((p.replace("\n", " ").replace("\t", " ") or "SELECT 1") for p in preds) + "\n")
    try:
        argv = ["evaluation.py", "--gold", str(gold_f.resolve()), "--pred", str(pred_f.resolve()),
                "--etype", "match", "--db", str(Path(db_dir).resolve()), "--table", str(Path(tables_json).resolve())]
        runner = ("import sys, runpy, _nltk_patch; sys.argv = " + repr(argv) +
                  "; runpy.run_path('evaluation.py', run_name='__main__')")
        out = subprocess.run([sys.executable, "-c", runner], cwd=str(repo), capture_output=True, text=True,
                             timeout=1800)
        text = out.stdout
        (work / "official_match_output.txt").write_text(text + "\n" + out.stderr[-5000:])
        m = re.search(r"^\s*exact match\s+([\d.\s]+)$", text, re.M)
        if not m:
            return None
        vals = [float(x) for x in m.group(1).split()]
        keys = ["easy", "medium", "hard", "extra", "all"]
        return dict(zip(keys, vals[:5]))
    except Exception as e:  # pragma: no cover
        print("[official-eval] exact match failed:", e)
        return None
