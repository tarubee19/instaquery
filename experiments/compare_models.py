"""Model comparison study (run on a Colab T4 by notebooks/InstaQuery_Model_Comparison.ipynb).

Questions this answers, each with the same prompt, decoder (greedy), evaluator and data:

1. Is Qwen2.5-Coder-1.5B the right base model? -> zero-shot accuracy and speed of 8 open models
   from 0.5B to 7B (Qwen2.5-Coder 0.5B/1.5B/3B/7B-AWQ, Qwen3 1.7B/4B, Llama-3.2-3B,
   DeepSeek-Coder-1.3B) against the QLoRA fine-tuned 1.5B.
2. Can the RAG failure (finding 3) be fixed? -> "skeleton RAG": retrieved examples with
   table/column names masked (src/instaquery/skeleton.py).
3. Does execution-guided self-correction help? -> one repair round for queries that fail to run.
4. Does it hold on harder, real-world data? -> 500-question stratified sample of BIRD dev.

Every run is cached under OUT/comparison/, so a Colab disconnect only loses the current run.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from instaquery.bird_data import bird_db_path, bird_match, download_bird_dev, load_bird_dev
from instaquery.prompts import SYSTEM_PROMPT, build_messages, extract_sql
from instaquery.repair import self_correct
from instaquery.schema import serialize_schema
from instaquery.skeleton import format_skeleton_examples
from instaquery.sql_eval import categorize_error, exec_match

QWEN3_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

# name -> (HF repo or local path, params in billions, extra request body)
MODELS = {
    "qwen2.5-coder-0.5b": ("Qwen/Qwen2.5-Coder-0.5B-Instruct", 0.5, {}),
    "qwen2.5-coder-1.5b": ("Qwen/Qwen2.5-Coder-1.5B-Instruct", 1.5, {}),
    "deepseek-coder-1.3b": ("deepseek-ai/deepseek-coder-1.3b-instruct", 1.3, {}),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", 1.7, QWEN3_NO_THINK),
    "llama-3.2-3b": ("unsloth/Llama-3.2-3B-Instruct", 3.2, {}),
    "qwen2.5-coder-3b": ("Qwen/Qwen2.5-Coder-3B-Instruct", 3.1, {}),
    "qwen3-4b": ("Qwen/Qwen3-4B-Instruct-2507", 4.0, {}),
    "qwen2.5-coder-7b-awq": ("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ", 7.6, {}),
}


# ----------------------------------------------------------------------------- helpers
def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Continuity-corrected McNemar test on paired correctness (a = reference, b = candidate)."""
    fixed = sum((not x) and y for x, y in zip(a, b))
    broke = sum(x and (not y) for x, y in zip(a, b))
    n = fixed + broke
    if n == 0:
        return {"fixed": 0, "broke": 0, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(fixed - broke) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2))  # survival of chi2 with 1 dof
    return {"fixed": fixed, "broke": broke, "chi2": round(chi2, 2), "p_value": p}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


class Study:
    def __init__(self, ws, out_dir: Path, bird_dir: Path, bird_n: int = 500):
        self.ws = ws
        self.out = Path(out_dir) / "comparison"
        (self.out / "metrics").mkdir(parents=True, exist_ok=True)
        (self.out / "predictions").mkdir(parents=True, exist_ok=True)
        self.bird_dir, self.bird_n = Path(bird_dir), bird_n
        self.bird = None
        self._gold_cache: dict = {}

    # ------------------------------------------------------------------ data
    def examples(self, dataset: str) -> list[dict]:
        if dataset == "spider_dev":
            return self.ws.dev
        if dataset == "quickcommerce":
            return self.ws.domain
        if dataset == "bird_dev":
            if self.bird is None:
                self.bird_root = download_bird_dev(self.bird_dir)
                self.bird = load_bird_dev(self.bird_root, self.bird_n)
                print(f"[bird] {len(self.bird)} questions from {self.bird_root}")
            return self.bird
        raise ValueError(dataset)

    def db_file(self, dataset: str, db_id: str) -> Path:
        return bird_db_path(self.bird_root, db_id) if dataset == "bird_dev" else self.ws.db_lookup(db_id)

    def messages(self, dataset: str, shots: str, tokenizer_len) -> list[list[dict]]:
        exs = self.examples(dataset)
        if shots == "skeleton":
            r = self.ws.build_retriever()
            shot_lists = r.retrieve_many([e["question"] for e in exs], self.ws.cfg.n_shots,
                                         exclude_db=[e["db_id"] for e in exs])
        else:
            shot_lists = [None] * len(exs)
        out = []
        for ex, sh in zip(exs, shot_lists):
            db = str(self.db_file(dataset, ex["db_id"]))
            schema = serialize_schema(db, self.ws.cfg.sample_values_per_column)
            if tokenizer_len(schema) > 3000:   # very wide schemas: drop sample values
                schema = serialize_schema(db, 0)
            msgs = build_messages(schema, ex["question"], None)
            if sh:
                user = msgs[1]["content"]
                k = user.index("Question:")
                msgs[1]["content"] = user[:k] + format_skeleton_examples(sh) + user[k:]
            out.append(msgs)
        return out

    # ------------------------------------------------------------- one run
    def run(self, gen, model: str, dataset: str, shots: str = "zero", repair: bool = False,
            extra_body: dict | None = None, params_b: float | None = None) -> dict:
        tag = f"{dataset}__{model}__{shots}{'__repair' if repair else ''}"
        mfile = self.out / "metrics" / f"{tag}.json"
        if mfile.exists():
            print("[cached]", tag)
            return json.loads(mfile.read_text())
        exs = self.examples(dataset)
        gen.extra_body = {"repetition_penalty": 1.0, "top_p": 1.0, **(extra_body or {})}
        msgs = self.messages(dataset, shots, lambda s: len(s) // 3)
        t0 = time.time()
        outs = gen.generate(msgs, desc=tag)
        gen_s = time.time() - t0
        preds = [extract_sql(o.text) for o in outs]
        info = [{"retried": False, "first_error": ""} for _ in preds]
        repair_s = 0.0
        if repair:
            t1 = time.time()
            preds, info = self_correct(gen, msgs, preds, [self.db_file(dataset, e["db_id"]) for e in exs],
                                       extra_body=extra_body)
            repair_s = time.time() - t1
        records = []
        for i, (ex, p) in enumerate(zip(exs, preds)):
            db = self.db_file(dataset, ex["db_id"])
            if dataset == "bird_dev":
                ok, err = bird_match(db, p, ex["query"], 30.0, self._gold_cache)
                if ok is None:
                    continue
                cat = "correct" if ok else ("wrong result" if err == "wrong result" else _cat(err))
                rec = {"correct": bool(ok), "pred_executable": err in ("", "wrong result"), "category": cat}
            else:
                r = exec_match(db, p, ex["query"], self.ws.cfg.sql_timeout_s, keep_distinct=(dataset != "spider_dev"))
                rec = {"correct": bool(r.correct), "pred_executable": bool(r.pred_executable),
                       "category": categorize_error(r)}
            rec.update({"i": i, "db_id": ex["db_id"], "hardness": ex.get("hardness", "n/a"),
                        "question": ex["question"], "gold": ex["query"], "pred": p, **info[i]})
            records.append(rec)
        n = len(records)
        k = sum(r["correct"] for r in records)
        by = {}
        for r in records:
            by.setdefault(r["hardness"], []).append(r["correct"])
        cats = {}
        for r in records:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        m = {"model": model, "dataset": dataset, "shots": shots, "repair": repair, "n": n,
             "params_b": params_b,
             "execution_accuracy": k / n, "ci95": wilson(k, n),
             "valid_sql_rate": sum(r["pred_executable"] for r in records) / n,
             "by_difficulty": {h: {"ex": sum(v) / len(v), "n": len(v)} for h, v in by.items()},
             "error_breakdown": cats, "generation_time_s": gen_s, "repair_time_s": repair_s,
             "retried": sum(r["retried"] for r in records),
             "avg_prompt_tokens": sum(o.prompt_tokens for o in outs) / len(outs)}
        with open(self.out / "predictions" / f"{tag}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        mfile.write_text(json.dumps(m, indent=2))
        print(f"[{tag}] EX={m['execution_accuracy']:.3f} valid={m['valid_sql_rate']:.3f} n={n} ({gen_s:.0f}s)")
        return m

    # ------------------------------------------------------------- summary
    def summarize(self, reference: str = "qwen2.5-coder-1.5b") -> dict:
        rows = [json.loads(p.read_text()) for p in sorted((self.out / "metrics").glob("*.json"))]
        preds = {}
        for p in (self.out / "predictions").glob("*.jsonl"):
            recs = [json.loads(line) for line in p.read_text().splitlines()]
            preds[p.stem] = {r["i"]: r["correct"] for r in recs}
        for m in rows:
            tag = f"{m['dataset']}__{m['model']}__{m['shots']}{'__repair' if m['repair'] else ''}"
            ref = f"{m['dataset']}__{reference}__zero"
            if tag != ref and ref in preds and tag in preds:
                common = sorted(set(preds[ref]) & set(preds[tag]))
                m["vs_reference"] = mcnemar([preds[ref][i] for i in common], [preds[tag][i] for i in common])
        (self.out / "summary.json").write_text(json.dumps(rows, indent=2))
        return {"rows": rows}


def _cat(err: str) -> str:
    e = err.lower()
    for key, name in (("no such column", "unknown column"), ("no such table", "unknown table"),
                      ("syntax error", "syntax error"), ("timeout", "timeout"), ("ambiguous", "ambiguous column")):
        if key in e:
            return name
    return "other execution error"


def seed_everything(seed: int = 0):
    random.seed(seed)
