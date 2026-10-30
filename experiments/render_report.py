"""Render the comparison tables in experiments/README.md (between the RESULTS markers) from
results/comparison/metrics.json, which collects the metric files written by experiments/compare_models.py."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"finetuned-1.5b": "Qwen2.5-Coder-1.5B + QLoRA (ours)", "qwen2.5-coder-0.5b": "Qwen2.5-Coder-0.5B",
         "qwen2.5-coder-1.5b": "Qwen2.5-Coder-1.5B (base)", "deepseek-coder-1.3b": "DeepSeek-Coder-1.3B",
         "qwen3-1.7b": "Qwen3-1.7B (no thinking)", "llama-3.2-3b": "Llama-3.2-3B", "qwen2.5-coder-3b": "Qwen2.5-Coder-3B",
         "qwen3-4b": "Qwen3-4B-Instruct-2507", "qwen2.5-coder-7b-awq": "Qwen2.5-Coder-7B (AWQ 4-bit)"}
PARAMS = {"finetuned-1.5b": 1.5, "qwen2.5-coder-0.5b": 0.5, "qwen2.5-coder-1.5b": 1.5, "deepseek-coder-1.3b": 1.3,
          "qwen3-1.7b": 1.7, "llama-3.2-3b": 3.2, "qwen2.5-coder-3b": 3.1, "qwen3-4b": 4.0, "qwen2.5-coder-7b-awq": 7.6}


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    rows = json.loads((ROOT / "results" / "comparison" / "metrics.json").read_text())
    get = {(r["model"], r["dataset"], r["shots"], r["repair"]): r for r in rows}
    models = [m for m in NAMES if any(k[0] == m for k in get)]
    L = ["| Model | Params | Spider dev EX (95% CI) | BIRD dev EX (95% CI) | Quick-commerce EX | Valid SQL (Spider / BIRD) |",
         "|---|---|---|---|---|---|"]

    def cell(m, d):
        r = get.get((m, d, "zero", False))
        if not r:
            return "-"
        lo, hi = wilson(r["execution_accuracy"], r["n"])
        return f"**{100 * r['execution_accuracy']:.1f}%** ({100 * lo:.0f}-{100 * hi:.0f})"

    for m in sorted(models, key=lambda m: -(get.get((m, "spider_dev", "zero", False)) or {"execution_accuracy": 0})["execution_accuracy"]):
        sp, bi = get.get((m, "spider_dev", "zero", False)), get.get((m, "bird_dev", "zero", False))
        qc = get.get((m, "quickcommerce", "zero", False))
        valid = f"{100 * sp['valid_sql_rate']:.0f}% / {100 * bi['valid_sql_rate']:.0f}%" if sp and bi else "-"
        L.append(f"| {NAMES[m]} | {PARAMS[m]}B | {cell(m, 'spider_dev')} | {cell(m, 'bird_dev')} | "
                 f"{'-' if not qc else f'{100 * qc['execution_accuracy']:.0f}%'} | {valid} |")
    L += ["", "| Technique (zero-shot unless stated) | Model | Spider dev | BIRD dev | Quick-commerce | Queries retried |",
          "|---|---|---|---|---|---|"]
    for m in ("qwen2.5-coder-1.5b", "finetuned-1.5b"):
        for shots, rep, label in (("zero", False, "zero-shot"), ("skeleton", False, "skeleton RAG (3 masked examples)"),
                                  ("zero", True, "+ execution-guided self-correction")):
            cells = []
            for d in ("spider_dev", "bird_dev", "quickcommerce"):
                r = get.get((m, d, shots, rep))
                cells.append("-" if not r else f"{100 * r['execution_accuracy']:.1f}%")
            r = get.get((m, "spider_dev", shots, rep))
            retried = "-" if not rep or not r else f"{r.get('retried', 0)} / {r['n']}"
            L.append(f"| {label} | {NAMES[m]} | {' | '.join(cells)} | {retried} |")
    readme = ROOT / "experiments" / "README.md"
    text = readme.read_text()
    a, b = text.index("<!-- RESULTS:START -->"), text.index("<!-- RESULTS:END -->")
    readme.write_text(text[: a + len("<!-- RESULTS:START -->")] + "\n" + "\n".join(L) + "\n" + text[b:])
    print("\n".join(L))


if __name__ == "__main__":
    main()
