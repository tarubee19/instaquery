"""Builds notebooks/InstaQuery_Colab.ipynb from the source tree (single self-contained notebook)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_FILES = ["__init__.py", "config.py", "spider_data.py", "schema.py", "prompts.py", "retrieval.py",
             "sql_eval.py", "official_eval.py", "generation.py", "train.py", "serving.py", "safety.py",
             "domain_db.py", "pipeline.py", "report.py"]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text, hidden=False):
    meta = {"cellView": "form"} if hidden else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# InstaQuery: fine-tuning a small open LLM for Text-to-SQL

End-to-end project on **one free Colab T4 GPU**:

1. **Data**: Spider (8.6K train questions, 1,034 dev questions on 20 unseen databases) + a synthetic quick-commerce database with 50 business questions
2. **Baselines**: base model zero-shot, fixed few-shot and **RAG** few-shot (dense retrieval of similar examples)
3. **QLoRA fine-tuning**: 4-bit NF4 base + LoRA adapters (Hugging Face transformers, PEFT, bitsandbytes)
4. **Evaluation**: execution accuracy (official Spider comparison logic), difficulty breakdown, error analysis
5. **Quantization**: 4-bit NF4 accuracy vs memory
6. **Serving**: vLLM OpenAI-compatible server, throughput/latency benchmark vs Hugging Face `generate`
7. **Report + demo**: charts, README with results, CV bullets, Gradio app

**How to run:** Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**.
`RUN_MODE="quick"` is a ~20-minute smoke test; `"full"` is the complete run (~2.5–3 h).
Every stage caches its outputs, so after a disconnect just **Run all** again and it resumes.
""")

md("## 0. Settings")
code('''
#@title Settings
RUN_MODE = "full"          #@param ["quick", "full"]
USE_DRIVE = True           #@param {type:"boolean"}
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  #@param {type:"string"}
DRIVE_DIR = "/content/drive/MyDrive/InstaQuery"  #@param {type:"string"}
''')

md("## 1. Environment")
code('''
import os, subprocess, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
T_START = time.time()
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print("GPU:", gpu or "NONE - switch the runtime to a T4 GPU before running")
assert gpu, "No GPU found"
''')
code('''
%pip install -q -U transformers peft accelerate bitsandbytes datasets sentence-transformers sqlglot sqlparse nltk gdown huggingface_hub httpx nest_asyncio
!pip uninstall -y -q torchao   # Colab's preinstalled torchao is too old for the new PEFT and is not needed here
import transformers, peft, torch
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
''')
code('''
import os
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_DIR = f"{DRIVE_DIR}/outputs_{RUN_MODE}"
else:
    OUTPUT_DIR = f"/content/instaquery_outputs_{RUN_MODE}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Outputs ->", OUTPUT_DIR)
''')

md("## 2. Project source code\nThe same files live in the GitHub repo under `src/instaquery/`.")
code("!mkdir -p /content/instaquery/src/instaquery /content/instaquery/data /content/instaquery/app "
     "/content/instaquery/templates /content/instaquery/tests")
for f in PKG_FILES:
    code(f"%%writefile /content/instaquery/src/instaquery/{f}\n" + (ROOT / "src" / "instaquery" / f).read_text())
code("%%writefile /content/instaquery/data/quickcommerce_eval.json\n" + (ROOT / "data" / "quickcommerce_eval.json").read_text())
code("%%writefile /content/instaquery/templates/README.md.tmpl\n" + (ROOT / "templates" / "README.md.tmpl").read_text())
code("%%writefile /content/instaquery/app/gradio_app.py\n" + (ROOT / "app" / "gradio_app.py").read_text())
code("%%writefile /content/instaquery/tests/test_core.py\n" + (ROOT / "tests" / "test_core.py").read_text())
code('''
!cd /content/instaquery && python -m pytest -q tests 2>&1 | tail -3
''')

md("## 3. Configuration and data")
code('''
import json, sys, importlib
from pathlib import Path
sys.path.insert(0, "/content/instaquery/src")
import instaquery
from instaquery.config import load_config
from instaquery.pipeline import Workspace
cfg = load_config(run_mode=RUN_MODE, output_dir=OUTPUT_DIR, base_model=BASE_MODEL)
cfg.save()
ws = Workspace(cfg, Path("/content/instaquery/data/quickcommerce_eval.json")).prepare_data()
''')
code('''
from instaquery.official_eval import ensure_repo, add_hardness
OFFICIAL = ensure_repo(Path(cfg.work_dir) / "test-suite-sql-eval")
add_hardness(ws.dev, OFFICIAL, ws.db_lookup)
from collections import Counter
print("Spider dev difficulty:", dict(Counter(e["hardness"] for e in ws.dev)))
print("Quick-commerce difficulty:", dict(Counter(e["hardness"] for e in ws.domain)))
''')
code('''
# Dataset statistics + one full prompt
import numpy as np, matplotlib.pyplot as plt
tok = ws.load_tokenizer()
sample = ws.dev[:300] + ws.train[:300]
lens = [len(tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True))["input_ids"])
        for m in ws.messages_for(sample, "zero")]
stats = {"train_questions": len(ws.train), "train_databases": len({e["db_id"] for e in ws.train}),
         "dev_questions": len(ws.dev), "dev_databases": len({e["db_id"] for e in ws.dev}),
         "quickcommerce_questions": len(ws.domain),
         "prompt_tokens_mean": float(np.mean(lens)), "prompt_tokens_p95": float(np.percentile(lens, 95))}
cfg.out("dataset_stats.json").write_text(json.dumps(stats, indent=2)); print(stats)
fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#fcfcfb")
ax.hist(lens, bins=40, color="#2a78d6", edgecolor="#fcfcfb", linewidth=1)
ax.set_xlabel("Prompt length (tokens)"); ax.set_ylabel("Questions"); ax.set_title("Prompt length distribution", loc="left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(cfg.out("assets", "prompt_lengths.png"), dpi=150); plt.show()
print(tok.apply_chat_template(ws.messages_for(ws.dev[:1], "rag")[0], tokenize=False, add_generation_prompt=True))
''')

md("## 4. vLLM (isolated environment)\nvLLM is installed into its own virtual env so it cannot clash with the training stack. "
   "If it fails, everything falls back to Hugging Face generation automatically.")
code('''
from instaquery.serving import install_vllm
VLLM_OK = install_vllm(cfg.vllm_version, log_file=cfg.out("logs", "vllm_install.log"))
ws.backend = "vllm" if VLLM_OK else "hf"
print("generation backend:", ws.backend)
''')

md("## 5. Baselines with the base model\nZero-shot, 3 fixed examples, and RAG (3 most similar training questions, never from the same database).")
code('''
base_plan = [("base_zeroshot", "spider_dev", "zero"), ("base_fewshot", "spider_dev", "fixed"),
             ("base_rag", "spider_dev", "rag"), ("base_zeroshot", "quickcommerce", "zero"),
             ("base_rag", "quickcommerce", "rag")]
base_results = ws.run_plan(cfg.base_model, base_plan, tag="base")
for m in base_results:
    print(f"{m['dataset']:14s} {m['variant']:15s} EX={100*m['execution_accuracy']:.1f}%")
''')

md("## 6. QLoRA fine-tuning")
code('''
from instaquery.train import train_qlora, merge_adapter
stats_file = cfg.adapter_dir / "train_stats.json"
if stats_file.exists():
    train_stats = json.loads(stats_file.read_text()); print("[train] cached adapter found")
else:
    train_stats = train_qlora(cfg, ws.train, ws.schema_for, cfg.adapter_dir, Path(cfg.output_dir) / "checkpoints")
print(json.dumps(train_stats, indent=2))
''')
code('''
from instaquery import report
report.chart_training_loss(json.loads((cfg.adapter_dir / "log_history.json").read_text()), cfg.out("assets", "training_loss.png"))
from IPython.display import Image, display
display(Image(str(cfg.out("assets", "training_loss.png"))))
''')
code('''
merged = merge_adapter(cfg.base_model, cfg.adapter_dir, cfg.merged_dir)
''')

md("## 7. Evaluate the fine-tuned model + vLLM serving benchmark")
code('''
from instaquery.generation import OpenAIGenerator
bench_file = Path(cfg.output_dir) / "benchmark.json"
bench = json.loads(bench_file.read_text()) if bench_file.exists() else {}
def _bench(gen):
    if isinstance(gen, OpenAIGenerator) and "vllm" not in bench:
        bench["vllm"] = ws.bench_vllm(gen)
ft_plan = [("finetuned", "spider_dev", "zero"), ("finetuned_rag", "spider_dev", "rag"),
           ("finetuned", "quickcommerce", "zero")]
ft_results = ws.run_plan(str(merged), ft_plan, tag="finetuned", after=None if "vllm" in bench else _bench)
if "hf_batched" not in bench:
    bench.update(ws.bench_hf(str(merged)))
bench_file.write_text(json.dumps(bench, indent=2))
for m in ft_results:
    print(f"{m['dataset']:14s} {m['variant']:15s} EX={100*m['execution_accuracy']:.1f}%")
print(json.dumps(bench, indent=2))
''')

md("## 8. 4-bit NF4 quantization: accuracy vs memory")
code('''
nf4_results = ws.run_plan(str(merged), [("finetuned_nf4", "spider_dev", "zero"), ("finetuned_nf4", "quickcommerce", "zero")],
                          tag="nf4", backend="hf", quant_4bit=True)
for m in nf4_results:
    print(f"{m['dataset']:14s} {m['variant']:15s} EX={100*m['execution_accuracy']:.1f}%  memory={m.get('model_memory_gb', 0):.2f} GB")
''')

md("## 9. Optional: paid API reference\nAdd an `OPENAI_API_KEY` in Colab **Secrets** (key icon on the left) to compare with a hosted model. Skipped otherwise.")
code('''
api_key = None
try:
    from google.colab import userdata
    api_key = userdata.get("OPENAI_API_KEY")
except Exception as e:
    print("No API key configured - skipping API reference.")
if api_key:
    api = OpenAIGenerator(cfg.api_base_url, cfg.api_model, api_key=api_key, concurrency=16)
    for d in ("spider_dev", "quickcommerce"):
        try:
            ws.run_variant(api, "api_zeroshot", d, "zero", extra_meta={"api_model": cfg.api_model})
        except Exception as e:
            print("API evaluation failed:", e)
''')

md("## 10. Official Spider exact-set-match (reference metric)")
code('''
em_file = Path(cfg.output_dir) / "official_exact_match.json"
official_em = json.loads(em_file.read_text()) if em_file.exists() else {}
from instaquery.official_eval import exact_match
for v in ("base_zeroshot", "finetuned"):
    p = Path(cfg.output_dir) / "predictions" / "spider_dev" / f"{v}.jsonl"
    if v in official_em or not p.exists():
        continue
    preds = [json.loads(l)["pred"] for l in p.read_text().splitlines()]
    res = exact_match(ws.dev, preds, OFFICIAL, Path(ws.spider_root) / "database", Path(ws.spider_root) / "tables.json",
                      Path(cfg.work_dir) / f"em_{v}")
    if res: official_em[v] = res
em_file.write_text(json.dumps(official_em, indent=2)); print(official_em)
''')

md("## 11. Report: tables, charts, README, CV bullets")
code('''
from IPython.display import Markdown, Image, display
importlib.reload(report)
metrics = report.load_metrics(cfg.output_dir)
assets = Path(cfg.output_dir) / "assets"
report.chart_ex_by_variant(metrics, assets / "ex_by_variant.png")
report.chart_by_difficulty(metrics, assets / "ex_by_difficulty.png")
report.chart_throughput(bench, assets / "throughput.png")
err_md = report.error_analysis(cfg.output_dir)
readme = report.build_readme(Path("/content/instaquery/templates/README.md.tmpl").read_text(), cfg, metrics,
                             train_stats, bench, err_md, official_em)
cv = report.cv_bullets(metrics, train_stats, bench, cfg.base_model.split("/")[-1])
(Path(cfg.output_dir) / "README.md").write_text(readme)
(Path(cfg.output_dir) / "cv_bullets.md").write_text(cv)
(Path(cfg.output_dir) / "summary_table.md").write_text(report.summary_table(metrics))
display(Markdown(report.summary_table(metrics)))
for f in ("ex_by_variant.png", "ex_by_difficulty.png", "throughput.png"):
    if (assets / f).exists(): display(Image(str(assets / f)))
display(Markdown(err_md))
display(Markdown("### CV bullets (real numbers from this run)\\n" + cv))
''')

md("## 12. Demo: ask the quick-commerce database in English")
code('''
%pip install -q gradio
import sys
sys.path.insert(0, "/content/instaquery/app")
from gradio_app import build_demo
from instaquery.generation import HFGenerator, load_hf_model
demo_model, demo_tok = load_hf_model(str(merged))
demo_gen = HFGenerator(demo_model, demo_tok, batch_size=1, max_new_tokens=256)
demo = build_demo(lambda msgs: demo_gen.generate_one_timed(msgs).text, str(cfg.domain_db_path))
demo.launch(share=False, inline=True, height=750)
''')

md("## 13. Package results")
code('''
import shutil
ws.timings["total_runtime_min"] = (time.time() - T_START) / 60
(Path(cfg.output_dir) / "timings.json").write_text(json.dumps(ws.timings, indent=2))
pkg = Path("/content/instaquery_results")
if pkg.exists(): shutil.rmtree(pkg)
shutil.copytree(cfg.output_dir, pkg, ignore=shutil.ignore_patterns("checkpoints", "*.safetensors", "*.bin", "optimizer.pt"))
shutil.copytree("/content/instaquery", pkg / "repo", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
shutil.copy(cfg.domain_db_path, pkg / "quickcommerce.sqlite")
zip_path = shutil.make_archive("/content/instaquery_results", "zip", pkg)
if USE_DRIVE: shutil.copy(zip_path, DRIVE_DIR)
print("Results zip:", zip_path, "| total runtime %.0f min" % ws.timings["total_runtime_min"])
print("ALL DONE")
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "notebooks" / "InstaQuery_Colab.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, len(cells), "cells", out.stat().st_size // 1024, "KB")
