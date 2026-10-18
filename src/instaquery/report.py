"""Collect metrics -> summary tables, charts, README results section and CV bullets."""

from __future__ import annotations

import json
from pathlib import Path

# validated categorical palette (light surface) - see README "charts"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"

LABELS = {
    "base_zeroshot": "Base, zero-shot",
    "base_fewshot": "Base, 3 fixed examples",
    "base_rag": "Base + RAG (3 retrieved examples)",
    "finetuned": "Fine-tuned (QLoRA)",
    "finetuned_rag": "Fine-tuned + RAG",
    "finetuned_nf4": "Fine-tuned, 4-bit NF4",
    "api_zeroshot": "Paid API model, zero-shot",
}


# one colour per variant family, identical in every chart
VARIANT_COLOR = {"base_zeroshot": MUTED, "base_fewshot": MUTED, "base_rag": AQUA, "finetuned": BLUE,
                 "finetuned_rag": BLUE, "finetuned_nf4": BLUE, "api_zeroshot": ORANGE}


def load_metrics(out_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((Path(out_dir) / "metrics").glob("*.json"))]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


DATASET_LABELS = {"spider_dev": "Spider dev (1,034)", "quickcommerce": "Quick-commerce (50)"}


def summary_table(metrics: list[dict]) -> str:
    rows = ["| Dataset | Variant | Execution accuracy | Valid SQL | Examples | Gen. time |",
            "|---|---|---|---|---|---|"]
    order = list(LABELS)
    dorder = ["spider_dev", "quickcommerce"]
    for m in sorted(metrics, key=lambda m: (dorder.index(m["dataset"]) if m["dataset"] in dorder else 9,
                                            order.index(m["variant"]) if m["variant"] in order else 99)):
        rows.append(f"| {DATASET_LABELS.get(m['dataset'], m['dataset'])} | {LABELS.get(m['variant'], m['variant'])} | "
                    f"**{100*m['execution_accuracy']:.1f}%** | {100*m['valid_sql_rate']:.1f}% | {m['n']} | "
                    f"{m['generation_time_s']:.0f}s |")
    return "\n".join(rows)


def chart_ex_by_variant(metrics: list[dict], path: Path):
    import matplotlib.pyplot as plt
    datasets = [d for d in ("spider_dev", "quickcommerce") if any(m["dataset"] == d for m in metrics)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.2), facecolor=SURFACE)
    axes = [axes] if len(datasets) == 1 else axes
    titles = {"spider_dev": "Spider dev (1,034 questions, 20 unseen databases)",
              "quickcommerce": "Quick-commerce DB (50 questions, unseen schema)"}
    for ax, d in zip(axes, datasets):
        ms = [m for m in metrics if m["dataset"] == d and m["variant"] in LABELS]
        ms.sort(key=lambda m: list(LABELS).index(m["variant"]))
        names = [LABELS[m["variant"]] for m in ms]
        vals = [100 * m["execution_accuracy"] for m in ms]
        cols = [VARIANT_COLOR.get(m["variant"], MUTED) for m in ms]
        y = range(len(ms))
        ax.barh(list(y), vals, color=cols, height=0.62, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v + 1, yi, f"{v:.1f}%", va="center", fontsize=9, color=INK)
        ax.set_yticks(list(y), names, fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Execution accuracy (%)", color=INK2, fontsize=9)
        ax.set_title(titles.get(d, d), fontsize=10, color=INK, loc="left")
        _style(ax)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.yaxis.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def chart_by_difficulty(metrics: list[dict], path: Path, dataset: str = "spider_dev",
                        variants=("base_zeroshot", "base_rag", "finetuned")):
    import matplotlib.pyplot as plt
    levels = ["easy", "medium", "hard", "extra"]
    ms = {m["variant"]: m for m in metrics if m["dataset"] == dataset}
    variants = [v for v in variants if v in ms]
    if not variants:
        return
    levels = [l for l in levels if any(l in ms[v]["by_difficulty"] for v in variants)]
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    w = 0.8 / len(variants)
    for i, v in enumerate(variants):
        vals = [100 * ms[v]["by_difficulty"].get(l, {}).get("ex", 0) for l in levels]
        xs = [j + (i - (len(variants) - 1) / 2) * w for j in range(len(levels))]
        ax.bar(xs, vals, width=w, color=VARIANT_COLOR.get(v, MUTED), label=LABELS[v], edgecolor=SURFACE, linewidth=2)
        for x, val in zip(xs, vals):
            ax.text(x, val + 1, f"{val:.0f}", ha="center", fontsize=8, color=INK2)
    counts = [ms[variants[0]]["by_difficulty"].get(l, {}).get("n", 0) for l in levels]
    ax.set_xticks(range(len(levels)), [f"{l}\n(n={c})" for l, c in zip(levels, counts)], color=INK)
    ax.set_ylabel("Execution accuracy (%)", color=INK2, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("Accuracy by Spider difficulty level", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def chart_training_loss(log_history: list[dict], path: Path):
    import matplotlib.pyplot as plt
    pts = [(h["step"], h["loss"]) for h in log_history if "loss" in h and "step" in h]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, linewidth=2)
    ax.set_xlabel("Optimizer step", color=INK2, fontsize=9)
    ax.set_ylabel("Training loss (SQL tokens)", color=INK2, fontsize=9)
    ax.set_title("QLoRA training loss", fontsize=10, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def chart_throughput(bench: dict, path: Path):
    import matplotlib.pyplot as plt
    v = bench.get("vllm") or []
    if not v:
        return
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    xs = [r["concurrency"] for r in v]
    ys = [r["req_per_s"] for r in v]
    ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=7, label="vLLM server (fp16)")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8,
                    color=INK)
    hf = bench.get("hf_batched")
    if hf:
        ax.axhline(hf["req_per_s"], color=ORANGE, linewidth=2, linestyle="--",
                   label=f"HF transformers, batch {hf['batch_size']}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_ylim(bottom=0, top=max(ys) * 1.15)
    ax.set_xlabel("Concurrent requests", color=INK2, fontsize=9)
    ax.set_ylabel("Requests per second", color=INK2, fontsize=9)
    ax.set_title("Serving throughput on one T4 GPU", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def error_analysis(out_dir: Path, dataset: str = "spider_dev", a: str = "base_zeroshot",
                   b: str = "finetuned", k: int = 5) -> str:
    pa = Path(out_dir) / "predictions" / dataset / f"{a}.jsonl"
    pb = Path(out_dir) / "predictions" / dataset / f"{b}.jsonl"
    if not (pa.exists() and pb.exists()):
        return ""
    ra = [json.loads(x) for x in pa.read_text().splitlines()]
    rb = [json.loads(x) for x in pb.read_text().splitlines()]
    fixed = [(x, y) for x, y in zip(ra, rb) if not x["correct"] and y["correct"]]
    broke = [(x, y) for x, y in zip(ra, rb) if x["correct"] and not y["correct"]]
    lines = [f"### What fine-tuning changed ({dataset})", "",
             f"* Questions the base model got wrong and the fine-tuned model got right: **{len(fixed)}**",
             f"* Questions the base model got right and the fine-tuned model got wrong: **{len(broke)}**", "",
             "| Error type | " + LABELS.get(a, a) + " | " + LABELS.get(b, b) + " |", "|---|---|---|"]
    from collections import Counter
    ca, cb = Counter(r["category"] for r in ra), Counter(r["category"] for r in rb)
    for cat in sorted(set(ca) | set(cb), key=lambda c: -(ca[c] + cb[c])):
        if cat != "correct":
            lines.append(f"| {cat} | {ca[cat]} | {cb[cat]} |")
    lines += ["", "Examples fixed by fine-tuning:", ""]
    for x, y in fixed[:k]:
        lines += [f"* **Q:** {x['question']}  ", f"  base: `{x['pred'][:220]}` ({x['category']})  ",
                  f"  fine-tuned: `{y['pred'][:220]}`"]
    return "\n".join(lines)


def cv_bullets(metrics: list[dict], train_stats: dict | None, bench: dict | None, model_name: str) -> str:
    get = {(m["dataset"], m["variant"]): m for m in metrics}
    def ex(d, v):
        m = get.get((d, v))
        return 100 * m["execution_accuracy"] if m else None
    b0, br, ft = ex("spider_dev", "base_zeroshot"), ex("spider_dev", "base_rag"), ex("spider_dev", "finetuned")
    q0, qft = ex("quickcommerce", "base_zeroshot"), ex("quickcommerce", "finetuned")
    nf4 = get.get(("spider_dev", "finetuned_nf4"))
    api = ex("spider_dev", "api_zeroshot")
    out = []
    if ft is not None and b0 is not None:
        tr = train_stats or {}
        out.append(f"Fine-tuned {model_name} for Text-to-SQL with QLoRA (4-bit NF4 + LoRA r={tr.get('lora', {}).get('r', 16)}, "
                   f"{tr.get('trainable_params', 0)/1e6:.1f}M trainable params) on {tr.get('train_examples', 0):,} Spider "
                   f"examples on a single T4 GPU in {tr.get('train_runtime_total_s', tr.get('train_runtime_s', 0))/60:.0f} min, raising execution accuracy on "
                   f"1,034 unseen-database questions from {b0:.1f}% to {ft:.1f}% (+{ft-b0:.1f} pts).")
    if br is not None and ft is not None:
        line = (f"Benchmarked zero-shot, fixed few-shot, retrieval-augmented few-shot (dense retrieval of similar "
                f"examples) and fine-tuned variants with an execution-based evaluator; RAG reached {br:.1f}% vs "
                f"{ft:.1f}% for the fine-tuned model")
        if api is not None:
            line += f", and a paid API model reached {api:.1f}%"
        out.append(line + ".")
    if q0 is not None and qft is not None:
        out.append(f"Built a synthetic quick-commerce SQLite database (7 tables, 25K orders) with 50 hand-verified "
                   f"business questions to test transfer to an unseen schema: {q0:.0f}% -> {qft:.0f}% execution accuracy.")
    if bench and bench.get("vllm"):
        best = max(bench["vllm"], key=lambda r: r["req_per_s"])
        hf = bench.get("hf_batched", {})
        speed = f" ({best['req_per_s']/hf['req_per_s']:.1f}x the throughput of batched HF generation)" if hf else ""
        out.append(f"Served the merged model with vLLM behind an OpenAI-compatible API at {best['req_per_s']:.1f} req/s "
                   f"and {best['output_tok_per_s']:.0f} output tokens/s at concurrency {best['concurrency']}{speed}.")
    if nf4 and ft is not None:
        out.append(f"Quantized the fine-tuned model to 4-bit NF4: {100*nf4['execution_accuracy']:.1f}% execution accuracy "
                   f"({100*nf4['execution_accuracy']-ft:+.1f} pts) in {nf4.get('model_memory_gb', 0):.2f} GB of GPU memory.")
    return "\n".join(f"- {x}" for x in out)


def build_readme(template: str, cfg, metrics: list[dict], train_stats: dict | None, bench: dict | None,
                 error_md: str, official_em: dict | None = None, findings_md: str = "",
                 engineering_md: str = "") -> str:
    get = {(m["dataset"], m["variant"]): m for m in metrics}
    b0, ft = get.get(("spider_dev", "base_zeroshot")), get.get(("spider_dev", "finetuned"))
    headline = ""
    if b0 and ft:
        headline = (f"> **Spider dev execution accuracy: {100*b0['execution_accuracy']:.1f}% → "
                    f"{100*ft['execution_accuracy']:.1f}%** after QLoRA fine-tuning "
                    f"({100*(ft['execution_accuracy']-b0['execution_accuracy']):+.1f} pts) "
                    f"on {ft['n']:,} questions over databases never seen in training.")
    ts = train_stats or {}
    train_rows = [
        f"| Training examples | {ts.get('train_examples', 'n/a'):,} ({ts.get('total_train_tokens', 0)/1e6:.2f}M tokens) |"
        if ts else "| Training examples | n/a |",
        f"| Trainable parameters | {ts.get('trainable_params', 0)/1e6:.1f}M of {ts.get('total_params', 0)/1e6:.0f}M "
        f"({100*ts.get('trainable_params', 0)/max(1, ts.get('total_params', 1)):.2f}%) |",
        f"| Steps / epochs | {ts.get('steps', 'n/a')} / {ts.get('epochs', 'n/a')} |",
        f"| Training time | {ts.get('train_runtime_total_s', ts.get('train_runtime_s', 0))/60:.0f} min on {ts.get('gpu', 'GPU')}"
        f"{' (' + ts['train_runtime_note'] + ')' if ts.get('train_runtime_note') else ''} |",
        f"| Peak GPU memory | {ts.get('peak_gpu_mem_gb', 0):.1f} GB |",
        f"| Final training loss | {ts.get('final_train_loss', float('nan')):.3f} |",
    ]
    if official_em:
        for k, v in official_em.items():
            train_rows.append(f"| Official exact-set-match, {LABELS.get(k, k)} | {100*v.get('all', 0):.1f}% |")
    bench_md = "_not run_"
    if bench and bench.get("vllm"):
        rows = ["| Setup | Requests/s | Output tokens/s | p50 latency |", "|---|---|---|---|"]
        hs = bench.get("hf_single")
        if hs:
            rows.append(f"| HF generate, 1 request at a time | {1/hs['mean_latency_s']:.2f} | - | {hs['p50_latency_s']:.2f}s |")
        hb = bench.get("hf_batched")
        if hb:
            rows.append(f"| HF generate, batch {hb['batch_size']} | {hb['req_per_s']:.2f} | {hb['output_tok_per_s']:.0f} | - |")
        for r in bench["vllm"]:
            rows.append(f"| vLLM, concurrency {r['concurrency']} | {r['req_per_s']:.2f} | {r['output_tok_per_s']:.0f} | "
                        f"{r['p50_latency_s']:.2f}s |")
        bench_md = "\n".join(rows)
    gpu = ts.get("gpu", "T4")
    rep = {"{{MODEL}}": cfg.base_model, "{{HEADLINE}}": headline, "{{RUN_MODE}}": cfg.run_mode, "{{GPU}}": gpu,
           "{{RESULTS_TABLE}}": summary_table(metrics), "{{TRAIN_TABLE}}": "\n".join(train_rows),
           "{{BENCH_TABLE}}": bench_md, "{{ERROR_ANALYSIS}}": error_md,
           "{{FINDINGS}}": findings_md, "{{ENGINEERING}}": engineering_md}
    for k, v in rep.items():
        template = template.replace(k, v)
    return template
