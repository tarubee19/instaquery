# Model comparison study

The main study fine-tunes one model (Qwen2.5-Coder-1.5B) and compares it with prompting baselines of the *same*
model. That leaves four questions an interviewer will ask, and this study answers them with the same prompt
format, greedy decoding, execution-accuracy evaluator and data as the main results:

1. **Was 1.5B the right base model?** Zero-shot accuracy of open models from 0.5B to 7B, including newer
   families (Qwen3, Llama-3.2) and another code model (DeepSeek-Coder), against the QLoRA model.
2. **Can the RAG failure be fixed?** Finding 3 of the main study: retrieved examples from *other* databases cut
   accuracy by 14 points because the small model copied their table and column names. *Skeleton RAG*
   (`src/instaquery/skeleton.py`) keeps the shape of the example SQL but masks every identifier and literal:
   `SELECT t1.col1 FROM table1 AS t1 JOIN table2 AS t2 ON ... WHERE t2.col4 > <num>`.
3. **Does execution-guided self-correction help?** (`src/instaquery/repair.py`) If a query fails to run, the
   SQLite error goes back to the model for one more try. Only failing queries are retried and the gold answer is
   never used, so this is a fair inference-time technique.
4. **Does it hold on harder, real-world data?** BIRD dev (Li et al., NeurIPS 2023): large, messy databases
   that need domain knowledge. We use a 500-question sample, stratified by difficulty (498 scored: 2 gold
   queries fail to execute), with the official "evidence" hint and BIRD's set-equality scoring.

Run it: `notebooks/InstaQuery_Model_Comparison.ipynb` on a Colab T4 (every run is cached in Drive). The tables
below are rendered by `python experiments/render_report.py` from `results/comparison/metrics.json`.

## Results

<!-- RESULTS:START -->
| Model | Params | Spider dev EX (95% CI) | BIRD dev EX (95% CI) | Quick-commerce EX | Valid SQL (Spider / BIRD) |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B + QLoRA (ours) | 1.5B | **74.2%** (71-77) | **32.3%** (28-37) | 74% | 92% / 66% |
| Qwen2.5-Coder-1.5B (base) | 1.5B | **68.4%** (65-71) | **29.1%** (25-33) | 70% | 85% / 55% |
| Qwen2.5-Coder-0.5B | 0.5B | **52.2%** (49-55) | **12.0%** (9-15) | 52% | 68% / 30% |

| Technique (zero-shot unless stated) | Model | Spider dev | BIRD dev | Quick-commerce | Queries retried |
|---|---|---|---|---|---|
| zero-shot | Qwen2.5-Coder-1.5B (base) | 68.4% | 29.1% | 70.0% | - |
| skeleton RAG (3 masked examples) | Qwen2.5-Coder-1.5B (base) | 62.6% | - | - | - |
| + execution-guided self-correction | Qwen2.5-Coder-1.5B (base) | 70.5% | 34.5% | 76.0% | 151 / 1034 |
| zero-shot | Qwen2.5-Coder-1.5B + QLoRA (ours) | 74.2% | 32.3% | 74.0% | - |
| skeleton RAG (3 masked examples) | Qwen2.5-Coder-1.5B + QLoRA (ours) | 70.0% | - | - | - |
| + execution-guided self-correction | Qwen2.5-Coder-1.5B + QLoRA (ours) | 74.6% | 34.9% | 76.0% | 78 / 1034 |
<!-- RESULTS:END -->

## What the numbers say

**Fine-tuning still wins at its size, and it transfers.** The QLoRA model reproduces the main result on Spider
(74.2% vs 68.4% for its base, re-run here with the new harness) and keeps an edge on BIRD, a benchmark it never
saw (32.3% vs 29.1%, i.e. +3.2 points). BIRD is much harder for every model: the main failure is still schema
grounding (unknown columns: 195 of 498 answers for the base model, 134 after fine-tuning).

**Skeleton RAG fixes most of what naive RAG broke, but does not beat zero-shot.** Base model: naive RAG 54.5%,
skeleton RAG 62.6%, zero-shot 68.4%. Masking stopped the copying ("unknown table" errors 92 -> 5), but the extra
context still costs accuracy, and it costs the fine-tuned model 4 points too (74.2% -> 70.0%). Conclusion: for a
1.5B model, examples from other databases are not worth the tokens. The next thing to try is same-database
examples, which this benchmark design (unseen dev databases) does not allow.

**Self-correction is the cheapest real gain, and it helps most where the model is weakest.** One repair round adds
+2.1 points on Spider for the base model (151 of 1,034 queries retried) but only +0.4 for the fine-tuned model,
which makes fewer execution errors to begin with (78 retried). On BIRD it adds +5.4 (base) and +2.6 (fine-tuned)
points: 29.1% -> 34.5% and 32.3% -> 34.9%. The extra cost is 12-51 s per run on a T4, against 30-60 s of
generation. It cannot fix wrong-but-valid SQL ("wrong result" errors even rise slightly as broken queries become
valid wrong ones), which is now the dominant error type.

**Size matters a lot below 1.5B.** Qwen2.5-Coder-0.5B scores 52.2% on Spider and 12.0% on BIRD, far below the
1.5B models.

## Caveats

* vLLM greedy decoding is not bit-for-bit deterministic across runs and GPUs: the base model scored 29.1% and
  29.5% on BIRD in two separate runs, and 68.4% here vs 68.5% in the main study.
* BIRD numbers are on a 498-question sample (95% CI about +/-4 points). Quick-commerce has 50 questions (+/-13 points),
  so differences there are not significant.
* Qwen3 models are run with thinking disabled, so that every model answers under the same token budget.
