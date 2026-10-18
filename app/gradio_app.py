"""InstaQuery demo: ask a question in English, get SQL + results from the quick-commerce database.

Run locally:   python app/gradio_app.py --model <merged_model_dir_or_hf_id> --db quickcommerce.sqlite
Or point it at a running vLLM server:  --api http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from instaquery.prompts import build_messages, extract_sql  # noqa: E402
from instaquery.safety import check_sql  # noqa: E402
from instaquery.schema import serialize_schema  # noqa: E402
from instaquery.sql_eval import execute_sql  # noqa: E402

EXAMPLES = [
    "Which 5 products generated the most revenue?",
    "What is the average delivery time of delivered orders in each city?",
    "Which pincodes have more than one dark store, and how many orders did each get?",
    "How many orders contain both bananas and strawberries?",
    "Which delivery partners in Pune have a rating above 4.8?",
]


def build_demo(generate_fn, db_path: str):
    import gradio as gr
    import pandas as pd

    schema = serialize_schema(str(db_path), 3)

    def answer(question: str):
        if not question.strip():
            return "", None, "Type a question."
        t0 = time.perf_counter()
        raw = generate_fn(build_messages(schema, question))
        sql = extract_sql(raw)
        ok, why = check_sql(sql)
        if not ok:
            return sql, None, f"Blocked by guardrail: {why}"
        flag, res = execute_sql(db_path, sql, timeout_s=10)
        dt = time.perf_counter() - t0
        if flag != "ok":
            return sql, None, f"SQL error: {res}"
        df = pd.DataFrame(res[:200])
        return sql, df, f"{len(res)} rows in {dt:.2f}s (read-only database, 10s timeout)"

    with gr.Blocks(title="InstaQuery: Text-to-SQL") as demo:
        gr.Markdown("## InstaQuery: ask the quick-commerce database in plain English\n"
                    "A QLoRA fine-tuned small LLM writes the SQL; queries run read-only with guardrails.")
        q = gr.Textbox(label="Question", placeholder=EXAMPLES[0])
        btn = gr.Button("Generate SQL and run", variant="primary")
        sql_out = gr.Code(label="Generated SQL", language="sql")
        table = gr.Dataframe(label="Result (first 200 rows)")
        status = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=q)
        btn.click(answer, q, [sql_out, table, status])
        q.submit(answer, q, [sql_out, table, status])
    return demo


def _hf_generate_fn(model_dir: str):
    from instaquery.generation import HFGenerator, load_hf_model
    model, tok = load_hf_model(model_dir)
    gen = HFGenerator(model, tok, batch_size=1, max_new_tokens=256)
    return lambda msgs: gen.generate_one_timed(msgs).text


def _api_generate_fn(base_url: str, model: str):
    from instaquery.generation import OpenAIGenerator
    gen = OpenAIGenerator(base_url, model)
    return lambda msgs: gen.generate([msgs])[0].text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--api", default=None)
    ap.add_argument("--served-name", default="instaquery")
    a = ap.parse_args()
    fn = _api_generate_fn(a.api, a.served_name) if a.api else _hf_generate_fn(a.model)
    build_demo(fn, a.db).launch()
