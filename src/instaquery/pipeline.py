"""Orchestration used by the Colab notebook: data, prompt building, cached evaluation runs."""

from __future__ import annotations

import contextlib
import gc
import json
import random
import time
from pathlib import Path

from . import spider_data
from .domain_db import build_quickcommerce_db
from .generation import HFGenerator, OpenAIGenerator, load_hf_model
from .prompts import build_messages, extract_sql
from .retrieval import ExampleRetriever, pick_fixed_shots
from .schema import serialize_schema
from .serving import VLLMServer
from .sql_eval import score

DOMAIN_DB_ID = "quickcommerce"


class Workspace:
    def __init__(self, cfg, eval_json: Path):
        self.cfg = cfg
        self.eval_json = Path(eval_json)
        self.tokenizer = None
        self.retriever = None
        self.backend = "hf"          # switched to "vllm" once the server is verified
        self.timings: dict[str, float] = {}

    # ------------------------------------------------------------------ data
    def prepare_data(self):
        cfg = self.cfg
        t0 = time.time()
        self.spider_root = spider_data.download_spider(cfg.spider_dir)
        train = [e.to_dict() for e in spider_data.load_train(self.spider_root)]
        dev = [e.to_dict() for e in spider_data.load_dev(self.spider_root)]
        build_quickcommerce_db(cfg.domain_db_path)
        domain = json.loads(self.eval_json.read_text())
        for i, d in enumerate(domain):
            d.update({"db_id": DOMAIN_DB_ID, "idx": i, "split": "quickcommerce", "hardness": d["difficulty"]})
        rng = random.Random(cfg.seed)
        if cfg.quick:
            train = rng.sample(train, cfg.quick_train_examples)
            dev = rng.sample(dev, cfg.quick_dev_examples)
        self.train, self.dev, self.domain = train, dev, domain
        self.timings["data_prep_s"] = time.time() - t0
        print(f"[data] train={len(train)} dev={len(dev)} quickcommerce={len(domain)}")
        return self

    def db_lookup(self, db_id: str) -> Path:
        if db_id == DOMAIN_DB_ID:
            return Path(self.cfg.domain_db_path)
        return spider_data.db_path(self.spider_root, db_id)

    def schema_for(self, db_id: str, with_values: bool = True) -> str:
        return serialize_schema(str(self.db_lookup(db_id)),
                                self.cfg.sample_values_per_column if with_values else 0)

    def load_tokenizer(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model)
        return self.tokenizer

    def build_retriever(self):
        if self.retriever is None:
            self.retriever = ExampleRetriever(self.train, self.cfg.retriever_model, device="cpu")
            print("[retriever]", self.retriever.method)
        return self.retriever

    # --------------------------------------------------------------- prompts
    def messages_for(self, examples: list[dict], shots: str = "zero") -> list[list[dict]]:
        """shots in {'zero', 'fixed', 'rag'}"""
        tok = self.tokenizer or self.load_tokenizer()
        if shots == "fixed":
            fixed = pick_fixed_shots(self.train, self.cfg.n_shots)
            shot_lists = [fixed] * len(examples)
        elif shots == "rag":
            r = self.build_retriever()
            shot_lists = r.retrieve_many([e["question"] for e in examples], self.cfg.n_shots,
                                         exclude_db=[e["db_id"] for e in examples])
        else:
            shot_lists = [None] * len(examples)
        out = []
        for ex, sh in zip(examples, shot_lists):
            msgs = build_messages(self.schema_for(ex["db_id"], True), ex["question"], sh)
            n = len(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)) // 3
            if n > self.cfg.max_prompt_tokens:  # cheap char estimate first, exact count only if close
                ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                          add_special_tokens=False)["input_ids"]
                if len(ids) > self.cfg.max_prompt_tokens:
                    msgs = build_messages(self.schema_for(ex["db_id"], False), ex["question"], sh)
            out.append(msgs)
        return out

    # ------------------------------------------------------------ backends
    @contextlib.contextmanager
    def generator(self, model_path: str, backend: str | None = None, quant_4bit: bool = False,
                  tag: str = "model"):
        backend = backend or self.backend
        cfg = self.cfg
        if backend == "vllm":
            srv = VLLMServer(model_path, cfg.served_model_name, cfg.vllm_port, cfg.vllm_gpu_mem,
                             cfg.vllm_max_model_len, log_file=Path(cfg.output_dir) / "logs" / f"vllm_{tag}.log")
            Path(cfg.output_dir, "logs").mkdir(parents=True, exist_ok=True)
            if not srv.start():
                raise RuntimeError("vLLM server failed to start")
            self.timings[f"vllm_startup_{tag}_s"] = srv.startup_s
            try:
                # plain greedy decoding (override the model's default repetition penalty, same as HF path)
                yield OpenAIGenerator(srv.base_url, cfg.served_model_name, max_new_tokens=cfg.max_new_tokens,
                                      concurrency=64, extra_body={"repetition_penalty": 1.0, "top_p": 1.0})
            finally:
                srv.stop()
        else:
            import torch
            model, tok = load_hf_model(model_path, quant_4bit=quant_4bit)
            try:
                gen = HFGenerator(model, tok, cfg.hf_eval_batch_size, cfg.max_new_tokens)
                gen.memory_gb = model.get_memory_footprint() / 1e9
                yield gen
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()

    # ------------------------------------------------------------ evaluation
    def run_variant(self, gen, variant: str, dataset: str, shots: str = "zero", force: bool = False,
                    extra_meta: dict | None = None) -> dict:
        cfg = self.cfg
        mfile = Path(cfg.output_dir) / "metrics" / f"{dataset}__{variant}.json"
        pfile = Path(cfg.output_dir) / "predictions" / dataset / f"{variant}.jsonl"
        if mfile.exists() and not force:
            print(f"[eval] cached: {dataset}/{variant}")
            return json.loads(mfile.read_text())
        examples = self.dev if dataset == "spider_dev" else self.domain
        msgs = self.messages_for(examples, shots)
        t0 = time.time()
        outs = gen.generate(msgs, desc=f"{dataset}/{variant}")
        gen_s = time.time() - t0
        failed = sum(o.text.startswith("-- request failed") for o in outs)
        if failed > 0.2 * len(outs):
            raise RuntimeError(f"{failed}/{len(outs)} generation requests failed: {outs[0].text[:300]}")
        preds = [extract_sql(o.text) for o in outs]
        extra = [{"raw": o.text, "latency_s": o.latency_s, "prompt_tokens": o.prompt_tokens,
                  "completion_tokens": o.completion_tokens} for o in outs]
        sc = score(examples, preds, self.db_lookup, cfg.sql_timeout_s, extra,
                   keep_distinct=(dataset != "spider_dev"))
        pfile.parent.mkdir(parents=True, exist_ok=True)
        with open(pfile, "w") as f:
            for r in sc.records:
                f.write(json.dumps(r) + "\n")
        n = len(examples)
        metrics = {
            "variant": variant, "dataset": dataset, "shots": shots, "n": n,
            "execution_accuracy": sc.ex, "valid_sql_rate": sc.exec_rate,
            "by_difficulty": {k: {"ex": v[0], "n": v[1]} for k, v in sc.by("hardness").items()},
            "error_breakdown": sc.error_counts(),
            "generation_time_s": gen_s, "examples_per_s": n / gen_s if gen_s else None,
            "avg_prompt_tokens": sum(e["prompt_tokens"] for e in extra) / n,
            "avg_completion_tokens": sum(e["completion_tokens"] for e in extra) / n,
            "backend": type(gen).__name__,
        }
        if hasattr(gen, "memory_gb"):
            metrics["model_memory_gb"] = gen.memory_gb
        if extra_meta:
            metrics.update(extra_meta)
        mfile.parent.mkdir(parents=True, exist_ok=True)
        mfile.write_text(json.dumps(metrics, indent=2))
        print(f"[eval] {dataset}/{variant}: EX={sc.ex:.3f} valid={sc.exec_rate:.3f} "
              f"({n} ex, {gen_s:.0f}s)")
        return metrics

    # ------------------------------------------------------- plans + bench
    def run_plan(self, model_path: str, plan: list[tuple[str, str, str]], tag: str,
                 backend: str | None = None, quant_4bit: bool = False, after=None) -> list[dict]:
        """plan: [(variant, dataset, shots)]. Falls back from vLLM to HF if the server fails.
        `after(gen)` runs inside the same model session (used for the throughput benchmark)."""
        cfg = self.cfg
        todo = [p for p in plan if not (Path(cfg.output_dir) / "metrics" / f"{p[1]}__{p[0]}.json").exists()]
        if not todo and after is None:
            return [json.loads((Path(cfg.output_dir) / "metrics" / f"{d}__{v}.json").read_text()) for v, d, _ in plan]
        backend = backend or self.backend
        try:
            with self.generator(model_path, backend=backend, quant_4bit=quant_4bit, tag=tag) as gen:
                res = [self.run_variant(gen, v, d, s) for v, d, s in plan]
                if after is not None:
                    after(gen)
                return res
        except RuntimeError as e:
            if backend != "vllm":
                raise
            print(f"[eval] vLLM unavailable ({e}); falling back to Hugging Face generation")
            self.backend = "hf"
            return self.run_plan(model_path, plan, tag, backend="hf", quant_4bit=quant_4bit, after=after)

    def bench_messages(self) -> list[list[dict]]:
        n = min(self.cfg.bench_requests if not self.cfg.quick else 50, len(self.dev))
        return self.messages_for(self.dev[:n], "zero")

    def bench_vllm(self, gen) -> list[dict]:
        msgs = self.bench_messages()
        gen.generate(msgs[:8], concurrency=8)  # warm-up
        rows = []
        for c in self.cfg.bench_concurrency:
            r = gen.timed_batch(msgs, concurrency=c)
            print(f"[bench] vLLM concurrency={c}: {r['req_per_s']:.2f} req/s, "
                  f"{r['output_tok_per_s']:.0f} tok/s, p50 {r['p50_latency_s']:.2f}s")
            rows.append(r)
        return rows

    def bench_hf(self, model_path: str, single: int = 20) -> dict:
        import torch
        msgs = self.bench_messages()
        out = {}
        with self.generator(model_path, backend="hf", tag="hf_bench") as gen:
            gen.generate(msgs[:4])  # warm-up
            t0 = time.perf_counter()
            res = gen.generate(msgs, desc="hf batched")
            wall = time.perf_counter() - t0
            toks = sum(o.completion_tokens for o in res)
            out["hf_batched"] = {"batch_size": gen.batch_size, "requests": len(msgs), "wall_s": wall,
                                 "req_per_s": len(msgs) / wall, "output_tok_per_s": toks / wall}
            lat = sorted(gen.generate_one_timed(m).latency_s for m in msgs[:single])
            out["hf_single"] = {"requests": len(lat), "p50_latency_s": lat[len(lat) // 2],
                                "mean_latency_s": sum(lat) / len(lat)}
            out["hf_fp16_memory_gb"] = gen.memory_gb
        print(f"[bench] HF batched: {out['hf_batched']['req_per_s']:.2f} req/s; "
              f"single-request p50 {out['hf_single']['p50_latency_s']:.2f}s")
        torch.cuda.empty_cache()
        return out
