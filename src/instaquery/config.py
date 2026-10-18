"""Central configuration. Every stage reads paths and hyper-parameters from here."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---- run mode -------------------------------------------------------
    # "quick": tiny subsets to smoke-test the whole pipeline (~20 min on a T4)
    # "full" : complete Spider train/dev (~2-3 h on a T4)
    run_mode: str = "full"

    # ---- paths ----------------------------------------------------------
    work_dir: str = "/content/instaquery_work"      # fast local disk (data, models)
    output_dir: str = "/content/instaquery_outputs"  # results (copied to Drive if mounted)

    # ---- model ----------------------------------------------------------
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    served_model_name: str = "instaquery"

    # ---- prompt / data --------------------------------------------------
    max_prompt_tokens: int = 1800       # drop sample values from schema above this
    max_seq_len: int = 2048             # prompt + answer during training
    sample_values_per_column: int = 3
    n_shots: int = 3                    # examples for few-shot and RAG variants
    retriever_model: str = "BAAI/bge-small-en-v1.5"

    # ---- QLoRA ----------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: float = 1.0
    per_device_batch_size: int = 4
    grad_accum_steps: int = 4
    warmup_ratio: float = 0.03
    save_steps: int = 100
    seed: int = 42

    # ---- generation / eval ----------------------------------------------
    max_new_tokens: int = 256
    hf_eval_batch_size: int = 16
    sql_timeout_s: float = 15.0

    # ---- serving --------------------------------------------------------
    vllm_version: str = "0.19.0"
    vllm_port: int = 8000
    vllm_gpu_mem: float = 0.85
    vllm_max_model_len: int = 4096
    bench_requests: int = 200
    bench_concurrency: list = field(default_factory=lambda: [1, 8, 32])

    # ---- quick-mode subset sizes ------------------------------------------
    quick_train_examples: int = 400
    quick_dev_examples: int = 100
    quick_max_steps: int = 40

    # ---- optional paid-API reference ----------------------------------------
    api_model: str = "gpt-4o-mini"
    api_base_url: str = "https://api.openai.com/v1"
    api_eval_examples: int = 1034

    # ------------------------------------------------------------------------
    @property
    def quick(self) -> bool:
        return self.run_mode == "quick"

    def path(self, *parts: str) -> Path:
        p = Path(self.work_dir).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def out(self, *parts: str) -> Path:
        p = Path(self.output_dir).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def spider_dir(self) -> Path:
        return Path(self.work_dir) / "spider"

    @property
    def adapter_dir(self) -> Path:
        return Path(self.output_dir) / "model" / "lora_adapter"

    @property
    def merged_dir(self) -> Path:
        return Path(self.work_dir) / f"merged_model_{self.run_mode}"

    @property
    def domain_db_path(self) -> Path:
        return Path(self.work_dir) / "quickcommerce" / "quickcommerce.sqlite"

    def save(self) -> None:
        self.out("config.json").write_text(json.dumps(asdict(self), indent=2))


def load_config(**overrides) -> Config:
    cfg = Config(**overrides)
    os.makedirs(cfg.work_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    return cfg
