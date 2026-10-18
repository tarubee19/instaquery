"""QLoRA supervised fine-tuning (Hugging Face transformers + PEFT + bitsandbytes).

* Base model loaded in 4-bit NF4 with double quantisation (QLoRA, Dettmers et al. 2023)
* LoRA adapters (r=16, alpha=32) on all attention + MLP projections
* Loss only on the SQL answer tokens (prompt tokens are masked with -100)
* Checkpoints every `save_steps`, so a Colab disconnect can resume
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .prompts import build_messages


def build_features(examples: list[dict], tokenizer, schema_for, max_seq_len: int, max_prompt_tokens: int):
    """Tokenise chat-formatted examples; mask the prompt so loss is on the SQL only."""
    feats, skipped = [], 0
    for ex in examples:
        schema = schema_for(ex["db_id"], True)
        msgs = build_messages(schema, ex["question"], answer=ex["query"])
        prompt_txt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        p_ids = tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]
        if len(p_ids) > max_prompt_tokens:  # retry without sample values
            schema = schema_for(ex["db_id"], False)
            msgs = build_messages(schema, ex["question"], answer=ex["query"])
            prompt_txt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            p_ids = tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]
        full_txt = tokenizer.apply_chat_template(msgs, tokenize=False)
        if not full_txt.startswith(prompt_txt):
            raise ValueError("chat template mismatch between prompt and full conversation")
        a_ids = tokenizer(full_txt[len(prompt_txt):], add_special_tokens=False)["input_ids"]
        ids = p_ids + a_ids
        if len(ids) > max_seq_len:
            skipped += 1
            continue
        feats.append({"input_ids": ids, "labels": [-100] * len(p_ids) + a_ids, "length": len(ids)})
    return feats, skipped


class PadCollator:
    """LEFT-pads so every answer sits at the end of its row. The trainer then only computes
    logits for the last K positions (K = longest answer in the batch) instead of the whole
    sequence - with Qwen's 152K-token vocabulary that saves several GB of GPU memory."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        keep = 1
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, n - L:] = torch.tensor(b["input_ids"])
            lab[i, n - L:] = torch.tensor(b["labels"])
            att[i, n - L:] = 1
            keep = max(keep, sum(1 for x in b["labels"] if x != -100))
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        return {"input_ids": ids, "labels": lab, "attention_mask": att, "position_ids": pos,
                "answer_len": torch.tensor(keep)}


class LengthGroupedSampler:
    """Shuffles, then sorts examples by length inside mega-batches of 64 so each batch holds
    similar lengths (much less padding); batch order is shuffled again. Deterministic per epoch."""

    def __init__(self, lengths: list[int], batch_size: int, seed: int = 42, mega: int = 64):
        self.lengths, self.bs, self.seed, self.mega, self.epoch = lengths, batch_size, seed, mega, 0

    def __len__(self):
        return len(self.lengths)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        import random
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        batches = []
        for s in range(0, len(idx), self.mega):
            chunk = sorted(idx[s:s + self.mega], key=lambda i: -self.lengths[i])
            batches += [chunk[j:j + self.bs] for j in range(0, len(chunk), self.bs)]
        rng.shuffle(batches)
        self.epoch += 1
        return iter([i for b in batches for i in b])


def make_trainer_cls():
    from transformers import Trainer

    class AnswerOnlyTrainer(Trainer):
        """Causal-LM loss on the answer tokens only, computing logits just for those positions."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.model_accepts_loss_kwargs = False  # we return a per-micro-batch mean; Trainer divides by GA

        def _get_train_sampler(self, *args, **kwargs):
            lengths = self.train_dataset["length"]
            return LengthGroupedSampler(list(lengths), self.args.per_device_train_batch_size, self.args.seed)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            import torch.nn.functional as F
            labels = inputs.pop("labels")
            k = int(inputs.pop("answer_len"))
            try:
                out = model(**inputs, logits_to_keep=k + 1, use_cache=False)
                logits = out.logits[:, :-1, :]
                tgt = labels[:, -k:]
            except TypeError:  # very old transformers: full logits
                out = model(**inputs, use_cache=False)
                logits = out.logits[:, :-1, :]
                tgt = labels[:, 1:]
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
            return (loss, out) if return_outputs else loss

    return AnswerOnlyTrainer


def train_qlora(cfg, train_examples: list[dict], schema_for, out_dir: Path, ckpt_dir: Path) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed

    from .generation import _dtype_kwargs

    set_seed(cfg.seed)
    out_dir, ckpt_dir = Path(out_dir), Path(ckpt_dir)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    feats, skipped = build_features(train_examples, tok, schema_for, cfg.max_seq_len, cfg.max_prompt_tokens)
    lengths = [f["length"] for f in feats]
    print(f"[train] {len(feats)} examples ({skipped} skipped > {cfg.max_seq_len} tokens); "
          f"mean length {sum(lengths)/len(lengths):.0f} tokens, total {sum(lengths)/1e6:.2f}M tokens")
    ds = Dataset.from_list(feats)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, quantization_config=bnb, device_map={"": 0},
                                                 **_dtype_kwargs(torch.float16))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    try:  # PEFT counts packed 4-bit weights correctly
        trainable, total = model.get_nb_trainable_parameters()
    except Exception:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable params: {trainable/1e6:.1f}M of {total/1e6:.0f}M ({100*trainable/total:.2f}%)")

    steps_per_epoch = math.ceil(len(ds) / (cfg.per_device_batch_size * cfg.grad_accum_steps))
    max_steps = cfg.quick_max_steps if cfg.quick else -1
    total_steps = max_steps if max_steps > 0 else math.ceil(steps_per_epoch * cfg.num_epochs)
    targs = dict(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        num_train_epochs=cfg.num_epochs,
        max_steps=max_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(cfg.warmup_ratio * total_steps)),
        logging_steps=10,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=cfg.seed,
        remove_unused_columns=False,
    )
    # keep only arguments this transformers version understands (APIs move between major versions)
    import dataclasses
    known = {f.name for f in dataclasses.fields(TrainingArguments)}
    dropped = [k for k in targs if k not in known]
    if dropped:
        print("[train] TrainingArguments: ignoring unsupported", dropped)
    args = TrainingArguments(**{k: v for k, v in targs.items() if k in known})
    trainer = make_trainer_cls()(model=model, args=args, train_dataset=ds, data_collator=PadCollator(tok.pad_token_id))
    resume = any(ckpt_dir.glob("checkpoint-*")) if ckpt_dir.exists() else False
    if resume:
        print("[train] resuming from last checkpoint in", ckpt_dir)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    train_s = time.time() - t1

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    stats = {
        "train_examples": len(feats), "skipped_too_long": skipped,
        "total_train_tokens": int(sum(lengths)), "mean_seq_len": sum(lengths) / len(lengths),
        "steps": int(trainer.state.global_step), "steps_per_epoch": steps_per_epoch,
        "epochs": cfg.num_epochs if not cfg.quick else round(trainer.state.global_step / steps_per_epoch, 3),
        "train_runtime_s": train_s, "prep_s": t1 - t0,
        "final_train_loss": float(result.training_loss),
        "trainable_params": int(trainable), "total_params": int(total),
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu": torch.cuda.get_device_name(0),
        "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout},
        "effective_batch": cfg.per_device_batch_size * cfg.grad_accum_steps, "lr": cfg.learning_rate,
    }
    (out_dir / "train_stats.json").write_text(json.dumps(stats, indent=2))
    (out_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    del trainer, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return stats


def merge_adapter(base_model: str, adapter_dir: Path, merged_dir: Path) -> Path:
    """Fold LoRA weights into the base model (fp16) so it can be served by vLLM."""
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .generation import _dtype_kwargs

    merged_dir = Path(merged_dir)
    if (merged_dir / "config.json").exists():
        print("[merge] already merged:", merged_dir)
        return merged_dir
    base = AutoModelForCausalLM.from_pretrained(base_model, device_map={"": 0}, **_dtype_kwargs(torch.float16))
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(merged_dir))
    # Use the base model's original config/tokenizer files (same architecture): files written by a newer
    # transformers can be unreadable or misread by the (older) transformers inside the vLLM environment.
    try:
        import fnmatch
        import shutil
        from huggingface_hub import snapshot_download
        # config.json too: a config written by a newer transformers (e.g. rope settings moved into
        # `rope_parameters`) is silently misread by the older transformers inside vLLM (wrong RoPE theta).
        pats = ["tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
                "generation_config.json", "config.json"]
        src = Path(snapshot_download(base_model, allow_patterns=pats))
        # the cached snapshot folder also holds the base *weights*: copy only the tokenizer files
        for f in src.iterdir():
            if f.is_file() and any(fnmatch.fnmatch(f.name, p) for p in pats):
                shutil.copy(f, merged_dir / f.name)
    except Exception as e:  # pragma: no cover
        print("[merge] could not copy original tokenizer files:", e)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    diff = verify_merge(base_model, merged_dir)
    print(f"[merge] saved {merged_dir} (max |merged - base| on a LoRA-targeted weight = {diff:.4g})")
    return merged_dir


def verify_merge(base_model: str, merged_dir: Path, key: str = "model.layers.0.self_attn.q_proj.weight") -> float:
    """Sanity check: the merged weights must differ from the base weights."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    def load(folder: Path):
        for f in sorted(Path(folder).glob("*.safetensors")):
            with safe_open(str(f), framework="pt") as sf:
                if key in sf.keys():
                    return sf.get_tensor(key).float()
        raise KeyError(key)

    base_dir = Path(snapshot_download(base_model, allow_patterns=["*.safetensors", "*.json"]))
    diff = float((load(merged_dir) - load(base_dir)).abs().max())
    if diff == 0.0:
        raise RuntimeError("Merged model is identical to the base model - LoRA weights were not applied")
    return diff
