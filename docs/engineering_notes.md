## Engineering notes (problems hit and how they were solved)

| Problem | Root cause | Fix |
|---|---|---|
| vLLM failed to start on the T4 | No FlashAttention-2 on sm75, so vLLM picked FlashInfer, which JIT-compiles kernels (`ninja` not on PATH) | `--attention-backend TRITON_ATTN` (no compilation step) |
| Out-of-memory during training (3.8 GB allocation) | Qwen's 152K-token vocabulary: full-sequence logits for a 4×2048 batch need ~5 GB in fp32 | Left-pad batches and compute logits only for the answer tokens (`logits_to_keep`); peak memory 5.2 GB |
| Training slower than needed | Random batches padded to the longest sequence (60% padding) | Length-grouped sampler (8% padding), about 1.4× faster |
| **Fine-tuned model scored 15 pts *below* the base model when served by vLLM** | transformers 5 saved `rope_theta` inside `rope_parameters`; the older transformers inside vLLM silently fell back to the default RoPE base (10,000 instead of 1,000,000) | Ship the base model's `config.json` with the merged weights. Caught because the 4-bit HF evaluation of the *same* weights scored 75% (a parity check between serving paths) |
| Merged model was identical to the base model (first attempt) | Copying tokenizer files from the Hugging Face cache snapshot also copied the base weights | Copy files by pattern only, plus a `verify_merge` check that fails if the LoRA delta is zero |
| Colab runtime recycled mid-training | Free-tier runtime disconnected at step ~92 | Checkpoints every 100 steps to Google Drive; the run resumed from step 100 |
| `TrainingArguments(warmup_ratio=...)` crashed | Argument removed in transformers 5 | Filter arguments against the installed version's dataclass fields |
