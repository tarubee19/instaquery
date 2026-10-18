## Key findings

1. **QLoRA fine-tuning gives a significant gain on unseen databases.** Spider dev execution accuracy rose from
   **68.5% to 74.1%** (95% CI 65.6–71.2% → 71.3–76.7%). Fine-tuning fixed 124 questions and broke 66
   (McNemar χ² = 17.1, **p < 0.0001**). Official exact-set-match rose much more (**42.6% → 63.3%**),
   because the model learned Spider's SQL conventions, not just correct answers.
2. **The gain comes from fewer schema hallucinations.** "Unknown column" errors halved (140 → 73), and
   the valid-SQL rate rose from 85.6% to 92.5%. The improvement is largest on *medium* (71.5 → 79.1%) and
   *hard* (54.0 → 62.6%) questions.
3. **Naive RAG hurts small models.** Adding the three most similar training questions from *other*
   databases dropped accuracy by 14 pts (68.5 → 54.5%). The model copied table and column names from the examples:
   "unknown table" errors went from 4 to 92. Retrieved examples need to be schema-aware (or from the same database).
4. **4-bit NF4 quantization is free here.** The quantized model scored 75.4% in 1.12 GB of GPU memory,
   vs 74.1% for fp16 in 3.09 GB (2.8× smaller). The gap is within noise; the two variants also use different decoders.
5. **vLLM serves 6× more traffic.** At 32 concurrent requests the model served 25.0 req/s (717 output tokens/s)
   vs 4.1 req/s for batched Hugging Face `generate` on the same T4. Single-request p50 latency fell from 0.59 s to 0.40 s.
6. **Transfer to a business schema is modest.** On the 50-question quick-commerce database the fine-tuned model
   scored 74% (base 72%, NF4 76%). With n = 50 the difference is not significant. Hard multi-join / date questions
   remain at 33–40%, so domain examples would be the next lever.
