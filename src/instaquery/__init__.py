"""InstaQuery: fine-tuning a small open LLM for Text-to-SQL with QLoRA.

Pipeline: Spider data -> prompt building (schema + question) -> baselines
(zero-shot, few-shot, retrieval-augmented few-shot) -> QLoRA fine-tuning ->
execution-accuracy evaluation -> 4-bit quantization -> vLLM serving + benchmark.
"""

__version__ = "1.0.0"
