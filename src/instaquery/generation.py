"""Two interchangeable generation back-ends with the same interface:

* HFGenerator      - Hugging Face transformers, batched greedy decoding on the GPU
* OpenAIGenerator  - any OpenAI-compatible endpoint (vLLM server, paid API)

Both take chat `messages` lists and return raw text plus timing/token stats.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass
class GenOutput:
    text: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ======================================================================= HF
def _dtype_kwargs(dtype):
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def load_hf_model(model_id: str, quant_4bit: bool = False, device_map="auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(device_map=device_map)
    if quant_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
    kwargs.update(_dtype_kwargs(torch.float16))
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tok


class HFGenerator:
    def __init__(self, model, tokenizer, batch_size: int = 16, max_new_tokens: int = 256):
        self.model, self.tok = model, tokenizer
        self.batch_size, self.max_new_tokens = batch_size, max_new_tokens

    def render(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, batch_messages: list[list[dict]], desc: str = "generating") -> list[GenOutput]:
        import torch
        from tqdm.auto import tqdm

        prompts = [self.render(m) for m in batch_messages]
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        outs: list[GenOutput | None] = [None] * len(prompts)
        self.tok.padding_side = "left"
        for s in tqdm(range(0, len(order), self.batch_size), desc=desc, leave=False):
            idx = order[s:s + self.batch_size]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          repetition_penalty=1.0, pad_token_id=self.tok.pad_token_id)
            dt = (time.perf_counter() - t0) / len(idx)
            new = gen[:, enc["input_ids"].shape[1]:]
            for row, i in enumerate(idx):
                ids = new[row]
                n_out = int((ids != self.tok.pad_token_id).sum())
                outs[i] = GenOutput(self.tok.decode(ids, skip_special_tokens=True), dt,
                                    int(enc["attention_mask"][row].sum()), n_out)
        return outs  # type: ignore

    def generate_one_timed(self, messages: list[dict]) -> GenOutput:
        """Unbatched request (used to compare single-request latency with vLLM)."""
        import torch
        enc = self.tok([self.render(messages)], return_tensors="pt", add_special_tokens=False).to(self.model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      repetition_penalty=1.0, pad_token_id=self.tok.pad_token_id)
        dt = time.perf_counter() - t0
        new = gen[0, enc["input_ids"].shape[1]:]
        return GenOutput(self.tok.decode(new, skip_special_tokens=True), dt,
                         int(enc["input_ids"].shape[1]), int(new.shape[0]))


# =================================================================== OpenAI
class OpenAIGenerator:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", max_new_tokens: int = 256,
                 concurrency: int = 32, timeout: float = 300.0, extra_body: dict | None = None):
        self.extra_body = extra_body or {}
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key = model, api_key
        self.max_new_tokens, self.concurrency, self.timeout = max_new_tokens, concurrency, timeout

    async def _one(self, client, sem, messages):
        payload = {"model": self.model, "messages": messages, "temperature": 0.0,
                   "max_tokens": self.max_new_tokens, **self.extra_body}
        async with sem:
            for attempt in range(4):
                t0 = time.perf_counter()
                try:
                    r = await client.post(f"{self.base_url}/chat/completions", json=payload)
                    if r.status_code == 429 or r.status_code >= 500:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    r.raise_for_status()
                    data = r.json()
                    dt = time.perf_counter() - t0
                    usage = data.get("usage") or {}
                    return GenOutput(data["choices"][0]["message"]["content"] or "", dt,
                                     usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                except Exception as e:
                    if attempt == 3:
                        return GenOutput(f"-- request failed: {e}", time.perf_counter() - t0)
                    await asyncio.sleep(2 ** attempt)

    async def _run(self, batch_messages, concurrency):
        import httpx
        sem = asyncio.Semaphore(concurrency)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
            tasks = [self._one(client, sem, m) for m in batch_messages]
            try:
                from tqdm.asyncio import tqdm_asyncio
                return await tqdm_asyncio.gather(*tasks, leave=False)
            except ImportError:
                return await asyncio.gather(*tasks)

    def generate(self, batch_messages: list[list[dict]], desc: str = "", concurrency: int | None = None):
        conc = concurrency or self.concurrency
        try:
            asyncio.get_running_loop()
            in_loop = True   # Jupyter/Colab already runs an event loop in this thread
        except RuntimeError:
            in_loop = False
        if in_loop:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(1) as ex:
                return ex.submit(asyncio.run, self._run(batch_messages, conc)).result()
        return asyncio.run(self._run(batch_messages, conc))

    def timed_batch(self, batch_messages, concurrency: int) -> dict:
        t0 = time.perf_counter()
        outs = self.generate(batch_messages, concurrency=concurrency)
        wall = time.perf_counter() - t0
        lat = sorted(o.latency_s for o in outs)
        toks = sum(o.completion_tokens for o in outs)
        return {"concurrency": concurrency, "requests": len(outs), "wall_s": wall,
                "req_per_s": len(outs) / wall, "output_tok_per_s": toks / wall,
                "p50_latency_s": lat[len(lat) // 2], "p95_latency_s": lat[int(len(lat) * 0.95) - 1],
                "failed": sum(o.text.startswith("-- request failed") for o in outs)}
