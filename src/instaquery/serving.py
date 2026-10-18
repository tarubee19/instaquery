"""Serve a model with vLLM (OpenAI-compatible API) from an isolated virtual env.

vLLM pins its own torch/CUDA build; installing it into a separate `uv` venv
keeps the training environment (transformers/peft/bitsandbytes) untouched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

VENV = Path("/content/vllm-env")


def vllm_python() -> Path:
    return VENV / "bin" / "python"


def install_vllm(version: str = "0.19.0", log_file: Path | None = None) -> bool:
    if vllm_python().exists():
        r = subprocess.run([str(vllm_python()), "-c", "import vllm;print(vllm.__version__)"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print("[vllm] already installed:", r.stdout.strip())
            return True
    t0 = time.time()
    log = open(log_file, "w") if log_file else subprocess.DEVNULL
    try:
        if shutil.which("uv") is None:
            subprocess.run(["pip", "install", "-q", "uv"], check=True)
        subprocess.run(["uv", "venv", str(VENV), "--python", "3.12", "--seed"], check=True, stdout=log, stderr=log)
        subprocess.run(["uv", "pip", "install", "--python", str(vllm_python()), f"vllm=={version}"],
                       check=True, stdout=log, stderr=log)
        print(f"[vllm] installed vllm=={version} in {time.time()-t0:.0f}s")
        return True
    except subprocess.CalledProcessError as e:
        print("[vllm] install failed:", e, "- see", log_file)
        return False
    finally:
        if log_file:
            log.close()


class VLLMServer:
    def __init__(self, model_path: str, served_name: str, port: int = 8000, gpu_mem: float = 0.85,
                 max_model_len: int = 4096, log_file: Path | None = None, extra_args: list[str] | None = None,
                 attention_backend: str | None = "TRITON_ATTN"):
        # T4 (sm75) has no FlashAttention-2; vLLM would pick FlashInfer, which JIT-compiles kernels.
        # The Triton backend needs no compilation step.
        self.attention_backend = attention_backend
        self.model_path, self.served_name, self.port = str(model_path), served_name, port
        self.gpu_mem, self.max_model_len = gpu_mem, max_model_len
        self.log_file = Path(log_file) if log_file else Path("/content/vllm_server.log")
        self.extra_args = extra_args or []
        self.proc: subprocess.Popen | None = None
        self.startup_s = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def start(self, timeout_s: int = 900) -> bool:
        cmd = [str(vllm_python()), "-m", "vllm.entrypoints.openai.api_server",
               "--model", self.model_path, "--served-model-name", self.served_name,
               "--port", str(self.port), "--dtype", "half",
               "--gpu-memory-utilization", str(self.gpu_mem),
               "--max-model-len", str(self.max_model_len), "--seed", "0", *self.extra_args]
        if self.attention_backend:
            cmd += ["--attention-backend", self.attention_backend]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PATH"] = f"{VENV / 'bin'}:{env.get('PATH', '')}"   # ninja etc. from the venv
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        self._log = open(self.log_file, "w")
        t0 = time.time()
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, env=env)
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                print("[vllm] server exited early; last log lines:\n" + self.tail())
                return False
            if self._healthy():
                self.startup_s = time.time() - t0
                print(f"[vllm] server ready in {self.startup_s:.0f}s at {self.base_url}")
                return True
            time.sleep(5)
        print("[vllm] timed out waiting for server; last log lines:\n" + self.tail())
        self.stop()
        return False

    def tail(self, n: int = 40) -> str:
        try:
            return "\n".join(self.log_file.read_text(errors="ignore").splitlines()[-n:])
        except Exception:
            return ""

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        try:
            self._log.close()
        except Exception:
            pass
        time.sleep(5)  # let the GPU memory be released
