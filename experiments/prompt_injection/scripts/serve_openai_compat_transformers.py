#!/usr/bin/env python3
"""Minimal OpenAI-compatible /v1/chat/completions server backed by Transformers.

This is intentionally dependency-light (stdlib HTTP server) so it runs inside the
existing Dome Poetry env without adding FastAPI/uvicorn.

It is useful for benchmarking guard LLMs (e.g. OpenGuardrails-Text) via the
OpenAI-compatible runner in `run_eval.py`.

Example:
  poetry -C /Users/dzen/Spaces/vjl/vijil-dome run \\
    python experiments/prompt_injection/scripts/serve_openai_compat_transformers.py \\
      --model-id openguardrails/OpenGuardrails-Text-4B-0124 \\
      --served-model-name OpenGuardrails-Text-4B-0124 \\
      --port 8000

Then in another terminal:
  poetry -C /Users/dzen/Spaces/vjl/vijil-dome run \\
    python experiments/prompt_injection/scripts/run_eval.py \\
      --results-root /Users/dzen/Spaces/vjl/vijil-dome/experiments/prompt_injection/data/results/ogr_oai_smoke \\
      --corpora benchmark_pool \\
      --max-rows-dev 20 --max-rows-holdout 20 \\
      --og-openai-base-url http://127.0.0.1:8000/v1 \\
      --og-openai-model OpenGuardrails-Text-4B-0124 \\
      --og-openai-positive-category S9 \\
      --og-openai-concurrency 1
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-id", type=str, default="openguardrails/OpenGuardrails-Text-4B-0124")
    p.add_argument("--served-model-name", type=str, default="")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps"],
        help="Device to run inference on. 'auto' prefers MPS if available.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32"],
        help="Torch dtype to load the model with.",
    )
    return p.parse_args()


def choose_device(device: str) -> str:
    if device != "auto":
        return device
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(dtype: str, device: str) -> torch.dtype | None:
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    # auto
    if device == "mps":
        return torch.float16
    return None


class ServerState:
    def __init__(self, tokenizer: Any, model: Any, device: str, served_model_name: str) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.served_model_name = served_model_name
        self._gen_lock = threading.Lock()

    def chat_completion(self, messages: list[dict[str, Any]], max_tokens: int, temperature: float) -> str:
        # Apply the model's chat template. For OpenGuardrails-Text this matters;
        # the backend expects a chat-formatted prompt even if the content is [INST]...[/INST].
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        do_sample = bool(temperature and temperature > 0.0)

        with self._gen_lock, torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=int(max(1, max_tokens)),
                do_sample=do_sample,
            )

        gen = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


class OpenAICompatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = {}
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if self.path.rstrip("/") == "/v1/models":
            st: ServerState = self.server.state  # type: ignore[attr-defined]
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": st.served_model_name,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return

        self._send_json(404, {"error": {"message": "Not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "Not found"}})
            return

        st: ServerState = self.server.state  # type: ignore[attr-defined]
        data = self._read_json()
        messages = data.get("messages") or []
        if not isinstance(messages, list) or not messages:
            self._send_json(400, {"error": {"message": "Missing 'messages'."}})
            return

        max_tokens = int(data.get("max_tokens") or 16)
        temperature = float(data.get("temperature") or 0.0)

        try:
            content = st.chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            self._send_json(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
            return

        now = int(time.time())
        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": now,
            "model": st.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self._send_json(200, resp)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # Keep the server quiet; the benchmark driver already logs errors per row.
        return


def main() -> None:
    # Reduce noisy HF logging for a long-running server process.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hf_logging.set_verbosity_error()

    args = parse_args()
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    served_name = args.served_model_name.strip() or args.model_id.split("/")[-1]

    print(f"Loading model {args.model_id} on {device} (dtype={dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    print("Model ready.")

    state = ServerState(tokenizer=tok, model=model, device=device, served_model_name=served_name)
    httpd: ThreadingHTTPServer = ThreadingHTTPServer((args.host, args.port), OpenAICompatHandler)
    httpd.state = state  # type: ignore[attr-defined]
    print(f"Listening on http://{args.host}:{args.port} (OpenAI base_url http://{args.host}:{args.port}/v1)")
    print("Health: GET /health")
    print("Chat: POST /v1/chat/completions")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

