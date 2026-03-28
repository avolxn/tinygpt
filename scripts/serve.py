"""
FastAPI web chat server with SSE streaming.

Serves both the chat UI and API from a single FastAPI instance.
Multiple GPUs load independent model copies; requests are distributed
to available workers.

Usage:
    python -m scripts.serve --checkpoint out/checkpoints/sft
    python -m scripts.serve --checkpoint out/checkpoints/sft --num-gpus 4 --port 8000
"""

import argparse
import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from tinygpt.checkpoint import build_model_from_checkpoint
from tinygpt.engine import Engine
from tinygpt.tokenizer import HuggingFaceTokenizer
from tinygpt.utils import autodetect_device_type, compute_init

# Abuse prevention limits
MAX_MESSAGES = 500
MAX_MSG_LEN = 8000
MAX_CONV_LEN = 32000
MAX_TEMP = 2.0
MAX_TOP_K = 200
MAX_GEN_TOKENS = 4096

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="tinygpt web server")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to fine-tuned checkpoint directory")
parser.add_argument("--tokenizer-dir", type=str, default="out/tokenizer")
parser.add_argument("-n", "--num-gpus", type=int, default=1)
parser.add_argument("-t", "--temperature", type=float, default=0.8)
parser.add_argument("-k", "--top-k", type=int, default=50)
parser.add_argument("-m", "--max-tokens", type=int, default=512)
parser.add_argument("-p", "--port", type=int, default=8000)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps", ""])
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
_, _, _, _, _base_device = compute_init(device_type)


@dataclass
class Worker:
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: HuggingFaceTokenizer


class WorkerPool:
    def __init__(self, num_gpus: int) -> None:
        self.num_gpus = num_gpus
        self.workers: list[Worker] = []
        self.available: asyncio.Queue[Worker] = asyncio.Queue()

    async def initialize(self) -> None:
        for gpu_id in range(self.num_gpus):
            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
            else:
                device = torch.device(device_type)
            model, _ = build_model_from_checkpoint(args.checkpoint, device, phase="eval")
            tokenizer = HuggingFaceTokenizer.from_directory(args.tokenizer_dir)
            engine = Engine(model, tokenizer)
            worker = Worker(gpu_id=gpu_id, device=device, engine=engine, tokenizer=tokenizer)
            self.workers.append(worker)
            await self.available.put(worker)
            logger.info(f"Worker {gpu_id} ready on {device}")

    async def acquire(self) -> Worker:
        return await self.available.get()

    async def release(self, worker: Worker) -> None:
        await self.available.put(worker)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    top_k: int | None = None


def validate_request(req: ChatRequest) -> None:
    if not req.messages:
        raise HTTPException(400, "At least one message required")
    if len(req.messages) > MAX_MESSAGES:
        raise HTTPException(400, f"Too many messages (max {MAX_MESSAGES})")
    total = 0
    for i, m in enumerate(req.messages):
        if not m.content:
            raise HTTPException(400, f"Message {i} has empty content")
        if len(m.content) > MAX_MSG_LEN:
            raise HTTPException(400, f"Message {i} too long (max {MAX_MSG_LEN} chars)")
        total += len(m.content)
        if m.role not in ("user", "assistant"):
            raise HTTPException(400, f"Invalid role '{m.role}' in message {i}")
    if total > MAX_CONV_LEN:
        raise HTTPException(400, f"Conversation too long (max {MAX_CONV_LEN} chars)")
    if req.temperature is not None and not (0 <= req.temperature <= MAX_TEMP):
        raise HTTPException(400, f"temperature must be in [0, {MAX_TEMP}]")
    if req.top_k is not None and not (0 <= req.top_k <= MAX_TOP_K):
        raise HTTPException(400, f"top_k must be in [0, {MAX_TOP_K}]")
    if req.max_tokens is not None and not (1 <= req.max_tokens <= MAX_GEN_TOKENS):
        raise HTTPException(400, f"max_tokens must be in [1, {MAX_GEN_TOKENS}]")


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = WorkerPool(args.num_gpus)
    await pool.initialize()
    app.state.pool = pool
    logger.info(f"Server ready at http://localhost:{args.port}")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    html = """<!DOCTYPE html>
<html><head><title>tinygpt chat</title></head>
<body>
<h2>tinygpt Chat</h2>
<div id="chat" style="height:400px;overflow:auto;border:1px solid #ccc;padding:8px;"></div>
<input id="input" type="text" style="width:80%" placeholder="Type a message..."/>
<button onclick="send()">Send</button>
<script>
const chat = document.getElementById("chat");
const input = document.getElementById("input");
let messages = [];
async function send() {
    const content = input.value.trim();
    if (!content) return;
    input.value = "";
    messages.push({role:"user", content});
    chat.innerHTML += "<p><b>You:</b> " + content + "</p>";
    chat.innerHTML += "<p><b>Assistant:</b> <span id='resp'></span></p>";
    const resp_el = document.getElementById("resp");
    resp_el.removeAttribute("id");
    const res = await fetch("/chat/completions", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({messages})
    });
    const reader = res.body.getReader();
    let assistant_text = "";
    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        const text = new TextDecoder().decode(value);
        for (const line of text.split("\\n")) {
            if (line.startsWith("data: ")) {
                const d = JSON.parse(line.slice(6));
                if (d.token) { assistant_text += d.token; resp_el.textContent += d.token; }
                if (d.done) { messages.push({role:"assistant",content:assistant_text}); }
            }
        }
    }
    chat.scrollTop = chat.scrollHeight;
}
input.addEventListener("keydown", e => { if (e.key === "Enter") send(); });
</script>
</body></html>"""
    return HTMLResponse(html)


@app.get("/health")
async def health():
    pool: WorkerPool = app.state.pool
    return {"status": "ok", "workers": pool.num_gpus}


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    validate_request(request)
    pool: WorkerPool = app.state.pool
    worker = await pool.acquire()

    temperature = min(max(request.temperature or args.temperature, 0.0), MAX_TEMP)
    top_k = min(max(request.top_k or args.top_k, 0), MAX_TOP_K)
    max_tokens = min(max(request.max_tokens or args.max_tokens, 1), MAX_GEN_TOKENS)

    tokenizer = worker.tokenizer
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    prompt_tokens = [bos]
    for msg in request.messages:
        if msg.role == "user":
            prompt_tokens += [user_start] + tokenizer.encode(msg.content) + [user_end]
        elif msg.role == "assistant":
            prompt_tokens += [assistant_start] + tokenizer.encode(msg.content) + [assistant_end]
    prompt_tokens += [assistant_start]

    async def stream_response() -> AsyncGenerator[str, None]:
        try:
            for token_column, _ in worker.engine.generate(
                prompt_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            ):
                token = token_column[0]
                if token in (assistant_end, bos):
                    break
                text = tokenizer.decode([token])
                yield f"data: {json.dumps({'token': text})}\n\n"
                await asyncio.sleep(0)
            yield f"data: {json.dumps({'done': True})}\n\n"
        finally:
            await pool.release(worker)

    return StreamingResponse(stream_response(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
