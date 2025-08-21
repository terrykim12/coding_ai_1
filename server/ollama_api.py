# Minimal Ollama-like API shim for our FastAPI server
# Endpoints:
#   POST /api/generate  {prompt, stream?, options?}
#   POST /api/chat      {messages:[{role,content}], stream?, options?}
#   GET  /api/tags      -> list available "models" (logical name)
#
# Notes:
# - Uses the already-loaded base/PEFT model in app.state (no reloads).
# - Streaming uses Server-Sent Events style: lines starting with "data: {json}\n\n".
# - Options mapping:
#     options.num_predict -> max_new_tokens (default 256)
#     options.temperature, options.top_p, options.repetition_penalty pass-through
# - This is not full Ollama parity, but it's enough for curl/CLI style usage.

from __future__ import annotations
import json, time, threading
from typing import Any, Dict, Iterable, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import torch

router = APIRouter()

# -----------------------------
# Pydantic schemas
# -----------------------------
class GenerateRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    stream: bool = True
    options: Dict[str, Any] = {}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = True
    options: Dict[str, Any] = {}

# -----------------------------
# Helpers
# -----------------------------
class WallClockBudget(StoppingCriteria):
    def __init__(self, s: float = 30.0):
        self.t0 = time.monotonic()
        self.S = s
    def __call__(self, input_ids, scores, **kwargs):
        return (time.monotonic() - self.t0) >= self.S


def _extract_gen_opts(opts: Dict[str, Any]) -> Dict[str, Any]:
    # Map Ollama-ish options to transformers.generate
    max_new = int(opts.get("num_predict", 256))
    temperature = float(opts.get("temperature", 0.0))
    top_p = float(opts.get("top_p", 1.0))
    rep = float(opts.get("repetition_penalty", 1.05))

    # If temperature == 0, force deterministic
    do_sample = not (temperature == 0.0 and top_p >= 1.0)

    out = dict(
        max_new_tokens=max_new,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rep,
        use_cache=True,
    )

    # Optional hard time budget (seconds)
    max_time = float(opts.get("max_time", 30.0))
    out["max_time"] = max_time
    out["stopping_criteria"] = StoppingCriteriaList([WallClockBudget(max_time)])
    return out


def _chat_to_prompt(msgs: List[ChatMessage]) -> str:
    # Simple role-tagged concatenation
    parts = []
    for m in msgs:
        role = (m.role or "user").lower()
        if role not in ("system", "user", "assistant"):
            role = "user"
        parts.append(f"<{role}>\n{m.content}\n</{role}>")
    return "\n\n".join(parts) + "\n<assistant>\n"


def _sse_event(obj: Dict[str, Any]) -> bytes:
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")


# -----------------------------
# Routes
# -----------------------------
@router.get("/api/tags")
async def api_tags(request: Request):
    # Present a logical model name + adapter info
    adapter = getattr(request.app.state, "adapter_path", "__none__")
    name = "qwen3-8b"
    if adapter and adapter != "__none__":
        name += "+peft"
    return {"models": [{"name": name, "adapter": adapter}]}


@router.post("/api/generate")
async def api_generate(req: GenerateRequest, request: Request):
    model = request.app.state.model
    tok = request.app.state.tok
    device = next(model.parameters()).device

    # Truncate prompt defensively (keeps API snappy)
    ctx_limit = int(getattr(request.app.state, "ollama_ctx_chars", 2000))
    prompt = (req.prompt or "")[:ctx_limit]

    inputs = tok(prompt, return_tensors="pt").to(device)
    gen_args = _extract_gen_opts(req.options or {})

    if not req.stream:
        out_ids = model.generate(**inputs, **gen_args)
        text = tok.decode(out_ids[0], skip_special_tokens=True)
        # Mimic Ollama-ish JSON
        return JSONResponse({
            "model": req.model or "qwen3-8b",
            "created": int(time.time()),
            "response": text[len(prompt):] if text.startswith(prompt) else text,
            "done": True,
        })

    # Streaming via TextIteratorStreamer
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def _worker():
        try:
            model.generate(**inputs, streamer=streamer, **gen_args)
        except Exception as e:
            # Send an error sentinel to streamer
            streamer.put(e)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    def _event_stream() -> Iterable[bytes]:
        try:
            for chunk in streamer:
                if isinstance(chunk, Exception):
                    yield _sse_event({
                        "model": req.model or "qwen3-8b",
                        "created": int(time.time()),
                        "error": str(chunk),
                        "done": True,
                    })
                    return
                yield _sse_event({
                    "model": req.model or "qwen3-8b",
                    "created": int(time.time()),
                    "response": chunk,
                    "done": False,
                })
        finally:
            yield _sse_event({
                "model": req.model or "qwen3-8b",
                "created": int(time.time()),
                "done": True,
            })

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@router.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    # Convert to a single prompt then delegate to /api/generate logic
    prompt = _chat_to_prompt(req.messages or [])
    gen_req = GenerateRequest(model=req.model, prompt=prompt, stream=req.stream, options=req.options)
    # Reuse same code path without HTTP roundtrip
    return await api_generate(gen_req, request)
