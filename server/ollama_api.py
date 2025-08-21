# Minimal Ollama-like API shim for our FastAPI server (v2.1 hotfix)
# Endpoints:
#   POST /api/generate  {prompt, stream?, options?}
#   POST /api/chat      {messages:[{role,content}], stream?, options?}
#   GET  /api/tags      -> list available "models" (logical name)
#
# Notes:
# - Uses the already-loaded base/PEFT model in app.state (no reloads).
# - Streaming uses Server-Sent Events style: lines starting with "data: {json}\n\n".
# - Qwen-specific chat template for proper conversation formatting.
# - Safe template handling with fallback for template-less tokenizers.
# - Options mapping:
#     options.num_predict -> max_new_tokens (default 64)
#     options.temperature, options.top_p, options.repetition_penalty pass-through
# - This is not full Ollama parity, but it's enough for curl/CLI style usage.

from __future__ import annotations
import json, time, threading
from typing import Any, Dict, Iterable, List, Optional

from fastapi import APIRouter, Request, HTTPException
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
    def __init__(self, s: float = 20.0):
        self.t0 = time.monotonic()
        self.S = s
    def __call__(self, input_ids, scores, **kwargs):
        return (time.monotonic() - self.t0) >= self.S


def _extract_gen_opts(opts: Dict[str, Any]) -> Dict[str, Any]:
    # Map Ollama-ish options to transformers.generate with safe defaults
    max_new = int(opts.get("num_predict", 64))  # Reduced from 256 to 64
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

    # Optional hard time budget (seconds) - reduced from 30s to 20s
    max_time = float(opts.get("max_time", 20.0))
    out["max_time"] = max_time
    out["stopping_criteria"] = StoppingCriteriaList([WallClockBudget(max_time)])
    return out


def _normalize_role(role: str) -> str:
    """Normalize role to Qwen-supported roles"""
    role = (role or "user").lower()
    if role not in ("system", "user", "assistant"):
        role = "user"
    return role


def _build_chat_input(tok, messages: List[Dict[str, str]], device) -> Dict[str, torch.Tensor]:
    """Build chat input with safe template handling"""
    try:
        # Try Qwen's built-in chat template first
        prompt_txt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback to simple concatenation if template fails
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            parts.append(f"<{role}>\n{content}\n</{role}>")
        prompt_txt = "\n\n".join(parts) + "\n<assistant>\n"
    
    # Encode after we have final text
    inputs = tok(prompt_txt, return_tensors="pt")
    return inputs.to(device)


def _sse_event(obj: Dict[str, Any]) -> bytes:
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")


# -----------------------------
# Routes
# -----------------------------
@router.get("/api/tags")
async def api_tags(request: Request):
    adapter = getattr(request.app.state, "adapter_path", "__none__")
    name = "qwen3-8b" + ("+peft" if adapter and adapter != "__none__" else "")
    return {"models": [{"name": name, "adapter": adapter}]}


@router.post("/api/generate")
async def api_generate(req: GenerateRequest, request: Request):
    model = request.app.state.model; tok = request.app.state.tok
    device = next(model.parameters()).device

    try:
        ctx_limit = int(getattr(request.app.state, "ollama_ctx_chars", 2000))
        user_prompt = (req.prompt or "")[:ctx_limit]
        system_prompt = (req.options or {}).get("system", "You are a helpful, concise assistant.")
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        inputs = _build_chat_input(tok, messages, device)
        gen_args = _extract_gen_opts(req.options or {})

        if not req.stream:
            with torch.no_grad():
                out_ids = model.generate(**inputs, **gen_args)
            gen_only = out_ids[0, inputs["input_ids"].shape[-1]:]
            text = tok.decode(gen_only, skip_special_tokens=True)
            return JSONResponse({"model": req.model or "qwen3-8b", "created": int(time.time()), "response": text, "done": True})

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        def _worker():
            try:
                model.generate(**inputs, streamer=streamer, **gen_args)
            except Exception as e:
                streamer.put(e)
        threading.Thread(target=_worker, daemon=True).start()

        def _event_stream():
            try:
                for chunk in streamer:
                    if isinstance(chunk, Exception):
                        yield _sse_event({"model": req.model or "qwen3-8b", "created": int(time.time()), "error": str(chunk), "done": True}); return
                    yield _sse_event({"model": req.model or "qwen3-8b", "created": int(time.time()), "response": chunk, "done": False})
            finally:
                yield _sse_event({"model": req.model or "qwen3-8b", "created": int(time.time()), "done": True})
        return StreamingResponse(_event_stream(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")


@router.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    model = request.app.state.model; tok = request.app.state.tok
    device = next(model.parameters()).device

    try:
        msgs = [{"role": _normalize_role(m.role), "content": m.content} for m in (req.messages or [])]
        if not msgs:
            raise HTTPException(400, "messages cannot be empty")
        inputs = _build_chat_input(tok, msgs, device)
        gen_args = _extract_gen_opts(req.options or {})

        if not req.stream:
            with torch.no_grad():
                out_ids = model.generate(**inputs, **gen_args)
            gen_only = out_ids[0, inputs["input_ids"].shape[-1]:]
            text = tok.decode(gen_only, skip_special_tokens=True)
            return JSONResponse({"model": req.model or "qwen3-8b", "created": int(time.time()), "response": text, "done": True})

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        def _worker():
            try:
                model.generate(**inputs, streamer=streamer, **gen_args)
            except Exception as e:
                streamer.put(e)
        threading.Thread(target=_worker, daemon=True).start()

        def _event_stream():
            try:
                for chunk in streamer:
                    if isinstance(chunk, Exception):
                        yield _sse_event({"model": req.model or "qwen3-8b", "created": int(time.time()), "error": str(chunk), "done": True}); return
                    yield _sse_event({"model": req.model or "qwen3-8b", "created": int(time.time()), "response": chunk, "done": False})
            finally:
                yield _sse_event({"model": req.model or "qwen3-8b", "created": int(time.time()), "done": True})
        return StreamingResponse(_event_stream(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")
