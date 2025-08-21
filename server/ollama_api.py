# Minimal Ollama-like API shim for our FastAPI server (v2.5 - safe cleanup fallback + adjusted stopping)
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
# - Robust response cleanup with safe fallback to prevent empty responses.
# - Enhanced error handling with logging and validation.
# - Fixed StopOnSeq tokenizer dependency issue.
# - Options mapping:
#     options.num_predict -> max_new_tokens (default 24)
#     options.temperature, options.top_p, options.repetition_penalty pass-through
# - This is not full Ollama parity, but it's enough for curl/CLI style usage.

from __future__ import annotations
import json, time, threading, re
from typing import Any, Dict, Iterable, List, Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import torch

router = APIRouter()

# --- Cleaners (robust) ---
import re as _re
import logging
logger = logging.getLogger("ollama_api")

# 분석 금지 시스템 프롬프트
NO_THINK_SYSTEM = (
    "You are a concise assistant. Answer directly with the final answer only. "
    "Do not include analysis, thoughts, plans, or meta commentary. "
    "If user greets, reply with a short greeting only."
)

# 클린업 패턴들
_CLEAN_THINK_RE = _re.compile(r"<think>.*?</think>\s*", _re.IGNORECASE | _re.DOTALL)
_CLEAN_TAGS_RE  = _re.compile(r"</?(assistant|user|system|message|finish|think)>", _re.IGNORECASE)
_META_LEAD_RE   = _re.compile(r'^\s*(?:Okay|Alright|Sure|Well)[, ]+(?:the user|you)\b[^.!?\n]*[.!?]\s+', _re.IGNORECASE)
_I_NEED_TO_RE   = _re.compile(r'^\s*(?:I\s+(?:need|should|will|am going)\s+to\b)[^\n]*\n?', _re.IGNORECASE)
_LET_ME_RE      = _re.compile(r"^\s*(?:Let(?:’|')?s|Let\s+me)\b[^\n]*\n?", _re.IGNORECASE)
_SINCE_USER_RE  = _re.compile(r'^\s*(?:Since|Because)\s+(?:they|you|the\s+user)\b[^\n]*\n?', _re.IGNORECASE)
_KO_META_RE     = _re.compile(r'^\s*(?:좋아|알겠어|음|자)\b[^!\?\.\n]*[!\?\.]\s+', _re.IGNORECASE)
_SAY_HELLO_RE   = _re.compile(r'\bsay\s+hello\b', _re.IGNORECASE)
_SAY_HI_RE      = _re.compile(r'\bsay\s+hi\b', _re.IGNORECASE)
_SAY_HEY_RE     = _re.compile(r'\bsay\s+hey\b', _re.IGNORECASE)

def _clean_response(text: str) -> str:
    """강력 정리 → 비면 특수 케이스/인용구 → 약한 정리 → 원문 순으로 폴백."""
    orig = text or ""
    # 약한 정리부터 준비
    weak = _CLEAN_TAGS_RE.sub("", _CLEAN_THINK_RE.sub("", orig)).strip()

    # 강한 정리
    t = _META_LEAD_RE.sub("", weak)
    t = _I_NEED_TO_RE.sub("", t)
    t = _LET_ME_RE.sub("", t)
    t = _SINCE_USER_RE.sub("", t)
    t = _KO_META_RE.sub("", t)
    t = t.strip()
    if t:
        return t

    # 폴백 1: 특수 케이스 - 사용자가 "say hello" 류를 요구한 흔적이 있으면 간결한 인사로 강제
    if _SAY_HELLO_RE.search(orig):
        return "Hello!"
    if _SAY_HI_RE.search(orig):
        return "Hi!"
    if _SAY_HEY_RE.search(orig):
        return "Hey!"

    # 폴백 2: 따옴표 내용이 있으면 사용
    m = _re.search(r'"([^"]{1,80})"', orig)
    if m:
        return m.group(1).strip()

    # 폴백 3: 약한 정리 결과
    if weak:
        return weak

    # 폴백 3: 원문
    return orig.strip()

def _to_device_inputs(enc, device):
    # BatchEncoding.to()가 없는 버전까지 커버
    if hasattr(enc, "to"):
        moved = enc.to(device)
        # 일부 버전에선 .to()가 None을 리턴 → dict로 강제 변환
        if moved is None:
            return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
        return moved
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}

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


class StopOnSeq(StoppingCriteria):
    """Stop when a specific token sequence appears at the end."""
    def __init__(self, tok, seq: str):
        if tok is None:
            raise ValueError("tokenizer must not be None")
        enc = tok(seq, add_special_tokens=False)
        self.ids = enc["input_ids"]
    def __call__(self, input_ids, scores, **kw):
        if not self.ids:
            return False
        L = len(self.ids)
        if L == 0 or input_ids.shape[-1] < L:
            return False
        tail = input_ids[0].tolist()[-L:]
        return tail == self.ids


def _extract_gen_opts(opts: Dict[str, Any]) -> Dict[str, Any]:
    max_new = int(opts.get("num_predict", 24))
    temperature = float(opts.get("temperature", 0.0))
    top_p = float(opts.get("top_p", 1.0))
    rep = float(opts.get("repetition_penalty", 1.05))
    do_sample = not (temperature == 0.0 and top_p >= 1.0)
    max_time = float(opts.get("max_time", 20.0))

    base = dict(
        max_new_tokens=max_new,
        repetition_penalty=rep,
        use_cache=True,
        max_time=max_time,   # 시간만 넘기고, stopping_criteria는 엔드포인트에서 구성
    )
    if do_sample:
        base.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    else:
        base.update(dict(do_sample=False))
    return base


def _normalize_role(role: str) -> str:
    """Normalize role to Qwen-supported roles"""
    role = (role or "user").lower()
    if role not in ("system", "user", "assistant"):
        role = "user"
    return role


def _build_chat_input(tok, messages: List[Dict[str, str]], device):
    """Try Qwen chat template; if unavailable, fall back to simple join."""
    try:
        prompt_txt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        sys = next((m["content"] for m in messages if m.get("role") == "system"), None)
        non_sys = [m for m in messages if m.get("role") != "system"]
        parts = []
        if sys:
            parts.append(f"<system>\n{sys}\n</system>")
        for m in non_sys:
            parts.append(f"<{m['role']}>\n{m['content']}\n</{m['role']}>")
        parts.append("<assistant>\n")
        prompt_txt = "\n\n".join(parts)
    enc = tok(prompt_txt, return_tensors="pt")
    return _to_device_inputs(enc, device)


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
    model = request.app.state.model
    tok = request.app.state.tok
    if tok is None:
        raise HTTPException(500, "tokenizer not loaded (app.state.tok is None)")
    device = next(model.parameters()).device
    try:
        ctx_limit = int(getattr(request.app.state, "ollama_ctx_chars", 2000))
        user_prompt = (req.prompt or "")[:ctx_limit]
        system_prompt = (req.options or {}).get("system", NO_THINK_SYSTEM)
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        inputs = _build_chat_input(tok, messages, device)
        gen_args = _extract_gen_opts(req.options or {})
        max_time = gen_args.get("max_time", 20.0)

        # 기본 시간 제한
        stops = [WallClockBudget(max_time)]

        # 기본은 끔 (원하면 클라이언트 options.stop_think=true로 켤 수 있게)
        stop_think = bool((req.options or {}).get("stop_think", False))
        if stop_think:
            try:
                stops.append(StopOnSeq(tok, "</think>"))
            except Exception as e:
                logger.warning(f"StopOnSeq disabled: {e}")

        gen_args["stopping_criteria"] = StoppingCriteriaList(stops)

        if not req.stream:
            with torch.no_grad():
                out_ids = model.generate(**inputs, **gen_args)
            gen_only = out_ids[0, inputs["input_ids"].shape[-1]:]
            text = tok.decode(gen_only, skip_special_tokens=True)
            if getattr(request.app.state, "clean_response_enabled", True):
                text = _clean_response(text)
            return JSONResponse({"model": req.model or "qwen3-8b",
                                 "created": int(time.time()),
                                 "response": text, "done": True})

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        def _worker():
            try:
                model.generate(**inputs, streamer=streamer, **gen_args)
            except Exception as e:
                logger.exception("api_generate worker failed")
                streamer.put(e)
        threading.Thread(target=_worker, daemon=True).start()

        def _event_stream():
            try:
                for chunk in streamer:
                    if isinstance(chunk, Exception):
                        yield _sse_event({"model": req.model or "qwen3-8b",
                                          "created": int(time.time()),
                                          "error": str(chunk), "done": True}); return
                    chunk_text = _clean_response(chunk) if getattr(request.app.state, "clean_response_enabled", True) else chunk
                    yield _sse_event({"model": req.model or "qwen3-8b",
                                      "created": int(time.time()),
                                      "response": chunk_text,
                                      "done": False})
            finally:
                yield _sse_event({"model": req.model or "qwen3-8b",
                                  "created": int(time.time()), "done": True})
        return StreamingResponse(_event_stream(), media_type="text/event-stream")
    except Exception as e:
        logger.exception("api_generate failed")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")


@router.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    model = request.app.state.model
    tok = request.app.state.tok
    if tok is None:
        raise HTTPException(500, "tokenizer not loaded (app.state.tok is None)")
    device = next(model.parameters()).device

    try:
        msgs = [{"role": _normalize_role(m.role), "content": m.content} for m in (req.messages or [])]
        if not msgs:
            raise HTTPException(400, "messages cannot be empty")
        
        # 사용자가 system을 주지 않은 경우에만 기본 삽입
        if msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": NO_THINK_SYSTEM}] + msgs
            
        inputs = _build_chat_input(tok, msgs, device)
        gen_args = _extract_gen_opts(req.options or {})
        max_time = gen_args.get("max_time", 20.0)

        # 기본 시간 제한
        stops = [WallClockBudget(max_time)]

        # 기본은 끔 (원하면 클라이언트 options.stop_think=true로 켤 수 있게)
        stop_think = bool((req.options or {}).get("stop_think", False))
        if stop_think:
            try:
                stops.append(StopOnSeq(tok, "</think>"))
            except Exception as e:
                logger.warning(f"StopOnSeq disabled: {e}")

        gen_args["stopping_criteria"] = StoppingCriteriaList(stops)

        if not req.stream:
            with torch.no_grad():
                out_ids = model.generate(**inputs, **gen_args)
            gen_only = out_ids[0, inputs["input_ids"].shape[-1]:]
            text = tok.decode(gen_only, skip_special_tokens=True)
            if getattr(request.app.state, "clean_response_enabled", True):
                text = _clean_response(text)
            return JSONResponse({"model": req.model or "qwen3-8b", "created": int(time.time()), "response": text, "done": True})

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        def _worker():
            try:
                model.generate(**inputs, streamer=streamer, **gen_args)
            except Exception as e:
                logger.exception("api_chat worker failed")
                streamer.put(e)
        threading.Thread(target=_worker, daemon=True).start()

        def _event_stream():
            try:
                for chunk in streamer:
                    if isinstance(chunk, Exception):
                        yield _sse_event({"model": req.model or "qwen3-8b", "created": int(time.time()), "error": str(chunk), "done": True}); return
                    chunk_text = _clean_response(chunk) if getattr(request.app.state, "clean_response_enabled", True) else chunk
                    yield _sse_event({"model": req.model or "qwen3-8b", "created": int(time.time()), "response": chunk_text, "done": False})
            finally:
                yield _sse_event({"model": req.model or "qwen3-8b", "created": int(time.time()), "done": True})
        return StreamingResponse(_event_stream(), media_type="text/event-stream")
    except Exception as e:
        logger.exception("api_chat failed")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")
