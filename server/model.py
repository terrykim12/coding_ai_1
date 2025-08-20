#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-8B 모델 래퍼 - 프리픽스 강제 + 타임아웃 최적화
"""

import os
import json
import torch
import re
import time
import ast
from typing import Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from peft import PeftModel
import logging

# --- PLAN 파싱 유틸리티 ---
_OBJ_HINT = re.compile(r'\"files\"\s*:\s*\[', re.S)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.M)

def _strip_fences(s: str) -> str:
    # ```json ... ``` / ``` ...``` 제거
    return _FENCE_RE.sub("", s).strip()

def _balanced_json_object(s: str) -> str | None:
    """문자열 내에서 가장 그럴듯한 JSON 오브젝트({ ... })를 균형 찾아 반환(문자열 내부 제외)."""
    s = s.strip()
    start = s.find("{")
    if start == -1: return None
    in_str = False; esc = False; depth = 0; beg = None
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == "\"": in_str = False
            continue
        else:
            if ch == "\"": in_str = True
            elif ch == "{":
                depth += 1
                if beg is None: beg = i
            elif ch == "}":
                depth -= 1
                if depth == 0 and beg is not None:
                    return s[beg:i+1]
    return None

def _json_fix_attempts(text: str) -> dict | None:
    """
    1) json.loads
    2) 코드펜스 제거 후 json.loads
    3) 'files' 힌트 포함된 균형 객체 추출 후 json.loads
    4) 흔한 실수 보정(싱글쿼트->더블, True/False/None 소문자) 후 ast.literal_eval
    """
    # 1
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2
    t2 = _strip_fences(text)
    try:
        return json.loads(t2)
    except Exception:
        pass
    # 3
    obj = _balanced_json_object(t2)
    if obj and _OBJ_HINT.search(obj):
        try:
            return json.loads(obj)
        except Exception:
            pass
    # 4 (관용 파서)
    t3 = t2.strip()
    if t3.startswith("{") and t3.endswith("}"):
        t4 = (t3
              .replace(""","\"").replace(""","\"").replace("'","'")
              .replace("\t"," ")
        )
        # 구조용 싱글쿼트 → 더블쿼트 (매우 보수적으로: 키/문자열 경계만)
        t4 = re.sub(r"(?<!\\)'", "\"", t4)  # 간단 치환 (완전치 않지만 SFT 출력엔 보통 충분)
        t4 = re.sub(r"\bTrue\b", "true", t4)
        t4 = re.sub(r"\bFalse\b", "false", t4)
        t4 = re.sub(r"\bNone\b", "null", t4)
        # trailing comma
        t4 = re.sub(r",\s*([}\]])", r"\1", t4)
        try:
            return json.loads(t4)
        except Exception:
            try:
                # 최후: 파이썬 dict 허용 파서
                return ast.literal_eval(t4)
            except Exception:
                pass
    return None

def _validate_plan(obj: dict) -> dict:
    """필수 키만 검증/보정. 불완전하면 최소 형태로 보정."""
    plan = {"files": [], "notes": ""}
    if isinstance(obj, dict):
        if isinstance(obj.get("files"), list):
            files = []
            for f in obj["files"]:
                if not isinstance(f, dict): continue
                path = f.get("path") or f.get("file") or f.get("target")
                if not path: continue
                files.append({
                    "path": str(path).replace("\\", "/"),
                    "reason": str(f.get("reason", "")),
                    "strategy": f.get("strategy") or "regex",
                    "tests": f.get("tests") if isinstance(f.get("tests"), list) else []
                })
            plan["files"] = files
        if "notes" in obj and isinstance(obj["notes"], str):
            plan["notes"] = obj["notes"]
    return plan

# PATCH 파싱용 정규식
EDITS_OBJ_RE = re.compile(r'"edits"\s*:\s*(\[[\s\S]*?\])')

logger = logging.getLogger(__name__)

# 마커 상수
START_PLAN = "<<<PLAN_JSON>>>"
START_PATCH = "<<<PATCH_JSON>>>"
END_MARK = "<<<END>>>"

class StopOnTime(StoppingCriteria):
    """시간 예산에 따른 조기 종료"""
    def __init__(self, deadline_s: float):
        self.deadline_s = deadline_s
    
    def __call__(self, input_ids, scores, **kwargs):
        return time.perf_counter() >= self.deadline_s

class Model:
    def __init__(self, model_path: str = None, adapter_path: str = None):
        # 환경변수 이름 호환: MODEL_PATH가 없으면 QWEN_BASE_MODEL 사용
        env_model = os.getenv("MODEL_PATH") or os.getenv("QWEN_BASE_MODEL")
        self.model_path = model_path or env_model or "Qwen/Qwen3-8B"

        self.adapter_path = adapter_path or os.getenv("ADAPTER_PATH", "training/qlora-out/adapter")
        self.device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = os.getenv("TORCH_DTYPE", "bfloat16" if torch.cuda.is_available() else "float16")

        # 4bit 사용 여부 (기본: 환경변수 QWEN_4BIT, Windows에서는 기본적으로 비활성화 권장)
        def _env_bool(name: str, default: bool = False) -> bool:
            v = os.getenv(name)
            if v is None:
                return default
            return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

        self.use_4bit = _env_bool("QWEN_4BIT", default=False)
        force_4bit = _env_bool("QWEN_FORCE_4BIT", default=False)
        if os.name == "nt" and self.use_4bit and not force_4bit:
            # Windows + bitsandbytes는 불안정 → 기본 비활성화. 강제 사용 시 QWEN_FORCE_4BIT=true 지정
            logger.warning("Windows 환경: 기본적으로 4-bit 비활성화(QWEN_4BIT=false). 강제 사용은 QWEN_FORCE_4BIT=true")
            self.use_4bit = False

        logger.info(f"모델 로딩 시작: {self.model_path}")
        logger.info(f"디바이스: {self.device}")
        logger.info(f"데이터 타입: {self.torch_dtype}")
        logger.info(f"4-bit 양자화 사용: {self.use_4bit}")

        self._load_model()
        logger.info("✅ 모델 로딩 완료!")

    def _load_model(self):
        """모델 및 토크나이저 로딩"""
        try:
            # 토크나이저 로딩
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                trust_remote_code=True
            )
            
            # EOS 토큰 설정
            if self.tokenizer.eos_token is None:
                self.tokenizer.eos_token = "<|endoftext|>"
            
            # 양자화/비양자화 설정
            if self.device == "cuda" and self.use_4bit:
                try:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=getattr(torch, self.torch_dtype)
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        quantization_config=bnb_config,
                        device_map={"": "cuda:0"},
                        trust_remote_code=True,
                        torch_dtype=getattr(torch, self.torch_dtype),
                        attn_implementation="sdpa"
                    )
                except Exception as qe:
                    logger.error(f"4-bit 로딩 실패, 비양자화로 폴백합니다: {qe}")
                    self.use_4bit = False
                    # 폴백: 비양자화 로딩
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        torch_dtype=getattr(torch, self.torch_dtype),
                        device_map={"": "cuda:0"},
                        attn_implementation="sdpa"
                    )
            else:
                # CPU 모드
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=getattr(torch, self.torch_dtype),
                    device_map={"": self.device if self.device == "cuda" else "cpu"},
                    attn_implementation="sdpa"
                )
            
            # LoRA 어댑터 로딩 (있는 경우)
            if self.adapter_path and os.path.exists(self.adapter_path):
                logger.info(f"LoRA 어댑터 로딩: {self.adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            raise

    def _gen_plain(self, prompt: str, max_new_tokens=384):
        """순수 문자열 프롬프트 기반 생성 (그리디 + 조기 종료)"""
        t0 = time.perf_counter()

        # ★★ 핵심: '<<<PLAN_JSON>>>{' 까지를 "입력 토큰"에 미리 넣는다.
        prefix = prompt + START_PLAN + "{"
        inputs = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        with torch.inference_mode():
            # 샘플링 파라미터 제거하여 경고 방지
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            out = self.model.generate(**inputs, **gen_kwargs)

        full = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # prefix 제거 → 모델이 생성한 부분만 추출
        assert full.startswith(prompt), "prefix mismatch"
        gen = full[len(prompt):]

        # <<<PLAN_JSON>>> 이후만 취함
        if START_PLAN in gen:
            gen = gen.split(START_PLAN, 1)[1]
        # END_MARK 나오면 거기서 컷
        if END_MARK in gen:
            gen = gen.split(END_MARK, 1)[0]

        # 맨 앞에 '{'는 우리가 시드했으므로 그대로 유지됨
        elapsed = time.perf_counter() - t0
        logger.info(f"생성 완료: {elapsed:.2f}초, 생성 길이: {len(gen)}자")
        
        return gen

    def _gen_plan(self, context_str: str, intent: str, max_new_tokens=192, time_budget_s=70):
        """시간 예산이 적용된 PLAN 생성"""
        t0 = time.perf_counter()
        
        # 프롬프트 + 프리픽스 강제
        prompt = (
            "You are a senior engineer. Output ONLY STRICT JSON plan. No markdown, no prose.\n"
            "Schema: {\"files\":[{\"path\":\"...\",\"reason\":\"...\",\"strategy\":\"anchor|range|regex|ast\",\"tests\":[\"...\"]}],\"notes\":\"...\"}\n"
            f"Begin immediately after {START_PLAN} and end with {END_MARK}.\n"
            "\n[INTENT]\n" + intent + "\n\n[CONTEXT]\n" + context_str + "\n\n"
        )
        
        # ★★ 핵심: '<<<PLAN_JSON>>>{' 까지를 "입력 토큰"에 미리 넣는다.
        prefix = prompt + START_PLAN + "{"
        inputs = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        # 시간 예산 설정
        deadline = time.perf_counter() + time_budget_s
        stops = StoppingCriteriaList([StopOnTime(deadline)])

        with torch.inference_mode():
            # 샘플링 파라미터 제거하여 경고 방지
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.eos_token_id,
                "stopping_criteria": stops
            }
            out = self.model.generate(**inputs, **gen_kwargs)

        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        gen = full[len(prompt):]  # 모델이 생성한 부분만

        if START_PLAN in gen:
            gen = gen.split(START_PLAN, 1)[1]
        if END_MARK in gen:
            gen = gen.split(END_MARK, 1)[0]

        # 강화된 파서: 어떤 형태든 JSON 회수
        raw = gen  # 모델 출력 문자열
        obj = _json_fix_attempts(raw)
        if not isinstance(obj, (dict,)):
            # 2차 시도: 모델에 초미니 재프롬프트(형식만 강제)
            guide = "Return ONLY a JSON object with keys: files(list), notes(string). No code fences."
            raw2 = self.generate_plan_minimal(intent, context_str, guide)  # 내부 재호출 함수(그리디, 짧게)
            obj = _json_fix_attempts(raw2)

        plan = _validate_plan(obj or {})
        # 빈 계획이면 fallback (요청 paths 기준 최소 플랜)
        if not plan["files"] and context_str:
            # context_str에서 paths 추출 시도
            paths = []
            if "examples/sample_py" in context_str:
                paths = ["examples/sample_py"]
            plan["files"] = [{"path": f"{p.rstrip('/')}/app.py", "reason": "default", "strategy": "regex", "tests": []}
                             for p in paths]
        
        elapsed = time.perf_counter() - t0
        logger.info(f"PLAN 생성 완료: {elapsed:.2f}초, 생성 길이: {len(gen)}자")
        
        return plan

    def generate_plan_minimal(self, intent: str, context_str: str, guide: str) -> str:
        """최소한의 PLAN 생성 (재시도용)"""
        prompt = f"{guide}\n\n[INTENT]\n{intent}\n\n[CONTEXT]\n{context_str}\n\n"
        prefix = prompt + START_PLAN + "{"
        
        inputs = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        with torch.inference_mode():
            gen_kwargs = {
                "max_new_tokens": 64,  # 짧게
                "do_sample": False,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            out = self.model.generate(**inputs, **gen_kwargs)
        
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        gen = full[len(prompt):]
        
        if START_PLAN in gen:
            gen = gen.split(START_PLAN, 1)[1]
        if END_MARK in gen:
            gen = gen.split(END_MARK, 1)[0]
        
        return gen

    def plan(self, context_str: str, intent: str, code_paste: str = "") -> Dict[str, Any]:
        """코드 수정 계획 생성 - 시간 예산 적용"""
        logger.info("PLAN 생성 시작...")
        return self._gen_plan(context_str, intent, max_new_tokens=192, time_budget_s=70)

    def patch(self, plan_json: Dict[str, Any], feedback: str = "") -> Dict[str, Any]:
        """
        모델에게 'edits 배열'만 생성시키고, 바깥 {version, edits} 래핑은 코드에서 수행.
        """
        logger.info("PATCH 생성 시작...")
        
        system = (
            "Output ONLY a JSON array named 'edits' (no markdown, no prose). "
            "Each item: {'path':str,'loc':{'type':'regex|anchor|range|ast',...},"
            "'action':'insert_before|insert_after|insert_after_block|replace_range|delete_range',"
            "'code':str,'once':true,'pre':{'must_contain':[...],'must_not_contain':[...]}}. "
            "Prefer function-level regex anchors or AST for Python. Keep patch minimal. "
            "All path values must be relative to the repository root and use forward slashes (/). "
            "Do NOT use absolute paths (no /home/... or C:\\...). "
            "For 'pre', treat patterns as LITERAL substrings. Use {\"regex\": true} only when you truly intend a regex. "
            "IMPORTANT: If you put a pattern in pre.must_contain, it MUST exist in the current file (BEFORE patch). "
            "If you put a pattern in pre.must_not_contain, it should NOT exist in the current file. "
            "Use pre.must_not_contain for new patterns you want to prevent from being duplicated. "
            "Return only the array items of edits (no outer object). Example: {...} is NOT allowed; write only objects separated by commas."
        )
        user = (
            "[PLAN]\n" + json.dumps(plan_json, ensure_ascii=False) +
            (("\n[FEEDBACK]\n" + feedback) if feedback else "") +
            "\n[INSTRUCTIONS]\n"
            "1) Return ONLY the JSON array of edits.\n"
            "2) Use double quotes only.\n"
            "3) Ensure valid JSON.\n"
            "4) Do not wrap with outer object.\n"
            "Begin immediately after the marker and end with <<<END>>>."
        )

        # ★ 프리픽스 강제: 외부 오브젝트까지는 우리가 제공한다.
        #   모델은 오직 배열 본문만 이어서 쓴다.
        prefix = (
            system + "\n\n" + user + "\n" +
            START_PATCH + "{\"version\":\"1\",\"edits\":"  # 여기까지는 고정
            "["                                          # 배열 여는 대괄호까지 강제
        )

        raw = self._gen_with_prefix(prefix, max_new_tokens=768, time_budget_s=90)

        # 센티넬 컷
        if END_MARK in raw:
            raw = raw.split(END_MARK, 1)[0]

        # 강화된 파서로 edits 배열 추출 (어떤 형태든 호환)
        raw_edits = self._extract_edits_any(raw)
        if not isinstance(raw_edits, list):
            # 파싱 실패 시 로깅으로 실제 출력 형태 확인
            logger.warning(f"PATCH 파싱 실패, raw[:400]: {raw[:400]}")
            raise ValueError("PATCH: JSON parse failed")

        # 문자열 → dict 코어싱
        def _coerce_edit_item(x):
            if isinstance(x, dict):
                return x
            if isinstance(x, str):
                # 코드펜스 제거
                t = x.strip().strip(",")
                # 균형 객체 추출
                i = t.find("{")
                if i >= 0:
                    depth = 0; j0 = None; in_str = False; esc = False
                    for j, ch in enumerate(t[i:], start=i):
                        if in_str:
                            if esc: esc = False
                            elif ch == "\\": esc = True
                            elif ch == '"': in_str = False
                            continue
                        if ch == '"': in_str = True
                        elif ch == "{":
                            depth += 1
                            if j0 is None: j0 = j
                        elif ch == "}":
                            depth -= 1
                            if depth == 0 and j0 is not None:
                                try:
                                    return json.loads(t[j0:j+1])
                                except Exception:
                                    pass
                                break
            return None

        coerced = []
        for it in raw_edits:
            ed = _coerce_edit_item(it)
            if ed is not None:
                coerced.append(ed)
        
        if not coerced:
            raise ValueError("PATCH: JSON parse failed (no valid edit objects)")
        
        # 최종 래핑
        return {"version": "1", "edits": coerced}

    def _extract_edits_any(self, s: str):
        """어떤 형태든 edits 배열만 뽑기 - SFT 전/후 모두 호환"""
        # A) {"version":"1","edits":[...]} 형태에서 'edits' 값만 추출
        m = EDITS_OBJ_RE.search(s)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        # B) 본문 어딘가의 첫 번째 '[' ~ 매칭되는 ']' 까지 균형 추출
        i = s.find("[")
        if i != -1:
            depth = 0; start = None
            for j, ch in enumerate(s[i:], start=i):
                if ch == "[":
                    depth += 1
                    if start is None: start = j
                elif ch == "]":
                    depth -= 1
                    if depth == 0 and start is not None:
                        frag = s[start:j+1]
                        try:
                            return json.loads(frag)
                        except Exception:
                            break

        # C) 아이템만 온 경우(대괄호가 없는 경우) → 강제로 [ ... ] 감싸 보기
        body = s.strip()
        if "[" not in body and "]" not in body and body:
            try:
                return json.loads("[" + body + "]")
            except Exception:
                pass
        return None

    def _extract_balanced_array(self, s: str):
        """s 안에서 첫 '['부터 대괄호 균형이 맞는 리스트 JSON을 파싱 (기존 호환성)"""
        i = s.find("[")
        if i == -1:
            return None
        depth = 0; start = None
        for j, ch in enumerate(s[i:], start=i):
            if ch == "[":
                depth += 1
                if start is None:
                    start = j
            elif ch == "]":
                depth -= 1
                if depth == 0 and start is not None:
                    frag = s[start:j+1]
                    try:
                        return json.loads(frag)
                    except Exception:
                        pass
        return None

    def _gen_with_prefix(self, prefix: str, max_new_tokens=512, time_budget_s=70):
        """prefix(입력) 뒤부터 이어쓰기. 그리디+시간예산+센티넬 컷."""
        inputs = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        deadline = time.perf_counter() + time_budget_s

        class StopOnTime:
            def __call__(self, input_ids, scores=None):
                return time.perf_counter() >= deadline

        with torch.inference_mode():
            # 샘플링 파라미터 제거하여 경고 방지
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.eos_token_id,
                "stopping_criteria": StoppingCriteriaList([StopOnTime()])
            }
            out = self.model.generate(**inputs, **gen_kwargs)
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return full[len(prefix):]  # suffix만 반환

    def _balanced_json(self, s: str):
        """균형 중괄호 블록을 복구 (기존 호환성 유지)"""
        stack = 0
        start = None
        for i, ch in enumerate(s):
            if ch == '{':
                if stack == 0: 
                    start = i
                stack += 1
            elif ch == '}':
                if stack > 0:
                    stack -= 1
                    if stack == 0 and start is not None:
                        frag = s[start:i+1]
                        try:
                            return json.loads(frag)
                        except Exception:
                            # 계속 탐색
                            pass
        return None

    def is_loaded(self) -> bool:
        """모델 로딩 상태 확인"""
        return self.model is not None and self.tokenizer is not None
    
    def get_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 반환"""
        info = {
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "model_loaded": self.is_loaded()
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.1f}GB",
                "gpu_memory_cached": f"{torch.cuda.memory_reserved(0) / 1024**3:.1f}GB"
            })
        
        return info

