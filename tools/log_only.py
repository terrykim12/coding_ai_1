#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로그 전용 실행기: 함수 스니펫만으로 /plan -> /patch 호출 (적용/테스트 없음)
성공 시 서버가 training/success_logs/*.jsonl에 자동 적재
"""
import json
import re
import argparse
from pathlib import Path
import requests


def get_snippet(path: str, func: str) -> str:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = -1
    for i, ln in enumerate(lines):
        if re.match(rf"^[\t ]*def\s+{re.escape(func)}\s*\(", ln):
            start = i
            break
    if start < 0:
        raise RuntimeError(f"스니펫 없음: {func} @ {path}")
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^\S", lines[j]):
            end = j
            break
    return "\n".join(lines[start:end])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--path", default="examples/sample_py/app.py")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    base = args.base
    path = args.path
    intents = [
        "add()에 입력 타입검사 추가 (int만 허용)",
        "divide()에 0 나누기 방지 로직과 명확한 에러 메시지 추가",
        "factorial()에 음수 입력 방지 및 ValueError 메시지 보강",
        "is_prime()에 2 미만 처리 보강 및 빠른 return",
        "fibonacci()에 n==1 처리 보강 및 입력검증 추가",
    ]
    func_by_intent = {
        "add": "add",
        "divide": "divide",
        "factorial": "factorial",
        "is_prime": "is_prime",
        "fibonacci": "fibonacci",
    }
    for _ in range(max(1, args.runs)):
        for intent in intents:
            func = next((v for k, v in func_by_intent.items() if k in intent), "add")
            try:
                snippet = get_snippet(path, func)
                plan_req = {"intent": intent, "paths": ["examples/sample_py"], "code_paste": snippet}
                r = requests.post(f"{base}/plan", json=plan_req, timeout=180)
                r.raise_for_status()
                plan = r.json()["plan"]
                patch_req = {"plan": plan}
                # 스트리밍 조기종료가 적용된 /patch_smart 사용
                r = requests.post(f"{base}/patch_smart", json=patch_req, timeout=300)
                r.raise_for_status()
                print("[OK]", intent)
            except Exception as e:
                print("[SKIP]", intent, ":", e)


if __name__ == "__main__":
    main()


