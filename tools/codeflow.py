#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTTP 클라이언트 (스니펫 기반): /plan -> /patch_smart -> /apply -> /test 순서 실행
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any

import requests


def get_function_snippet(path: str, func_name: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    # def NAME(...): 시그니처 끝 콜론까지(반환타입 주석 허용) → 본문 ~ 다음 톱레벨 또는 파일 끝
    pat = re.compile(
        rf"(?ms)^[\t ]*def\s+{re.escape(func_name)}\([^\n]*\)[^\n]*:[\s\S]*?(?=^\S|\Z)"
    )
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"함수 스니펫을 찾을 수 없습니다: {func_name} @ {path}")
    return m.group(0)


def invoke_code_edit(base_url: str, intent: str, file_path: str, func_name: str) -> Dict[str, Any]:
    project_root = Path.cwd()
    abs_path = (project_root / file_path).resolve()
    rel_path = str(abs_path.relative_to(project_root)).replace("\\", "/")

    code_paste = get_function_snippet(str(abs_path), func_name)

    # PLAN
    plan_req = {"intent": intent, "paths": [rel_path], "code_paste": code_paste}
    r = requests.post(f"{base_url}/plan", json=plan_req, timeout=180)
    r.raise_for_status()
    plan_resp = r.json()
    plan = plan_resp["plan"]

    # PATCH_SMART
    patch_req = {"plan": plan}
    r = requests.post(f"{base_url}/patch_smart", json=patch_req, timeout=240)
    if r.status_code >= 400:
        return {"error": f"patch_smart {r.status_code}", "detail": r.text}
    patch_resp = r.json()
    patch = patch_resp["patch"]
    edits = patch.get("edits", [])

    # APPLY (dry-run -> real)
    allowed = [str(Path(rel_path).parent).replace("\\", "/")]
    apply_req = {"patch": patch, "allowed_paths": allowed, "dry_run": True}
    r = requests.post(f"{base_url}/apply", json=apply_req, timeout=180)
    r.raise_for_status()
    apply_dry = r.json()

    apply_req["dry_run"] = False
    r = requests.post(f"{base_url}/apply", json=apply_req, timeout=180)
    r.raise_for_status()
    apply_real = r.json()

    # TEST
    test_req = {"paths": [str(Path(rel_path).parent).replace("\\", "/")]}
    r = requests.post(f"{base_url}/test", json=test_req, timeout=300)
    r.raise_for_status()
    test_resp = r.json()

    return {
        "plan_id": plan_resp.get("plan_id"),
        "edits": len(edits),
        "apply_dry": apply_dry,
        "apply_real": apply_real,
        "test": test_resp.get("summary", {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8765")
    ap.add_argument("--intent", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--function", required=True)
    args = ap.parse_args()

    out = invoke_code_edit(args.base_url, args.intent, args.path, args.function)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()


