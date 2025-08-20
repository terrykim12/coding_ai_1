#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
중복 감지 로직 직접 디버깅 스크립트
"""

import json
import hashlib
import os

def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()

def _file_sha(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return _sha(content)

def _edit_id(edit: dict) -> str:
    key = json.dumps({
        "path": edit.get("path"),
        "loc": edit.get("loc"),
        "action": edit.get("action"),
        "code": edit.get("code", "")
    }, sort_keys=True, ensure_ascii=False)
    return _sha(key)[:16]

def _load_ledger() -> dict:
    ledger_path = ".llm_patch/ledger.json"
    if os.path.exists(ledger_path):
        try:
            return json.load(open(ledger_path, "r", encoding="utf-8"))
        except Exception:
            pass
    return {"applied": {}}

# 테스트용 패치 데이터
patch_data = {
    "version": "1",
    "edits": [
        {
            "path": "examples/sample_py/app.py",
            "loc": {
                "type": "regex",
                "pattern": r"def add\(a: Union\[int, float\], b: Union\[int, float\]\) -> Union\[int, float\]:\s*\n\s*\"\"\"\s*\n\s*두 숫자를 더합니다\.\s*\n\s*Args:\s*\n\s*a: 첫 번째 숫자\s*\n\s*b: 두 번째 숫자\s*\n\s*Returns:\s*\n\s*두 숫자의 합\s*\n\s*Note: 이 함수에는 버그가 있습니다 - 음수 검증이 누락되어 있습니다\.\s*\n\s*\"\"\"\s*\n\s*# 버그: 음수 검증 누락\s*\n\s*return a \+ b"
            },
            "action": "replace_range",
            "once": True,
            "code": """def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    \"\"\"
    두 숫자를 더합니다.
    
    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자
        
    Returns:
        두 숫자의 합
        
    Note: 이 함수에는 버그가 있습니다 - 음수 검증이 누락되어 있습니다.
    \"\"\"
    # 버그: 음수 검증 누락
    if a < 0 or b < 0:
        raise ValueError(\"Negative numbers not allowed\")
    return a + b"""
        }
    ]
}

print("🔍 중복 감지 로직 직접 디버깅")
print("=" * 60)

# 1. Ledger 로드
ledger = _load_ledger()
print("📁 Ledger 내용:")
print(json.dumps(ledger, indent=2, ensure_ascii=False))

# 2. 현재 파일 SHA 계산
file_path = "examples/sample_py/app.py"
current_sha = _file_sha(file_path)
print(f"\n📄 현재 파일 SHA: {current_sha}")

# 3. edit_id 계산
edit = patch_data["edits"][0]
edit_id = _edit_id(edit)
print(f"🆔 계산된 edit_id: {edit_id}")

# 4. 중복 감지 로직 테스트
print(f"\n🔍 중복 감지 로직 테스트:")
print(f"   path: {file_path}")
print(f"   path in ledger['applied']: {file_path in ledger['applied']}")
if file_path in ledger['applied']:
    print(f"   edit_id in ledger['applied'][path]: {edit_id in ledger['applied'][file_path]}")
    if edit_id in ledger['applied'][file_path]:
        ledger_sha = ledger['applied'][file_path][edit_id]
        print(f"   ledger SHA: {ledger_sha}")
        print(f"   current SHA: {current_sha}")
        print(f"   SHA 일치: {ledger_sha == current_sha}")

# 5. 중복 감지 조건 확인
once = edit.get("once", True)
print(f"\n✅ 중복 감지 조건:")
print(f"   once: {once}")
print(f"   path in ledger: {file_path in ledger['applied']}")
print(f"   edit_id in ledger[path]: {edit_id in ledger['applied'].get(file_path, {})}")

if once and file_path in ledger['applied'] and edit_id in ledger['applied'][file_path]:
    print("   🎉 중복 감지되어야 함!")
else:
    print("   ⚠️ 중복 감지되지 않음")
    if not once:
        print("      - once가 False")
    if file_path not in ledger['applied']:
        print("      - path가 ledger에 없음")
    if edit_id not in ledger['applied'].get(file_path, {}):
        print("      - edit_id가 ledger[path]에 없음")
