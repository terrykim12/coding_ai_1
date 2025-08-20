#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edit_id 계산 디버깅 스크립트
"""

import json
import hashlib

def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()

def _edit_id(edit: dict) -> str:
    # path + loc + action + code 기반 안정적 해시
    key = json.dumps({
        "path": edit.get("path"),
        "loc": edit.get("loc"),
        "action": edit.get("action"),
        "code": edit.get("code", "")
    }, sort_keys=True, ensure_ascii=False)
    return _sha(key)[:16]

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

print("🔍 edit_id 계산 디버깅")
print("=" * 50)

edit = patch_data["edits"][0]
edit_id = _edit_id(edit)

print(f"📁 파일 경로: {edit['path']}")
print(f"🔧 액션: {edit['action']}")
print(f"📍 위치 타입: {edit['loc']['type']}")
print(f"📝 코드 길이: {len(edit['code'])}자")
print(f"🆔 계산된 edit_id: {edit_id}")

# JSON 키 확인
key = json.dumps({
    "path": edit.get("path"),
    "loc": edit.get("loc"),
    "action": edit.get("action"),
    "code": edit.get("code", "")
}, sort_keys=True, ensure_ascii=False)

print(f"\n🔑 JSON 키 (정렬됨):")
print(f"   길이: {len(key)}자")
print(f"   해시: {_sha(key)}")
print(f"   edit_id: {_sha(key)[:16]}")

# 여러 번 계산해보기
print(f"\n🔄 여러 번 계산 테스트:")
for i in range(3):
    test_id = _edit_id(edit)
    print(f"   {i+1}번째: {test_id} {'✅' if test_id == edit_id else '❌'}")
