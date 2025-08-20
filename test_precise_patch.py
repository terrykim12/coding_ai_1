#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정확한 위치 패치 적용 테스트 스크립트
정확한 앵커를 사용하여 패치를 올바른 위치에 적용
"""

import requests
import json

def test_precise_patch():
    """정확한 위치 패치 적용 테스트"""
    print("🚀 정확한 위치 패치 적용 테스트 시작...")
    print("=" * 60)
    
    # 더 정확한 앵커를 사용한 패치 데이터
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
    
    # 1. 정확한 위치에 패치 적용
    print("\n📝 1단계: 정확한 위치에 PATCH 적용")
    print("-" * 30)
    
    try:
        response = requests.post(
            "http://127.0.0.1:8765/apply",
            json={"patch": patch_data, "allowed_paths": ["examples/sample_py"], "dry_run": False},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ PATCH 적용 성공!")
            print(f"   적용된 파일: {result.get('applied', [])}")
            print(f"   건너뛴 파일: {result.get('skipped', [])}")
            print(f"   실패한 파일: {result.get('failed', [])}")
            print(f"   상세 정보: {json.dumps(result.get('details', []), indent=2, ensure_ascii=False)}")
            
            # 파일 상태 확인
            if result.get('applied'):
                print("\n📁 파일 상태 확인:")
                try:
                    with open("examples/sample_py/app.py", "r", encoding="utf-8") as f:
                        content = f.read()
                        if "if a < 0 or b < 0:" in content and "raise ValueError" in content:
                            print("   ✅ 음수 검증 코드가 정확한 위치에 추가되었습니다!")
                        else:
                            print("   ⚠️ 음수 검증 코드가 예상과 다른 위치에 추가되었습니다.")
                except Exception as e:
                    print(f"   ❌ 파일 읽기 실패: {e}")
                    
        else:
            print(f"❌ PATCH 적용 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ PATCH 테스트 실패: {e}")
        return None
    
    # 2. 중복 적용 테스트
    print("\n🔄 2단계: 동일 PATCH 중복 적용 테스트")
    print("-" * 30)
    
    try:
        response = requests.post(
            "http://127.0.0.1:8765/apply",
            json={"patch": patch_data, "allowed_paths": ["examples/sample_py"], "dry_run": False},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 중복 적용 응답 성공!")
            print(f"   적용된 파일: {result.get('applied', [])}")
            print(f"   건너뛴 파일: {result.get('skipped', [])}")
            print(f"   실패한 파일: {result.get('failed', [])}")
            
            # 중복 방지 확인
            if result.get('skipped'):
                print("\n🎉 중복 방지 기능 정상 작동!")
                print("   동일한 PATCH가 자동으로 건너뛰어졌습니다.")
            else:
                print("\n⚠️ 중복 방지 기능 확인 필요")
                
        else:
            print(f"❌ 중복 적용 테스트 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            
    except Exception as e:
        print(f"❌ 중복 적용 테스트 실패: {e}")
    
    print("\n🎯 테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    test_precise_patch()
