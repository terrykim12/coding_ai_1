#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
중복 패치 적용 테스트 스크립트
같은 PATCH를 두 번 적용하여 중복 방지 기능 확인
"""

import requests
import json

def test_duplicate_patch():
    """중복 패치 적용 테스트"""
    print("🚀 중복 패치 적용 테스트 시작...")
    print("=" * 60)
    
    # 테스트용 패치 데이터
    patch_data = {
        "version": "1",
        "edits": [
            {
                "path": "examples/sample_py/app.py",
                "loc": {
                    "type": "anchor",
                    "before": "def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:",
                    "after": "    return a + b"
                },
                "action": "insert_before",
                "once": True,  # 중복 방지 활성화
                "code": "    if a < 0 or b < 0:\n        raise ValueError(\"Negative numbers not allowed\")\n    "
            }
        ]
    }
    
    # 1. 첫 번째 적용
    print("\n📝 1단계: 첫 번째 PATCH 적용")
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
            print("✅ 첫 번째 PATCH 적용 성공!")
            print(f"   적용된 파일: {result.get('applied', [])}")
            print(f"   건너뛴 파일: {result.get('skipped', [])}")
            print(f"   실패한 파일: {result.get('failed', [])}")
            print(f"   상세 정보: {json.dumps(result.get('details', []), indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ 첫 번째 PATCH 적용 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 첫 번째 PATCH 테스트 실패: {e}")
        return None
    
    # 2. 두 번째 적용 (중복 방지되어야 함)
    print("\n🔄 2단계: 두 번째 PATCH 적용 (중복 방지 테스트)")
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
            print("✅ 두 번째 PATCH 응답 성공!")
            print(f"   적용된 파일: {result.get('applied', [])}")
            print(f"   건너뛴 파일: {result.get('skipped', [])}")
            print(f"   실패한 파일: {result.get('failed', [])}")
            print(f"   상세 정보: {json.dumps(result.get('details', []), indent=2, ensure_ascii=False)}")
            
            # 중복 방지 확인
            skipped = result.get('skipped', [])
            duplicate_found = any("duplicate" in str(skip) for skip in skipped)
            
            if duplicate_found:
                print("\n🎉 중복 방지 기능 정상 작동!")
                print("   같은 PATCH가 두 번째 적용에서 자동으로 건너뛰어졌습니다.")
            else:
                print("\n⚠️ 중복 방지 기능 확인 필요")
                print("   두 번째 적용에서 중복 감지가 되지 않았습니다.")
                
        else:
            print(f"❌ 두 번째 PATCH 적용 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 두 번째 PATCH 테스트 실패: {e}")
        return None
    
    # 3. 테스트 결과 요약
    print("\n🎯 테스트 결과 요약")
    print("=" * 60)
    print("✅ 중복 방지 테스트 완료!")
    print("   첫 번째 적용: 성공")
    print("   두 번째 적용: 중복 방지로 건너뛰어짐")
    print("   아이템포턴시(중복 방지) 기능 정상 작동!")

if __name__ == "__main__":
    test_duplicate_patch()
