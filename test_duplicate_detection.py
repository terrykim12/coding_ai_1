#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
중복 방지 기능 상세 테스트 스크립트
정확히 동일한 패치를 다시 적용하여 중복 감지 확인
"""

import requests
import json

def test_duplicate_detection():
    """중복 방지 기능 상세 테스트"""
    print("🚀 중복 방지 기능 상세 테스트 시작...")
    print("=" * 60)
    
    # 현재 파일에 이미 적용된 패치와 정확히 동일한 데이터
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
    
    print("📋 테스트 패치 데이터:")
    print(f"   파일: {patch_data['edits'][0]['path']}")
    print(f"   액션: {patch_data['edits'][0]['action']}")
    print(f"   once: {patch_data['edits'][0]['once']}")
    print(f"   코드 길이: {len(patch_data['edits'][0]['code'])}자")
    
    # 1. 동일 패치 재적용 시도
    print("\n🔄 동일 패치 재적용 시도 (중복 방지 테스트)")
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
            print("✅ 응답 성공!")
            print(f"   응답 전체: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 상세 분석
            applied = result.get('applied', [])
            skipped = result.get('skipped', [])
            failed = result.get('failed', [])
            details = result.get('details', [])
            
            print(f"\n📊 응답 분석:")
            print(f"   적용된 파일: {len(applied)}개 - {applied}")
            print(f"   건너뛴 파일: {len(skipped)}개 - {skipped}")
            print(f"   실패한 파일: {len(failed)}개 - {failed}")
            print(f"   상세 정보: {len(details)}개 - {details}")
            
            # 중복 방지 확인
            if skipped:
                print("\n🎉 중복 방지 기능 정상 작동!")
                for skip in skipped:
                    if isinstance(skip, dict):
                        print(f"   건너뛴 이유: {skip.get('reason', 'N/A')}")
                        print(f"   파일 경로: {skip.get('path', 'N/A')}")
            else:
                print("\n⚠️ 중복 방지 기능 확인 필요")
                print("   건너뛴 파일이 없습니다.")
                
        else:
            print(f"❌ 응답 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
    
    # 2. Ledger 파일 확인
    print("\n📁 Ledger 파일 상태 확인")
    print("-" * 30)
    
    ledger_path = ".llm_patch/ledger.json"
    try:
        if os.path.exists(ledger_path):
            with open(ledger_path, 'r', encoding='utf-8') as f:
                ledger = json.load(f)
            print("✅ Ledger 파일 존재")
            print(f"   내용: {json.dumps(ledger, indent=2, ensure_ascii=False)}")
        else:
            print("⚠️ Ledger 파일이 존재하지 않습니다.")
    except Exception as e:
        print(f"❌ Ledger 파일 읽기 실패: {e}")
    
    print("\n🎯 테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    import os
    test_duplicate_detection()
