#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 중복 패치 테스트
"""

import requests
import json

def test_simple_duplicate():
    """간단한 중복 패치 테스트"""
    print("🚀 간단한 중복 패치 테스트")
    print("=" * 50)
    
    # 매우 간단한 패치 데이터
    patch_data = {
        "version": "1",
        "edits": [
            {
                "path": "examples/sample_py/app.py",
                "loc": {
                    "type": "anchor",
                    "before": "def add\\(a: Union\\[int, float\\], b: Union\\[int, float\\]\\) -> Union\\[int, float\\]:",
                    "after": "return a + b"
                },
                "action": "insert_before",
                "once": True,
                "code": "    # 테스트 주석\n    "
            }
        ]
    }
    
    print("📋 테스트 패치:")
    print(f"   파일: {patch_data['edits'][0]['path']}")
    print(f"   액션: {patch_data['edits'][0]['action']}")
    print(f"   once: {patch_data['edits'][0]['once']}")
    
    # API 호출
    try:
        response = requests.post(
            "http://127.0.0.1:8765/apply",
            json={"patch": patch_data, "allowed_paths": ["examples/sample_py"], "dry_run": False},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ 응답 성공!")
            print(f"   응답 전체: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 상세 분석
            print(f"\n📊 응답 분석:")
            print(f"   applied: {len(result.get('applied', []))}개")
            print(f"   skipped: {len(result.get('skipped', []))}개")
            print(f"   failed: {len(result.get('failed', []))}개")
            print(f"   details: {len(result.get('details', []))}개")
            
        else:
            print(f"❌ 응답 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    test_simple_duplicate()
