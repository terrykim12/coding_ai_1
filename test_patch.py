#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATCH 단계 테스트 스크립트
"""

import requests
import json

def test_patch():
    """PATCH 단계 테스트"""
    print("🚀 PATCH 단계 테스트...")
    
    # PLAN 결과 (이전 테스트에서 얻은 것)
    plan_data = {
        "files": [
            {
                "path": "examples/sample_py\\app.py",
                "reason": "Add input validation to add function",
                "strategy": "range",
                "tests": [
                    "TestBasicOperations.test_add_negative",
                    "TestBasicOperations.test_add_float"
                ]
            }
        ],
        "notes": "Add input validation to prevent errors with invalid inputs."
    }
    
    data = {
        "plan": plan_data
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8765/patch",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ PATCH 생성 성공!")
            print(f"   패치 ID: {result.get('patch_id', 'N/A')}")
            print(f"   패치 내용: {json.dumps(result.get('patch', {}), indent=2, ensure_ascii=False)}")
            return result
        else:
            print(f"❌ PATCH 생성 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ PATCH 테스트 실패: {e}")
        return None

if __name__ == "__main__":
    test_patch()
