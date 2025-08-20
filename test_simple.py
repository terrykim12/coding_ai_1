#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 API 테스트 스크립트
"""

import requests
import json

def test_plan():
    """PLAN 단계 간단 테스트"""
    print("🚀 PLAN 단계 테스트...")
    
    data = {
        "intent": "Add input validation to add function",
        "paths": ["examples/sample_py"],
        "code_paste": "def add(a,b): return a+b"
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8765/plan",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ PLAN 생성 성공!")
            print(f"   응답 시간: 성공")
            print(f"   계획 ID: {result.get('plan_id', 'N/A')}")
            print(f"   계획 내용: {json.dumps(result.get('plan', {}), indent=2, ensure_ascii=False)}")
            return result
        else:
            print(f"❌ PLAN 생성 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ PLAN 테스트 실패: {e}")
        return None

if __name__ == "__main__":
    test_plan()
