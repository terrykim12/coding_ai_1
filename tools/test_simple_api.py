#!/usr/bin/env python3
import requests
import json
import time

def test_plan_api():
    """PLAN API 테스트"""
    url = "http://127.0.0.1:8765/plan"
    
    # 간단한 요청
    data = {
        "intent": "add() 함수에 음수 방지 추가",
        "paths": ["examples/sample_py"],
        "code_paste": "def add(a, b):\n    return a + b"
    }
    
    try:
        print("PLAN API 호출 중...")
        response = requests.post(url, json=data, timeout=180)
        print(f"상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("성공!")
            print(f"PLAN ID: {result.get('plan_id')}")
            return result
        else:
            print(f"오류: {response.text}")
            return None
            
    except Exception as e:
        print(f"예외 발생: {e}")
        return None

def test_health():
    """Health API 테스트"""
    url = "http://127.0.0.1:8765/health"
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Health 상태: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.status_code == 200
    except Exception as e:
        print(f"Health 체크 실패: {e}")
        return False

if __name__ == "__main__":
    print("=== API 테스트 시작 ===")
    
    # Health 체크
    if not test_health():
        print("서버가 실행되지 않음")
        exit(1)
    
    # PLAN API 테스트
    plan_result = test_plan_api()
    
    if plan_result:
        print("=== PLAN API 성공 ===")
    else:
        print("=== PLAN API 실패 ===")
