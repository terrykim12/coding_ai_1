#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-8B Local Coding AI API 테스트 스크립트
Python으로 실행하세요
"""

import requests
import json
import sys

def test_api():
    base_url = "http://127.0.0.1:8765"
    
    print("🚀 Qwen3-8B Local Coding AI API 테스트 시작")
    print("=" * 50)
    
    # 1. 서버 상태 확인
    print("\n1️⃣ 서버 상태 확인...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 서버 상태: {data.get('status', 'unknown')}")
            print(f"   모델 로드: {data.get('model_loaded', 'unknown')}")
            print(f"   디버그 프로세스: {data.get('debug_processes', 'unknown')}")
        else:
            print(f"❌ 서버 응답 오류: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        return
    
    # 2. PLAN 단계 테스트
    print("\n2️⃣ PLAN 단계 테스트...")
    plan_data = {
        "intent": "Add negative validation to add() function",
        "paths": ["examples/sample_py"],
        "code_paste": "def add(a,b): return a+b"
    }
    
    try:
        response = requests.post(
            f"{base_url}/plan",
            json=plan_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            plan_result = response.json()
            print("✅ PLAN 생성 성공!")
            print(f"   계획 ID: {plan_result.get('plan_id', 'N/A')}")
            print(f"   계획 내용: {json.dumps(plan_result.get('plan', {}), indent=2, ensure_ascii=False)}")
            
            # 3. PATCH 단계 테스트
            print("\n3️⃣ PATCH 단계 테스트...")
            patch_data = {"plan": plan_result.get('plan', {})}
            
            patch_response = requests.post(
                f"{base_url}/patch",
                json=patch_data,
                headers={"Content-Type": "application/json"}
            )
            
            if patch_response.status_code == 200:
                patch_result = patch_response.json()
                print("✅ PATCH 생성 성공!")
                print(f"   패치 ID: {patch_result.get('patch_id', 'N/A')}")
                print(f"   패치 내용: {json.dumps(patch_result.get('patch', {}), indent=2, ensure_ascii=False)}")
                
                # 4. APPLY 단계 테스트 (DRY RUN)
                print("\n4️⃣ APPLY 단계 테스트 (DRY RUN)...")
                apply_data = {
                    "patch": patch_result.get('patch', {}),
                    "allowed_paths": ["examples/sample_py"],
                    "dry_run": True
                }
                
                apply_response = requests.post(
                    f"{base_url}/apply",
                    json=apply_data,
                    headers={"Content-Type": "application/json"}
                )
                
                if apply_response.status_code == 200:
                    apply_result = apply_response.json()
                    print("✅ 패치 적용 성공 (DRY RUN)!")
                    print(f"   적용된 파일: {apply_result.get('applied', [])}")
                    if apply_result.get('failed'):
                        print(f"   실패한 파일: {apply_result.get('failed', [])}")
                else:
                    print(f"❌ 패치 적용 실패: {apply_response.status_code}")
                    print(f"   오류: {apply_response.text}")
            else:
                print(f"❌ PATCH 생성 실패: {patch_response.status_code}")
                print(f"   오류: {patch_response.text}")
        else:
            print(f"❌ PLAN 생성 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            
    except Exception as e:
        print(f"❌ API 호출 중 오류 발생: {e}")
    
    # 5. TEST 단계 테스트
    print("\n5️⃣ TEST 단계 테스트...")
    try:
        test_response = requests.post(
            f"{base_url}/test",
            json={},
            headers={"Content-Type": "application/json"}
        )
        
        if test_response.status_code == 200:
            test_result = test_response.json()
            print("✅ 테스트 실행 성공!")
            summary = test_result.get('summary', {})
            print(f"   상태: {summary.get('status', 'N/A')}")
            print(f"   총 테스트: {summary.get('total_tests', 'N/A')}")
            print(f"   통과: {summary.get('passed', 'N/A')}")
            print(f"   실패: {summary.get('failed', 'N/A')}")
            print(f"   에러: {summary.get('errors', 'N/A')}")
        else:
            print(f"❌ 테스트 실행 실패: {test_response.status_code}")
            print(f"   오류: {test_response.text}")
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
    
    print("\n🎉 API 테스트 완료!")

if __name__ == "__main__":
    test_api()

