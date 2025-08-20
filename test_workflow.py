#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전체 워크플로우 테스트 스크립트
PLAN → PATCH → APPLY → TEST 순서로 진행
"""

import requests
import json
import time

def test_workflow():
    """전체 워크플로우 테스트"""
    print("🚀 전체 워크플로우 테스트 시작...")
    print("=" * 60)
    
    # 1. PLAN 단계
    print("\n📋 1단계: PLAN 생성")
    print("-" * 30)
    
    plan_data = {
        "intent": "add() 함수에 음수 입력 검증 추가",
        "paths": ["examples/sample_py"],
        "code_paste": "def add(a,b): return a+b"
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8765/plan",
            json=plan_data,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response.status_code == 200:
            plan_result = response.json()
            print("✅ PLAN 생성 성공!")
            print(f"   계획 ID: {plan_result.get('plan_id', 'N/A')}")
            print(f"   계획 내용: {json.dumps(plan_result.get('plan', {}), indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ PLAN 생성 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ PLAN 테스트 실패: {e}")
        return None
    
    # 2. PATCH 단계
    print("\n🔧 2단계: PATCH 생성")
    print("-" * 30)
    
    patch_data = {
        "plan": plan_result.get('plan')
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8765/patch",
            json=patch_data,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response.status_code == 200:
            patch_result = response.json()
            print("✅ PATCH 생성 성공!")
            print(f"   패치 ID: {patch_result.get('patch_id', 'N/A')}")
            print(f"   패치 내용: {json.dumps(patch_result.get('patch', {}), indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ PATCH 생성 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ PATCH 테스트 실패: {e}")
        return None
    
    # 3. APPLY 단계 (dry_run으로 먼저 테스트)
    print("\n📝 3단계: PATCH 적용 (dry_run)")
    print("-" * 30)
    
    apply_data = {
        "patch": patch_result.get('patch'),
        "allowed_paths": ["examples/sample_py"],
        "dry_run": True
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8765/apply",
            json=apply_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            apply_result = response.json()
            print("✅ PATCH 적용 (dry_run) 성공!")
            print(f"   적용될 파일: {apply_result.get('applied', [])}")
            print(f"   실패한 파일: {apply_result.get('failed', [])}")
            print(f"   Dry run: {apply_result.get('dry_run', False)}")
        else:
            print(f"❌ PATCH 적용 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ PATCH 적용 테스트 실패: {e}")
        return None
    
    # 4. 실제 적용 (dry_run=False)
    print("\n💾 4단계: PATCH 실제 적용")
    print("-" * 30)
    
    apply_data["dry_run"] = False
    
    try:
        response = requests.post(
            "http://127.0.0.1:8765/apply",
            json=apply_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            apply_result = response.json()
            print("✅ PATCH 실제 적용 성공!")
            print(f"   적용된 파일: {apply_result.get('applied', [])}")
            print(f"   실패한 파일: {apply_result.get('failed', [])}")
            print(f"   Dry run: {apply_result.get('dry_run', False)}")
        else:
            print(f"❌ PATCH 실제 적용 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ PATCH 실제 적용 테스트 실패: {e}")
        return None
    
    # 5. 테스트 실행
    print("\n🧪 5단계: 테스트 실행")
    print("-" * 30)
    
    test_data = {
        "paths": ["examples/sample_py"],
        "coverage": False
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8765/test",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            test_result = response.json()
            print("✅ 테스트 실행 성공!")
            print(f"   테스트 요약: {json.dumps(test_result.get('summary', {}), indent=2, ensure_ascii=False)}")
            print(f"   테스트 출력: {test_result.get('output', '')[:500]}...")
        else:
            print(f"❌ 테스트 실행 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 테스트 실행 실패: {e}")
        return None
    
    # 6. 워크플로우 요약
    print("\n🎉 워크플로우 완료!")
    print("=" * 60)
    print("✅ 모든 단계가 성공적으로 완료되었습니다!")
    print(f"   PLAN ID: {plan_result.get('plan_id', 'N/A')}")
    print(f"   PATCH ID: {patch_result.get('patch_id', 'N/A')}")
    print(f"   적용된 파일: {apply_result.get('applied', [])}")
    print(f"   테스트 결과: {test_result.get('summary', {}).get('passed', 0)}개 통과")
    
    return {
        "plan": plan_result,
        "patch": patch_result,
        "apply": apply_result,
        "test": test_result
    }

if __name__ == "__main__":
    test_workflow()
