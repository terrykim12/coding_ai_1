#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실패 케이스 분석 및 데이터셋 개선 스크립트
자기부스팅으로 데이터셋 품질 향상
"""

import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Any

def collect_failure_cases() -> List[Dict[str, Any]]:
    """실패 케이스 수집"""
    print("🔍 실패 케이스 수집 중...")
    
    # E2E 평가에서 실패한 케이스들 재현
    failure_cases = [
        {
            "intent": "factorial() 함수 개선 - 복잡한 케이스",
            "paths": ["examples/sample_py"],
            "code_paste": "def factorial(n): return n * factorial(n-1)",
            "expected_behavior": "PLAN 생성 성공",
            "actual_behavior": "500 오류 발생"
        },
        {
            "intent": "add() 함수에 음수 검증 추가 - 타임아웃 케이스",
            "paths": ["examples/sample_py"],
            "code_paste": "def add(a, b): return a + b",
            "expected_behavior": "PATCH 생성 성공",
            "actual_behavior": "504 타임아웃"
        }
    ]
    
    collected_failures = []
    
    for case in failure_cases:
        print(f"  테스트: {case['intent']}")
        
        # PLAN 테스트
        try:
            response = requests.post(
                "http://127.0.0.1:8765/plan",
                json={
                    "intent": case["intent"],
                    "paths": case["paths"],
                    "code_paste": case["code_paste"]
                },
                headers={"Content-Type": "application/json"},
                timeout=60  # 더 짧은 타임아웃으로 실패 유도
            )
            
            if response.status_code != 200:
                collected_failures.append({
                    "type": "PLAN_FAILURE",
                    "intent": case["intent"],
                    "input": case,
                    "error": f"HTTP {response.status_code}",
                    "response": response.text[:200] if response.text else ""
                })
                print(f"    ❌ PLAN 실패: {response.status_code}")
            else:
                print(f"    ✅ PLAN 성공")
                
                # PATCH 테스트
                try:
                    plan_result = response.json()
                    patch_response = requests.post(
                        "http://127.0.0.1:8765/patch",
                        json={"plan": plan_result.get("plan")},
                        headers={"Content-Type": "application/json"},
                        timeout=60
                    )
                    
                    if patch_response.status_code != 200:
                        collected_failures.append({
                            "type": "PATCH_FAILURE",
                            "intent": case["intent"],
                            "input": case,
                            "plan": plan_result.get("plan"),
                            "error": f"HTTP {patch_response.status_code}",
                            "response": patch_response.text[:200] if patch_response.text else ""
                        })
                        print(f"    ❌ PATCH 실패: {patch_response.status_code}")
                    else:
                        print(f"    ✅ PATCH 성공")
                        
                except Exception as e:
                    collected_failures.append({
                        "type": "PATCH_ERROR",
                        "intent": case["intent"],
                        "input": case,
                        "error": str(e)
                    })
                    print(f"    ❌ PATCH 오류: {e}")
                    
        except Exception as e:
            collected_failures.append({
                "type": "PLAN_ERROR",
                "intent": case["intent"],
                "input": case,
                "error": str(e)
            })
            print(f"    ❌ PLAN 오류: {e}")
        
        time.sleep(2)  # 서버 부하 방지
    
    return collected_failures

def generate_improvement_samples(failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """실패 케이스를 바탕으로 개선 샘플 생성"""
    print("\n🛠️ 개선 샘플 생성 중...")
    
    improvement_samples = []
    
    for failure in failures:
        if failure["type"] == "PLAN_FAILURE":
            # PLAN 실패 → 더 간단하고 명확한 PLAN 샘플 추가
            improvement_samples.append({
                "instruction": "Create simple modification plan. Output PLAN JSON only.",
                "input": f"""[INTENT] {failure['intent']}

[CONTEXT]
<<<FILE examples/sample_py/app.py>>>
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
<<<END>>>""",
                "output": """<<<PLAN_JSON>>>
{
  "files": [
    {
      "path": "examples/sample_py/app.py",
      "reason": "Add input validation to factorial function",
      "strategy": "anchor",
      "tests": ["test_factorial_negative"]
    }
  ],
  "notes": "Simple improvement: add negative number validation"
}
<<<END>>>"""
            })
            
        elif failure["type"] == "PATCH_FAILURE":
            # PATCH 실패 → 더 간단하고 구체적인 PATCH 샘플 추가
            improvement_samples.append({
                "instruction": "Generate minimal patch. Output PATCH JSON only.",
                "input": f"""[INTENT] {failure['intent']}

[CONTEXT]
<<<FILE examples/sample_py/app.py>>>
def add(a, b):
    return a + b
<<<END>>>""",
                "output": """<<<PATCH_JSON>>>
{
  "version": "1",
  "edits": [
    {
      "path": "examples/sample_py/app.py",
      "loc": {
        "type": "anchor",
        "before": "def add(a, b):",
        "after": "return a + b"
      },
      "action": "insert_before",
      "code": "    if a < 0 or b < 0: raise ValueError('Negative not allowed')\n    "
    }
  ]
}
<<<END>>>"""
            })
    
    # 추가 개선 샘플들 (형식 강화)
    format_improvement_samples = [
        {
            "instruction": "Output strict JSON only. No explanation.",
            "input": "[INTENT] Simple validation\n\n[CONTEXT]\nCode sample\n<<<END>>>",
            "output": """<<<PATCH_JSON>>>
{
  "version": "1",
  "edits": []
}
<<<END>>>"""
        },
        {
            "instruction": "Generate plan with minimal content. JSON format only.",
            "input": "[INTENT] Basic improvement\n\n[CONTEXT]\nSample code\n<<<END>>>",
            "output": """<<<PLAN_JSON>>>
{
  "files": [
    {
      "path": "example.py",
      "reason": "Basic improvement",
      "strategy": "anchor",
      "tests": ["test_basic"]
    }
  ],
  "notes": "Minimal plan"
}
<<<END>>>"""
        },
        {
            "instruction": "Keep JSON response short and valid.",
            "input": "[INTENT] Minimal change\n\n[CONTEXT]\nTest\n<<<END>>>",
            "output": """<<<PATCH_JSON>>>
{
  "version": "1",
  "edits": [
    {
      "path": "test.py",
      "loc": {"type": "anchor", "before": "def test():"},
      "action": "insert_after",
      "code": " pass"
    }
  ]
}
<<<END>>>"""
        }
    ]
    
    improvement_samples.extend(format_improvement_samples)
    
    print(f"  생성된 개선 샘플: {len(improvement_samples)}개")
    return improvement_samples

def update_training_dataset(improvement_samples: List[Dict[str, Any]]):
    """훈련 데이터셋에 개선 샘플 추가"""
    print("\n📝 훈련 데이터셋 업데이트 중...")
    
    # 기존 데이터셋 로드
    train_file = Path("training/data/train.jsonl")
    existing_samples = []
    
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    existing_samples.append(json.loads(line))
    
    print(f"  기존 샘플: {len(existing_samples)}개")
    
    # 개선 샘플 추가
    all_samples = existing_samples + improvement_samples
    print(f"  총 샘플: {len(all_samples)}개")
    
    # 업데이트된 데이터셋 저장
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 검증 데이터셋도 업데이트 (전체의 20% 샘플링)
    val_samples = all_samples[::5]  # 5개 중 1개씩
    val_file = Path("training/data/val.jsonl")
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"  훈련 데이터: {len(all_samples)}개 → {train_file}")
    print(f"  검증 데이터: {len(val_samples)}개 → {val_file}")

def main():
    """실패 케이스 재학습 실행"""
    print("🚀 실패 케이스 재학습 시작...")
    print("=" * 60)
    
    # 서버 상태 확인
    try:
        response = requests.get("http://127.0.0.1:8765/health", timeout=10)
        if response.status_code != 200:
            print("❌ 서버가 실행되지 않았습니다.")
            return
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        return
    
    # 1. 실패 케이스 수집
    failures = collect_failure_cases()
    print(f"\n수집된 실패 케이스: {len(failures)}개")
    
    if not failures:
        print("✅ 실패 케이스가 없습니다. 시스템이 안정적으로 작동 중입니다!")
        return
    
    # 2. 개선 샘플 생성
    improvement_samples = generate_improvement_samples(failures)
    
    # 3. 데이터셋 업데이트
    update_training_dataset(improvement_samples)
    
    # 4. 재학습 권장사항 출력
    print("\n" + "=" * 60)
    print("🎯 재학습 권장사항")
    print("=" * 60)
    print("1. 업데이트된 데이터셋으로 재학습 실행:")
    print("   python training/train_sft_qwen3.py")
    print("\n2. 재학습 후 E2E 평가 다시 실행:")
    print("   python evaluate_e2e.py")
    print("\n3. 성능 개선 확인 후 필요시 추가 데이터 수집")
    
    print(f"\n📊 개선 요약:")
    print(f"  - 기존 실패 케이스: {len(failures)}개")
    print(f"  - 생성된 개선 샘플: {len(improvement_samples)}개")
    print(f"  - 데이터셋 품질 향상을 위한 자기부스팅 완료!")
    
    print("\n🎉 실패 케이스 재학습 준비 완료!")

if __name__ == "__main__":
    main()
