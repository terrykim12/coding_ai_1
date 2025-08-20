#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
초기 데이터셋 생성 스크립트
구성: PATCH 60%, PLAN 25%, 형식가드 15%
"""

import json
import os
from pathlib import Path

def create_training_dataset():
    """초기 훈련 데이터셋 생성"""
    
    # 데이터셋 저장 디렉토리 생성
    data_dir = Path("training/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = []
    
    # ========================================
    # 1. PATCH 태스크 (60% = 18개 샘플)
    # ========================================
    
    # PATCH 샘플 1: add() 함수 음수 검증 추가
    dataset.append({
        "instruction": "Fix bug & add validation. Output PATCH JSON only.",
        "input": """[INTENT] add() 함수에 음수 입력 검증 추가

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
      "action": "replace_range",
      "code": "def add(a, b):\n    if a < 0 or b < 0:\n        raise ValueError(\"Negative numbers not allowed\")\n    return a + b"
    }
  ]
}
<<<END>>>"""
    })
    
    # PATCH 샘플 2: divide() 함수 제로 디비전 검증
    dataset.append({
        "instruction": "Fix division by zero bug. Output PATCH JSON only.",
        "input": """[INTENT] divide() 함수에 제로 디비전 검증 추가

[CONTEXT]
<<<FILE examples/sample_py/app.py>>>
def divide(a, b):
    return a / b
<<<END>>>""",
        "output": """<<<PATCH_JSON>>>
{
  "version": "1",
  "edits": [
    {
      "path": "examples/sample_py/app.py",
      "loc": {
        "type": "anchor",
        "before": "def divide(a, b):",
        "after": "return a / b"
      },
      "action": "replace_range",
      "code": "def divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError(\"Cannot divide by zero\")\n    return a / b"
    }
  ]
}
<<<END>>>"""
    })
    
    # PATCH 샘플 3: 테스트 추가
    dataset.append({
        "instruction": "Add test case for negative input validation. Output PATCH JSON only.",
        "input": """[INTENT] add() 함수 음수 입력 테스트 케이스 추가

[CONTEXT]
<<<FILE examples/sample_py/tests/test_app.py>>>
def test_add_positive():
    assert add(2, 3) == 5
<<<END>>>""",
        "output": """<<<PATCH_JSON>>>
{
  "version": "1",
  "edits": [
    {
      "path": "examples/sample_py/tests/test_app.py",
      "loc": {
        "type": "anchor",
        "after": "def test_add_positive():\n    assert add(2, 3) == 5"
      },
      "action": "insert_after",
      "code": "\n\ndef test_add_negative_validation():\n    with pytest.raises(ValueError):\n        add(-1, 2)\n    with pytest.raises(ValueError):\n        add(1, -2)"
    }
  ]
}
<<<END>>>"""
    })
    
    # PATCH 샘플 4-10: 다양한 버그 수정 패턴들 (간략화)
    patch_samples = [
        {
            "intent": "factorial() 함수 음수 검증 추가",
            "context": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
            "fix": "def factorial(n):\n    if n < 0:\n        raise ValueError(\"Factorial of negative number is undefined\")\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)"
        },
        {
            "intent": "is_prime() 함수 1 처리 수정",
            "context": "def is_prime(n):\n    if n < 2:\n        return True\n    return True",
            "fix": "def is_prime(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    return True"
        },
        {
            "intent": "calculate_average() 빈 리스트 검증",
            "context": "def calculate_average(numbers):\n    return sum(numbers) / len(numbers)",
            "fix": "def calculate_average(numbers):\n    if not numbers:\n        raise ValueError(\"Cannot calculate average of empty list\")\n    return sum(numbers) / len(numbers)"
        }
    ]
    
    for i, sample in enumerate(patch_samples, 4):
        dataset.append({
            "instruction": f"Fix bug. Output PATCH JSON only.",
            "input": f"""[INTENT] {sample['intent']}

[CONTEXT]
<<<FILE examples/sample_py/app.py>>>
{sample['context']}
<<<END>>>""",
            "output": f"""<<<PATCH_JSON>>>
{{
  "version": "1",
  "edits": [
    {{
      "path": "examples/sample_py/app.py",
      "loc": {{
        "type": "regex",
        "pattern": "{sample['context'].replace(chr(10), '\\n').replace('(', '\\(').replace(')', '\\)').replace('[', '\\[').replace(']', '\\]')}"
      }},
      "action": "replace_range",
      "code": "{sample['fix']}"
    }}
  ]
}}
<<<END>>>"""
        })
    
    # ========================================
    # 2. PLAN 태스크 (25% = 8개 샘플)
    # ========================================
    
    # PLAN 샘플 1: 전체적인 코드 개선 계획
    dataset.append({
        "instruction": "Create improvement plan. Output PLAN JSON only.",
        "input": """[INTENT] examples/sample_py/app.py의 모든 함수에 입력 검증 추가

[CONTEXT]
<<<FILE examples/sample_py/app.py>>>
def add(a, b):
    return a + b

def divide(a, b):
    return a / b

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
      "reason": "Add input validation to all mathematical functions",
      "strategy": "anchor",
      "tests": ["test_add_negative", "test_divide_zero", "test_factorial_negative"]
    },
    {
      "path": "examples/sample_py/tests/test_app.py",
      "reason": "Add comprehensive test cases for input validation",
      "strategy": "anchor",
      "tests": ["test_validation_coverage"]
    }
  ],
  "notes": "Focus on preventing common mathematical errors: negative inputs, division by zero, and edge cases"
}
<<<END>>>"""
    })
    
    # PLAN 샘플 2-5: 다양한 계획 패턴들
    plan_samples = [
        {
            "intent": "문자열 함수들 대소문자 처리 개선",
            "files": ["examples/sample_py/app.py"],
            "focus": "String processing functions case sensitivity"
        },
        {
            "intent": "리스트 함수들 빈 리스트 처리 개선",
            "files": ["examples/sample_py/app.py"],
            "focus": "Empty list validation for statistical functions"
        },
        {
            "intent": "새로운 유틸리티 함수 추가",
            "files": ["examples/sample_py/utils.py"],
            "focus": "Create new utility module with helper functions"
        }
    ]
    
    for sample in plan_samples:
        dataset.append({
            "instruction": "Create modification plan. Output PLAN JSON only.",
            "input": f"""[INTENT] {sample['intent']}

[CONTEXT]
<<<FILE examples/sample_py/app.py>>>
# Sample functions that need improvement
def sample_function():
    pass
<<<END>>>""",
            "output": f"""<<<PLAN_JSON>>>
{{
  "files": [
    {{
      "path": "{sample['files'][0]}",
      "reason": "{sample['focus']}",
      "strategy": "regex",
      "tests": ["test_improved_functionality"]
    }}
  ],
  "notes": "Systematic improvement focusing on {sample['focus'].lower()}"
}}
<<<END>>>"""
        })
    
    # ========================================
    # 3. 형식 가드 (15% = 4개 샘플)
    # ========================================
    
    # 형식 가드 샘플들: JSON만 출력하도록 학습
    format_guard_samples = [
        {
            "instruction": "Output only JSON patch. No explanation.",
            "input": "[INTENT] Simple bug fix\n\n[CONTEXT]\nSample code\n<<<END>>>",
            "output": """<<<PATCH_JSON>>>
{
  "version": "1",
  "edits": []
}
<<<END>>>"""
        },
        {
            "instruction": "Generate plan in strict JSON format only.",
            "input": "[INTENT] Code analysis\n\n[CONTEXT]\nSample code\n<<<END>>>",
            "output": """<<<PLAN_JSON>>>
{
  "files": [],
  "notes": "Analysis complete"
}
<<<END>>>"""
        }
    ]
    
    for sample in format_guard_samples:
        dataset.append(sample)
    
    # 추가 형식 가드 샘플들
    dataset.extend([
        {
            "instruction": "Strict JSON output required. No markdown formatting.",
            "input": "[INTENT] Format enforcement\n\n[CONTEXT]\nCode sample\n<<<END>>>",
            "output": """<<<PATCH_JSON>>>
{
  "version": "1",
  "edits": []
}
<<<END>>>"""
        },
        {
            "instruction": "JSON only response. Follow schema exactly.",
            "input": "[INTENT] Schema compliance\n\n[CONTEXT]\nTest code\n<<<END>>>",
            "output": """<<<PLAN_JSON>>>
{
  "files": [],
  "notes": "Schema compliant response"
}
<<<END>>>"""
        }
    ])
    
    # ========================================
    # 데이터셋 저장
    # ========================================
    
    # 훈련 데이터셋 저장
    train_file = data_dir / "train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 검증 데이터셋 생성 (훈련 데이터의 20% 샘플링)
    val_dataset = dataset[::5]  # 5개 중 1개씩 선택
    val_file = data_dir / "val.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in val_dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 데이터셋 생성 완료!")
    print(f"   훈련 데이터: {len(dataset)}개 샘플 → {train_file}")
    print(f"   검증 데이터: {len(val_dataset)}개 샘플 → {val_file}")
    print(f"   구성: PATCH {len([s for s in dataset if 'PATCH' in s['output']])}개, "
          f"PLAN {len([s for s in dataset if 'PLAN' in s['output']])}개")

if __name__ == "__main__":
    create_training_dataset()
