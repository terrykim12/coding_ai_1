#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2E 평가 스크립트
JSON 형식, 패치 적용, 테스트 통과율 측정
"""

import requests
import json
import time
import csv
from pathlib import Path
from typing import Dict, List, Any

def test_json_format_success_rate():
    """JSON 형식 성공률 테스트"""
    print("🔍 JSON 형식 성공률 테스트...")
    
    test_cases = [
        {
            "intent": "add() 함수에 음수 검증 추가",
            "paths": ["examples/sample_py"],
            "code_paste": "def add(a, b): return a + b"
        },
        {
            "intent": "divide() 함수에 제로 디비전 검증 추가",
            "paths": ["examples/sample_py"],
            "code_paste": "def divide(a, b): return a / b"
        },
        {
            "intent": "factorial() 함수 개선",
            "paths": ["examples/sample_py"],
            "code_paste": "def factorial(n): return n * factorial(n-1)"
        }
    ]
    
    results = {"total": 0, "json_success": 0, "plan_success": 0, "patch_success": 0}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"  테스트 {i}/{len(test_cases)}: {test_case['intent']}")
        results["total"] += 1
        
        # PLAN 테스트
        try:
            response = requests.post(
                "http://127.0.0.1:8765/plan",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            
            if response.status_code == 200:
                plan_result = response.json()
                plan_data = plan_result.get("plan")
                
                if isinstance(plan_data, dict) and "files" in plan_data:
                    results["json_success"] += 1
                    results["plan_success"] += 1
                    print(f"    ✅ PLAN JSON 성공")
                    
                    # PATCH 테스트
                    try:
                        patch_response = requests.post(
                            "http://127.0.0.1:8765/patch",
                            json={"plan": plan_data},
                            headers={"Content-Type": "application/json"},
                            timeout=120
                        )
                        
                        if patch_response.status_code == 200:
                            patch_result = patch_response.json()
                            patch_data = patch_result.get("patch")
                            
                            if isinstance(patch_data, dict) and "edits" in patch_data:
                                results["patch_success"] += 1
                                print(f"    ✅ PATCH JSON 성공")
                            else:
                                print(f"    ❌ PATCH JSON 실패: 잘못된 스키마")
                        else:
                            print(f"    ❌ PATCH 요청 실패: {patch_response.status_code}")
                            
                    except Exception as e:
                        print(f"    ❌ PATCH 테스트 실패: {e}")
                else:
                    print(f"    ❌ PLAN JSON 실패: 잘못된 스키마")
            else:
                print(f"    ❌ PLAN 요청 실패: {response.status_code}")
                
        except Exception as e:
            print(f"    ❌ PLAN 테스트 실패: {e}")
        
        time.sleep(1)  # 서버 부하 방지
    
    return results

def test_patch_application_success_rate():
    """패치 적용 성공률 테스트"""
    print("\n🔧 패치 적용 성공률 테스트...")
    
    # 간단한 패치 테스트 케이스
    test_patches = [
        {
            "name": "간단한 주석 추가",
            "patch": {
                "version": "1",
                "edits": [
                    {
                        "path": "examples/sample_py/app.py",
                        "loc": {
                            "type": "anchor",
                            "before": "def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:",
                            "after": "return a + b"
                        },
                        "action": "insert_before",
                        "once": True,
                        "code": "    # E2E 테스트 주석\n    "
                    }
                ]
            }
        }
    ]
    
    results = {"total": 0, "apply_success": 0, "syntax_valid": 0}
    
    for test_case in test_patches:
        print(f"  테스트: {test_case['name']}")
        results["total"] += 1
        
        try:
            response = requests.post(
                "http://127.0.0.1:8765/apply",
                json={
                    "patch": test_case["patch"],
                    "allowed_paths": ["examples/sample_py"],
                    "dry_run": False
                },
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                apply_result = response.json()
                applied_files = apply_result.get("applied", [])
                failed_files = apply_result.get("failed", [])
                
                # 새로운 상태 시스템: applied, noop_identical, noop_present, duplicate는 모두 성공
                success_statuses = {"applied", "noop_identical", "noop_present", "duplicate"}
                
                # details에서 성공 상태 확인
                details = apply_result.get("details", [])
                success_count = 0
                total_count = len(details)
                
                for detail in details:
                    if detail.get("status") in success_statuses:
                        success_count += 1
                
                if total_count > 0:
                    success_rate = success_count / total_count
                    if success_rate >= 0.8:  # 80% 이상 성공 시 통과
                        results["apply_success"] += 1
                        print(f"    ✅ 패치 적용 성공 ({success_rate:.1%})")
                        
                        # 상태별 상세 정보
                        for detail in details:
                            status = detail.get("status", "unknown")
                            path = detail.get("path", "unknown")
                            print(f"      - {path}: {status}")
                        
                        # 문법 검증 (Python 파일인 경우)
                        try:
                            with open("examples/sample_py/app.py", "r", encoding="utf-8") as f:
                                code_content = f.read()
                            
                            compile(code_content, "examples/sample_py/app.py", "exec")
                            results["syntax_valid"] += 1
                            print(f"    ✅ 문법 검증 통과")
                            
                        except SyntaxError as e:
                            print(f"    ❌ 문법 오류: {e}")
                        except Exception as e:
                            print(f"    ⚠️ 문법 검증 실패: {e}")
                    else:
                        print(f"    ❌ 패치 적용 실패: 성공률 {success_rate:.1%} ({success_count}/{total_count})")
                else:
                    print(f"    ❌ 패치 적용 실패: 상세 정보 없음")
            else:
                print(f"    ❌ 패치 요청 실패: {response.status_code}")
                
        except Exception as e:
            print(f"    ❌ 패치 테스트 실패: {e}")
    
    return results

def test_duplicate_prevention():
    """중복 방지 테스트"""
    print("\n🔄 중복 방지 테스트...")
    
    # 동일한 패치를 두 번 적용
    test_patch = {
        "version": "1",
        "edits": [
            {
                "path": "examples/sample_py/app.py",
                "loc": {
                    "type": "anchor",
                    "before": "def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:",
                    "after": "return a + b"
                },
                "action": "insert_before",
                "once": True,
                "code": "    # 중복 방지 테스트 주석\n    "
            }
        ]
    }
    
    results = {"first_apply": False, "second_skip": False}
    
    # 첫 번째 적용
    try:
        response = requests.post(
            "http://127.0.0.1:8765/apply",
            json={
                "patch": test_patch,
                "allowed_paths": ["examples/sample_py"],
                "dry_run": False
            },
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            apply_result = response.json()
            if apply_result.get("applied"):
                results["first_apply"] = True
                print("  ✅ 첫 번째 적용 성공")
            
    except Exception as e:
        print(f"  ❌ 첫 번째 적용 실패: {e}")
    
    time.sleep(1)
    
    # 두 번째 적용 (중복 방지되어야 함)
    try:
        response = requests.post(
            "http://127.0.0.1:8765/apply",
            json={
                "patch": test_patch,
                "allowed_paths": ["examples/sample_py"],
                "dry_run": False
            },
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            apply_result = response.json()
            details = apply_result.get("details", [])
            
            # duplicate 상태 확인
            duplicate_count = 0
            for detail in details:
                if detail.get("status") == "duplicate":
                    duplicate_count += 1
            
            if duplicate_count > 0:
                results["second_skip"] = True
                print(f"  ✅ 두 번째 적용 중복 방지됨 (duplicate: {duplicate_count})")
            else:
                print("  ❌ 중복 방지 실패")
            
    except Exception as e:
        print(f"  ❌ 두 번째 적용 테스트 실패: {e}")
    
    return results

def save_evaluation_results(results: Dict[str, Any]):
    """평가 결과를 CSV로 저장"""
    evaluation_dir = Path("evaluation")
    evaluation_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_file = evaluation_dir / f"e2e_evaluation_{timestamp}.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value", "Percentage"])
        
        # JSON 형식 성공률
        json_results = results["json_format"]
        if json_results["total"] > 0:
            writer.writerow([
                "JSON Format Success Rate", 
                f"{json_results['json_success']}/{json_results['total']}", 
                f"{json_results['json_success']/json_results['total']*100:.1f}%"
            ])
            writer.writerow([
                "PLAN Success Rate", 
                f"{json_results['plan_success']}/{json_results['total']}", 
                f"{json_results['plan_success']/json_results['total']*100:.1f}%"
            ])
            writer.writerow([
                "PATCH Success Rate", 
                f"{json_results['patch_success']}/{json_results['total']}", 
                f"{json_results['patch_success']/json_results['total']*100:.1f}%"
            ])
        
        # 패치 적용 성공률
        patch_results = results["patch_application"]
        if patch_results["total"] > 0:
            writer.writerow([
                "Patch Application Success Rate", 
                f"{patch_results['apply_success']}/{patch_results['total']}", 
                f"{patch_results['apply_success']/patch_results['total']*100:.1f}%"
            ])
            writer.writerow([
                "Syntax Validation Success Rate", 
                f"{patch_results['syntax_valid']}/{patch_results['total']}", 
                f"{patch_results['syntax_valid']/patch_results['total']*100:.1f}%"
            ])
        
        # 중복 방지
        duplicate_results = results["duplicate_prevention"]
        writer.writerow([
            "Duplicate Prevention", 
            "Pass" if duplicate_results["first_apply"] and duplicate_results["second_skip"] else "Fail",
            "100%" if duplicate_results["first_apply"] and duplicate_results["second_skip"] else "0%"
        ])
    
    print(f"\n📊 평가 결과 저장됨: {csv_file}")

def main():
    """E2E 평가 실행"""
    print("🚀 E2E 평가 시작...")
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
    
    results = {}
    
    # 1. JSON 형식 성공률 테스트
    results["json_format"] = test_json_format_success_rate()
    
    # 2. 패치 적용 성공률 테스트
    results["patch_application"] = test_patch_application_success_rate()
    
    # 3. 중복 방지 테스트
    results["duplicate_prevention"] = test_duplicate_prevention()
    
    # 4. 결과 요약
    print("\n" + "=" * 60)
    print("📊 E2E 평가 결과 요약")
    print("=" * 60)
    
    json_results = results["json_format"]
    if json_results["total"] > 0:
        print(f"JSON 형식 성공률: {json_results['json_success']}/{json_results['total']} ({json_results['json_success']/json_results['total']*100:.1f}%)")
        print(f"PLAN 성공률: {json_results['plan_success']}/{json_results['total']} ({json_results['plan_success']/json_results['total']*100:.1f}%)")
        print(f"PATCH 성공률: {json_results['patch_success']}/{json_results['total']} ({json_results['patch_success']/json_results['total']*100:.1f}%)")
    
    patch_results = results["patch_application"]
    if patch_results["total"] > 0:
        print(f"패치 적용 성공률: {patch_results['apply_success']}/{patch_results['total']} ({patch_results['apply_success']/patch_results['total']*100:.1f}%)")
        print(f"문법 검증 성공률: {patch_results['syntax_valid']}/{patch_results['total']} ({patch_results['syntax_valid']/patch_results['total']*100:.1f}%)")
    
    duplicate_results = results["duplicate_prevention"]
    duplicate_success = duplicate_results["first_apply"] and duplicate_results["second_skip"]
    print(f"중복 방지 기능: {'✅ 정상' if duplicate_success else '❌ 실패'}")
    
    # 5. 결과 저장
    save_evaluation_results(results)
    
    print("\n🎉 E2E 평가 완료!")

if __name__ == "__main__":
    main()
