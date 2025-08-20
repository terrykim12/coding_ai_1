#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-8B Local Coding AI CUDA 테스트 스크립트
GPU 가속 성능을 테스트합니다
"""

import requests
import json
import time
import torch
from typing import Dict, Any

def print_cuda_info():
    """CUDA 환경 정보 출력"""
    print("🚀 CUDA 환경 정보")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA 사용 가능")
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"   CUDA 버전: {torch.version.cuda}")
        print(f"   PyTorch 버전: {torch.__version__}")
        
        # GPU 메모리 상태
        print(f"   현재 할당된 메모리: {torch.cuda.memory_allocated(0) / 1024**3:.1f}GB")
        print(f"   현재 캐시된 메모리: {torch.cuda.memory_reserved(0) / 1024**3:.1f}GB")
        
    else:
        print("❌ CUDA 사용 불가")
        print("   CPU 모드로 실행됩니다")
    
    print()

def test_server_health(base_url: str) -> bool:
    """서버 상태 확인"""
    print("1️⃣ 서버 상태 확인...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 서버 상태: {data.get('status', 'unknown')}")
            print(f"   모델 로드: {data.get('model_loaded', 'unknown')}")
            
            # CUDA 정보 표시
            if data.get('cuda_available'):
                print(f"   GPU: {data.get('gpu_name', 'unknown')}")
                print(f"   GPU 메모리: {data.get('gpu_memory_total', 'unknown')}")
                print(f"   할당된 메모리: {data.get('gpu_memory_allocated', 'unknown')}")
                print(f"   캐시된 메모리: {data.get('gpu_memory_cached', 'unknown')}")
                print(f"   CUDA 버전: {data.get('cuda_version', 'unknown')}")
            else:
                print("   CUDA 사용 불가")
            
            return True
        else:
            print(f"❌ 서버 응답 오류: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        return False

def test_plan_generation(base_url: str) -> Dict[str, Any]:
    """PLAN 단계 테스트 (성능 측정)"""
    print("\n2️⃣ PLAN 단계 테스트 (성능 측정)...")
    
    plan_data = {
        "intent": "Add input validation to add() function",
        "paths": ["examples/sample_py"],
        "code_paste": "def add(a, b): return a + b"
    }
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/plan",
            json=plan_data,
            headers={"Content-Type": "application/json"},
            timeout=120  # 2분 타임아웃
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            plan_result = response.json()
            print(f"✅ PLAN 생성 성공!")
            print(f"   응답 시간: {duration:.2f}초")
            print(f"   계획 ID: {plan_result.get('plan_id', 'N/A')}")
            
            # GPU 메모리 사용량 확인
            try:
                health_response = requests.get(f"{base_url}/health")
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    if health_data.get('cuda_available'):
                        print(f"   GPU 메모리 사용량: {health_data.get('gpu_memory_allocated', 'unknown')}")
            except:
                pass
            
            return plan_result
        else:
            print(f"❌ PLAN 생성 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return {}
            
    except Exception as e:
        print(f"❌ PLAN 테스트 실패: {e}")
        return {}

def test_patch_generation(base_url: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """PATCH 단계 테스트 (성능 측정)"""
    if not plan:
        print("❌ PLAN이 없어서 PATCH 테스트를 건너뜁니다")
        return {}
    
    print("\n3️⃣ PATCH 단계 테스트 (성능 측정)...")
    
    patch_data = {"plan": plan.get('plan', {})}
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/patch",
            json=patch_data,
            headers={"Content-Type": "application/json"},
            timeout=180  # 3분 타임아웃
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            patch_result = response.json()
            print(f"✅ PATCH 생성 성공!")
            print(f"   응답 시간: {duration:.2f}초")
            print(f"   패치 ID: {patch_result.get('patch_id', 'N/A')}")
            
            # GPU 메모리 사용량 확인
            try:
                health_response = requests.get(f"{base_url}/health")
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    if health_data.get('cuda_available'):
                        print(f"   GPU 메모리 사용량: {health_data.get('gpu_memory_allocated', 'unknown')}")
            except:
                pass
            
            return patch_result
        else:
            print(f"❌ PATCH 생성 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return {}
            
    except Exception as e:
        print(f"❌ PATCH 테스트 실패: {e}")
        return {}

def test_workflow(base_url: str) -> Dict[str, Any]:
    """전체 워크플로우 테스트 (성능 측정)"""
    print("\n4️⃣ 전체 워크플로우 테스트 (성능 측정)...")
    
    workflow_data = {
        "intent": "Fix the bug in calculate_average function",
        "paths": ["examples/sample_py"],
        "code_paste": "def calculate_average(numbers): return sum(numbers) / len(numbers)",
        "auto_apply": False,  # 안전을 위해 자동 적용 비활성화
        "auto_test": False
    }
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/workflow",
            json=workflow_data,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5분 타임아웃
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            workflow_result = response.json()
            print(f"✅ 워크플로우 실행 성공!")
            print(f"   총 실행 시간: {duration:.2f}초")
            print(f"   워크플로우 ID: {workflow_result.get('workflow_id', 'N/A')}")
            
            return workflow_result
        else:
            print(f"❌ 워크플로우 실행 실패: {response.status_code}")
            print(f"   오류: {response.text}")
            return {}
            
    except Exception as e:
        print(f"❌ 워크플로우 테스트 실패: {e}")
        return {}

def benchmark_gpu_performance():
    """GPU 성능 벤치마크"""
    print("\n5️⃣ GPU 성능 벤치마크...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다")
        return
    
    try:
        # 간단한 텐서 연산으로 GPU 성능 테스트
        device = torch.device("cuda")
        
        # 메모리 할당 테스트
        start_time = time.time()
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        allocation_time = time.time() - start_time
        
        # 행렬 곱셈 테스트
        start_time = time.time()
        z = torch.mm(x, y)
        torch.cuda.synchronize()  # GPU 연산 완료 대기
        computation_time = time.time() - start_time
        
        # 메모리 정리
        del x, y, z
        torch.cuda.empty_cache()
        
        print(f"✅ GPU 성능 테스트 완료")
        print(f"   메모리 할당 시간: {allocation_time:.4f}초")
        print(f"   1000x1000 행렬 곱셈: {computation_time:.4f}초")
        
    except Exception as e:
        print(f"❌ GPU 성능 테스트 실패: {e}")

def main():
    """메인 테스트 함수"""
    base_url = "http://127.0.0.1:8765"
    
    print("🚀 Qwen3-8B Local Coding AI CUDA 테스트 시작")
    print("=" * 60)
    
    # 1. CUDA 환경 정보
    print_cuda_info()
    
    # 2. GPU 성능 벤치마크
    benchmark_gpu_performance()
    
    # 3. 서버 상태 확인
    if not test_server_health(base_url):
        print("❌ 서버가 실행되지 않았습니다. 먼저 서버를 시작하세요.")
        return
    
    # 4. PLAN 단계 테스트
    plan_result = test_plan_generation(base_url)
    
    # 5. PATCH 단계 테스트
    patch_result = test_patch_generation(base_url, plan_result)
    
    # 6. 전체 워크플로우 테스트
    workflow_result = test_workflow(base_url)
    
    # 7. 최종 결과 요약
    print("\n🎉 CUDA 테스트 완료!")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("✅ CUDA 가속이 활성화되어 있습니다")
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   사용 가능한 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️ CUDA를 사용할 수 없습니다 (CPU 모드)")
    
    print("\n📊 테스트 결과:")
    if plan_result:
        print("   ✅ PLAN 단계: 성공")
    else:
        print("   ❌ PLAN 단계: 실패")
    
    if patch_result:
        print("   ✅ PATCH 단계: 성공")
    else:
        print("   ❌ PATCH 단계: 실패")
    
    if workflow_result:
        print("   ✅ 워크플로우: 성공")
    else:
        print("   ❌ 워크플로우: 실패")

if __name__ == "__main__":
    main()
