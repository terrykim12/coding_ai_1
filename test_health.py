#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 health check 스크립트
"""

import requests

def test_health():
    """서버 상태 확인"""
    print("🏥 서버 상태 확인...")
    
    try:
        response = requests.get(
            "http://127.0.0.1:8765/health",
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 서버 정상!")
            print(f"   상태: {result.get('status', 'N/A')}")
            print(f"   모델 로딩: {result.get('model_loaded', 'N/A')}")
            print(f"   CUDA 사용: {result.get('cuda_available', 'N/A')}")
            if result.get('cuda_available'):
                print(f"   GPU: {result.get('gpu_name', 'N/A')}")
                print(f"   메모리: {result.get('gpu_memory_total', 'N/A')}")
            return True
        else:
            print(f"❌ 서버 오류: {response.status_code}")
            print(f"   응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return False

if __name__ == "__main__":
    test_health()
