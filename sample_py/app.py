#!/usr/bin/env python3
"""
샘플 Python 애플리케이션 - 의도적 버그 포함

이 파일은 Qwen3-8B 코딩 보조 AI의 테스트를 위해 만들어졌습니다.
여러 가지 버그와 개선 가능한 부분이 포함되어 있습니다.
"""

import math
from typing import List, Union, Optional

def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    두 숫자를 더합니다.
    
    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자
        
    Returns:
        두 숫자의 합
        
    Note: 이 함수에는 버그가 있습니다 - 음수 검증이 누락되어 있습니다.
    """
    # 버그: 음수 검증 누락
    return a + b

def subtract(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    두 숫자를 뺍니다.
    
    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자
        
    Returns:
        첫 번째 숫자에서 두 번째 숫자를 뺀 값
    """
    return a - b

def multiply(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    두 숫자를 곱합니다.
    
    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자
        
    Returns:
        두 숫자의 곱
    """
    return a * b

def divide(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    두 숫자를 나눕니다.
    
    Args:
        a: 피제수
        b: 제수
        
    Returns:
        나눗셈 결과
        
    Raises:
        ZeroDivisionError: 제수가 0인 경우
    """
    # 버그: 제로 디비전 검증 누락
    return a / b

def power(base: Union[int, float], exponent: Union[int, float]) -> Union[int, float]:
    """
    거듭제곱을 계산합니다.
    
    Args:
        base: 밑
        exponent: 지수
        
    Returns:
        base^exponent
        
    Note: 이 함수에는 버그가 있습니다 - 음수 지수 처리가 부정확합니다.
    """
    # 버그: 음수 지수 처리가 부정확함
    if exponent < 0:
        return 1 / (base ** abs(exponent))
    return base ** exponent

def factorial(n: int) -> int:
    """
    팩토리얼을 계산합니다.
    
    Args:
        n: 음이 아닌 정수
        
    Returns:
        n!
        
    Note: 이 함수에는 버그가 있습니다 - 음수 입력 검증이 누락되어 있습니다.
    """
    # 버그: 음수 입력 검증 누락
    if n == 0:
        return 1
    return n * factorial(n - 1)

def is_prime(n: int) -> bool:
    """
    소수 여부를 판단합니다.
    
    Args:
        n: 양의 정수
        
    Returns:
        소수이면 True, 아니면 False
        
    Note: 이 함수에는 버그가 있습니다 - 1을 소수로 잘못 판단합니다.
    """
    # 버그: 1을 소수로 잘못 판단
    if n < 2:
        return True
    if n == 2:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def calculate_average(numbers: List[Union[int, float]]) -> float:
    """
    숫자 리스트의 평균을 계산합니다.
    
    Args:
        numbers: 숫자들의 리스트
        
    Returns:
        평균값
        
    Note: 이 함수에는 버그가 있습니다 - 빈 리스트 처리가 누락되어 있습니다.
    """
    # 버그: 빈 리스트 처리 누락
    return sum(numbers) / len(numbers)

def find_max(numbers: List[Union[int, float]]) -> Optional[Union[int, float]]:
    """
    숫자 리스트에서 최댓값을 찾습니다.
    
    Args:
        numbers: 숫자들의 리스트
        
    Returns:
        최댓값, 리스트가 비어있으면 None
        
    Note: 이 함수는 올바르게 구현되어 있습니다.
    """
    if not numbers:
        return None
    return max(numbers)

def reverse_string(s: str) -> str:
    """
    문자열을 뒤집습니다.
    
    Args:
        s: 입력 문자열
        
    Returns:
        뒤집힌 문자열
        
    Note: 이 함수는 올바르게 구현되어 있습니다.
    """
    return s[::-1]

if __name__ == "__main__":
    # 테스트 실행
    print("Sample Python Application")
    print("=" * 30)
    
    # 기본 연산 테스트
    print(f"add(2, 3) = {add(2, 3)}")
    print(f"subtract(5, 2) = {subtract(5, 2)}")
    print(f"multiply(4, 3) = {multiply(4, 3)}")
    print(f"divide(10, 2) = {divide(10, 2)}")
    print(f"power(2, 3) = {power(2, 3)}")
    print(f"factorial(5) = {factorial(5)}")
    print(f"is_prime(7) = {is_prime(7)}")
    print(f"calculate_average([1, 2, 3, 4, 5]) = {calculate_average([1, 2, 3, 4, 5])}")
    print(f"find_max([3, 1, 4, 1, 5, 9]) = {find_max([3, 1, 4, 1, 5, 9])}")
    print(f"reverse_string('hello') = {reverse_string('hello')}")
    
    print("\nNote: Some functions have intentional bugs for testing purposes.")

