#!/usr/bin/env python3
"""
샘플 Python 애플리케이션의 테스트 파일

이 파일은 Qwen3-8B 코딩 보조 AI의 테스트를 위해 만들어졌습니다.
여러 가지 테스트 케이스와 의도적 실패 케이스가 포함되어 있습니다.
"""

import pytest
import sys
import os

# 상위 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import (
    add, subtract, multiply, divide, power, factorial, fibonacci,
    is_prime, find_primes_up_to, calculate_average, find_max, find_min,
    reverse_string, is_palindrome, count_vowels
)

class TestBasicOperations:
    """기본 연산 함수들 테스트"""
    
    def test_add_positive(self):
        """양수 덧셈 테스트"""
        assert add(5, 3) == 8
        assert add(0, 0) == 0
        assert add(100, 200) == 300
    
    def test_add_negative(self):
        """음수 덧셈 테스트"""
        assert add(-5, -3) == -8
        assert add(-10, 5) == -5
        assert add(10, -5) == 5
    
    def test_add_float(self):
        """실수 덧셈 테스트"""
        assert add(3.14, 2.86) == 6.0
        assert add(0.1, 0.2) == pytest.approx(0.3, rel=1e-10)
    
    def test_subtract(self):
        """뺄셈 테스트"""
        assert subtract(10, 4) == 6
        assert subtract(0, 5) == -5
        assert subtract(-5, -3) == -2
    
    def test_multiply(self):
        """곱셈 테스트"""
        assert multiply(6, 7) == 42
        assert multiply(0, 100) == 0
        assert multiply(-3, 4) == -12
    
    def test_divide(self):
        """나눗셈 테스트"""
        assert divide(20, 5) == 4.0
        assert divide(10, 3) == pytest.approx(3.333333, rel=1e-5)
        assert divide(-10, 2) == -5.0
    
    def test_divide_by_zero(self):
        """0으로 나누기 테스트"""
        with pytest.raises(ZeroDivisionError):
            divide(10, 0)

class TestAdvancedFunctions:
    """고급 함수들 테스트"""
    
    def test_power_positive(self):
        """양수 거듭제곱 테스트"""
        assert power(2, 3) == 8
        assert power(5, 0) == 1
        assert power(10, 1) == 10
    
    def test_power_negative_exponent(self):
        """음수 지수 테스트"""
        assert power(2, -1) == 0.5
        assert power(3, -2) == pytest.approx(0.111111, rel=1e-5)
    
    def test_power_zero_base(self):
        """0의 거듭제곱 테스트"""
        assert power(0, 5) == 0
        # 0^0은 정의되지 않음 (현재 구현에서는 1 반환)
        assert power(0, 0) == 1
    
    def test_factorial_positive(self):
        """양수 팩토리얼 테스트"""
        assert factorial(0) == 1
        assert factorial(1) == 1
        assert factorial(5) == 120
        assert factorial(10) == 3628800
    
    def test_factorial_negative(self):
        """음수 팩토리얼 테스트 - 현재는 무한 재귀 발생"""
        # 이 테스트는 현재 구현에서 실패해야 함
        with pytest.raises(RecursionError):
            factorial(-1)
    
    def test_fibonacci_positive(self):
        """양수 피보나치 테스트"""
        assert fibonacci(0) == 0
        assert fibonacci(1) == 1
        assert fibonacci(2) == 1
        assert fibonacci(8) == 21
    
    def test_fibonacci_negative(self):
        """음수 피보나치 테스트 - 현재는 무한 재귀 발생"""
        # 이 테스트는 현재 구현에서 실패해야 함
        with pytest.raises(RecursionError):
            fibonacci(-1)

class TestPrimeFunctions:
    """소수 관련 함수들 테스트"""
    
    def test_is_prime_positive(self):
        """양수 소수 판별 테스트"""
        assert is_prime(2) == True
        assert is_prime(3) == True
        assert is_prime(17) == True
        assert is_prime(4) == False
        assert is_prime(9) == False
    
    def test_is_prime_edge_cases(self):
        """소수 판별 경계 케이스 테스트"""
        assert is_prime(1) == False  # 1은 소수가 아님
        assert is_prime(0) == False
        assert is_prime(-1) == False
    
    def test_find_primes_up_to(self):
        """범위 내 소수 찾기 테스트"""
        assert find_primes_up_to(10) == [2, 3, 5, 7]
        assert find_primes_up_to(20) == [2, 3, 5, 7, 11, 13, 17, 19]
        assert find_primes_up_to(2) == [2]

class TestListFunctions:
    """리스트 관련 함수들 테스트"""
    
    def test_calculate_average(self):
        """평균 계산 테스트"""
        assert calculate_average([1, 2, 3, 4, 5]) == 3.0
        assert calculate_average([0, 0, 0]) == 0.0
        assert calculate_average([1.5, 2.5]) == 2.0
    
    def test_calculate_average_empty_list(self):
        """빈 리스트 평균 계산 테스트 - 현재는 ZeroDivisionError 발생"""
        # 이 테스트는 현재 구현에서 실패해야 함
        with pytest.raises(ZeroDivisionError):
            calculate_average([])
    
    def test_find_max(self):
        """최댓값 찾기 테스트"""
        assert find_max([1, 2, 3, 4, 5]) == 5
        assert find_max([-5, -3, -1]) == -1
        assert find_max([0]) == 0
    
    def test_find_max_empty_list(self):
        """빈 리스트 최댓값 찾기 테스트"""
        assert find_max([]) == None
    
    def test_find_min(self):
        """최솟값 찾기 테스트"""
        assert find_min([1, 2, 3, 4, 5]) == 1
        assert find_min([-5, -3, -1]) == -5
        assert find_min([0]) == 0
    
    def test_find_min_empty_list(self):
        """빈 리스트 최솟값 찾기 테스트"""
        assert find_min([]) == None

class TestStringFunctions:
    """문자열 관련 함수들 테스트"""
    
    def test_reverse_string(self):
        """문자열 뒤집기 테스트"""
        assert reverse_string("hello") == "olleh"
        assert reverse_string("") == ""
        assert reverse_string("a") == "a"
        assert reverse_string("12345") == "54321"
    
    def test_is_palindrome(self):
        """회문 판별 테스트"""
        assert is_palindrome("racecar") == True
        assert is_palindrome("hello") == False
        assert is_palindrome("") == True
        assert is_palindrome("a") == True
    
    def test_is_palindrome_case_sensitive(self):
        """회문 판별 대소문자 구분 테스트"""
        # 현재 구현은 대소문자를 구분함
        assert is_palindrome("Racecar") == False
        assert is_palindrome("Mom") == False
    
    def test_count_vowels(self):
        """모음 개수 세기 테스트"""
        assert count_vowels("hello") == 2
        assert count_vowels("world") == 1
        assert count_vowels("aeiou") == 5
        assert count_vowels("xyz") == 0
        assert count_vowels("") == 0
    
    def test_count_vowels_case_sensitive(self):
        """모음 개수 세기 대소문자 구분 테스트"""
        # 현재 구현은 소문자 모음만 고려함
        assert count_vowels("HELLO") == 0
        assert count_vowels("Hello") == 2

class TestEdgeCases:
    """경계 케이스 테스트"""
    
    def test_large_numbers(self):
        """큰 숫자 테스트"""
        # 매우 큰 숫자에서의 동작 확인
        assert add(999999999, 1) == 1000000000
        assert multiply(1000000, 1000000) == 1000000000000
    
    def test_floating_point_precision(self):
        """부동소수점 정밀도 테스트"""
        # 부동소수점 연산의 정밀도 확인
        result = add(0.1, 0.2)
        assert result == pytest.approx(0.3, rel=1e-10)
    
    def test_type_consistency(self):
        """타입 일관성 테스트"""
        # 입력 타입에 따른 출력 타입 확인
        assert isinstance(add(1, 2), int)
        assert isinstance(add(1.0, 2), float)
        assert isinstance(add(1, 2.0), float)

if __name__ == "__main__":
    # pytest로 실행
    pytest.main([__file__])

