import subprocess
import sys
import os
import json
import re
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TestResult:
    passed: bool
    output: str
    error_count: int
    failure_count: int
    test_count: int
    duration: float
    error_details: List[Dict[str, str]]

def _parse_pytest_output(output: str) -> TestResult:
    """pytest 출력 파싱"""
    error_count = 0
    failure_count = 0
    test_count = 0
    duration = 0.0
    error_details = []
    
    # 테스트 결과 요약 파싱
    summary_match = re.search(r'=+ (.*?) in ([\d.]+)s =+', output)
    if summary_match:
        summary = summary_match.group(1)
        duration = float(summary_match.group(2))
        
        # 통계 파싱
        stats = re.findall(r'(\d+) (passed|failed|error|skipped)', summary)
        for count, status in stats:
            count = int(count)
            if status == 'passed':
                test_count += count
            elif status == 'failed':
                failure_count += count
                test_count += count
            elif status == 'error':
                error_count += count
                test_count += count
            elif status == 'skipped':
                test_count += count
    
    # 에러 상세 정보 파싱
    error_pattern = r'=+ FAILURES =+\n(.*?)(?=\n=+|\Z)'
    error_matches = re.findall(error_pattern, output, re.DOTALL)
    
    for error in error_matches:
        # 테스트 이름과 에러 메시지 추출
        test_name_match = re.search(r'([^\n]+)', error)
        if test_name_match:
            test_name = test_name_match.group(1).strip()
            
            # 에러 메시지 추출
            error_msg_match = re.search(r'([^\n]*Error[^\n]*)', error)
            error_msg = error_msg_match.group(1) if error_msg_match else "Unknown error"
            
            error_details.append({
                "test": test_name,
                "error": error_msg,
                "full_output": error[:500]  # 첫 500자만
            })
    
    passed = (error_count == 0 and failure_count == 0)
    
    return TestResult(
        passed=passed,
        output=output,
        error_count=error_count,
        failure_count=failure_count,
        test_count=test_count,
        duration=duration,
        error_details=error_details
    )

def run_pytest(
    test_path: Optional[str] = None,
    args: Optional[List[str]] = None,
    timeout: int = 180,
    capture_output: bool = True
) -> TestResult:
    """pytest 실행"""
    
    cmd = [sys.executable, "-m", "pytest"]
    
    if test_path:
        cmd.append(test_path)
    
    # 기본 옵션
    default_args = [
        "-q",  # quiet mode
        "--tb=short",  # 간단한 traceback
        "--strict-markers",  # 마커 검증
        "--disable-warnings"  # 경고 숨김
    ]
    
    if args:
        cmd.extend(args)
    else:
        cmd.extend(default_args)
    
    try:
        # 환경 변수 설정
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')
        
        # pytest 실행
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            env=env,
            cwd=os.getcwd()
        )
        
        if capture_output:
            output = result.stdout + "\n" + result.stderr
        else:
            output = "Tests completed"
        
        # 결과 파싱
        test_result = _parse_pytest_output(output)
        
        # subprocess 에러 처리
        if result.returncode != 0 and test_result.test_count == 0:
            test_result.passed = False
            test_result.error_count += 1
            test_result.error_details.append({
                "test": "pytest_execution",
                "error": f"pytest failed with return code {result.returncode}",
                "full_output": output
            })
        
        return test_result
        
    except subprocess.TimeoutExpired:
        return TestResult(
            passed=False,
            output=f"pytest timed out after {timeout} seconds",
            error_count=1,
            failure_count=0,
            test_count=0,
            duration=timeout,
            error_details=[{
                "test": "pytest_timeout",
                "error": f"pytest timed out after {timeout} seconds",
                "full_output": ""
            }]
        )
        
    except Exception as e:
        return TestResult(
            passed=False,
            output=f"Failed to run pytest: {str(e)}",
            error_count=1,
            failure_count=0,
            test_count=0,
            duration=0.0,
            error_details=[{
                "test": "pytest_error",
                "error": str(e),
                "full_output": ""
            }]
        )

def run_specific_test(
    test_file: str,
    test_function: Optional[str] = None,
    timeout: int = 60
) -> TestResult:
    """특정 테스트 파일/함수 실행"""
    
    if test_function:
        test_path = f"{test_file}::{test_function}"
    else:
        test_path = test_file
    
    return run_pytest(test_path, timeout=timeout)

def run_tests_with_coverage(
    test_path: Optional[str] = None,
    timeout: int = 300
) -> TestResult:
    """커버리지와 함께 테스트 실행"""
    
    coverage_args = [
        "--cov=.",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov"
    ]
    
    return run_pytest(test_path, coverage_args, timeout)

def get_test_summary(test_result: TestResult) -> Dict[str, any]:
    """테스트 결과 요약"""
    return {
        "status": "passed" if test_result.passed else "failed",
        "summary": {
            "total_tests": test_result.test_count,
            "passed": test_result.test_count - test_result.error_count - test_result.failure_count,
            "failed": test_result.failure_count,
            "errors": test_result.error_count,
            "duration": f"{test_result.duration:.2f}s"
        },
        "errors": test_result.error_details[:5],  # 상위 5개 에러만
        "output_preview": test_result.output[:1000]  # 첫 1000자만
    }

def run_tests_in_loop(
    test_path: Optional[str] = None,
    max_iterations: int = 5,
    delay: float = 1.0
) -> List[TestResult]:
    """테스트를 반복 실행하여 안정성 확인"""
    
    results = []
    
    for i in range(max_iterations):
        result = run_pytest(test_path, timeout=60)
        results.append(result)
        
        if result.passed:
            break
        
        # 실패 시 잠시 대기 후 재시도
        if i < max_iterations - 1:
            import time
            time.sleep(delay)
    
    return results

def check_test_health(test_path: Optional[str] = None) -> Dict[str, any]:
    """테스트 상태 점검"""
    
    # 빠른 테스트 실행
    quick_result = run_pytest(test_path, ["-x"], timeout=30)
    
    health_status = {
        "healthy": quick_result.passed,
        "test_count": quick_result.test_count,
        "last_run": quick_result.duration,
        "recommendations": []
    }
    
    if quick_result.error_count > 0:
        health_status["recommendations"].append("Fix test errors before proceeding")
    
    if quick_result.duration > 10:
        health_status["recommendations"].append("Tests are running slowly, consider optimization")
    
    if quick_result.test_count == 0:
        health_status["recommendations"].append("No tests found, add test files")
    
    return health_status

