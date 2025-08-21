import pytest
from examples.sample_py.app import divide

def test_divide_normal():
    assert divide(6, 3) == 2

def test_divide_zero():
    try:
        divide(1, 0)
    except Exception as e:
        # 적용 정책에 맞춰서 한 가지 조건만 체크 (예: 메시지 포함)
        assert "0" in str(e) or "zero" in str(e).lower()
    else:
        # 예외를 안 던지는 정책이라면, 여기서 반환값/메시지 검증으로 바꿔주세요.
        pytest.fail("expected guard behavior on divide by zero")
