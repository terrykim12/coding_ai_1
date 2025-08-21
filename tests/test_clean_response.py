from server.ollama_api import _clean_response


def test_clean_hello_en():
    s = 'Okay, the user said "Hello". I need to respond with a short greeting only. Hello!'
    out = _clean_response(s)
    assert out.lower().startswith("hello")


def test_clean_hello_ko():
    s = "좋아, 사용자가 '안녕'이라고 했어. 이제 간단히 인사만 하면 돼. 안녕하세요!"
    out = _clean_response(s)
    assert "안녕하세요" in out


