from app.llm.mock_provider import MockProvider

def test_mock_generate():
    p = MockProvider()
    res = p.generate("hello")
    assert "[MOCK]" in res["text"]
