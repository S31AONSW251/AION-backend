import json
import pytest

from server import app


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as c:
        yield c


def test_solve_math_llm_fallback(monkeypatch, client):
    # Monkeypatch the _call_ollama_generate to return a predictable response
    def fake_ollama(payload, timeout=60, retries=1):
        return {'response': 'LLM solved: 2+2=4'}

    monkeypatch.setattr('server._call_ollama_generate', fake_ollama)

    resp = client.post('/solve-math', json={'problem': 'What is 2+2?'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['ok'] is True
    assert data['method'] == 'llm'
    assert '2+2' in data['result'] or '4' in data['result']


@pytest.mark.skipif(not pytest.importorskip('sympy', reason='sympy not installed'), reason='sympy not installed')
def test_solve_math_sympy(client):
    # Simple expression that SymPy can simplify
    resp = client.post('/solve-math', json={'problem': '2+2'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['ok'] is True
    assert data['method'] == 'sympy'
    assert data['result'] == '4' or '4' in data['result']
