#!/usr/bin/env python3
import requests
import os

BACKEND_URL = os.environ.get('AION_BACKEND_URL', 'http://127.0.0.1:5000')

if __name__ == '__main__':
    # Ingest a simple document
    doc = { 'documents': [ { 'id': 'test-1', 'text': 'Test AION memory about the test scenario', 'metadata': { 'source': 'test' } } ] }
    r = requests.post(BACKEND_URL + '/api/rag/ingest', json=doc, timeout=10)
    print('Ingest', r.status_code, r.text)
    # Query
    q = { 'query': 'test scenario', 'top_k': 5 }
    r = requests.post(BACKEND_URL + '/api/rag/query', json=q, timeout=10)
    print('Query', r.status_code, r.text)
