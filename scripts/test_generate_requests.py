#!/usr/bin/env python3
"""Small test script to exercise backend endpoints:
- GET /health
- POST /api/generate/test
- POST /api/generate (non-stream)
- POST /api/generate/stream (streaming - print NDJSON lines)

Run with the backend running: .venv\Scripts\python.exe scripts\test_generate_requests.py
"""
import json
import sys
import time
from pathlib import Path

import requests


BASE = "http://127.0.0.1:5000"


def get_health():
    url = BASE + "/health"
    print(f"GET {url}")
    r = requests.get(url, timeout=5)
    print("status:", r.status_code)
    print(r.text)


def post_test_stream():
    url = BASE + "/api/generate/test"
    print(f"POST {url} (test stream)")
    r = requests.post(url, json={"prompt": "test"}, stream=True, timeout=20)
    print("status:", r.status_code)
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        print("LINE:", line)


def post_generate():
    url = BASE + "/api/generate"
    print(f"POST {url} (non-stream)")
    payload = {"prompt": "Hello from test script. Introduce yourself in one sentence."}
    r = requests.post(url, json=payload, timeout=30)
    print("status:", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)


def post_generate_stream():
    url = BASE + "/api/generate/stream"
    print(f"POST {url} (stream) — streaming NDJSON lines")
    payload = {"prompt": "Stream a short greeting token-by-token."}
    with requests.post(url, json=payload, stream=True, timeout=120) as r:
        print("status:", r.status_code)
        if r.status_code != 200:
            print(r.text)
            return
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            print("STREAM_LINE:", line)


def main():
    try:
        get_health()
    except Exception as e:
        print("Health check failed:", e)
    print('\n--- test stream ---')
    try:
        post_test_stream()
    except Exception as e:
        print("Test stream failed:", e)
    print('\n--- non-stream generate ---')
    try:
        post_generate()
    except Exception as e:
        print("Generate failed:", e)
    print('\n--- streaming generate ---')
    try:
        post_generate_stream()
    except Exception as e:
        print("Streaming generate failed:", e)


if __name__ == "__main__":
    main()
