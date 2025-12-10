#!/usr/bin/env python3
"""
Index all memories from the DB into the vector store via the admin endpoint.
"""
import os
import sys
import requests

BACKEND_URL = os.environ.get('AION_BACKEND_URL', 'http://127.0.0.1:5000')
ADMIN_KEY = os.environ.get('AION_ADMIN_KEY', 'changeme')

if __name__ == '__main__':
    url = f"{BACKEND_URL}/admin/vector/sync-memory?admin_key={ADMIN_KEY}"
    print(f"Calling {url}")
    resp = requests.post(url, timeout=60)
    print('Status:', resp.status_code)
    try:
        print(resp.json())
    except Exception:
        print(resp.text)
