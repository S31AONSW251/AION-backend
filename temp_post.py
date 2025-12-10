import requests
payload = {"documents":[{"text":"Test document: AION vector ingest probe","metadata":{"source":"probe"}}]}
try:
    r = requests.post("http://127.0.0.1:5000/admin/vector/ingest?admin_key=changeme", json=payload, timeout=5)
    print(r.status_code)
    print(r.text)
except Exception as e:
    print('Error:', e)
