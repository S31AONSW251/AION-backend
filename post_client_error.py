import requests
try:
    r = requests.post('http://127.0.0.1:5000/api/log-client-error', json={
        'error':'Simulated client error for automated capture test',
        'stack':'Trace: simulated',
        'when':'automated-test'
    }, timeout=5)
    print('status', r.status_code)
    print(r.text)
except Exception as e:
    print('error', e)
