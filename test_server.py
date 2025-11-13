import argparse
import json
import os
import sys
import time

try:
    import requests
except Exception:
    print("The 'requests' package is required. Install with: pip install requests")
    sys.exit(1)


def pretty(resp):
    try:
        return json.dumps(resp.json(), indent=2, ensure_ascii=False)
    except Exception:
        return resp.text[:1000]


def do_get(base, path, timeout=6):
    url = base.rstrip("/") + path
    print(f"\nGET {url}")
    try:
        r = requests.get(url, timeout=timeout)
        print(f" -> {r.status_code}")
        print(pretty(r))
    except Exception as e:
        print(" -> ERROR:", e)


def do_post(base, path, payload=None, timeout=10):
    url = base.rstrip("/") + path
    print(f"\nPOST {url} -> payload: {payload}")
    try:
        r = requests.post(url, json=payload or {}, timeout=timeout)
        print(f" -> {r.status_code}")
        print(pretty(r))
    except Exception as e:
        print(" -> ERROR:", e)


def main():
    p = argparse.ArgumentParser(description="Quick AION backend tester")
    p.add_argument("--url", "-u", default=os.environ.get("AION_TEST_URL", "http://127.0.0.1:5000"),
                   help="Base URL of AION backend (default: http://127.0.0.1:5000)")
    args = p.parse_args()
    base = args.url.rstrip("/")

    print(f"Testing AION server at {base} ...")

    # 1: health
    do_get(base, "/api/health")

    # 2: root
    do_get(base, "/")

    # 3: consciousness state
    do_get(base, "/consciousness/state")

    # 4: simple retrieve (RAG)
    do_post(base, "/api/retrieve", {"query": "project AION design notes"})

    # 5: advanced search (small payload)
    do_post(base, "/api/advanced-search", {"query": "stable diffusion video generation", "filters": {"file_type": "pdf"}})

    # 6: attempt an insight (posting an example URL) — safe if AION_ALLOW_EXTERNAL enabled
    example_url = "https://example.com"
    do_post(base, "/api/insight", {"url": example_url})

    # 7: check agent control endpoints (toggle pause/resume) — send admin_key if required
    print("\nAttempting agent control (pause/resume) without admin_key — may be rejected:")
    do_post(base, "/api/agent/control", {"action": "pause"})
    do_post(base, "/api/agent/control", {"action": "resume"})

    # 8: list assets (may be empty)
    do_get(base, "/api/assets")

    # 9: optional: add an episodic memory (consciousness)
    do_post(base, "/consciousness/add-episodic-memory", {"event_type": "test", "content": "Health-check event from test_server.py"})

    print("\nBasic tests complete. If some endpoints returned errors, inspect server logs and CORS/network config.")
    print("Tip: set AION_ALLOW_EXTERNAL=1 in the server environment if you need the server to fetch external URLs for /api/insight or /api/fetch-asset.")


if __name__ == "__main__":
    main()
