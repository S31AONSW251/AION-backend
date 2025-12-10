#!/usr/bin/env python3
"""
AION System Final Verification & Configuration Script
Tests all critical endpoints and systems
"""
import requests
import json
import sys
from datetime import datetime

BASE_URL = "http://127.0.0.1:5000"
TESTS_PASSED = 0
TESTS_FAILED = 0

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {level}: {msg}")

def test_endpoint(name, method, path, payload=None, expected_status=200):
    global TESTS_PASSED, TESTS_FAILED
    try:
        url = f"{BASE_URL}{path}"
        headers = {"Content-Type": "application/json"}
        
        if method.upper() == "GET":
            r = requests.get(url, headers=headers, timeout=10)
        elif method.upper() == "POST":
            r = requests.post(url, json=payload, headers=headers, timeout=10)
        else:
            r = requests.request(method, url, json=payload, headers=headers, timeout=10)
        
        if r.status_code == expected_status:
            log(f"✓ {name} [{method.upper()} {path}] -> {r.status_code}", "PASS")
            TESTS_PASSED += 1
            return True
        else:
            log(f"✗ {name} [{method.upper()} {path}] -> {r.status_code} (expected {expected_status})", "FAIL")
            log(f"  Response: {r.text[:200]}", "ERROR")
            TESTS_FAILED += 1
            return False
    except Exception as e:
        log(f"✗ {name} -> {str(e)}", "FAIL")
        TESTS_FAILED += 1
        return False

print("\n" + "="*70)
print("   AION SYSTEM COMPREHENSIVE VERIFICATION")
print("="*70 + "\n")

# Test 1: Health
log("Starting AION System Tests...", "INFO")
print()

print("BACKEND CONNECTIVITY TESTS")
print("-" * 70)
test_endpoint("Health Check", "GET", "/api/health", expected_status=200)
test_endpoint("Root Endpoint", "GET", "/", expected_status=200)

print("\nCONSCIOUSNESS SYSTEM TESTS")
print("-" * 70)
test_endpoint("Consciousness State", "GET", "/consciousness/state", expected_status=200)
test_endpoint("Episodic Memories", "GET", "/consciousness/memories?limit=5", expected_status=200)
test_endpoint("Insights", "GET", "/consciousness/insights?limit=5", expected_status=200)
test_endpoint("Add Episodic Memory", "POST", "/consciousness/add-episodic-memory", 
              {"event_type": "test", "content": "Test memory from verification script"}, expected_status=200)

print("\nAI GENERATION TESTS")
print("-" * 70)
test_endpoint("Code Generation", "POST", "/generate", 
              {"type": "code", "prompt": "print hello world"}, expected_status=200)
test_endpoint("Direct Code Gen", "POST", "/generate-code",
              {"prompt": "write a function to add two numbers"}, expected_status=200)

print("\nSEARCH & RETRIEVAL TESTS")
print("-" * 70)
test_endpoint("API Retrieve", "POST", "/api/retrieve",
              {"query": "AION artificial intelligence"}, expected_status=200)
test_endpoint("Advanced Search", "POST", "/api/advanced-search",
              {"query": "machine learning", "filters": {}}, expected_status=200)

print("\nSTATUS & PROVIDER TESTS")
print("-" * 70)
test_endpoint("Provider Status", "GET", "/status/providers", expected_status=200)
test_endpoint("Check Updates", "GET", "/check-updates", expected_status=200)

print("\nCONVERSATION SYNC TESTS")
print("-" * 70)
test_endpoint("Sync Conversation", "POST", "/sync-conversation",
              {"user_id": "test", "messages": [{"role": "user", "content": "hello"}]}, expected_status=200)

print("\nASYNCHRONOUS JOB TESTS")
print("-" * 70)
test_endpoint("List Assets", "GET", "/api/assets", expected_status=200)
test_endpoint("Agent Control - Pause", "POST", "/api/agent/control",
              {"action": "pause"}, expected_status=200)
test_endpoint("Agent Control - Resume", "POST", "/api/agent/control",
              {"action": "resume"}, expected_status=200)

print("\nINSIGHT & URL CACHE TESTS")
print("-" * 70)
# This might fail if URL is unreachable, that's OK for this test
test_endpoint("URL Insight Cache", "POST", "/api/insight",
              {"url": "https://example.com"}, expected_status=200)

print("\n" + "="*70)
print("                        TEST SUMMARY")
print("="*70)
print(f"✓ PASSED: {TESTS_PASSED}")
print(f"✗ FAILED: {TESTS_FAILED}")
total = TESTS_PASSED + TESTS_FAILED
if total > 0:
    success_rate = (TESTS_PASSED / total) * 100
    print(f"📊 SUCCESS RATE: {success_rate:.1f}%")
print("="*70)

if TESTS_FAILED == 0:
    print("\n✨ ALL TESTS PASSED - AION IS FULLY OPERATIONAL ✨\n")
    sys.exit(0)
else:
    print(f"\n⚠ {TESTS_FAILED} test(s) failed. Check logs above.\n")
    sys.exit(1)
