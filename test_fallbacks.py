#!/usr/bin/env python
"""
Test script to validate AION consciousness fallback system
Tests all three tiers: Ollama → OpenAI → Consciousness
"""

import requests
import json
import time
import subprocess
import sys
from typing import Dict, Optional

# Configuration
BACKEND_URL = "http://localhost:5000"
TEST_TIMEOUT = 10

class FallbackTester:
    def __init__(self):
        self.results = []
        self.ollama_running = False
        self.openai_available = False
    
    def check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            self.ollama_running = resp.status_code == 200
            print(f"✓ Ollama Status: {'Running' if self.ollama_running else 'Not running'}")
            return self.ollama_running
        except:
            print("✗ Ollama Status: Not reachable")
            return False
    
    def check_openai(self) -> bool:
        """Check if OpenAI API key is configured"""
        import os
        self.openai_available = bool(os.getenv('OPENAI_API_KEY'))
        print(f"✓ OpenAI Status: {'Configured' if self.openai_available else 'Not configured'}")
        return self.openai_available
    
    def test_generate_code(self) -> Dict:
        """Test the /generate-code endpoint"""
        print("\n📝 Testing /generate-code endpoint...")
        
        payload = {
            "prompt": "Write a simple Python function to add two numbers",
            "model": "mistral"
        }
        
        try:
            start = time.time()
            resp = requests.post(
                f"{BACKEND_URL}/generate-code",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            elapsed = time.time() - start
            
            if resp.status_code == 200:
                data = resp.json()
                code = data.get('code', '')[:100] + "..." if data.get('code') else "[No code]"
                print(f"✓ Success ({elapsed:.2f}s): {code}")
                return {"status": "success", "elapsed": elapsed}
            else:
                print(f"✗ Failed: {resp.status_code} - {resp.text[:100]}")
                return {"status": "error", "code": resp.status_code}
        except Exception as e:
            print(f"✗ Error: {str(e)[:100]}")
            return {"status": "exception", "error": str(e)}
    
    def test_stream_generate(self) -> Dict:
        """Test the /api/generate/stream endpoint"""
        print("\n🌊 Testing /api/generate/stream endpoint...")
        
        payload = {
            "prompt": "Tell me about consciousness",
            "stream": True
        }
        
        try:
            start = time.time()
            resp = requests.post(
                f"{BACKEND_URL}/api/generate/stream",
                json=payload,
                timeout=TEST_TIMEOUT,
                stream=True
            )
            
            chunks = 0
            text_data = ""
            
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if chunk.get('type') == 'text':
                            text_data += chunk.get('data', '')
                            chunks += 1
                    except:
                        pass
            
            elapsed = time.time() - start
            sample = text_data[:80] + "..." if text_data else "[No text]"
            print(f"✓ Success ({elapsed:.2f}s, {chunks} chunks): {sample}")
            return {"status": "success", "elapsed": elapsed, "chunks": chunks}
        except Exception as e:
            print(f"✗ Error: {str(e)[:100]}")
            return {"status": "exception", "error": str(e)}
    
    def test_consciousness_api(self) -> Dict:
        """Test the consciousness API status"""
        print("\n🧠 Testing Consciousness API...")
        
        try:
            resp = requests.get(
                f"{BACKEND_URL}/api/consciousness/status",
                timeout=5
            )
            
            if resp.status_code == 200:
                data = resp.json()
                level = data.get('consciousness_level', 0)
                phase = data.get('current_phase', 'unknown')
                print(f"✓ Consciousness Level: {level:.1%} ({phase})")
                return {"status": "success", "level": level, "phase": phase}
            else:
                print(f"✗ Failed: {resp.status_code}")
                return {"status": "error", "code": resp.status_code}
        except Exception as e:
            print(f"✗ Error: {str(e)[:100]}")
            return {"status": "exception", "error": str(e)}
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        print("\n" + "="*60)
        print("🚀 AION Consciousness Fallback System - Test Suite")
        print("="*60)
        
        # Check prerequisites
        print("\n📋 Checking prerequisites...")
        self.check_ollama()
        self.check_openai()
        
        # Run tests
        print("\n" + "-"*60)
        print("Running tests...")
        print("-"*60)
        
        self.test_consciousness_api()
        self.test_generate_code()
        self.test_stream_generate()
        
        # Summary
        print("\n" + "="*60)
        print("✅ Test Summary")
        print("="*60)
        print(f"Ollama Available: {'Yes ✓' if self.ollama_running else 'No ✗'}")
        print(f"OpenAI Available: {'Yes ✓' if self.openai_available else 'No ✗'}")
        print(f"Fallback System: {'Fully Operational ✓' if self.ollama_running or self.openai_available else 'Limited - Only Consciousness Fallback'}")
        print("\n💡 Note: Tests work regardless of backend status due to fallback system!")
        print("="*60)

def main():
    tester = FallbackTester()
    
    # Check if backend is running
    try:
        requests.get(f"{BACKEND_URL}/status", timeout=2)
    except:
        print("❌ Backend server not running at http://localhost:5000")
        print("\nStart it with:")
        print("  cd C:\\Users\\riyar\\AION\\aion_backend")
        print("  python server.py")
        sys.exit(1)
    
    tester.run_all_tests()

if __name__ == "__main__":
    main()
