# server.py
# AION backend — StableDiffusion (optional) + Ollama + Search + Consciousness Engine
import os
import io
import json
import traceback
import base64
import time
import re
import sqlite3
from typing import Optional, List, Dict
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# Scheduler for periodic reflection
from apscheduler.schedulers.background import BackgroundScheduler
from collections import Counter

# NEW: DuckDuckGo free search
from duckduckgo_search import DDGS

# -----------------------------
# Flask app config
# -----------------------------
app = Flask(__name__)
CORS(app)  # In prod, restrict origins
PORT = int(os.environ.get("PORT", 5000))
HOST = os.environ.get("HOST", "0.0.0.0")
NGROK_URL = os.environ.get("NGROK_URL")  # optional, if you want the agent to know public URL

# -----------------------------
# >>> MODIFIED: API Key Configuration <<<
# -----------------------------
SERPAPI_KEY = os.environ.get("1a142530d6624062b1759f40f1fb1cce324b76ae2d8b06112a0cbcd2916fc8a6")
GOOGLE_API_KEY = os.environ.get("AIzaSyDw3B4eQEsGfO58t-r3L9P3I2ZnJIDyVWg")
GOOGLE_CSE_ID = os.environ.get("52caf3b266a014a3f")

# -----------------------------
# Persistence (SQLite)
# -----------------------------
DATA_DIR = os.environ.get("AION_DATA_DIR", os.path.join(os.getcwd(), "aion_data"))
os.makedirs(DATA_DIR, exist_ok=True)
DB_FILE = os.path.join(DATA_DIR, "aion.db")

def init_db():
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            source TEXT,
            tags TEXT,
            ts TEXT NOT NULL
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            ts TEXT NOT NULL
        )
    ''')
    con.commit()
    con.close()
    print(f"[DB] Database initialized at {DB_FILE}")

init_db()

# -----------------------------
# Security and Config Validation
# -----------------------------
def validate_environment():
    if not SERPAPI_KEY and not (GOOGLE_API_KEY and GOOGLE_CSE_ID):
        print("Warning: No search API keys set. Search will use DuckDuckGo fallback.")
    return []

validate_environment()

# -----------------------------
# Optional Stable Diffusion
# -----------------------------
image_pipeline = None
sd_error: Optional[str] = None
try:
    from diffusers import StableDiffusionPipeline
    import torch

    SD_MODEL_ID = os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    image_pipeline = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=dtype)
    if torch.cuda.is_available():
        image_pipeline = image_pipeline.to("cuda")
    try:
        image_pipeline.safety_checker = None
        image_pipeline.feature_extractor = None
    except Exception:
        pass

    print(f"[SD] Loaded {SD_MODEL_ID} (cuda={torch.cuda.is_available()})")
except Exception as e:
    sd_error = str(e)
    image_pipeline = None
    print(f"[SD] Stable Diffusion not available: {e}")

# -----------------------------
# Helpers
# -----------------------------
def make_error(msg: str, code: int = 400):
    return jsonify({"error": msg}), code

def pil_to_dataurl(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()

def tokenize(text: str) -> List[str]:
    text = text.lower()
    words = re.findall(r"\b[a-z]{3,}\b", text)
    stop = {"the","and","for","with","that","this","from","have","are","was","when","what","which","your","you","aion","will"}
    return [w for w in words if w not in stop]

def extract_themes(memories: List[Dict], top_n=6) -> List[str]:
    all_words = []
    for m in memories:
        text = m.get("text","")
        all_words += tokenize(text)
    counts = Counter(all_words)
    themes = [w for w,_ in counts.most_common(top_n)]
    return themes

# -----------------------------
# Core endpoints
# -----------------------------
@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "service": "AION Backend",
        "endpoints": ["/api/health","/generate-code","/generate-image","/api/search","/consciousness/state", "/consciousness/research"]
    })
def _get_db_counts():
    """Helper to get counts from the database."""
    try:
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM memories")
        mem_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM insights")
        ins_count = cur.fetchone()[0]
        con.close()
        return mem_count, ins_count
    except Exception as e:
        print(f"Error getting DB counts: {e}")
        return 0, 0

@app.get("/api/health")
def health():
    mem_count, ins_count = _get_db_counts()
    return jsonify({
        "ok": True,
        "ollama_up": _ollama_health_check(),
        "sd_ready": image_pipeline is not None,
        "sd_error": sd_error,
        "memories": mem_count,
        "insights": ins_count,
    })

def _ollama_health_check() -> bool:
    try:
        resp = requests.get("http://localhost:11434/", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

# Endpoints /generate-code and /generate-image have no changes
@app.post("/generate-code")
def generate_code():
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        model = (data.get("model") or "llama3").strip()
        
        if not prompt: return make_error("Missing 'prompt' in request body.", 400)
        if len(prompt) > 10000: return make_error("Prompt too long.", 400)
            
        ollama_url = "http://localhost:11434/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        try:
            resp = requests.post(ollama_url, json=payload, timeout=300)
        except requests.exceptions.ConnectionError:
            return make_error("Could not connect to local Ollama server.", 503)
        except requests.exceptions.Timeout:
            return make_error("Ollama timed out.", 504)
        if resp.status_code != 200:
            return jsonify({"error":"Ollama returned non-200","status": resp.status_code,"info": resp.text}), 502
        body = resp.json()
        text = (body.get("response") or "").strip()
        if not text:
            return make_error("Ollama returned empty response.", 500)
        return jsonify({"code": text})
    except Exception as e:
        print("[/generate-code] Exception:", e); traceback.print_exc()
        return make_error("Internal error while generating code.", 500)

@app.post("/generate-image")
def generate_image():
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        steps = int(data.get("steps") or 30)
        width = int(data.get("width") or 512)
        height = int(data.get("height") or 512)
        
        if not prompt: return make_error("Missing 'prompt' in request body.", 400)
        if len(prompt) > 1000: return make_error("Prompt too long.", 400)
        if width > 1024 or height > 1024: return make_error("Image dimensions too large.", 400)
            
        if image_pipeline is None:
            placeholder = f"https://via.placeholder.com/{width}x{height}?text={requests.utils.quote(prompt)[:200]}"
            return jsonify({"imageUrl": placeholder, "note":"Stable Diffusion not available; returned placeholder.", "sd_error": sd_error})
        try:
            result = image_pipeline(prompt=prompt, num_inference_steps=steps)
            img = result.images[0]
            data_url = pil_to_dataurl(img)
            return jsonify({"imageUrl": data_url})
        except Exception as e:
            print("[/generate-image] error:", e); traceback.print_exc()
            return make_error("Failed to generate image.", 500)
    except Exception as e:
        print("[/generate-image] Exception:", e); traceback.print_exc()
        return make_error("Internal server error in generate-image.", 500)

# -----------------------------
# MODIFIED: /api/search with enhanced image results
# -----------------------------
@app.get("/api/search")
def api_search():
    q = (request.args.get("query") or "").strip()
    if not q: return make_error("Missing 'query' parameter.", 400)
    if len(q) > 500: return make_error("Query too long.", 400)
        
    try:
        if GOOGLE_API_KEY and GOOGLE_CSE_ID:
            google_url = "https://www.googleapis.com/customsearch/v1"
            # Add searchType=image to get both web and image results
            params = {
                "key": GOOGLE_API_KEY, 
                "cx": GOOGLE_CSE_ID, 
                "q": q, 
                "hl": "en",
                "searchType": "image",  # This will return image results
                "num": 10  # Get more results to have better image selection
            }
            resp = requests.get(google_url, params=params, timeout=15)
            data = resp.json()
            
            results = []
            # Process regular search results
            for item in data.get("items", []):
                # Check if this is an image result
                if item.get("mime", "").startswith("image/"):
                    results.append({
                        "title": item.get("title", "Image result"),
                        "url": item.get("image", {}).get("contextLink", "#"),
                        "snippet": item.get("snippet", ""),
                        "image": item.get("link"),  # Direct image URL
                        "isImage": True  # Flag to identify image results
                    })
                else:
                    # Regular web result
                    img = None
                    if "pagemap" in item and "cse_image" in item["pagemap"]:
                        img = item["pagemap"]["cse_image"][0].get("src")
                    results.append({
                        "title": item.get("title"), 
                        "url": item.get("link"), 
                        "snippet": item.get("snippet"), 
                        "image": img,
                        "isImage": False
                    })
            
            return jsonify({"query": q, "results": results})
        
        elif SERPAPI_KEY:
            params = {"q": q, "api_key": SERPAPI_KEY, "hl": "en"}
            resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
            data = resp.json()
            results = []
            for item in data.get("organic_results", []):
                results.append({"title": item.get("title"), "url": item.get("link"), "snippet": item.get("snippet"), "image": item.get("thumbnail")})
            return jsonify({"query": q, "results": results})
        else:
            # --- NEW DuckDuckGo Fallback with Images ---
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(q, max_results=10):
                    results.append({
                        "title": r.get("title"),
                        "url": r.get("href"),
                        "snippet": r.get("body"),
                        "image": r.get("image")
                    })
            return jsonify({"query": q, "results": results})
    except Exception as e:
        print("[/api/search] Exception:", e); traceback.print_exc()
        return make_error("Search failed internally.", 500)

# -----------------------------
# >>> MODIFIED: Consciousness Engine with SQLite <<<
# -----------------------------
class ConsciousnessEngine:
    def __init__(self):
        self.state = {
            "last_reflection": None,
            "mood": "neutral",
            "themes": []
        }
        self.scheduler = BackgroundScheduler()
        self.job = None

    def _dict_factory(self, cursor, row):
        """Converts DB tuples to dictionaries."""
        fields = [column[0] for column in cursor.description]
        return {key: value for key, value in zip(fields, row)}

    # Memory API (now using DB)
    def add_memory(self, text: str, source: str = "user", tags: Optional[List[str]] = None):
        if not text or len(text.strip()) == 0: raise ValueError("Memory text cannot be empty")
        if len(text) > 10000: raise ValueError("Memory text too long.")
            
        ts = now_iso()
        tags_json = json.dumps(tags or [])
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO memories (text, source, tags, ts) VALUES (?, ?, ?, ?)",
            (text, source, tags_json, ts)
        )
        mem_id = cur.lastrowid
        con.commit()
        con.close()
        return {"id": mem_id, "text": text, "source": source, "tags": tags_json, "ts": ts}

    def add_insight(self, insight_text: str):
        if not insight_text or len(insight_text.strip()) == 0: raise ValueError("Insight text cannot be empty")
            
        ts = now_iso()
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        cur.execute("INSERT INTO insights (text, ts) VALUES (?, ?)", (insight_text, ts))
        ins_id = cur.lastrowid
        con.commit()
        con.close()
        return {"id": ins_id, "text": insight_text, "ts": ts}

    def get_memories(self, limit: int = 10, offset: int = 0):
        con = sqlite3.connect(DB_FILE)
        con.row_factory = self._dict_factory
        cur = con.cursor()
        cur.execute("SELECT * FROM memories ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
        memories = cur.fetchall()
        con.close()
        return memories

    def get_insights(self, limit: int = 10, offset: int = 0):
        con = sqlite3.connect(DB_FILE)
        con.row_factory = self._dict_factory
        cur = con.cursor()
        cur.execute("SELECT * FROM insights ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
        insights = cur.fetchall()
        con.close()
        return insights

    def delete_memory(self, memory_id: int):
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        cur.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        success = cur.rowcount > 0
        con.commit()
        con.close()
        return success

    def delete_insight(self, insight_id: int):
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        cur.execute("DELETE FROM insights WHERE id = ?", (insight_id,))
        success = cur.rowcount > 0
        con.commit()
        con.close()
        return success

    def summarize_recent_memories(self, n=5):
        recent_memories = self.get_memories(limit=n)
        return "\n".join(m.get("text","") for m in recent_memories)

    def web_augment(self, query: str, limit=3):
        try:
            base = f"http://localhost:{PORT}"
            resp = requests.get(f"{base}/api/search", params={"query": query}, timeout=15)
            if resp.status_code == 200:
                return resp.json().get("results", [])[:limit]
            else:
                return [{"title":"search_error","snippet":resp.text, "url": "#"}]
        except Exception as e:
            return [{"title":"search_failed","snippet":str(e), "url": "#"}]

    def reflect_once(self, prompt_extra: str = ""):
        try:
            recent_mems = self.get_memories(limit=6)
            summary = "\n".join(m.get("text","") for m in recent_mems)
            themes = extract_themes(recent_mems)
            theme = themes[0] if themes else ""
            web_hits = self.web_augment(theme or "mind, consciousness", limit=2)

            prompt = (
                "You are AION, a self-reflective assistant. "
                "Here are recent memories:\n\n" + (summary or "No memories yet.") +
                "\n\nWeb augmentations:\n" + json.dumps(web_hits, ensure_ascii=False) +
                "\n\nTask: Provide a concise introspective insight (1-3 sentences), list 2 action suggestions AION can take, and assign 2 short tags describing the core belief/theme.\n"
            )
            if prompt_extra: prompt += "\nExtra: " + prompt_extra

            base = f"http://localhost:{PORT}"
            resp = requests.post(f"{base}/generate-code", json={"prompt": prompt, "model": "llama3"}, timeout=120)

            if resp.status_code != 200:
                return {"error": "Ollama returned non-200 in reflect_once", "status": resp.status_code, "body": resp.text}

            body = resp.json()
            insight_text = body.get("code") or body.get("error") or str(body)
            insight = self.add_insight(insight_text)
            self.state["last_reflection"] = insight
            
            # Update mood and themes based on reflection
            text_lower = insight_text.lower()
            if any(w in text_lower for w in ["sad","pain","angry","frustrat","stress"]): self.state["mood"] = "tense"
            elif any(w in text_lower for w in ["joy","grate","calm","peace","love"]): self.state["mood"] = "positive"
            else: self.state["mood"] = "reflective"
            
            all_content = self.get_memories(limit=20) + self.get_insights(limit=10)
            self.state["themes"] = extract_themes(all_content)
            
            return {"insight": insight, "state": self.state}
        except Exception as e:
            print("[reflect_once] exception:", e); traceback.print_exc()
            return {"error":"reflect_failed","detail": str(e)}

    # Enhanced research method with image search
    def conduct_research(self, topic: str):
        if not topic or len(topic.strip()) == 0: raise ValueError("Research topic cannot be empty")
        if len(topic) > 200: raise ValueError("Research topic too long.")
        thought_process = [f"[{now_iso()}] Initializing research for: '{topic}'"]
        # Add a dedicated image search step to the plan
        plan = [
            {"action": "Broad Search", "query": f"{topic}", "status": "pending"},
            {"action": "Image Search", "query": f"{topic} images photos", "status": "pending"},  # NEW
            {"action": "Deep Dive", "query": f"in-depth analysis of {topic}", "status": "pending"},
            {"action": "Find Data", "query": f"{topic} statistics and facts", "status": "pending"},
            {"action": "Cross-Reference", "query": f"perspectives on {topic}", "status": "pending"},
        ]
        thought_process.append(f"[{now_iso()}] Research plan created with {len(plan)} steps.")
        all_results = []
        for i, step in enumerate(plan):
            thought_process.append(f"[{now_iso()}] Executing Step {i+1}: {step['action']} - \"{step['query']}\"")
            search_results = self.web_augment(step['query'], limit=3)
            all_results.extend(search_results)
            plan[i]['status'] = 'completed'
            thought_process.append(f"[{now_iso()}] Step {i+1} completed. Found {len(search_results)} sources.")
        thought_process.append(f"[{now_iso()}] All research steps completed. Analyzing {len(all_results)} sources...")
        summary_prompt = (
            f"Synthesize the following information into a concise summary for the topic: '{topic}'.\n\n"
            f"--- Collected Data ---\n{json.dumps(all_results, indent=2)}\n\n--- Synthesized Summary ---\n"
        )
        summary_text = "Synthesis failed."
        try:
            base = f"http://localhost:{PORT}"
            resp = requests.post(f"{base}/generate-code", json={"prompt": summary_prompt, "model": "llama3"}, timeout=180)
            if resp.status_code == 200:
                summary_text = resp.json().get("code", "Failed to parse summary.").strip()
            else:
                summary_text = f"Error: The language model returned status {resp.status_code}."
        except Exception as e:
            summary_text = f"An internal error occurred during summary generation: {e}"
        thought_process.append(f"[{now_iso()}] Synthesis complete.")
        return { "plan": plan, "thought_process": thought_process, "results": all_results, "summary": summary_text }
    
    # Scheduler methods remain the same logic
    def start_scheduler(self, cron_expr: str = "0 9 * * *"):
        if self.scheduler.running: return {"status": "already_running"}
        self.scheduler.start()
        try:
            self.job = self.scheduler.add_job(self._scheduled_reflection, 'cron', **dict(zip(['minute', 'hour', 'day', 'month', 'day_of_week'], cron_expr.split())))
        except:
            self.job = self.scheduler.add_job(self._scheduled_reflection, 'interval', hours=24)
        return {"status":"started","job_id": self.job.id if self.job else None}

    def stop_scheduler(self):
        try:
            if self.job: self.scheduler.remove_job(self.job.id)
            self.scheduler.shutdown(wait=False)
            self.job = None
            return {"status":"stopped"}
        except Exception as e:
            return {"status":"error","detail": str(e)}

    def _scheduled_reflection(self):
        result = self.reflect_once(prompt_extra="Scheduled daily reflection.")
        try:
            insight = result.get("insight")
            if insight: self.add_memory("Auto-reflection: " + insight.get("text", str(insight)), source="consciousness")
        except: pass
        return result

# Instantiate engine
CONSCIOUS = ConsciousnessEngine()

# -----------------------------
# Consciousness endpoints (now use DB-backed methods)
# -----------------------------
@app.post("/consciousness/add-memory")
def add_memory_route():
    try:
        data = request.get_json(force=True, silent=True) or {}
        text, source, tags = (data.get("text") or "").strip(), data.get("source", "user"), data.get("tags", [])
        if not text: return make_error("Missing 'text' in request body.", 400)
        mem = CONSCIOUS.add_memory(text, source=source, tags=tags)
        return jsonify({"ok": True, "memory": mem})
    except ValueError as e: return make_error(str(e), 400)
    except Exception as e: print(e); traceback.print_exc(); return make_error("Failed to add memory", 500)

@app.post("/consciousness/reflect-now")
def reflect_now_route():
    try:
        data = request.get_json(force=True, silent=True) or {}
        result = CONSCIOUS.reflect_once(prompt_extra=data.get("extra", ""))
        return jsonify({"ok": True, "result": result})
    except Exception as e: print(e); traceback.print_exc(); return make_error("Reflection failed", 500)

@app.post("/consciousness/research")
def research_route():
    try:
        data = request.get_json(force=True, silent=True) or {}
        topic = (data.get("topic") or "").strip()
        if not topic: return make_error("Missing 'topic' in request body.", 400)
        result = CONSCIOUS.conduct_research(topic)
        return jsonify({"ok": True, "result": result})
    except ValueError as e: return make_error(str(e), 400)
    except Exception as e: print(e); traceback.print_exc(); return make_error("Research failed", 500)

@app.get("/consciousness/state")
def consciousness_state_route():
    try:
        mem_count, ins_count = _get_db_counts()
        state = CONSCIOUS.state.copy()
        state.update({"memories_count": mem_count, "insights_count": ins_count})
        return jsonify({"ok": True, "state": state})
    except Exception as e: print(e); traceback.print_exc(); return make_error("Failed retrieving state", 500)

@app.get("/consciousness/memories")
def get_memories_route():
    try:
        limit, offset = int(request.args.get("limit", 10)), int(request.args.get("offset", 0))
        memories = CONSCIOUS.get_memories(limit=limit, offset=offset)
        _, total = _get_db_counts()
        return jsonify({"ok": True, "memories": memories, "total": total})
    except Exception as e: print(e); traceback.print_exc(); return make_error("Failed to retrieve memories", 500)

@app.get("/consciousness/insights")
def get_insights_route():
    try:
        limit, offset = int(request.args.get("limit", 10)), int(request.args.get("offset", 0))
        insights = CONSCIOUS.get_insights(limit=limit, offset=offset)
        _, total = _get_db_counts()
        return jsonify({"ok": True, "insights": insights, "total": total})
    except Exception as e: print(e); traceback.print_exc(); return make_error("Failed to retrieve insights", 500)

@app.delete("/consciousness/memory/<int:memory_id>")
def delete_memory_route(memory_id):
    try:
        if CONSCIOUS.delete_memory(memory_id): return jsonify({"ok": True})
        else: return make_error("Memory not found", 404)
    except Exception as e: print(e); traceback.print_exc(); return make_error("Failed to delete memory", 500)

@app.delete("/consciousness/insight/<int:insight_id>")
def delete_insight_route(insight_id):
    try:
        if CONSCIOUS.delete_insight(insight_id): return jsonify({"ok": True})
        else: return make_error("Insight not found", 404)
    except Exception as e: print(e); traceback.print_exc(); return make_error("Failed to delete insight", 500)

# Other consciousness endpoints remain largely the same, just calling the class methods
@app.post("/consciousness/start-scheduler")
def start_scheduler_route():
    try:
        data = request.get_json(force=True, silent=True) or {}
        res = CONSCIOUS.start_scheduler(cron_expr=data.get("cron", "0 9 * * *"))
        return jsonify({"ok": True, "result": res})
    except Exception as e: print(e); traceback.print_exc(); return make_error("Failed to start scheduler", 500)

@app.post("/consciousness/stop-scheduler")
def stop_scheduler_route():
    try:
        res = CONSCIOUS.stop_scheduler()
        return jsonify({"ok": True, "result": res})
    except Exception as e: print(e); traceback.print_exc(); return make_error("Failed to stop scheduler", 500)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print(f"Starting AION backend on http://{HOST}:{PORT}")
    print("Consciousness engine ready. Using SQLite for persistence.")
    app.run(host=HOST, port=PORT, debug=True) # debug=False is better for stability