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
from flask import Flask, request, jsonify, Response
from flask import send_from_directory, abort
from werkzeug.utils import secure_filename
import uuid
from flask_cors import CORS
import hashlib

# Scheduler for periodic reflection
from apscheduler.schedulers.background import BackgroundScheduler
from collections import Counter

# NEW: DuckDuckGo free search (optional import)
try:
    from ddgs import DDGS
except Exception:
    DDGS = None

# Add these imports at the top
import redis
from flask_socketio import SocketIO, emit
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# -----------------------------
# Flask app config
# -----------------------------
app = Flask(__name__)
# Fix CORS configuration to allow frontend requests
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"])  # Adjust to your frontend URL
PORT = int(os.environ.get("PORT", 5000))
HOST = os.environ.get("HOST", "0.0.0.0")
NGROK_URL = os.environ.get("NGROK_URL")  # optional, if you want the agent to know public URL

# Add after Flask app initialization
socketio = SocketIO(app, cors_allowed_origins="*")
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])

# Redis initialization
redis_client = None
try:
    redis_client = redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True
    )
    redis_client.ping()
    print("[Redis] Connected successfully")
except Exception as e:
    print(f"[Redis] Connection failed: {e}")
    redis_client = None

# -----------------------------
# API Key Configuration (use environment variables)
# -----------------------------
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
FAST_PROVIDER = os.environ.get("FAST_PROVIDER")  # e.g. 'replicate'
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
# Global flag to allow or disallow external HTTP requests. Defaults to enabled.
AION_ALLOW_EXTERNAL = str(os.environ.get('AION_ALLOW_EXTERNAL', '1')).lower() in ('1', 'true', 'yes')
ADMIN_KEY = os.environ.get('AION_ADMIN_KEY', 'changeme')


def safe_get(url, *args, **kwargs):
    """Wrapper around requests.get that respects AION_ALLOW_EXTERNAL."""
    if not AION_ALLOW_EXTERNAL:
        # Provide a helpful error that points to how to enable external access
        raise RuntimeError('External HTTP requests are disabled by AION_ALLOW_EXTERNAL. Set env var `AION_ALLOW_EXTERNAL=1` or call the admin endpoint `/admin/allow-external` with a valid admin key to enable.')
    # enforce a sensible default timeout to avoid long hangs when callers forget to set one
    # shorter default (6s) helps fail fast and prevents the server from being tied up
    if 'timeout' not in kwargs:
        kwargs['timeout'] = 6
    return requests.get(url, *args, **kwargs)


def safe_post(url, *args, **kwargs):
    """Wrapper around requests.post that respects AION_ALLOW_EXTERNAL."""
    if not AION_ALLOW_EXTERNAL:
        raise RuntimeError('External HTTP requests are disabled by AION_ALLOW_EXTERNAL. Set env var `AION_ALLOW_EXTERNAL=1` or call the admin endpoint `/admin/allow-external` with a valid admin key to enable.')
    # shorter default timeout to fail fast on slow upstreams
    if 'timeout' not in kwargs:
        kwargs['timeout'] = 6
    return requests.post(url, *args, **kwargs)
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
    # Add search_history table for analytics
    cur.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY,
            query TEXT NOT NULL,
            result_count INTEGER,
            timestamp TEXT NOT NULL
        )
    ''')
    # NEW: Add episodic_memory table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS episodic_memory (
            id INTEGER PRIMARY KEY,
            event_type TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    # NEW: Add procedural_memory table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS procedural_memory (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            steps TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_used TEXT
        )
    ''')
    # Insights cache table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            insight TEXT,
            summary TEXT,
            created_at TEXT
        )
    ''')
    # Video jobs table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS video_jobs (
            id TEXT PRIMARY KEY,
            prompt TEXT,
            status TEXT,
            result_path TEXT,
            error TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    ''')
    # Image jobs table for async image generation
    cur.execute('''
        CREATE TABLE IF NOT EXISTS image_jobs (
            id TEXT PRIMARY KEY,
            prompt TEXT,
            steps INTEGER,
            width INTEGER,
            height INTEGER,
            status TEXT,
            result_path TEXT,
            error TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    ''')
    con.commit()
    con.close()
    print(f"[DB] Database initialized at {DB_FILE}")



init_db()

# --- Startup / diagnostic logs ---
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL')
# Timeouts (connect, read) for model proxying
AION_MODEL_CONNECT_TIMEOUT = float(os.environ.get('AION_MODEL_CONNECT_TIMEOUT', os.environ.get('AION_MODEL_TIMEOUT', 3)))
AION_MODEL_READ_TIMEOUT = float(os.environ.get('AION_MODEL_READ_TIMEOUT', os.environ.get('AION_MODEL_READ_TIMEOUT', 60)))
print(f"[Startup] OLLAMA_BASE_URL={OLLAMA_BASE_URL or '<none>'}")
print(f"[Startup] AION_MODEL_CONNECT_TIMEOUT={AION_MODEL_CONNECT_TIMEOUT}, AION_MODEL_READ_TIMEOUT={AION_MODEL_READ_TIMEOUT}")

# Concise startup summary (helpful for quick debugging)
def _startup_summary():
    try:
        openai_present = bool(OPENAI_API_KEY)
        ddgs_present = 'yes' if DDGS is not None else 'no'
        model_host = OLLAMA_BASE_URL or 'http://localhost:11434'
        print('[Startup Summary]')
        print(f'  - Ollama base: {model_host}')
        print(f'  - OpenAI fallback configured: {openai_present}')
        print(f'  - DuckDuckGo (ddgs) available: {ddgs_present}')
        print(f'  - Model connect/read timeouts: {AION_MODEL_CONNECT_TIMEOUT}/{AION_MODEL_READ_TIMEOUT} (s)')
    except Exception as e:
        print(f'[Startup Summary] failed: {e}')

_startup_summary()

# -----------------------------
# Job queue and limits
# -----------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

# Limits (tweak as needed)
MAX_STEPS = int(os.environ.get('MAX_STEPS', 40))
MAX_FRAMES = int(os.environ.get('MAX_FRAMES', 64))
MAX_WIDTH = int(os.environ.get('MAX_WIDTH', 1024))
MAX_HEIGHT = int(os.environ.get('MAX_HEIGHT', 1024))

executor = ThreadPoolExecutor(max_workers=int(os.environ.get('VIDEO_WORKERS', 2)))


def _cache_get(key: str) -> Optional[str]:
    try:
        if redis_client:
            return redis_client.get(key)
    except Exception:
        pass
    return None


def _cache_set(key: str, value: str, ex: int = 3600):
    try:
        if redis_client:
            redis_client.set(key, value, ex=ex)
    except Exception:
        pass


def _save_file_from_url(url: str, max_bytes: int = 50 * 1024 * 1024):
    """Fetch a remote URL and save to DATA_DIR. Returns (ok, filename, error)"""
    try:
        # Basic validation
        if not url.startswith('http'):
            return False, None, 'Invalid URL'

        resp = safe_get(url, stream=True, timeout=20)
        resp.raise_for_status()

        # Content length guard
        content_length = resp.headers.get('content-length')
        if content_length and int(content_length) > max_bytes:
            return False, None, 'File too large'

        # Determine filename
        parsed_name = os.path.basename(url.split('?')[0]) or ''
        ext = os.path.splitext(parsed_name)[1] or ''
        safe_name = secure_filename(parsed_name) or hashlib.sha256(url.encode()).hexdigest()[:16] + ext
        os.makedirs(DATA_DIR, exist_ok=True)
        out_path = os.path.join(DATA_DIR, safe_name)

        # Stream write with size enforcement
        total = 0
        with open(out_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    f.close()
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
                    return False, None, 'File exceeds allowed size'
                f.write(chunk)

        return True, safe_name, None
    except Exception as e:
        return False, None, str(e)


@app.route('/api/fetch-asset', methods=['POST'])
def api_fetch_asset():
    try:
        data = request.get_json(force=True, silent=True) or {}
        url = data.get('url')
        if not url:
            return jsonify({'ok': False, 'error': 'Missing url'}), 400

        ok, filename, err = _save_file_from_url(url)
        if not ok:
            return jsonify({'ok': False, 'error': err}), 400

        files_url = request.host_url.rstrip('/') + f"/files/{filename}"
        return jsonify({'ok': True, 'url': files_url, 'filename': filename})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500


def _extract_text_from_html(html: str) -> str:
    # Very lightweight extraction: remove scripts/styles and tags
    try:
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', html, flags=re.S|re.I)
        text = re.sub(r'<!--.*?-->', ' ', text, flags=re.S)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception:
        return ''


def _summarize_text_extractive(text: str, max_sentences: int = 5) -> str:
    # Simple TF-IDF sentence scoring summarizer using sklearn TfidfVectorizer
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as _np

        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= max_sentences:
            return text[:4000]

        vec = TfidfVectorizer(stop_words='english', max_features=5000)
        X = vec.fit_transform(sentences)
        scores = X.sum(axis=1).A1
        ranked_idx = _np.argsort(scores)[::-1][:max_sentences]
        ranked_idx.sort()
        summary = ' '.join([sentences[i] for i in ranked_idx])
        return summary
    except Exception:
        # fallback: return beginning of text
        return text[:1000]


@app.route('/api/insight', methods=['POST'])
def api_insight():
    try:
        data = request.get_json(force=True, silent=True) or {}
        url = data.get('url')
        if not url:
            return jsonify({'ok': False, 'error': 'Missing url'}), 400

        cache_key = f"insight:{hashlib.sha256(url.encode()).hexdigest()}"
        # Try redis first
        try:
            if redis_client:
                cached = redis_client.get(cache_key)
                if cached:
                    return jsonify({'ok': True, 'insight': cached})
        except Exception:
            pass

        # Try sqlite cache
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        cur.execute('SELECT insight FROM insights WHERE url=?', (url,))
        row = cur.fetchone()
        if row:
            con.close()
            return jsonify({'ok': True, 'insight': row[0]})

        # Fetch page
        resp = safe_get(url, timeout=15)
        resp.raise_for_status()
        content_type = resp.headers.get('content-type', '')
        if 'html' not in content_type and 'text' not in content_type:
            # Non-HTML: return file metadata
            insight = f"Non-HTML resource: {content_type}; size={resp.headers.get('content-length','unknown')}"
            # persist
            now = datetime.utcnow().isoformat()
            try:
                cur.execute('INSERT OR REPLACE INTO insights (url, insight, summary, created_at) VALUES (?,?,?,?)', (url, insight, insight, now))
                con.commit()
            except Exception:
                pass
            con.close()
            try:
                if redis_client:
                    redis_client.set(cache_key, insight, ex=60*60*24)
            except Exception:
                pass
            return jsonify({'ok': True, 'insight': insight})

        html = resp.text
        text = _extract_text_from_html(html)
        summary = _summarize_text_extractive(text, max_sentences=6)
        insight = summary

        now = datetime.utcnow().isoformat()
        try:
            cur.execute('INSERT OR REPLACE INTO insights (url, insight, summary, created_at) VALUES (?,?,?,?)', (url, insight, summary, now))
            con.commit()
        except Exception:
            pass
        con.close()

        try:
            if redis_client:
                redis_client.set(cache_key, insight, ex=60*60*24)
        except Exception:
            pass

        return jsonify({'ok': True, 'insight': insight, 'summary': summary})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500

def _insert_job(job_id, prompt):
    now = now_iso()
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("INSERT OR REPLACE INTO video_jobs (id,prompt,status,created_at,updated_at) VALUES (?,?,?,?,?)",
                (job_id, prompt, 'queued', now, now))
    con.commit(); con.close()

def _update_job(job_id, status=None, result_path=None, error=None):
    now = now_iso()
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    if status:
        cur.execute("UPDATE video_jobs SET status=?, updated_at=? WHERE id=?", (status, now, job_id))
    if result_path:
        cur.execute("UPDATE video_jobs SET result_path=?, status='done', updated_at=? WHERE id= ?", (result_path, now, job_id))
        # cache the result by a video-specific cache key so repeated requests can return quickly
        try:
            cache_key = f"video:{hashlib.sha256(result_path.encode()).hexdigest()}"
            _cache_set(cache_key, result_path, ex=60*60*6)
        except Exception:
            pass
    if error:
        cur.execute("UPDATE video_jobs SET error=?, status='error', updated_at=? WHERE id=?", (error, now, job_id))
    con.commit(); con.close()


def _insert_image_job(job_id, prompt, steps, width, height):
    now = now_iso()
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("INSERT OR REPLACE INTO image_jobs (id,prompt,steps,width,height,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (job_id, prompt, steps, width, height, 'queued', now, now))
    con.commit(); con.close()


def _update_image_job(job_id, status=None, result_path=None, error=None):
    now = now_iso()
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    if status:
        cur.execute("UPDATE image_jobs SET status=?, updated_at=? WHERE id=?", (status, now, job_id))
    if result_path:
        cur.execute("UPDATE image_jobs SET result_path=?, status='done', updated_at=? WHERE id=?", (result_path, now, job_id))
    if error:
        cur.execute("UPDATE image_jobs SET error=?, status='error', updated_at=? WHERE id=?", (error, now, job_id))
    con.commit(); con.close()


def _get_image_job(job_id):
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT id,prompt,steps,width,height,status,result_path,error,created_at,updated_at FROM image_jobs WHERE id=?", (job_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return dict(id=row[0], prompt=row[1], steps=row[2], width=row[3], height=row[4], status=row[5], result_path=row[6], error=row[7], created_at=row[8], updated_at=row[9])

def _get_job(job_id):
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT id,prompt,status,result_path,error,created_at,updated_at FROM video_jobs WHERE id=?", (job_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return dict(id=row[0], prompt=row[1], status=row[2], result_path=row[3], error=row[4], created_at=row[5], updated_at=row[6])


def compute_video_cache_key(prompt, steps, width, height, num_frames, preset='balanced'):
    s = f"{prompt}|{steps}|{width}x{height}|{num_frames}|{preset}"
    return f"video:{hashlib.sha256(s.encode()).hexdigest()}"

def _generate_video_job(job_id, prompt, steps, width, height, num_frames, preset='balanced'):
    try:
        _update_job(job_id, status='running')
        # call existing generation logic but write to file — replicate the code path
        # generate keyframe
        import torch
        if torch.cuda.is_available():
            with torch.inference_mode():
                with torch.autocast('cuda'):
                    img_result = image_pipeline(prompt=prompt, num_inference_steps=steps, width=width, height=height)
        else:
            with torch.inference_mode():
                img_result = image_pipeline(prompt=prompt, num_inference_steps=steps, width=width, height=height)
        init_image = img_result.images[0]

        # generate frames
        if torch.cuda.is_available():
            with torch.inference_mode():
                with torch.autocast('cuda'):
                    video_result = video_pipeline(image=init_image, num_frames=num_frames, decode_chunk_size=8)
        else:
            with torch.inference_mode():
                video_result = video_pipeline(image=init_image, num_frames=num_frames, decode_chunk_size=8)

        video = video_result.frames[0] if hasattr(video_result, 'frames') else video_result

        # write to disk
        os.makedirs(DATA_DIR, exist_ok=True)
        safe_ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        fname = f"video_{safe_ts}_{job_id}.mp4"
        file_path = os.path.join(DATA_DIR, fname)
        try:
            import imageio.v3 as iio
            buf = io.BytesIO()
            with iio.imopen(buf, 'w', plugin='pyav', file_format='mp4') as writer:
                for frame in video:
                    if hasattr(frame, 'convert'):
                        frame = np.array(frame.convert('RGB'))
                    writer.append(frame)
            data_bytes = buf.getvalue()
        except Exception:
            import imageio
            buf = io.BytesIO()
            imageio.mimsave(buf, video, format='mp4', fps=8)
            data_bytes = buf.getvalue()
        with open(file_path, 'wb') as f:
            f.write(data_bytes)

        _update_job(job_id, result_path=fname)
    except Exception as e:
        tb = traceback.format_exc()
        _update_job(job_id, error=str(e) + '\n' + tb)


def _generate_image_job(job_id, prompt, steps, width, height):
    try:
        _update_image_job(job_id, status='running')
        import torch
        if image_pipeline is None:
            raise RuntimeError('Image pipeline is not available')

        if torch.cuda.is_available():
            with torch.inference_mode():
                with torch.autocast('cuda'):
                    result = image_pipeline(prompt=prompt, num_inference_steps=steps, width=width, height=height)
        else:
            with torch.inference_mode():
                result = image_pipeline(prompt=prompt, num_inference_steps=steps, width=width, height=height)

        img = result.images[0]
        # Save image to disk
        os.makedirs(DATA_DIR, exist_ok=True)
        fname = f"img_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{job_id}.png"
        path = os.path.join(DATA_DIR, fname)
        try:
            img.save(path)
        except Exception:
            # try PIL conversion
            from PIL import Image
            arr = np.array(img.convert('RGB')) if hasattr(img, 'convert') else np.array(img)
            Image.fromarray(arr).save(path)

        _update_image_job(job_id, result_path=fname, status='done')
    except Exception as e:
        tb = traceback.format_exc()
        _update_image_job(job_id, error=str(e) + '\n' + tb)


# -----------------------------
# Fast remote provider helpers (Replicate scaffold)
# -----------------------------
def _call_replicate_image(prompt, model='stability-ai/stable-diffusion', **kwargs):
    """Call Replicate's image generation model. Return dict with 'ok' and 'url' or 'data'."""
    if not REPLICATE_API_TOKEN:
        return {'ok': False, 'error': 'Replicate token not configured'}
    headers = {
        'Authorization': f'Token {REPLICATE_API_TOKEN}',
        'Content-Type': 'application/json'
    }
    payload = {
        'version': kwargs.get('version'),
        'input': {
            'prompt': prompt,
            **{k: v for k, v in kwargs.items() if k not in ('version',)}
        }
    }
    try:
        resp = safe_post('https://api.replicate.com/v1/predictions', headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return {'ok': True, 'data': data}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def _call_replicate_video(prompt, model='stability-ai/stable-video-diffusion', **kwargs):
    if not REPLICATE_API_TOKEN:
        return {'ok': False, 'error': 'Replicate token not configured'}
    headers = {
        'Authorization': f'Token {REPLICATE_API_TOKEN}',
        'Content-Type': 'application/json'
    }
    payload = {
        'version': kwargs.get('version'),
        'input': {
            'prompt': prompt,
            **{k: v for k, v in kwargs.items() if k not in ('version',)}
        }
    }
    try:
        resp = safe_post('https://api.replicate.com/v1/predictions', headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return {'ok': True, 'data': data}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


# Fast endpoints
@app.route('/generate-image-fast', methods=['POST'])
def generate_image_fast():
    payload = request.json or {}
    prompt = payload.get('prompt')
    if not prompt:
        return jsonify({'ok': False, 'error': 'Missing prompt'}), 400
    provider = FAST_PROVIDER or payload.get('provider')
    if not provider:
        return jsonify({'ok': False, 'error': 'No FAST_PROVIDER configured'}), 501
    if provider == 'replicate':
        res = _call_replicate_image(prompt, **payload.get('options', {}))
        if not res.get('ok'):
            return jsonify({'ok': False, 'error': res.get('error')}), 500
        return jsonify({'ok': True, 'provider': 'replicate', 'result': res.get('data')})
    return jsonify({'ok': False, 'error': 'Provider not supported'}), 501


@app.route('/generate-video-fast', methods=['POST'])
def generate_video_fast():
    payload = request.json or {}
    prompt = payload.get('prompt')
    if not prompt:
        return jsonify({'ok': False, 'error': 'Missing prompt'}), 400
    provider = FAST_PROVIDER or payload.get('provider')
    if not provider:
        return jsonify({'ok': False, 'error': 'No FAST_PROVIDER configured'}), 501
    if provider == 'replicate':
        res = _call_replicate_video(prompt, **payload.get('options', {}))
        if not res.get('ok'):
            return jsonify({'ok': False, 'error': res.get('error')}), 500
        return jsonify({'ok': True, 'provider': 'replicate', 'result': res.get('data')})
    return jsonify({'ok': False, 'error': 'Provider not supported'}), 501



# -----------------------------
# Security and Config Validation
# -----------------------------
def validate_environment():
    if not SERPAPI_KEY and not (GOOGLE_API_KEY and GOOGLE_CSE_ID):
        print("Warning: No search API keys set. Search will use DuckDuckGo fallback.")
    return []

validate_environment()


# Minimal file upload endpoint for frontend multi-modal ingestion
from file_analyzer import FileAnalyzer

# Initialize file analyzer
file_analyzer = FileAnalyzer()

@app.route('/api/upload', methods=['POST'])
def api_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'ok': False, 'error': 'No file uploaded'}), 400
        f = request.files['file']
        if not f or f.filename == '':
            return jsonify({'ok': False, 'error': 'Empty filename'}), 400

        # Sanitize and create stored filename
        orig_name = secure_filename(f.filename)
        ext = os.path.splitext(orig_name)[1] or ''
        file_id = str(uuid.uuid4())
        stored_name = f"{file_id}{ext}"
        os.makedirs(DATA_DIR, exist_ok=True)
        save_path = os.path.join(DATA_DIR, stored_name)

        # Save file to disk
        f.save(save_path)
        
        # Analyze the uploaded file
        try:
            analysis_result = file_analyzer.analyze_file(save_path)
        except Exception as e:
            analysis_result = {
                "error": f"Analysis failed: {str(e)}",
                "mime_type": f.mimetype or 'application/octet-stream',
                "size_bytes": os.path.getsize(save_path)
            }
        try:
            if mime == 'application/pdf' or ext.lower() == '.pdf':
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(save_path)
                    text_parts = []
                    for p in range(min(3, doc.page_count)):
                        page = doc.load_page(p)
                        text_parts.append(page.get_text())
                    excerpt = '\n'.join(text_parts)[:2000]
                except Exception:
                    excerpt = None
        except Exception:
            excerpt = None

        resp = {
            'ok': True,
            'fileId': file_id,
            'filename': stored_name,
            'originalName': orig_name,
            'url': f"/files/{stored_name}",
            'analysis': analysis_result
        }
        return jsonify(resp)
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'ok': False, 'error': str(e), 'trace': tb}), 500


# Serve uploaded files (static) from DATA_DIR
@app.route('/files/<path:filename>', methods=['GET'])
def serve_uploaded_file(filename):
    # prevent path traversal
    safe_path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(safe_path):
        return abort(404)
    return send_from_directory(DATA_DIR, filename)

# -----------------------------
# Optional Stable Diffusion
# -----------------------------
image_pipeline = None
sd_error: Optional[str] = None
# Allow skipping heavy model imports (diffusers/torch) in test or CI environments
_SKIP_SD = str(os.environ.get('AION_SKIP_MODEL_LOAD', '')).lower() in ('1', 'true', 'yes')
if _SKIP_SD:
    image_pipeline = None
    video_pipeline = None
    sd_error = 'skipped model load via AION_SKIP_MODEL_LOAD'
    print('[SD] Skipping Stable Diffusion model load (AION_SKIP_MODEL_LOAD set)')
else:
    try:
        from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
        import torch
        import contextlib

        SD_MODEL_ID = os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
        # Set globals for device/dtype so endpoints can reuse them safely
        TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

        # set cuDNN benchmark to potentially improve performance on fixed-size inputs
        try:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
        except Exception:
            pass

        image_pipeline = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=TORCH_DTYPE)
        if torch.cuda.is_available():
            image_pipeline = image_pipeline.to(TORCH_DEVICE)
        try:
            image_pipeline.safety_checker = None
            image_pipeline.feature_extractor = None
        except Exception:
            pass

        # Video pipeline setup
        VIDEO_MODEL_ID = os.environ.get("VIDEO_MODEL_ID", "stabilityai/stable-video-diffusion-img2vid-xt")
        video_pipeline = StableVideoDiffusionPipeline.from_pretrained(VIDEO_MODEL_ID, torch_dtype=TORCH_DTYPE)
        if torch.cuda.is_available():
            video_pipeline = video_pipeline.to(TORCH_DEVICE)

        # Try to enable memory/attention optimizations if available (safe, non-fatal)
        try:
            if hasattr(image_pipeline, "enable_attention_slicing"):
                image_pipeline.enable_attention_slicing()
            if hasattr(video_pipeline, "enable_attention_slicing"):
                video_pipeline.enable_attention_slicing()
        except Exception:
            pass
        try:
            if hasattr(image_pipeline, "enable_xformers_memory_efficient_attention"):
                image_pipeline.enable_xformers_memory_efficient_attention()
            if hasattr(video_pipeline, "enable_xformers_memory_efficient_attention"):
                video_pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            # xformers may not be installed — it's only an optional perf win
            pass

        print(f"[SD] Loaded {SD_MODEL_ID} (cuda={torch.cuda.is_available()})")
    except Exception as e:
        sd_error = str(e)
        image_pipeline = None
        video_pipeline = None
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

# --- NEW: Small in-memory agent state used by SSE/control endpoints ---
agent_status = {"status": "idle", "last_ts": now_iso()}

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
        cur.execute("SELECT COUNT(*) FROM episodic_memory")
        episodic_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM procedural_memory")
        procedural_count = cur.fetchone()[0]
        con.close()
        return mem_count, ins_count, episodic_count, procedural_count
    except Exception as e:
        print(f"Error getting DB counts: {e}")
        return 0, 0, 0, 0

@app.get("/api/health")
def health():
    mem_count, ins_count, episodic_count, procedural_count = _get_db_counts()
    return jsonify({
        "ok": True,
        "ollama_up": _ollama_health_check(),
        "sd_ready": image_pipeline is not None,
        "sd_error": sd_error,
        "memories": mem_count,
        "insights": ins_count,
        "episodic_memories": episodic_count,
        "procedural_memories": procedural_count,
        "redis_connected": redis_client is not None and redis_client.ping()
    })

def _ollama_health_check() -> bool:
    try:
        resp = safe_get("http://localhost:11434/", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


# -----------------------------
# Admin & Diagnostics
# -----------------------------
@app.post('/admin/allow-external')
def admin_allow_external():
    """Toggle AION_ALLOW_EXTERNAL at runtime. Requires ADMIN_KEY in JSON body.
    This only affects the in-memory flag for the running process and will not persist across restarts.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        key = data.get('admin_key')
        enable = data.get('enable')
        if key != ADMIN_KEY:
            return make_error('Unauthorized', 401)
        if enable in (True, '1', 'true', 1, 'yes'):
            global AION_ALLOW_EXTERNAL
            AION_ALLOW_EXTERNAL = True
            return jsonify({'ok': True, 'AION_ALLOW_EXTERNAL': True})
        else:
            AION_ALLOW_EXTERNAL = False
            return jsonify({'ok': True, 'AION_ALLOW_EXTERNAL': False})
    except Exception as e:
        traceback.print_exc()
        return make_error('Failed to toggle external access', 500)


@app.get('/admin/check-dns')
def admin_check_dns():
    """Resolve a set of well-known hostnames to test DNS. Returns the IPs or errors."""
    hosts = ['example.com', 'google.com', 'github.com']
    import socket
    out = {}
    for h in hosts:
        try:
            out[h] = socket.gethostbyname_ex(h)[2]
        except Exception as e:
            out[h] = {'error': str(e)}
    return jsonify({'ok': True, 'dns': out})


@app.get('/admin/check-http')
def admin_check_http():
    """Make quick HTTP GETs to a few endpoints to validate outbound HTTP. Respects AION_ALLOW_EXTERNAL flag."""
    targets = ['https://example.com', 'https://www.google.com', 'https://api.ipify.org?format=json']
    results = {}
    for t in targets:
        try:
            r = safe_get(t, timeout=5)
            results[t] = {'status_code': r.status_code, 'len': len(r.text or '')}
        except Exception as e:
            results[t] = {'error': str(e)}
    return jsonify({'ok': True, 'results': results})


@app.get('/env-info')
def env_info():
    info = {}
    # Torch / CUDA
    try:
        import torch
        info['torch_version'] = getattr(torch, '__version__', 'unknown')
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_version'] = getattr(torch.version, 'cuda', 'unknown')
            try:
                info['cuda_device'] = torch.cuda.get_device_name(0)
            except Exception:
                info['cuda_device'] = 'unknown'
    except Exception as e:
        info['torch_error'] = str(e)

    # Diffusers
    try:
        import diffusers
        info['diffusers_version'] = getattr(diffusers, '__version__', 'unknown')
    except Exception as e:
        info['diffusers_error'] = str(e)

    # ImageIO / PyAV / ffmpeg
    try:
        import imageio
        info['imageio_version'] = getattr(imageio, '__version__', 'unknown')
    except Exception as e:
        info['imageio_error'] = str(e)
    try:
        import imageio.v3 as iio
        info['imageio_v3'] = True
    except Exception:
        info['imageio_v3'] = False
    try:
        import av
        info['pyav_version'] = getattr(av, '__version__', 'unknown')
    except Exception as e:
        info['pyav_error'] = str(e)
    # Check ffmpeg on PATH
    try:
        import shutil
        ffmpeg_path = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
        info['ffmpeg_path'] = ffmpeg_path
    except Exception as e:
        info['ffmpeg_error'] = str(e)

    # Optional perf libs
    try:
        import xformers
        info['xformers_available'] = True
    except Exception:
        info['xformers_available'] = False
    try:
        import numpy as np
        info['numpy_version'] = getattr(np, '__version__', 'unknown')
    except Exception as e:
        info['numpy_error'] = str(e)

    # SD pipeline availability
    info['sd_ready'] = image_pipeline is not None
    info['sd_error'] = sd_error

    return jsonify(info)


@app.get('/ollama/models')
def ollama_models():
    """Probe local Ollama server for available models and return a normalized list.
    This function tries a few common Ollama endpoints and returns the first successful JSON response.
    """
    candidates = [
        'http://localhost:11434/models',
        'http://localhost:11434/api/models',
        'http://localhost:11434/models/list',
        'http://localhost:11434/api/list_models'
    ]
    tried = []
    for url in candidates:
        try:
            print(f"[/ollama/models] Trying {url}")
            resp = safe_get(url, timeout=3)
            tried.append({'url': url, 'status': resp.status_code})
            if resp.status_code != 200:
                continue
            try:
                data = resp.json()
            except Exception:
                # If response is plain text, try to parse lines
                text = resp.text.strip()
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                return jsonify({'ok': True, 'models': lines, 'source': url, 'tried': tried})

            # Normalize a few common shapes
            models = []
            if isinstance(data, dict):
                # Common: {'models': [...]} or {'data': [...]} or keys -> list
                if 'models' in data and isinstance(data['models'], list):
                    models = [m.get('name') if isinstance(m, dict) else m for m in data['models']]
                elif 'data' in data and isinstance(data['data'], list):
                    models = [m.get('id') if isinstance(m, dict) else m for m in data['data']]
                else:
                    # Flatten dict values that are strings
                    for k,v in data.items():
                        if isinstance(v, str):
                            models.append(v)
            elif isinstance(data, list):
                models = [m.get('name') if isinstance(m, dict) else m for m in data]

            return jsonify({'ok': True, 'models': models, 'source': url, 'tried': tried})
        except Exception as e:
            print(f"[/ollama/models] Error probing {url}: {e}")
            tried.append({'url': url, 'error': str(e)})

    return jsonify({'ok': False, 'models': [], 'error': 'Could not reach Ollama on localhost:11434', 'tried': tried}), 502


def _call_ollama_generate(payload, timeout=8, retries=1):
    """Call Ollama's generate endpoint with basic retry and normalization logic.

    Defaults use a short connect/read timeout so callers fail fast when the model
    backend is down or unresponsive. Callers may override timeout if needed.
    """
    base_url = os.environ.get('OLLAMA_BASE_URL')
    if base_url:
        base_url = base_url.rstrip('/')
        urls = [
            f"{base_url}/generate",
            f"{base_url}/api/generate",
            f"{base_url}/v1/generate",
            f"{base_url}/chat",
            f"{base_url}/api/chat",
            f"{base_url}/v1/chat"
        ]
    else:
        urls = [
            'http://localhost:11434/api/generate',
            'http://localhost:11434/generate',
            'http://localhost:11434/v1/generate',
            'http://localhost:11434/chat',
            'http://localhost:11434/api/chat',
            'http://localhost:11434/v1/chat'
        ]
    # Timeouts can be configured via env vars: AION_MODEL_CONNECT_TIMEOUT and AION_MODEL_READ_TIMEOUT
    try:
        connect_to = float(os.environ.get('AION_MODEL_CONNECT_TIMEOUT', '3'))
        read_to = float(os.environ.get('AION_MODEL_READ_TIMEOUT', str(timeout)))
        timeout_value = (connect_to, read_to) if connect_to and read_to else float(read_to)
    except Exception:
        timeout_value = timeout

    last_err = None
    for url in urls:
        for attempt in range(max(1, retries)):
            try:
                print(f"[_call_ollama_generate] POST {url} attempt={attempt} timeout={timeout_value}")
                # allow tuple timeout (connect, read) or single number
                resp = safe_post(url, json=payload, timeout=timeout_value)
                if resp.status_code != 200:
                    last_err = f"status {resp.status_code}: {resp.text}"
                    continue
                try:
                    j = resp.json()
                    return _normalize_model_response(j)
                except Exception:
                    return _normalize_model_response(resp.text)
            except Exception as e:
                last_err = str(e)
                print(f"[_call_ollama_generate] error contacting {url}: {e}")
    # Provide a clearer message including attempted URLs for easier debugging
    raise RuntimeError(f"Failed to call model endpoints ({', '.join(urls)}): {last_err}")


def _call_openai_generate(prompt: str, model: str = None, max_tokens: int = 512, temperature: float = 0.2):
    """Call OpenAI Chat Completions as a fallback when local model is unavailable.

    Returns a dict-like response similar to Ollama JSON (keeps 'response' key with text).
    """
    if not OPENAI_API_KEY:
        raise RuntimeError('OPENAI_API_KEY not set')
    m = model or OPENAI_MODEL or 'gpt-4o-mini'
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'}
    body = {
        'model': m,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': temperature,
    }
    # Respect model timeouts env var
    try:
        connect_to = float(os.environ.get('AION_MODEL_CONNECT_TIMEOUT', '3'))
        read_to = float(os.environ.get('AION_MODEL_READ_TIMEOUT', '15'))
        timeout_value = (connect_to, read_to)
    except Exception:
        timeout_value = 10
    resp = safe_post(url, json=body, headers=headers, timeout=timeout_value)
    resp.raise_for_status()
    j = resp.json()
    # Extract text
    try:
        choices = j.get('choices') or []
        if choices:
            text = choices[0].get('message', {}).get('content') or choices[0].get('text')
        else:
            text = j.get('text') or ''
    except Exception:
        text = str(j)
    return {'response': text, 'source': 'openai'}


def _stream_openai_generate(prompt: str, model: str = None, max_tokens: int = 512, temperature: float = 0.2):
    """Stream OpenAI chat completions (server-side) and yield token strings.

    Yields dict pieces: {'type':'text','data': token, 'delta': True}
    """
    if not OPENAI_API_KEY:
        raise RuntimeError('OPENAI_API_KEY not set')
    m = model or OPENAI_MODEL or 'gpt-4o-mini'
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'}
    body = {
        'model': m,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': temperature,
        'stream': True,
    }
    try:
        connect_to = float(os.environ.get('AION_MODEL_CONNECT_TIMEOUT', '3'))
        read_to = float(os.environ.get('AION_MODEL_READ_TIMEOUT', '60'))
        timeout_value = (connect_to, read_to)
    except Exception:
        timeout_value = (3, 60)

    resp = safe_post(url, json=body, headers=headers, stream=True, timeout=timeout_value)
    resp.raise_for_status()

    # OpenAI streaming uses lines beginning with 'data: '
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            continue
        # Some lines may be like 'data: [DONE]'
        if line.startswith('data:'):
            payload = line[len('data:'):].strip()
        else:
            payload = line

        if payload == '[DONE]':
            break

        try:
            j = json.loads(payload)
            # Extract delta content
            choices = j.get('choices') or []
            if choices:
                delta = choices[0].get('delta', {})
                token = delta.get('content')
                if token:
                    yield {'type': 'text', 'data': token, 'delta': True}
                else:
                    # no token, maybe role or other metadata
                    continue
        except Exception:
            # Not JSON or parse error — forward raw
            yield {'type': 'text', 'data': payload}


def _normalize_model_response(body) -> dict:
    """Normalize various model backend response shapes into a consistent dict.

    Returns a dict: {
        ok: bool,
        result: { text: str, choices: list, model: str|None, finish_reason: str|None, usage: dict, created_at: iso }
        raw: original_body
    }
    This attempts to handle OpenAI shapes, Ollama shapes, and plain text.
    """
    try:
        now = datetime.utcnow().isoformat()
        normalized = {
            'ok': False,
            'result': {
                'text': '',
                'choices': [],
                'model': None,
                'finish_reason': None,
                'usage': {},
                'created_at': now,
            },
            'raw': body,
        }

        # Already normalized
        if isinstance(body, dict) and 'result' in body and 'ok' in body:
            return body

        # OpenAI-like responses
        if isinstance(body, dict) and 'choices' in body and isinstance(body['choices'], list):
            texts = []
            choices = []
            for c in body.get('choices', []):
                if isinstance(c, dict):
                    # Chat completion shape
                    msg = c.get('message') or {}
                    content = msg.get('content') or c.get('text') or ''
                    role = msg.get('role') or c.get('role')
                    finish = c.get('finish_reason')
                    choices.append({'text': content, 'role': role, 'finish_reason': finish})
                    texts.append(content)
                else:
                    choices.append({'text': str(c)})
                    texts.append(str(c))

            normalized['ok'] = True
            normalized['result']['text'] = ''.join(texts).strip()
            normalized['result']['choices'] = choices
            normalized['result']['model'] = body.get('model') or OPENAI_MODEL
            normalized['result']['usage'] = body.get('usage', {}) or {}
            return normalized

        # Ollama / local model shapes
        if isinstance(body, dict):
            if 'response' in body and isinstance(body['response'], str):
                normalized['ok'] = True
                normalized['result']['text'] = body['response']
                normalized['result']['model'] = body.get('model') or body.get('model_name')
                return normalized

            if 'text' in body and isinstance(body['text'], str):
                normalized['ok'] = True
                normalized['result']['text'] = body['text']
                normalized['result']['model'] = body.get('model') or body.get('model_name')
                return normalized

            if 'outputs' in body and isinstance(body['outputs'], list):
                parts = []
                for o in body['outputs']:
                    if isinstance(o, dict):
                        parts.append(o.get('content') or o.get('text') or '')
                    else:
                        parts.append(str(o))
                text = ' '.join([p for p in parts if p]).strip()
                normalized['ok'] = True
                normalized['result']['text'] = text
                normalized['result']['model'] = body.get('model') or body.get('model_name')
                return normalized

            # Fallback: find first non-empty string value
            for k, v in body.items():
                if isinstance(v, str) and v.strip():
                    normalized['ok'] = True
                    normalized['result']['text'] = v.strip()
                    return normalized

        # Plain string
        if isinstance(body, str):
            normalized['ok'] = True
            normalized['result']['text'] = body.strip()
            return normalized

        # Last resort stringify
        normalized['ok'] = True
        normalized['result']['text'] = str(body)
        return normalized
    except Exception as e:
        print(f"[_normalize_model_response] normalization error: {e}")
        return {'ok': False, 'result': {'text': '', 'choices': [], 'model': None, 'created_at': datetime.utcnow().isoformat()}, 'raw': body}


@app.post('/ollama/generate')
def ollama_generate_proxy():
    try:
        data = request.get_json(force=True, silent=True) or {}
        # Minimal validation
        prompt = (data.get('prompt') or '').strip()
        model = data.get('model')
        if not prompt:
            return make_error('Missing prompt', 400)
        payload = {'prompt': prompt, 'stream': False}
        if model:
            payload['model'] = model
        try:
            # Use short timeout so we fail fast when the model backend is unreachable
            body = _call_ollama_generate(payload, timeout=8, retries=1)
            return jsonify({'ok': True, 'body': body})
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[/ollama/generate] Error calling model backend: {e}\n{tb}")
            return make_error('Model backend unreachable. Start local Ollama or set OLLAMA_BASE_URL to a reachable model server.', 502)
    except Exception as e:
        print(f"[/ollama/generate] Request error: {e}")
        return make_error('Invalid request to Ollama proxy', 400)


@app.post('/api/generate/stream')
def api_generate_stream():
    """Streaming proxy endpoint expected by the frontend.
    Forwards the JSON payload to Ollama (or OLLAMA_BASE_URL) and streams
    lines back to the client as NDJSON / text lines.
    """
    data = request.get_json(force=True, silent=True) or {}
    prompt = (data.get('prompt') or data.get('input') or '').strip()
    model = data.get('model')
    if not prompt:
        return make_error('Missing prompt', 400)

    # Build payload for Ollama; keep 'stream': True to request streaming
    payload = {'prompt': prompt, 'stream': True}
    if model:
        payload['model'] = model

    # Determine candidate URLs (same logic as _call_ollama_generate)
    base_url = os.environ.get('OLLAMA_BASE_URL')
    if base_url:
        base_url = base_url.rstrip('/')
        urls = [
            f"{base_url}/generate",
            f"{base_url}/api/generate",
            f"{base_url}/v1/generate",
            f"{base_url}/chat",
            f"{base_url}/api/chat",
            f"{base_url}/v1/chat",
        ]
    else:
        urls = [
            'http://localhost:11434/api/generate',
            'http://localhost:11434/generate',
            'http://localhost:11434/v1/generate',
            'http://localhost:11434/chat',
            'http://localhost:11434/api/chat',
            'http://localhost:11434/v1/chat',
        ]

    # Try each URL until we get a streaming response
    last_err = None
    for url in urls:
        try:
            print(f"[/api/generate/stream] proxying to {url}")
            # connect timeout short (3s) to fail fast; read timeout longer for streaming (60s)
            resp = safe_post(url, json=payload, stream=True, timeout=(3, 60))
            if resp.status_code != 200:
                last_err = f"status {resp.status_code}: {resp.text}"
                continue

            def stream_gen(r):
                try:
                    for raw in r.iter_lines(decode_unicode=True):
                        if raw is None:
                            continue
                        line = raw.decode('utf-8') if isinstance(raw, bytes) else raw
                        if not line:
                            continue

                        # Try to parse JSON lines returned by model backends
                        piece_obj = None
                        try:
                            parsed = json.loads(line)
                            # Common shapes: {'token': '...', 'delta': true}, {'response': '...'}, {'text': '...'}
                            if isinstance(parsed, dict):
                                # If it's an 'error' object, forward as-is
                                if parsed.get('error'):
                                    piece_obj = {'type': 'error', 'data': parsed}
                                # If it looks like a token/delta
                                elif parsed.get('token') or parsed.get('delta') or parsed.get('chunk'):
                                    text = parsed.get('token') or parsed.get('chunk') or parsed.get('data') or ''
                                    piece_obj = {'type': 'text', 'data': text, 'delta': True}
                                # Ollama-like simple response
                                elif 'response' in parsed or 'text' in parsed:
                                    text = parsed.get('response') or parsed.get('text') or ''
                                    piece_obj = {'type': 'text', 'data': text, 'delta': False}
                                else:
                                    # unknown JSON object — forward under 'json' type
                                    piece_obj = {'type': 'json', 'data': parsed}
                            else:
                                piece_obj = {'type': 'text', 'data': str(parsed)}
                        except Exception:
                            # Not JSON: forward as plain text chunk
                            piece_obj = {'type': 'text', 'data': line}

                        try:
                            yield (json.dumps(piece_obj) + '\n')
                        except Exception:
                            yield (json.dumps({'type': 'text', 'data': line}) + '\n')
                except GeneratorExit:
                    return
                except Exception as e:
                    print(f"[/api/generate/stream] stream error: {e}")
                    try:
                        yield json.dumps({'error': str(e)}) + '\n'
                    except Exception:
                        yield 'error\n'

            return Response(stream_gen(resp), mimetype='application/x-ndjson')
        except Exception as e:
            last_err = str(e)
            print(f"[/api/generate/stream] error proxying to {url}: {e}")

    # Return a clear guidance message to the frontend so users know how to fix it
    print(f"[/api/generate/stream] all attempts failed: {last_err}")
    # If OpenAI fallback is configured, try a single non-stream OpenAI call and stream it as one NDJSON piece
    if OPENAI_API_KEY:
        try:
            print("[/api/generate/stream] Attempting OpenAI fallback for streaming endpoint")
            def oa_stream_gen():
                try:
                    for piece in _stream_openai_generate(prompt, model=model, max_tokens=int(os.environ.get('AION_OPENAI_MAX_TOKENS', '512'))):
                        yield (json.dumps(piece) + '\n')
                    # final marker
                    yield (json.dumps({'type': 'meta', 'message': 'openai-stream-done'}) + '\n')
                except Exception as _e:
                    try:
                        yield (json.dumps({'type': 'error', 'data': str(_e)}) + '\n')
                    except Exception:
                        yield ('error\n')

            return Response(oa_stream_gen(), mimetype='application/x-ndjson')
        except Exception as e:
            print(f"[/api/generate/stream] OpenAI fallback error: {e}")

    # Final fallback: return the simple test stream so frontend can display something useful
    try:
        print("[/api/generate/stream] Returning test stream fallback")
        def test_gen():
            messages = [
                {'type': 'meta', 'message': 'test-stream-start'},
                {'type': 'text', 'data': 'Streaming fallback: model backend unreachable.'},
                {'type': 'text', 'data': 'Please start the local model server (Ollama) or set OLLAMA_BASE_URL.'},
                {'type': 'meta', 'message': 'test-stream-end'},
            ]
            for m in messages:
                yield (json.dumps(m) + '\n')
                time.sleep(0.15)
        return Response(test_gen(), mimetype='application/x-ndjson')
    except Exception as e:
        print(f"[/api/generate/stream] fallback stream error: {e}")
        return make_error('Streaming model backend unreachable. Please ensure the local model server (e.g., Ollama) is running or set OLLAMA_BASE_URL.', 502)


@app.post('/api/generate')
def api_generate():
    """Non-streaming generate endpoint expected by some frontends.
    Proxies the request to the same Ollama-style endpoints used by the streaming proxy.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = (data.get('prompt') or data.get('input') or '').strip()
        model = data.get('model')
        if not prompt:
            return make_error('Missing prompt', 400)

        payload = {'prompt': prompt, 'stream': False}
        if model:
            payload['model'] = model

        try:
            # Fail fast by default; allow caller to override if they need longer
            body = _call_ollama_generate(payload, timeout=8, retries=1)
            normalized = _normalize_model_response(body)
            # Provide a clear standardized shape: top-level `result` (object) and `raw`
            out = {'ok': True, 'result': normalized.get('result'), 'raw': normalized.get('raw')}
            return jsonify(out)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[/api/generate] Ollama error: {e}\n{tb}")
            # Attempt OpenAI fallback if configured
            if OPENAI_API_KEY:
                try:
                    print("[/api/generate] Falling back to OpenAI generate")
                    openai_res = _call_openai_generate(prompt, model=model)
                    return jsonify({'ok': True, 'body': openai_res, 'fallback': 'openai'})
                except Exception as e2:
                    print(f"[/api/generate] OpenAI fallback error: {e2}")
                    return make_error('Both local model backend and OpenAI fallback failed. See server logs for details.', 502)
            return make_error('Model backend unreachable. Start local Ollama or set OLLAMA_BASE_URL to a reachable server.', 502)
    except Exception as e:
        print(f"[/api/generate] Request error: {e}")
        return make_error('Invalid request to generate', 400)


@app.post('/api/generate/test')
def api_generate_test():
    """Simple test streaming endpoint that emits a few NDJSON lines so the frontend
    streaming parser can be validated without a model backend.
    """
    def gen():
        try:
            messages = [
                {'type': 'meta', 'message': 'test-stream-start'},
                {'type': 'text', 'data': 'This is a test chunk 1.'},
                {'type': 'text', 'data': 'This is a test chunk 2.'},
                {'type': 'text', 'data': 'Final chunk.'},
                {'type': 'meta', 'message': 'test-stream-end'},
            ]
            for m in messages:
                yield (json.dumps(m) + '\n')
                time.sleep(0.2)
        except GeneratorExit:
            return

    return Response(gen(), mimetype='application/x-ndjson')


@app.get('/api/status/providers')
def api_status_providers():
    """Return a simple list of configured/assumed providers for the frontend to display."""
    try:
        providers = []
        # Ollama: assume available if OLLAMA_BASE_URL set or default local port is used by proxies
        base = os.environ.get('OLLAMA_BASE_URL')
        if base or True:
            providers.append('ollama')
        # OpenAI
        if OPENAI_API_KEY:
            providers.append('openai')
        # Fast providers (replicate etc.)
        if FAST_PROVIDER:
            providers.append(FAST_PROVIDER)

        return jsonify({'ok': True, 'providers': providers})
    except Exception as e:
        print(f"[/api/status/providers] Error: {e}")
        return make_error('Error checking providers', 500)


@app.get('/debug/ddg')
def debug_ddg_search():
    """Simple debug endpoint: run a DuckDuckGo (DDGS) search and return top results.
    Example: /debug/ddg?q=python+flask
    """
    try:
        q = (request.args.get('q') or '').strip()
        if not q:
            return make_error('Missing query param q', 400)
        if DDGS is None:
            return make_error('ddgs package not installed', 500)
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(q, timelimit=5):
                    results.append({'title': r.get('title'), 'url': r.get('url'), 'body': r.get('body')})
                    if len(results) >= 10:
                        break
        except Exception as e:
            print(f"[/debug/ddg] ddgs search error: {e}")
            return make_error('DDG search failed: ' + str(e), 502)
        return jsonify({'ok': True, 'query': q, 'results': results})
    except Exception as e:
        print(f"[/debug/ddg] Error: {e}")
        return make_error('Internal error', 500)
        return make_error('Failed to enumerate providers', 500)


@app.get('/health')
def health_check():
    """Basic health endpoint that checks the server and model backend reachability.
    Useful for quick debugging when the frontend reports the chat API is unresponsive.
    """
    status = {'ok': True, 'services': {}}
    # Redis
    try:
        status['services']['redis'] = bool(redis_client and redis_client.ping())
    except Exception:
        status['services']['redis'] = False

    # Database file exists?
    try:
        status['services']['sqlite_db'] = os.path.exists(DB_FILE)
    except Exception:
        status['services']['sqlite_db'] = False

    # Ollama/model backend quick check
    try:
        base = os.environ.get('OLLAMA_BASE_URL') or 'http://localhost:11434'
        url = base.rstrip('/') + '/api/generate'
        # Small probe with short timeout
        try:
            resp = safe_post(url, json={'prompt': 'ping', 'stream': False}, timeout=5)
            status['services']['model_backend'] = (resp.status_code == 200)
            try:
                status['services']['model_backend_body'] = resp.text[:200]
            except Exception:
                status['services']['model_backend_body'] = None
        except Exception as e:
            status['services']['model_backend'] = False
            status['services']['model_backend_error'] = str(e)
    except Exception as e:
        status['services']['model_backend'] = False
        status['services']['model_backend_error'] = str(e)

    return jsonify(status)


@app.get('/files/<path:filename>')
def serve_file(filename: str):
    # Serve files from DATA_DIR only. Reject traversal attempts.
    if '..' in filename or filename.startswith('/'):
        abort(400)
    full_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(full_path):
        return make_error('File not found', 404)
    return send_from_directory(DATA_DIR, filename, as_attachment=True)


@app.get('/files/assets/<path:filename>')
def serve_asset_file(filename: str):
    # serve from DATA_DIR/assets
    if '..' in filename or filename.startswith('/'):
        abort(400)
    assets_dir = os.path.join(DATA_DIR, 'assets')
    full_path = os.path.join(assets_dir, filename)
    if not os.path.exists(full_path):
        return make_error('Asset not found', 404)
    return send_from_directory(assets_dir, filename, as_attachment=True)


# New: fetch external asset and store locally, returning a served URL
@app.post('/api/fetch-asset')
@limiter.limit('20 per hour')
def fetch_asset():
    try:
        data = request.get_json(force=True, silent=True) or {}
        url = data.get('url')
        suggested_name = data.get('filename')
        if not url:
            return make_error('Missing url in request body', 400)

        # Basic validation
        if not (url.startswith('http://') or url.startswith('https://')):
            return make_error('Only http/https URLs are supported', 400)

        # Create assets dir
        assets_dir = os.path.join(DATA_DIR, 'assets')
        os.makedirs(assets_dir, exist_ok=True)

        # Try to determine filename
        if suggested_name:
            base = secure_filename(suggested_name)
        else:
            # extract from URL path
            base = secure_filename(url.split('/')[-1]) or 'asset'

        # If no extension, try to infer later from content-type
        name = base
        dest_path = os.path.join(assets_dir, name)

        # Avoid collisions by appending short uuid if exists
        if os.path.exists(dest_path):
            name = f"{os.path.splitext(base)[0]}_{uuid.uuid4().hex[:8]}{os.path.splitext(base)[1]}"
            dest_path = os.path.join(assets_dir, name)

        # Stream download with size limit (100MB)
        MAX_BYTES = int(os.environ.get('MAX_ASSET_BYTES', 100 * 1024 * 1024))
        headers = {'User-Agent': 'AION/1.0 (+https://example.local)'}
        with safe_get(url, stream=True, timeout=30, headers=headers) as r:
            r.raise_for_status()
            total = 0
            # If server provides content-length and it's too large, abort
            cl = r.headers.get('Content-Length')
            if cl and int(cl) > MAX_BYTES:
                return make_error('Remote file too large', 413)

            # Write to a temp file first
            tmp_name = name + '.download'
            tmp_path = os.path.join(assets_dir, tmp_name)
            with open(tmp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_BYTES:
                        f.close()
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                        return make_error('Remote file exceeded maximum allowed size', 413)
                    f.write(chunk)

        # Optionally inspect content-type and rename to add extension
        content_type = ''
        try:
            head = requests.head(url, timeout=6, headers=headers)
            content_type = head.headers.get('Content-Type', '')
        except Exception:
            content_type = r.headers.get('Content-Type', '') if 'r' in locals() else ''

        # If file has no extension but content-type known, add extension
        if os.path.splitext(name)[1] == '' and content_type:
            ext = ''
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            elif 'gif' in content_type:
                ext = '.gif'
            elif 'pdf' in content_type:
                ext = '.pdf'
            elif 'mp4' in content_type or 'video' in content_type:
                ext = '.mp4'
            if ext:
                final_name = name + ext
            else:
                final_name = name
        else:
            final_name = name

        final_path = os.path.join(assets_dir, final_name)
        # move tmp to final (handle rename if changed)
        try:
            os.rename(tmp_path, final_path)
        except Exception:
            # fallback to copy
            with open(tmp_path, 'rb') as src, open(final_path, 'wb') as dst:
                dst.write(src.read())
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        # Return served URL
        served_url = request.host_url.rstrip('/') + f"/files/assets/{final_name}"
        return jsonify({'ok': True, 'filename': final_name, 'url': served_url, 'size': os.path.getsize(final_path), 'content_type': content_type})

    except requests.HTTPError as e:
        tb = traceback.format_exc()
        return make_error(f'Failed to fetch remote asset: {str(e)}', 502)
    except Exception as e:
        tb = traceback.format_exc()
        print('[/api/fetch-asset] Exception', e, tb)
        return make_error('Internal server error while fetching asset', 500)

# -----------------------------
# NEW: Simple retrieval endpoint used by frontend RAG (returns 'contexts' array)
# -----------------------------
@app.post('/api/retrieve')
def api_retrieve():
    try:
        data = request.get_json(force=True, silent=True) or {}
        query = (data.get('query') or '').strip()
        if not query:
            return make_error("Missing 'query' in request body.", 400)

        # Very small retrieval: search insights and memories for the query keywords
        keywords = [w for w in re.findall(r'\w{3,}', query.lower())][:6]
        contexts = []

        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        # Search memories
        try:
            q = '%' + '%'.join(keywords) + '%'
            cur.execute("SELECT text, source, ts FROM memories WHERE text LIKE ? LIMIT 6", (q,))
            for row in cur.fetchall():
                contexts.append(f"Memory ({row[2]}): {row[0][:800]}")
        except Exception:
            pass

        # Search insights
        try:
            q = '%' + '%'.join(keywords) + '%'
            cur.execute("SELECT insight, created_at FROM insights WHERE insight LIKE ? LIMIT 4", (q,))
            for row in cur.fetchall():
                contexts.append(f"Insight ({row[1]}): {row[0][:800]}")
        except Exception:
            pass

        con.close()

        # Fallback: if nothing found, return a short hint/fallback contexts
        if not contexts:
            contexts = [
                "Document: Project AION Design Notes — core objectives: soulful AI, context-aware responses.",
                "Memory: No close matches found in local memory index."
            ]

        return jsonify({"ok": True, "contexts": contexts[:10]})
    except Exception as e:
        traceback.print_exc()
        return make_error("Failed to retrieve contexts", 500)


# -----------------------------
# NEW: Lightweight analyze-file endpoint (used after upload)
# -----------------------------
@app.post('/api/analyze-file')
def api_analyze_file():
    try:
        data = request.get_json(force=True, silent=True) or {}
        file_url = data.get('file_url') or data.get('url') or ''
        if not file_url:
            return make_error("Missing 'file_url' in request body.", 400)

        excerpt = ""
        summary = ""
        # If local files path (served by /files), read from disk
        try:
            if file_url.startswith('/files/') or file_url.startswith(request.host_url.rstrip('/')):
                # normalize to filename
                fname = file_url.split('/')[-1]
                fpath = os.path.join(DATA_DIR, fname)
                if os.path.exists(fpath):
                    # try to read small text or first bytes for PDF/other
                    try:
                        with open(fpath, 'rb') as fh:
                            data_bytes = fh.read(200000)
                            txt = None
                            try:
                                txt = data_bytes.decode('utf-8', errors='ignore')
                            except Exception:
                                txt = None
                            excerpt = (txt or '')[:2000]
                    except Exception:
                        excerpt = ""
        except Exception:
            excerpt = ""

        # If no local excerpt, try to fetch remote (with safe_get)
        if not excerpt:
            try:
                ok, filename, err = _save_file_from_url(file_url, max_bytes=5 * 1024 * 1024)
                if ok and filename:
                    fpath = os.path.join(DATA_DIR, filename)
                    try:
                        with open(fpath, 'rb') as fh:
                            txt = fh.read(200000).decode('utf-8', errors='ignore')
                            excerpt = txt[:2000]
                    except Exception:
                        excerpt = ""
                else:
                    # attempt one-shot HTTP fetch for small pages
                    r = safe_get(file_url, timeout=6)
                    r.raise_for_status()
                    if 'html' in (r.headers.get('content-type') or ''):
                        text = _extract_text_from_html(r.text)
                        excerpt = text[:2000]
                    else:
                        excerpt = f"Remote resource: {r.headers.get('content-type','unknown')}"
            except Exception:
                excerpt = ""

        # Summarize excerpt
        try:
            summary = _summarize_text_extractive(excerpt or (file_url or ""), max_sentences=4)
        except Exception:
            summary = (excerpt or "")[:1000]

        # Build a minimal serverAnalysis object consistent with frontend expectations
        analysis = {
            "status": "done",
            "excerpt": excerpt,
            "analysis": {
                "ai_understanding": {
                    "summary": summary,
                    "auto_generated": True
                },
                "content_summary": summary
            }
        }
        return jsonify({"ok": True, "analysis": analysis})
    except Exception as e:
        traceback.print_exc()
        return make_error("Failed to analyze file", 500)


# -----------------------------
# NEW: Simple index-file endpoint (accepts file upload or URL and returns index metadata)
# -----------------------------
@app.post('/api/index-file')
def api_index_file():
    try:
        # Accept multipart/form-data file upload OR JSON { file_url: "..." }
        if 'file' in request.files:
            f = request.files['file']
            orig = secure_filename(f.filename or 'uploaded')
            ext = os.path.splitext(orig)[1] or ''
            file_id = uuid.uuid4().hex
            stored_name = f"{file_id}{ext}"
            fpath = os.path.join(DATA_DIR, stored_name)
            os.makedirs(DATA_DIR, exist_ok=True)
            f.save(fpath)
            excerpt = ''
            try:
                with open(fpath, 'rb') as fh:
                    excerpt = fh.read(200000).decode('utf-8', errors='ignore')[:2000]
            except Exception:
                excerpt = ''
            # Insert a lightweight index entry (into insights table for simplicity)
            con = sqlite3.connect(DB_FILE)
            cur = con.cursor()
            try:
                now = now_iso()
                cur.execute("INSERT INTO insights (text, ts) VALUES (?, ?)", (excerpt or f"Indexed file: {orig}", now))
                con.commit()
            except Exception:
                pass
            con.close()
            return jsonify({"ok": True, "indexed": True, "filename": stored_name, "url": f"/files/{stored_name}"})

        data = request.get_json(force=True, silent=True) or {}
        file_url = data.get('file_url') or data.get('url') or data.get('fileUrl') or ''
        if not file_url:
            return make_error("Missing file upload or 'file_url' in body", 400)

        # Try to fetch & cache small content or just register the URL in insights
        try:
            ok, filename, err = _save_file_from_url(file_url, max_bytes=5 * 1024 * 1024)
            if ok and filename:
                text_excerpt = ""
                try:
                    with open(os.path.join(DATA_DIR, filename), 'rb') as fh:
                        text_excerpt = fh.read(200000).decode('utf-8', errors='ignore')[:2000]
                except Exception:
                    text_excerpt = ""
                con = sqlite3.connect(DB_FILE)
                cur = con.cursor()
                cur.execute("INSERT INTO insights (text, ts) VALUES (?, ?)", (text_excerpt or f"Indexed url: {file_url}", now_iso()))
                con.commit(); con.close()
                return jsonify({"ok": True, "indexed": True, "filename": filename, "url": f"/files/{filename}"})
        except Exception:
            pass

        # fallback: register URL only
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        cur.execute("INSERT INTO insights (text, ts) VALUES (?, ?)", (f"Indexed url: {file_url}", now_iso()))
        con.commit(); con.close()
        return jsonify({"ok": True, "indexed": True, "url": file_url})
    except Exception as e:
        traceback.print_exc()
        return make_error("Failed to index file", 500)


# -----------------------------
# NEW: Simple sync endpoint used by the frontend to POST conversation data
# -----------------------------
@app.post('/api/sync-conversation')
def api_sync_conversation():
    try:
        data = request.get_json(force=True, silent=True) or {}
        # For local dev just accept and return counts; optionally persist as insight
        try:
            conv = data.get('conversation') or []
            # small persistence: store a JSON summary as an insight
            if conv:
                txt = json.dumps(conv[-5:], ensure_ascii=False)[:2000]
                con = sqlite3.connect(DB_FILE)
                cur = con.cursor()
                cur.execute("INSERT INTO insights (text, ts) VALUES (?, ?)", (f"Synced conversation snippet: {txt}", now_iso()))
                con.commit(); con.close()
        except Exception:
            pass
        return jsonify({"ok": True, "synced": True})
    except Exception as e:
        traceback.print_exc()
        return make_error("Failed to sync conversation", 500)


# -----------------------------
# NEW: Agent control and SSE stream endpoints expected by frontend
# -----------------------------
@app.post('/api/agent/control')
def api_agent_control():
    try:
        data = request.get_json(force=True, silent=True) or {}
        action = (data.get('action') or '').lower()
        if action == 'pause':
            agent_status['status'] = 'paused'
        elif action == 'resume':
            agent_status['status'] = 'running'
        elif action == 'stop':
            agent_status['status'] = 'stopped'
        else:
            return make_error("Unknown action", 400)
        agent_status['last_ts'] = now_iso()
        return jsonify({"ok": True, "status": agent_status['status']})
    except Exception as e:
        traceback.print_exc()
        return make_error("Failed to control agent", 500)


@app.get('/api/agent/stream')
def api_agent_stream():
    """Simple SSE stream that emits agent status and occasional events.
    Frontend expects EventSource; this produces 'message' events containing JSON."""
    def gen():
        try:
            # initial event
            yield f"data: {json.dumps({'type':'status','status': agent_status['status'], 'ts': now_iso()})}\n\n"
            count = 0
            # simple loop that yields heartbeat + occasional fake events for UI
            while True:
                time.sleep(5)
                count += 1
                # heartbeat status
                yield f"data: {json.dumps({'type':'status','status': agent_status['status'], 'ts': now_iso()})}\n\n"
                # emit an event every 3 cycles for UI demonstration
                if count % 3 == 0:
                    ev = {'type':'event','id': uuid.uuid4().hex, 'ts': now_iso(), 'message': f'Agent heartbeat #{count}', 'level': 'info'}
                    yield f"data: {json.dumps(ev)}\n\n"
        except GeneratorExit:
            return
        except Exception as e:
            try:
                yield f"data: {json.dumps({'type':'error','error': str(e)})}\n\n"
            except Exception:
                pass
    return Response(gen(), mimetype='text/event-stream')


# -----------------------------
# NEW: Lightweight check-updates endpoint for frontend polling
# -----------------------------
@app.get('/api/check-updates')
def api_check_updates():
    try:
        # Minimal response: no forced updates by default
        return jsonify({"updateAvailable": False, "version": None, "note": "No automatic updates configured."})
    except Exception:
        return make_error("Failed to check updates", 500)

# Endpoints /generate-code and /generate-image have no changes
@app.post("/generate-code")
def generate_code():
    try:
        # Log incoming request payload for easier debugging
        raw = request.get_data(as_text=True)
        print("[/generate-code] Raw request body:", raw)
        data = request.get_json(force=True, silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        model = (data.get("model") or "llama3").strip()
        
        if not prompt: return make_error("Missing 'prompt' in request body.", 400)
        if len(prompt) > 10000: return make_error("Prompt too long.", 400)
            
        # Use centralized Ollama proxy helper
        payload = {"model": model, "prompt": prompt, "stream": False}
        try:
            body = _call_ollama_generate(payload, timeout=180, retries=2)
        except Exception as e:
            print(f"[/generate-code] Ollama proxy error: {e}")
            tb = traceback.format_exc()
            print(tb)
            return make_error(f"Ollama generate failed: {str(e)}", 502)

        # The proxy may return {'response': '...'} or {'body': {...}}
        if isinstance(body, dict):
            text = (body.get('response') or body.get('body') or '')
            if isinstance(text, dict):
                text = (text.get('response') or '')
            text = (text or '').strip()
        else:
            text = str(body).strip()

        if not text:
            print("[/generate-code] Ollama returned empty response via proxy:", body)
            return make_error("Ollama returned empty response.", 502)

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
            import torch
            # Build cache key from prompt+params
            cache_key = f"img:{hashlib.sha256(f'{prompt}|{steps}|{width}x{height}'.encode()).hexdigest()}"
            cached = _cache_get(cache_key)
            if cached:
                return jsonify({"imageUrl": cached, "note": "from_cache"})

            t0 = time.time()
            # Use inference_mode + autocast for faster, memory-efficient inference when CUDA is available
            if torch.cuda.is_available():
                with torch.inference_mode():
                    with torch.autocast("cuda"):
                        result = image_pipeline(prompt=prompt, num_inference_steps=steps, width=width, height=height)
            else:
                with torch.inference_mode():
                    result = image_pipeline(prompt=prompt, num_inference_steps=steps, width=width, height=height)
            img = result.images[0]
            t1 = time.time()
            print(f"[/generate-image] Inference time: {t1-t0:.2f}s")

            # If caller requests 'save' in query params or JSON, write file and return URL
            save = False
            try:
                save = bool(int(request.args.get('save') or (request.json or {}).get('save') or 0))
            except Exception:
                save = False

            if save:
                os.makedirs(DATA_DIR, exist_ok=True)
                fname = f"img_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256((prompt+str(time.time())).encode()).hexdigest()[:8]}.png"
                fpath = os.path.join(DATA_DIR, fname)
                try:
                    img.save(fpath)
                except Exception:
                    try:
                        from PIL import Image
                        arr = np.array(img.convert('RGB')) if hasattr(img, 'convert') else np.array(img)
                        Image.fromarray(arr).save(fpath)
                    except Exception:
                        pass
                file_url = request.host_url.rstrip('/') + f"/files/{fname}"
                try:
                    _cache_set(cache_key, file_url, ex=1800)
                except Exception:
                    pass
                return jsonify({"imageUrl": file_url, "note": "saved_to_disk"})

            # default: return small base64 data URL (may be large)
            data_url = pil_to_dataurl(img)
            try:
                _cache_set(cache_key, data_url, ex=1800)
            except Exception:
                pass
            return jsonify({"imageUrl": data_url})
        except Exception as e:
            print("[/generate-image] error:", e); traceback.print_exc()
            return make_error("Failed to generate image.", 500)
    except Exception as e:
        print("[/generate-image] Exception:", e); traceback.print_exc()
        return make_error("Internal server error in generate-image.", 500)


@app.post('/generate-image-async')
def generate_image_async():
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = (data.get('prompt') or '').strip()
        steps = int(data.get('steps') or 30)
        width = int(data.get('width') or 512)
        height = int(data.get('height') or 512)

        if not prompt:
            return make_error("Missing 'prompt' in request body.", 400)

        job_id = uuid.uuid4().hex
        _insert_image_job(job_id, prompt, steps, width, height)
        # Submit to executor
        executor.submit(_generate_image_job, job_id, prompt, steps, width, height)

        return jsonify({'job_id': job_id, 'status': 'queued', 'job_url': request.host_url.rstrip('/') + f"/image-job/{job_id}"})
    except Exception as e:
        traceback.print_exc()
        return make_error('Failed to queue image job', 500)


@app.get('/image-job/<job_id>')
def image_job_status(job_id: str):
    job = _get_image_job(job_id)
    if not job:
        return make_error('Job not found', 404)
    if job.get('status') == 'done' and job.get('result_path'):
        job['download_url'] = request.host_url.rstrip('/') + f"/files/{job.get('result_path')}"
    return jsonify(job)

# Video generation endpoint
@app.post("/generate-video")
def generate_video():
    try:
        # 1. Log initial request
        print("[/generate-video] Starting video generation request")
        
        # 2. Validate input data
        data = request.get_json(force=True, silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        steps = int(data.get("steps") or 30)
        width = int(data.get("width") or 512)
        height = int(data.get("height") or 512)
        num_frames = int(data.get("num_frames") or 24)
        preset = (data.get('preset') or 'balanced')

        # Apply preset overrides for faster generation
        if preset == 'fast':
            steps = min(20, steps)
            num_frames = min(12, num_frames)
            width = min(512, width)
            height = min(512, height)
        elif preset == 'quality':
            steps = max(steps, 40)
            num_frames = max(num_frames, 32)

        # Check cache first
        try:
            cache_key = compute_video_cache_key(prompt, steps, width, height, num_frames, preset)
            cached = _cache_get(cache_key)
            if cached:
                # Return direct files URL
                files_url = request.host_url.rstrip('/') + f"/files/{cached}"
                return jsonify({"videoUrl": files_url, "note": "from_cache"})
        except Exception:
            pass

        print(f"[/generate-video] Request parameters: prompt='{prompt}', steps={steps}, size={width}x{height}, frames={num_frames}")

        # 3. Input validation
        if not prompt:
            return make_error("Missing 'prompt' in request body.", 400)
        if len(prompt) > 1000:
            return make_error("Prompt too long.", 400)
        if width > 1024 or height > 1024:
            return make_error("Video dimensions too large.", 400)

        # 4. Check pipeline availability
        if video_pipeline is None or image_pipeline is None:
            print("[/generate-video] Pipeline not available. video_pipeline:", video_pipeline is not None, "image_pipeline:", image_pipeline is not None)
            if sd_error:
                print("[/generate-video] SD Error:", sd_error)
            placeholder = f"https://via.placeholder.com/{width}x{height}?text=Video+not+available"
            return jsonify({
                "videoUrl": placeholder, 
                "note": "Stable Video Diffusion not available; returned placeholder.",
                "error_details": sd_error if sd_error else "Pipeline initialization failed"
            })

        try:
            import torch
            # 5. Generate keyframe image
            print("[/generate-video] Generating initial keyframe image...")
            t_start = time.time()
            if torch.cuda.is_available():
                with torch.inference_mode():
                    with torch.autocast("cuda"):
                        img_result = image_pipeline(
                            prompt=prompt,
                            num_inference_steps=steps,
                            width=width,
                            height=height
                        )
            else:
                with torch.inference_mode():
                    img_result = image_pipeline(
                        prompt=prompt,
                        num_inference_steps=steps,
                        width=width,
                        height=height
                    )
            init_image = img_result.images[0]
            t_keyframe = time.time()
            print(f"[/generate-video] Keyframe generated in {t_keyframe-t_start:.2f}s")

            # 6. Generate video
            print("[/generate-video] Starting video generation...")
            # Many video pipelines accept a `generator` / schedule; keep defaults for safety
            if torch.cuda.is_available():
                with torch.inference_mode():
                    with torch.autocast("cuda"):
                        video_result = video_pipeline(
                            image=init_image,
                            num_frames=num_frames,
                            decode_chunk_size=8
                        )
            else:
                with torch.inference_mode():
                    video_result = video_pipeline(
                        image=init_image,
                        num_frames=num_frames,
                        decode_chunk_size=8
                    )
            t_frames = time.time()
            print(f"[/generate-video] Video frames generated in {t_frames-t_keyframe:.2f}s (total {t_frames-t_start:.2f}s)")
            # video_result.frames is typically a list of PIL images or numpy arrays
            video = video_result.frames[0] if hasattr(video_result, 'frames') else video_result

            # 7. Convert to MP4
            print("[/generate-video] Converting to MP4...")
            import imageio.v3 as iio
            # imageio v3 prefers writing to a file path; use BytesIO with v2 API fallback
            try:
                buf = io.BytesIO()
                # try v3 writer
                with iio.imopen(buf, "w", plugin="pyav", file_format="mp4") as writer:
                    for frame in video:
                        # ensure numpy array in HWC uint8
                        if hasattr(frame, 'convert'):
                            frame = np.array(frame.convert('RGB'))
                        writer.append(frame)
                data_bytes = buf.getvalue()
            except Exception:
                # fallback to imageio v2
                import imageio
                buf = io.BytesIO()
                imageio.mimsave(buf, video, format='mp4', fps=8)
                data_bytes = buf.getvalue()

            # Save bytes to disk to avoid sending huge base64 in responses
            os.makedirs(DATA_DIR, exist_ok=True)
            safe_ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            fname = f"video_{safe_ts}_{num_frames}f_{width}x{height}.mp4"
            file_path = os.path.join(DATA_DIR, fname)
            with open(file_path, 'wb') as f:
                f.write(data_bytes)

            # cache final video file by computed key
            try:
                cache_key = compute_video_cache_key(prompt, steps, width, height, num_frames, preset)
                _cache_set(cache_key, fname, ex=60*60*6)
            except Exception:
                pass
            # set cache key for this prompt/params
            try:
                cache_key = compute_video_cache_key(prompt, steps, width, height, num_frames, preset)
                _cache_set(cache_key, fname, ex=60*60*6)
            except Exception:
                pass
            t_done = time.time()
            print(f"[/generate-video] Video saved to {file_path} in {t_done-t_frames:.2f}s (end-to-end {t_done-t_start:.2f}s)")

            # Return a files endpoint URL (client should GET this URL to download)
            files_url = request.host_url.rstrip('/') + f"/files/{fname}"
            return jsonify({
                "videoUrl": files_url,
                "success": True,
                "details": {
                    "frames": num_frames,
                    "size": f"{width}x{height}",
                    "steps": steps,
                    "timings": {
                        "keyframe": f"{t_keyframe-t_start:.2f}s",
                        "frames": f"{t_frames-t_keyframe:.2f}s",
                        "conversion": f"{t_done-t_frames:.2f}s",
                        "total": f"{t_done-t_start:.2f}s"
                    }
                }
            })

        except Exception as e:
            print("[/generate-video] Generation error:", str(e))
            traceback.print_exc()
            # Check CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated(0)
                memory_reserved = torch.cuda.memory_reserved(0)
                print(f"[/generate-video] CUDA memory status - Allocated: {memory_allocated/1024**2:.1f}MB, Reserved: {memory_reserved/1024**2:.1f}MB")
            return make_error(f"Failed to generate video: {str(e)}", 500)

    except Exception as e:
        print("[/generate-video] Request processing error:", str(e))
        tb = traceback.format_exc()
        print(tb)
        # Return an explicit error response so clients receive JSON with details
        return make_error(f"Internal error processing request: {str(e)}\n{tb}", 500)


@app.post('/generate-video-async')
def generate_video_async():
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = (data.get('prompt') or '').strip()
        steps = int(data.get('steps') or 30)
        width = int(data.get('width') or 512)
        height = int(data.get('height') or 512)
        num_frames = int(data.get('num_frames') or 24)
        preset = (data.get('preset') or 'balanced')

        # enforce limits
        if not prompt:
            return make_error("Missing 'prompt' in request body.", 400)
        if steps > MAX_STEPS or num_frames > MAX_FRAMES or width > MAX_WIDTH or height > MAX_HEIGHT:
            return make_error(f"Request exceeds server limits (max_steps={MAX_STEPS}, max_frames={MAX_FRAMES}, max_size={MAX_WIDTH}x{MAX_HEIGHT}).", 400)

        # create job
        job_id = uuid.uuid4().hex
        _insert_job(job_id, prompt)
        # submit to executor
        executor.submit(_generate_video_job, job_id, prompt, steps, width, height, num_frames, preset)

        return jsonify({"job_id": job_id, "status": "queued", "job_url": request.host_url.rstrip('/') + f"/job/{job_id}"})
    except Exception as e:
        traceback.print_exc()
        return make_error('Failed to queue video job', 500)


@app.get('/job/<job_id>')
def job_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        return make_error('Job not found', 404)
    if job.get('status') == 'done' and job.get('result_path'):
        job['download_url'] = request.host_url.rstrip('/') + f"/files/{job.get('result_path')}"
    return jsonify(job)

# -----------------------------
# MODIFIED: /api/search with enhanced image results
# -----------------------------
@app.get("/api/search")
def api_search():
    q = (request.args.get("query") or "").strip()
    if not q: return make_error("Missing 'query' parameter.", 400)
    if len(q) > 500: return make_error("Query too long.", 400)
        
    try:
        t_start = time.time()
        # Log search query for analytics
        try:
            con = sqlite3.connect(DB_FILE)
            cur = con.cursor()
            cur.execute("INSERT INTO search_history (query, result_count, timestamp) VALUES (?, ?, ?)",
                      (q, 0, now_iso()))
            con.commit()
            con.close()
        except Exception as e:
            print(f"Failed to log search query: {e}")

        # Try Google CSE first if configured
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
            try:
                t0 = time.time()
                resp = safe_get(google_url, params=params, timeout=6)
                t_google = time.time() - t0
                print(f"[/api/search] Google CSE time: {t_google:.2f}s")
                data = resp.json()

                results = []
                # Process regular search results
                for item in data.get("items", []):
                    # Check if this is an image result
                    try:
                        mime = item.get("mime", "")
                    except Exception:
                        mime = ''
                    if isinstance(mime, str) and mime.startswith("image/"):
                        results.append({
                            "title": item.get("title", "Image result"),
                            "url": item.get("image", {}).get("contextLink", "#"),
                            "snippet": item.get("snippet", ""),
                            "image": item.get("link"),  # Direct image URL
                            "isImage": True
                        })
                    else:
                        img = None
                        try:
                            if "pagemap" in item and "cse_image" in item["pagemap"]:
                                img = item["pagemap"]["cse_image"][0].get("src")
                        except Exception:
                            img = None
                        results.append({
                            "title": item.get("title"),
                            "url": item.get("link"),
                            "snippet": item.get("snippet"),
                            "image": img,
                            "isImage": False
                        })

                total_time = time.time() - t_start
                print(f"[/api/search] Completed Google CSE path in {total_time:.2f}s")
                return jsonify({"query": q, "results": results})
            except Exception as e:
                print(f"[/api/search] Google CSE failed: {e}")
                # fallthrough to try SERPAPI or duckduckgo below
        
        # Next try SerpAPI if available
        if SERPAPI_KEY:
            params = {"q": q, "api_key": SERPAPI_KEY, "hl": "en"}
            try:
                t0 = time.time()
                resp = safe_get("https://serpapi.com/search", params=params, timeout=6)
                t_serp = time.time() - t0
                print(f"[/api/search] SerpAPI time: {t_serp:.2f}s")
                data = resp.json()
                results = []
                for item in data.get("organic_results", []):
                    results.append({"title": item.get("title"), "url": item.get("link"), "snippet": item.get("snippet"), "image": item.get("thumbnail")})
                return jsonify({"query": q, "results": results})
            except Exception as e:
                print(f"[/api/search] SerpAPI failed: {e}")
                # fallthrough to duckduckgo below
        # Final fallback: DuckDuckGo (DDGS package)
        # Timed DuckDuckGo fallback: don't allow the DDG provider to block for long
        try:
            t0 = time.time()
            results = _duckduckgo_search(q, num=10)
            t_ddg = time.time() - t0
            print(f"[/api/search] DuckDuckGo time: {t_ddg:.2f}s")
            if results:
                total_time = time.time() - t_start
                print(f"[/api/search] Completed search in {total_time:.2f}s")
                return jsonify({"query": q, "results": results})
            else:
                print(f"[/api/search] DuckDuckGo returned no results or timed out")
                return make_error('Search failed: no providers available or all providers failed', 502)
        except Exception as e:
            print(f"[/api/search] DuckDuckGo fallback error: {e}")
            return make_error('Search failed: no providers available or all providers failed', 502)
    except Exception as e:
        print("[/api/search] Exception:", e); traceback.print_exc()
        return make_error("Search failed internally.", 500)

# Add this endpoint for advanced search
@app.post("/api/advanced-search")
@limiter.limit("10 per minute")
def advanced_search():
    try:
        data = request.get_json(force=True, silent=True) or {}
        query = (data.get("query") or "").strip()
        filters = data.get("filters", {})
        
        if not query:
            return make_error("Missing 'query' parameter.", 400)
        
        # Build cache key
        cache_key = f"search:{query}:{json.dumps(filters, sort_keys=True)}"
        
        # Check cache first
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                return jsonify(json.loads(cached))
        
        # Enhanced search with filters
        results = perform_advanced_search(query, filters)
        
        # Apply ML clustering to results
        clustered_results = cluster_search_results(results, query)
        
        # Cache results
        if redis_client:
            redis_client.setex(cache_key, 3600, json.dumps(clustered_results))  # Cache for 1 hour
        
        # Emit real-time update via WebSocket
        socketio.emit('search_completed', {
            'query': query,
            'result_count': len(results),
            'timestamp': now_iso()
        })
        
        return jsonify(clustered_results)
        
    except Exception as e:
        print("[/api/advanced-search] Exception:", e)
        traceback.print_exc()
        return make_error("Advanced search failed", 500)

def perform_advanced_search(query, filters):
    """Perform search with advanced filters"""
    # Implementation with multiple search providers and filters
    # This would integrate with your existing search logic
    # but add support for date ranges, domains, content types, etc.
    try:
        # Emit start event for UI progress
        try:
            socketio.emit('search_progress', {'phase': 'start', 'query': query, 'ts': now_iso()})
        except Exception:
            pass
        # Use existing search as base; prefer Google CSE when configured
        if GOOGLE_API_KEY and GOOGLE_CSE_ID:
            google_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": GOOGLE_API_KEY, 
                "cx": GOOGLE_CSE_ID, 
                "q": query, 
                "hl": "en",
                "num": 10
            }
            
            # Apply filters
            if filters.get("date_range"):
                params["dateRestrict"] = filters["date_range"]
            if filters.get("file_type"):
                params["fileType"] = filters["file_type"]
                
            try:
                resp = safe_get(google_url, params=params, timeout=15)
                data = resp.json()
                results = []
                for item in data.get("items", []):
                    results.append({
                        "title": item.get("title"), 
                        "url": item.get("link"), 
                        "snippet": item.get("snippet"), 
                        "image": item.get("pagemap", {}).get("cse_image", [{}])[0].get("src") if "pagemap" in item else None
                    })
                return results
            except Exception as e:
                print(f"[perform_advanced_search] Google CSE failed: {e}")
                # fallthrough to duckduckgo below
            
        # Fallback to regular DuckDuckGo search
        try:
            if DDGS is None:
                raise RuntimeError('DDGS package not installed')
            results = []
            with DDGS() as ddgs:
                for i, r in enumerate(ddgs.text(query, max_results=10)):
                    results.append({
                        "title": r.get("title"),
                        "url": r.get("href"),
                        "snippet": r.get("body"),
                        "image": r.get("image")
                    })
                    # emit per-provider progress
                    try:
                        socketio.emit('search_progress', {'phase': 'provider', 'provider': 'duckduckgo', 'index': i+1, 'query': query, 'ts': now_iso()})
                    except Exception:
                        pass
            # final provider done
            try:
                socketio.emit('search_progress', {'phase': 'provider_done', 'provider': 'duckduckgo', 'query': query, 'count': len(results), 'ts': now_iso()})
            except Exception:
                pass
            return results
        except Exception as e:
            print(f"[perform_advanced_search] DuckDuckGo fallback failed: {e}")
            return []
                
    except Exception as e:
        print(f"Advanced search error: {e}")
        return []

def cluster_search_results(results, query):
    """Cluster search results using ML algorithms"""
    if not results or len(results) < 3:
        return {"query": query, "results": results, "clusters": []}
    
    # Extract text for clustering
    texts = [f"{r.get('title', '')} {r.get('snippet', '')}" for r in results]
    
    # Vectorize text
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    # Perform clustering
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(X)
    labels = clustering.labels_
    
    # Group results by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(results[i])
    
    # Extract cluster themes
    cluster_info = []
    for label, items in clusters.items():
        if label == -1:  # Noise points
            continue
            
        # Extract common keywords for cluster theme
        all_text = " ".join([f"{r.get('title', '')} {r.get('snippet', '')}" for r in items])
        words = tokenize(all_text)
        top_words = Counter(words).most_common(3)
        theme = ", ".join([word for word, count in top_words])
        
        cluster_info.append({
            "theme": theme,
            "count": len(items),
            "items": items
        })
    
    return {
        "query": query,
        "total_results": len(results),
        "clusters": cluster_info,
        "raw_results": results
    }


# -----------------------------
# Hybrid search providers & endpoint
# -----------------------------
def _google_cse_search(query, num=5):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return []
    try:
        url = 'https://www.googleapis.com/customsearch/v1'
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CSE_ID, 'q': query, 'num': num}
        resp = safe_get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        out = []
        for item in data.get('items', []):
            out.append({
                'title': item.get('title'),
                'url': item.get('link'),
                'snippet': item.get('snippet'),
                'source': 'google',
                'image': item.get('pagemap', {}).get('cse_image', [{}])[0].get('src') if item.get('pagemap') else None
            })
        return out
    except Exception as e:
        print('[_google_cse_search] error', e)
        return []


def _youtube_search(query, num=5):
    # Use YouTube Data API if API key present, otherwise return empty
    YT_KEY = os.environ.get('YOUTUBE_API_KEY')
    if not YT_KEY:
        return []
    try:
        url = 'https://www.googleapis.com/youtube/v3/search'
        params = {'part': 'snippet', 'q': query, 'key': YT_KEY, 'maxResults': num, 'type': 'video'}
        resp = safe_get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        out = []
        for it in data.get('items', []):
            sid = it.get('id', {}).get('videoId')
            snippet = it.get('snippet', {})
            out.append({
                'title': snippet.get('title'),
                'url': f'https://www.youtube.com/watch?v={sid}' if sid else snippet.get('channelId'),
                'snippet': snippet.get('description'),
                'source': 'youtube',
                'video_id': sid,
                'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url')
            })
        return out
    except Exception as e:
        print('[_youtube_search] error', e)
        return []


def _reddit_search(query, num=5):
    # Lightweight Reddit search via public endpoint (rate-limited and unofficial)
    try:
        url = 'https://www.reddit.com/search.json'
        headers = {'User-Agent': 'AION/1.0'}
        params = {'q': query, 'limit': num, 'sort': 'relevance'}
        resp = safe_get(url, params=params, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        out = []
        for ch in data.get('data', {}).get('children', []):
            d = ch.get('data', {})
            out.append({
                'title': d.get('title'),
                'url': 'https://reddit.com' + d.get('permalink', ''),
                'snippet': d.get('selftext') or d.get('title'),
                'source': 'reddit'
            })
        return out
    except Exception as e:
        print('[_reddit_search] error', e)
        return []


def _duckduckgo_search(query, num=10):
    # Updated: support an optional timeout via keyword-only arg when called from api_search
    # We'll implement a small worker that collects results and returns them within the time budget
    def _collect_ddg(q, n):
        out = []
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(q, max_results=n)):
                out.append({
                    'title': r.get('title'),
                    'url': r.get('href'),
                    'snippet': r.get('body'),
                    'source': 'duckduckgo',
                    'image': r.get('image')
                })
        return out

    # If DDGS not available, return empty quickly
    if DDGS is None:
        print('[_duckduckgo_search] DDGS package not installed')
        return []

    # Support optional timeout via kwargs by checking local frame (fallback default)
    # We'll use executor to bound execution time. Default timeout here is handled by callers.
    future = None
    try:
        future = executor.submit(_collect_ddg, query, num)
        # For compatibility with existing callers that expect immediate return, wait with a small timeout.
        results = future.result(timeout=5)
        if results:
            return results
    except Exception as e:
        try:
            if future is not None:
                future.cancel()
        except Exception:
            pass
        print('[_duckduckgo_search] ddgs error or timeout', e)

    # Fallback: simple HTML scrape of DuckDuckGo's /html endpoint when ddgs isn't available or fails
    try:
        print('[_duckduckgo_search] falling back to HTML scrape')
        url = 'https://html.duckduckgo.com/html'
        resp = safe_get(url, params={'q': query}, timeout=6)
        resp.raise_for_status()
        html_text = resp.text or ''
        # Find anchor hrefs; prefer links that look like external results
        anchors = re.findall(r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>', html_text, flags=re.I|re.S)
        out = []
        seen = set()
        import html as _html
        for href, title_html in anchors:
            # Clean title: strip tags and unescape
            title = re.sub(r'<[^>]+>', '', title_html)
            title = _html.unescape(title).strip()
            if not href or href in seen:
                continue
            seen.add(href)
            out.append({'title': title or href, 'url': href, 'snippet': '', 'source': 'duckduckgo'})
            if len(out) >= num:
                break
        return out
    except Exception as e:
        print('[_duckduckgo_search] HTML scrape failed', e)
        return []


def _instagram_search(query, num=5):
    """
    Placeholder Instagram adapter. Instagram's official API requires App Review and access tokens.
    This function currently returns an empty list unless `INSTAGRAM_BEARER` is set in env.
    TODO: Implement proper Graph API calls with paging, rate-limiting, and privacy checks.
    """
    INSTAGRAM_BEARER = os.environ.get('INSTAGRAM_BEARER') or os.environ.get('INSTAGRAM_ACCESS_TOKEN')
    INSTAGRAM_USER_ID = os.environ.get('INSTAGRAM_USER_ID')
    # If no token is configured, return empty list (safe fallback)
    if not INSTAGRAM_BEARER:
        return []

    try:
        out = []
        # Prefer the Facebook Graph API media endpoint for a given user if USER_ID is provided.
        # Note: access token may be passed as a query param for the Graph API. This is a best-effort adapter
        # and will only work when valid credentials are provided. We keep it optional and robust to failure.
        if INSTAGRAM_USER_ID:
            url = f"https://graph.facebook.com/v17.0/{INSTAGRAM_USER_ID}/media"
            params = {
                'fields': 'id,caption,media_url,permalink,timestamp,media_type',
                'limit': min(max(1, int(num)), 50),
                'access_token': INSTAGRAM_BEARER
            }
            resp = safe_get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get('data', []):
                caption = item.get('caption') or ''
                # Basic match: include posts where the caption contains the query (case-insensitive)
                if query and query.lower() not in caption.lower() and len(out) >= 0:
                    # we still include items but prefer matches; keep behavior permissive
                    pass
                out.append({
                    'title': (caption[:120] or 'Instagram post'),
                    'url': item.get('permalink') or item.get('media_url'),
                    'snippet': caption,
                    'source': 'instagram',
                    'image': item.get('media_url'),
                    'date': item.get('timestamp')
                })
                if len(out) >= num:
                    break
            return out

        # If no user id, attempt a simple hashtag search via Graph API if token allows (best-effort)
        # Note: hashtag search requires additional permissions; if it's not available, return empty list.
        # We keep this non-fatal and return [] on error.
        # For now, fall back to empty list if user id not provided.
        return []
    except Exception as e:
        print('[_instagram_search] error', e)
        return []


def _twitter_search(query, num=5):
    """
    Placeholder Twitter adapter. Use `TWITTER_BEARER` (v2 API) or client credentials.
    Real integration needs elevated access for full search and careful rate-limit handling.
    """
    TWITTER_BEARER = os.environ.get('TWITTER_BEARER') or os.environ.get('X_TWITTER_BEARER')
    if not TWITTER_BEARER:
        return []
    try:
        headers = {'Authorization': f'Bearer {TWITTER_BEARER}', 'User-Agent': 'AION/1.0'}
        url = 'https://api.twitter.com/2/tweets/search/recent'
        params = {
            'query': query,
            'max_results': min(max(1, int(num)), 100),
            'tweet.fields': 'text,author_id,created_at,attachments',
            'expansions': 'attachments.media_keys',
            'media.fields': 'url,preview_image_url,type'
        }
        resp = safe_get(url, params=params, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        out = []
        media_map = {}
        for m in data.get('includes', {}).get('media', []):
            media_map[m.get('media_key')] = m

        for t in data.get('data', []):
            text = t.get('text') or ''
            tweet_id = t.get('id')
            item = {
                'title': text[:120],
                'url': f"https://twitter.com/i/web/status/{tweet_id}",
                'snippet': text,
                'source': 'twitter',
                'date': t.get('created_at')
            }
            # Attach media if present
            attachments = t.get('attachments', {})
            if attachments and attachments.get('media_keys'):
                keys = attachments.get('media_keys')
                # take first media's url if available
                mm = media_map.get(keys[0])
                if mm:
                    item['image'] = mm.get('preview_image_url') or mm.get('url')
            out.append(item)
            if len(out) >= num:
                break
        return out
    except Exception as e:
        print('[_twitter_search] error', e)
        return []


@app.post('/api/hybrid-search')
@limiter.limit('20 per minute')
def api_hybrid_search():
    try:
        data = request.get_json(force=True, silent=True) or {}
        query = (data.get('query') or '').strip()
        providers = data.get('providers') or []
        filters = data.get('filters') or {}
        if not query:
            return make_error('Missing query', 400)

        # Delegate aggregation to helper so it can be unit tested
        aggregate = _aggregate_hybrid_results(query, providers, filters=filters)
        resp = {'query': query, 'results': aggregate, 'count': len(aggregate)}
        return jsonify(resp)
    except Exception as e:
        print('[/api/hybrid-search] error', e); traceback.print_exc()
        return make_error('Hybrid search failed', 500)


def _aggregate_hybrid_results(query: str, providers: list, filters: dict = None) -> list:
    """
    Aggregate results from selected providers, deduplicate by URL, and attach a simple score.
    Returns a list of normalized result dicts.
    This helper is separated for testing and reuse.
    """
    if filters is None:
        filters = {}

    if not providers:
        providers = ['aion', 'duckduckgo']

    aggregate = []
    seen = set()

    # Local AION research (fast): perform_advanced_search may return a list or a dict with raw_results
    if 'aion' in providers:
        try:
            local_raw = perform_advanced_search(query, filters)
            # Support both list return and dict with raw_results
            if isinstance(local_raw, dict) and 'raw_results' in local_raw:
                local = local_raw.get('raw_results', [])
            else:
                local = local_raw or []
            for r in (local[:5] if isinstance(local, list) else []):
                r['source'] = r.get('source') or 'aion'
                key = (r.get('url') or '')
                if key and key not in seen:
                    seen.add(key); aggregate.append(r)
        except Exception as e:
            print('[_aggregate_hybrid_results] aion error', e)

    # Provider adapters
    if 'google' in providers:
        for r in _google_cse_search(query, num=5):
            key = r.get('url')
            if key and key not in seen:
                seen.add(key); aggregate.append(r)

    if 'youtube' in providers:
        for r in _youtube_search(query, num=5):
            key = r.get('url')
            if key and key not in seen:
                seen.add(key); aggregate.append(r)

    if 'reddit' in providers:
        for r in _reddit_search(query, num=5):
            key = r.get('url')
            if key and key not in seen:
                seen.add(key); aggregate.append(r)

    if 'duckduckgo' in providers:
        # If DDGS package is unavailable, adapter returns []
        for r in _duckduckgo_search(query, num=8):
            key = r.get('url')
            if key and key not in seen:
                seen.add(key); aggregate.append(r)

    if 'instagram' in providers:
        for r in _instagram_search(query, num=5):
            key = r.get('url')
            if key and key not in seen:
                seen.add(key); aggregate.append(r)

    if 'twitter' in providers:
        for r in _twitter_search(query, num=5):
            key = r.get('url')
            if key and key not in seen:
                seen.add(key); aggregate.append(r)

    # Basic ranking: prefer AION/local then google then youtube then others; keep original order
    source_score = {'aion': 1.0, 'google': 0.9, 'youtube': 0.8, 'reddit': 0.6, 'duckduckgo': 0.5, 'twitter': 0.55, 'instagram': 0.55}
    for i, r in enumerate(aggregate):
        base = source_score.get(r.get('source'), 0.4)
        r['score'] = round(base - (i * 0.0001), 4)

    return aggregate


@app.get('/api/assets')
def list_assets():
    """List saved assets in DATA_DIR/assets with basic metadata"""
    assets_dir = os.path.join(DATA_DIR, 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    items = []
    for fname in sorted(os.listdir(assets_dir)):
        fpath = os.path.join(assets_dir, fname)
        if not os.path.isfile(fpath):
            continue
        stat = os.stat(fpath)
        items.append({
            'filename': fname,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            'url': request.host_url.rstrip('/') + f"/files/assets/{fname}"
        })
    return jsonify({'ok': True, 'assets': items})


@app.delete('/api/assets/<path:filename>')
def delete_asset(filename: str):
    assets_dir = os.path.join(DATA_DIR, 'assets')
    safe = filename.replace('..', '').replace('/', '')
    fpath = os.path.join(assets_dir, safe)
    if not os.path.exists(fpath):
        return make_error('Asset not found', 404)
    try:
        os.remove(fpath)
        return jsonify({'ok': True, 'filename': safe})
    except Exception as e:
        print('delete_asset error', e)
        return make_error('Failed to delete asset', 500)


# --- Math solver endpoint: try SymPy first, then fallback to LLM proxy
@app.post('/solve-math')
def solve_math():
    try:
        data = request.get_json(force=True, silent=True) or {}
        problem = (data.get('problem') or data.get('expression') or data.get('query') or '').strip()
        if not problem:
            return make_error('Missing math problem in request body (use `problem` or `expression`).', 400)

        # Try SymPy if available for instant local solves
        try:
            import sympy as sp
            # Try to detect common operations
            if any(k in problem.lower() for k in ['integrate', 'integral', '∫']):
                expr_text = problem.replace('integrate', '').replace('integral', '')
                expr = sp.sympify(expr_text)
                res = sp.integrate(expr)
                return jsonify({'ok': True, 'method': 'sympy', 'result': str(res)})
            if any(k in problem.lower() for k in ['differentiate', 'derivative', "d/d"]):
                expr_text = problem
                # simple parse: differentiate expression
                expr = sp.sympify(expr_text)
                res = sp.diff(expr)
                return jsonify({'ok': True, 'method': 'sympy', 'result': str(res)})
            # try evaluate/simplify or solve
            try:
                expr = sp.sympify(problem)
                res = sp.simplify(expr)
                return jsonify({'ok': True, 'method': 'sympy', 'result': str(res)})
            except Exception:
                # try to solve equations
                try:
                    if '=' in problem:
                        left, right = problem.split('=', 1)
                        sol = sp.solve(sp.Eq(sp.sympify(left), sp.sympify(right)))
                        return jsonify({'ok': True, 'method': 'sympy', 'result': str(sol)})
                except Exception:
                    pass
        except Exception:
            # SymPy not available or failed; fallback to LLM proxy
            pass

        # Fallback: use Ollama/OpenAI via proxy helper
        try:
            prompt = f"Solve this math problem and show steps: {problem}"
            payload = {'prompt': prompt, 'stream': False}
            body = _call_ollama_generate(payload, timeout=60, retries=1)
            if isinstance(body, dict):
                text = body.get('response') or body.get('body') or str(body)
            else:
                text = str(body)
            return jsonify({'ok': True, 'method': 'llm', 'result': text})
        except Exception as e:
            return make_error('Failed to solve math problem via LLM fallback: ' + str(e), 502)
    except Exception as e:
        tb = traceback.format_exc()
        print('[/solve-math] Exception:', e, tb)
        return make_error('Internal error while solving math problem', 500)

# Add WebSocket endpoint for real-time updates
@socketio.on('connect')
def handle_connect():
    emit('connection_established', {'data': 'Connected to AION search'})

@socketio.on('search_progress')
def handle_search_progress(data):
    # Broadcast search progress to all clients
    emit('search_update', data, broadcast=True)

# Add search analytics endpoint
@app.post("/api/search-analytics")
def search_analytics():
    try:
        data = request.get_json(force=True, silent=True) or {}
        days = int(data.get("days", 7))
        
        # Get search statistics from database
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        
        # Get popular queries
        cur.execute("""
            SELECT query, COUNT(*) as count, AVG(result_count) as avg_results 
            FROM search_history 
            WHERE timestamp > datetime('now', ?) 
            GROUP BY query 
            ORDER BY count DESC 
            LIMIT 10
        """, (f"-{days} days",))
        
        popular_queries = []
        for row in cur.fetchall():
            popular_queries.append({
                "query": row[0],
                "count": row[1],
                "avg_results": row[2]
            })
        
        # Get search success rate
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN result_count > 0 THEN 1 ELSE 0 END) as successful
            FROM search_history 
            WHERE timestamp > datetime('now', ?)
        """, (f"-{days} days",))
        
        total, successful = cur.fetchone()
        success_rate = (successful / total * 100) if total > 0 else 0
        
        con.close()
        
        return jsonify({
            "popular_queries": popular_queries,
            "success_rate": success_rate,
            "total_searches": total,
            "time_period_days": days
        })
        
    except Exception as e:
        print("[/api/search-analytics] Exception:", e)
        traceback.print_exc()
        return make_error("Analytics retrieval failed", 500)

# -----------------------------
# NEW: Episodic Memory Endpoints
# -----------------------------
@app.post("/consciousness/add-episodic-memory")
def add_episodic_memory():
    try:
        data = request.get_json(force=True, silent=True) or {}
        event_type = data.get("event_type", "user_interaction")
        content = data.get("content", "")
        
        if not content:
            return make_error("Missing 'content' in request body.", 400)
            
        ts = now_iso()
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO episodic_memory (event_type, content, timestamp) VALUES (?, ?, ?)",
            (event_type, content, ts)
        )
        con.commit()
        con.close()
        
        return jsonify({"ok": True, "event_type": event_type, "timestamp": ts})
    except Exception as e:
        print("[/consciousness/add-episodic-memory] Exception:", e)
        traceback.print_exc()
        return make_error("Failed to add episodic memory", 500)

@app.get("/consciousness/episodic-memories")
def get_episodic_memories():
    try:
        limit = int(request.args.get("limit", 10))
        offset = int(request.args.get("offset", 0))
        
        con = sqlite3.connect(DB_FILE)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM episodic_memory ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
        memories = [dict(row) for row in cur.fetchall()]
        con.close()
        
        return jsonify({"ok": True, "memories": memories})
    except Exception as e:
        print("[/consciousness/episodic-memories] Exception:", e)
        traceback.print_exc()
        return make_error("Failed to retrieve episodic memories", 500)

# -----------------------------
# NEW: Procedural Memory Endpoints
# -----------------------------
@app.post("/consciousness/add-procedure")
def add_procedure():
    try:
        data = request.get_json(force=True, silent=True) or {}
        name = data.get("name", "")
        steps = data.get("steps", [])
        
        if not name or not steps:
            return make_error("Missing 'name' or 'steps' in request body.", 400)
            
        ts = now_iso()
        steps_json = json.dumps(steps)
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO procedural_memory (name, steps, created_at) VALUES (?, ?, ?)",
            (name, steps_json, ts)
        )
        con.commit()
        con.close()
        
        return jsonify({"ok": True, "name": name, "created_at": ts})
    except Exception as e:
        print("[/consciousness/add-procedure] Exception:", e)
        traceback.print_exc()
        return make_error("Failed to add procedure", 500)

@app.get("/consciousness/procedures")
def get_procedures():
    try:
        limit = int(request.args.get("limit", 10))
        offset = int(request.args.get("offset", 0))
        
        con = sqlite3.connect(DB_FILE)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM procedural_memory ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
        procedures = []
        for row in cur.fetchall():
            procedure = dict(row)
            procedure["steps"] = json.loads(procedure["steps"])
            procedures.append(procedure)
        con.close()
        
        return jsonify({"ok": True, "procedures": procedures})
    except Exception as e:
        print("[/consciousness/procedures] Exception:", e)
        traceback.print_exc()
        return make_error("Failed to retrieve procedures", 500)

@app.get("/consciousness/procedure/<name>")
def get_procedure(name):
    try:
        con = sqlite3.connect(DB_FILE)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM procedural_memory WHERE name = ?", (name,))
        row = cur.fetchone()
        con.close()
        
        if not row:
            return make_error("Procedure not found", 404)
            
        procedure = dict(row)
        procedure["steps"] = json.loads(procedure["steps"])
        
        return jsonify({"ok": True, "procedure": procedure})
    except Exception as e:
        print("[/consciousness/procedure] Exception:", e)
        traceback.print_exc()
        return make_error("Failed to retrieve procedure", 500)

# -----------------------------
# MODIFIED: Consciousness Engine with SQLite <<<
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
        cur.execute("SELECT * FROM insights ORDER by id DESC LIMIT ? OFFSET ?", (limit, offset))
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
            # Use internal search helper directly to avoid local HTTP overhead
            results = perform_advanced_search(query, {})
            if isinstance(results, dict) and 'raw_results' in results:
                raw = results.get('raw_results', [])
            else:
                raw = results or []
            return raw[:limit]
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

            # Call Ollama helper directly instead of local HTTP to avoid request overhead
            try:
                body = _call_ollama_generate({'prompt': prompt, 'model': 'llama3', 'stream': False}, timeout=120, retries=1)
            except Exception as e:
                return {"error": "Ollama generate failed in reflect_once", "detail": str(e)}

            if isinstance(body, dict):
                insight_text = body.get('response') or body.get('code') or body.get('body') or str(body)
            else:
                insight_text = str(body)
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
            # Call model helper directly to avoid extra HTTP proxy and potential 502 returns
            body = None
            try:
                body = _call_ollama_generate({'prompt': summary_prompt, 'model': 'llama3', 'stream': False}, timeout=180, retries=1)
            except Exception as e:
                # If local model fails, try OpenAI fallback helper
                try:
                    body = _call_openai_generate(summary_prompt, model=OPENAI_MODEL, max_tokens=512)
                except Exception:
                    body = None

            if body is None:
                summary_text = "Error: Model backend unavailable or returned an error."
            else:
                # Normalize various response shapes and extract text
                normalized = _normalize_model_response(body)
                summary_text = (normalized.get('result', {}).get('text') or '').strip()
                if not summary_text:
                    # fallback to raw if normalization didn't find text
                    try:
                        summary_text = str(normalized.get('raw') or body)[:2000]
                    except Exception:
                        summary_text = "Error: Could not extract summary from model response."
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
        mem_count, ins_count, episodic_count, procedural_count = _get_db_counts()
        state = CONSCIOUS.state.copy()
        state.update({
            "memories_count": mem_count, 
            "insights_count": ins_count,
            "episodic_memories_count": episodic_count,
            "procedural_memories_count": procedural_count
        })
        return jsonify({"ok": True, "state": state})
    except Exception as e: print(e); traceback.print_exc(); return make_error("Failed retrieving state", 500)

@app.get("/consciousness/memories")
def get_memories_route():
    try:
        limit, offset = int(request.args.get("limit", 10)), int(request.args.get("offset", 0))
        memories = CONSCIOUS.get_memories(limit=limit, offset=offset)
        mem_count, _, _, _ = _get_db_counts()
        return jsonify({"ok": True, "memories": memories, "total": mem_count})
    except Exception as e: print(e); traceback.print_exc(); return make_error("Failed to retrieve memories", 500)

@app.get("/consciousness/insights")
def get_insights_route():
    try:
        limit, offset = int(request.args.get("limit", 10)), int(request.args.get("offset", 0))
        insights = CONSCIOUS.get_insights(limit=limit, offset=offset)
        _, ins_count, _, _ = _get_db_counts()
        return jsonify({"ok": True, "insights": insights, "total": ins_count})
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