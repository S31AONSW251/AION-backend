# AION Backend — Developer Setup (Windows)

This file contains quick steps to get the backend running on Windows for local testing.

Prerequisites
- Python 3.10+ and virtualenv
- Git
- (Optional) Ollama installed and running locally if you want local model inference: https://ollama.com
- (Optional) An OpenAI API key if you want OpenAI fallback streaming (set `OPENAI_API_KEY`).
- FFmpeg installed and on PATH if you plan to use PyAV/image/video features.

Create virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Environment variables (use PowerShell setx or set for the session)

- `OLLAMA_BASE_URL` — optional override (default: http://localhost:11434)
- `OPENAI_API_KEY` — optional; if present, backend will use OpenAI as fallback for generation
- `AION_MODEL_CONNECT_TIMEOUT` and `AION_MODEL_READ_TIMEOUT` — seconds for model proxy timeouts

Starting the server

```powershell
cd aion_backend
.venv\Scripts\python.exe run_server.py
```

Health and quick checks

1. Health endpoint

   ```powershell
   Invoke-RestMethod -Uri http://127.0.0.1:5000/health
   ```

2. Test NDJSON streaming endpoint

   ```powershell
   # This returns a small NDJSON test stream
   Invoke-RestMethod -Uri http://127.0.0.1:5000/api/generate/test -Method POST -Body (@{prompt='hello'} | ConvertTo-Json)
   ```

3. Real generate (non-stream)

   ```powershell
   Invoke-RestMethod -Uri http://127.0.0.1:5000/api/generate -Method POST -Body (@{prompt='Say hello'} | ConvertTo-Json)
   ```

Notes on Ollama and OpenAI
- Ollama: if you want to use a local model, install and run the Ollama daemon and make sure it listens at http://localhost:11434 or set `OLLAMA_BASE_URL` accordingly.
- OpenAI: set `OPENAI_API_KEY` in the environment before starting the backend for OpenAI fallback to be available.

Native/wheels caveats on Windows
- Some optional packages (xformers, diffusers) may require GPU-specific wheels or extra steps on Windows. If you see import errors for image/video pipelines, they are optional and only needed for image-generation features.
- Install FFmpeg (https://ffmpeg.org/download.html) and add to PATH for PyAV/imageio-ffmpeg to work.

If you want, I can run the health and test calls for you and paste the outputs here.
