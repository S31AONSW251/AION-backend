Performance changes made to server.py

What I changed
- Enabled cuDNN benchmark when CUDA is available to improve fixed-size input speed.
- Attempted to enable attention slicing and xformers memory-efficient attention (safe no-op if not installed).
- Wrapped inference calls with `torch.inference_mode()` and `torch.autocast('cuda')` when CUDA is available.
- Added detailed timing logs for `/generate-image` and `/generate-video` endpoints.
- Improved MP4 conversion to prefer `imageio.v3`/pyav and fallback to `imageio`.
- Changed `/generate-video` to save output MP4 files into `DATA_DIR` and return a download URL `/files/<filename>` instead of embedding base64.
- Added a `/files/<filename>` route to serve files from `DATA_DIR`.

How to test
1) Start server and call `/api/health` to make sure SD pipelines loaded.
2) POST to `/generate-video` with a small test payload (low steps and frames) and watch the console logs for timings.

Example PowerShell POST (adjust host/port):
```powershell
$body = @{prompt='a scenic mountain sunrise'; steps=12; width=384; height=216; num_frames=8} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/generate-video -Method Post -Body $body -ContentType 'application/json'
```

You should see timing prints like:
- `[/generate-video] Keyframe generated in X.XXs`
- `[/generate-video] Video frames generated in Y.YYs (total Z.ZZs)`
- `[/generate-video] Video saved to ...` 

Notes & next steps
- Installing `xformers` typically provides the biggest speed/memory win for attention-heavy models.
- Consider streaming large videos to disk and returning a URL (already implemented) to avoid memory spikes.
- If you want further improvements I can add batching, use accelerate offloading, or add a background job queue for long-running video jobs.
