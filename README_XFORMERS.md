Windows installation notes for xformers, pyav (av), and video helpers

This document helps install optional performance packages that greatly speed up Stable Diffusion / diffusers attention and video encoding on Windows.

Important: Match your PyTorch CUDA version. Run the check below first to see what PyTorch/CUDA you have.

Check PyTorch and CUDA (PowerShell):
```powershell
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda device:', torch.cuda.get_device_name(0))
    import sys
    print('cuda arch:', torch.version.cuda)
PY
```

1) Install or update PyTorch (if needed)
- Visit https://pytorch.org/get-started/locally/ and pick the correct command for your CUDA version.
- Example (CUDA 11.8, stable):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Adjust `cu118` to your CUDA version (cu117, cu121 etc).

2) Install xformers (optional but recommended)
- xformers binaries for Windows may not be available via pip for every PyTorch/CUDA combo. Try the simple pip install first:
```powershell
pip install xformers
```
- If that fails, use a prebuilt wheel matching your torch and CUDA. Search GitHub releases or community-built wheels (e.g., "xformers wheels <torch version> <cuda>").
- Alternative: use `--use-pep517` or build from source (long; requires Visual Studio and correct CUDA toolkit).

3) Install pyav (av) and ffmpeg
- `pyav` provides bindings to libav/ffmpeg. It's practical to install `imageio-ffmpeg` instead for pure-Python ffmpeg access.
```powershell
pip install av imageio imageio-ffmpeg
```
- If `pip install av` fails, install a binary wheel or install ffmpeg separately and rely on `imageio-ffmpeg` for encoding/decoding.

4) If you use imageio and need faster mp4 encoding, install `pyav` with ffmpeg present or ensure `imageio-ffmpeg` is installed.

5) Troubleshooting
- If you can't get `xformers` on Windows, you still benefit from attention slicing (already enabled in server). xformers gives the biggest speed + memory wins but is optional.
- For any pip install failures, capture the error and search for a matching wheel for your torch+cuda combination.

6) Example final environment setup (PowerShell):
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
# install torch (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate
pip install xformers || Write-Host "xformers install failed — consult README_XFORMERS.md"
pip install av imageio imageio-ffmpeg
```

7) After installing: restart the server and watch logs. The server tries to enable xformers if available and will fall back safely if not.

If you'd like, I can generate specific wheel links for `xformers` matching your installed `torch` version — share the output of the PyTorch check above and I'll look up wheels that match.

Server-side env helper
----------------------
After starting the server you can GET `/env-info` to return the server's Python environment info (torch version, cuda availability, diffusers version). Example:

```powershell
Invoke-RestMethod -Uri http://localhost:5000/env-info
```

Paste that output when you want me to look up exact wheel links for `xformers` matching your environment.