# model_server.py
# Lightweight FastAPI service to host Stable Diffusion on a single process.
# Run on a GPU machine and expose an HTTP endpoint for image generation.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import traceback
import logging
from typing import Optional
import base64
from io import BytesIO

app = FastAPI()
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("sd-server")

try:
    from diffusers import StableDiffusionPipeline
    import torch
    SD_MODEL_ID = os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info("Loading SD model %s (cuda=%s)", SD_MODEL_ID, torch.cuda.is_available())
    pipeline = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=dtype)
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
    pipeline.safety_checker = None
    pipeline.feature_extractor = None
    logger.info("SD model loaded")
except Exception as e:
    logger.exception("Failed to load SD model")
    pipeline = None

class ImageRequest(BaseModel):
    prompt: str
    steps: Optional[int] = 30
    width: Optional[int] = 512
    height: Optional[int] = 512

def pil_to_dataurl(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

@app.post("/generate-image")
def generate_image(req: ImageRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Stable Diffusion not available on this server")
    if req.width > 1024 or req.height > 1024:
        raise HTTPException(status_code=400, detail="Max resolution 1024x1024")
    try:
        result = pipeline(prompt=req.prompt, num_inference_steps=req.steps)
        img = result.images[0]
        return {"imageUrl": pil_to_dataurl(img)}
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))