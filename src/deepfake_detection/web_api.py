from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import FramePredictor

DEFAULT_CHECKPOINT = "checkpoints/frame_deepfake_v1/best_model.pt"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

app = FastAPI(
    title="Deepfake Detection Dev API",
    version="1.0.0",
    description="Local frame-model inference for the webcam development UI.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_predictor: FramePredictor | None = None
_predictor_lock = Lock()


def _checkpoint_path() -> Path:
    return Path(os.getenv("DEEPFAKE_CHECKPOINT", DEFAULT_CHECKPOINT)).expanduser()


def get_predictor() -> FramePredictor:
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                checkpoint = _checkpoint_path()
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"frame checkpoint not found at {checkpoint}. "
                        "Set DEEPFAKE_CHECKPOINT to a valid frame model."
                    )
                _predictor = FramePredictor(
                    checkpoint,
                    device_name=os.getenv("DEEPFAKE_DEVICE") or None,
                )
    return _predictor


@app.get("/health")
def health() -> dict:
    try:
        predictor = get_predictor()
    except Exception as error:
        return {
            "status": "unavailable",
            "error": str(error),
            "checkpoint": str(_checkpoint_path()),
        }
    return {"status": "ready", **predictor.metadata}


@app.post("/predict-frame")
async def predict_frame(image: Annotated[UploadFile, File()]) -> dict:
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 12 MB limit.")
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="The uploaded image could not be decoded.")
    started = perf_counter()
    try:
        faces = get_predictor().predict(frame)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference failed: {error}") from error
    height, width = frame.shape[:2]
    return {
        "faces": faces,
        "image": {"width": width, "height": height},
        "inference_ms": round((perf_counter() - started) * 1000, 1),
    }
