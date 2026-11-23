# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""ML-Backend Custom Server.

Loads models from S3 (via Model Registry metadata) and serves
Label Studio ML Backend compatible HTTP inference endpoints.

Supported frameworks:
- PP-YOLOE (PaddleDetection) - Apache-2.0
- paddleocr (ocr)
- pytorch (generic)
- tensorflow (generic)
- onnxruntime (generic)

NO ultralytics/YOLO - AGPL-3.0 removed
"""

import os
import logging
import base64
from io import BytesIO
from typing import Optional, List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-backend-custom")

app = FastAPI(title="ML-Backend Custom", version="1.0.0")

# ---------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "")
BACKEND_TYPE = os.getenv("BACKEND_TYPE", "det")
FRAMEWORK = os.getenv("FRAMEWORK", "paddlex")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333")
MODEL_ROOT = os.getenv("MODEL_ROOT", "/app/models")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
DEVICE = os.getenv("DEVICE", "cpu")

_model = None
_model_names = {}


def download_from_s3(s3_path: str, local_path: str) -> str:
    """Download model from S3 (seaweedfs)."""
    import boto3
    
    # Parse s3://bucket/key
    rest = s3_path[len("s3://"):]
    bucket, key = rest.split("/", 1)
    
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id="",
        aws_secret_access_key="",
        region_name="us-east-1",
    )
    
    logger.info(f"Downloading model from {s3_path} to {local_path}")
    s3.download_file(bucket, key, local_path)
    logger.info(f"Model downloaded successfully")
    
    return local_path


def load_model():
    """Load model based on framework type."""
    global _model, _model_names
    
    if _model is not None:
        return _model
    
    if not MODEL_PATH:
        raise ValueError("MODEL_PATH environment variable is required")
    
    # Download model if it's on S3
    if MODEL_PATH.startswith("s3://"):
        local_path = os.path.join(MODEL_ROOT, "model.pt")
        os.makedirs(MODEL_ROOT, exist_ok=True)
        if not os.path.exists(local_path):
            download_from_s3(MODEL_PATH, local_path)
        model_path = local_path
    else:
        model_path = MODEL_PATH
    
    logger.info(f"Loading model: {model_path} (framework={FRAMEWORK}, device={DEVICE})")
    
    if FRAMEWORK == "paddlex":
        # PP-YOLOE (PaddleDetection) - Apache-2.0
        from paddlex import create_model
        _model = create_model(model_name="PP-YOLOE-L", pretrained=True)
        _model_names = {}
    elif FRAMEWORK == "paddleocr":
        from paddleocr import PaddleOCR
        _model = PaddleOCR(lang='ch', use_angle_cls=True)
        _model_names = {}
    elif FRAMEWORK == "pytorch":
        import torch
        _model = torch.load(model_path, map_location=DEVICE)
        _model_names = getattr(_model, "names", {})
    elif FRAMEWORK == "tensorflow":
        import tensorflow as tf
        _model = tf.keras.models.load_model(model_path)
        _model_names = {}
    elif FRAMEWORK == "onnxruntime":
        import onnxruntime as ort
        _model = ort.InferenceSession(model_path)
        _model_names = {}
    else:
        raise ValueError(f"Unsupported framework: {FRAMEWORK}. Supported: paddlex, paddleocr, pytorch, tensorflow, onnxruntime")
    
    logger.info(f"Model loaded successfully, names: {_model_names}")
    return _model


def decode_image(data) -> Image.Image:
    """Decode input: base64 string, raw bytes, or URL."""
    if isinstance(data, str):
        if data.startswith("data:"):
            # data URI: data:image/jpeg;base64,...
            b64 = data.split(",", 1)[1]
            return Image.open(BytesIO(base64.b64decode(b64)))
        if data.startswith("http"):
            import urllib.request
            with urllib.request.urlopen(data) as resp:
                return Image.open(BytesIO(resp.read()))
        # bare base64
        try:
            return Image.open(BytesIO(base64.b64decode(data)))
        except Exception:
            raise HTTPException(400, "invalid image string")
    if isinstance(data, (bytes, bytearray)):
        return Image.open(BytesIO(bytes(data)))
    raise HTTPException(400, "unsupported image input")


class PredictRequest(BaseModel):
    tasks: Optional[List[Dict[str, Any]]] = None
    label_config: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model")
def model_info():
    m = load_model()
    return {
        "name": "custom-model",
        "ready": True,
        "version": getattr(m, "version", "1.0"),
        "framework": FRAMEWORK,
        "backend_type": BACKEND_TYPE,
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    """Label Studio ML Backend compatible predict endpoint."""
    m = load_model()
    
    tasks = req.tasks or []
    results = []
    
    for task in tasks:
        task_data = task.get("data", {})
        image_data = task_data.get("image", "")
        
        if not image_data:
            results.append({"result": []})
            continue
        
        try:
            img = decode_image(image_data)
            preds = _predict_single(m, img)
            results.append({"result": preds})
        except Exception as e:
            logger.error(f"Prediction failed for task {task.get('id')}: {e}")
            results.append({"result": [], "error": str(e)})
    
    return {"results": results}


def _predict_single(model, img: Image.Image) -> List[Dict]:
    """Run prediction on a single image based on framework."""
    
    if FRAMEWORK == "paddlex":
        # PP-YOLOE (PaddleDetection) - Apache-2.0
        from paddlex import create_model
        result = model.predict(img)
        return _format_paddlex(result)
    elif FRAMEWORK == "paddleocr":
        import numpy as np
        img_array = np.array(img)
        preds = model.ocr(img_array, cls=True)
        return _format_paddleocr(preds)
    elif FRAMEWORK == "onnxruntime":
        # TODO: Implement ONNX inference
        return []
    else:
        # Generic framework - return empty for now
        return []


def _format_paddlex(result) -> List[Dict]:
    """Format PP-YOLOE (PaddleDetection) predictions to Label Studio format."""
    items = []
    if isinstance(result, dict) and "boxes" in result:
        boxes = result["boxes"]
        scores = result.get("scores", [])
        labels = result.get("labels", [])
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            items.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(w),
                    "height": float(h),
                    "rectanglelabels": ["object"],
                    "rotation": 0,
                },
                "score": float(scores[i]) if i < len(scores) else 0.5,
            })
    return items


def _format_paddleocr(preds) -> List[Dict]:
    """Format PaddleOCR predictions to Label Studio format."""
    items = []
    if not preds or not preds[0]:
        return items
    
    for line in preds[0]:
        box, (text, conf) = line
        # Convert box to rectangle format
        x_min = min(p[0] for p in box)
        y_min = min(p[1] for p in box)
        x_max = max(p[0] for p in box)
        y_max = max(p[1] for p in box)
        
        items.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": round(x_min, 2),
                "y": round(y_min, 2),
                "width": round(x_max - x_min, 2),
                "height": round(y_max - y_min, 2),
                "rectanglelabels": ["text"],
                "rotation": 0,
            },
            "score": round(conf, 4),
        })
    
    return items


if __name__ == "__main__":
    port = int(os.getenv("PORT", "9090"))
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=port)
