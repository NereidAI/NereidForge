# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import os
import logging
from typing import List, Dict, Optional

from label_studio_ml.model import LabelStudioMLBase

logger = logging.getLogger(__name__)
if not os.getenv("LOG_LEVEL"):
    logger.setLevel(logging.INFO)


class PPYOLOEModel(LabelStudioMLBase):
    """Label Studio ML Backend based on PP-YOLOE (PaddleDetection) - Apache-2.0"""

    def setup(self):
        """Configure model parameters"""
        self.set("model_version", "ppyoloe")
        # Lazy-load PP-YOLOE model
        self._model = None
        self._model_path = os.getenv("MODEL_PATH", "/app/models/ppyoloe_crn_l_300e_coco.pdparams")
        self._config_path = os.getenv("CONFIG_PATH", "/app/configs/ppyoloe_crn_l_300e_coco.yml")

    def _load_model(self):
        """Lazy load PP-YOLOE model"""
        if self._model is not None:
            return
        try:
            import paddle  # noqa: F401  # Initialize the framework before PaddleX model creation.
            from paddlex import create_model

            self._model = create_model(model_name="PP-YOLOE-L", pretrained=True)
            logger.info("PP-YOLOE model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PP-YOLOE model: {e}")
            raise

    def detect_control_models(self):
        """Detect control models based on the labeling config"""
        control_models = []

        label_interface = getattr(self, "label_interface", None)
        if label_interface is None:
            raise ValueError(
                "PP-YOLOE model label_interface not initialized. "
                "This model must be used within a Label Studio project context "
                "(deployed via Label Studio ML backend framework), not as a standalone HTTP service. "
                "Ensure the model is deployed through Label Studio's model management, "
                "or configure the model through the Label Studio UI project settings."
            )

        label_interface = getattr(self, "label_interface", None)
        if label_interface is None:
            return []

        control_models = []
        for control in label_interface.controls:
            if not control.to_name:
                continue

            # Only handle rectangle labels for object detection
            if control.tag == "RectangleLabels":
                from control_models.rectangle_labels import RectangleLabelsModel
                instance = RectangleLabelsModel.create(self, control)
                if instance and instance.label_map:
                    control_models.append(instance)
                    logger.info(f"Control tag with model detected: {instance}")

        if not control_models:
            raise ValueError(
                f"No suitable control tags (e.g., RectangleLabels connected to Image) "
                f"detected in the label config:\n{self.label_config}"
            )

        return control_models

    def _run_inference(self, image_path: str):
        """Run PP-YOLOE inference on an image"""
        if self._model is None:
            self._load_model()
        import cv2

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Use paddlex for simpler inference
        try:
            from paddlex import create_model
            model = create_model(model_name="PP-YOLOE-L", pretrained=True)
            result = model.predict(image_path)
            return result
        except Exception as e:
            logger.error(f"PP-YOLOE inference failed: {e}")
            raise

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ):
        """Run PP-YOLOE predictions on the tasks"""
        logger.info(f"Run prediction on {len(tasks)} tasks, project ID = {self.project_id}")

        import time as _time

        t_start = _time.time()
        predictions = []

        for idx, task in enumerate(tasks):
            regions = []

            # Get image path
            image_path = task.get("data", {}).get("image") or task.get("image")
            if not image_path:
                logger.warning(f"Task {idx} has no image")
                continue

            # Handle S3 URLs
            if image_path.startswith("s3://"):
                import boto3
                from urllib.parse import urlparse
                parsed = urlparse(image_path)
                bucket = parsed.netloc
                key = parsed.path.lstrip("/")
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    s3 = boto3.client("s3")
                    s3.download_file(bucket, key, tmp.name)
                    image_path = tmp.name

            try:
                result = self._run_inference(image_path)
                regions = self._parse_results(result)
                logger.info(
                    "predict task=%d -> %d regions",
                    idx, len(regions)
                )
            except Exception as e:
                logger.exception(f"predict task={idx} failed: {e}")
                raise

            # Calculate final score
            all_scores = [region["score"] for region in regions if "score" in region]
            avg_score = sum(all_scores) / max(len(all_scores), 1)

            # Compose final prediction
            prediction = {
                "result": regions,
                "score": avg_score,
                "model_version": "ppyoloe",
            }
            predictions.append(prediction)

        logger.info(
            "prediction done: %d tasks, %d total regions, %.2fs",
            len(tasks),
            sum(len(p["result"]) for p in predictions),
            _time.time() - t_start,
        )
        return {"predictions": predictions}

    def _parse_results(self, result):
        """Parse PP-YOLOE results to Label Studio format"""
        regions = []
        if isinstance(result, dict) and "boxes" in result:
            boxes = result["boxes"]
            scores = result.get("scores", [])
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                region = {
                    "original_width": 1.0,
                    "original_height": 1.0,
                    "image_rotation": 0,
                    "value": {
                        "x": float(x1),
                        "y": float(y1),
                        "width": float(w),
                        "height": float(h),
                        "rotation": 0,
                        "rectanglelabels": ["object"],
                    },
                    "score": float(scores[i]) if i < len(scores) else 0.5,
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                }
                regions.append(region)
        return regions

    def fit(self, event, data, **kwargs):
        """Training not supported in this version - use PaddleDetection directly"""
        logger.warning("Training not supported in PP-YOLOE Label Studio backend. Use PaddleDetection directly for training.")
        return {"message": "Training not supported. Use PaddleDetection directly for training."}
