"""In-process Grounding DINO (Hugging Face transformers port), loaded once at startup.

Replaces the notebook's HTTP-hosted model: the frame stays an in-memory ndarray end-to-end.
torch/transformers are imported lazily inside `load()` so importing this module is cheap.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

import cv2
import numpy as np

from src.segment.postprocess import Detection, normalize_prompt

__all__ = ["GroundingDinoSegmentor", "Segmentor", "create_segmentor", "normalize_prompt"]

logger = logging.getLogger(__name__)


class Segmentor(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def device(self) -> str: ...

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def detect(
        self,
        frame_bgr: np.ndarray,
        *,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        frame_bytes: bytes | None = None,
        exif_orientation: int = 1,
    ) -> list[Detection]: ...

    def close(self) -> None: ...


class GroundingDinoSegmentor:
    def __init__(self, model_id: str, device: str = "cuda", max_side: int = 1333) -> None:
        self.model_id = model_id
        self._requested_device = device
        self._device = "cpu"
        self.max_side = max_side
        self._lock = threading.Lock()
        self._model: Any = None
        self._processor: Any = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = self._requested_device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("GDINO_DEVICE=cuda but CUDA is unavailable; falling back to cpu")
            device = "cpu"
        self._device = device
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self._model = model.to(device).eval()
        if device == "cuda":
            cap = torch.cuda.get_device_capability()
            logger.info(
                "Grounding DINO %s on %s (sm_%d%d)",
                self.model_id,
                torch.cuda.get_device_name(),
                *cap,
            )
        else:
            logger.info("Grounding DINO %s on cpu", self.model_id)

    def warmup(self) -> None:
        frame = np.full((480, 640, 3), 127, dtype=np.uint8)
        self.detect(frame, prompt="barcode", box_threshold=0.3, text_threshold=0.25)

    def detect(
        self,
        frame_bgr: np.ndarray,
        *,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        frame_bytes: bytes | None = None,
        exif_orientation: int = 1,
    ) -> list[Detection]:
        import torch

        if not self.ready:
            raise RuntimeError("segmentor not loaded")
        orig_h, orig_w = frame_bgr.shape[:2]
        img = frame_bgr
        long_side = max(orig_h, orig_w)
        if long_side > self.max_side:
            # The processor would resize anyway; INTER_AREA here is cheaper and cleaner.
            s = self.max_side / long_side
            img = cv2.resize(
                img,
                (max(1, round(orig_w * s)), max(1, round(orig_h * s))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        text = normalize_prompt(prompt)

        with self._lock, torch.inference_mode():
            inputs = self._processor(images=rgb, text=text, return_tensors="pt").to(self._device)
            outputs = self._model(**inputs)
            # Boxes come back normalised internally, so target_sizes=original frame maps them
            # straight into full-resolution pixel coordinates.
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(orig_h, orig_w)],
            )[0]

        labels = results.get("text_labels") or results.get("labels") or []
        boxes = results["boxes"].detach().cpu().numpy().tolist()
        scores = results["scores"].detach().cpu().numpy().tolist()
        return [
            Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), str(lbl))
            for b, s, lbl in zip(boxes, scores, labels, strict=False)
        ]

    def close(self) -> None:
        model, self._model = self._model, None
        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def create_segmentor(settings: Any) -> Segmentor:
    """Local HF model by default; the legacy hosted HTTP API when USE_SEGMENTOR_API=true."""
    if settings.USE_SEGMENTOR_API:
        from src.segment.remote_api import RemoteSegmentor

        return RemoteSegmentor(
            settings.SEGMENTOR_API_URL, timeout_s=settings.SEGMENTOR_API_TIMEOUT_S
        )
    return GroundingDinoSegmentor(
        settings.GDINO_MODEL_ID, settings.GDINO_DEVICE, settings.GDINO_MAX_SIDE
    )
