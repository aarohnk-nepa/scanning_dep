"""OCR enrichment (ENABLE_OCR=true only).

Nothing here imports an OCR library at module level; main.py imports this module only when the
flag is on, and the engine loads its weights in __init__. Install with: uv sync --extra ocr
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import numpy as np

from src.schemas import BBox, OcrLine, OcrResult

logger = logging.getLogger(__name__)


class OcrEngine(Protocol):
    def recognize(self, crop_bgr: np.ndarray) -> OcrResult: ...


def _quad_to_bbox(quad: Any) -> BBox | None:
    try:
        pts = np.asarray(quad, dtype=float).reshape(-1, 2)
        return BBox.of((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
    except Exception:
        return None


class RapidOcrEngine:
    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise RuntimeError(
                "ENABLE_OCR=true but the OCR extra is not installed: uv sync --extra ocr"
            ) from None
        self._engine = RapidOCR()

    def recognize(self, crop_bgr: np.ndarray) -> OcrResult:
        result, _ = self._engine(np.ascontiguousarray(crop_bgr))
        lines = [
            OcrLine(text=str(text), score=float(score), bbox=_quad_to_bbox(quad))
            for quad, text, score in (result or [])
        ]
        return OcrResult(text="\n".join(ln.text for ln in lines), lines=lines)


class PaddleOcrEngine:
    """Optional backend. PaddleOCR is not in the lockfile: `uv add paddleocr paddlepaddle`."""

    def __init__(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise RuntimeError("OCR_BACKEND=paddleocr but paddleocr is not installed") from None
        self._engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def recognize(self, crop_bgr: np.ndarray) -> OcrResult:
        pages = self._engine.ocr(np.ascontiguousarray(crop_bgr), cls=True) or []
        lines = []
        for page in pages:
            for quad, (text, score) in page or []:
                lines.append(OcrLine(text=str(text), score=float(score), bbox=_quad_to_bbox(quad)))
        return OcrResult(text="\n".join(ln.text for ln in lines), lines=lines)


def create_ocr_engine(backend: str) -> OcrEngine:
    if backend == "paddleocr":
        return PaddleOcrEngine()
    return RapidOcrEngine()


async def run_ocr(engine: OcrEngine, segments: Sequence[Any], executor: ThreadPoolExecutor) -> None:
    """OCR every segment crop; per-segment errors are recorded, never raised."""
    loop = asyncio.get_running_loop()

    async def one(seg: Any) -> None:
        try:
            seg.ocr_result = await loop.run_in_executor(executor, engine.recognize, seg.crop)
        except Exception as exc:
            seg.ocr_error = str(exc) if isinstance(exc, ValueError) else type(exc).__name__

    await asyncio.gather(*(one(s) for s in segments))
