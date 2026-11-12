"""Shared fakes and synthetic barcode images. No GPU, no model weights, no network."""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest
import zxingcpp

# macOS + Homebrew: let pyzbar's ctypes loader find libzbar.
if sys.platform == "darwin":
    os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib:/usr/local/lib")

from src.decode.code_reader import load_pyzbar  # noqa: E402
from src.segment.postprocess import Detection  # noqa: E402

HAS_PYZBAR = load_pyzbar() is not None
needs_pyzbar = pytest.mark.skipif(not HAS_PYZBAR, reason="libzbar not installed")


def make_code(text: str, fmt: str = "QRCode", scale: int = 4) -> np.ndarray:
    """Grayscale uint8 barcode image with a white quiet zone."""
    bc = zxingcpp.create_barcode(text, getattr(zxingcpp.BarcodeFormat, fmt))
    img = np.asarray(bc.to_image(scale=scale), dtype=np.uint8)
    return np.pad(img, 8 * scale // 2, constant_values=255)


def canvas_with_codes(
    placements: list[tuple[np.ndarray, int, int]], size: tuple[int, int] = (900, 1200)
) -> np.ndarray:
    """White BGR canvas (h, w) with grayscale codes pasted at (x, y)."""
    h, w = size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for code, x, y in placements:
        ch, cw = code.shape[:2]
        canvas[y : y + ch, x : x + cw] = cv2.cvtColor(code, cv2.COLOR_GRAY2BGR)
    return canvas


def jpeg_bytes(img: np.ndarray, quality: int = 95) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return buf.tobytes()


def png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class FakeSegmentor:
    """Returns fixed detections per prompt; records every call."""

    def __init__(self, detections=None, by_prompt: dict[str, list[Detection]] | None = None):
        self.detections = detections or []
        self.by_prompt = by_prompt or {}
        self.calls: list[dict] = []

    ready = True
    device = "fake"

    def load(self):
        pass

    def warmup(self):
        pass

    def close(self):
        pass

    def detect(
        self,
        frame_bgr,
        *,
        prompt,
        box_threshold,
        text_threshold,
        frame_bytes=None,
        exif_orientation=1,
    ):
        from src.segment.postprocess import normalize_prompt

        p = normalize_prompt(prompt)
        self.calls.append(
            {
                "prompt": p,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "shape": frame_bgr.shape,
            }
        )
        return list(self.by_prompt.get(p, self.detections))


class FakeDispatcher:
    def __init__(self, full: bool = False):
        self.full = full
        self.jobs: list = []
        self.stopped = False

    @property
    def alive(self) -> bool:
        return True

    def submit(self, job) -> bool:
        if self.full:
            return False
        self.jobs.append(job)
        return True

    def stop(self, timeout_s: float) -> None:
        self.stopped = True
