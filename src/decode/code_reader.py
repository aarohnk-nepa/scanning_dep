"""Parallel barcode/QR decoding of every padded segment crop.

Engine order per crop (fixed): pyzbar first - one pass returns every ZBar-supported symbol;
zxing-cpp ONLY when pyzbar found nothing (DataMatrix, PDF417, damaged/dense codes). The
engines' answers are never merged.

There is deliberately NO upscale retry: the segmentor targets "barcode" on the full-resolution
frame and pads generously, so synthetic enlargement adds CPU time without adding information.

Both engines release the GIL, so crops are decoded concurrently on a persistent
ThreadPoolExecutor. Format names are reported in ZBar's vocabulary whichever engine read them.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
import zxingcpp

logger = logging.getLogger(__name__)

# zxing-cpp's format names, upper-cased, already match ZBar's except for these.
_ZXING_TO_ZBAR_FORMAT = {"ITF": "I25"}

Rect = tuple[int, int, int, int]  # x, y, w, h


@dataclass(frozen=True)
class DecodedBarcode:
    symbology: str
    value: str
    engine: Literal["pyzbar", "zxing"]
    rect_local: Rect  # in crop pixels
    bbox_global: tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels


def _text(data: bytes) -> str:
    """Payload bytes as text: UTF-8 when it is, otherwise Latin-1, which loses nothing."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _zxing_format(fmt: Any) -> str:
    name = getattr(fmt, "name", None)
    if not isinstance(name, str):
        name = str(fmt).rsplit(".", 1)[-1]
    name = name.upper()
    return _ZXING_TO_ZBAR_FORMAT.get(name, name)


def _zxing_rect(position: Any) -> Rect:
    """Axis-aligned bounding box of zxing's four corner points."""
    corners = [position.top_left, position.top_right, position.bottom_right, position.bottom_left]
    xs = [int(c.x) for c in corners]
    ys = [int(c.y) for c in corners]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def load_pyzbar() -> Callable[[np.ndarray], list] | None:
    """pyzbar's decode, or None when the ZBar C library is missing (then zxing only)."""
    try:
        from pyzbar.pyzbar import decode
    except Exception as exc:  # ImportError or OSError from ctypes
        logger.warning("pyzbar unavailable (%s); decoding with zxing-cpp only", type(exc).__name__)
        return None
    return decode


def _pyzbar_results(decode: Callable, gray: np.ndarray) -> list[tuple[str, str, Rect, str]]:
    return [
        (
            str(s.type),
            _text(s.data),
            (int(s.rect.left), int(s.rect.top), int(s.rect.width), int(s.rect.height)),
            "pyzbar",
        )
        for s in decode(gray)
    ]


def _zxing_results(read: Callable, gray: np.ndarray) -> list[tuple[str, str, Rect, str]]:
    return [(_zxing_format(r.format), r.text, _zxing_rect(r.position), "zxing") for r in read(gray)]


def to_global(rect: Rect, padded_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    ox, oy = padded_bbox[0], padded_bbox[1]
    return (ox + x, oy + y, ox + x + w, oy + y + h)


class CodeReader:
    def __init__(
        self,
        executor: ThreadPoolExecutor,
        *,
        pyzbar_decode: Callable[[np.ndarray], list] | None | Literal["auto"] = "auto",
        zxing_read: Callable[[np.ndarray], list] | None = zxingcpp.read_barcodes,
    ) -> None:
        self._executor = executor
        self._pyzbar = load_pyzbar() if pyzbar_decode == "auto" else pyzbar_decode
        self._zxing = zxing_read

    def read_codes(self, gray: np.ndarray) -> list[tuple[str, str, Rect, str]]:
        """pyzbar first; zxing-cpp only if pyzbar returned nothing."""
        if self._pyzbar is not None:
            found = _pyzbar_results(self._pyzbar, gray)
            if found:
                return found
        if self._zxing is not None:
            return _zxing_results(self._zxing, gray)
        return []

    def decode_crop(
        self, crop: np.ndarray, padded_bbox: tuple[int, int, int, int]
    ) -> tuple[list[DecodedBarcode], str | None]:
        """Decode one crop; exceptions are isolated and reported, never raised."""
        try:
            img = np.ascontiguousarray(crop)
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.ascontiguousarray(gray)
            found = self.read_codes(gray)
        except Exception as exc:  # one bad crop must not sink the rest
            return [], (str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
        return [
            DecodedBarcode(
                symbology=fmt,
                value=text,
                engine=engine,  # type: ignore[arg-type]
                rect_local=rect,
                bbox_global=to_global(rect, padded_bbox),
            )
            for fmt, text, rect, engine in found
        ], None

    def decode_segments(
        self, segments: Sequence[Any]
    ) -> list[tuple[list[DecodedBarcode], str | None]]:
        """Synchronous: submit every crop to the pool, collect in segment order."""
        futures = [self._executor.submit(self.decode_crop, s.crop, s.padded_bbox) for s in segments]
        return [f.result() for f in futures]

    async def adecode_segments(
        self, segments: Sequence[Any]
    ) -> list[tuple[list[DecodedBarcode], str | None]]:
        """Async: all crops in parallel on the pool; results attached to the segments."""
        loop = asyncio.get_running_loop()
        results = await asyncio.gather(
            *(
                loop.run_in_executor(self._executor, self.decode_crop, s.crop, s.padded_bbox)
                for s in segments
            )
        )
        for seg, (codes, err) in zip(segments, results, strict=True):
            seg.barcodes = codes
            seg.decode_error = err
        return list(results)


def dedupe_barcodes(segments: Sequence[Any]) -> list[tuple[int, DecodedBarcode]]:
    """Flat list across segments, deduplicated by (symbology, value); first segment wins.

    Adjacent padded crops can overlap, so the same physical code may be read twice.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[int, DecodedBarcode]] = []
    for seg in sorted(segments, key=lambda s: s.segment_index):
        for code in seg.barcodes:
            key = (code.symbology, code.value)
            if key in seen:
                continue
            seen.add(key)
            out.append((seg.segment_index, code))
    return out
