"""Detections -> segments: outer-box containment suppression, NMS, generous padding, crops.

numpy only (no torch) so it is cheap to test and safe for the dataset worker to import.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detector box in global frame pixels (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str = ""

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Segment:
    segment_index: int
    score: float
    label: str
    raw_bbox: tuple[float, float, float, float]
    padded_bbox: tuple[int, int, int, int]
    crop: np.ndarray  # view into the frame (no copy)
    source: str = "barcode"
    barcodes: list[Any] = field(default_factory=list)  # list[DecodedBarcode]
    decode_error: str | None = None
    ocr_result: Any = None
    ocr_error: str | None = None
    agent_result: Any = None
    agent_error: str | None = None


def normalize_prompt(prompt: str) -> str:
    """Grounding DINO wants lowercase phrases terminated by '.': 'Barcode' -> 'barcode.'"""
    p = re.sub(r"\s+", " ", prompt.strip().lower())
    if not p:
        raise ValueError("prompt is empty")
    return p if p.endswith(".") else p + "."


def sanitize_detections(dets: Iterable[Detection], width: int, height: int) -> list[Detection]:
    """Drop non-finite/degenerate boxes, clamp to the frame, sort by score (desc)."""
    out: list[Detection] = []
    for d in dets:
        vals = (d.x1, d.y1, d.x2, d.y2, d.score)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            continue
        x1, x2 = sorted((min(max(d.x1, 0.0), width), min(max(d.x2, 0.0), width)))
        y1, y2 = sorted((min(max(d.y1, 0.0), height), min(max(d.y2, 0.0), height)))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        out.append(Detection(float(x1), float(y1), float(x2), float(y2), float(d.score), d.label))
    out.sort(key=lambda d: d.score, reverse=True)
    return out


def box_area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)


def _pairwise_intersection(boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def suppress_contained(boxes: np.ndarray, threshold: float) -> np.ndarray:
    """Keep-mask that drops every OUTER box containing a smaller box.

    For a pair with Area(B_i) > Area(B_j): if Area(B_i n B_j) / Area(B_j) >= threshold, the
    outer B_i is discarded and the inner B_j kept. Equal areas are left to NMS.
    """
    n = len(boxes)
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return keep
    areas = box_area(boxes)
    inter = _pairwise_intersection(boxes)
    with np.errstate(divide="ignore", invalid="ignore"):
        containment = np.where(areas[None, :] > 0, inter / areas[None, :], 0.0)  # [i, j]
    larger = areas[:, None] > areas[None, :]
    keep &= ~np.any(larger & (containment >= threshold), axis=1)
    return keep


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy class-agnostic NMS. Returns kept indices in descending score order."""
    if len(boxes) == 0:
        return []
    areas = box_area(boxes)
    inter = _pairwise_intersection(boxes)
    union = areas[:, None] + areas[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    order = list(np.argsort(-scores, kind="stable"))
    kept: list[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        kept.append(int(i))
        suppressed |= iou[i] > iou_threshold
    return kept


def pad_box(
    box: tuple[float, float, float, float], pad_x: float, pad_y: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """pad_x/pad_y are fractions of the box's own width/height; result clamped to the frame."""
    x1, y1, x2, y2 = box
    px = (x2 - x1) * pad_x
    py = (y2 - y1) * pad_y
    return (
        max(0, math.floor(x1 - px)),
        max(0, math.floor(y1 - py)),
        min(width, math.ceil(x2 + px)),
        min(height, math.ceil(y2 + py)),
    )


def build_segments(
    frame: np.ndarray,
    detections: Iterable[Detection],
    *,
    containment_threshold: float,
    nms_iou: float,
    pad_x: float,
    pad_y: float,
    min_side_px: int = 4,
    max_count: int = 20,
    source: str = "barcode",
) -> list[Segment]:
    """sanitize -> containment suppression -> NMS -> top-N -> pad -> crop views."""
    height, width = frame.shape[:2]
    dets = sanitize_detections(detections, width, height)
    if not dets:
        return []
    boxes = np.array([d.box for d in dets], dtype=np.float64)
    scores = np.array([d.score for d in dets], dtype=np.float64)

    keep = np.flatnonzero(suppress_contained(boxes, containment_threshold))
    kept_local = nms(boxes[keep], scores[keep], nms_iou)
    survivors = [dets[keep[i]] for i in kept_local]
    survivors.sort(key=lambda d: d.score, reverse=True)

    segments: list[Segment] = []
    for d in survivors[:max_count]:
        px1, py1, px2, py2 = pad_box(d.box, pad_x, pad_y, width, height)
        if px2 - px1 < min_side_px or py2 - py1 < min_side_px:
            continue
        segments.append(
            Segment(
                segment_index=len(segments),
                score=d.score,
                label=d.label,
                raw_bbox=d.box,
                padded_bbox=(px1, py1, px2, py2),
                crop=frame[py1:py2, px1:px2],
                source=source,
            )
        )
    return segments
