"""Write one verified sample to the YOLO dataset on disk.

Layout under DATASET_DIR:
    images/{request_id}.jpg                 full upright frame
    labels/{request_id}.txt                 YOLO: "0 cx cy w h" per positive segment (raw box)
    crops/{positive,negative,unverified}/{request_id}_seg{idx}.jpg
    metadata/{request_id}.json              full record; written LAST (commit marker)

Label rule: positive  <=> gemma_verified is True OR scanner_decoded
            negative  <=> gemma_verified is False AND NOT scanner_decoded
            unverified<=> gemma_verified is None (Gemma unreachable / unparseable) AND NOT decoded
Unverified crops are never mixed into negatives; such samples get label_complete=false.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

FinalLabel = Literal["positive", "negative", "unverified"]


@dataclass(frozen=True)
class SegmentVerdict:
    segment_index: int
    score: float
    label: str
    raw_bbox: tuple[float, float, float, float]
    padded_bbox: tuple[int, int, int, int]
    scanner_decoded: bool
    gemma_verified: bool | None
    barcodes: list[dict[str, Any]] = field(default_factory=list)


def final_label(gemma_verified: bool | None, scanner_decoded: bool) -> FinalLabel:
    if gemma_verified is True or scanner_decoded:
        return "positive"
    if gemma_verified is False:
        return "negative"
    return "unverified"


def yolo_line(raw_bbox: Sequence[float], width: int, height: int, class_id: int = 0) -> str:
    x1, y1, x2, y2 = (float(v) for v in raw_bbox)

    def clamp(v: float) -> float:
        return min(max(v, 0.0), 1.0)

    cx = clamp((x1 + x2) / 2 / width)
    cy = clamp((y1 + y2) / 2 / height)
    w = clamp((x2 - x1) / width)
    h = clamp((y2 - y1) / height)
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _jpeg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


class DatasetStorage:
    def __init__(self, root: str | Path, jpeg_quality: int = 95) -> None:
        self.root = Path(root)
        self.jpeg_quality = jpeg_quality

    def ensure_layout(self) -> None:
        for sub in (
            "images",
            "labels",
            "metadata",
            "crops/positive",
            "crops/negative",
            "crops/unverified",
        ):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        classes = self.root / "classes.txt"
        if not classes.exists():
            classes.write_text("barcode\n")
        data_yaml = self.root / "data.yaml"
        if not data_yaml.exists():
            data_yaml.write_text(
                "# Ultralytics-style config; split images/ into train/val before training\n"
                "path: .\ntrain: images\nval: images\nnames:\n  0: barcode\n"
            )

    def write_sample(
        self,
        *,
        request_id: str,
        timestamp: str,
        frame: np.ndarray,
        verdicts: Sequence[SegmentVerdict],
        provenance: dict[str, Any] | None = None,
    ) -> Path:
        height, width = frame.shape[:2]
        records = []
        label_lines = []
        for v in verdicts:
            fl = final_label(v.gemma_verified, v.scanner_decoded)
            px1, py1, px2, py2 = v.padded_bbox
            crop = frame[py1:py2, px1:px2]
            if crop.size:
                name = f"{request_id}_seg{v.segment_index}.jpg"
                _atomic_write_bytes(self.root / "crops" / fl / name, _jpeg(crop, self.jpeg_quality))
            if fl == "positive":
                label_lines.append(yolo_line(v.raw_bbox, width, height))
            x1, y1, x2, y2 = v.raw_bbox
            rec = asdict(v)
            rec.update(
                raw_bbox=[round(float(c), 2) for c in v.raw_bbox],
                padded_bbox=list(v.padded_bbox),
                raw_size_px=[round(x2 - x1, 2), round(y2 - y1, 2)],
                padded_size_px=[px2 - px1, py2 - py1],
                gdino_score=round(float(v.score), 4),
                final_label=fl,
            )
            rec.pop("score", None)
            records.append(rec)

        _atomic_write_bytes(
            self.root / "images" / f"{request_id}.jpg", _jpeg(frame, self.jpeg_quality)
        )
        label_text = "\n".join(label_lines) + ("\n" if label_lines else "")
        _atomic_write_bytes(self.root / "labels" / f"{request_id}.txt", label_text.encode())

        meta = {
            "request_id": request_id,
            "timestamp": timestamp,
            "image_width": width,
            "image_height": height,
            "label_complete": all(r["final_label"] != "unverified" for r in records),
            "num_positive": sum(r["final_label"] == "positive" for r in records),
            "segments": records,
            "provenance": provenance or {},
        }
        meta_path = self.root / "metadata" / f"{request_id}.json"
        _atomic_write_bytes(meta_path, json.dumps(meta, indent=2).encode())
        return meta_path
