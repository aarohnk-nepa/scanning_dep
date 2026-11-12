"""Pydantic request/response models for /api/v1/scan."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

StickerType = Literal["shipping_label", "product_label", "handling", "warning", "other", "unknown"]


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def of(cls, box) -> BBox:
        x1, y1, x2, y2 = box
        return cls(
            x1=round(float(x1), 1),
            y1=round(float(y1), 1),
            x2=round(float(x2), 1),
            y2=round(float(y2), 1),
        )


class Barcode(BaseModel):
    symbology: str  # ZBar vocabulary: "QRCODE", "CODE128", "EAN13", "I25", "DATAMATRIX", ...
    value: str
    bbox_global: BBox
    segment_index: int
    engine: Literal["pyzbar", "zxing"]


class OcrLine(BaseModel):
    text: str
    score: float
    bbox: BBox | None = None


class OcrResult(BaseModel):
    text: str
    lines: list[OcrLine] = Field(default_factory=list)


class AgentResult(BaseModel):
    # Key order mirrors the model's output contract (raw_text first).
    raw_text: str
    fields: dict[str, str | None]
    has_machine_code: bool
    type: StickerType
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_verification: bool


class SegmentOut(BaseModel):
    segment_index: int
    source: Literal["barcode", "enrich"] = "barcode"
    score: float
    label: str
    raw_bbox: BBox
    padded_bbox: BBox
    barcodes: list[Barcode] = Field(default_factory=list)
    decode_error: str | None = None
    ocr_result: OcrResult | None = None
    ocr_error: str | None = None
    agent_result: AgentResult | None = None
    agent_error: str | None = None


class TimingMs(BaseModel):
    decode_image_ms: float
    segment_ms: float
    code_reader_ms: float
    enrich_ms: float
    total_ms: float


class ScanResponse(BaseModel):
    request_id: str
    image_width: int
    image_height: int
    barcodes: list[Barcode]
    segments: list[SegmentOut]
    timing_ms: TimingMs
    dataset_queued: bool


class ScanOptions(BaseModel):
    """Optional per-request overrides of the .env defaults."""

    model_config = ConfigDict(extra="ignore")

    prompt: str | None = Field(None, min_length=1, max_length=256)
    padding: float | None = Field(None, ge=0, le=2, description="Sets both pad_x and pad_y")
    pad_x: float | None = Field(None, ge=0, le=2)
    pad_y: float | None = Field(None, ge=0, le=2)
    box_threshold: float | None = Field(None, ge=0, le=1)
    text_threshold: float | None = Field(None, ge=0, le=1)
    enrich_prompt: str | None = Field(
        None,
        min_length=1,
        max_length=256,
        description="Second detection pass (e.g. 'shipping label') whose crops OCR/agent read "
        "instead of the barcode crops.",
    )


class ScanJsonRequest(ScanOptions):
    image_base64: str = Field(min_length=1)
