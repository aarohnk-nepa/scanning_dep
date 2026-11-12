"""FastAPI app: GET/POST /api/v1/scan.

Pipeline per request (fixed order):
    receive image -> decode + EXIF-normalise (once) -> segment (Grounding DINO)
      -> parallel code reader (pyzbar -> zxing-cpp)  -> [enqueue dataset job, non-blocking]
      -> optional OCR / VLM agent enrichment          -> JSON response

The frame and crops stay in-memory numpy arrays end-to-end. Run a single uvicorn worker: each
worker would load its own model and spawn its own dataset collector.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Any, Protocol

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from src.config import Settings, get_settings
from src.decode.code_reader import CodeReader, dedupe_barcodes
from src.schemas import (
    Barcode,
    BBox,
    ScanJsonRequest,
    ScanOptions,
    ScanResponse,
    SegmentOut,
    TimingMs,
)
from src.segment.postprocess import Segment, build_segments
from src.utils.image_io import (
    ImageDecodeError,
    ImageTooLargeError,
    decode_base64_image,
    load_image,
)

logger = logging.getLogger("outbound_vision")

_IMAGE_FIELDS = ("file", "image")
_RAW_IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp", "application/octet-stream")


class DatasetDispatcher(Protocol):
    @property
    def alive(self) -> bool: ...

    def submit(self, job: Any) -> bool: ...

    def stop(self, timeout_s: float) -> None: ...


@dataclass
class PipelineState:
    settings: Settings
    segmentor: Any
    code_reader: CodeReader
    gpu_executor: ThreadPoolExecutor
    scan_executor: ThreadPoolExecutor
    dispatcher: DatasetDispatcher | None = None
    ocr: Any = None
    ocr_executor: ThreadPoolExecutor | None = None
    agent: Any = None


# --------------------------------------------------------------------------- input parsing


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


async def read_scan_input(request: Request, settings: Settings) -> tuple[bytes, ScanOptions]:
    """One handler for every way a client can send an image (GET or POST).

    multipart/form-data (field `file` or `image`), application/json ({"image_base64": ...}),
    a raw image/* body, or - with an empty body - ?image_base64= in the query string.
    Override precedence: body > query > settings.
    """
    max_bytes = settings.MAX_IMAGE_BYTES
    # Multipart/base64 overhead: allow ~1.4x the image limit for the whole body.
    body_limit = int(max_bytes * 1.4) + 64 * 1024
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > body_limit:
        raise HTTPException(413, f"request body too large (limit {body_limit} bytes)")

    query = {k: v for k, v in request.query_params.items()}
    q_image = query.pop("image_base64", None)
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    options: dict[str, Any] = dict(query)
    image: bytes | None = None

    if ctype in ("multipart/form-data", "application/x-www-form-urlencoded"):
        form = await request.form()
        for key, value in form.multi_items():
            if key in _IMAGE_FIELDS and isinstance(value, UploadFile):
                if image is None:
                    image = await value.read()
            elif key == "image_base64" and isinstance(value, str):
                image = decode_base64_image(value)
            elif isinstance(value, str):
                options[key] = value
    elif ctype == "application/json":
        raw = await request.body()
        if raw.strip():
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                raise HTTPException(422, "body is not valid JSON") from None
            if not isinstance(payload, dict):
                raise HTTPException(422, "JSON body must be an object")
            if "image_base64" in payload:
                try:
                    req = ScanJsonRequest.model_validate({**options, **payload})
                except ValidationError as exc:
                    raise HTTPException(
                        422, exc.errors(include_url=False, include_input=False)
                    ) from None
                return decode_base64_image(req.image_base64), req
            options.update(payload)
    elif ctype in _RAW_IMAGE_TYPES or ctype.startswith("image/"):
        image = await request.body() or None
    elif ctype:
        body = await request.body()
        if body:
            raise HTTPException(415, f"unsupported content type {ctype!r}")

    if image is None and q_image:
        image = decode_base64_image(q_image)
    if image is None:
        raise HTTPException(
            422,
            "no image: send multipart field 'file', JSON {'image_base64': ...}, "
            "a raw image/* body, or ?image_base64=",
        )
    try:
        opts = ScanOptions.model_validate(options)
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_url=False, include_input=False)) from None
    return image, opts


# --------------------------------------------------------------------------- pipeline


async def _detect(
    state: PipelineState,
    frame,
    data: bytes,
    orientation: int,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
):
    loop = asyncio.get_running_loop()
    fn = partial(
        state.segmentor.detect,
        frame,
        prompt=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        frame_bytes=data,
        exif_orientation=orientation,
    )
    try:
        return await loop.run_in_executor(state.gpu_executor, fn)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except Exception as exc:
        name = type(exc).__name__
        if name == "OutOfMemoryError" or getattr(exc, "outage", False):
            logger.warning("segmentation unavailable: %s", exc)
            raise HTTPException(503, f"segmentation unavailable ({name})") from None
        if hasattr(exc, "outage"):  # remote API refused the request (4xx)
            raise HTTPException(502, str(exc)) from None
        logger.exception("segmentation failed")
        raise HTTPException(500, f"segmentation failed ({name})") from None


def _barcode_dicts(seg: Segment) -> list[dict[str, Any]]:
    return [
        {
            "symbology": c.symbology,
            "value": c.value,
            "engine": c.engine,
            "bbox_global": list(c.bbox_global),
        }
        for c in seg.barcodes
    ]


def _dispatch_dataset(
    state: PipelineState,
    request_id: str,
    data: bytes,
    decoded,
    segments: list[Segment],
    provenance: dict[str, Any],
) -> bool:
    if state.dispatcher is None or not state.settings.ENABLE_DATA_COLLECTION:
        return False
    from src.dataset_worker.worker import DatasetJob, SegmentPayload

    job = DatasetJob(
        request_id=request_id,
        timestamp=datetime.now(UTC).isoformat(),
        frame_bytes=data,
        frame_shape=(decoded.height, decoded.width),
        segments_payload=[
            SegmentPayload(
                segment_index=s.segment_index,
                score=float(s.score),
                label=s.label,
                raw_bbox=tuple(float(v) for v in s.raw_bbox),
                padded_bbox=tuple(int(v) for v in s.padded_bbox),
                scanner_decoded=bool(s.barcodes),
                barcodes=_barcode_dicts(s),
            )
            for s in segments
        ],
        provenance=provenance,
    )
    try:
        return bool(state.dispatcher.submit(job))
    except Exception:
        logger.exception("dataset dispatch failed")
        return False


def _segment_out(seg: Segment) -> SegmentOut:
    return SegmentOut(
        segment_index=seg.segment_index,
        source=seg.source,  # type: ignore[arg-type]
        score=round(float(seg.score), 4),
        label=seg.label,
        raw_bbox=BBox.of(seg.raw_bbox),
        padded_bbox=BBox.of(seg.padded_bbox),
        barcodes=[
            Barcode(
                symbology=c.symbology,
                value=c.value,
                bbox_global=BBox.of(c.bbox_global),
                segment_index=seg.segment_index,
                engine=c.engine,
            )
            for c in seg.barcodes
        ],
        decode_error=seg.decode_error,
        ocr_result=seg.ocr_result,
        ocr_error=seg.ocr_error,
        agent_result=seg.agent_result,
        agent_error=seg.agent_error,
    )


async def run_scan(state: PipelineState, data: bytes, opts: ScanOptions) -> ScanResponse:
    s = state.settings
    if not getattr(state.segmentor, "ready", True):
        raise HTTPException(503, "segmentor not ready")
    request_id = str(uuid.uuid4())
    t_total = time.perf_counter()

    # 1. decode + EXIF normalisation (once)
    t0 = time.perf_counter()
    try:
        decoded = await asyncio.to_thread(load_image, data, s.MAX_IMAGE_BYTES)
    except ImageTooLargeError as exc:
        raise HTTPException(413, str(exc)) from None
    except ImageDecodeError as exc:
        raise HTTPException(400, str(exc)) from None
    decode_ms = _ms(t0)

    prompt = opts.prompt or s.SEGMENT_PROMPT
    box_t = opts.box_threshold if opts.box_threshold is not None else s.GDINO_BOX_THRESHOLD
    text_t = opts.text_threshold if opts.text_threshold is not None else s.GDINO_TEXT_THRESHOLD
    pad_x = (
        opts.pad_x
        if opts.pad_x is not None
        else (opts.padding if opts.padding is not None else s.SEGMENT_PAD_X)
    )
    pad_y = (
        opts.pad_y
        if opts.pad_y is not None
        else (opts.padding if opts.padding is not None else s.SEGMENT_PAD_Y)
    )
    seg_kwargs = dict(
        containment_threshold=s.CONTAINMENT_THRESHOLD,
        nms_iou=s.NMS_IOU_THRESHOLD,
        pad_x=pad_x,
        pad_y=pad_y,
        min_side_px=s.MIN_CROP_SIDE_PX,
        max_count=s.SEGMENT_MAX_COUNT,
    )

    # 2. segment
    t0 = time.perf_counter()
    dets = await _detect(state, decoded.bgr, data, decoded.exif_orientation, prompt, box_t, text_t)
    segments = build_segments(decoded.bgr, dets, **seg_kwargs)
    segment_ms = _ms(t0)

    # 3. parallel code reader
    t0 = time.perf_counter()
    if segments:
        await state.code_reader.adecode_segments(segments)
    code_reader_ms = _ms(t0)

    # 4. background dataset collector (non-blocking put_nowait)
    provenance = {
        "prompt": prompt,
        "box_threshold": box_t,
        "text_threshold": text_t,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "containment_threshold": s.CONTAINMENT_THRESHOLD,
        "nms_iou_threshold": s.NMS_IOU_THRESHOLD,
        "segmentor": "remote_api" if s.USE_SEGMENTOR_API else s.GDINO_MODEL_ID,
    }
    dataset_queued = _dispatch_dataset(state, request_id, data, decoded, segments, provenance)

    # 5. optional enrichment (OCR, then VLM agent)
    t0 = time.perf_counter()
    enrich_segments: list[Segment] = []
    if state.ocr is not None or state.agent is not None:
        targets = segments
        if opts.enrich_prompt:
            e_dets = await _detect(
                state,
                decoded.bgr,
                data,
                decoded.exif_orientation,
                opts.enrich_prompt,
                box_t,
                text_t,
            )
            enrich_segments = build_segments(decoded.bgr, e_dets, **seg_kwargs, source="enrich")
            for seg in enrich_segments:
                seg.segment_index += len(segments)
            targets = enrich_segments
        if state.ocr is not None and targets:
            from src.enrich.ocr_engine import run_ocr

            await run_ocr(state.ocr, targets, state.ocr_executor or state.scan_executor)
        if state.agent is not None and targets:
            await state.agent.enrich(targets)
    enrich_ms = _ms(t0)

    barcodes = [
        Barcode(
            symbology=c.symbology,
            value=c.value,
            bbox_global=BBox.of(c.bbox_global),
            segment_index=idx,
            engine=c.engine,
        )
        for idx, c in dedupe_barcodes(segments)
    ]
    return ScanResponse(
        request_id=request_id,
        image_width=decoded.width,
        image_height=decoded.height,
        barcodes=barcodes,
        segments=[_segment_out(seg) for seg in [*segments, *enrich_segments]],
        timing_ms=TimingMs(
            decode_image_ms=decode_ms,
            segment_ms=segment_ms,
            code_reader_ms=code_reader_ms,
            enrich_ms=enrich_ms,
            total_ms=_ms(t_total),
        ),
        dataset_queued=dataset_queued,
    )


# --------------------------------------------------------------------------- app factory

_SCAN_OPENAPI = {
    "requestBody": {
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "format": "binary"},
                        **{k: {"type": "string"} for k in ScanOptions.model_fields},
                    },
                    "required": ["file"],
                }
            },
            "application/json": {"schema": ScanJsonRequest.model_json_schema()},
            "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
        },
        "required": False,
    }
}


def create_app(
    settings: Settings | None = None,
    *,
    segmentor: Any = None,
    dispatcher: DatasetDispatcher | None = None,
    ocr: Any = None,
    agent: Any = None,
) -> FastAPI:
    """Build the app. Keyword arguments inject components (tests); None means build from env."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.LOG_LEVEL.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned_dispatcher = False
        disp = dispatcher
        if settings.ENABLE_DATA_COLLECTION and disp is None:
            from src.dataset_worker.worker import DatasetWorkerProcess

            disp = DatasetWorkerProcess(
                settings.worker_config(), maxsize=settings.DATASET_QUEUE_MAXSIZE
            )
            disp.start()
            owned_dispatcher = True

        scan_executor = ThreadPoolExecutor(
            settings.SCANNER_THREADS, thread_name_prefix="code-reader"
        )
        gpu_executor = ThreadPoolExecutor(1, thread_name_prefix="segmentor")

        seg = segmentor
        owned_segmentor = seg is None
        if seg is None:
            from src.segment.grounding_dino import create_segmentor

            seg = create_segmentor(settings)
            t0 = time.perf_counter()
            await asyncio.to_thread(seg.load)
            await asyncio.to_thread(seg.warmup)
            logger.info("segmentor ready on %s in %.1fs", seg.device, time.perf_counter() - t0)

        ocr_engine, ocr_executor = ocr, None
        if settings.ENABLE_OCR:
            if ocr_engine is None:
                from src.enrich.ocr_engine import create_ocr_engine

                ocr_engine = await asyncio.to_thread(create_ocr_engine, settings.OCR_BACKEND)
            ocr_executor = ThreadPoolExecutor(1, thread_name_prefix="ocr")
        else:
            ocr_engine = None

        vlm = agent if settings.ENABLE_AGENT else None
        owned_agent = False
        if settings.ENABLE_AGENT and vlm is None:
            from src.enrich.vlm_agent import VlmAgent

            vlm = VlmAgent(
                base_url=settings.agent_base_url,
                api_key=settings.agent_api_key,
                model=settings.agent_model,
                timeout_s=settings.AGENT_TIMEOUT_S,
                max_concurrency=settings.AGENT_MAX_CONCURRENCY,
                use_system_role=settings.GEMMA_USE_SYSTEM_ROLE,
            )
            owned_agent = True

        app.state.pipeline = PipelineState(
            settings=settings,
            segmentor=seg,
            code_reader=CodeReader(scan_executor),
            gpu_executor=gpu_executor,
            scan_executor=scan_executor,
            dispatcher=disp,
            ocr=ocr_engine,
            ocr_executor=ocr_executor,
            agent=vlm,
        )
        logger.info(
            "ready: data_collection=%s ocr=%s agent=%s segmentor=%s",
            settings.ENABLE_DATA_COLLECTION,
            settings.ENABLE_OCR,
            settings.ENABLE_AGENT,
            "remote_api" if settings.USE_SEGMENTOR_API else settings.GDINO_MODEL_ID,
        )
        try:
            yield
        finally:
            if owned_agent and vlm is not None:
                await vlm.aclose()
            if owned_dispatcher and disp is not None:
                await asyncio.to_thread(disp.stop, settings.DATASET_SHUTDOWN_TIMEOUT_S)
            for ex in (scan_executor, gpu_executor, ocr_executor):
                if ex is not None:
                    ex.shutdown(wait=True, cancel_futures=True)
            if owned_segmentor:
                seg.close()

    app = FastAPI(
        title="Outbound Vision API",
        version="0.1.0",
        docs_url="/api/v1/docs",
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
    )
    router = APIRouter(prefix="/api/v1")

    @router.api_route(
        "/scan",
        methods=["GET", "POST"],
        response_model=ScanResponse,
        openapi_extra=_SCAN_OPENAPI,
        summary="Segment barcodes in an image and decode them",
    )
    async def scan(request: Request) -> ScanResponse:
        state: PipelineState = request.app.state.pipeline
        try:
            data, opts = await read_scan_input(request, state.settings)
        except ImageDecodeError as exc:
            raise HTTPException(400, str(exc)) from None
        return await run_scan(state, data, opts)

    @router.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        state: PipelineState | None = getattr(request.app.state, "pipeline", None)
        if state is None:
            return {"status": "starting", "ready": False}
        disp = state.dispatcher
        return {
            "status": "healthy",
            "ready": bool(getattr(state.segmentor, "ready", True)),
            "device": getattr(state.segmentor, "device", "unknown"),
            "segmentor": "remote_api" if settings.USE_SEGMENTOR_API else settings.GDINO_MODEL_ID,
            "flags": {
                "ENABLE_DATA_COLLECTION": settings.ENABLE_DATA_COLLECTION,
                "ENABLE_OCR": settings.ENABLE_OCR,
                "ENABLE_AGENT": settings.ENABLE_AGENT,
            },
            "dataset_worker_alive": bool(disp is not None and disp.alive),
            "dataset_queue_size": getattr(disp, "approx_qsize", None) if disp else None,
        }

    app.include_router(router)
    return app


app = create_app()
