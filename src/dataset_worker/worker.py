"""Background dataset collector: a dedicated OS process fed by a multiprocessing.Queue.

The API process enqueues a DatasetJob (the ORIGINAL encoded upload bytes + segment boxes and
scanner results) with put_nowait() and never waits. The worker decodes the frame with the same
image_io.load_image (same EXIF handling), re-slices the padded crops, verifies them with Gemma
and writes the sample via DatasetStorage.

IMPORT HYGIENE: this module (and everything it imports) must never import torch/transformers.
The child is started with the "spawn" method, so it re-imports this module from scratch and
must not touch CUDA.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import queue as queue_mod
import signal
from dataclasses import dataclass, field
from typing import Any

from src.config import WorkerConfig
from src.dataset_worker.storage import DatasetStorage, SegmentVerdict
from src.utils.image_io import load_image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentPayload:
    segment_index: int
    score: float
    label: str
    raw_bbox: tuple[float, float, float, float]
    padded_bbox: tuple[int, int, int, int]
    scanner_decoded: bool
    barcodes: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetJob:
    request_id: str
    timestamp: str  # ISO-8601 UTC
    frame_bytes: bytes  # original upload bytes (~10x smaller than the decoded 12MP array)
    frame_shape: tuple[int, int]  # expected upright (height, width), checked after decoding
    segments_payload: list[SegmentPayload]
    provenance: dict[str, Any] = field(default_factory=dict)


def _build_verifier(cfg: WorkerConfig):
    if not cfg.gemma_base_url:
        logger.warning("GEMMA_API_BASE_URL is empty; samples are stored unverified")
        return None
    from src.dataset_worker.gemma_verifier import GemmaVerifier

    return GemmaVerifier(
        base_url=cfg.gemma_base_url,
        api_key=cfg.gemma_api_key,
        model=cfg.gemma_model,
        mode=cfg.gemma_mode,
        max_batch=cfg.gemma_max_batch,
        timeout_s=cfg.gemma_timeout_s,
        max_concurrency=cfg.gemma_max_concurrency,
        crop_max_side=cfg.gemma_crop_max_side,
        use_system_role=cfg.gemma_use_system_role,
        retries=cfg.gemma_retries,
        cooldown_s=cfg.gemma_cooldown_s,
    )


def process_job(
    job: DatasetJob,
    storage: DatasetStorage,
    verifier: Any,
    loop: asyncio.AbstractEventLoop,
    *,
    skip_gemma: bool = False,
    save_empty: bool = True,
) -> bool:
    """Verify and persist one job. Returns True when a sample was written."""
    if not job.segments_payload and not save_empty:
        return False
    decoded = load_image(job.frame_bytes)
    if (decoded.height, decoded.width) != tuple(job.frame_shape):
        logger.warning(
            "dataset job %s dropped: decoded shape %s != expected %s",
            job.request_id,
            (decoded.height, decoded.width),
            tuple(job.frame_shape),
        )
        return False
    frame = decoded.bgr
    crops = [
        frame[p.padded_bbox[1] : p.padded_bbox[3], p.padded_bbox[0] : p.padded_bbox[2]]
        for p in job.segments_payload
    ]

    verdicts: list[bool | None] = [None] * len(crops)
    if crops and verifier is not None and not skip_gemma:
        try:
            verdicts = loop.run_until_complete(verifier.verify(crops))
        except Exception as exc:
            logger.warning("Gemma verification failed (%s)", type(exc).__name__)

    seg_verdicts = [
        SegmentVerdict(
            segment_index=p.segment_index,
            score=p.score,
            label=p.label,
            raw_bbox=tuple(p.raw_bbox),
            padded_bbox=tuple(p.padded_bbox),
            scanner_decoded=p.scanner_decoded,
            gemma_verified=v,
            barcodes=list(p.barcodes),
        )
        for p, v in zip(job.segments_payload, verdicts, strict=True)
    ]
    provenance = dict(job.provenance)
    if verifier is not None:
        provenance.setdefault("gemma_model", getattr(verifier, "model", None))
        provenance.setdefault("gemma_mode", getattr(verifier, "mode", None))
    storage.write_sample(
        request_id=job.request_id,
        timestamp=job.timestamp,
        frame=frame,
        verdicts=seg_verdicts,
        provenance=provenance,
    )
    return True


def run_worker(job_queue: Any, fast_drain: Any, cfg: WorkerConfig) -> None:
    """Spawn target. Consumes jobs until the None sentinel arrives."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Ctrl+C in a dev terminal hits the whole process group; let the parent drain us instead.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(
        level=cfg.log_level.upper(),
        format="%(asctime)s %(levelname)s [dataset-worker] %(name)s: %(message)s",
    )
    storage = DatasetStorage(cfg.dataset_dir, cfg.jpeg_quality)
    storage.ensure_layout()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    verifier = _build_verifier(cfg)
    logger.info("dataset worker started (pid=%d, dir=%s)", os.getpid(), cfg.dataset_dir)
    written = 0
    try:
        while True:
            job = job_queue.get()
            if job is None:
                break
            try:
                if process_job(
                    job,
                    storage,
                    verifier,
                    loop,
                    skip_gemma=fast_drain.is_set(),
                    save_empty=cfg.save_empty,
                ):
                    written += 1
            except Exception:
                logger.exception("dataset job failed")
    finally:
        if verifier is not None:
            try:
                loop.run_until_complete(verifier.aclose())
            except Exception:
                pass
        loop.close()
        logger.info("dataset worker exiting (%d samples written)", written)


class DatasetWorkerProcess:
    """Parent-side handle: owns the queue and the spawned child process."""

    def __init__(self, cfg: WorkerConfig, maxsize: int = 32) -> None:
        self.cfg = cfg
        self._ctx = mp.get_context("spawn")
        self._queue = self._ctx.Queue(maxsize=maxsize)
        self._fast_drain = self._ctx.Event()
        self._proc: Any = None
        self._closed = False

    def start(self) -> None:
        self._proc = self._ctx.Process(
            target=run_worker,
            args=(self._queue, self._fast_drain, self.cfg),
            name="dataset-worker",
            daemon=False,
        )
        self._proc.start()
        logger.info("dataset worker spawned (pid=%s)", self._proc.pid)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    @property
    def approx_qsize(self) -> int | None:
        try:
            return self._queue.qsize()
        except NotImplementedError:  # macOS
            return None

    def submit(self, job: DatasetJob) -> bool:
        """Non-blocking enqueue. Never raises; False when the sample was dropped."""
        if self._closed or not self.alive:
            return False
        try:
            self._queue.put_nowait(job)
            return True
        except queue_mod.Full:
            logger.warning("dataset queue full; dropping sample %s", job.request_id)
        except (ValueError, OSError):
            logger.warning("dataset queue closed; dropping sample %s", job.request_id)
        return False

    def stop(self, timeout_s: float = 20.0) -> None:
        """Drain gracefully: sentinel -> join -> fast drain (skip Gemma) -> terminate."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        terminated = False
        if proc.is_alive():
            try:
                self._queue.put(None, timeout=2)
            except Exception:
                pass
            proc.join(timeout_s * 0.5)
            if proc.is_alive():
                logger.warning("dataset worker still draining; skipping Gemma for the rest")
                self._fast_drain.set()
                proc.join(timeout_s * 0.5)
            if proc.is_alive():
                logger.warning("dataset worker did not drain in time; terminating")
                proc.terminate()
                proc.join(5)
                terminated = True
        if terminated:
            self._queue.cancel_join_thread()
        self._queue.close()
        logger.info("dataset worker stopped (exitcode=%s)", proc.exitcode)
