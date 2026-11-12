"""All configuration, loaded once from the environment / `.env`.

Field names are UPPERCASE so code reads exactly like the env file (`settings.ENABLE_OCR`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class WorkerConfig:
    """Plain, picklable subset of the settings handed to the spawned dataset worker."""

    dataset_dir: str
    jpeg_quality: int
    save_empty: bool
    log_level: str
    max_image_bytes: int
    gemma_base_url: str
    gemma_api_key: str
    gemma_model: str
    gemma_mode: str
    gemma_max_batch: int
    gemma_timeout_s: float
    gemma_max_concurrency: int
    gemma_crop_max_side: int
    gemma_use_system_role: bool
    gemma_retries: int
    gemma_cooldown_s: float


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # --- Pipeline feature flags ---
    ENABLE_DATA_COLLECTION: bool = True
    ENABLE_OCR: bool = False
    ENABLE_AGENT: bool = False

    # --- Segmentor ---
    USE_SEGMENTOR_API: bool = False
    SEGMENTOR_API_URL: str = ""
    SEGMENTOR_API_TIMEOUT_S: float = Field(180.0, gt=0)
    GDINO_MODEL_ID: str = "IDEA-Research/grounding-dino-tiny"
    GDINO_DEVICE: Literal["cuda", "cpu"] = "cuda"
    GDINO_MAX_SIDE: int = Field(1333, ge=64)
    SEGMENT_PROMPT: str = "barcode"
    GDINO_BOX_THRESHOLD: float = Field(0.30, ge=0, le=1)
    GDINO_TEXT_THRESHOLD: float = Field(0.25, ge=0, le=1)
    CONTAINMENT_THRESHOLD: float = Field(0.80, gt=0, le=1)
    NMS_IOU_THRESHOLD: float = Field(0.50, gt=0, le=1)
    SEGMENT_PAD_X: float = Field(0.25, ge=0, le=2)
    SEGMENT_PAD_Y: float = Field(0.20, ge=0, le=2)
    SEGMENT_MAX_COUNT: int = Field(20, ge=1)
    MIN_CROP_SIDE_PX: int = Field(4, ge=1)
    MAX_IMAGE_BYTES: int = Field(30 * 1024 * 1024, ge=1024)

    # --- Parallel code reader ---
    SCANNER_THREADS: int = Field(4, ge=1)

    # --- Background dataset collector & Gemma verifier ---
    DATASET_DIR: str = "./dataset"
    DATASET_QUEUE_MAXSIZE: int = Field(32, ge=1)
    DATASET_SHUTDOWN_TIMEOUT_S: float = Field(20.0, gt=0)
    DATASET_JPEG_QUALITY: int = Field(95, ge=1, le=100)
    DATASET_SAVE_EMPTY: bool = True
    GEMMA_API_BASE_URL: str = "http://host.docker.internal:8080/v1"
    GEMMA_API_KEY: str = ""
    GEMMA_MODEL_NAME: str = "gemma-3-27b-it"
    GEMMA_BATCH_MODE: Literal["multi_image", "single_token_concurrent"] = "multi_image"
    GEMMA_MAX_BATCH_SIZE: int = Field(8, ge=1)
    GEMMA_TIMEOUT_S: float = Field(10.0, gt=0)
    GEMMA_MAX_CONCURRENCY: int = Field(8, ge=1)
    GEMMA_CROP_MAX_SIDE: int = Field(896, ge=32)
    GEMMA_USE_SYSTEM_ROLE: bool = False
    GEMMA_RETRIES: int = Field(1, ge=0)
    GEMMA_CIRCUIT_COOLDOWN_S: float = Field(60.0, ge=0)

    # --- OCR enrichment ---
    OCR_BACKEND: Literal["rapidocr", "paddleocr"] = "rapidocr"

    # --- VLM agent enrichment (empty -> reuse GEMMA_*) ---
    AGENT_API_BASE_URL: str = ""
    AGENT_API_KEY: str = ""
    AGENT_MODEL_NAME: str = ""
    AGENT_TIMEOUT_S: float = Field(30.0, gt=0)
    AGENT_MAX_CONCURRENCY: int = Field(1, ge=1)

    # --- Server ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    def worker_config(self) -> WorkerConfig:
        return WorkerConfig(
            dataset_dir=self.DATASET_DIR,
            jpeg_quality=self.DATASET_JPEG_QUALITY,
            save_empty=self.DATASET_SAVE_EMPTY,
            log_level=self.LOG_LEVEL,
            max_image_bytes=self.MAX_IMAGE_BYTES,
            gemma_base_url=self.GEMMA_API_BASE_URL,
            gemma_api_key=self.GEMMA_API_KEY,
            gemma_model=self.GEMMA_MODEL_NAME,
            gemma_mode=self.GEMMA_BATCH_MODE,
            gemma_max_batch=self.GEMMA_MAX_BATCH_SIZE,
            gemma_timeout_s=self.GEMMA_TIMEOUT_S,
            gemma_max_concurrency=self.GEMMA_MAX_CONCURRENCY,
            gemma_crop_max_side=self.GEMMA_CROP_MAX_SIDE,
            gemma_use_system_role=self.GEMMA_USE_SYSTEM_ROLE,
            gemma_retries=self.GEMMA_RETRIES,
            gemma_cooldown_s=self.GEMMA_CIRCUIT_COOLDOWN_S,
        )

    @property
    def agent_base_url(self) -> str:
        return self.AGENT_API_BASE_URL or self.GEMMA_API_BASE_URL

    @property
    def agent_api_key(self) -> str:
        return self.AGENT_API_KEY or self.GEMMA_API_KEY

    @property
    def agent_model(self) -> str:
        return self.AGENT_MODEL_NAME or self.GEMMA_MODEL_NAME


@lru_cache
def get_settings() -> Settings:
    return Settings()
