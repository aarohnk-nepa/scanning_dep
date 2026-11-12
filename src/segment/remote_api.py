"""Remote Grounding DINO over HTTP (the legacy hosted API), behind the same Segmentor protocol.

Enabled with USE_SEGMENTOR_API=true + SEGMENTOR_API_URL=<full URL of POST .../detect>.
Contract (see legacy/hosting-grounding-dino.ipynb): multipart `image` + form fields `prompt`,
`box_threshold`, `text_threshold`; detections JSON in the `X-Detections` response header:
[{"label", "confidence", "box": {"x1", "y1", "x2", "y2"}}] in the pixels of the image sent.

The hosted API does not apply EXIF orientation, so when the upload has a non-identity
orientation tag we send a re-encoded upright frame: boxes then line up with our frame.

Errors carry the status and size only, never the response body (it can echo the upload).
"""

from __future__ import annotations

import json

import httpx
import numpy as np

from src.segment.postprocess import Detection, normalize_prompt
from src.utils.image_io import encode_jpeg


class SegmentationAPIError(Exception):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def outage(self) -> bool:
        """The SERVER is down, rather than refusing this one request."""
        return self.status is None or self.status >= 500


_CONTENT_TYPES = {b"\xff\xd8\xff": "image/jpeg", b"\x89PNG": "image/png"}


def _content_type(data: bytes) -> str:
    for magic, ct in _CONTENT_TYPES.items():
        if data.startswith(magic):
            return ct
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


class RemoteSegmentor:
    def __init__(
        self, url: str, *, timeout_s: float = 180.0, client: httpx.Client | None = None
    ) -> None:
        if not url:
            raise ValueError("USE_SEGMENTOR_API=true requires SEGMENTOR_API_URL")
        self.url = url
        self._client = client or httpx.Client(timeout=timeout_s)

    @property
    def ready(self) -> bool:
        return True

    @property
    def device(self) -> str:
        return "remote"

    def load(self) -> None:
        pass

    def warmup(self) -> None:
        pass

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
        if frame_bytes is not None and exif_orientation == 1:
            payload, ctype = frame_bytes, _content_type(frame_bytes)
        else:
            payload, ctype = encode_jpeg(frame_bgr, 95), "image/jpeg"

        try:
            resp = self._client.post(
                self.url,
                files={"image": ("frame.jpg", payload, ctype)},
                data={
                    "prompt": normalize_prompt(prompt),
                    "box_threshold": str(box_threshold),
                    "text_threshold": str(text_threshold),
                },
            )
        except Exception as exc:
            raise SegmentationAPIError(
                f"segmentation API unreachable ({type(exc).__name__})"
            ) from None
        if resp.status_code != 200:
            raise SegmentationAPIError(
                f"segmentation API answered {resp.status_code} ({len(resp.content)} bytes)",
                status=resp.status_code,
            )
        try:
            raw = json.loads(resp.headers["X-Detections"])
            if not isinstance(raw, list):
                raise TypeError("X-Detections is not a list")
        except Exception as exc:
            raise SegmentationAPIError(
                f"segmentation API answered unreadably ({type(exc).__name__})",
                status=resp.status_code,
            ) from None

        dets: list[Detection] = []
        for d in raw:
            try:
                b = d["box"]
                dets.append(
                    Detection(
                        float(b["x1"]),
                        float(b["y1"]),
                        float(b["x2"]),
                        float(b["y2"]),
                        float(d.get("confidence", 0.0)),
                        str(d.get("label", "")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue  # one malformed detection must not sink the rest
        return dets

    def close(self) -> None:
        self._client.close()
