"""Image bytes <-> upright BGR ndarray, once, at the ingestion boundary.

Depends only on numpy, OpenCV and Pillow so the dataset worker can import it without torch.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

_EXIF_ORIENTATION_TAG = 0x0112


class ImageDecodeError(ValueError):
    """The bytes are not a decodable JPEG/PNG/WebP image. Message is safe to return."""


class ImageTooLargeError(ValueError):
    """The upload exceeds MAX_IMAGE_BYTES."""


@dataclass(frozen=True)
class DecodedImage:
    bgr: np.ndarray  # upright, C-contiguous, read-only
    width: int
    height: int
    format: str  # "jpeg" | "png" | "webp"
    exif_orientation: int


def decode_base64_image(value: str) -> bytes:
    """base64 (optionally a `data:` URL) -> raw bytes."""
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    value = "".join(value.split())
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ImageDecodeError("image_base64 is not valid base64") from None


def sniff_image_format(data: bytes) -> str:
    """Format from the magic bytes, never from what the caller claims."""
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise ImageDecodeError("unsupported image format (expected JPEG, PNG or WebP)")


def read_exif_orientation(data: bytes) -> int:
    """EXIF orientation tag (1-8); 1 when absent or unreadable. Reads the header only."""
    try:
        with Image.open(BytesIO(data)) as im:
            value = int(im.getexif().get(_EXIF_ORIENTATION_TAG, 1))
    except Exception:
        return 1
    return value if 1 <= value <= 8 else 1


def apply_exif_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
    """Rotate/flip a stored-orientation image into its upright display orientation."""
    if orientation == 2:
        return cv2.flip(img, 1)
    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(img, 0)
    if orientation == 5:
        return cv2.transpose(img)
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(img), -1)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def load_image(data: bytes, max_bytes: int | None = None) -> DecodedImage:
    """Raw upload bytes -> upright BGR frame. Orientation is applied explicitly, not by OpenCV."""
    if not data:
        raise ImageDecodeError("empty image")
    if max_bytes is not None and len(data) > max_bytes:
        raise ImageTooLargeError(f"image is {len(data)} bytes; limit is {max_bytes}")
    fmt = sniff_image_format(data)
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if img is None:
        raise ImageDecodeError("could not decode the image")
    orientation = read_exif_orientation(data) if fmt in ("jpeg", "webp") else 1
    img = np.ascontiguousarray(apply_exif_orientation(img, orientation))
    # The frame is shared with background consumers; nobody may mutate it in place.
    img.flags.writeable = False
    h, w = img.shape[:2]
    return DecodedImage(bgr=img, width=w, height=h, format=fmt, exif_orientation=orientation)


def cap_long_side(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale (never upscale) so the long side is <= max_side."""
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img
    scale = max_side / long_side
    return cv2.resize(
        img,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def encode_jpeg(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def to_data_url(img: np.ndarray, max_side: int | None = None, quality: int = 90) -> str:
    """In-memory crop -> `data:image/jpeg;base64,...` for an outbound VLM request."""
    if max_side is not None:
        img = cap_long_side(img, max_side)
    b64 = base64.b64encode(encode_jpeg(img, quality)).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
