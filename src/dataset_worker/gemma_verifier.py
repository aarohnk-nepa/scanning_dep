"""Batched binary barcode verification of segment crops with Gemma (OpenAI-compatible API).

Modes:
  multi_image             - up to GEMMA_MAX_BATCH_SIZE crops in ONE request labelled
                            "image 1".."image K"; answer "image i: true|false" per line.
                            temperature=0, max_tokens=8*K.
  single_token_concurrent - one request per crop, fired together with asyncio.gather;
                            answer exactly "true" / "false"; max_tokens=2.

Every verdict is True / False / None. None means "unknown" (unreachable, timed out, or an
unparseable answer) and is never coerced to False. Failures never raise out of `verify`.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Sequence

import httpx
import numpy as np

from src.utils.image_io import to_data_url

logger = logging.getLogger(__name__)

MULTI_IMAGE_PROMPT = (
    "You are a binary barcode segmentation verifier. You are given {n} cropped images labeled "
    "image 1 through image {n}. For each crop, determine whether it contains a real 1D barcode, "
    "QR code, DataMatrix, or 2D machine-readable code (even if blurry, angled, or low contrast). "
    "Respond ONLY with one line per image in the exact format `image 1: true` or "
    "`image 1: false`. Output no other words, punctuation, or explanation."
)

SINGLE_TOKEN_PROMPT = (
    "You are a binary barcode segmentation verifier. Does this cropped image contain a real 1D "
    "barcode, QR code, DataMatrix, or 2D machine-readable code (even if blurry, angled, or low "
    "contrast)? Answer with exactly one word: true or false."
)

_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}
_SPECIAL_TOKENS = re.compile(r"<[^<>\s]{1,32}>")
_LINE_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?image\s*#?\s*(\d+)\s*[:=\-–)\]]?\s*(true|false|yes|no)\b",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = _SPECIAL_TOKENS.sub(" ", text or "")
    return text.replace("```", "\n").replace("*", "").replace("`", "").replace("_", " ")


def _token_bool(token: str) -> bool | None:
    t = token.strip().strip(".,;:!\"'()[]{}").lower()
    if t in _TRUE:
        return True
    if t in _FALSE:
        return False
    return None


def parse_single_token_response(text: str) -> bool | None:
    words = _clean(text).split()
    return _token_bool(words[0]) if words else None


def parse_multi_image_response(text: str, n: int) -> list[bool | None]:
    """'image 1: true\\nimage 2: false' -> [True, False]. Missing/conflicting -> None."""
    cleaned = _clean(text)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    found: dict[int, bool | None] = {}
    for ln in lines:
        m = _LINE_RE.match(ln)
        if not m:
            continue
        idx = int(m.group(1))
        if not 1 <= idx <= n:
            continue
        val = _token_bool(m.group(2))
        if idx in found and found[idx] != val:
            found[idx] = None  # contradictory answers for the same image
        elif idx not in found:
            found[idx] = val
    if found:
        return [found.get(i) for i in range(1, n + 1)]
    # Positional fallback: exactly n bare true/false tokens, one per line.
    if len(lines) == n:
        vals = [_token_bool(ln) for ln in lines]
        if all(v is not None for v in vals):
            return vals
    return [None] * n


def _with_prompt(prompt: str, content: list[dict], use_system_role: bool) -> list[dict]:
    if use_system_role:
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]
    # Some Gemma endpoints reject the system role; put the instruction first in the user turn.
    return [{"role": "user", "content": [{"type": "text", "text": prompt}, *content]}]


def build_multi_image_messages(data_urls: Sequence[str], *, use_system_role: bool) -> list[dict]:
    n = len(data_urls)
    content: list[dict] = []
    for i, url in enumerate(data_urls, start=1):
        content.append({"type": "text", "text": f"image {i}:"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append(
        {"type": "text", "text": f"Answer now for image 1 through image {n}, one line each."}
    )
    return _with_prompt(MULTI_IMAGE_PROMPT.format(n=n), content, use_system_role)


def build_single_token_messages(data_url: str, *, use_system_role: bool) -> list[dict]:
    content = [{"type": "image_url", "image_url": {"url": data_url}}]
    return _with_prompt(SINGLE_TOKEN_PROMPT, content, use_system_role)


class GemmaUnavailable(Exception):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def outage(self) -> bool:
        return self.status is None or self.status >= 500


class GemmaVerifier:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str,
        mode: str = "multi_image",
        max_batch: int = 8,
        timeout_s: float = 10.0,
        max_concurrency: int = 8,
        crop_max_side: int = 896,
        use_system_role: bool = False,
        retries: int = 1,
        cooldown_s: float = 60.0,
        outage_trip: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if mode not in ("multi_image", "single_token_concurrent"):
            raise ValueError(f"unknown GEMMA_BATCH_MODE {mode!r}")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.mode = mode
        self.max_batch = max(1, max_batch)
        self.crop_max_side = crop_max_side
        self.use_system_role = use_system_role
        self.retries = retries
        self.cooldown_s = cooldown_s
        self.outage_trip = outage_trip
        self._sem = asyncio.Semaphore(max(1, max_concurrency))
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout_s, headers=headers)
        if client is not None and api_key:
            self._client.headers.update(headers)
        self._consecutive_outages = 0
        self._open_until = 0.0

    @property
    def circuit_open(self) -> bool:
        return time.monotonic() < self._open_until

    async def aclose(self) -> None:
        await self._client.aclose()

    async def verify(self, crops: Sequence[np.ndarray]) -> list[bool | None]:
        if not crops:
            return []
        if self.circuit_open:
            return [None] * len(crops)
        try:
            urls = [to_data_url(c, self.crop_max_side, 90) for c in crops]
        except Exception as exc:
            logger.warning("could not encode crops for Gemma (%s)", type(exc).__name__)
            return [None] * len(crops)
        if self.mode == "multi_image":
            out: list[bool | None] = []
            for start in range(0, len(urls), self.max_batch):
                out.extend(await self._verify_chunk(urls[start : start + self.max_batch]))
            return out
        return list(await asyncio.gather(*(self._verify_one(u) for u in urls)))

    async def _verify_chunk(self, urls: list[str]) -> list[bool | None]:
        k = len(urls)
        msgs = build_multi_image_messages(urls, use_system_role=self.use_system_role)
        text = await self._safe_chat(msgs, max_tokens=8 * k)
        return [None] * k if text is None else parse_multi_image_response(text, k)

    async def _verify_one(self, url: str) -> bool | None:
        msgs = build_single_token_messages(url, use_system_role=self.use_system_role)
        text = await self._safe_chat(msgs, max_tokens=2)
        return None if text is None else parse_single_token_response(text)

    async def _safe_chat(self, messages: list[dict], max_tokens: int) -> str | None:
        if self.circuit_open:
            return None
        async with self._sem:
            for attempt in range(self.retries + 1):
                try:
                    text = await self._chat(messages, max_tokens)
                    self._consecutive_outages = 0
                    return text
                except GemmaUnavailable as exc:
                    if not exc.outage:
                        logger.warning("Gemma refused a verification request: %s", exc)
                        return None
                    if attempt < self.retries:
                        await asyncio.sleep(0.5)
                        continue
                    logger.warning("Gemma unavailable: %s", exc)
                    self._consecutive_outages += 1
                    if self._consecutive_outages >= self.outage_trip:
                        self._open_until = time.monotonic() + self.cooldown_s
                        logger.warning(
                            "Gemma circuit open for %.0fs after %d outages",
                            self.cooldown_s,
                            self._consecutive_outages,
                        )
                    return None
        return None

    async def _chat(self, messages: list[dict], max_tokens: int) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        try:
            resp = await self._client.post(self.url, json=body)
        except Exception as exc:
            raise GemmaUnavailable(f"unreachable ({type(exc).__name__})") from None
        if resp.status_code != 200:
            raise GemmaUnavailable(
                f"answered {resp.status_code} ({len(resp.content)} bytes)",
                status=resp.status_code,
            )
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            raise GemmaUnavailable(
                f"unreadable answer ({type(exc).__name__})", status=resp.status_code
            ) from None
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return str(content or "")
