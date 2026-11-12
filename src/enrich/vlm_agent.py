"""VLM agent enrichment: full structured sticker reading with Gemma (ENABLE_AGENT=true only).

Ported from legacy/agent.py without the agent framework (BaseAgent, ExecutionContext, events,
tool registry, gateway service). The extraction prompt, the closed sticker-type vocabulary,
the confidence bands and the mechanical `needs_verification` rule are preserved verbatim.

Crops are passed in memory and only encoded to a data URL at the outbound HTTP boundary.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any

import httpx

from src.schemas import AgentResult
from src.utils.image_io import to_data_url

logger = logging.getLogger(__name__)

AGENT_CROP_MAX_SIDE = 1536


# Below this a person re-checks the reading. Applied mechanically in _validate_reading,
# not left to the model: a mechanical rule must not be delegated to a stochastic process.
VERIFY_BELOW_CONFIDENCE = 0.70

STICKER_TYPES = frozenset(
    {"shipping_label", "product_label", "handling", "warning", "other", "unknown"}
)

_REQUIRED_KEYS = (
    "raw_text",
    "fields",
    "has_machine_code",
    "type",
    "summary",
    "confidence",
    "needs_verification",
)

# THE STICKER TYPES AND THE CONFIDENCE BANDS ARE THE PART TO LEAVE ALONE once measured; the
# key order below is load-bearing. The model generates left to right, so `raw_text` comes
# FIRST: the transcription is how it reaches the verdict, and `fields`, `type`, `summary`
# and `confidence` are then read off text that already exists. With the verdict first, a
# type and a confidence would be committed before a character was transcribed.
SYSTEM_PROMPT = """\
You are the sticker-reading step of a Vision Agent that inspects cardboard boxes. An upstream
detector has located a sticker on a box and cropped it. You receive ONE cropped sticker image
per request, labeled "Sticker N:". Read it and return structured context.

SOURCE OF TRUTH
The pixels of the crop are the only source. You are not given the box, the other stickers, the
shipper or any metadata, and you must not infer what a sticker "usually" says. A value you
cannot see is not there.

STICKER TYPES
This list is CLOSED. Choose exactly one.
- shipping_label - carrier, routing, address, tracking or destination information.
- product_label - what is inside: product name, SKU/model, lot or batch, quantity, weight,
  dates, origin.
- handling - handling instructions or pictograms: this side up, fragile, keep dry, stacking.
- warning - hazard, safety, or regulatory notices.
- other - a legible sticker that fits none of the above.
- unknown - the crop is not a sticker, is blank, or is too unreadable to classify.

READING DISCIPLINE (required, and written BEFORE the verdict)
`raw_text` is the FIRST field of the answer because it is how you reach the verdict, not a
summary composed after one. Do not settle on a type before you have written it.
- Transcribe every legible character from top to bottom, left to right. Start a new line for
  each printed line.
- Copy text verbatim: preserve casing, digits, punctuation and units. Only normalise
  whitespace.
- Anything you cannot read - blurred, glare, torn, cut off at the crop edge - is written
  as [illegible]. NEVER complete a partial word or number from what it probably says. A
  confident wrong digit in a tracking number or SKU is worse than a gap.
- Where 0/O, 1/l/I or 5/S cannot be told apart in context, keep what the pixels show and lower
  your confidence.
- Text printed beneath a barcode IS legible text: transcribe it. Do NOT try to decode the
  QR code or barcode itself - a separate scanner tool does that. Only say whether one is
  present.
- `fields` come from `raw_text` only. Use the label as it is PRINTED ("Lot", "Net Wt",
  "SKU"); the value is null when the label is printed but its value is unreadable. If the
  sticker has no labelled values, `fields` is an empty object.
- Report what is in the pixels. A brand, address or number that you can only half read is
  [illegible], never assumed.

CONFIDENCE and REVIEW
- confidence: 0.0-1.0, your certainty that `raw_text` and `fields` are correct and complete.
  Use the whole range. This number decides which readings a person re-checks, so one that is
  always high tells them nothing and spends the review effort at random.
    0.90-1.00  every line is sharp and fully legible; nothing is cut off.
    0.70-0.89  legible, but small, angled, glared, or cut off in one place.
    0.40-0.69  several lines or values are uncertain or [illegible].
    below 0.40 you are inferring more than you can actually see.
- Below 0.70, set needs_verification=true. type=unknown also sets it true.

OUTPUT - respond with ONLY this JSON object, in this exact key order, no other text:
{
  "raw_text": "all legible text, top to bottom, one printed line per line, [illegible] where unreadable",
  "fields": {"<label as printed>": "<value as printed, or null if unreadable>"},
  "has_machine_code": true | false,
  "type": "shipping_label" | "product_label" | "handling" | "warning" | "other" | "unknown",
  "summary": "one short sentence describing what the sticker says",
  "confidence": number between 0.0 and 1.0,
  "needs_verification": boolean
}
"""


def parse_json(text: str) -> Any:
    """Parse LLM output as JSON, tolerating accidental markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def validate_reading(raw: Any) -> dict:
    """
    Check the answer against the closed vocabulary and apply the review rule.

    Raises ValueError with a message that never echoes the model's text, so it is safe to
    log and to return.
    """
    if not isinstance(raw, dict):
        raise ValueError("answer was not a JSON object")
    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"missing keys: {', '.join(missing)}")
    if raw["type"] not in STICKER_TYPES:
        raise ValueError("type outside the closed vocabulary")
    confidence = raw["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError("confidence outside 0.0-1.0")
    if not isinstance(raw["fields"], dict):
        raise ValueError("fields was not an object")

    raw["needs_verification"] = (
        bool(raw["needs_verification"])
        or confidence < VERIFY_BELOW_CONFIDENCE
        or raw["type"] == "unknown"
    )
    return raw


def crop_messages(data_url: str, crop_id: Any, *, use_system_role: bool = False) -> list[dict]:
    """OpenAI-compatible multimodal messages for ONE crop ("Sticker N:" + image data URL).

    One crop per call: a bad read, a timeout or a 400 costs one sticker, not the box.
    """
    user_parts = [
        {"type": "text", "text": "Read the sticker below and answer with the JSON object only."},
        {"type": "text", "text": f"Sticker {crop_id}:"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    if use_system_role:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_parts},
        ]
    return [{"role": "user", "content": [{"type": "text", "text": SYSTEM_PROMPT}, *user_parts]}]


class AgentUnavailable(Exception):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def outage(self) -> bool:
        return self.status is None or self.status >= 500


class VlmAgent:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str,
        timeout_s: float = 30.0,
        max_concurrency: int = 1,
        use_system_role: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.use_system_role = use_system_role
        self._max_concurrency = max(1, max_concurrency)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=timeout_s, headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _chat(self, messages: list[dict]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0,  # transcription: keep it deterministic
        }
        try:
            resp = await self._client.post(self.url, json=body)
        except Exception as exc:
            raise AgentUnavailable(f"unreachable ({type(exc).__name__})") from None
        if resp.status_code != 200:
            raise AgentUnavailable(
                f"answered {resp.status_code} ({len(resp.content)} bytes)", status=resp.status_code
            )
        try:
            return str(resp.json()["choices"][0]["message"]["content"] or "")
        except Exception as exc:
            raise AgentUnavailable(
                f"unreadable answer ({type(exc).__name__})", status=resp.status_code
            ) from None

    async def read_crop(self, crop: Any, crop_id: Any) -> AgentResult:
        """Raises AgentUnavailable (transport) or ValueError (rejected answer)."""
        url = to_data_url(crop, AGENT_CROP_MAX_SIDE, 92)
        content = await self._chat(
            crop_messages(url, crop_id, use_system_role=self.use_system_role)
        )
        try:
            reading = validate_reading(parse_json(content))
        except json.JSONDecodeError:
            raise ValueError("model output was not valid JSON") from None
        except ValueError as exc:
            raise ValueError(f"model output rejected: {exc}") from None
        return AgentResult.model_validate(reading)

    async def enrich(self, segments: Sequence[Any]) -> None:
        """Populate agent_result / agent_error on each segment. Never raises."""
        sem = asyncio.Semaphore(self._max_concurrency)
        outage = asyncio.Event()

        async def one(seg: Any) -> None:
            async with sem:
                if outage.is_set():
                    seg.agent_error = "gemma unavailable"
                    return
                try:
                    seg.agent_result = await self.read_crop(seg.crop, seg.segment_index)
                except AgentUnavailable as exc:
                    logger.warning("agent read failed for segment %s: %s", seg.segment_index, exc)
                    seg.agent_error = str(exc)
                    # No answer / 5xx says everything about the server: skip the rest.
                    if exc.outage:
                        outage.set()
                except ValueError as exc:
                    seg.agent_error = str(exc)
                except Exception as exc:
                    seg.agent_error = type(exc).__name__

        await asyncio.gather(*(one(s) for s in segments))
