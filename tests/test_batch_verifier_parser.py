import base64
import json
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from src.dataset_worker.gemma_verifier import (
    GemmaVerifier,
    build_multi_image_messages,
    parse_multi_image_response,
    parse_single_token_response,
)
from src.dataset_worker.storage import DatasetStorage, SegmentVerdict, final_label, yolo_line

# --------------------------------------------------------------------------- parsers


@pytest.mark.parametrize(
    "text,n,expected",
    [
        ("image 1: true\nimage 2: false", 2, [True, False]),
        ("Image 1: TRUE\nImage 2: False\n", 2, [True, False]),
        ("**image 1:** true\n- image 2: false", 2, [True, False]),
        ("image 2: false\nimage 1: true", 2, [True, False]),
        ("image 1 - yes\nimage 2 = no", 2, [True, False]),
        ("Sure!\nimage 1: true\nimage 3: false\nThanks", 3, [True, None, False]),
        ("image 1: true\nimage 9: false", 2, [True, None]),
        ("image 1: true\nimage 1: false\nimage 2: true", 2, [None, True]),
        ("image 1: true\nimage 1: true", 1, [True]),
        ("```\nimage 1: true\nimage 2: false\n```", 2, [True, False]),
        ("", 2, [None, None]),
        ("true\nfalse", 2, [True, False]),  # positional fallback
        ("true\nfalse\ntrue", 2, [None, None]),
        ("image 1: maybe", 1, [None]),
    ],
)
def test_parse_multi_image(text, n, expected):
    assert parse_multi_image_response(text, n) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("true", True),
        (" True.", True),
        ("FALSE\n", False),
        ("`true`", True),
        ("yes", True),
        ("no", False),
        ("true<end_of_turn>", True),
        ("<start_of_turn>false", False),
        ("maybe", None),
        ("", None),
    ],
)
def test_parse_single_token(text, expected):
    assert parse_single_token_response(text) is expected


def test_multi_image_payload_shape():
    urls = [f"data:image/jpeg;base64,AAA{i}" for i in range(3)]
    msgs = build_multi_image_messages(urls, use_system_role=False)
    assert [m["role"] for m in msgs] == ["user"]
    parts = msgs[0]["content"]
    assert "image 1 through image 3" in parts[0]["text"]
    assert [p["text"] for p in parts if p["type"] == "text"][1:4] == [
        "image 1:",
        "image 2:",
        "image 3:",
    ]
    assert [p["image_url"]["url"] for p in parts if p["type"] == "image_url"] == urls
    assert build_multi_image_messages(urls, use_system_role=True)[0]["role"] == "system"


# --------------------------------------------------------------------------- HTTP (mocked)


def _crop(value: int) -> np.ndarray:
    return np.full((40, 60, 3), value, np.uint8)


def _decode_mean(url: str) -> float:
    raw = base64.b64decode(url.split(",", 1)[1])
    return float(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE).mean())


def _reply(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def _verifier(handler, **kw) -> GemmaVerifier:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    defaults = dict(base_url="http://gemma/v1", model="gemma-test", retries=0, client=client)
    return GemmaVerifier(**{**defaults, **kw})


async def test_multi_image_batches_and_stitches_indices():
    bodies = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        bodies.append(body)
        urls = [
            p["image_url"]["url"]
            for p in body["messages"][0]["content"]
            if p["type"] == "image_url"
        ]
        # dark crop -> true, light crop -> false
        lines = [
            f"image {i}: {'true' if _decode_mean(u) < 128 else 'false'}"
            for i, u in enumerate(urls, 1)
        ]
        return _reply("\n".join(lines))

    crops = [_crop(0 if i % 3 == 0 else 255) for i in range(10)]
    v = _verifier(handler, mode="multi_image", max_batch=8)
    out = await v.verify(crops)
    assert out == [i % 3 == 0 for i in range(10)]
    assert len(bodies) == 2
    assert [b["max_tokens"] for b in bodies] == [64, 16]
    assert all(b["temperature"] == 0.0 and b["model"] == "gemma-test" for b in bodies)
    await v.aclose()


async def test_single_token_concurrent_maps_back_by_crop():
    seen_max_tokens = []

    def handler(req):
        body = json.loads(req.content)
        seen_max_tokens.append(body["max_tokens"])
        url = next(
            p["image_url"]["url"]
            for p in body["messages"][0]["content"]
            if p["type"] == "image_url"
        )
        return _reply("true" if _decode_mean(url) < 128 else "false")

    v = _verifier(handler, mode="single_token_concurrent")
    out = await v.verify([_crop(255), _crop(0), _crop(255), _crop(0)])
    assert out == [False, True, False, True]
    assert seen_max_tokens == [2, 2, 2, 2]
    await v.aclose()


@pytest.mark.parametrize("failure", ["connect", "500", "timeout"])
async def test_outage_returns_unknown_and_opens_circuit(failure):
    calls = []

    def handler(req):
        calls.append(1)
        if failure == "connect":
            raise httpx.ConnectError("down")
        if failure == "timeout":
            raise httpx.ReadTimeout("slow")
        return httpx.Response(500, text="boom")

    v = _verifier(handler, mode="multi_image", outage_trip=3, cooldown_s=60)
    for _ in range(3):
        assert await v.verify([_crop(0), _crop(255)]) == [None, None]
    n = len(calls)
    assert v.circuit_open
    assert await v.verify([_crop(0)]) == [None]
    assert len(calls) == n  # circuit open: no HTTP call
    await v.aclose()


async def test_4xx_is_not_an_outage():
    v = _verifier(lambda r: httpx.Response(400, text="bad"), mode="multi_image")
    for _ in range(5):
        assert await v.verify([_crop(0)]) == [None]
    assert not v.circuit_open
    await v.aclose()


# --------------------------------------------------------------------------- label rule + storage


def test_final_label_rule():
    assert final_label(True, False) == "positive"
    assert final_label(False, True) == "positive"  # scanner success overrides Gemma
    assert final_label(None, True) == "positive"
    assert final_label(False, False) == "negative"
    assert final_label(None, False) == "unverified"


def test_yolo_line_normalisation():
    assert yolo_line((100, 100, 300, 200), 1000, 500) == "0 0.200000 0.300000 0.200000 0.200000"


def _verdict(i, gemma, decoded):
    x = 50 + 200 * i
    return SegmentVerdict(
        segment_index=i,
        score=0.9,
        label="barcode",
        raw_bbox=(x, 100, x + 100, 200),
        padded_bbox=(x - 25, 80, x + 125, 220),
        scanner_decoded=decoded,
        gemma_verified=gemma,
        barcodes=[{"symbology": "QRCODE", "value": "X"}] if decoded else [],
    )


def test_storage_writes_yolo_crops_and_metadata(tmp_path: Path):
    st = DatasetStorage(tmp_path, 90)
    st.ensure_layout()
    frame = np.full((500, 1000, 3), 200, np.uint8)
    verdicts = [
        _verdict(0, True, False),
        _verdict(1, False, True),
        _verdict(2, False, False),
        _verdict(3, None, False),
    ]
    meta_path = st.write_sample(
        request_id="req1",
        timestamp="t",
        frame=frame,
        verdicts=verdicts,
        provenance={"prompt": "barcode."},
    )

    labels = (tmp_path / "labels" / "req1.txt").read_text().splitlines()
    assert labels == [
        yolo_line(verdicts[0].raw_bbox, 1000, 500),
        yolo_line(verdicts[1].raw_bbox, 1000, 500),
    ]
    assert sorted(p.name for p in (tmp_path / "crops/positive").iterdir()) == [
        "req1_seg0.jpg",
        "req1_seg1.jpg",
    ]
    assert [p.name for p in (tmp_path / "crops/negative").iterdir()] == ["req1_seg2.jpg"]
    assert [p.name for p in (tmp_path / "crops/unverified").iterdir()] == ["req1_seg3.jpg"]
    assert (tmp_path / "images" / "req1.jpg").exists()

    meta = json.loads(meta_path.read_text())
    assert meta["image_width"] == 1000 and meta["image_height"] == 500
    assert meta["label_complete"] is False
    seg1 = meta["segments"][1]
    assert seg1["scanner_decoded"] is True and seg1["gemma_verified"] is False
    assert seg1["final_label"] == "positive" and seg1["padded_size_px"] == [150, 140]
    assert "gdino_score" in seg1
    assert not list(tmp_path.rglob("*.tmp"))


def test_storage_background_sample_gets_empty_label(tmp_path: Path):
    st = DatasetStorage(tmp_path)
    st.ensure_layout()
    st.write_sample(
        request_id="bg",
        timestamp="t",
        frame=np.zeros((10, 10, 3), np.uint8),
        verdicts=[_verdict(0, False, False)],
    )
    assert (tmp_path / "labels" / "bg.txt").read_text() == ""
    meta = json.loads((tmp_path / "metadata" / "bg.json").read_text())
    assert meta["label_complete"] is True


def test_worker_process_job_end_to_end(tmp_path: Path):
    """process_job: bytes -> decode -> Gemma (mocked) -> label = gemma OR scanner."""
    import asyncio

    from src.dataset_worker.worker import DatasetJob, SegmentPayload, process_job
    from tests.conftest import jpeg_bytes

    frame = np.full((300, 600, 3), 255, np.uint8)
    frame[50:150, 50:150] = 0  # dark -> Gemma true
    job = DatasetJob(
        request_id="job1",
        timestamp="t",
        frame_bytes=jpeg_bytes(frame),
        frame_shape=(300, 600),
        segments_payload=[
            SegmentPayload(0, 0.9, "barcode", (50, 50, 150, 150), (40, 40, 160, 160), False),
            SegmentPayload(1, 0.8, "barcode", (300, 50, 400, 150), (290, 40, 410, 160), True),
            SegmentPayload(2, 0.7, "barcode", (450, 50, 550, 150), (440, 40, 560, 160), False),
        ],
    )

    def handler(req):
        body = json.loads(req.content)
        urls = [
            p["image_url"]["url"]
            for p in body["messages"][0]["content"]
            if p["type"] == "image_url"
        ]
        return _reply(
            "\n".join(
                f"image {i}: {'true' if _decode_mean(u) < 128 else 'false'}"
                for i, u in enumerate(urls, 1)
            )
        )

    loop = asyncio.new_event_loop()
    try:
        v = _verifier(handler)
        st = DatasetStorage(tmp_path)
        st.ensure_layout()
        assert process_job(job, st, v, loop)
        loop.run_until_complete(v.aclose())
    finally:
        loop.close()
    lines = (tmp_path / "labels" / "job1.txt").read_text().splitlines()
    assert len(lines) == 2  # seg0 (gemma) + seg1 (scanner); seg2 negative
    meta = json.loads((tmp_path / "metadata" / "job1.json").read_text())
    assert [s["final_label"] for s in meta["segments"]] == ["positive", "positive", "negative"]
    assert meta["provenance"]["gemma_mode"] == "multi_image"
