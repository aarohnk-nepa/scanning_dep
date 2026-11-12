import base64
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.config import Settings
from src.main import create_app
from src.schemas import OcrResult
from src.segment.postprocess import Detection
from tests.conftest import (
    FakeDispatcher,
    FakeSegmentor,
    canvas_with_codes,
    make_code,
    png_bytes,
)

QR_TEXT = "PALLET-0001"
QR = make_code(QR_TEXT, "QRCode", 4)
QX, QY = 300, 250
QH, QW = QR.shape
FRAME = canvas_with_codes([(QR, QX, QY)])
IMG = png_bytes(FRAME)  # lossless, so the QR decodes deterministically
OUTER = Detection(QX - 100, QY - 100, QX + QW + 300, QY + QH + 150, 0.95, "barcode")
INNER = Detection(QX + 8, QY + 8, QX + QW - 8, QY + QH - 8, 0.60, "barcode")


def settings(**kw) -> Settings:
    base = dict(
        ENABLE_DATA_COLLECTION=True,
        ENABLE_OCR=False,
        ENABLE_AGENT=False,
        MAX_IMAGE_BYTES=5 * 1024 * 1024,
    )
    return Settings(_env_file=None, **{**base, **kw})


@pytest.fixture
def ctx():
    seg = FakeSegmentor([OUTER, INNER])
    disp = FakeDispatcher()
    app = create_app(settings(), segmentor=seg, dispatcher=disp)
    with TestClient(app) as client:
        yield client, seg, disp


def assert_scan_ok(r, disp=None):
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["segments"]) == 1, "outer box must be suppressed"
    seg = body["segments"][0]
    assert seg["raw_bbox"]["x1"] == INNER.x1
    assert [b["value"] for b in body["barcodes"]] == [QR_TEXT]
    bc = body["barcodes"][0]
    assert bc["symbology"] == "QRCODE" and bc["segment_index"] == 0
    pb = seg["padded_bbox"]
    assert pb["x1"] <= bc["bbox_global"]["x1"] and bc["bbox_global"]["x2"] <= pb["x2"]
    assert set(body["timing_ms"]) == {
        "decode_image_ms",
        "segment_ms",
        "code_reader_ms",
        "enrich_ms",
        "total_ms",
    }
    assert all(v >= 0 for v in body["timing_ms"].values())
    if disp is not None:
        assert body["dataset_queued"] is True
        job = disp.jobs[-1]
        assert job.request_id == body["request_id"]
        assert job.frame_shape == FRAME.shape[:2]
        assert job.segments_payload[0].scanner_decoded is True
        assert job.segments_payload[0].barcodes[0]["value"] == QR_TEXT
    return body


def test_post_multipart(ctx):
    client, seg, disp = ctx
    r = client.post("/api/v1/scan", files={"file": ("x.png", IMG, "image/png")})
    assert_scan_ok(r, disp)
    assert seg.calls[-1]["prompt"] == "barcode."


def test_post_json_base64_with_overrides(ctx):
    client, seg, disp = ctx
    b64 = "data:image/png;base64," + base64.b64encode(IMG).decode()
    r = client.post(
        "/api/v1/scan",
        json={"image_base64": b64, "prompt": "QR Code", "padding": 0.5, "box_threshold": 0.4},
    )
    body = assert_scan_ok(r, disp)
    assert seg.calls[-1]["prompt"] == "qr code."
    assert seg.calls[-1]["box_threshold"] == 0.4
    w = INNER.x2 - INNER.x1
    assert body["segments"][0]["padded_bbox"]["x1"] == pytest.approx(INNER.x1 - 0.5 * w, abs=1)


def test_get_with_multipart_json_query_and_raw(ctx):
    client, _, disp = ctx
    assert_scan_ok(
        client.request("GET", "/api/v1/scan", files={"file": ("x.png", IMG, "image/png")}), disp
    )
    assert_scan_ok(
        client.request(
            "GET", "/api/v1/scan", json={"image_base64": base64.b64encode(IMG).decode()}
        ),
        disp,
    )
    assert_scan_ok(
        client.request("GET", "/api/v1/scan", content=IMG, headers={"Content-Type": "image/png"}),
        disp,
    )
    small = png_bytes(FRAME[QY - 60 : QY + QH + 60, QX - 60 : QX + QW + 60])
    seg_small = Detection(68, 68, 60 + QW - 8, 60 + QH - 8, 0.9)
    client.app.state.pipeline.segmentor.detections = [seg_small]
    r4 = client.get("/api/v1/scan", params={"image_base64": base64.b64encode(small).decode()})
    assert r4.status_code == 200 and [b["value"] for b in r4.json()["barcodes"]] == [QR_TEXT]


def test_error_codes(ctx):
    client, _, _ = ctx
    assert (
        client.post("/api/v1/scan", files={"file": ("x.jpg", b"garbage", "image/jpeg")}).status_code
        == 400
    )
    assert (
        client.post(
            "/api/v1/scan", content=b"hello", headers={"Content-Type": "text/plain"}
        ).status_code
        == 415
    )
    assert client.post("/api/v1/scan", data={"prompt": "x"}).status_code == 422
    assert client.post("/api/v1/scan", files={"other": ("a", b"x")}).status_code == 422
    assert client.post("/api/v1/scan", json={"image_base64": "!!notbase64!!"}).status_code == 400
    big = b"\xff\xd8\xff" + b"0" * (8 * 1024 * 1024)
    assert (
        client.post("/api/v1/scan", content=big, headers={"Content-Type": "image/jpeg"}).status_code
        == 413
    )


def test_queue_full_never_blocks_response():
    disp = FakeDispatcher(full=True)
    app = create_app(settings(), segmentor=FakeSegmentor([INNER]), dispatcher=disp)
    with TestClient(app) as client:
        r = client.post("/api/v1/scan", files={"file": ("x.png", IMG, "image/png")})
    assert r.status_code == 200 and r.json()["dataset_queued"] is False


def test_data_collection_disabled():
    disp = FakeDispatcher()
    app = create_app(
        settings(ENABLE_DATA_COLLECTION=False), segmentor=FakeSegmentor([INNER]), dispatcher=disp
    )
    with TestClient(app) as client:
        r = client.post("/api/v1/scan", files={"file": ("x.png", IMG, "image/png")})
    assert r.json()["dataset_queued"] is False and disp.jobs == []


def test_exif_orientation_normalised(ctx):
    client, seg, _ = ctx
    stored = np.zeros((100, 200, 3), np.uint8)  # stored landscape
    stored[:, :20] = 255  # bright stripe on the stored LEFT edge
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW for display
    buf = io.BytesIO()
    Image.fromarray(stored).save(buf, format="JPEG", exif=exif, quality=95)
    r = client.post("/api/v1/scan", files={"file": ("r.jpg", buf.getvalue(), "image/jpeg")})
    body = r.json()
    assert (body["image_width"], body["image_height"]) == (100, 200)
    assert seg.calls[-1]["shape"][:2] == (200, 100)

    from src.utils.image_io import load_image

    upright = load_image(buf.getvalue()).bgr
    ref = np.asarray(Image.open(io.BytesIO(buf.getvalue())).transpose(Image.Transpose.ROTATE_270))
    assert upright.shape[:2] == ref.shape[:2]
    assert upright[:20].mean() > 200  # stored-left stripe is now on top


@pytest.mark.parametrize("orientation", range(1, 9))
def test_apply_exif_orientation_matches_pil(orientation):
    from PIL import ImageOps

    from src.utils.image_io import apply_exif_orientation

    rng = np.random.default_rng(orientation)
    stored = rng.integers(0, 255, (6, 9, 3), dtype=np.uint8)
    im = Image.fromarray(stored)
    exif = im.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    im.save(buf, format="PNG", exif=exif)
    expected = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(buf.getvalue()))))
    np.testing.assert_array_equal(apply_exif_orientation(stored, orientation), expected)


class FakeOcr:
    def recognize(self, crop):
        return OcrResult(text=f"{crop.shape[1]}x{crop.shape[0]}", lines=[])


def _agent_handler(conf):
    def handler(req):
        reading = {
            "raw_text": "PALLET-0001",
            "fields": {"Lot": None},
            "has_machine_code": True,
            "type": "product_label",
            "summary": "a label",
            "confidence": conf,
            "needs_verification": False,
        }
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(reading)}}]}
        )

    return handler


def test_ocr_and_agent_enrichment_on_barcode_crops():
    from src.enrich.vlm_agent import VlmAgent

    agent = VlmAgent(
        base_url="http://gemma/v1",
        model="m",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_agent_handler(0.6))),
    )
    app = create_app(
        settings(ENABLE_OCR=True, ENABLE_AGENT=True),
        segmentor=FakeSegmentor([INNER]),
        dispatcher=FakeDispatcher(),
        ocr=FakeOcr(),
        agent=agent,
    )
    with TestClient(app) as client:
        r = client.post("/api/v1/scan", files={"file": ("x.png", IMG, "image/png")})
    seg = r.json()["segments"][0]
    assert seg["source"] == "barcode"
    assert seg["ocr_result"]["text"]
    assert seg["agent_result"]["type"] == "product_label"
    assert seg["agent_result"]["needs_verification"] is True  # 0.6 < 0.70, applied mechanically


def test_enrich_prompt_runs_second_detection():
    label_box = Detection(100, 100, 900, 700, 0.8, "shipping label")
    seg = FakeSegmentor([INNER], by_prompt={"shipping label.": [label_box]})
    app = create_app(
        settings(ENABLE_OCR=True), segmentor=seg, dispatcher=FakeDispatcher(), ocr=FakeOcr()
    )
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/scan",
            files={"file": ("x.png", IMG, "image/png")},
            data={"enrich_prompt": "Shipping Label"},
        )
    body = r.json()
    assert [c["prompt"] for c in seg.calls] == ["barcode.", "shipping label."]
    sources = [(s["source"], s["segment_index"]) for s in body["segments"]]
    assert sources == [("barcode", 0), ("enrich", 1)]
    assert body["segments"][0]["ocr_result"] is None
    assert body["segments"][1]["ocr_result"]["text"]
    assert [b["value"] for b in body["barcodes"]] == [QR_TEXT]


def test_remote_segmentor_contract():
    from src.segment.remote_api import RemoteSegmentor, SegmentationAPIError

    seen = {}

    def handler(req: httpx.Request):
        seen["body"] = req.content
        dets = [
            {
                "label": "barcode",
                "confidence": 0.77,
                "box": {"x1": 1.5, "y1": 2, "x2": 30, "y2": 40},
            },
            {"bad": 1},
        ]
        return httpx.Response(200, content=b"jpeg", headers={"X-Detections": json.dumps(dets)})

    rs = RemoteSegmentor(
        "http://dino/api/v1/detect", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    out = rs.detect(
        FRAME, prompt="Barcode", box_threshold=0.3, text_threshold=0.25, frame_bytes=IMG
    )
    assert out == [Detection(1.5, 2.0, 30.0, 40.0, 0.77, "barcode")]
    assert b'name="prompt"' in seen["body"] and b"barcode." in seen["body"]

    down = RemoteSegmentor(
        "http://dino",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))),
    )
    with pytest.raises(SegmentationAPIError) as ei:
        down.detect(FRAME, prompt="barcode", box_threshold=0.3, text_threshold=0.25)
    assert ei.value.outage


def test_remote_segmentor_outage_maps_to_503():
    from src.segment.remote_api import RemoteSegmentor

    rs = RemoteSegmentor(
        "http://dino",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(502))),
    )
    app = create_app(settings(), segmentor=rs, dispatcher=FakeDispatcher())
    with TestClient(app) as client:
        r = client.post("/api/v1/scan", files={"file": ("x.png", IMG, "image/png")})
    assert r.status_code == 503


def test_dataset_worker_never_imports_torch():
    code = (
        "import sys, src.dataset_worker.worker, src.dataset_worker.gemma_verifier; "
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    )
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root)


@pytest.mark.slow
def test_real_worker_process_end_to_end(tmp_path):
    """Full app with the real spawned DatasetWorkerProcess (Gemma disabled)."""
    s = settings(DATASET_DIR=str(tmp_path), GEMMA_API_BASE_URL="")
    app = create_app(s, segmentor=FakeSegmentor([OUTER, INNER]))
    with TestClient(app) as client:
        assert client.get("/api/v1/health").json()["dataset_worker_alive"] is True
        r = client.post("/api/v1/scan", files={"file": ("x.png", IMG, "image/png")})
        rid = r.json()["request_id"]
        assert r.json()["dataset_queued"] is True
        meta = tmp_path / "metadata" / f"{rid}.json"
        deadline = time.time() + 30
        while not meta.exists() and time.time() < deadline:
            time.sleep(0.2)
    data = json.loads(meta.read_text())
    assert data["segments"][0]["scanner_decoded"] is True
    assert data["segments"][0]["final_label"] == "positive"
    assert data["segments"][0]["gemma_verified"] is None
    assert (tmp_path / "labels" / f"{rid}.txt").read_text().count("\n") == 1
