import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.decode import code_reader as cr_mod
from src.decode.code_reader import CodeReader, dedupe_barcodes
from src.segment.postprocess import Detection, build_segments
from tests.conftest import canvas_with_codes, make_code, needs_pyzbar

KW = dict(containment_threshold=0.8, nms_iou=0.5, pad_x=0.25, pad_y=0.20)


def _sym(text, x=0, y=0, w=10, h=10, kind="QRCODE"):
    return SimpleNamespace(
        type=kind, data=text.encode(), rect=SimpleNamespace(left=x, top=y, width=w, height=h)
    )


def _segments(n=4, size=(400, 1600)):
    f = np.full((*size, 3), 255, np.uint8)
    dets = [Detection(20 + 380 * i, 100, 300 + 380 * i, 300, 0.9 - i * 0.01) for i in range(n)]
    return build_segments(f, dets, **KW)


@pytest.fixture
def pool():
    with ThreadPoolExecutor(4) as ex:
        yield ex


async def test_segments_decode_in_parallel(pool):
    barrier = threading.Barrier(4, timeout=5)

    def fake_pyzbar(gray):
        barrier.wait()  # only passes if all 4 crops are in flight at once
        return [_sym(f"code-{threading.get_ident()}")]

    reader = CodeReader(pool, pyzbar_decode=fake_pyzbar, zxing_read=None)
    segs = _segments(4)
    await reader.adecode_segments(segs)
    assert all(s.decode_error is None for s in segs)
    assert all(len(s.barcodes) == 1 for s in segs)


def test_no_upscale_retry(pool, monkeypatch):
    assert "resize" not in inspect.getsource(cr_mod)
    assert "INTER_CUBIC" not in inspect.getsource(cr_mod)

    def boom(*a, **k):
        raise AssertionError("cv2.resize must not be called")

    monkeypatch.setattr(cv2, "resize", boom)
    calls = []
    reader = CodeReader(
        pool,
        pyzbar_decode=lambda g: calls.append("z") or [],
        zxing_read=lambda g: calls.append("x") or [],
    )
    segs = _segments(2)
    reader.decode_segments(segs)
    # exactly one pyzbar + one zxing call per crop, no retry pass
    assert calls.count("z") == 2 and calls.count("x") == 2


def test_engine_order_pyzbar_first_zxing_fallback(pool):
    zx_calls = []

    def zx(gray):
        zx_calls.append(1)
        pos = SimpleNamespace(
            top_left=SimpleNamespace(x=1, y=2),
            top_right=SimpleNamespace(x=11, y=2),
            bottom_right=SimpleNamespace(x=11, y=12),
            bottom_left=SimpleNamespace(x=1, y=12),
        )
        return [SimpleNamespace(format=SimpleNamespace(name="ITF"), text="123", position=pos)]

    hit = CodeReader(pool, pyzbar_decode=lambda g: [_sym("A")], zxing_read=zx)
    segs = _segments(1)
    hit.decode_segments(segs)
    codes, _ = hit.decode_crop(segs[0].crop, segs[0].padded_bbox)
    assert codes[0].engine == "pyzbar" and zx_calls == []

    miss = CodeReader(pool, pyzbar_decode=lambda g: [], zxing_read=zx)
    codes, _ = miss.decode_crop(segs[0].crop, segs[0].padded_bbox)
    assert len(zx_calls) == 1
    assert codes[0].engine == "zxing" and codes[0].symbology == "I25"
    px1, py1 = segs[0].padded_bbox[:2]
    assert codes[0].bbox_global == (px1 + 1, py1 + 2, px1 + 11, py1 + 12)


def test_crops_are_c_contiguous_and_errors_isolated(pool):
    def fake(gray):
        assert gray.flags.c_contiguous
        if gray.shape[1] == 999:
            raise RuntimeError("bad crop")
        return [_sym("ok")]

    reader = CodeReader(pool, pyzbar_decode=fake, zxing_read=None)
    segs = _segments(3)
    assert not segs[1].crop.flags.c_contiguous  # ROI views are strided
    segs[1].crop = np.full((50, 999, 3), 255, np.uint8)
    results = reader.decode_segments(segs)
    assert results[1] == ([], "RuntimeError")
    assert results[0][0][0].value == "ok" and results[2][0][0].value == "ok"


@needs_pyzbar
async def test_real_qr_and_code128_with_global_coordinates(pool):
    qr = make_code("QR-PAYLOAD-1", "QRCode", 4)
    c128 = make_code("ABC-12345", "Code128", 3)
    frame = canvas_with_codes([(qr, 100, 120), (c128, 500, 500)])
    qh, qw = qr.shape
    dets = [
        Detection(100, 120, 100 + qw, 120 + qh, 0.9),
        Detection(500, 500, 500 + c128.shape[1], 500 + c128.shape[0], 0.8),
    ]
    segs = build_segments(frame, dets, **KW)
    await CodeReader(pool).adecode_segments(segs)
    by_val = {c.value: c for s in segs for c in s.barcodes}
    assert by_val["QR-PAYLOAD-1"].symbology == "QRCODE"
    assert by_val["QR-PAYLOAD-1"].engine == "pyzbar"
    x1, y1, x2, y2 = by_val["QR-PAYLOAD-1"].bbox_global
    pad = 32  # zxing's own quiet zone (16) + make_code's extra padding (16) at scale 4
    assert abs(x1 - (100 + pad)) <= 3 and abs(y1 - (120 + pad)) <= 3
    assert by_val["ABC-12345"].symbology == "CODE128"


async def test_real_datamatrix_falls_back_to_zxing(pool):
    dm = make_code("DM-42", "DataMatrix", 6)
    frame = canvas_with_codes([(dm, 300, 300)])
    segs = build_segments(
        frame, [Detection(300, 300, 300 + dm.shape[1], 300 + dm.shape[0], 0.9)], **KW
    )
    await CodeReader(pool).adecode_segments(segs)
    (code,) = segs[0].barcodes
    assert (code.symbology, code.value, code.engine) == ("DATAMATRIX", "DM-42", "zxing")


async def test_duplicates_across_overlapping_crops_deduped(pool):
    qr = make_code("SAME-CODE", "QRCode", 4)
    frame = canvas_with_codes([(qr, 200, 200)])
    h, w = qr.shape
    # two different boxes that both cover the QR (neither contains the other >= 0.8)
    dets = [
        Detection(180, 190, 200 + w + 10, 200 + h + 5, 0.9),
        Detection(200, 160, 200 + w + 60, 200 + h + 40, 0.8),
    ]
    segs = build_segments(
        frame, dets, containment_threshold=0.99, nms_iou=0.95, pad_x=0.25, pad_y=0.2
    )
    assert len(segs) == 2
    reader = CodeReader(pool, pyzbar_decode="auto")
    await reader.adecode_segments(segs)
    assert all(len(s.barcodes) == 1 for s in segs)  # per-segment lists keep every hit
    flat = dedupe_barcodes(segs)
    assert len(flat) == 1 and flat[0][0] == 0
