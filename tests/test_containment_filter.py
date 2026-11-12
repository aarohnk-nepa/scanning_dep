import math

import numpy as np
import pytest

from src.segment.postprocess import (
    Detection,
    build_segments,
    nms,
    normalize_prompt,
    pad_box,
    suppress_contained,
)

PAD_X, PAD_Y = 0.25, 0.20
KW = dict(containment_threshold=0.80, nms_iou=0.50, pad_x=PAD_X, pad_y=PAD_Y)


def frame(h=1000, w=1000):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_outer_box_suppressed_inner_kept_and_padded():
    f = frame()
    outer = Detection(100, 100, 500, 500, 0.9, "barcode")
    inner = Detection(150, 150, 450, 300, 0.6, "barcode")
    segs = build_segments(f, [outer, inner], **KW)

    assert len(segs) == 1
    s = segs[0]
    assert s.raw_bbox == (150, 150, 450, 300)
    # pad_x = 300 * 0.25 = 75, pad_y = 150 * 0.20 = 30
    assert s.padded_bbox == (75, 120, 525, 330)
    assert s.crop.shape == (210, 450, 3)
    assert np.shares_memory(s.crop, f)  # a view, not a copy
    assert s.segment_index == 0


def test_partial_overlap_keeps_both():
    a = Detection(100, 100, 300, 300, 0.9)
    b = Detection(250, 100, 450, 300, 0.8)  # 25% overlap
    assert len(build_segments(frame(), [a, b], **KW)) == 2


def test_nested_chain_keeps_innermost_and_siblings():
    a = Detection(0, 0, 900, 900, 0.9)
    b = Detection(100, 100, 800, 800, 0.8)
    c = Detection(200, 200, 400, 300, 0.7)
    d = Detection(500, 500, 700, 600, 0.6)  # sibling of c, also inside a and b
    segs = build_segments(frame(), [a, b, c, d], **KW)
    assert sorted(s.raw_bbox for s in segs) == [(200, 200, 400, 300), (500, 500, 700, 600)]


def test_containment_runs_before_nms():
    # IoU(outer, inner) = 0.81: NMS alone would keep the higher-scored OUTER box.
    outer = Detection(100, 100, 300, 300, 0.9)
    inner = Detection(110, 110, 290, 290, 0.5)
    boxes = np.array([outer.box, inner.box], dtype=float)
    assert nms(boxes, np.array([0.9, 0.5]), 0.5) == [0]  # control
    segs = build_segments(frame(), [outer, inner], **KW)
    assert [s.raw_bbox for s in segs] == [inner.box]


def test_nms_removes_near_duplicate_equal_area():
    a = Detection(100, 100, 300, 300, 0.9)
    b = Detection(105, 100, 305, 300, 0.7)  # same area, IoU ~0.95
    segs = build_segments(frame(), [a, b], **KW)
    assert [s.raw_bbox for s in segs] == [a.box]


def test_suppress_contained_threshold_boundary():
    boxes = np.array([[0, 0, 100, 100], [50, 0, 150, 100], [60, 0, 110, 50]], dtype=float)
    # small box [60,0,110,50] is 80% inside box 0 and 100% inside box 1 -> both outers drop
    assert suppress_contained(boxes, 0.80).tolist() == [False, False, True]
    assert suppress_contained(boxes, 0.85).tolist() == [True, False, True]


def test_padding_clamped_to_frame():
    assert pad_box((0, 0, 100, 50), PAD_X, PAD_Y, 110, 55) == (0, 0, 110, 55)
    segs = build_segments(frame(200, 200), [Detection(150, 150, 210, 230, 0.9)], **KW)
    # clamped box 150..200 (50x50): pad 12.5 / 10
    assert segs[0].padded_bbox == (137, 140, 200, 200)


def test_bad_tiny_and_capped_detections():
    dets = [
        Detection(math.nan, 0, 10, 10, 0.9),
        Detection(0, 0, 10, 10, math.inf),
        Detection(10, 10, 10, 50, 0.9),  # zero width
        Detection(10, 10, 12, 12, 0.8),  # padded crop < 4px
    ] + [Detection(100 * i, 0, 100 * i + 50, 50, 0.5 + i / 100) for i in range(5)]
    segs = build_segments(frame(), dets, **KW, max_count=3)
    assert len(segs) == 3
    assert [s.segment_index for s in segs] == [0, 1, 2]
    assert [s.score for s in segs] == sorted([s.score for s in segs], reverse=True)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("barcode", "barcode."),
        ("Barcode", "barcode."),
        ("barcode.", "barcode."),
        ("  QR   Code ", "qr code."),
    ],
)
def test_normalize_prompt(raw, expected):
    assert normalize_prompt(raw) == expected
