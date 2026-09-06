"""학습 라벨은 `keypoint_annotations` 를 쓴다 — 그 계약을 못 박는다.

배경: 같은 GT JSON 안에 keypoint 좌표가 두 벌 있고 인덱스 배정이 다를 수 있다.
2026-09-06 `live_capture_gt` 851장 전수에서 camera-facing 0123 규약
(0 왼쪽/1 오른쪽, 0·1 위/3·2 아래) 위반은 keypoint_annotations 0장,
projected_cuboid 198장(23.3%)이었다.  근거:
`_docs/audits/accuracy_root_cause_v1/REAL_LABEL_AUDIT.md`
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "challenge/yolo_pose_one_model/scripts"))
sys.path.insert(0, str(ROOT / "challenge/scripts/dataset"))

from prepare_yolo_pose import SENTINEL, load_kps  # noqa: E402
from gen_flip_noise_aug import FLIP_PERM_8, flip  # noqa: E402

LIVE_GT = ROOT / "challenge/data/01_real/live_capture_gt"


def _write(tmp_path, obj):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"objects": [obj]}))
    return p


def test_load_kps_prefers_keypoint_annotations(tmp_path):
    """두 필드가 다르면 keypoint_annotations 를 따라야 한다."""
    ann = [{"xy": [float(i), float(i) + 0.5]} for i in range(9)]
    proj = [[100.0 + i, 200.0 + i] for i in range(8)]
    kps = load_kps(_write(tmp_path, {
        "keypoint_annotations": ann,
        "projected_cuboid": proj,
        "projected_cuboid_centroid": [999.0, 999.0],
    }))
    assert kps == [(float(i), float(i) + 0.5) for i in range(9)]


def test_load_kps_falls_back_for_synthetic(tmp_path):
    """합성 GT 에는 keypoint_annotations 가 없다 — 기존 경로가 그대로 유지돼야 한다."""
    proj = [[float(i), float(i)] for i in range(8)]
    kps = load_kps(_write(tmp_path, {
        "projected_cuboid": proj, "projected_cuboid_centroid": [7.0, 7.0],
    }))
    assert kps == [(float(i), float(i)) for i in range(8)] + [(7.0, 7.0)]


def test_load_kps_null_xy_becomes_sentinel(tmp_path):
    ann = [{"xy": [1.0, 2.0]} for _ in range(9)]
    ann[3] = {"xy": None}
    kps = load_kps(_write(tmp_path, {"keypoint_annotations": ann}))
    assert kps[3] == (SENTINEL, SENTINEL)


def test_flip_also_flips_keypoint_annotations():
    """이미지를 뒤집으면 keypoint_annotations 도 같이 뒤집혀야 한다.

    안 그러면 뒤집힌 이미지에 안 뒤집힌 라벨이 붙는다 — load_kps 가 그 필드를
    우선하므로 조용히 틀린 데이터셋이 만들어진다.
    """
    w = 640
    img = np.zeros((480, w, 3), np.uint8)
    ann = [{"xy": [10.0 * i, 5.0 * i], "visibility": 2} for i in range(9)]
    proj = [[10.0 * i, 5.0 * i] for i in range(8)]
    _, out = flip(img, {"projected_cuboid": proj,
                        "projected_cuboid_centroid": [80.0, 40.0],
                        "keypoint_annotations": ann})
    got = out["keypoint_annotations"]
    assert len(got) == 9
    for dst, src in enumerate(FLIP_PERM_8):
        assert got[dst]["xy"] == [w - 1.0 - ann[src]["xy"][0], ann[src]["xy"][1]]
    # centroid 는 좌우만 뒤집히고 자리는 그대로다
    assert got[8]["xy"] == [w - 1.0 - ann[8]["xy"][0], ann[8]["xy"][1]]
    # 뒤집힌 라벨이 원본과 같으면(= 안 뒤집혔으면) 실패해야 한다
    assert got[0]["xy"] != ann[0]["xy"]


def _lr_ok(pts):
    """camera-facing 0123: 0 이 1 의 왼쪽, 3 이 2 의 왼쪽."""
    return pts[0][0] < pts[1][0] and pts[3][0] < pts[2][0]


@pytest.mark.skipif(not LIVE_GT.exists(), reason="live_capture_gt 없음")
def test_live_capture_gt_keypoint_annotations_obey_convention():
    """실데이터 불변식 — 이게 깨지면 정본 필드 선택의 근거가 무너진다."""
    checked = violations = 0
    for jp in sorted(LIVE_GT.glob("*/*.json")):
        try:
            objs = json.loads(jp.read_text()).get("objects") or []
        except Exception:
            continue
        if not objs:
            continue
        ann = objs[0].get("keypoint_annotations")
        if not isinstance(ann, list) or len(ann) < 4:
            continue
        pts = [e.get("xy") for e in ann[:4]]
        if any(p is None for p in pts):
            continue
        checked += 1
        if not _lr_ok(pts):
            violations += 1
    assert checked > 500, f"검사 프레임이 너무 적다: {checked}"
    assert violations == 0, f"{violations}/{checked} 프레임이 규약을 어긴다"
