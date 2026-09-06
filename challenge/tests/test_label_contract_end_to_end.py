"""JSON -> load_kps -> pad -> to_line -> YOLO txt 를 **끝까지** 통과시키는 계약 테스트.

기존 `test_keypoint_field_contract.py` 는 loader 수준에서 끝나서
sentinel 이 padding 을 통과해 감독 대상이 되는 것을 잡지 못했다:

    xy=None -> (-0.5, -0.5) -> +PAD -> (99.5, 99.5) -> 캔버스 안 -> v=2

게다가 그 점이 bbox 에도 들어가 8x8 px 상자가 308x248 px 로 늘어났다.
계약 정본은 `scripts/annotate/real_gt_v2_schema.keypoint_annotations_to_ultralytics`:
visibility 0 또는 xy None 이면 학습 타깃은 [0, 0, 0] 이다.
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "challenge/yolo_pose_one_model/scripts"))
sys.path.insert(0, str(ROOT / "challenge/scripts/dataset"))

import prepare_yolo_pose as pyp  # noqa: E402
from gen_flip_noise_aug import FLIP_PERM_8, flip  # noqa: E402

W, H = 640, 480


def _convert(tmp_path, obj, img=None):
    """JSON + 이미지를 실제 변환 경로에 태우고 파싱된 YOLO 라벨을 돌려준다."""
    img = np.zeros((H, W, 3), np.uint8) if img is None else img
    ip, ap = tmp_path / "i.png", tmp_path / "a.json"
    cv2.imwrite(str(ip), img)
    ap.write_text(json.dumps({"objects": [obj]}), encoding="utf-8")
    op, lp = tmp_path / "o.png", tmp_path / "o.txt"
    res = pyp.one(("s", str(ip), str(ap), str(op), str(lp)))
    if res != "ok":
        return res, None, None
    f = lp.read_text(encoding="utf-8").strip().split()
    pw, ph = img.shape[1] + 2 * pyp.PAD, img.shape[0] + 2 * pyp.PAD
    bbox = [float(v) for v in f[1:5]]
    kps = [(float(f[5 + 3 * i]) * pw, float(f[6 + 3 * i]) * ph, int(f[7 + 3 * i]))
           for i in range(9)]
    return res, bbox, kps


def _ann(vis=2):
    return [{"xy": [300.0 + i, 240.0 + i], "visibility": vis, "in_frame": True,
             "source": "manual_click", "reason": "visible"} for i in range(9)]


def test_none_keypoint_stays_unsupervised_after_padding(tmp_path):
    """xy=None 은 padding 을 통과해도 v=0 이어야 하고 bbox 에도 들어가면 안 된다."""
    ann = _ann()
    ann[3] = {"xy": None, "visibility": 0, "in_frame": False,
              "source": "unknown", "reason": "unknown"}
    res, bbox, kps = _convert(tmp_path, {"keypoint_annotations": ann})
    assert res == "ok"
    assert kps[3] == (0.0, 0.0, 0), "좌표를 모르는 점이 감독되고 있다"
    pw, ph = W + 2 * pyp.PAD, H + 2 * pyp.PAD
    x0 = (bbox[0] - bbox[2] / 2) * pw
    y0 = (bbox[1] - bbox[3] / 2) * ph
    # sentinel 이 bbox 에 섞이면 좌상단이 (99.5, 99.5) 로 끌려간다
    assert x0 > 300.0 and y0 > 300.0, f"bbox 가 sentinel 쪽으로 끌렸다: {x0:.1f},{y0:.1f}"


def test_visibility_zero_with_legacy_xy_is_not_supervised(tmp_path):
    """좌표는 남아 있지만 provenance 를 모르는 점(visibility 0)도 감독하지 않는다."""
    ann = _ann()
    ann[5] = {"xy": [310.0, 250.0], "visibility": 0, "in_frame": True,
              "source": "unknown", "reason": "unknown"}
    _, _, kps = _convert(tmp_path, {"keypoint_annotations": ann})
    assert kps[5] == (0.0, 0.0, 0)


def test_known_out_of_frame_keypoint_follows_contract_after_padding(tmp_path):
    """원본 밖이지만 padding 캔버스 안인 '아는' 점은 계약대로 감독된다.

    reflect padding 이 그 자리를 채우므로 현행 계약은 이 점을 감독 대상으로 둔다.
    모르는 점과 달리 좌표가 실재하므로 v=2 이고 위치가 보존돼야 한다.
    """
    ann = _ann()
    ann[2] = {"xy": [-40.0, 240.0], "visibility": 2, "in_frame": False,
              "source": "manual_click", "reason": "visible"}
    _, _, kps = _convert(tmp_path, {"keypoint_annotations": ann})
    assert kps[2][2] == 2
    assert kps[2][0] == pytest.approx(-40.0 + pyp.PAD, abs=1.0)

    # 캔버스 밖(-200)이면 아는 점이라도 v=0
    ann[2]["xy"] = [-200.0, 240.0]
    _, _, kps = _convert(tmp_path, {"keypoint_annotations": ann})
    assert kps[2] == (0.0, 0.0, 0)


def test_real_keypoint_annotations_survive_full_conversion(tmp_path):
    """keypoint_annotations 가 projected_cuboid 를 이기고 끝까지 살아남는다."""
    ann = _ann()
    obj = {"keypoint_annotations": ann,
           "projected_cuboid": [[10.0 + i, 20.0 + i] for i in range(8)],
           "projected_cuboid_centroid": [11.0, 21.0]}
    _, _, kps = _convert(tmp_path, obj)
    for i in range(9):
        assert kps[i][2] == 2
        assert kps[i][0] == pytest.approx(300.0 + i + pyp.PAD, abs=1.0)
        assert kps[i][1] == pytest.approx(240.0 + i + pyp.PAD, abs=1.0)


def test_flip_full_conversion_preserves_index_contract(tmp_path):
    """flip 산출물을 실제로 변환했을 때 인덱스 순열과 미러링이 함께 적용된다."""
    img = np.zeros((H, W, 3), np.uint8)
    ann = _ann()
    src_obj = {"keypoint_annotations": ann,
               "projected_cuboid": [[300.0 + i, 240.0 + i] for i in range(8)],
               "projected_cuboid_centroid": [308.0, 248.0]}
    fimg, fobj = flip(img, json.loads(json.dumps(src_obj)))

    _, _, kps = _convert(tmp_path, fobj, img=fimg)
    for dst, src in enumerate(FLIP_PERM_8):
        assert kps[dst][0] == pytest.approx(W - 1.0 - (300.0 + src) + pyp.PAD, abs=1.0), (
            f"index {dst} 가 원본 {src} 의 미러 위치에 있지 않다")
        assert kps[dst][1] == pytest.approx(240.0 + src + pyp.PAD, abs=1.0)
    assert kps[8][0] == pytest.approx(W - 1.0 - 308.0 + pyp.PAD, abs=1.0)


def test_synthetic_fallback_still_converts(tmp_path):
    """합성 경로(projected_cuboid) 도 같은 변환을 통과해야 한다."""
    obj = {"projected_cuboid": [[300.0 + i, 240.0 + i] for i in range(8)],
           "projected_cuboid_centroid": [308.0, 248.0]}
    res, _, kps = _convert(tmp_path, obj)
    assert res == "ok"
    assert all(k[2] == 2 for k in kps)


def test_to_line_accepts_bare_pairs():
    """기존 호출자(prepare_real_ft) 가 쓰는 (x, y) 2-튜플 경로가 계속 동작한다."""
    line = pyp.to_line(840, 680, [(400.0 + i, 340.0 + i) for i in range(9)])
    assert line is not None and line.split()[-1] == "2"
