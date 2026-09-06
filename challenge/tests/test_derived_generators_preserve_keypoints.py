"""파생 생성기가 정본 keypoint 필드를 보존·변환하는지 못 박는다 (§6).

2026-09-06 감사에서 파생 생성기 9개 중 `keypoint_annotations` 를 보존하는 것이
`gen_flip_noise_aug.py` 하나뿐이었다.  나머지는 결과 JSON 을 **새 dict 로** 만들며
그 필드를 떨어뜨렸고, 그러면 학습 변환기가 `projected_cuboid` fallback 으로
내려간다 — 그 필드는 live_capture_gt 851장에서 규약을 198장(23.3%) 어긴다.
근거: `_docs/audits/next_accuracy_v2/DERIVED_DATA_AUDIT.md`

여기서는 **생성기가 실제로 그 필드를 쓰는지**(정적)와
**공용 변환 헬퍼가 좌표·순열·상태를 맞게 다루는지**(단위)를 본다.
디스크 산출물 검사는 `test_derived_artifact_invariants.py` 가 맡는다.
"""
from __future__ import annotations

import ast
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATASET = ROOT / "challenge/scripts/dataset"
sys.path.insert(0, str(DATASET))

from keypoint_annotations_transform import (  # noqa: E402
    FLIP_PERM_8, provenance, transform_annotations)

# 파생 real 데이터를 내는 생성기.  전부 정본 필드를 내보내야 한다.
GENERATORS = [
    "gen_truncation_crops.py",
    "pad_truncation_crops.py",
    "augment_ratio_robust.py",
    "augment_dataset.py",
    "make_pseudo_gt.py",
    "gen_flip_noise_aug.py",
]


@pytest.mark.parametrize("name", GENERATORS)
def test_generator_emits_keypoint_annotations(name):
    """생성기 소스에 `keypoint_annotations` 를 **쓰는** 경로가 있어야 한다.

    문자열 검색이 아니라 AST 로 본다 — 주석·docstring 에만 있는 것을 통과시키지
    않기 위해서다.
    """
    src = (DATASET / name).read_text(encoding="utf-8")
    tree = ast.parse(src)
    emits = False
    for node in ast.walk(tree):
        # obj["keypoint_annotations"] = ...
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            if isinstance(node.slice, ast.Constant) and \
                    node.slice.value == "keypoint_annotations":
                emits = True
        # dict 리터럴의 키로 들어가는 경우
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and k.value == "keypoint_annotations":
                    emits = True
        # kat.attach(...) 로 위임하는 경우
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("attach", "transform_annotations"):
                emits = True
    assert emits, (
        f"{name} 이 keypoint_annotations 를 내보내지 않는다 — "
        "산출물이 projected_cuboid fallback 으로 내려간다")


@pytest.mark.parametrize("name", GENERATORS)
def test_generator_records_provenance(name):
    """`keypoint_source` / `parent_frame` / `transformation` 을 남겨야 한다."""
    src = (DATASET / name).read_text(encoding="utf-8")
    tree = ast.parse(src)
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in ("keypoint_source", "parent_frame", "transformation"):
                seen.add(node.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("attach", "provenance"):
                seen |= {"keypoint_source", "parent_frame", "transformation"}
    missing = {"keypoint_source", "parent_frame", "transformation"} - seen
    assert not missing, f"{name}: provenance 필드 누락 {sorted(missing)}"


def _ann(n=9):
    return [{"xy": [10.0 * i, 5.0 * i], "visibility": 2, "in_frame": True,
             "source": "manual_click", "reason": "visible"} for i in range(n)]


def test_transform_applies_affine():
    out = transform_annotations({"keypoint_annotations": _ann()},
                                [[0.5, 0.0, -100.0], [0.0, 0.5, -50.0]], 640, 480)
    assert out[2]["xy"] == pytest.approx([0.5 * 20 - 100, 0.5 * 10 - 50])


def test_transform_flip_permutes_and_mirrors():
    ann = _ann()
    w = 640
    out = transform_annotations({"keypoint_annotations": ann},
                                [[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0]],
                                w, 480, perm=FLIP_PERM_8)
    for dst, srci in enumerate(FLIP_PERM_8):
        assert out[dst]["xy"] == pytest.approx(
            [w - 1.0 - ann[srci]["xy"][0], ann[srci]["xy"][1]])
    # centroid 는 제자리에서 미러만 된다
    assert out[8]["xy"] == pytest.approx([w - 1.0 - ann[8]["xy"][0], ann[8]["xy"][1]])


def test_transform_keeps_unknown_points_unknown():
    """xy=None 은 좌표가 아니라 상태다 — 옮기지 않는다."""
    ann = _ann()
    ann[3] = {"xy": None, "visibility": 0, "in_frame": False,
              "source": "unknown", "reason": "unknown"}
    out = transform_annotations({"keypoint_annotations": ann},
                                [[2.0, 0.0, 5.0], [0.0, 2.0, 5.0]], 640, 480)
    assert out[3]["xy"] is None
    assert out[3]["visibility"] == 0


def test_transform_updates_in_frame_for_new_canvas():
    ann = _ann()
    out = transform_annotations({"keypoint_annotations": ann},
                                [[1.0, 0.0, 1000.0], [0.0, 1.0, 0.0]], 640, 480)
    assert all(e["in_frame"] is False for e in out), "새 캔버스 밖인데 in_frame 이 True 다"


def test_transform_returns_none_when_parent_has_no_field():
    """합성 GT 처럼 부모에 필드가 없으면 None — 호출부가 만들어 낼지 정한다."""
    assert transform_annotations({"projected_cuboid": [[0, 0]] * 8},
                                 [[1, 0, 0], [0, 1, 0]], 640, 480) is None


def test_provenance_has_three_fields():
    p = provenance("a/b.json", {"kind": "crop"})
    assert set(p) == {"keypoint_source", "parent_frame", "transformation"}
    assert p["keypoint_source"] == "keypoint_annotations"
