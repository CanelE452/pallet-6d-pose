"""파생 real 데이터가 정본 keypoint 필드를 잃지 않게 하는 공용 변환.

배경.  파생 생성기 대부분이 결과 JSON 을 **새 dict 로** 만들면서
``keypoint_annotations`` 를 떨어뜨렸다.  그러면 학습 변환기
``prepare_yolo_pose.load_kps`` 가 ``projected_cuboid`` fallback 으로 내려가는데,
그 필드는 ``live_capture_gt`` 851장에서 camera-facing 0123 규약을
198장(23.3%) 어긴다.  즉 부모에 규약을 지키는 필드가 있어도 파생 단계에서 소실됐다.
근거: ``_docs/audits/next_accuracy_v2/DERIVED_DATA_AUDIT.md``

여기 한 벌만 두고 생성기들이 import 한다 — 다섯 곳에 베끼면 갈라진다.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# camera-facing 0123 에서 좌우를 뒤집으면 코너 짝이 바뀐다. centroid(8)는 제자리.
FLIP_PERM_8 = (1, 0, 3, 2, 5, 4, 7, 6)


def transform_annotations(src_obj, M, width, height, *, perm=None):
    """``keypoint_annotations`` 를 2x3 affine ``M`` 으로 옮긴다.

    ``M`` 은 ``[[a, b, tx], [c, d, ty]]`` (list 또는 numpy).
    ``perm`` 은 좌우가 바뀌는 변환(hflip)에서만 준다 — ``FLIP_PERM_8``.

    ``xy`` 가 None 인 점(좌표를 모르는 점)은 그대로 None 으로 둔다.  좌표가 아니라
    상태이므로 옮길 것이 없다.  ``in_frame`` 은 새 캔버스 기준으로 다시 계산한다.

    부모에 이 필드가 없으면(합성 GT 등) ``None`` 을 돌려준다 — 호출부는 그때
    아무 것도 쓰지 않으면 된다.
    """
    ann = src_obj.get("keypoint_annotations")
    if not (isinstance(ann, list) and len(ann) >= 9):
        return None

    moved = []
    for entry in ann[:9]:
        e = dict(entry) if isinstance(entry, dict) else {}
        xy = e.get("xy")
        if xy is not None:
            x = float(M[0][0]) * xy[0] + float(M[0][1]) * xy[1] + float(M[0][2])
            y = float(M[1][0]) * xy[0] + float(M[1][1]) * xy[1] + float(M[1][2])
            e["xy"] = [x, y]
            e["in_frame"] = bool(0 <= x < width and 0 <= y < height)
        moved.append(e)

    if perm is not None:
        moved = [moved[i] for i in perm] + [moved[8]]
    return moved


def _rel(path):
    """저장소 상대경로로 정규화한다.

    이 저장소는 Windows 와 Ubuntu 두 환경에서 공유되므로 절대경로를 남기면
    다른 머신에서 부모를 못 찾는다.
    """
    if path is None:
        return None
    try:
        return os.path.relpath(Path(path).resolve(), REPO).replace(os.sep, "/")
    except (ValueError, OSError):
        return str(path)


def provenance(parent_frame, transformation):
    """파생본이 어디서 어떻게 나왔는지 — 오브젝트에 그대로 얹는다."""
    return {"keypoint_source": "keypoint_annotations",
            "parent_frame": _rel(parent_frame),
            "transformation": transformation}


def attach(obj, src_obj, M, width, height, *, perm=None,
           parent_frame=None, transformation=None):
    """``obj`` 에 변환된 keypoint_annotations 와 provenance 를 얹는다(제자리).

    부모에 필드가 없어 옮길 것이 없으면 **provenance 도 찍지 않는다.**
    ``keypoint_source`` 만 남기면 거짓말이 되고, 그 필드 유무로 낡은 산출물을
    걸러내는 빌더 가드(`prepare_yolo_pose_from_live_gt.assert_derived_is_current`)를
    통과시켜 버린다.
    """
    moved = transform_annotations(src_obj, M, width, height, perm=perm)
    if moved is None:
        return obj
    obj["keypoint_annotations"] = moved
    obj.update(provenance(parent_frame, transformation))
    return obj
