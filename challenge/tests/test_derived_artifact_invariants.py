"""디스크에 있는 파생 산출물 자체를 검사한다 — 생성기 코드가 아니라.

계약 테스트(`test_keypoint_field_contract.py`, `test_label_contract_end_to_end.py`)는
**코드**를 묶는다.  생성기를 고쳐도 디스크에 남은 낡은 증강본은 여전히 초록불로
통과하고, 그걸 학습에 쓰면 뒤집힌 이미지에 안 뒤집힌 라벨이 붙는다.
실제로 2026-09-06 `flip_noise_aug_livegt` 가 그 상태였다 —
생성기는 `f2b2739` 에 고쳐졌으나 산출물은 09-04 빌드였다.

여기서는 산출물의 두 keypoint 필드가 부모와 **미러 + FLIP_PERM_8** 관계인지 본다.
"""
from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIVE_GT = ROOT / "challenge/data/01_real/live_capture_gt"
DERIVED = ROOT / "challenge/data/03_derived"
FLIP_PERM_8 = (1, 0, 3, 2, 5, 4, 7, 6)
TOL = 1e-6

# 낡아서 학습에 쓰면 안 되는 폴더.  지우지 않고 여기에 적어 둔다 —
# 근거: _docs/audits/next_accuracy_v2/DERIVED_DATA_AUDIT.md
STALE_FLIP_DIRS = {"flip_noise_aug_livegt"}


def _parent(stem: str):
    """`<session>_<frame>_f` -> live_capture_gt 의 부모 JSON 경로."""
    body = stem[:-2]
    sess, _, frame = body.rpartition("_")
    p = LIVE_GT / f"{sess}_manual_gt" / f"{frame}.json"
    return p if p.is_file() else None


def _obj(p):
    return json.loads(p.read_text(encoding="utf-8"))["objects"][0]


def _flip_dirs():
    if not DERIVED.is_dir():
        return []
    out = []
    for d in sorted(DERIVED.iterdir()):
        if d.is_dir() and d.name not in STALE_FLIP_DIRS and any(d.glob("*_f.json")):
            out.append(d)
    return out


@pytest.mark.skipif(not _flip_dirs(), reason="검사할 flip 파생 폴더 없음")
@pytest.mark.parametrize("d", _flip_dirs(), ids=lambda p: p.name)
def test_flip_artifact_mirrors_both_keypoint_fields(d):
    checked = ann_checked = 0
    bad_proj, bad_ann = [], []
    for fp in sorted(d.glob("*_f.json")):
        pp = _parent(fp.stem)
        if pp is None:
            continue
        src, dst = _obj(pp), _obj(fp)
        w = json.loads(pp.read_text(encoding="utf-8"))["camera_data"]["width"]
        checked += 1

        sp, dp = src.get("projected_cuboid"), dst.get("projected_cuboid")
        if sp and dp and len(sp) >= 8 and len(dp) >= 8:
            for i, s in enumerate(FLIP_PERM_8):
                if (abs(dp[i][0] - (w - 1.0 - sp[s][0])) > 1e-3
                        or abs(dp[i][1] - sp[s][1]) > 1e-3):
                    bad_proj.append(fp.name)
                    break

        sa, da = src.get("keypoint_annotations"), dst.get("keypoint_annotations")
        if isinstance(sa, list) and isinstance(da, list) and len(sa) >= 9 and len(da) >= 9:
            ann_checked += 1
            for i, s in enumerate(FLIP_PERM_8 + (8,)):
                sx, dx = sa[s].get("xy"), da[i].get("xy")
                if sx is None or dx is None:
                    if sx is not None or dx is not None:
                        bad_ann.append(fp.name)
                        break
                    continue
                if (abs(dx[0] - (w - 1.0 - sx[0])) > 1e-3
                        or abs(dx[1] - sx[1]) > 1e-3):
                    bad_ann.append(fp.name)
                    break

    assert checked > 0, f"{d.name}: 부모를 찾은 flip 프레임이 없다"
    assert not bad_proj, (
        f"{d.name}: projected_cuboid 가 미러+순열이 아니다 "
        f"{len(bad_proj)}/{checked} 예) {bad_proj[:3]}")
    assert ann_checked > 0, (
        f"{d.name}: keypoint_annotations 를 가진 flip 산출물이 없다 — "
        "생성기가 그 필드를 떨어뜨렸다는 뜻이다")
    assert not bad_ann, (
        f"{d.name}: keypoint_annotations 가 미러+순열이 아니다 "
        f"{len(bad_ann)}/{ann_checked} 예) {bad_ann[:3]}  "
        "(생성기는 고쳤는데 산출물이 낡았을 때 이 줄이 뜬다)")


@pytest.mark.skipif(not (DERIVED / "flip_noise_aug_livegt").is_dir(),
                    reason="낡은 폴더 없음")
def test_known_stale_flip_dir_is_still_stale():
    """격리 목록이 최신인지 확인한다 — 낡은 게 고쳐졌으면 목록에서 빼라."""
    d = DERIVED / "flip_noise_aug_livegt"
    for fp in sorted(d.glob("*_f.json"))[:20]:
        pp = _parent(fp.stem)
        if pp is None:
            continue
        sa = _obj(pp).get("keypoint_annotations")
        da = _obj(fp).get("keypoint_annotations")
        if isinstance(sa, list) and isinstance(da, list):
            if sa[0].get("xy") != da[0].get("xy"):
                pytest.fail(
                    f"{d.name} 이 더는 낡지 않았다 — STALE_FLIP_DIRS 에서 빼고 "
                    "일반 검사 대상으로 돌려라")
            return
    pytest.skip("비교할 표본 없음")


# --------------------------------------------------------------------------- #
# 빌더가 낡은 파생 폴더를 거부하는가
# --------------------------------------------------------------------------- #

def _guard():
    import sys
    sys.path.insert(0, str(ROOT / "challenge/yolo_pose_one_model/scripts"))
    from prepare_yolo_pose_from_live_gt import assert_derived_is_current
    return assert_derived_is_current


@pytest.mark.skipif(not (DERIVED / "flip_noise_aug_livegt").is_dir(),
                    reason="낡은 폴더 없음")
def test_builder_rejects_artifact_without_provenance():
    """필드는 있는데 provenance 가 없으면 고치기 전 산출물이다 — 거부해야 한다."""
    with pytest.raises(SystemExit) as e:
        _guard()(DERIVED / "flip_noise_aug_livegt")
    assert "keypoint_source" in str(e.value)


@pytest.mark.skipif(not (DERIVED / "truncation_crops_livegt").is_dir(),
                    reason="crop 폴더 없음")
def test_builder_rejects_artifact_without_keypoint_annotations():
    """필드 자체가 없으면 projected_cuboid fallback 으로 내려간다 — 거부해야 한다."""
    with pytest.raises(SystemExit) as e:
        _guard()(DERIVED / "truncation_crops_livegt")
    assert "keypoint_annotations" in str(e.value)


@pytest.mark.skipif(not (DERIVED / "flip_noise_aug_livegt_v2").is_dir(),
                    reason="재생성본 없음")
def test_builder_accepts_regenerated_artifact():
    """고친 생성기로 다시 만든 것은 통과해야 한다 — 안 그러면 가드가 과하다."""
    _guard()(DERIVED / "flip_noise_aug_livegt_v2")
