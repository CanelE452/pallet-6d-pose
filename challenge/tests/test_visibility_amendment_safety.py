"""visibility-only 반영이 좌표를 건드릴 수 없다는 것을 강제한다.

이 저장소에는 어노테이션 도구가 좌표/가시성을 조용히 망가뜨린 사고가 있었다
(`u<0` 오판으로 72 프레임 153 코너가 invisible 로 저장됨).  그래서 "안 바꾼다" 를
선언이 아니라 검사로 둔다.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APPLY = REPO_ROOT / "scripts" / "annotate" / "apply_visibility_amendments.py"
REVIEW = REPO_ROOT / "scripts" / "annotate" / "review_visibility_only.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def apply_module():
    return load(APPLY, "apply_visibility_amendments")


def payload() -> dict:
    return {
        "schema_version": "real_pallet_gt_v2",
        "camera_data": {"width": 640, "height": 480,
                        "intrinsics": {"fx": 614.0, "fy": 614.0,
                                       "cx": 329.0, "cy": 234.0}},
        "objects": [{
            "object_type": "plastic_standard_110x130x11",
            "pose_transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]],
            "projected_cuboid": [[10.0, 10.0]] * 8,
            "keypoint_annotations": [
                {"xy": [float(i), float(i)], "visibility": 0, "in_frame": True,
                 "source": "unknown", "reason": "unknown"}
                for i in range(9)
            ],
        }],
    }


# ── 허용 편집은 통과한다 ────────────────────────────────────────────────

def test_visibility_and_reason_are_allowed(apply_module):
    before = payload()
    after = copy.deepcopy(before)
    for point in after["objects"][0]["keypoint_annotations"]:
        point["visibility"] = 2
        point["reason"] = "visible"
    assert apply_module.forbidden_diff(before, after) == []


def test_source_is_not_a_visibility_field(apply_module):
    """`source` 는 좌표의 출처다.  가시성만 본 리뷰가 바꿀 수 없다.

    한때 여기에 `human_visibility_review` 를 써서 64 프레임이 GT-v2 enum 을
    위반했고 평가기가 통째로 멈췄다.
    """

    before = payload()
    after = copy.deepcopy(before)
    after["objects"][0]["keypoint_annotations"][0]["source"] = "manual_click"
    assert apply_module.forbidden_diff(before, after)


def test_written_values_stay_inside_the_gt_v2_enums(apply_module):
    """쓰는 값이 스키마 enum 안에 있는지 — 쓰기 전에 계약으로 막는다."""

    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "real_gt_v2_schema",
        REPO_ROOT / "scripts" / "annotate" / "real_gt_v2_schema.py")
    schema = _ilu.module_from_spec(spec)
    spec.loader.exec_module(schema)
    for decision in apply_module.STATES.values():
        if decision is None:
            continue
        visibility, reason = decision
        assert visibility in schema.VISIBILITY_VALUES
        assert reason in schema.KEYPOINT_REASONS


def test_apply_validates_the_schema_before_writing(apply_module):
    """스키마 검증이 실제로 배선돼 있는지 — 위 검사가 공허해지지 않게."""

    source = APPLY.read_text()
    assert "validate_gt_v2(after)" in source


# ── 금지 편집은 전부 잡힌다 ─────────────────────────────────────────────

@pytest.mark.parametrize("mutate, label", [
    (lambda p: p["objects"][0]["keypoint_annotations"][3].__setitem__(
        "xy", [99.0, 99.0]), "keypoint xy"),
    (lambda p: p["objects"][0]["keypoint_annotations"][0].__setitem__(
        "in_frame", False), "in_frame"),
    (lambda p: p["objects"][0].__setitem__(
        "pose_transform", [[0, 0, 0, 0]] * 4), "pose_transform"),
    (lambda p: p["objects"][0].__setitem__(
        "projected_cuboid", [[0.0, 0.0]] * 8), "projected_cuboid"),
    (lambda p: p["objects"][0].__setitem__("object_type", "wood_small_80x59x14"),
     "object_type"),
    (lambda p: p["camera_data"]["intrinsics"].__setitem__("fx", 1.0), "intrinsics"),
    (lambda p: p["objects"][0]["keypoint_annotations"].pop(), "keypoint 삭제"),
])
def test_forbidden_changes_are_detected(apply_module, mutate, label):
    before = payload()
    after = copy.deepcopy(before)
    mutate(after)
    problems = apply_module.forbidden_diff(before, after)
    assert problems, f"{label} 변경이 잡히지 않았다"


def test_a_single_pixel_coordinate_change_is_detected(apply_module):
    """1 px 도 통과시키지 않는다."""

    before = payload()
    after = copy.deepcopy(before)
    after["objects"][0]["keypoint_annotations"][5]["xy"][0] += 1.0
    assert apply_module.forbidden_diff(before, after)


def test_visibility_change_together_with_coordinate_change_is_still_rejected(
        apply_module):
    """허용 편집에 금지 편집을 섞어도 통과하지 않는다."""

    before = payload()
    after = copy.deepcopy(before)
    after["objects"][0]["keypoint_annotations"][2]["visibility"] = 2
    after["objects"][0]["keypoint_annotations"][2]["xy"] = [1.0, 2.0]
    assert apply_module.forbidden_diff(before, after)


# ── unknown 은 쓰지 않는다 ─────────────────────────────────────────────

def test_unknown_state_is_held_not_written(apply_module):
    assert apply_module.STATES["u"] is None
    assert apply_module.STATES["v"] == (2, "visible")
    assert apply_module.STATES["o"] == (1, "occluded")
    assert apply_module.STATES["t"] == (0, "truncated")


# ── review 도구가 모델 정보를 읽지 않는다 ───────────────────────────────

# 실제 모델 사용을 가리키는 토큰만 본다.  "prediction-blinded" 는 안전 표기이므로
# 그 문구는 먼저 지우고 검사한다 — 안전 장치를 위반으로 세면 안 된다.
FORBIDDEN_IN_REVIEW = (
    "ARM_RESULTS", "model.predict", "ultralytics", "YOLO", "corner_median",
    "box_conf", "fpr95", "auroc", "R5_PROPOSED", "checkpoint", "torch",
    "paper_eval_v1/arms",
)
SAFETY_PHRASES = ("prediction-blinded", "prediction_blinded", "모델 예측")


def test_review_tool_never_touches_model_artifacts():
    source = REVIEW.read_text()
    for phrase in SAFETY_PHRASES:
        source = source.replace(phrase, " ")
    offending = [token for token in FORBIDDEN_IN_REVIEW if token in source]
    assert not offending, f"review 도구가 모델 관련 심볼을 참조한다: {offending}"


def test_review_tool_declares_the_blinded_protocol():
    """안전 표기가 실제로 있는지도 확인한다 — 위 검사가 공허해지지 않게."""

    assert "prediction-blinded" in REVIEW.read_text()


def test_review_tool_has_no_coordinate_edit_path():
    """좌표를 바꾸는 코드가 존재하지 않아야 한다 — 플래그가 아니라 구조로 막는다."""

    source = REVIEW.read_text()
    for token in ('["xy"] =', "['xy'] =", "xy\"] +=", "setMouseCallback",
                  "pose_transform\"] ="):
        assert token not in source, f"좌표/포즈 편집 경로가 있다: {token}"


def test_amendment_layer_is_separate_from_ground_truth():
    """review 도구는 GT 파일에 쓰지 않는다."""

    source = REVIEW.read_text()
    assert "AMENDMENTS" in source
    # 저장은 amendment 경로로만 간다.
    assert "save_amendments" in source
    assert "annotation_path\"]).write_text" not in source


# ── 자동 분류 반영 ──────────────────────────────────────────────────────

def test_auto_states_are_inside_the_gt_v2_enums(apply_module):
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "real_gt_v2_schema",
        REPO_ROOT / "scripts" / "annotate" / "real_gt_v2_schema.py")
    schema = _ilu.module_from_spec(spec)
    spec.loader.exec_module(schema)
    for visibility, reason in apply_module.AUTO_STATES.values():
        assert visibility in schema.VISIBILITY_VALUES
        assert reason in schema.KEYPOINT_REASONS


def test_auto_mapping_matches_the_pre_registered_policy(apply_module):
    """규정은 모델 결과를 보기 전에 고정됐다.  코드가 그걸 따라야 한다."""

    assert apply_module.AUTO_STATES == {
        "AUTO_TRUNCATED": (0, "truncated"),
        "AUTO_SELF_OCCLUDED": (1, "occluded"),
        "AUTO_CENTROID_OCCLUDED": (1, "occluded"),
        "SELF_VISIBLE_CANDIDATE": (2, "visible"),
    }


def test_external_occlusion_is_never_decided_automatically(apply_module):
    """depth 신호만으로 가림을 확정하지 않는다 — 그건 사람이 본다."""

    assert "EXTERNAL_OCCLUSION_CANDIDATE" not in apply_module.AUTO_STATES


def test_auto_never_overwrites_an_existing_decision():
    """이미 판정이 있는 점을 기하 추정으로 덮지 않는다."""

    source = APPLY.read_text()
    assert 'points[index].get("visibility") != 0' in source
    assert "auto_skipped_already_decided" in source


def test_human_review_wins_over_the_automatic_verdict():
    source = APPLY.read_text()
    assert "auto_superseded_by_human" in source
