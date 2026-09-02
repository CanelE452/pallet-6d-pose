"""§12 — `visibility = 0` 계약을 영수증에 대고 단언한다.

측정은 `scripts/self_training_yolo/v2/verify_keypoint_mask_contract.py` 가 한다
(ultralytics 8.4.60 이 필요해 `pallet-yolo26` 에서만 돈다).  pytest 는
`pallet-pose` 에 있으므로 여기서는 그 영수증을 검사한다.

영수증이 없으면 skip 이 아니라 **실패**다 — 계약을 확인하지 않은 채 V2 를 학습하는
일이 없어야 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RECEIPT = REPO_ROOT / "data/pallet/results/paper_selftrain_v2/KEYPOINT_MASK_CONTRACT.json"
LOCK = REPO_ROOT / "data/pallet/results/paper_selftrain_v2/SELFTRAIN_V2_METHOD_LOCK.json"

pytestmark = pytest.mark.skipif(
    not LOCK.exists(), reason="V2 트랙이 아직 없다")


@pytest.fixture(scope="module")
def receipt():
    assert RECEIPT.exists(), (
        "KEYPOINT_MASK_CONTRACT.json 이 없다.  V2 학습 전에 "
        "pallet-yolo26 에서 verify_keypoint_mask_contract 를 돌려라")
    return json.loads(RECEIPT.read_text())


def test_measured_on_the_pinned_ultralytics(receipt):
    assert receipt["ultralytics_version"] == "8.4.60"


# ── A · C · D  box supervision 은 마스크와 무관하다 ───────────────────

def test_box_supervision_survives_a_full_keypoint_mask(receipt):
    assert receipt["findings"]["box_supervision_survives_full_mask"] is True


def test_box_gradient_is_unchanged_by_masking(receipt):
    ratio = receipt["findings"]["box_gradient_ratio_masked_over_visible"]
    assert ratio == pytest.approx(1.0, abs=1e-6), (
        f"마스크가 box supervision 을 바꾼다 (비 {ratio}) — V2 의 전제가 깨진다")


def test_a_box_only_sample_is_not_dropped(receipt):
    assert receipt["runs"]["all_masked"]["total_loss"] > 0.0


# ── B  keypoint 마스크가 좌표 단위로 먹는다 ───────────────────────────

def test_pose_terms_vanish_when_every_keypoint_is_masked(receipt):
    assert receipt["findings"]["pose_terms_zero_when_fully_masked"] is True


def test_masked_point_coordinates_are_ignored(receipt):
    assert receipt["findings"]["masked_point_coordinates_are_ignored"] is True


def test_visible_point_coordinates_still_matter(receipt):
    """이게 False 면 손실이 포화된 것이고, 위 검사들이 공허해진다.

    실제로 한 번 그렇게 '통과' 했다 — 정규화 안 된 keypoint 를 넣어 전부 화면 밖으로
    나갔다.
    """

    assert receipt["findings"]["visible_point_coordinates_matter"] is True


# ── 함정: 순수한 ignore 가 아니다 ─────────────────────────────────────

def test_the_lock_records_that_masking_also_supervises_visibility(receipt):
    assert receipt["findings"]["masking_still_supervises_keypoint_objectness"] is True
    lock = json.loads(LOCK.read_text())
    constraint = lock["known_constraint_keypoint_objectness"]
    assert constraint["masking_still_supervises_keypoint_objectness"] is True
    assert constraint["kobj_gain_changed"] is False


def test_the_lock_matches_the_receipt(receipt):
    lock = json.loads(LOCK.read_text())
    constraint = lock["known_constraint_keypoint_objectness"]
    assert constraint["box_gradient_ratio_masked_over_visible"] == pytest.approx(
        receipt["findings"]["box_gradient_ratio_masked_over_visible"], abs=1e-6)
