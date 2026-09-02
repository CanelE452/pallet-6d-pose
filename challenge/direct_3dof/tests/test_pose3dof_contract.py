"""3DoF 규약 sanity — 모델 코드를 쓰기 전에 좌표계/대칭이 맞는지부터 고정한다.

여기가 깨진 채로 학습 scaffold 를 진행하면 규약 mismatch 를 나중에 데이터 탓으로
오진하게 된다.  ``3DOF_CONTRACT.md`` 의 각 문장에 대응하는 테스트를 둔다.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

TRACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK))

from pose3dof import (  # noqa: E402
    SYMMETRY_FOLD,
    YAW_PERIOD,
    XZNormalizer,
    decode_prediction,
    decode_yaw,
    encode_target,
    encode_yaw,
    normalize_yaw_vector,
    wrap_yaw,
    xz_from_translation,
    yaw_error,
    yaw_from_rotation,
)


def _rotation_about_y(angle: float) -> np.ndarray:
    """카메라 +Y 는 아래를 향한다.  이 축 둘레 회전이 곧 팔레트 yaw 다."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=np.float64)


# ── 1~2. deployment convention 그대로인가 ────────────────────────────────────

def test_yaw_matches_deployment_formula() -> None:
    """배포 _angles_from_R 은 atan2(R[0,2], R[2,2]) 와 같아야 한다."""
    for angle in (-1.2, -0.3, 0.0, 0.4, 1.1):
        r = _rotation_about_y(angle)
        expected = math.atan2(r[0, 2], r[2, 2])
        assert yaw_from_rotation(r) == pytest.approx(expected, abs=1e-12)


def test_identity_rotation_is_yaw_zero() -> None:
    """정면 pose → yaw ≈ 0."""
    assert yaw_from_rotation(np.eye(3)) == pytest.approx(0.0, abs=1e-12)


# ── 3~5. x / z 부호와 단위 ───────────────────────────────────────────────────

def test_pallet_shifted_right_gives_positive_x() -> None:
    x, _ = xz_from_translation([0.42, 0.0, 3.0])
    assert x > 0


def test_pallet_shifted_left_gives_negative_x() -> None:
    x, _ = xz_from_translation([-0.42, 0.0, 3.0])
    assert x < 0


def test_further_pallet_gives_larger_z() -> None:
    _, near = xz_from_translation([0.0, 0.0, 1.5])
    _, far = xz_from_translation([0.0, 0.0, 4.5])
    assert far > near
    assert (near, far) == (1.5, 4.5)          # metre 그대로, 변환 없음


# ── 6~9. 4-fold 대칭 ────────────────────────────────────────────────────────

def test_symmetry_fold_is_four() -> None:
    assert SYMMETRY_FOLD == 4
    assert YAW_PERIOD == pytest.approx(math.pi / 2)


@pytest.mark.parametrize("base", [0.0, 0.17, 0.6, 1.2])
def test_equivalent_rotations_encode_identically(base: float) -> None:
    """90° 씩 돌린 네 포즈는 물리적으로 같으므로 타깃도 같아야 한다.

    (sin ψ, cos ψ) 였다면 여기서 서로 다른 값이 나와 학습 신호가 상쇄된다.
    """
    reference = encode_yaw(base)
    for k in range(1, SYMMETRY_FOLD):
        rotated = encode_yaw(base + k * YAW_PERIOD)
        assert rotated[0] == pytest.approx(reference[0], abs=1e-9)
        assert rotated[1] == pytest.approx(reference[1], abs=1e-9)


def test_encoding_separates_non_equivalent_angles() -> None:
    """등가가 아닌 각도까지 접어버리면 안 된다 (0° 와 45° 는 달라야 한다)."""
    a = np.asarray(encode_yaw(0.0))
    b = np.asarray(encode_yaw(math.radians(45.0)))
    assert np.linalg.norm(a - b) > 0.5


def test_wrap_maps_into_canonical_range() -> None:
    values = wrap_yaw([-3.0, -0.1, 0.0, 0.3, 5.0])
    assert np.all(values >= 0.0)
    assert np.all(values < YAW_PERIOD)


# ── 10~11. sin/cos 연속성 — 경계에서 튀지 않는가 ─────────────────────────────

def test_encoding_is_continuous_across_the_period_boundary() -> None:
    """등가 경계(89.9° vs 90.1°)에서 타깃이 튀면 회귀가 그 지점을 못 배운다."""
    just_below = np.asarray(encode_yaw(math.radians(89.9)))
    just_above = np.asarray(encode_yaw(math.radians(90.1)))
    assert np.linalg.norm(just_below - just_above) < 0.05


def test_error_across_the_boundary_is_small_not_huge() -> None:
    """89.9° 와 90.1° 의 오차는 0.2° 여야지 180° 가 아니다."""
    err = yaw_error(math.radians(89.9), math.radians(90.1))
    assert math.degrees(err) == pytest.approx(0.2, abs=1e-6)


def test_yaw_error_is_bounded_by_quarter_period() -> None:
    rng = np.random.default_rng(0)
    a = rng.uniform(-10.0, 10.0, size=512)
    b = rng.uniform(-10.0, 10.0, size=512)
    err = yaw_error(a, b)
    assert np.all(err >= 0.0)
    assert np.all(err <= YAW_PERIOD / 2 + 1e-12)


# ── 12~14. round-trip ───────────────────────────────────────────────────────

def test_yaw_encode_decode_round_trip() -> None:
    angles = np.linspace(0.0, YAW_PERIOD, 37, endpoint=False)
    recovered = decode_yaw(*encode_yaw(angles))
    assert np.max(yaw_error(recovered, angles)) < 1e-9


def test_decode_is_invariant_to_prediction_magnitude() -> None:
    """예측 벡터가 단위 길이가 아니어도 각도는 같아야 한다."""
    s, c = encode_yaw(0.37)
    assert decode_yaw(s * 7.3, c * 7.3) == pytest.approx(decode_yaw(s, c), abs=1e-12)


def test_normalize_yaw_vector_handles_degenerate_zero() -> None:
    s, c = normalize_yaw_vector(0.0, 0.0)
    assert np.isfinite(s) and np.isfinite(c)
    assert (float(s), float(c)) == (0.0, 1.0)


def test_translation_unit_round_trip_through_normalizer() -> None:
    norm = XZNormalizer(x_offset=0.05, x_scale=0.4, z_offset=3.0, z_scale=1.2)
    x, z = 0.37, 4.15
    back_x, back_z = norm.decode(*norm.encode(x, z))
    assert float(back_x) == pytest.approx(x, abs=1e-12)
    assert float(back_z) == pytest.approx(z, abs=1e-12)


def test_full_target_round_trip() -> None:
    norm = XZNormalizer.from_samples([-0.5, 0.0, 0.5], [1.5, 3.0, 4.5])
    x, z, psi = -0.21, 2.6, math.radians(37.0)
    vector = encode_target(x, z, psi, norm)
    assert vector.shape[-1] == 4
    got_x, got_z, got_psi = decode_prediction(vector, norm)
    assert float(got_x) == pytest.approx(x, abs=1e-9)
    assert float(got_z) == pytest.approx(z, abs=1e-9)
    assert float(yaw_error(got_psi, psi)) < 1e-9


def test_normalizer_serialises_round_trip() -> None:
    """통계는 config/checkpoint/export 세 곳에 같은 값으로 실려야 한다."""
    norm = XZNormalizer.from_samples([-0.4, 0.1, 0.6], [1.0, 2.0, 5.0])
    assert XZNormalizer.from_dict(norm.to_dict()) == norm


def test_normalizer_rejects_zero_scale() -> None:
    with pytest.raises(ValueError):
        XZNormalizer(x_offset=0.0, x_scale=0.0, z_offset=0.0, z_scale=1.0)
