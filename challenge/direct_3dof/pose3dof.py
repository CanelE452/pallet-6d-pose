"""3DoF pose 규약을 코드로 고정한다.  규약 근거는 ``3DOF_CONTRACT.md``.

여기 있는 함수만이 ``x / z / yaw`` 를 만들고 비교하는 공식 경로다.  모델·로더·평가가
각자 atan2 를 다시 쓰기 시작하면 규약이 조용히 갈라진다.

핵심 두 가지:

* ``yaw`` 는 배포 코드(``depth_cam/calib/geometry.py:_angles_from_R``)와 같은 식으로
  뽑는다 — ``atan2(R[0,2], R[2,2])``.  새로 만들지 않았다.
* 현장 팔레트는 4방향 포크 진입이라 90° 회전이 등가다.  그래서 yaw 를
  ``(sin 4ψ, cos 4ψ)`` 로 인코딩한다.  ``(sin ψ, cos ψ)`` 를 쓰면 등가인 네 회전이
  서로 다른 타깃을 받아 학습 신호가 상쇄된다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# 등가 회전 수.  4 = 90° 마다 같은 포즈 (4방향 진입 팔레트).
SYMMETRY_FOLD = 4
YAW_PERIOD = 2.0 * math.pi / SYMMETRY_FOLD          # π/2
_EPS = 1e-12


# ── deployment convention ────────────────────────────────────────────────────

def yaw_from_rotation(rotation) -> float:
    """회전행렬 → yaw [rad].  배포 ``_angles_from_R`` 과 같은 식.

    ``n = R @ [0,0,1] = R[:, 2]`` 이므로 ``atan2(n[0], n[2]) == atan2(R[0,2], R[2,2])``.
    """
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return float(math.atan2(r[0, 2], r[2, 2]))


def xz_from_translation(translation) -> tuple[float, float]:
    """카메라 좌표 tvec → (x, z) [m].  tvec 이 전면 중심이라 변환이 없다."""
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    return float(t[0]), float(t[2])


# ── 대칭을 반영한 yaw 표현 ────────────────────────────────────────────────────

def wrap_yaw(psi):
    """등가 회전을 접어 ``[0, π/2)`` 로 보낸다.  스칼라/배열 모두 받는다."""
    return np.mod(np.asarray(psi, dtype=np.float64), YAW_PERIOD)


def encode_yaw(psi):
    """ψ → ``(sin 4ψ, cos 4ψ)``.  등가인 네 회전이 같은 값으로 접힌다."""
    a = SYMMETRY_FOLD * np.asarray(psi, dtype=np.float64)
    return np.sin(a), np.cos(a)


def decode_yaw(sin_component, cos_component):
    """``(sin 4ψ, cos 4ψ)`` → ψ ∈ ``[0, π/2)``.

    예측 벡터는 단위 길이가 아닐 수 있으므로 atan2 를 그대로 쓴다(크기에 불변).
    """
    s = np.asarray(sin_component, dtype=np.float64)
    c = np.asarray(cos_component, dtype=np.float64)
    return wrap_yaw(np.arctan2(s, c) / SYMMETRY_FOLD)


def normalize_yaw_vector(sin_component, cos_component):
    """추론 전 ``[sin, cos]`` 를 단위 벡터로 만든다.  0 벡터는 (0, 1) 로 둔다."""
    s = np.asarray(sin_component, dtype=np.float64)
    c = np.asarray(cos_component, dtype=np.float64)
    norm = np.sqrt(s * s + c * c)
    safe = norm > _EPS
    out_s = np.where(safe, s / np.where(safe, norm, 1.0), 0.0)
    out_c = np.where(safe, c / np.where(safe, norm, 1.0), 1.0)
    return out_s, out_c


def yaw_error(predicted, target):
    """등가를 반영한 yaw 오차 [rad].  ``0 ≤ err ≤ π/4``.

    ``179° vs -179°`` 를 358° 로 세는 실수와 4-fold 등가를 무시하는 실수를 동시에 막는다.
    """
    d = np.mod(np.asarray(predicted, dtype=np.float64)
               - np.asarray(target, dtype=np.float64), YAW_PERIOD)
    return np.minimum(d, YAW_PERIOD - d)


# ── x / z 정규화 ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class XZNormalizer:
    """x/z 를 loss 가 다루기 좋은 스케일로 옮긴다.

    통계는 config·checkpoint·export metadata 세 곳에 함께 저장해야 한다.  한 곳만
    바뀌면 배포에서 조용히 어긋난다.
    """

    x_offset: float
    x_scale: float
    z_offset: float
    z_scale: float
    kind: str = "standardize"

    def __post_init__(self) -> None:
        if not (self.x_scale > 0.0 and self.z_scale > 0.0):
            raise ValueError("normalizer scale must be positive")

    def encode(self, x, z):
        return ((np.asarray(x, dtype=np.float64) - self.x_offset) / self.x_scale,
                (np.asarray(z, dtype=np.float64) - self.z_offset) / self.z_scale)

    def decode(self, x_norm, z_norm):
        return (np.asarray(x_norm, dtype=np.float64) * self.x_scale + self.x_offset,
                np.asarray(z_norm, dtype=np.float64) * self.z_scale + self.z_offset)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "x_offset": float(self.x_offset), "x_scale": float(self.x_scale),
            "z_offset": float(self.z_offset), "z_scale": float(self.z_scale),
        }

    @classmethod
    def from_dict(cls, payload) -> "XZNormalizer":
        return cls(
            x_offset=float(payload["x_offset"]), x_scale=float(payload["x_scale"]),
            z_offset=float(payload["z_offset"]), z_scale=float(payload["z_scale"]),
            kind=str(payload.get("kind", "standardize")),
        )

    @classmethod
    def from_samples(cls, x_values, z_values, *, kind: str = "standardize") -> "XZNormalizer":
        x = np.asarray(x_values, dtype=np.float64).reshape(-1)
        z = np.asarray(z_values, dtype=np.float64).reshape(-1)
        if x.size == 0 or z.size == 0:
            raise ValueError("normalizer needs samples")
        if kind != "standardize":
            raise ValueError(f"unsupported normalizer kind: {kind}")
        return cls(
            x_offset=float(x.mean()), x_scale=float(max(x.std(), 1e-6)),
            z_offset=float(z.mean()), z_scale=float(max(z.std(), 1e-6)),
            kind=kind,
        )


# ── 하나의 샘플 타깃 ─────────────────────────────────────────────────────────

def encode_target(x_m, z_m, yaw_rad, normalizer: XZNormalizer):
    """(x, z, ψ) → 모델이 배우는 4-vector ``(x_norm, z_norm, sin4ψ, cos4ψ)``."""
    x_norm, z_norm = normalizer.encode(x_m, z_m)
    sin4, cos4 = encode_yaw(yaw_rad)
    return np.stack([np.asarray(x_norm), np.asarray(z_norm),
                     np.asarray(sin4), np.asarray(cos4)], axis=-1)


def decode_prediction(vector, normalizer: XZNormalizer):
    """모델 4-vector → ``(x_m, z_m, yaw_rad)``."""
    v = np.asarray(vector, dtype=np.float64)
    x_m, z_m = normalizer.decode(v[..., 0], v[..., 1])
    sin4, cos4 = normalize_yaw_vector(v[..., 2], v[..., 3])
    return x_m, z_m, decode_yaw(sin4, cos4)
