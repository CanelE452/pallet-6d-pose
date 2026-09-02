"""direct yaw 회귀 loss.

scalar 각도 MSE 를 쓰지 않는다.  각도는 순환량이라 179° 와 -179° 가 358° 떨어진 것으로
계산되고, 게다가 이 팔레트는 90° 마다 등가라 그 실수가 네 배로 커진다.
대신 ``(sin 4ψ, cos 4ψ)`` 단위벡터 사이의 코사인 거리를 쓴다::

    L_yaw = 1 - dot(normalize(pred), gt)      # 0 (일치) ~ 2 (정반대)

여기에 예측 벡터가 단위원에서 멀어지지 않도록 하는 항을 더한다::

    L_unit = (||pred|| - 1)^2

정규화 없이 코사인만 쓰면 크기가 자유롭게 줄어들어 gradient 가 죽는 구간이 생긴다.

anchor 대응은 detection 쪽 TAL 결과를 그대로 쓴다.  ``fg_mask`` 로 positive anchor 만
고르고 ``target_gt_idx`` 로 그 anchor 가 담당하는 object 의 yaw 를 가져온다 —
keypoint loss 가 하는 방식과 같다.  이렇게 해야 detection 과 pose 의 association 이
loss 단계에서도 유지된다.
"""

from __future__ import annotations

import torch
from torch import nn

_EPS = 1e-6


def normalize_yaw_vector(vector: torch.Tensor) -> torch.Tensor:
    """마지막 축이 (sin, cos) 인 텐서를 단위벡터로.  0 근처는 안전하게 통과."""
    norm = vector.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return vector / norm


def decode_yaw(vector: torch.Tensor, fold: int = 4) -> torch.Tensor:
    """(sin nψ, cos nψ) → ψ ∈ [0, 2π/n).  평가·추론용."""
    period = 2.0 * torch.pi / fold
    angle = torch.atan2(vector[..., 0], vector[..., 1]) / fold
    return torch.remainder(angle, period)


class DirectYawLoss(nn.Module):
    """positive anchor 에서만 계산하는 순환 yaw loss.

    Args:
        unit_weight: 단위노름 정규화 항의 비중.  loss 스케일을 먼저 재고 정하라 —
            숫자를 처음부터 임의로 확정하지 않는다.
    """

    def __init__(self, unit_weight: float = 0.1) -> None:
        super().__init__()
        self.unit_weight = float(unit_weight)

    def forward(self, pred_yaw: torch.Tensor, target_yaw: torch.Tensor,
                fg_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            pred_yaw:   (B, A, 2) — anchor 마다 (sin4ψ, cos4ψ) 예측
            target_yaw: (B, A, 2) — 그 anchor 가 담당하는 object 의 인코딩된 yaw
            fg_mask:    (B, A) bool — TAL 이 고른 positive anchor

        Returns:
            ``{"yaw": …, "unit": …}`` — 각각 스칼라.  positive 가 없으면 0.
        """
        zero = pred_yaw.sum() * 0.0
        if fg_mask is None or fg_mask.sum() == 0:
            return {"yaw": zero, "unit": zero}

        selected = pred_yaw[fg_mask]                 # (N, 2)
        target = target_yaw[fg_mask]                 # (N, 2)

        unit_pred = normalize_yaw_vector(selected)
        cosine = (unit_pred * target).sum(dim=-1)
        yaw_loss = (1.0 - cosine).mean()

        magnitude = selected.norm(dim=-1)
        unit_loss = ((magnitude - 1.0) ** 2).mean()

        return {"yaw": yaw_loss, "unit": self.unit_weight * unit_loss}


def gather_anchor_targets(object_yaw: torch.Tensor, target_gt_idx: torch.Tensor,
                          fg_mask: torch.Tensor) -> torch.Tensor:
    """object 별 yaw 인코딩을 anchor 격자로 펼친다.

    Args:
        object_yaw:    (B, M, 2) — 배치의 object 마다 (sin4ψ, cos4ψ)
        target_gt_idx: (B, A) — anchor 가 담당하는 object index (TAL 산출)
        fg_mask:       (B, A) bool

    Returns:
        (B, A, 2).  negative anchor 자리는 0 이고 loss 에서 mask 로 걸러진다.
    """
    batch, anchors = target_gt_idx.shape
    index = target_gt_idx.clamp_min(0).unsqueeze(-1).expand(batch, anchors, 2)
    gathered = object_yaw.gather(1, index)
    return gathered * fg_mask.unsqueeze(-1).to(gathered.dtype)
