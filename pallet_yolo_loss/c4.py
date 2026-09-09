#!/usr/bin/env python3
"""정사각 팔레트 전용 C4 (yaw 90도) symmetry-aware pose loss.

기존 ``symmetry.py`` 의 A1/ASC 는 건드리지 않는다.  저 실험들의 재현성을 깨지 않도록
새 class 로 opt-in 한다 — 환경변수 ``C4_CONFIG`` 가 없으면 이 loss 는
``PoseLoss26`` 과 **완전히 같은 경로**를 탄다.

## 무엇을 바꾸는가

keypoint **target 의 동치류**만 바꾼다.  box / cls / dfl / anchor assignment 는
stock 그대로다.

정사각 팔레트는 yaw 90도 회전이 물리적으로 같은 물체라, 어느 면을 앞면(0~3)으로
부를지가 임의다.  indexed loss 는 위치를 정확히 맞춘 예측에도 "번호가 90도 돌았다"는
이유로 큰 loss 를 준다.  그래서 네 개의 cuboid permutation 중 loss 가 최소인 것을
target 으로 삼는다::

    Lsym = min over g in {0, 90, 180, 270} of L(pred, GT o perm_g)

## 하지 않는 것

* pred[i] -> 가장 가까운 GT point (금지)
* Hungarian(pred, GT) (금지)
* keypoint 마다 독립적인 permutation (금지)

선택 단위는 **object 하나**다.  같은 GT object 를 담당하는 positive anchor 들은
모두 같은 branch 를 쓴다.  좌표와 visibility 는 같은 permutation 으로 함께 움직인다
(텐서 마지막 축 전체를 재배열하므로 구조적으로 어긋날 수 없다).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os

import torch

from ultralytics.utils.loss import PoseLoss26
from ultralytics.utils.ops import xyxy2xywh

from .symmetry import per_instance_kpt_loss

# C4_PERMUTATIONS.json 에서 기하로 유도한 값.  여기 하드코딩된 것은 기본값일 뿐이고,
# 실제 실행에서는 config 파일이 준 값을 쓰며 아래 검증을 통과해야만 활성화된다.
DEFAULT_PERMUTATIONS = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8),
    (1, 5, 6, 2, 0, 4, 7, 3, 8),
    (5, 4, 7, 6, 1, 0, 3, 2, 8),
    (4, 0, 3, 7, 5, 1, 2, 6, 8),
)
CUBOID_EDGES = frozenset(map(frozenset, [
    (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]))


def validate_permutations(perms) -> list[str]:
    """cuboid 구조를 보존하는지 검사한다.  실패 목록을 돌려준다.

    감으로 적은 순열이 조용히 학습에 들어가는 것을 막는다.  centroid 가 움직이거나
    edge 가 깨지면 그건 회전이 아니라 물체를 다른 것으로 바꾸는 변환이다.
    """
    fails = []
    for k, p in enumerate(perms):
        if len(p) != 9:
            fails.append(f"perm[{k}]: 길이가 9가 아니다 ({len(p)})")
            continue
        if sorted(p[:8]) != list(range(8)):
            fails.append(f"perm[{k}]: 0~7 bijection 아님")
        if p[8] != 8:
            fails.append(f"perm[{k}]: centroid 가 움직임 (perm[8]={p[8]})")
        if {frozenset({p[a], p[b]}) for a, b in map(tuple, CUBOID_EDGES)} != CUBOID_EDGES:
            fails.append(f"perm[{k}]: cuboid 12-edge 보존 안 됨")
    if perms and tuple(perms[0]) != tuple(range(9)):
        fails.append("perm[0] 이 identity 가 아니다")
    return fails


@dataclass
class C4Config:
    """``C4_CONFIG`` 가 가리키는 JSON 으로 켠다.  없으면 stock 동작."""

    enabled: bool = False
    permutations: tuple = field(default_factory=lambda: DEFAULT_PERMUTATIONS)
    # 진단용 — branch 선택 히스토그램을 남길 파일
    stats_path: str = ""

    @classmethod
    def from_env(cls) -> "C4Config":
        path = os.environ.get("C4_CONFIG")
        if not path or not os.path.exists(path):
            return cls()
        d = json.load(open(path, encoding="utf-8"))
        perms = tuple(tuple(p) for p in d.get("permutations", DEFAULT_PERMUTATIONS))
        cfg = cls(enabled=bool(d.get("enabled", False)), permutations=perms,
                  stats_path=str(d.get("stats_path", "")))
        if cfg.enabled:
            bad = validate_permutations(cfg.permutations)
            if bad:
                raise ValueError("C4 permutation 검증 실패: " + "; ".join(bad))
        return cfg


class ChallengeC4PoseLoss(PoseLoss26):
    """object 단위로 C4 branch 를 골라 keypoint target 을 재배열한다."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.c4 = C4Config.from_env()
        self._perm_t = None
        # branch 히스토그램 — smoke 에서 한쪽으로 쏠리면 구현 버그를 의심한다.
        self.c4_branch_hist = [0, 0, 0, 0]

    def _perms(self, device) -> torch.Tensor:
        if self._perm_t is None or self._perm_t.device != device:
            self._perm_t = torch.as_tensor(self.c4.permutations, dtype=torch.long,
                                           device=device)
        return self._perm_t

    @staticmethod
    def _object_slot(batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        """GT 행 i 가 자기 이미지 안에서 몇 번째 object 인지.

        ``_select_target_keypoints`` 가 쓰는 것과 같은 규칙(이미지별 등장 순서)이라야
        ``target_gt_idx`` 와 맞물린다.
        """
        b = batch_idx.flatten().long()
        counts = torch.bincount(b, minlength=batch_size)
        offset = torch.cumsum(counts, 0) - counts
        return torch.arange(len(b), device=b.device) - offset[b]

    def _choose_branches(self, masks, target_gt_idx, keypoints, batch_idx,
                         stride_tensor, target_bboxes, pred_kpts):
        """object 마다 loss 가 최소인 permutation index 를 고른다.

        선택에는 gradient 가 필요 없다.  ``no_grad`` 로 네 branch 를 재고 argmin 만
        가져온다.
        """
        device = keypoints.device
        batch_size = masks.shape[0]
        perms = self._perms(device)
        n_branch = perms.shape[0]

        b_idx = masks.nonzero(as_tuple=True)[0]              # (N_pos,)
        g_idx = target_gt_idx[masks]                          # (N_pos,)
        max_gt = int(target_gt_idx.max().item()) + 1 if target_gt_idx.numel() else 1
        obj_key = b_idx * max_gt + g_idx                      # positive anchor -> object
        n_obj_key = batch_size * max_gt

        # target_bboxes 는 상위에서 in-place 로 나눠지므로 여기서는 복사본을 쓴다.
        boxes = (target_bboxes / stride_tensor)[masks]
        area = xyxy2xywh(boxes)[:, 2:].prod(1, keepdim=True)
        pred = pred_kpts[masks]

        cost = torch.zeros(n_branch, n_obj_key, device=device)
        for k in range(n_branch):
            kp = keypoints[:, perms[k], :]                    # 좌표+visibility 동시 재배열
            sel = self._select_target_keypoints(kp, batch_idx, target_gt_idx, masks)
            sel = sel.clone()
            sel[..., :2] /= stride_tensor.view(1, -1, 1, 1)
            gt = sel[masks]
            m = gt[..., 2] != 0 if gt.shape[-1] == 3 else torch.full_like(gt[..., 0], True)
            per_anchor = per_instance_kpt_loss(self.keypoint_loss, pred, gt, m, area)
            cost[k].scatter_add_(0, obj_key, per_anchor)

        best = cost.argmin(0)                                 # (n_obj_key,)
        for k in range(n_branch):
            self.c4_branch_hist[k] += int((best[obj_key] == k).sum().item())
        return best, max_gt

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                                 stride_tensor, target_bboxes, pred_kpts):
        """C4 가 꺼져 있으면 stock 과 한 글자도 다르지 않은 경로를 탄다."""
        if not self.c4.enabled or not masks.any():
            return super().calculate_keypoints_loss(
                masks, target_gt_idx, keypoints, batch_idx, stride_tensor,
                target_bboxes, pred_kpts)

        with torch.no_grad():
            best, max_gt = self._choose_branches(
                masks, target_gt_idx, keypoints, batch_idx, stride_tensor,
                target_bboxes, pred_kpts)
            # 고른 branch 를 GT 행에 매핑한다.
            slot = self._object_slot(batch_idx, masks.shape[0])
            row_key = batch_idx.flatten().long() * max_gt + slot
            row_branch = best[row_key.clamp(max=best.numel() - 1)]
            perms = self._perms(keypoints.device)
            gather = perms[row_branch]                        # (N_gt_rows, 9)

        keypoints = torch.gather(
            keypoints, 1, gather.unsqueeze(-1).expand(-1, -1, keypoints.shape[-1]))
        return super().calculate_keypoints_loss(
            masks, target_gt_idx, keypoints, batch_idx, stride_tensor,
            target_bboxes, pred_kpts)
