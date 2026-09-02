"""synthetic-pretrained YOLO26n-pose 체크포인트 → direct-yaw 모델.

``Pose26`` head 를 ``PoseDirectYaw26`` 으로 갈아끼우고 나머지(backbone·neck·기존 head
가중치)를 그대로 옮긴다.  yaml 을 새로 만들지 않고 로드된 모듈을 교체하는 이유는,
체크포인트가 들고 있는 stride/anchor 같은 런타임 상태를 그대로 승계하기 위해서다.
yaml 경로로 다시 만들면 그 상태를 손으로 재현해야 하고 거기서 어긋나기 쉽다.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

MODELS = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELS))

from head import PoseDirectYaw26  # noqa: E402
from loader import LoadReport, load_pretrained_into  # noqa: E402


def _head_input_channels(head) -> tuple[int, ...]:
    """head 가 받는 P3/P4/P5 채널 수를 cv4 trunk 의 첫 conv 에서 읽는다."""
    channels = []
    for branch in head.cv4:
        first = branch[0]
        conv = getattr(first, "conv", first)
        channels.append(int(conv.in_channels))
    return tuple(channels)


def _copy_runtime_state(source, target) -> None:
    """stride/anchor 등 forward 에 필요한 비학습 상태를 승계한다."""
    for attribute in ("stride", "nc", "no", "reg_max", "end2end", "max_det",
                      "shape", "anchors", "strides", "export", "format",
                      "xyxy", "legacy", "kpt_shape", "nk", "nk_sigma"):
        if hasattr(source, attribute):
            setattr(target, attribute, getattr(source, attribute))


def build_direct_yaw_model(checkpoint_path: str | Path,
                           *, verbose: bool = True):
    """체크포인트를 읽어 direct-yaw 모델과 적재 리포트를 돌려준다.

    Returns:
        ``(model, report)`` — model 은 ultralytics ``PoseModel`` 이고 마지막 모듈만
        ``PoseDirectYaw26`` 으로 교체돼 있다.
    """
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model = model.float()

    old_head = model.model[-1]
    channels = _head_input_channels(old_head)
    new_head = PoseDirectYaw26(
        nc=int(old_head.nc),
        kpt_shape=tuple(old_head.kpt_shape),
        reg_max=int(getattr(old_head, "reg_max", 1)),
        end2end=bool(getattr(old_head, "end2end", False)),
        ch=channels,
    )
    _copy_runtime_state(old_head, new_head)

    report = load_pretrained_into(new_head, old_head.state_dict())
    new_head.i, new_head.f, new_head.type = old_head.i, old_head.f, old_head.type
    model.model[-1] = new_head
    if hasattr(model, "yaml") and isinstance(model.yaml, dict):
        model.yaml = dict(model.yaml)

    if verbose:
        print("── head 교체 리포트 (Pose26 → PoseDirectYaw26) ──")
        print(report.render())
    return model, report


def count_parameters(module) -> tuple[int, int]:
    """(tensors, params)."""
    state = module.state_dict()
    return len(state), sum(t.numel() for t in state.values())
