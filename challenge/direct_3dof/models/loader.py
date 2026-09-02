"""synthetic-pretrained YOLO26n-pose 를 direct-yaw 모델로 옮겨 싣는다.

**silent partial loading 을 하지 않는다.**  무엇이 실렸고 무엇이 안 실렸는지를 전부
세어서 돌려준다.  "대충 실렸겠지" 로 넘어가면 backbone 이 랜덤인 채로 학습이 돌아가고,
그 실패를 나중에 데이터나 loss 탓으로 오진하게 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class LoadReport:
    """무엇이 재사용됐는지에 대한 감사 기록."""

    loaded_tensors: int = 0
    loaded_params: int = 0
    total_target_tensors: int = 0
    total_target_params: int = 0
    shape_mismatch: list[str] = field(default_factory=list)
    missing_in_source: list[str] = field(default_factory=list)
    unexpected_in_source: list[str] = field(default_factory=list)
    newly_initialised: list[str] = field(default_factory=list)

    @property
    def loaded_tensor_ratio(self) -> float:
        return self.loaded_tensors / max(self.total_target_tensors, 1)

    @property
    def loaded_param_ratio(self) -> float:
        return self.loaded_params / max(self.total_target_params, 1)

    def render(self) -> str:
        lines = [
            f"loaded tensors        : {self.loaded_tensors} / {self.total_target_tensors}"
            f"  ({100 * self.loaded_tensor_ratio:.1f}%)",
            f"loaded parameters     : {self.loaded_params:,} / {self.total_target_params:,}"
            f"  ({100 * self.loaded_param_ratio:.1f}%)",
            f"shape-mismatch        : {len(self.shape_mismatch)}",
            f"missing in source     : {len(self.missing_in_source)}",
            f"unexpected in source  : {len(self.unexpected_in_source)}",
            f"newly initialised     : {len(self.newly_initialised)}",
        ]
        if self.newly_initialised:
            preview = ", ".join(sorted(self.newly_initialised)[:6])
            lines.append(f"  new: {preview}"
                         + (" …" if len(self.newly_initialised) > 6 else ""))
        if self.shape_mismatch:
            preview = ", ".join(sorted(self.shape_mismatch)[:6])
            lines.append(f"  mismatch: {preview}"
                         + (" …" if len(self.shape_mismatch) > 6 else ""))
        return "\n".join(lines)


def load_pretrained_into(target: torch.nn.Module, source_state: dict) -> LoadReport:
    """이름과 shape 이 모두 맞는 텐서만 옮기고, 나머지를 전부 기록한다."""
    target_state = target.state_dict()
    report = LoadReport(
        total_target_tensors=len(target_state),
        total_target_params=sum(t.numel() for t in target_state.values()),
    )

    transferable = {}
    for name, tensor in target_state.items():
        if name not in source_state:
            report.missing_in_source.append(name)
            report.newly_initialised.append(name)
            continue
        candidate = source_state[name]
        if tuple(candidate.shape) != tuple(tensor.shape):
            report.shape_mismatch.append(name)
            report.newly_initialised.append(name)
            continue
        transferable[name] = candidate
        report.loaded_tensors += 1
        report.loaded_params += tensor.numel()

    report.unexpected_in_source = [k for k in source_state if k not in target_state]
    target.load_state_dict(transferable, strict=False)
    return report


def extract_state_dict(checkpoint_path: str | Path) -> dict:
    """ultralytics 체크포인트에서 순수 state_dict 를 꺼낸다."""
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = model.state_dict() if hasattr(model, "state_dict") else model
    return {k: v.float() if hasattr(v, "float") else v for k, v in state.items()}


def summarise_by_section(state: dict) -> dict[str, tuple[int, int]]:
    """model.<idx> 를 backbone / neck / head 로 갈라 (tensors, params) 를 센다.

    경계는 yolo26-pose.yaml 실측: backbone 0–10, neck 11–22, head 23.
    """
    sections = {"backbone": [0, 10], "neck": [11, 22], "head": [23, 23]}
    counts = {name: [0, 0] for name in sections}
    for name, tensor in state.items():
        parts = name.split(".")
        if len(parts) < 2 or parts[0] != "model" or not parts[1].isdigit():
            continue
        index = int(parts[1])
        for section, (low, high) in sections.items():
            if low <= index <= high:
                counts[section][0] += 1
                counts[section][1] += tensor.numel()
                break
    return {k: (v[0], v[1]) for k, v in counts.items()}
