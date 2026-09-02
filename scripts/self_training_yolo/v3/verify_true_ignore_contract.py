"""§9 — true-ignore 가 실제로 gradient 를 0 으로 만드는지 실측한다.

이 계약이 통과하기 전에는 smoke 도 돌리지 않는다.  V2 에서 "ignore" 라고 믿었던 것이
실은 negative supervision 이었던 전례가 있다 — 소스를 읽는 것으로는 부족하다.

실행: `pallet-yolo26` 환경에서, `paper_selftrain_v3` 결과 폴더를 CWD 로 두고 돌린다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from true_ignore_pose_loss import make_criterion  # noqa: E402

R0 = REPO_ROOT / (
    "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
    "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")
OUT = REPO_ROOT / "data/pallet/results/paper_selftrain_v3/TRUE_IGNORE_LOSS_CONTRACT.json"

N_KPT = 9
IMGSZ = 320
BOX_PREFIXES = ("cv2", "cv3", "one2one_cv2", "one2one_cv3")
KPT_PREFIXES = ("cv4", "cv4_kpts", "cv4_sigma",
                "one2one_cv4", "one2one_cv4_kpts", "one2one_cv4_sigma")
ITEM_NAMES = ("box", "kpt_location", "kpt_visibility", "cls", "dfl", "rle")
TOLERANCE = 1e-6

_SAMPLE = None


def real_sample():
    import cv2

    manifest = REPO_ROOT / "challenge/real_gt_v2/manifests/PAPER_EVAL_PLASTIC_POS.json"
    item = json.loads(manifest.read_text())["items"][0]
    image = cv2.imread(str(REPO_ROOT / item["image_path"]))
    if image is None:
        raise SystemExit(f"UNREADABLE_IMAGE: {item['image_path']}")
    height, width = image.shape[:2]
    payload = json.loads((REPO_ROOT / item["gt_v2_path"]).read_text())
    points = payload["objects"][0]["keypoint_annotations"]
    xy = torch.tensor([point["xy"] for point in points], dtype=torch.float32)
    normalised = torch.stack([xy[:, 0] / width, xy[:, 1] / height], dim=1)
    corners = xy[:8]
    x0, y0 = corners.min(0).values
    x1, y1 = corners.max(0).values
    box = torch.tensor([[((x0 + x1) / 2) / width, ((y0 + y1) / 2) / height,
                         (x1 - x0) / width, (y1 - y0) / height]])
    resized = cv2.resize(image, (IMGSZ, IMGSZ))
    tensor = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1)[None]
    return tensor.float() / 255.0, normalised, box


def make_batch(visibility, shift=None):
    global _SAMPLE
    if _SAMPLE is None:
        _SAMPLE = real_sample()
    image, normalised, box = _SAMPLE
    keypoints = torch.cat([normalised, torch.zeros(N_KPT, 1)], dim=1)[None].clone()
    keypoints[0, :, 2] = torch.tensor(visibility, dtype=torch.float32)
    if shift is not None:
        index, delta_px = shift
        keypoints[0, index, :2] += delta_px / IMGSZ
    return {
        "img": image.clone(),
        "batch_idx": torch.zeros(1),
        "cls": torch.zeros(1, 1),
        "bboxes": box.clone(),
        "keypoints": keypoints,
    }


def build(true_ignore: bool):
    model = YOLO(str(R0), task="pose").model.float().train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.args = get_cfg(DEFAULT_CFG)
    model.args.epochs = 10
    model.criterion = make_criterion(model) if true_ignore else None
    return model


def measure(visibility, true_ignore=True, shift=None, logit_scale=None):
    model = build(true_ignore)
    if logit_scale is not None:
        # ignored keypoint 의 objectness logit 을 크게 흔든다.  head 의 마지막
        # keypoint 분기 bias 를 건드려 예측 자체를 바꾼다.
        head = list(model.model)[-1]
        with torch.no_grad():
            for name, parameter in head.named_parameters():
                if name.startswith("one2one_cv4_sigma") or name.startswith("cv4_sigma"):
                    parameter.add_(logit_scale)
    model.zero_grad(set_to_none=True)
    loss, items = model.loss(make_batch(visibility, shift))
    loss.sum().backward()

    head = list(model.model)[-1]
    box = keypoint = 0.0
    for name, parameter in head.named_parameters():
        if parameter.grad is None:
            continue
        magnitude = float(parameter.grad.abs().sum())
        prefix = name.split(".")[0]
        if prefix in KPT_PREFIXES:
            keypoint += magnitude
        elif prefix in BOX_PREFIXES:
            box += magnitude
    return {
        "total_loss": float(loss.sum()),
        "items": dict(zip(ITEM_NAMES, [float(v) for v in items])),
        "box_branch_grad": box,
        "keypoint_branch_grad": keypoint,
    }


def close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE * max(1.0, abs(a), abs(b))


def main() -> int:
    import ultralytics

    all_supervised = [2.0] * N_KPT
    all_ignored = [1.0] * N_KPT
    one_ignored = [2.0] * 8 + [1.0]
    half = [2.0] * 4 + [1.0] * 5
    stock_invisible = [2.0] * 4 + [0.0] * 5

    runs = {
        "A_all_supervised": measure(all_supervised),
        "B_one_ignored": measure(one_ignored),
        "C_all_ignored": measure(all_ignored),
        "half_ignored": measure(half),
        "half_ignored_shift_ignored_point": measure(half, shift=(8, 60.0)),
        "half_ignored_shift_supervised_point": measure(half, shift=(0, 60.0)),
        # 진단용으로만 남긴다.  `cv4_sigma` 는 supervised 점의 RLE 에도 들어가므로
        # 이 교란은 "ignored 점의 objectness 만" 을 격리하지 못한다 — 증거로 쓰지
        # 않는다.  kobj gradient 가 ignored 점에 닿지 않는다는 결정적 근거는
        # `all_ignored_gives_zero_keypoint_gradient` 다 (정확히 0).
        "half_ignored_logit_perturbed_DIAGNOSTIC_ONLY": measure(half, logit_scale=3.0),
        # synthetic parity: sentinel 이 하나도 없는 배치는 stock 과 같아야 한다.
        "synthetic_stock": measure(stock_invisible, true_ignore=False),
        "synthetic_true_ignore_loss": measure(stock_invisible, true_ignore=True),
    }

    parity = runs["synthetic_stock"], runs["synthetic_true_ignore_loss"]
    findings = {
        # BOX
        "box_loss_identical_all_supervised_vs_all_ignored": close(
            runs["A_all_supervised"]["items"]["box"],
            runs["C_all_ignored"]["items"]["box"]),
        "box_gradient_ratio_ignored_over_supervised": (
            runs["C_all_ignored"]["box_branch_grad"]
            / runs["A_all_supervised"]["box_branch_grad"]),
        # LOCATION
        "ignored_point_coordinates_do_not_move_the_loss": close(
            runs["half_ignored"]["total_loss"],
            runs["half_ignored_shift_ignored_point"]["total_loss"]),
        "supervised_point_coordinates_do_move_the_loss": not close(
            runs["half_ignored"]["total_loss"],
            runs["half_ignored_shift_supervised_point"]["total_loss"]),
        # KOBJ
        "all_ignored_kills_every_keypoint_term": (
            runs["C_all_ignored"]["items"]["kpt_location"] == 0.0
            and runs["C_all_ignored"]["items"]["kpt_visibility"] == 0.0
            and runs["C_all_ignored"]["items"]["rle"] == 0.0),
        "all_ignored_gives_zero_keypoint_gradient": (
            runs["C_all_ignored"]["keypoint_branch_grad"] == 0.0),
        # SYNTHETIC PARITY
        "synthetic_parity_total_loss": close(
            parity[0]["total_loss"], parity[1]["total_loss"]),
        "synthetic_parity_items": all(
            close(parity[0]["items"][key], parity[1]["items"][key])
            for key in ITEM_NAMES),
        "synthetic_parity_box_gradient": close(
            parity[0]["box_branch_grad"], parity[1]["box_branch_grad"]),
        "synthetic_parity_keypoint_gradient": close(
            parity[0]["keypoint_branch_grad"], parity[1]["keypoint_branch_grad"]),
    }
    # V2 대비: stock 에서 전부 masked 이면 keypoint gradient 가 남았는데 (V2 함정),
    # true-ignore 에서는 0 이어야 한다.
    stock_all_masked = measure([0.0] * N_KPT, true_ignore=False)
    runs["stock_all_masked_for_reference"] = stock_all_masked
    findings["stock_all_masked_still_has_keypoint_gradient"] = (
        stock_all_masked["keypoint_branch_grad"] > 0.0)

    status = "PASS" if all(
        value is True for key, value in findings.items()
        if key not in ("box_gradient_ratio_ignored_over_supervised",)
    ) and close(findings["box_gradient_ratio_ignored_over_supervised"], 1.0) else "FAIL"

    report = {
        "schema_version": "v3_true_ignore_loss_contract_v1",
        "status": status,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "tolerance": TOLERANCE,
        "sentinel": {"supervised": 2, "true_ignore": 1, "invisible": 0},
        "item_names": list(ITEM_NAMES),
        "runs": runs,
        "findings": findings,
        "interpretation": (
            "visibility == 1 인 keypoint 는 location · RLE · keypoint objectness 어느 "
            "항에도 들어가지 않으며 keypoint branch gradient 를 만들지 않는다.  "
            "box/cls/dfl 은 영향을 받지 않는다.  sentinel 이 없는 배치는 stock 과 "
            "수치적으로 동일하다."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"{'run':38} {'total':>10} {'box grad':>11} {'kpt grad':>11}")
    print("-" * 74)
    for name, run in runs.items():
        print(f"{name:38} {run['total_loss']:10.4f} "
              f"{run['box_branch_grad']:11.2f} {run['keypoint_branch_grad']:11.2f}")
    print()
    for key, value in findings.items():
        print(f"  {key:56} {value}")
    print(f"\nCONTRACT {status}")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
