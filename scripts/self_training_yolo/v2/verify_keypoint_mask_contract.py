"""visibility = 0 이 Ultralytics 8.4.60 에서 실제로 무엇을 하는지 gradient 로 잰다.

V2 전체가 이 계약 위에 선다.  "keypoint 는 무시하고 box/class supervision 은 남긴다"
가 성립하지 않으면 학습할 이유가 없다.  소스만 읽고 넘어가지 않는다 — 이 저장소에는
선언과 실제가 갈라진 이력이 있다.

pytest 는 `pallet-pose` 에만 있고 ultralytics 8.4.60 은 `pallet-yolo26` 에만 있다.
그래서 측정은 여기서 하고 **영수증(JSON)** 을 남긴다.  단언은
`challenge/tests/test_v2_keypoint_mask_contract.py` 가 그 영수증에 대고 한다.

실행: `pallet-yolo26` 환경에서, `paper_selftrain_v2` 결과 폴더를 CWD 로 두고 이 파일을
돌린다 (전역 purpose_gate 훅이 CWD 의 PURPOSE.md 를 본다).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

REPO_ROOT = Path(__file__).resolve().parents[3]
R0 = REPO_ROOT / (
    "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
    "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")
OUT = REPO_ROOT / "data/pallet/results/paper_selftrain_v2/KEYPOINT_MASK_CONTRACT.json"

N_KPT = 9
BOX_PREFIXES = ("cv2", "cv3", "one2one_cv2", "one2one_cv3")
KPT_PREFIXES = ("cv4", "cv4_kpts", "cv4_sigma",
                "one2one_cv4", "one2one_cv4_kpts", "one2one_cv4_sigma")
# 손실 주석: loss(box, kpt_location, kpt_visibility, cls, dfl[, rle])
ITEM_NAMES = ("box", "kpt_location", "kpt_visibility", "cls", "dfl", "rle")


# 손실은 `batch["keypoints"]` 를 **정규화 좌표**로 받아 imgsz 를 곱한다.  픽셀값을
# 그대로 넣으면 전부 화면 밖으로 나가 `1 - exp(-e)` 가 1 로 포화되고, 그러면 무엇을
# 흔들어도 손실이 안 변해 검사가 공허해진다 — 실제로 한 번 그렇게 "통과" 했다.
IMGSZ = 320
_SAMPLE = None


def real_sample():
    """실제 평가 프레임 하나.  예측이 GT 근처라 손실이 포화되지 않는다."""

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


def make_batch(visibility, shift=None) -> dict:
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


def measure(visibility, shift=None) -> dict:
    model = YOLO(str(R0), task="pose").model.float().train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.args = get_cfg(DEFAULT_CFG)
    model.args.epochs = 10
    model.criterion = None
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


def main() -> int:
    import ultralytics

    visible = [2.0] * N_KPT
    masked = [0.0] * N_KPT
    half = [2.0] * 4 + [0.0] * 5

    runs = {
        "all_visible": measure(visible),
        "half_visible": measure(half),
        "all_masked": measure(masked),
        # 마스크가 좌표 단위로 먹는지: 가려진 점의 GT 를 크게 흔들어도 손실이 그대로여야
        # 하고, 보이는 점을 흔들면 달라져야 한다.
        "half_visible_shift_masked_point": measure(half, shift=(8, 60.0)),
        "half_visible_shift_visible_point": measure(half, shift=(0, 60.0)),
    }

    report = {
        "schema_version": "v2_keypoint_mask_contract_v1",
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "checkpoint": str(R0.relative_to(REPO_ROOT)),
        "kpt_shape": YOLO(str(R0), task="pose").model.yaml.get("kpt_shape"),
        "item_names": list(ITEM_NAMES),
        "runs": runs,
        "findings": {
            "box_supervision_survives_full_mask":
                runs["all_masked"]["box_branch_grad"] > 0.0,
            "box_gradient_ratio_masked_over_visible":
                runs["all_masked"]["box_branch_grad"]
                / runs["all_visible"]["box_branch_grad"],
            "pose_terms_zero_when_fully_masked": (
                runs["all_masked"]["items"]["kpt_location"] == 0.0
                and runs["all_masked"]["items"]["rle"] == 0.0),
            "masked_point_coordinates_are_ignored": (
                abs(runs["half_visible_shift_masked_point"]["total_loss"]
                    - runs["half_visible"]["total_loss"]) < 1e-6),
            "visible_point_coordinates_matter": (
                abs(runs["half_visible_shift_visible_point"]["total_loss"]
                    - runs["half_visible"]["total_loss"]) > 1e-6),
            "masking_still_supervises_keypoint_objectness":
                runs["all_masked"]["keypoint_branch_grad"] > 0.0,
            "keypoint_gradient_ratio_masked_over_visible":
                runs["all_masked"]["keypoint_branch_grad"]
                / runs["all_visible"]["keypoint_branch_grad"],
        },
        "interpretation": (
            "kpt_shape[-1] == 3 이므로 masked keypoint 는 pose/RLE 항에서 빠지지만 "
            "bce_pose(pred_kpt[..., 2], kpt_mask) 를 통해 '보이지 않음' 을 적극 "
            "학습한다.  V2 의 per-keypoint mask 는 순수한 ignore 가 아니라 '이 코너는 "
            "신뢰할 수 없다' 는 negative visibility supervision 을 겸한다.  배포가 "
            "kp_conf >= 0.5 를 쓰므로 DEV 에서 kp_conf 분포를 반드시 감시한다."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"{'run':36} {'total':>10} {'box grad':>11} {'kpt grad':>11}")
    print("-" * 72)
    for name, run in runs.items():
        print(f"{name:36} {run['total_loss']:10.4f} "
              f"{run['box_branch_grad']:11.2f} {run['keypoint_branch_grad']:11.2f}")
    print()
    for key, value in report["findings"].items():
        print(f"  {key:50} {value}")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
