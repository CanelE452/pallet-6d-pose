"""annotate.py — State + JSON I/O 모듈.

class State                    : 한 프레임 라벨링 세션 상태
make_annotation(...)            : NDDS 호환 GT JSON dict 생성
save_frame_json(...)            : JSON + PNG link 저장
load_existing_annotation(...)   : 기존 라벨 JSON 로드해 State 채우기
"""
from __future__ import annotations
import json
import os
import shutil

import numpy as np

from annotate_pnp import PALLET_DIMS


class State:
    """한 프레임 라벨링 상태 — main loop 가 read/write."""
    img = None
    img_shape = None
    kps_2d = None       # length 9, each [x, y] or None
    extrap_mask = None  # length 9, bool — True = t/x 외삽 점 (PnP weight 0.3, v7)
    active = 0
    pose = None
    zoom = 1.0
    pan = [0, 0]
    dirty = False       # 미저장 변경
    last_mouse = None
    split = "eval"      # 이 프레임의 용도: "eval"(평가용) or "train". v 키로 토글, JSON 저장
    # Goto (임의 frame 점프): trackbar 클릭/드래그 + G/: 번호 입력
    goto = None         # 점프 목표 (selected 인덱스). 설정되면 main 루프가 소비
    goto_mode = False   # 번호 입력 중
    goto_buf = ""       # 입력 버퍼
    annot_only = False  # True = n/p 가 어노된 frame(out_dir JSON 존재)만 이동
    # MANIPULATE mode (6DoF pose 직접 편집)
    mode = "click"      # "click" or "manip"
    locked_pose = None  # manip 진입 시 PnP pose snapshot (dict: R, t)
    trans_step = 0.02   # m (translate step)
    rot_step_deg = 5.0  # degrees (rotate step)
    # TWO-LINE intersection sub-mode (CLICK 모드 내)
    line_mode = False
    line_pts = None     # list of [x, y] (max 4)


def make_annotation(kps_2d, pose, image_shape, K, dims=None, split="eval"):
    """NDDS 호환 JSON dict 생성.

    GT = 사용자가 클릭한 manual_kps 그대로. 안 찍은 점은 PnP projection 으로 fallback,
    그것도 image 밖이면 [-1, -1] sentinel (NDDS loader 가 invisible 처리).
    dims 는 pose 가 결정한 auto-selected 값.
    """
    if dims is None:
        dims = pose.get("dims", PALLET_DIMS) if pose else PALLET_DIMS
    h, w = image_shape[:2]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = pose["R"]
    T[:3, 3] = pose["t"]
    proj = pose["projected_all"]
    cuboid = []
    for i in range(8):
        if i < len(kps_2d) and kps_2d[i] is not None:
            cuboid.append([float(kps_2d[i][0]), float(kps_2d[i][1])])
        elif proj[i][0] >= 0:
            cuboid.append([float(proj[i][0]), float(proj[i][1])])
        else:
            cuboid.append([-1.0, -1.0])
    # Centroid: 사용자 클릭 → fallback PnP projection → fallback corners 평균
    if len(kps_2d) > 8 and kps_2d[8] is not None:
        centroid = [float(kps_2d[8][0]), float(kps_2d[8][1])]
    elif len(proj) > 8 and proj[8][0] >= 0:
        centroid = [float(proj[8][0]), float(proj[8][1])]
    else:
        valid = [c for c in cuboid if c[0] >= 0]
        centroid = [float(np.mean([c[0] for c in valid])),
                    float(np.mean([c[1] for c in valid]))] if valid else [-1.0, -1.0]
    return {
        "camera_data": {
            "width": w, "height": h,
            "intrinsics": {
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            },
        },
        "objects": [{
            "class": "pallet",
            "name": "real_pallet",
            "visibility": 1,
            "pose_transform": T.tolist(),
            "projected_cuboid": cuboid,
            "projected_cuboid_centroid": list(centroid),
            "dimensions_m": {
                "width": dims[0], "height": dims[2], "depth": dims[1],
            },
            "gt_source": "manual",
            "split": split,
            "manual_kps": [list(p) if p is not None else None for p in kps_2d],
            "reproj_error_px": pose["reproj_error_px"],
        }],
    }


def save_frame_json(out_json, out_png, src_png_path, ann):
    """JSON 저장 + PNG hardlink/copy."""
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2)
    if not os.path.exists(out_png):
        try:
            os.link(src_png_path, out_png)
        except (OSError, NotImplementedError):
            shutil.copy2(src_png_path, out_png)


def load_existing_annotation(state, out_json):
    """기존 JSON 있으면 manual_kps 를 state.kps_2d 로 로드. 없으면 noop.
    active 는 첫 None idx 로 자동 설정."""
    if not os.path.exists(out_json):
        return False
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        manual = data["objects"][0].get("manual_kps")
        state.split = data["objects"][0].get("split", "eval")
        if manual:
            state.kps_2d = [list(p) if p is not None else None for p in manual]
            state.active = next((i for i, k in enumerate(state.kps_2d) if k is None), 8)
            return True
    except Exception as e:
        print(f"[WARN] load {out_json} failed: {e}")
    return False
