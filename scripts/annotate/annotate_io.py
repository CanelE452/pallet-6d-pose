"""annotate.py — State + JSON I/O 모듈.

class State                    : 한 프레임 라벨링 세션 상태
make_annotation(...)            : NDDS 호환 GT JSON dict 생성
save_frame_json(...)            : JSON + PNG link 저장
load_existing_annotation(...)   : 기존 라벨 JSON 로드해 State 채우기
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

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
    toast = None        # (텍스트, BGR, 만료시각) — 화면 위 짧은 알림


def make_annotation(kps_2d, pose, image_shape, K, dims=None, split="eval",
                    extrap_mask=None):
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

    def _behind(p):
        """project_3d 의 sentinel 은 (-1,-1) **쌍** 이다(카메라 뒤).

        u<0 하나만 보고 버리면 화면 왼쪽으로 잘린 정상 코너(z>0, 예 u=-18.0/v=89.3)가
        "안 보임" 으로 저장된다. 정본 426장 중 44장이 이렇게 59개를 잃었다(2026-08-15).
        위쪽 밖(v<0)은 그대로 저장되고 있었으니 좌우/상하가 비대칭이기도 했다."""
        return float(p[0]) == -1.0 and float(p[1]) == -1.0

    cuboid = []
    for i in range(8):
        if i < len(kps_2d) and kps_2d[i] is not None:
            cuboid.append([float(kps_2d[i][0]), float(kps_2d[i][1])])
        elif not _behind(proj[i]):
            cuboid.append([float(proj[i][0]), float(proj[i][1])])
        else:
            cuboid.append([-1.0, -1.0])
    # Centroid: 사용자 클릭 → fallback PnP projection → fallback corners 평균
    if len(kps_2d) > 8 and kps_2d[8] is not None:
        centroid = [float(kps_2d[8][0]), float(kps_2d[8][1])]
    elif len(proj) > 8 and not _behind(proj[8]):
        centroid = [float(proj[8][0]), float(proj[8][1])]
    else:
        # 투영의 평균 != 평균의 투영. 여기 오는 건 centroid 가 카메라 뒤인 퇴화 경우뿐이라
        # 근사로 둔다. 화면 밖 코너는 위에서 살렸으므로 한쪽으로 치우치지도 않는다.
        valid = [c for c in cuboid if not (c[0] == -1.0 and c[1] == -1.0)]
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
            # 어느 점이 사람 클릭이고 어느 점이 x/t/g/c 자동채움인지. 저장하지 않으면
            # 다시 열었을 때 전부 클릭으로 취급돼 PnP 가중치가 달라지고, 파일을 열었을
            # 뿐인데 pose 가 바뀐다(비멱등). 소비처는 키 이름으로 읽으므로 추가돼도 안전.
            "extrapolated_mask": ([bool(b) for b in extrap_mask]
                                  if extrap_mask is not None else None),
            "reproj_error_px": pose["reproj_error_px"],
        }],
    }


def save_frame_json(out_json, out_png, src_png_path, ann):
    """JSON 저장 + PNG hardlink/copy.

    JSON 은 임시 파일에 쓴 뒤 os.replace 로 바꿔치기한다. 대상 파일을 열어 놓고 쓰다가
    중간에 죽으면(Ctrl+C, 디스크 부족) 반쯤 쓰인 JSON 이 남아 그 프레임의 라벨을 잃는다.
    replace 는 같은 파일시스템에서 원자적이라 실패해도 옛 내용이 그대로 남는다.
    """
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    tmp = out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_json)
    if not os.path.exists(out_png):
        try:
            os.link(src_png_path, out_png)
        except (OSError, NotImplementedError):
            shutil.copy2(src_png_path, out_png)


def load_existing_annotation(state, out_json):
    """기존 JSON 있으면 manual_kps 를 state.kps_2d 로 로드. 없으면 noop.
    active 는 첫 None idx 로 자동 설정.

    split 이 없는 JSON 을 열 때 기본값을 "eval" 로 두면, 저장하는 순간 그 프레임이
    평가셋으로 바뀐다. 옛 파일 상당수가 split 필드가 없어서 학습에서 통째로 빠질 수
    있었다. 호출자가 정한 기본값(state.split, --default_split)을 그대로 두는 쪽이 맞다.

    JSON 이 깨져 있으면 조용히 빈 화면으로 시작해 그 위에 덮어쓰게 된다. 원본을
    .corrupt 로 옮겨 두고 알린다.
    """
    if not os.path.exists(out_json):
        return False
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = data["objects"][0]
        manual = obj.get("manual_kps")
        if obj.get("split"):
            state.split = obj["split"]
        if manual:
            state.kps_2d = [list(p) if p is not None else None for p in manual]
            em = obj.get("extrapolated_mask")
            state.extrap_mask = ([bool(b) for b in em] if em and len(em) == 9
                                 else [False] * 9)
            state.active = next((i for i, k in enumerate(state.kps_2d) if k is None), 8)
            return True
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        # 진짜로 깨진 파일만 치운다. 예전엔 KeyError/TypeError(스키마가 다를 뿐인 멀쩡한
        # JSON)도 "깨짐" 으로 보고 원본을 옮겨 버려, 그 프레임이 "어노 없음" 으로 보이고
        # 새로 덮어써졌다. 같은 이름의 .corrupt 를 조용히 덮어쓰지도 않는다.
        bak = out_json + ".corrupt"
        i = 1
        while os.path.exists(bak):
            bak = f"{out_json}.corrupt.{i}"
            i += 1
        try:
            os.replace(out_json, bak)
            print(f"[WARN] {os.path.basename(out_json)} 를 읽지 못해 "
                  f"{os.path.basename(bak)} 로 옮겼다: {e}")
        except OSError:
            print(f"[WARN] load {out_json} failed: {e}")
    except Exception as e:
        # 스키마가 다른 멀쩡한 파일. 건드리지 않고 알리기만 한다 — 덮어쓰면 잃는다.
        print(f"[WARN] {os.path.basename(out_json)} 형식이 예상과 다르다 "
              f"({type(e).__name__}: {e}) — 파일은 그대로 두고 빈 상태로 시작한다. "
              f"저장하면 덮어쓰게 되니 주의.")
    return False
