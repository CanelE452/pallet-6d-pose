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

import copy
import json
import os
import shutil

import numpy as np

from annotate_pnp import PALLET_DIMS

try:
    from pallet_geometry import (
        AxisAssignment,
        camera_facing_to_canonical_pose,
        canonical_dimensions,
        canonical_to_camera_facing_keypoint_permutation,
        canonical_to_camera_facing_transform,
    )
except ImportError:  # geometry module is optional for legacy standalone use.
    AxisAssignment = None
    camera_facing_to_canonical_pose = None
    canonical_dimensions = None
    canonical_to_camera_facing_keypoint_permutation = None
    canonical_to_camera_facing_transform = None

try:
    from real_gt_v2_schema import validate_gt_v2
except ImportError:  # Legacy standalone imports may not ship the v2 validator.
    validate_gt_v2 = None


GT_V2_SCHEMA_VERSION = "real_pallet_gt_v2"
KEYPOINT_FRAME = "camera_dynamic_0123_v4"
KEYPOINT_SOURCES = {
    "manual_click", "extrapolated", "pnp_projected", "centroid_auto", "unknown",
}
KEYPOINT_REASONS = {"visible", "occluded", "truncated", "unknown"}


class State:
    """한 프레임 라벨링 상태 — main loop 가 read/write."""
    img = None
    img_shape = None
    kps_2d = None       # length 9, each [x, y] or None
    extrap_mask = None  # length 9, bool — True = t/x 외삽 점 (PnP weight 0.3, v7)
    keypoint_annotations = None  # GT v2 length-9 visibility/source/reason entries
    axis_assignment = None       # signed YAW_0/90/180/270; human confirmation required
    axis_assignment_candidates = None
    axis_assignment_confirmed = False
    population_role = "DEV"
    capture_metadata = None
    # Full snapshots of the document/object loaded from disk.  They are kept
    # separate from the editable v2 state so an old label can be saved into a
    # new v2 namespace without rewriting or dropping any compatibility field.
    legacy_document = None
    legacy_object = None
    camera_facing_hypothesis_override = None
    occlusion_level = "unknown"
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


def _axis_name(value):
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _axis_object(value):
    name = _axis_name(value)
    if name is None or AxisAssignment is None:
        return name
    if isinstance(value, AxisAssignment):
        return value
    return AxisAssignment(name)


def _canonical_xyz_dict():
    if canonical_dimensions is None:
        return {"x": 1.1, "y": 0.11, "z": 1.3}
    dims = canonical_dimensions()
    return {
        "x": float(dims.x_m),
        "y": float(dims.y_m),
        "z": float(dims.z_m),
    }


def _fallback_axis_transform(name):
    angles = {"YAW_0": 0.0, "YAW_90": 90.0, "YAW_180": 180.0, "YAW_270": 270.0}
    if name not in angles:
        raise ValueError(f"unknown axis assignment: {name!r}")
    a = np.deg2rad(angles[name])
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _canonical_pose_record(R_cf, t_cf, axis_assignment):
    name = _axis_name(axis_assignment)
    if name is None:
        raise ValueError("a signed axis assignment is required")

    axis = _axis_object(axis_assignment)
    if camera_facing_to_canonical_pose is not None:
        R_can, t_can = camera_facing_to_canonical_pose(R_cf, t_cf, axis)
        A = canonical_to_camera_facing_transform(axis)
        perm = canonical_to_camera_facing_keypoint_permutation(axis)
    else:
        A = _fallback_axis_transform(name)
        R_can, t_can = np.asarray(R_cf) @ A, np.asarray(t_cf)
        perm = tuple(range(9))
    T_can = np.eye(4, dtype=np.float64)
    T_can[:3, :3] = np.asarray(R_can, dtype=np.float64)
    T_can[:3, 3] = np.asarray(t_can, dtype=np.float64).reshape(3)
    return {
        "axis_assignment": name,
        "pose_transform": T_can.tolist(),
        "canonical_to_camera_facing_rotation": np.asarray(A).tolist(),
        "canonical_to_camera_facing_keypoint_permutation": [int(v) for v in perm],
    }


def _canonical_pose_payload(R_cf, t_cf, axis_assignment, confirmed, candidates):
    if not confirmed or axis_assignment is None:
        return None
    return _canonical_pose_record(R_cf, t_cf, axis_assignment)


def _normalise_axis_candidates(values, dims):
    names = []
    for value in values or []:
        name = _axis_name(value)
        if name in {"YAW_0", "YAW_90", "YAW_180", "YAW_270"} \
                and name not in names:
            names.append(name)
    if not names:
        # Backward-compatible callers do not know about signed axes.  Preserve
        # both signs for the selected W/D parity; never fabricate one sign.
        if len(dims) >= 2 and float(dims[0]) > float(dims[1]):
            names = ["YAW_90", "YAW_270"]
        else:
            names = ["YAW_0", "YAW_180"]
    return names


def _point_in_frame(point, width, height):
    return bool(point is not None and len(point) >= 2
                and 0.0 <= float(point[0]) < float(width)
                and 0.0 <= float(point[1]) < float(height))


def _normalise_keypoint_annotations(kps_2d, projected, width, height,
                                    keypoint_annotations=None, extrap_mask=None):
    """Return the explicit v2 per-keypoint contract without guessing old visibility."""
    supplied = keypoint_annotations if isinstance(keypoint_annotations, list) else []
    result = []
    for i in range(9):
        base = copy.deepcopy(supplied[i]) if i < len(supplied) and isinstance(supplied[i], dict) else {}
        manual_xy = kps_2d[i] if i < len(kps_2d) else None
        fallback_xy = projected[i] if i < len(projected) else None
        is_behind = (fallback_xy is None or
                     (float(fallback_xy[0]) == -1.0 and float(fallback_xy[1]) == -1.0))
        if base:
            # Explicit v2 state wins.  In particular, a new visibility=0
            # point remains xy=null instead of silently acquiring a PnP label.
            xy = manual_xy if manual_xy is not None else base.get("xy")
        else:
            xy = manual_xy if manual_xy is not None else (None if is_behind else fallback_xy)
        if xy is not None:
            xy = [float(xy[0]), float(xy[1])]

        extrapolated = bool(extrap_mask is not None and i < len(extrap_mask)
                            and extrap_mask[i])
        if base:
            visibility = int(base.get("visibility", 0))
            source = str(base.get("source", "unknown"))
            reason = str(base.get("reason", "unknown"))
        elif manual_xy is None and xy is not None:
            visibility, source, reason = 1, "pnp_projected", "unknown"
        elif extrapolated:
            visibility = 1
            source = "centroid_auto" if i == 8 else "extrapolated"
            reason = "unknown"
        elif manual_xy is not None:
            visibility, source, reason = 2, "manual_click", "visible"
        else:
            visibility, source, reason = 0, "unknown", "unknown"
        if visibility not in (0, 1, 2):
            visibility = 0
        if source not in KEYPOINT_SOURCES:
            source = "unknown"
        if reason not in KEYPOINT_REASONS:
            reason = "unknown"
        result.append({
            "xy": xy,
            "visibility": visibility,
            "in_frame": _point_in_frame(xy, width, height),
            "source": source,
            "reason": reason,
        })
    return result


def _truncation_payload(cuboid, width, height):
    outside = [i for i, point in enumerate(cuboid)
               if not _point_in_frame(point, width, height)]
    finite = [point for point in cuboid
              if point is not None and not (point[0] == -1.0 and point[1] == -1.0)]
    fraction = None
    if finite:
        x0, x1 = min(p[0] for p in finite), max(p[0] for p in finite)
        y0, y1 = min(p[1] for p in finite), max(p[1] for p in finite)
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        ix0, ix1 = max(0.0, x0), min(float(width), x1)
        iy0, iy1 = max(0.0, y0), min(float(height), y1)
        inside = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        fraction = float(1.0 - inside / area) if area > 0 else None
    return {
        "is_truncated": bool(outside),
        "outside_keypoints": outside,
        "bbox_outside_fraction": fraction,
    }


def make_annotation(kps_2d, pose, image_shape, K, dims=None, split="eval",
                    extrap_mask=None, keypoint_annotations=None,
                    axis_assignment=None, axis_assignment_candidates=None,
                    axis_assignment_confirmed=False, legacy_object=None,
                    legacy_document=None, population_role="DEV", metadata=None,
                    occlusion_level="unknown"):
    """NDDS 호환 JSON dict 생성.

    GT = 사용자가 클릭한 manual_kps 그대로. 안 찍은 점은 PnP projection 으로 fallback,
    그것도 image 밖이면 [-1, -1] sentinel (NDDS loader 가 invisible 처리).
    dims 는 pose 가 결정한 auto-selected 값.
    """
    population_role = str(population_role).upper()
    if population_role not in {"DEV", "FINAL"}:
        raise ValueError("population_role must be DEV or FINAL")
    if dims is None:
        dims = pose.get("dims", PALLET_DIMS) if pose else PALLET_DIMS
    dims = tuple(float(v) for v in dims)
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
    candidates = _normalise_axis_candidates(
        axis_assignment_candidates or
        pose.get("_axis_assignment_candidates", []) or [], dims)
    if axis_assignment is None and axis_assignment_confirmed:
        axis_assignment = pose.get("_axis_assignment")
    axis_name = _axis_name(axis_assignment) if axis_assignment_confirmed else None
    if axis_name is not None and axis_name not in candidates:
        raise ValueError(
            f"confirmed axis {axis_name} is not in PnP candidates {candidates}")
    kp_ann = _normalise_keypoint_annotations(
        kps_2d, proj, w, h,
        keypoint_annotations=keypoint_annotations,
        extrap_mask=extrap_mask,
    )
    original_object = (
        copy.deepcopy(legacy_object) if isinstance(legacy_object, dict) else {})
    # A v2 file already has a dedicated legacy payload.  A pre-v2 file does
    # not, so seed the payload from its original camera-facing compatibility
    # fields.  In both cases setdefault is additive and never changes a value
    # that came from disk.
    legacy = copy.deepcopy(original_object.get("legacy") or {})
    legacy.setdefault("dimensions_m", copy.deepcopy(original_object.get(
        "dimensions_m", {
            "width": dims[0], "height": dims[2], "depth": dims[1],
        })))
    legacy.setdefault("pose_transform", copy.deepcopy(
        original_object.get("pose_transform", T.tolist())))
    legacy.setdefault("fix_swap", copy.deepcopy(
        original_object.get("fix_swap")))
    candidate_summaries = []
    for item in pose.get("_wd_candidates", []) or []:
        summary = dict(item)
        if "dims" in summary:
            summary["dims"] = [float(v) for v in summary["dims"]]
        candidate_summaries.append(summary)
    obj = {
        "class": "pallet",
        "name": "real_pallet",
        "visibility": 1,
        # Legacy fields remain camera-facing and are intentionally not paper-facing truth.
        "pose_transform": T.tolist(),
        "projected_cuboid": cuboid,
        "projected_cuboid_centroid": list(centroid),
        "dimensions_m": {
            "width": dims[0], "height": dims[2], "depth": dims[1],
        },
        "gt_source": "manual",
        "split": split,
        "manual_kps": [list(p) if p is not None else None for p in kps_2d],
        "extrapolated_mask": ([bool(b) for b in extrap_mask]
                              if extrap_mask is not None else None),
        "reproj_error_px": float(pose["reproj_error_px"]),
        "keypoint_frame": KEYPOINT_FRAME,
        "physical_dimensions_m": _canonical_xyz_dict(),
        "camera_facing_pnp": {
            "axis_assignment": axis_name,
            "axis_assignment_candidates": [_axis_name(v) for v in candidates],
            "axis_assignment_confirmed": bool(axis_assignment_confirmed),
            "dimensions_m": {
                "width": dims[0], "height": dims[2], "depth": dims[1],
            },
            "pose_transform": T.tolist(),
            "selected_hypothesis": pose.get("_camera_facing_hypothesis"),
            "selection_reason": pose.get("_wd_selection_reason", "legacy_geometry_rank"),
            "hypotheses": candidate_summaries,
        },
        "canonical_pose": _canonical_pose_payload(
            pose["R"], pose["t"], axis_assignment,
            bool(axis_assignment_confirmed), candidates),
        "canonical_pose_candidates": [
            _canonical_pose_record(pose["R"], pose["t"], value)
            for value in candidates
        ],
        "pose_status": ("CANONICAL_POSE_CONFIRMED" if axis_assignment_confirmed
                        else "UNCONFIRMED_SIGNED_AXIS"),
        "migration_status": ("CANONICAL_POSE_CONFIRMED" if axis_assignment_confirmed
                             else "MANUAL_REVIEW_REQUIRED"),
        "legacy": legacy,
        "keypoint_annotations": kp_ann,
        "occlusion_level": (occlusion_level if occlusion_level in
                            {"none", "partial", "heavy", "unknown"} else "unknown"),
        "truncation": _truncation_payload(cuboid, w, h),
    }
    # GT v2 is additive.  Every pre-existing object-level compatibility field
    # (including less common fields such as tag_id and sentinel_repaired) must
    # survive old -> load -> save byte-for-byte at the JSON-value level.  Only
    # the explicit v2 fields below are regenerated from the editor state.
    generated_v2_fields = {
        "physical_dimensions_m", "camera_facing_pnp", "canonical_pose",
        "canonical_pose_candidates", "pose_status", "migration_status",
        "legacy", "keypoint_annotations", "occlusion_level", "truncation",
    }
    for key, value in original_object.items():
        if key not in generated_v2_fields:
            obj[key] = copy.deepcopy(value)

    generated_camera_data = {
        "width": w, "height": h,
        "intrinsics": {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        },
    }
    # Preserve root-level compatibility/custom fields as well.  ``objects``
    # and the two v2 contract fields are the only deliberate replacements.
    result = (
        copy.deepcopy(legacy_document)
        if isinstance(legacy_document, dict) else {})
    result["schema_version"] = GT_V2_SCHEMA_VERSION
    result["population_role"] = population_role
    result["camera_data"] = copy.deepcopy(
        result.get("camera_data", generated_camera_data))
    result["objects"] = [obj]
    for key, value in (metadata or {}).items():
        if key in {"capture_session_id", "camera_serial", "capture_timestamp",
                   "lighting_condition"} and value is not None:
            result[key] = value
    return result


def save_frame_json(out_json, out_png, src_png_path, ann):
    """JSON 저장 + PNG hardlink/copy.

    JSON 은 임시 파일에 쓴 뒤 os.replace 로 바꿔치기한다. 대상 파일을 열어 놓고 쓰다가
    중간에 죽으면(Ctrl+C, 디스크 부족) 반쯤 쓰인 JSON 이 남아 그 프레임의 라벨을 잃는다.
    replace 는 같은 파일시스템에서 원자적이라 실패해도 옛 내용이 그대로 남는다.
    """
    if ann.get("schema_version") == GT_V2_SCHEMA_VERSION and validate_gt_v2 is not None:
        validate_gt_v2(ann)
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


def load_existing_annotation(state, out_json, *, read_only=False):
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
        kp_ann = obj.get("keypoint_annotations")
        is_v2 = data.get("schema_version") == GT_V2_SCHEMA_VERSION
        if (isinstance(kp_ann, list) and len(kp_ann) == 9
                and all(isinstance(entry, dict) for entry in kp_ann)):
            state.keypoint_annotations = copy.deepcopy(kp_ann)
            if is_v2 or not manual:
                # GT-v2 edits live here.  Its top-level ``manual_kps`` is a
                # frozen legacy compatibility field and must not overwrite a
                # reviewed v2 coordinate when the file is opened again.
                manual = [entry.get("xy") for entry in kp_ann]
        if obj.get("split"):
            state.split = obj["split"]
        if manual:
            state.kps_2d = [list(p) if p is not None else None for p in manual]
            if is_v2 and state.keypoint_annotations is not None:
                state.extrap_mask = [
                    entry.get("source") in {
                        "extrapolated", "pnp_projected", "centroid_auto"}
                    for entry in state.keypoint_annotations
                ]
            else:
                em = obj.get("extrapolated_mask")
                state.extrap_mask = ([bool(b) for b in em] if em and len(em) == 9
                                     else [False] * 9)
            if state.keypoint_annotations is None:
                camera = data.get("camera_data") or {}
                state.keypoint_annotations = _normalise_keypoint_annotations(
                    state.kps_2d,
                    list(obj.get("projected_cuboid") or [])
                    + [obj.get("projected_cuboid_centroid")],
                    int(camera.get("width") or 0), int(camera.get("height") or 0),
                    keypoint_annotations=[{
                        "xy": (list(p) if p is not None else None),
                        "visibility": 0,
                        "source": ("extrapolated" if state.extrap_mask[i] else "unknown"),
                        "reason": "unknown",
                    } for i, p in enumerate(state.kps_2d)],
                    extrap_mask=state.extrap_mask,
                )
            cf = obj.get("camera_facing_pnp") or {}
            state.axis_assignment = cf.get("axis_assignment")
            state.axis_assignment_candidates = list(
                cf.get("axis_assignment_candidates") or [])
            state.axis_assignment_confirmed = bool(
                cf.get("axis_assignment_confirmed", state.axis_assignment is not None))
            state.population_role = str(data.get("population_role", "DEV")).upper()
            state.capture_metadata = {
                key: data.get(key) for key in (
                    "capture_session_id", "camera_serial", "capture_timestamp",
                    "lighting_condition") if data.get(key) is not None
            }
            state.legacy_document = copy.deepcopy(data)
            state.legacy_object = copy.deepcopy(obj)
            state.camera_facing_hypothesis_override = (
                cf.get("selected_hypothesis")
                if cf.get("selection_reason") == "manual_camera_facing_override"
                else None)
            state.occlusion_level = obj.get("occlusion_level", "unknown")
            state.active = next((i for i, k in enumerate(state.kps_2d) if k is None), 8)
            return True
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        # 진짜로 깨진 파일만 치운다. 예전엔 KeyError/TypeError(스키마가 다를 뿐인 멀쩡한
        # JSON)도 "깨짐" 으로 보고 원본을 옮겨 버려, 그 프레임이 "어노 없음" 으로 보이고
        # 새로 덮어써졌다. 같은 이름의 .corrupt 를 조용히 덮어쓰지도 않는다.
        if read_only:
            print(f"[WARN] read-only label {out_json} is unreadable ({e}); "
                  "source file was not moved or changed")
            return False
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
        suffix = ("read-only source remains unchanged."
                  if read_only else "저장하면 덮어쓰게 되니 주의.")
        print(f"[WARN] {os.path.basename(out_json)} 형식이 예상과 다르다 "
              f"({type(e).__name__}: {e}) — 파일은 그대로 두고 빈 상태로 시작한다. "
              f"{suffix}")
    return False
