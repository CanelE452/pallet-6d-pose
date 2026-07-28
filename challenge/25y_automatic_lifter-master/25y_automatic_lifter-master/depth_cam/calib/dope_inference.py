# calib/dope_inference.py
# -----------------------------------------------------------------------------
# DOPE 6D pose inference adapter (challenge/scripts/run_live.py 와 동일 path).
#
# 변경 이력 (2026-05-27):
#   - 기존 _extract_peaks (채널별 단일 argmax) 제거.
#   - challenge run_live.py 와 동일하게 ObjectDetector.find_object_poses 사용
#     (affinity field 기반 corner↔centroid grouping, multi-instance 지원).
#   - CuboidPNPSolver 로 PnP 수행 (location: cm, quaternion).
#   - enforce_camera_facing (camera-z 비교 → 0~3 가까운 면 swap).
#   - Sanity gate (kp count / reproj / z range / depth-PnP / edge ratio).
#   - Temporal confirm — N 연속 통과 후 confirmed.
#   - half precision 일관성: t.to(device, dtype=ref.dtype).
#   - infer_pose(): R/t/raw_points/proj_points/info dict 반환 (HUD 시각화용).
#   - infer_keypoints9(): backward compat — 내부 infer_pose 호출 후 kps9 만 반환.
# -----------------------------------------------------------------------------
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms


# Deep_Object_Pose/common 을 sys.path 에 추가 (detector/cuboid/cuboid_pnp_solver 임포트용)
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[3]
_DOPE_COMMON = _REPO_ROOT / "Deep_Object_Pose" / "common"
if str(_DOPE_COMMON) not in sys.path:
    sys.path.insert(0, str(_DOPE_COMMON))

from detector import ModelData, ObjectDetector  # type: ignore  # noqa: E402
from cuboid import Cuboid3d  # type: ignore  # noqa: E402
from cuboid_pnp_solver import CuboidPNPSolver  # type: ignore  # noqa: E402
from pyrr import Quaternion, matrix33  # type: ignore  # noqa: E402


_IMAGENET_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def _scale_K(K: np.ndarray, s: float) -> np.ndarray:
    """K 의 fx, fy, cx, cy 모두 ×s. (resize 보정)"""
    K2 = K.copy()
    K2[0, 0] *= s; K2[1, 1] *= s; K2[0, 2] *= s; K2[1, 2] *= s
    return K2


def _sample_depth(depth_frame, x_orig: float, y_orig: float, radius: int = 3) -> Optional[float]:
    """RealSense depth_frame 의 (x, y) 주변 radius 픽셀 중앙값 (m). None 이면 sample 불가."""
    if depth_frame is None:
        return None
    try:
        fw, fh = int(depth_frame.get_width()), int(depth_frame.get_height())
    except Exception:
        return None
    vals = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = int(x_orig) + dx, int(y_orig) + dy
            if 0 <= nx < fw and 0 <= ny < fh:
                try:
                    d = float(depth_frame.get_distance(nx, ny))
                except Exception:
                    d = 0.0
                if d > 0.05:
                    vals.append(d)
    return float(np.median(vals)) if vals else None


def _enforce_camera_facing(result, pnp_solver):
    """0~3 이 카메라 가까운 면 (near face) 되도록 raw/projected_points swap.

    challenge/scripts/run_live.py 의 enforce_camera_facing 와 동일.
    학습 데이터는 object-fixed (0~3 = SDG 의 z_max corner) 로 생성됐지만 추론 단에서
    camera-facing 으로 통일. 0~3 corner 의 z_cam 평균이 4~7 보다 멀면 Ry(180°)
    swap 적용 ([5,4,7,6,1,0,3,2,8]).
    """
    loc = result.get("location")
    quat = result.get("quaternion")
    raw = result.get("raw_points")
    proj = result.get("projected_points")
    if loc is None or quat is None or pnp_solver._cuboid3d is None:
        return result

    q = Quaternion(quat)
    R = matrix33.create_from_quaternion(q)
    t = np.array(loc, dtype=np.float64)
    obj_pts = np.array(pnp_solver._cuboid3d.get_vertices()[:8], dtype=np.float64)
    pts_cam = (R @ obj_pts.T).T + t
    z_cam = pts_cam[:, 2]
    if z_cam[:4].mean() <= z_cam[4:].mean():
        return result   # 이미 0~3 이 가까이

    swap_map = [5, 4, 7, 6, 1, 0, 3, 2, 8]
    if raw is not None:
        result["raw_points"] = [raw[swap_map[i]] if swap_map[i] < len(raw) else None
                                for i in range(9)]
    if proj is not None:
        new_proj = [None] * 9
        for i in range(9):
            j = swap_map[i]
            if j < len(proj):
                new_proj[i] = proj[j]
        result["projected_points"] = new_proj
    return result


def _kp_count(raw_points) -> int:
    return sum(1 for p in raw_points if p is not None)


def _reproj_error(raw_points, proj_points) -> float:
    if proj_points is None:
        return float("inf")
    errs = []
    for r, p in zip(raw_points, proj_points):
        if r is None or p is None:
            continue
        errs.append(float(np.hypot(r[0] - p[0], r[1] - p[1])))
    return float(np.mean(errs)) if errs else float("inf")


def _edge_ratio_ok(proj_points, tol: float) -> bool:
    """proj_points 0~7 의 X/Y bbox 종횡비가 cuboid 가능 범위인가."""
    if proj_points is None:
        return True
    pts = [p for p in proj_points[:8] if p is not None]
    if len(pts) < 8:
        return True
    arr = np.array(proj_points[:8], dtype=np.float32)
    xs, ys = arr[:, 0], arr[:, 1]
    w_px = float(xs.max() - xs.min())
    h_px = float(ys.max() - ys.min())
    if w_px < 5 or h_px < 5:
        return False
    ratio_px = max(w_px, h_px) / max(min(w_px, h_px), 1.0)
    return 0.8 <= ratio_px <= (8.0 * (1.0 + tol))


def _evaluate_result(result, gates: dict, depth_cm: Optional[float]) -> Tuple[bool, str, dict]:
    """PnP 결과에 모든 gate 적용. (ok, reason, info) 반환."""
    raw = result.get("raw_points")
    if raw is None:
        return False, "no_raw_points", {}
    n_kp = _kp_count(raw)
    if n_kp < gates["min_kp"]:
        return False, f"kp={n_kp}<{gates['min_kp']}", {"n_kp": n_kp}

    loc = result.get("location")
    if loc is None:
        return False, "pnp_failed", {"n_kp": n_kp}
    z_m = float(loc[2]) / 100.0   # location 은 cm 단위
    if z_m < gates["z_min_m"] or z_m > gates["z_max_m"]:
        return False, f"z={z_m:.2f}m out of range", {"n_kp": n_kp, "z_m": z_m}

    proj = result.get("projected_points")
    reproj = _reproj_error(raw, proj)
    if reproj > gates["max_reproj_px"]:
        return False, f"reproj={reproj:.1f}px", {"n_kp": n_kp, "z_m": z_m, "reproj": reproj}

    if not _edge_ratio_ok(proj, gates["edge_ratio_tol"]):
        return False, "edge_ratio", {"n_kp": n_kp, "z_m": z_m, "reproj": reproj}

    if depth_cm is not None:
        z_cm = z_m * 100.0
        rel = abs(depth_cm - z_cm) / max(z_cm, 1e-6)
        if rel > gates["depth_rel"]:
            return False, f"depth_z_diff={rel:.2f}", {
                "n_kp": n_kp, "z_m": z_m, "reproj": reproj, "depth_rel": rel
            }
        return True, "ok", {"n_kp": n_kp, "z_m": z_m, "reproj": reproj, "depth_rel": rel}
    return True, "ok", {"n_kp": n_kp, "z_m": z_m, "reproj": reproj}


class _DopeCfg:
    """ObjectDetector.find_object_poses 가 요구하는 cfg."""
    mask_edges = 1
    mask_faces = 1
    vertex = 1
    threshold = 0.30
    softmax = 1000
    thresh_angle = 0.5
    thresh_map = 0.30
    sigma = 3
    thresh_points = 0.30


class DopePoseEstimator:
    """DOPE 6D pose estimator — ObjectDetector + CuboidPNPSolver + gate + temporal confirm.

    Note:
        - VGG-19 (DopeNetwork) 가중치만 지원. Mobilenet 변형 미지원.
        - challengenight.pth 는 학습 시 height=400 (run_live.py 와 동일 proc_scale)
          기준. input_height 인자는 호환 위해 노출하지만 내부적으로 400 으로 고정 권장.
    """

    def __init__(
        self,
        weights_path: str,
        pallet_width_m: float,
        pallet_height_m: float,
        pallet_depth_m: float,
        input_height: int = 400,
        peak_threshold: float = 0.30,
        peak_sigma: float = 3.0,
        thresh_map: float = 0.30,
        thresh_points: float = 0.30,
        thresh_angle: float = 0.5,
        softmax: int = 1000,
        gates: Optional[dict] = None,
        confirm_frames: int = 2,
        device: Optional[str] = None,
    ):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"DOPE weights not found: {weights_path}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.input_height = int(input_height)
        self.confirm_frames = int(confirm_frames)

        # gate default
        self.gates = {
            "min_kp": 7,
            "max_reproj_px": 8.0,
            "z_min_m": 0.3,
            "z_max_m": 5.0,
            "depth_rel": 0.30,
            "edge_ratio_tol": 0.30,
        }
        if gates:
            self.gates.update(gates)

        # DataParallel checkpoint 자동 판별
        sd = torch.load(weights_path, map_location="cpu")
        first_key = next(iter(sd.keys())) if isinstance(sd, dict) and len(sd) > 0 else ""
        parallel = isinstance(first_key, str) and first_key.startswith("module.")

        print(f"[DopePose] weights : {weights_path}")
        print(f"[DopePose] device  : {self.device}")
        print(f"[DopePose] parallel ckpt: {parallel}")

        if self.device.startswith("cuda"):
            self.model = ModelData(name="pallet", net_path=weights_path, parallel=parallel)
            self.model.load_net_model()
            self.net = self.model.net   # .cuda().eval()
        else:
            from detector import DopeNetwork  # type: ignore
            from collections import OrderedDict
            net = DopeNetwork()
            state_dict = sd
            if parallel:
                new_sd = OrderedDict()
                for k, v in state_dict.items():
                    new_sd[k[7:]] = v
                state_dict = new_sd
            net.load_state_dict(state_dict)
            net.eval()
            self.net = net
            self.model = None

        # Half precision (FP16) — GPU 일 때만. config.USE_HALF=True 가 default.
        # FP16 추론은 RTX 3060 Laptop 같은 모바일 GPU 에서 FP32 대비 5~10x 빠름.
        try:
            from calib.config import USE_HALF as _USE_HALF
        except ImportError:
            _USE_HALF = True
        if self.device.startswith("cuda") and _USE_HALF:
            self.net.half()
            print(f"[DopePose] half precision : True (FP16)")
        else:
            print(f"[DopePose] half precision : False (FP32, device={self.device})")

        # Warmup forward — 첫 추론의 lazy compile / cuDNN benchmark 비용을 init 으로 분산.
        # 두 번째 frame 부터 정상 추론 속도.
        try:
            ref_param = next(self.net.parameters())
            dummy_h = int(input_height)
            dummy_w = (int(input_height * 4 / 3) // 8) * 8   # 4:3 비율 대략 (RealSense 640x480 → 533)
            dummy = torch.zeros(
                1, 3, dummy_h, dummy_w,
                device=ref_param.device, dtype=ref_param.dtype,
            )
            with torch.no_grad():
                _ = self.net(dummy)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            print(f"[DopePose] warmup done ({dummy_h}x{dummy_w})")
        except Exception as e:
            print(f"[DopePose] warmup skipped: {e}")

        # CuboidPNPSolver — Cuboid3d size 는 cm 단위 (location 도 cm 반환)
        dim_cm = [pallet_width_m * 100.0, pallet_height_m * 100.0, pallet_depth_m * 100.0]
        self.pnp_solver = CuboidPNPSolver("pallet", cuboid3d=Cuboid3d(dim_cm))
        self.pnp_solver.set_dist_coeffs(np.zeros((4, 1), dtype=np.float64))

        # ObjectDetector cfg
        self.cfg = _DopeCfg()
        self.cfg.threshold = float(peak_threshold)
        self.cfg.thresh_map = float(thresh_map)
        self.cfg.thresh_points = float(thresh_points)
        self.cfg.thresh_angle = float(thresh_angle)
        self.cfg.sigma = int(round(peak_sigma))
        self.cfg.softmax = int(softmax)

        # temporal confirm 상태
        self.consecutive_ok = 0
        self.last_proc_scale: float = 1.0
        self.last_reason: str = "init"

    def _forward(self, img_rgb_np: np.ndarray):
        """RGB(H,W,3,uint8) → (vertex2, aff) — 마지막 stage tensor.

        net 이 FP16 (half precision) 일 때 forward 는 FP16 으로 수행하되,
        반환 tensor 는 FP32 로 cast (ObjectDetector 의 numpy 처리가 fp16 미지원).
        """
        t = _IMAGENET_TF(img_rgb_np)
        ref = next(self.net.parameters())
        t = t.to(device=ref.device, dtype=ref.dtype).unsqueeze(0)
        with torch.no_grad():
            out, seg = self.net(t)
        return out[-1][0].float(), seg[-1][0].float()

    # ------------------------------------------------------------------ public
    def infer_pose(
        self,
        bgr: np.ndarray,
        camera_matrix: np.ndarray,
        depth_frame=None,
    ) -> Optional[dict]:
        """BGR image + 원본 K (+ depth_frame) → 6D pose dict 또는 None.

        Returns:
            None — 미검출 또는 게이트 실패 (self.last_reason 에 사유 기록)
            dict:
                "R"          : np.ndarray (3,3)  pallet local axes in camera frame
                "t_m"        : np.ndarray (3,)   미터 단위 location
                "raw_points_orig"     : list[(u,v) or None]  9 점 (원본 좌표)
                "proj_points_orig"    : list[(u,v) or None]  PnP reproj (원본 좌표)
                "kps9_orig"  : list[(u,v)]  v4 backward-compat (NaN 가능)
                "K_small"    : np.ndarray (3,3)  resize 적용된 K
                "proc_scale" : float
                "consecutive_ok" : int
                "confirmed"  : bool   N 연속 통과 여부
                "info"       : dict   n_kp / z_m / reproj
                "yaw_deg_cam": float  atan2(R[0,2], R[2,2]) (degrees, raw — 180° wrap 적용 전)
        """
        if bgr is None or bgr.ndim != 3 or camera_matrix is None:
            self.last_reason = "bad_input"
            return None
        h, w = bgr.shape[:2]
        if h <= 0 or w <= 0:
            self.last_reason = "bad_shape"
            return None

        # height=input_height 로 resize (VGG stride 8 호환 width 정렬)
        proc_scale = float(self.input_height) / float(h)
        new_w = int(round(w * proc_scale)) & ~7
        new_w = max(new_w, 8)
        img_small = cv2.resize(bgr, (new_w, self.input_height))
        img_rgb = img_small[..., ::-1].copy()
        self.last_proc_scale = proc_scale

        K_orig = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        K_small = _scale_K(K_orig, proc_scale)
        self.pnp_solver.set_camera_intrinsic_matrix(K_small)

        vertex2, aff = self._forward(img_rgb)

        try:
            results = ObjectDetector.find_object_poses(vertex2, aff, self.pnp_solver, self.cfg)
        except Exception as e:
            print(f"[DopePose] find_object_poses error: {e}")
            results = []

        # gate 평가 — 첫 통과 result 채택. gate 실패해도 첫 result 는 시각화용 보존.
        best = None
        best_info = None
        last_reason = "no_result"
        first_result = None
        first_info = None
        for result in results:
            result = _enforce_camera_facing(result, self.pnp_solver)
            depth_cm = None
            raw = result.get("raw_points")
            if depth_frame is not None and raw is not None and raw[8] is not None:
                d_m = _sample_depth(
                    depth_frame,
                    raw[8][0] / proc_scale,
                    raw[8][1] / proc_scale,
                )
                if d_m is not None:
                    depth_cm = d_m * 100.0
            ok, reason, info = _evaluate_result(result, self.gates, depth_cm)
            last_reason = reason
            if first_result is None:
                first_result = result
                first_info = dict(info)
                first_info["fail_reason"] = reason if not ok else "ok"
            if ok:
                best = result
                best_info = info
                break

        if best is None:
            self.consecutive_ok = 0
            self.last_reason = last_reason
            # gate 실패 — raw_points 가 있으면 시각화용으로 반환 (gate_passed=False)
            if first_result is not None and first_result.get("raw_points") is not None:
                return self._build_pose_dict(
                    first_result, K_small, proc_scale,
                    consecutive_ok=0, confirmed=False, gate_passed=False,
                    info=first_info or {"fail_reason": last_reason},
                )
            return None

        self.consecutive_ok += 1
        confirmed = self.consecutive_ok >= self.confirm_frames
        self.last_reason = "ok" if confirmed else f"pending {self.consecutive_ok}/{self.confirm_frames}"
        return self._build_pose_dict(
            best, K_small, proc_scale,
            consecutive_ok=int(self.consecutive_ok),
            confirmed=bool(confirmed), gate_passed=True,
            info=best_info or {},
        )

    @staticmethod
    def _build_pose_dict(result, K_small, proc_scale, consecutive_ok, confirmed, gate_passed, info):
        """공통 dict 빌더 — gate pass / fail 분기에서 동일 형식 반환."""
        inv_s = 1.0 / max(proc_scale, 1e-9)
        # `or []` 사용 금지 — raw_points/projected_points 가 numpy array 일 경우
        # truth value ambiguity 발생. 명시적 None 검사로 안전 처리.
        raw = result.get("raw_points")
        if raw is None:
            raw = []
        proj = result.get("projected_points")
        if proj is None:
            proj = []
        raw_orig = [None if p is None else (float(p[0]) * inv_s, float(p[1]) * inv_s) for p in raw]
        proj_orig = [None if p is None else (float(p[0]) * inv_s, float(p[1]) * inv_s) for p in proj]
        while len(raw_orig) < 9:
            raw_orig.append(None)
        kps9_orig = [
            (float("nan"), float("nan")) if p is None else (p[0], p[1])
            for p in raw_orig
        ]

        # R / t (m) — PnP 성공한 경우만. gate 실패라도 PnP 가 성공하면 R/t 있음.
        loc = result.get("location")
        quat = result.get("quaternion")
        R = None
        t_m = None
        yaw_deg = None
        if loc is not None and quat is not None:
            q = Quaternion(quat)
            R = np.array(matrix33.create_from_quaternion(q), dtype=np.float64)
            t_m = np.asarray(loc, dtype=np.float64) / 100.0
            yaw_deg = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))

        return {
            "R": R,
            "t_m": t_m,
            "raw_points_orig": raw_orig,
            "proj_points_orig": proj_orig,
            "kps9_orig": kps9_orig,
            "K_small": K_small,
            "proc_scale": proc_scale,
            "consecutive_ok": consecutive_ok,
            "confirmed": confirmed,
            "gate_passed": gate_passed,
            "info": info,
            "yaw_deg_cam": yaw_deg,
        }

    def infer_keypoints9(self, bgr: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """Backward-compat — kps9 (원본 좌표, NaN 가능) 만 반환.

        Note: 새 코드는 infer_pose() 를 직접 호출하고 R/t/HUD 정보 사용 권장.
        K 가 필요해 camera_matrix=identity 가정 — PnP 결과가 부정확하므로 main_rec
        은 반드시 infer_pose 를 쓰고 이 함수는 시각화 용도 외에 사용하지 말 것.
        """
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = float(self.input_height)
        K[0, 2] = float(bgr.shape[1]) / 2.0
        K[1, 2] = float(bgr.shape[0]) / 2.0
        out = self.infer_pose(bgr, K, depth_frame=None)
        if out is None:
            return None
        return out["kps9_orig"]
