# calib/yolo_inference.py
# -----------------------------------------------------------------------------
# YOLO26-pose 6D pose inference adapter — DopePoseEstimator 와 동일 인터페이스.
#
# main_rec.py / pose6d_adapter.py 가 DOPE 와 똑같이 쓸 수 있도록 infer_pose() 가
# 동일 시그니처/반환 dict 를 갖는다. FSM 배선은 main_rec.py 에서 estimator 만
# DopePoseEstimator → YoloPoseEstimator 로 바꾸면 그대로 동작.
#
# 검증된 YOLO 추론 로직 복제 (절대 변경 금지 항목):
#   - 추론 : eval_ab_crop.predict / pallet_jetson_deploy/infer_fps.predict
#            (100px reflect pad → ultralytics .predict → keypoint (-pad) shift,
#             box conf 최대 instance 선택)
#   - 9 kp : annotate camera-facing (0~3 near/front, 4~7 far/rear, 8 centroid).
#            절대 재배열 금지.
#   - PnP  : eval_ab_crop.solve_pnp = SQPnP (cv2.SOLVEPNP_SQPNP + RefineLM),
#            conf>=kp_conf 점만, n>=6, median reproj>12px reject.
#   - 3D   : annotate_pnp.make_pallet_keypoints_3d(W, D, H) (인자순서 width,depth,height).
#
# 좌표 convention: OpenCV (+X right, +Y down, +Z forward). 절대 변경 금지.
#
# ★ R/t convention — DOPE 와 일치시키기 위한 R_fix (가장 중요) ----------------
#   FSM 의 pose6d_adapter.pose6d_to_align_vars 는 R[:,2] (= pallet local +Z 축의
#   camera frame 표현) 를 "pallet forward" 로 보고, 카메라가 정면을 볼 때 두 +Z 가
#   마주봐 atan2 가 ±180° → 내부 +180° wrap 으로 ψ≈0 을 만든다. 이 컨벤션은 DOPE
#   Cuboid3d (front face = +Z = +depth/2) 기준으로 튜닝됐다.
#
#   YOLO 의 make_pallet_keypoints_3d 는 0~3(near/front) 을 -Z(-depth/2), 4~7(far)
#   을 +Z 에 둔다. 즉 YOLO local +Z 는 "far(rear)" 방향 → DOPE 의 +Z(front) 와 정확히
#   반대. 두 모델 모두 X=width(right), Y=height(down), Z=depth(forward) 로 axis triad
#   는 동일하나, +Z polarity 가 반대(=Y축 180° 회전 차이) 다.
#
#   실측 검증 (forklift_raw_20260528_163408/gt_manual, GT pose_transform 은
#   make_pallet_keypoints_3d 컨벤션):
#     - raw YOLO R 직접 입력 → 정면 프레임에서 ψ≈±179° (틀림)
#     - R_dope = R_yolo @ Ry(180°) 적용 → 정면 ψ≈0° (000000:-0.7°, 000030:+0.1°),
#       회전 시 ψ 가 매끄럽게 ±증가, d_lat/d_fwd 물리적으로 타당.
#   따라서 R_FIX = Ry(180°) 를 R_yolo 의 오른쪽에 곱해 (model frame 을 회전)
#   DOPE convention 으로 맞춘다.  DOPE 를 기준으로 YOLO 를 맞춘 것.
#
#   t 는 둘 다 centroid(=local origin) 위치이므로 변환 불필요. R_fix(Y축 회전) 는
#   centroid 를 옮기지 않으므로 t 에 영향 없음.
#
# weights:
#   .pt / .onnx / .engine 모두 ultralytics.YOLO 가 자동 처리.
#   ★ .engine(TensorRT) 사용 시 LD_LIBRARY_PATH 에 torch/lib + tensorrt_libs 필요:
#     export LD_LIBRARY_PATH=$ENV/lib/python3.10/site-packages/torch/lib:\
#       $ENV/lib/python3.10/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
# -----------------------------------------------------------------------------
from __future__ import annotations
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


# 검증된 3D 모델 (annotate_pnp.make_pallet_keypoints_3d 그대로 복제 — repo 모듈
# import 의존성 없이 self-contained). 인자순서 = (width, depth, height).
def _make_pallet_keypoints_3d(width: float, depth: float, height: float) -> np.ndarray:
    """Camera-facing convention 9-keypoint 3D 모델.

    cuboid local frame: X=right(+), Y=down(OpenCV +y=bottom), Z=forward(+).
      0: near-top-LEFT     (-w/2, -h/2, -d/2)   near = Z_local 작은 쪽
      1: near-top-RIGHT    (+w/2, -h/2, -d/2)
      2: near-bottom-RIGHT (+w/2, +h/2, -d/2)
      3: near-bottom-LEFT  (-w/2, +h/2, -d/2)
      4: far-top-LEFT      (-w/2, -h/2, +d/2)   far = Z_local 큰 쪽
      5: far-top-RIGHT     (+w/2, -h/2, +d/2)
      6: far-bottom-RIGHT  (+w/2, +h/2, +d/2)
      7: far-bottom-LEFT   (-w/2, +h/2, +d/2)
      8: centroid
    """
    w, h, d = width / 2.0, height / 2.0, depth / 2.0
    corners = np.array([
        [-w, -h, -d],   # 0 near-top-LEFT
        [+w, -h, -d],   # 1 near-top-RIGHT
        [+w, +h, -d],   # 2 near-bottom-RIGHT
        [-w, +h, -d],   # 3 near-bottom-LEFT
        [-w, -h, +d],   # 4 far-top-LEFT
        [+w, -h, +d],   # 5 far-top-RIGHT
        [+w, +h, +d],   # 6 far-bottom-RIGHT
        [-w, +h, +d],   # 7 far-bottom-LEFT
    ], dtype=np.float64)
    centroid = corners.mean(axis=0, keepdims=True)
    return np.vstack([corners, centroid])


# R_FIX — YOLO local frame → DOPE local frame (Y축 180° 회전).
#   X→-X, Z→-Z, Y 유지. R_dope = R_yolo @ R_FIX.
_R_FIX = np.array([
    [-1.0, 0.0,  0.0],
    [0.0,  1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=np.float64)


SQPNP_MAX_MED_REPROJ = 12.0   # px. SQPnP median reproj 이보다 크면 PnP 실패 처리


def _sample_depth(depth_frame, x_orig: float, y_orig: float, radius: int = 3) -> Optional[float]:
    """RealSense depth_frame 의 (x, y) 주변 radius 픽셀 중앙값 (m). None=sample 불가.

    dope_inference._sample_depth 와 동일.
    """
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
    """proj_points 0~7 의 X/Y bbox 종횡비가 cuboid 가능 범위인가.

    dope_inference._edge_ratio_ok 와 동일.
    """
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


def _evaluate_result(raw, proj, R, t_m, gates: dict, depth_cm: Optional[float]
                     ) -> Tuple[bool, str, dict]:
    """PnP 결과에 모든 gate 적용 — dope_inference._evaluate_result 와 동일 게이트."""
    if raw is None:
        return False, "no_raw_points", {}
    n_kp = _kp_count(raw)
    if n_kp < gates["min_kp"]:
        return False, f"kp={n_kp}<{gates['min_kp']}", {"n_kp": n_kp}

    if R is None or t_m is None:
        return False, "pnp_failed", {"n_kp": n_kp}
    z_m = float(t_m[2])
    if z_m < gates["z_min_m"] or z_m > gates["z_max_m"]:
        return False, f"z={z_m:.2f}m out of range", {"n_kp": n_kp, "z_m": z_m}

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


class YoloPoseEstimator:
    """YOLO26-pose 6D pose estimator — DopePoseEstimator 와 동일 위치/역할.

    YOLO keypoint(9) → SQPnP → R/t (DOPE convention 으로 R_fix 적용) → gate →
    temporal confirm. infer_pose() 시그니처/반환 dict 는 DopePoseEstimator 와 동일.

    Note:
        - peak_threshold / peak_sigma / thresh_* / softmax 등 DOPE 전용 인자는
          호환 위해 받되 무시한다 (ultralytics 는 자체 conf 사용).
        - weights 는 .pt / .onnx / .engine 모두 ultralytics 가 처리.
          .engine 사용 시 LD_LIBRARY_PATH 주의 (모듈 docstring 참조).
    """

    def __init__(
        self,
        weights_path: str,
        pallet_width_m: float,
        pallet_height_m: float,
        pallet_depth_m: float,
        # --- YOLO 추론 파라미터 ---
        det_conf: float = 0.4,        # ultralytics detection conf
        kp_conf: float = 0.5,         # keypoint vis thr (PnP / 그리기)
        imgsz: int = 640,
        pad: int = 100,               # reflect pad 폭 (A 모델 학습과 동일)
        # --- DOPE 호환 인자 (무시) ---
        input_height: int = 400,
        peak_threshold: float = 0.30,
        peak_sigma: float = 3.0,
        thresh_map: float = 0.30,
        thresh_points: float = 0.30,
        thresh_angle: float = 0.5,
        softmax: int = 1000,
        # --- gate / confirm ---
        gates: Optional[dict] = None,
        confirm_frames: int = 2,
        device: Optional[str] = None,
    ):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

        self.confirm_frames = int(confirm_frames)
        self.det_conf = float(det_conf)
        self.kp_conf = float(kp_conf)
        self.imgsz = int(imgsz)
        self.pad = int(pad)

        # dims (m). make_pallet_keypoints_3d 인자순서 = (width, depth, height).
        self.pallet_width_m = float(pallet_width_m)
        self.pallet_height_m = float(pallet_height_m)
        self.pallet_depth_m = float(pallet_depth_m)
        self.dims = (self.pallet_width_m, self.pallet_depth_m, self.pallet_height_m)

        # gate default — DopePoseEstimator 와 동일 기본값.
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

        from ultralytics import YOLO  # lazy import — pallet-yolo26 env
        print(f"[YoloPose] weights : {weights_path}")
        self.model = YOLO(weights_path)
        if device is not None:
            try:
                self.model.to(device)
            except Exception as e:
                print(f"[YoloPose] .to({device}) skipped: {e}")
        print(f"[YoloPose] dims(W,D,H) m: {self.dims}")
        print(f"[YoloPose] det_conf={self.det_conf} kp_conf={self.kp_conf} "
              f"imgsz={self.imgsz} pad={self.pad}")
        print(f"[YoloPose] R_fix = Ry(180) (YOLO->DOPE convention)")

        # warmup forward — 첫 추론 lazy compile / cuDNN benchmark 비용 분산.
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model.predict(dummy, verbose=False, conf=self.det_conf,
                                   imgsz=self.imgsz)
            print(f"[YoloPose] warmup done ({self.imgsz}x{self.imgsz})")
        except Exception as e:
            print(f"[YoloPose] warmup skipped: {e}")

        # temporal confirm 상태 (DOPE 와 동일 필드)
        self.consecutive_ok = 0
        self.last_proc_scale: float = 1.0
        self.last_reason: str = "init"

    # ------------------------------------------------------------------ predict
    def _predict_kps(self, bgr: np.ndarray):
        """100px reflect pad → predict → (-pad) shift, box conf 최대 instance.

        eval_ab_crop.predict / infer_fps.predict 복제.
        반환 (kps(9,2), conf(9,)) 또는 (None, None).
        """
        if self.pad > 0:
            inp = cv2.copyMakeBorder(bgr, self.pad, self.pad, self.pad, self.pad,
                                     cv2.BORDER_REFLECT)
        else:
            inp = bgr
        r = self.model.predict(inp, verbose=False, conf=self.det_conf,
                               imgsz=self.imgsz)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return None, None
        allkp = r.keypoints.data.cpu().numpy().astype(np.float64)   # (N,9,3)
        if self.pad > 0:
            allkp = allkp.copy()
            allkp[:, :, 0] -= self.pad
            allkp[:, :, 1] -= self.pad
        if allkp.shape[0] == 1:
            bi = 0
        elif r.boxes is not None:
            bi = int(np.argmax(r.boxes.conf.cpu().numpy()))   # box conf 최대 instance
        else:
            bi = 0
        kp = allkp[bi]
        return kp[:, :2].copy(), kp[:, 2].copy()

    # ------------------------------------------------------------------ PnP
    def _solve_pnp(self, kps_2d: np.ndarray, kp_conf: np.ndarray, K: np.ndarray):
        """SQPnP (eval_ab_crop.solve_pnp 복제) + R_FIX 적용 → DOPE convention.

        반환 (R_dope, t, n_used) 또는 (None, None, n_used). R_dope 는 R_FIX 적용 후.
        """
        kp3d = _make_pallet_keypoints_3d(*self.dims)
        dist = np.zeros((5, 1), dtype=np.float64)

        obj_pts, img_pts = [], []
        for i in range(9):
            if kp_conf[i] >= self.kp_conf:
                obj_pts.append(kp3d[i])
                img_pts.append([float(kps_2d[i, 0]), float(kps_2d[i, 1])])
        n = len(obj_pts)
        if n < 6:
            return None, None, n

        obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
        img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)

        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            return None, None, n
        if not ok:
            return None, None, n

        try:   # 1-step LM 정제
            rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec)
        except cv2.error:
            pass

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        med_reproj = float(np.median(
            np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)))
        if med_reproj > SQPNP_MAX_MED_REPROJ:
            return None, None, n

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        if t[2] < 0:   # 카메라 뒤로 풀리면 부호 뒤집기 (eval_ab_crop 과 동일)
            t, R = -t, -R

        # ★ R_FIX — YOLO local frame → DOPE local frame. t 는 centroid 라 불변.
        R_dope = R @ _R_FIX
        return R_dope, t, n

    def _reproj_from_pose(self, R_dope: np.ndarray, t: np.ndarray, K: np.ndarray):
        """R_dope/t 로 9 keypoint 투영 (원본 좌표 [u,v]).

        주의: R_dope 는 R_FIX 적용본. 3D 모델도 R_FIX 의 역(=동일, R_FIX 자기역원)을
        적용해야 동일 점. R_FIX 는 involution(R_FIX@R_FIX=I) 이므로
        proj = (R_dope @ R_FIX @ kp3d.T) + t = (R_yolo @ kp3d.T) + t 와 동일.
        """
        kp3d = _make_pallet_keypoints_3d(*self.dims)
        kp3d_dope = (_R_FIX @ kp3d.T).T   # R_FIX involution → R_yolo 모델로 환원
        Pc = (R_dope @ kp3d_dope.T).T + t
        out = []
        for p in Pc:
            if p[2] <= 1e-6:
                out.append(None)
            else:
                u = K[0, 0] * p[0] / p[2] + K[0, 2]
                v = K[1, 1] * p[1] / p[2] + K[1, 2]
                out.append((float(u), float(v)))
        return out

    # ------------------------------------------------------------------ public
    def infer_pose(
        self,
        bgr: np.ndarray,
        camera_matrix: np.ndarray,
        depth_frame=None,
    ) -> Optional[dict]:
        """BGR image + 원본 K (+ depth_frame) → 6D pose dict 또는 None.

        DopePoseEstimator.infer_pose 와 동일 시그니처/반환.

        Returns:
            None — 미검출 (self.last_reason 에 사유 기록)
            dict:
                "R"          : np.ndarray (3,3)  pallet local axes in camera frame
                               (DOPE convention, R_FIX 적용). R[:,2]=pallet forward.
                "t_m"        : np.ndarray (3,)   centroid 위치 (m)
                "raw_points_orig"  : list[(u,v) or None]  9 점 (원본 좌표)
                "proj_points_orig" : list[(u,v) or None]  PnP reproj (원본 좌표)
                "kps9_orig"  : list[(u,v)]  (NaN 가능)
                "K_small"    : np.ndarray (3,3)  (YOLO 는 resize 없음 → K_orig)
                "proc_scale" : float (=1.0)
                "consecutive_ok" : int
                "confirmed"  : bool   N 연속 통과 여부
                "gate_passed": bool   sanity gate 통과
                "info"       : dict   n_kp / z_m / reproj
                "yaw_deg_cam": float  atan2(R[0,2], R[2,2]) (degrees, raw)
        """
        if bgr is None or bgr.ndim != 3 or camera_matrix is None:
            self.last_reason = "bad_input"
            return None
        h, w = bgr.shape[:2]
        if h <= 0 or w <= 0:
            self.last_reason = "bad_shape"
            return None

        # YOLO 는 내부에서 imgsz 로 letterbox → keypoint 가 원본 좌표로 복원되므로
        # 추가 resize / K scale 불필요. DOPE 호환 위해 proc_scale=1.0, K_small=K_orig.
        K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.last_proc_scale = 1.0

        kps_2d, kp_conf = self._predict_kps(bgr)
        if kps_2d is None:
            self.consecutive_ok = 0
            self.last_reason = "no_detection"
            return None

        # raw_points (원본 좌표) — conf<thr 점은 None (DOPE raw_points 와 동일 의미).
        raw_orig = [
            (float(kps_2d[i, 0]), float(kps_2d[i, 1])) if kp_conf[i] >= self.kp_conf
            else None
            for i in range(9)
        ]

        R, t, n_used = self._solve_pnp(kps_2d, kp_conf, K)
        t_m = None if t is None else np.asarray(t, dtype=np.float64)

        proj_orig = None
        if R is not None and t is not None:
            proj_orig = self._reproj_from_pose(R, t, K)

        # depth gate (옵션)
        depth_cm = None
        if depth_frame is not None and raw_orig[8] is not None:
            d_m = _sample_depth(depth_frame, raw_orig[8][0], raw_orig[8][1])
            if d_m is not None:
                depth_cm = d_m * 100.0

        ok, reason, info = _evaluate_result(raw_orig, proj_orig, R, t_m,
                                            self.gates, depth_cm)

        if not ok:
            self.consecutive_ok = 0
            self.last_reason = reason
            info = dict(info)
            info["fail_reason"] = reason
            # gate 실패라도 raw_points 있으면 시각화용 반환 (gate_passed=False).
            return self._build_pose_dict(
                raw_orig, proj_orig, R, t_m, K,
                consecutive_ok=0, confirmed=False, gate_passed=False, info=info,
            )

        self.consecutive_ok += 1
        confirmed = self.consecutive_ok >= self.confirm_frames
        self.last_reason = ("ok" if confirmed
                            else f"pending {self.consecutive_ok}/{self.confirm_frames}")
        info = dict(info)
        info["fail_reason"] = "ok"
        return self._build_pose_dict(
            raw_orig, proj_orig, R, t_m, K,
            consecutive_ok=int(self.consecutive_ok),
            confirmed=bool(confirmed), gate_passed=True, info=info,
        )

    @staticmethod
    def _build_pose_dict(raw_orig, proj_orig, R, t_m, K_small,
                         consecutive_ok, confirmed, gate_passed, info):
        """공통 dict 빌더 — DopePoseEstimator._build_pose_dict 와 동일 형식."""
        raw = list(raw_orig) if raw_orig is not None else []
        while len(raw) < 9:
            raw.append(None)
        proj = list(proj_orig) if proj_orig is not None else []
        while len(proj) < 9:
            proj.append(None)
        kps9_orig = [
            (float("nan"), float("nan")) if p is None else (p[0], p[1])
            for p in raw
        ]
        yaw_deg = None
        if R is not None:
            yaw_deg = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))
        return {
            "R": R,
            "t_m": t_m,
            "raw_points_orig": raw,
            "proj_points_orig": proj,
            "kps9_orig": kps9_orig,
            "K_small": K_small,
            "proc_scale": 1.0,
            "consecutive_ok": consecutive_ok,
            "confirmed": confirmed,
            "gate_passed": gate_passed,
            "info": info,
            "yaw_deg_cam": yaw_deg,
        }

    def infer_keypoints9(self, bgr: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """Backward-compat — kps9 (원본 좌표, NaN 가능) 만 반환.

        DopePoseEstimator.infer_keypoints9 와 동일 역할. K=identity 가정이라 R/t 는
        부정확 — 시각화 외 용도 금지, 새 코드는 infer_pose() 사용.
        """
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = K[1, 1] = float(self.imgsz)
        K[0, 2] = float(bgr.shape[1]) / 2.0
        K[1, 2] = float(bgr.shape[0]) / 2.0
        out = self.infer_pose(bgr, K, depth_frame=None)
        if out is None:
            return None
        return out["kps9_orig"]
