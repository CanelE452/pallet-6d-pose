#!/usr/bin/env python3
# verify_depth_fusion.py
# -----------------------------------------------------------------------------
# depth scale 보정 + cam→fork extrinsic 골격 검증.
#
# 데이터:
#   color/GT : challenge/data/capturepalletNN_manual_gt/{stem}.png + {stem}.json
#              (json.objects[0].pose_transform = 4x4 OpenCV camera-frame pose,
#               translation in meters; projected_cuboid_centroid = centroid px)
#   depth    : data/outside/capturepalletNN/depth/{stem}.png  (uint16, mm)
#
# 검증 항목:
#   1) monocular(GT pose) centroid z  vs  RealSense depth(centroid px) 불일치 정량
#      → 이것이 depth scale 보정으로 제거하려는 거리 오차.
#   2) depth_scale_correct 적용 후 t.z == depth(±소수), x,y 가 같은 비율로 스케일.
#   3) apply_cam_to_fork 기본값(0)에서 입력=출력(항등).
# -----------------------------------------------------------------------------
import os
import sys
import json
import glob

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib.pose6d_adapter import depth_scale_correct, apply_cam_to_fork  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEPTH_SCALE_M = 0.001  # RealSense z16 uint16 mm → m


def find_depth_png(seq: str, stem: str):
    p = os.path.join(REPO_ROOT, "data", "outside", f"capturepallet{seq}", "depth", f"{stem}.png")
    return p if os.path.exists(p) else None


def sample_depth_png(depth_mm, u, v, radius=3):
    """depth png (uint16 mm) 의 (u,v) 주변 median depth (m). dope._sample_depth 와 동일 로직."""
    h, w = depth_mm.shape[:2]
    vals = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = int(u) + dx, int(v) + dy
            if 0 <= nx < w and 0 <= ny < h:
                d = float(depth_mm[ny, nx]) * DEPTH_SCALE_M
                if d > 0.05:
                    vals.append(d)
    return float(np.median(vals)) if vals else None


def run_seq(seq: str):
    gt_dir = os.path.join(REPO_ROOT, "challenge", "data", f"capturepallet{seq}_manual_gt")
    jsons = sorted(glob.glob(os.path.join(gt_dir, "*.json")))
    jsons = [j for j in jsons if not j.endswith(".bak")]
    rows = []
    for jp in jsons:
        stem = os.path.splitext(os.path.basename(jp))[0]
        dpng = find_depth_png(seq, stem)
        if dpng is None:
            continue
        d = json.load(open(jp))
        objs = d.get("objects") or []
        if not objs:
            continue
        o = objs[0]
        T = np.array(o["pose_transform"], dtype=np.float64)
        t_mono = T[:3, 3].copy()            # GT/monocular centroid (camera frame, m)
        cuv = o.get("projected_cuboid_centroid")
        if cuv is None:
            continue
        depth_mm = cv2.imread(dpng, cv2.IMREAD_UNCHANGED)
        if depth_mm is None or depth_mm.dtype != np.uint16:
            continue
        d_cen = sample_depth_png(depth_mm, cuv[0], cuv[1])
        if d_cen is None:
            continue

        t_corr, ok = depth_scale_correct(t_mono, cuv, d_cen)
        t_corr = np.array(t_corr, dtype=np.float64)

        rows.append({
            "stem": stem,
            "mono_z": float(t_mono[2]),
            "depth_z": float(d_cen),
            "dz_abs": abs(float(t_mono[2]) - float(d_cen)),
            "corr_z": float(t_corr[2]),
            "corr_ok": ok,
            "t_mono": t_mono,
            "t_corr": t_corr,
        })
    return rows


def main():
    seqs = ["07", "08", "09"]
    all_rows = []
    print("=" * 78)
    print("작업1 검증 — monocular centroid z  vs  RealSense depth(centroid px)")
    print("=" * 78)
    for seq in seqs:
        rows = run_seq(seq)
        if not rows:
            print(f"[cp{seq}] (no matched depth+GT frames)")
            continue
        # corr_ok=True 이고 mono_z 가 물리적으로 그럴듯한 frame 만 헤드라인 통계.
        # (일부 GT pose_transform 이 깨져 mono_z 가 수백~수천 m 인 frame 존재 →
        #  helper 는 depth 로 정확히 보정하지만 'before' 오차 통계를 왜곡하므로 제외.
        #  이런 frame 이야말로 depth 보정이 monocular scale 을 구제하는 사례다.)
        valid = [r for r in rows if r["corr_ok"] and 0.3 < r["mono_z"] < 12.0]
        n_rej = len(rows) - len(valid)
        dz = np.array([r["dz_abs"] for r in valid])
        print(f"\n[cp{seq}] n={len(rows)} frames  (depth 유효 보정 {len(valid)}, "
              f"z_max 밖 무보정 {n_rej})")
        print(f"  |mono_z - depth_z|   mean={dz.mean():.4f} m  median={np.median(dz):.4f} m  "
              f"min={dz.min():.4f}  max={dz.max():.4f}")
        # 보정 후 z == depth 확인 (유효 frame 만)
        zerr = np.array([abs(r["corr_z"] - r["depth_z"]) for r in valid])
        print(f"  보정후 |corr_z - depth_z|  max={zerr.max():.2e} m  (≈0 이어야 함)")
        # x,y 비율 일치 확인: t_corr 가 t_mono * (depth/mono_z) 와 같은지
        ratio_ok = True
        for r in valid:
            tm, tc = r["t_mono"], r["t_corr"]
            s_expected = r["depth_z"] / tm[2]
            if not np.allclose(tc, tm * s_expected, atol=1e-9):
                ratio_ok = False
        print(f"  x,y 동일비율 스케일 (t_corr == t_mono * s):  {'PASS' if ratio_ok else 'FAIL'}")
        # 거리 오차 개선: 보정 전(mono_z) vs 보정 후(corr_z), reference = depth_z
        print(f"  거리(z)오차 감소:  before(mono)={dz.mean():.4f} m  →  after(corr)={zerr.mean():.4e} m")
        for r in valid[:3]:
            print(f"    {r['stem'][:13]}..  mono_z={r['mono_z']:.3f}  depth_z={r['depth_z']:.3f}  "
                  f"corr_z={r['corr_z']:.3f}  t_mono={np.round(r['t_mono'],3)}  t_corr={np.round(r['t_corr'],3)}")
        all_rows.extend(valid)

    if all_rows:
        dz = np.array([r["dz_abs"] for r in all_rows])
        print("\n" + "-" * 78)
        print(f"[ALL cp07/08/09] n={len(all_rows)}  |mono_z - depth_z|  "
              f"mean={dz.mean():.4f} m  median={np.median(dz):.4f} m")
        print("-" * 78)

    # ---- 작업2 검증: extrinsic 항등 (기본값 0 → 입력=출력) ----
    print("\n" + "=" * 78)
    print("작업2 검증 — apply_cam_to_fork 기본값(0) 항등 (회귀 없음)")
    print("=" * 78)
    R = np.array([
        [0.98638303, 0.09184943, 0.13642655],
        [-0.0861514, 0.99516753, -0.04711172],
        [-0.14009446, 0.03471686, 0.98952932],
    ])
    t = np.array([0.79809807, 0.22484211, 5.51028586])
    R_f, t_f = apply_cam_to_fork(R, t)  # 기본값 → 항등
    R_f = np.array(R_f); t_f = np.array(t_f)
    r_id = np.allclose(R_f, R, atol=1e-12)
    t_id = np.allclose(t_f, t, atol=1e-12)
    print(f"  R_fork == R_cam :  {'PASS' if r_id else 'FAIL'}  (max diff {np.abs(R_f-R).max():.2e})")
    print(f"  t_fork == t_cam :  {'PASS' if t_id else 'FAIL'}  (max diff {np.abs(t_f-t).max():.2e})")

    # 비-항등 sanity: yaw=90deg 회전이 실제로 적용되는지 (골격 동작 확인)
    R_f2, t_f2 = apply_cam_to_fork(np.eye(3), [1.0, 0.0, 0.0],
                                   cam_to_fork_t=[0.0, 0.0, 0.5],
                                   cam_to_fork_rpy_deg=[0.0, 0.0, 90.0])
    t_f2 = np.array(t_f2)
    # yaw=90: x->y, t=[1,0,0]+offset[0,0,0.5] → [0,1,0.5]
    nonid_ok = np.allclose(t_f2, [0.0, 1.0, 0.5], atol=1e-9)
    print(f"  비-항등 sanity (yaw90,+z0.5):  {'PASS' if nonid_ok else 'FAIL'}  t_fork={np.round(t_f2,4)}")

    # depth 무효 시 무보정 + 플래그
    t_nc, ok_nc = depth_scale_correct([1, 2, 3], (10, 10), None)
    print(f"\n  depth 무효 시 무보정:  {'PASS' if (t_nc==(1.0,2.0,3.0) and not ok_nc) else 'FAIL'}  "
          f"(t={t_nc}, flag={ok_nc})")


if __name__ == "__main__":
    main()
