"""debug_frame_000005.py — mixed_v8_train/000005 단일 프레임 분석.

목적:
  - 8 corner 3D (object frame + camera frame) + projected 출력
  - 4 vertical face 후보 각각의:
      * top/bot corner index
      * image polygon area (v4 기준)
      * face normal vs los yaw_alignment (v5 기준)
      * face center camera distance (v5 보조)
  - v4 vs v5 vs 사용자 직관 (이미지에서 가장 큰 면) 비교
  - 4 face 를 색깔별로 시각화한 디버그 PNG 생성
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import cv2


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_PATH = os.path.join(REPO, "data", "pallet", "training_data", "mixed_v8_train", "000005.json")
PNG_PATH  = os.path.join(REPO, "data", "pallet", "training_data", "mixed_v8_train", "000005.png")
OUT_PATH  = os.path.join(REPO, "debug", "converted_gt_viz_v4", "_debug_000005.png")


def polyarea(pts):
    pts = np.asarray(pts, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(
        x[0] * y[1] - x[1] * y[0] +
        x[1] * y[2] - x[2] * y[1] +
        x[2] * y[3] - x[3] * y[2] +
        x[3] * y[0] - x[0] * y[3]
    )


def plane_normal(pts4):
    p0, p1, p2 = pts4[0], pts4[1], pts4[2]
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n)
    return n / nn if nn > 1e-9 else None


def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    obj = data["objects"][0]
    cuboid = np.array(obj["cuboid"], dtype=np.float64)           # (8,3) object frame
    proj   = np.array(obj["projected_cuboid"], dtype=np.float64) # (8,2) image
    M      = np.array(obj["pose_transform"], dtype=np.float64)   # (4,4) object->camera
    R = M[:3, :3]
    t = M[:3, 3]
    pts_cam = (R @ cuboid.T).T + t  # (8,3) camera frame (OpenCV +z forward)

    print("=" * 78)
    print("FRAME mixed_v8_train/000005.json")
    print("=" * 78)
    print(f"camera intrinsics: fx={data['camera_data']['intrinsics']['fx']:.2f}  "
          f"cx={data['camera_data']['intrinsics']['cx']}  cy={data['camera_data']['intrinsics']['cy']}")
    print(f"object location (cam frame): {t}")
    print(f"camera distance to body: {np.linalg.norm(t):.3f} m")
    print()
    print(f"{'idx':>3}  {'cuboid(obj,xyz)':>32}  {'pts_cam(xyz)':>32}  {'proj(uv)':>22}")
    for i in range(8):
        c, pc, pj = cuboid[i], pts_cam[i], proj[i]
        print(f"{i:>3}  "
              f"({c[0]:>+8.3f},{c[1]:>+8.3f},{c[2]:>+8.3f})  "
              f"({pc[0]:>+8.3f},{pc[1]:>+8.3f},{pc[2]:>+8.3f})  "
              f"({pj[0]:>7.1f},{pj[1]:>7.1f})")
    print()

    # ── 1. top/bot by object z (height)
    z_order = np.argsort(cuboid[:, 2])[::-1]
    top4 = [int(i) for i in z_order[:4]]
    bot4 = [int(i) for i in z_order[4:]]
    print(f"object z (height) order desc: {[int(i) for i in z_order]}")
    print(f"  top4 (z=0.137): {top4}    z vals: {cuboid[top4, 2].tolist()}")
    print(f"  bot4 (z=0.000): {bot4}    z vals: {cuboid[bot4, 2].tolist()}")
    print()

    # ── 2. vertical pairing
    top_to_bot = {}
    used = set()
    for ti in top4:
        cands = sorted(bot4, key=lambda b: ((cuboid[ti,0]-cuboid[b,0])**2 +
                                             (cuboid[ti,1]-cuboid[b,1])**2))
        for bj in cands:
            if bj not in used:
                top_to_bot[ti] = bj
                used.add(bj)
                break
    print("vertical pairing (top -> bot, by xy distance):")
    for ti in top4:
        bj = top_to_bot[ti]
        d = np.linalg.norm(cuboid[ti,:2] - cuboid[bj,:2])
        print(f"  {ti} -> {bj}   xy_dist={d:.6f} m   (vertical edge length)")
    print()

    # ── 3. top4 splits + parallelism
    splits = [
        ((top4[0], top4[1]), (top4[2], top4[3])),
        ((top4[0], top4[2]), (top4[1], top4[3])),
        ((top4[0], top4[3]), (top4[1], top4[2])),
    ]
    print(f"top4 splits (eA, eB) & parallelism cos:")
    parallel_splits = []
    for k, (sA, sB) in enumerate(splits):
        eA = cuboid[sA[1]] - cuboid[sA[0]]
        eB = cuboid[sB[1]] - cuboid[sB[0]]
        nA = np.linalg.norm(eA); nB = np.linalg.norm(eB)
        cos = abs(np.dot(eA, eB) / (nA * nB))
        is_par = cos >= 0.95
        print(f"  split{k}: {sA} | {sB}   |eA|={nA:.3f}  |eB|={nB:.3f}  "
              f"cos|.|={cos:.4f}  {'PARALLEL' if is_par else 'diag'}")
        if is_par:
            parallel_splits.append((sA, sB))
    print()

    # ── 4. 4 vertical face 후보
    print("4 vertical face candidates (face = [t1, t2, bot(t2), bot(t1)]):")
    body_center_cam = pts_cam.mean(axis=0)
    cam_origin = np.zeros(3)
    face_records = []
    for (sA, sB) in parallel_splits:
        faceA = [sA[0], sA[1], top_to_bot[sA[1]], top_to_bot[sA[0]]]
        faceB = [sB[0], sB[1], top_to_bot[sB[1]], top_to_bot[sB[0]]]
        for face in (faceA, faceB):
            face_pts_cam = pts_cam[face]
            face_pts_proj = proj[face]
            face_center_cam = face_pts_cam.mean(axis=0)
            face_center_obj = cuboid[face].mean(axis=0)
            normal_cam = plane_normal(face_pts_cam)
            out_dir = face_center_cam - body_center_cam
            if np.dot(normal_cam, out_dir) < 0:
                normal_cam = -normal_cam
            los = face_center_cam - cam_origin
            d = np.linalg.norm(los)
            los_unit = los / d
            yaw_align = float(np.dot(normal_cam, -los_unit))
            area = polyarea(face_pts_proj)
            face_records.append({
                "face": face,
                "area_img": area,
                "yaw_align": yaw_align,
                "dist": d,
                "normal_cam": normal_cam,
                "center_obj": face_center_obj,
                "center_cam": face_center_cam,
            })

    # 가까울수록 prox 1.0
    dmin = min(r["dist"] for r in face_records)
    dmax = max(r["dist"] for r in face_records)
    drange = max(dmax - dmin, 1e-9)
    for r in face_records:
        r["prox"] = 1.0 - (r["dist"] - dmin) / drange
        r["v5_score"] = 0.7 * r["yaw_align"] + 0.3 * r["prox"]

    print(f"  {'#':>2}  {'face':>16}  {'area_img':>9}  {'yaw_align':>9}  "
          f"{'dist':>6}  {'prox':>5}  {'v5_score':>8}  center_obj")
    for i, r in enumerate(face_records):
        c = r["center_obj"]
        print(f"  {i:>2}  {str(r['face']):>16}  {r['area_img']:>9.1f}  "
              f"{r['yaw_align']:>+9.3f}  {r['dist']:>6.3f}  {r['prox']:>5.2f}  "
              f"{r['v5_score']:>+8.3f}  ({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})")
    print()

    # ── 5. v4 (max area) vs v5 (max score)
    v4_pick = max(range(len(face_records)), key=lambda i: face_records[i]["area_img"])
    v5_pick = max(range(len(face_records)), key=lambda i: face_records[i]["v5_score"])
    print(f"v4 selects FRONT = face#{v4_pick}  {face_records[v4_pick]['face']}  "
          f"(max image area = {face_records[v4_pick]['area_img']:.1f} px²)")
    print(f"v5 selects FRONT = face#{v5_pick}  {face_records[v5_pick]['face']}  "
          f"(max score = {face_records[v5_pick]['v5_score']:+.3f}, "
          f"yaw={face_records[v5_pick]['yaw_align']:+.3f})")
    print()

    # SDG 원본 0~3 이 어느 face 인지 식별
    sdg_front = [0, 1, 2, 3]
    print(f"SDG original face {{0,1,2,3}} ?= which candidate?")
    for i, r in enumerate(face_records):
        if set(r["face"]) == set(sdg_front):
            print(f"  -> face#{i}  area={r['area_img']:.1f}  yaw={r['yaw_align']:+.3f}  "
                  f"score={r['v5_score']:+.3f}")
    print()

    # ── 6. 시각화
    img = cv2.imread(PNG_PATH)
    if img is None:
        print(f"!! image not found: {PNG_PATH}")
        return

    # 2x upscale + 좌/우 비교 + 하단 패널
    SCALE = 2
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    H, W = img.shape[:2]
    canvas = np.zeros((H + 280, W * 2 + 20, 3), dtype=np.uint8)
    canvas[:H, :W] = img.copy()
    canvas[:H, W+20:W*2+20] = img.copy()
    proj_v = proj * SCALE  # for drawing

    # 좌측: 원본 8 corner 점만 + 인덱스
    for i in range(8):
        u, v = int(proj_v[i, 0]), int(proj_v[i, 1])
        cv2.circle(canvas[:H, :W], (u, v), 6, (0, 255, 255), -1)
        cv2.putText(canvas[:H, :W], str(i), (u + 7, v - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas[:H, :W], "ORIGINAL SDG indices 0..7",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # 우측: 4 vertical face 색칠
    colors = [(255, 80, 80), (80, 255, 80), (80, 200, 255), (255, 80, 255)]  # BGR
    right = canvas[:H, W+20:W*2+20]
    for i, r in enumerate(face_records):
        poly = proj_v[r["face"]].astype(np.int32)
        overlay = right.copy()
        cv2.fillPoly(overlay, [poly], colors[i])
        cv2.addWeighted(overlay, 0.35, right, 0.65, 0, dst=right)
        cv2.polylines(right, [poly], True, colors[i], 3)
    for i, r in enumerate(face_records):
        c = proj_v[r["face"]].mean(axis=0).astype(int)
        cv2.putText(right, f"F{i}", (c[0] - 12, c[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i], 3, cv2.LINE_AA)
    for i in range(8):
        u, v = int(proj_v[i, 0]), int(proj_v[i, 1])
        cv2.circle(right, (u, v), 6, (255, 255, 255), -1)
        cv2.circle(right, (u, v), 7, (0, 0, 0), 2)
        cv2.putText(right, str(i), (u + 7, v - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, "4 vertical faces (F0..F3)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, f"v4 pick: F{v4_pick}  {face_records[v4_pick]['face']}  (max area)",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(right, f"v5 pick: F{v5_pick}  {face_records[v5_pick]['face']}  (max yaw_align)",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2, cv2.LINE_AA)

    # 하단 텍스트 패널
    y0 = H + 24
    cv2.putText(canvas, "4-face stats (face / area_img / yaw_align / dist / v5_score):",
                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    for i, r in enumerate(face_records):
        y = y0 + 30 + i * 28
        txt = (f"F{i}  face={r['face']}  area={r['area_img']:7.1f}  "
               f"yaw={r['yaw_align']:+.3f}  dist={r['dist']:.3f}  "
               f"score={r['v5_score']:+.3f}")
        cv2.putText(canvas, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2, cv2.LINE_AA)
    y = y0 + 30 + 4 * 28 + 8
    cv2.putText(canvas, "v4=max(area_img)  v5=0.7*yaw_align+0.3*prox  SDG_front_face={0,1,2,3}=F1",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, canvas)
    print(f"saved debug PNG: {OUT_PATH}")


if __name__ == "__main__":
    main()
