"""diagnose_v8_zoom.py — zoom 시각화 + click vs projection 비교 + axis 표시."""
import os, sys, json
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from annotate_pnp import (
    solve_pose, make_pallet_keypoints_3d, project_3d, PALLET_DIMS,
    _CUBE_FLIPS_DEG, _rot_axis_angle, _seed_from_ippe_face, _CUBOID_FACES,
    _refine_with_init, _eval_pair_invariants, _eval_v8_tilt, _reproj_err_dict,
    V8_TILT_SOFT_THR, V8_TILT_HARD_THR,
)

REPO  = r"C:\Users\minjae\Documents\github\FoundationPose"
IMG   = os.path.join(REPO, "data/outside/capturepallet08/rgb/1778653498432396288.png")
K_TXT = os.path.join(REPO, "data/outside/capturepallet08/cam_K.txt")
OUT   = os.path.join(REPO, "data/pallet/results/annotate_v8_oblique")

clicks = [
    [366.0, 264.0],  # 0
    [462.0, 251.0],  # 1
    [484.0, 278.0],  # 2
    [416.0, 272.0],  # 3
    None,            # 4
    [549.0, 258.0],  # 5
    [552.0, 270.0],  # 6
    None,            # 7
    None,            # 8 centroid
]

colors_corner = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
    (255, 255, 0), (255, 0, 0), (255, 0, 255), (255, 255, 255),
]

K = np.loadtxt(K_TXT)
img = cv2.imread(IMG)
H, W = img.shape[:2]
print(f"img {W}x{H}")

pose = solve_pose(clicks, K, img_shape=img.shape)
print(f"R=\n{pose['R']}")
print(f"t={pose['t']}")
print(f"dims={pose['dims']}")
print(f"reproj_error_px={pose['reproj_error_px']:.2f}")
print(f"v6_strict_passed={pose['_v6_strict_passed']}")

# Crop pallet region with margin
ux = [p[0] for p in clicks if p is not None] + [p[0] for p in pose['projected_all'][:8] if p[0] >= 0]
vy = [p[1] for p in clicks if p is not None] + [p[1] for p in pose['projected_all'][:8] if p[0] >= 0]
xmin = max(0, int(min(ux) - 40))
xmax = min(W, int(max(ux) + 40))
ymin = max(0, int(min(vy) - 40))
ymax = min(H, int(max(vy) + 40))

# Annotated full image, then crop
vis = img.copy()
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
proj = pose['projected_all']

# Draw 12 cuboid edges + clear face colors
# FRONT face edges (RED), BACK face edges (BLUE), connectors (GREEN)
front_edges = [(0,1),(1,2),(2,3),(3,0)]
back_edges  = [(4,5),(5,6),(6,7),(7,4)]
conn_edges  = [(0,4),(1,5),(2,6),(3,7)]
for a,b in front_edges:
    pa,pb = proj[a], proj[b]
    if pa[0] == -1 or pb[0] == -1: continue
    cv2.line(vis, (int(pa[0]),int(pa[1])), (int(pb[0]),int(pb[1])), (0,0,255), 2, cv2.LINE_AA)
for a,b in back_edges:
    pa,pb = proj[a], proj[b]
    if pa[0] == -1 or pb[0] == -1: continue
    cv2.line(vis, (int(pa[0]),int(pa[1])), (int(pb[0]),int(pb[1])), (255,0,0), 2, cv2.LINE_AA)
for a,b in conn_edges:
    pa,pb = proj[a], proj[b]
    if pa[0] == -1 or pb[0] == -1: continue
    cv2.line(vis, (int(pa[0]),int(pa[1])), (int(pb[0]),int(pb[1])), (0,200,0), 1, cv2.LINE_AA)

# Cuboid corners labeled
for i in range(8):
    p = proj[i]
    if p[0] == -1: continue
    u,v = int(p[0]), int(p[1])
    # projected corner = filled circle
    cv2.circle(vis, (u,v), 5, colors_corner[i], -1)
    cv2.circle(vis, (u,v), 5, (255,255,255), 1)
    cv2.putText(vis, f"P{i}", (u+7, v-7), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255,255,255), 1, cv2.LINE_AA)

# User clicks (open ring, hollow)
for i, p in enumerate(clicks[:8]):
    if p is None: continue
    u,v = int(p[0]), int(p[1])
    cv2.circle(vis, (u,v), 8, colors_corner[i], 2)
    cv2.putText(vis, f"C{i}", (u-30, v+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                colors_corner[i], 1, cv2.LINE_AA)
    # Draw arrow from click to projected
    pp = proj[i]
    if pp[0] != -1:
        cv2.arrowedLine(vis, (u, v), (int(pp[0]), int(pp[1])),
                        (0, 255, 255), 1, tipLength=0.2)

# Cuboid local axes from centroid (X=right red, Y=down green, Z=forward blue)
centroid_3d = make_pallet_keypoints_3d(*pose['dims'])[8]
R, t = pose['R'], pose['t']
ax_len = 0.4
origin_cam = (R @ centroid_3d) + t
u0 = int(K[0,0] * origin_cam[0]/origin_cam[2] + K[0,2])
v0 = int(K[1,1] * origin_cam[1]/origin_cam[2] + K[1,2])
ax_colors = [(0,0,255), (0,255,0), (255,0,0)]
ax_labels = ['X', 'Y', 'Z']
for k in range(3):
    end_3d = centroid_3d + np.eye(3)[k] * ax_len
    end_cam = (R @ end_3d) + t
    u1 = int(K[0,0] * end_cam[0]/end_cam[2] + K[0,2])
    v1 = int(K[1,1] * end_cam[1]/end_cam[2] + K[1,2])
    cv2.arrowedLine(vis, (u0,v0), (u1,v1), ax_colors[k], 2, tipLength=0.2)
    cv2.putText(vis, ax_labels[k], (u1+3, v1+3), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, ax_colors[k], 2, cv2.LINE_AA)

# Annotation header
cv2.rectangle(vis, (0, 0), (W, 60), (0, 0, 0), -1)
txt1 = f"v7 SELECTED  dims={pose['dims']}  reproj={pose['reproj_error_px']:.1f}px  tz={pose['t'][2]:.2f}m"
txt2 = f"strict={pose['_v6_strict_passed']}  n_strict_ok={pose['_v6_n_strict_ok']}/{pose['_v6_n_candidates']}"
cv2.putText(vis, txt1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
cv2.putText(vis, txt2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

# Legend (bottom)
cv2.rectangle(vis, (0, H-50), (W, H), (0, 0, 0), -1)
legend = "FRONT face=RED   BACK face=BLUE   connectors=GREEN   CLICK=hollow ring   PROJ=filled disc"
cv2.putText(vis, legend, (10, H-22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
cv2.putText(vis, "yellow arrow = click->proj direction (reproj error)", (10, H-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

cv2.imwrite(os.path.join(OUT, "v8_oblique_full.png"), vis)
print(f"saved: v8_oblique_full.png")

# Cropped version with 3x zoom
crop = vis[ymin:ymax, xmin:xmax]
crop_3x = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(OUT, "v8_oblique_zoom.png"), crop_3x)
print(f"saved: v8_oblique_zoom.png (crop {xmin},{ymin}..{xmax},{ymax} x3)")

# Print all 8 projected positions
print(f"\nAll 8 projected corners:")
for i in range(8):
    p = proj[i]
    note = ""
    if i < len(clicks) and clicks[i] is not None:
        d = np.hypot(p[0]-clicks[i][0], p[1]-clicks[i][1])
        note = f"  click=({clicks[i][0]:.0f},{clicks[i][1]:.0f})  err={d:.1f}px"
    print(f"  [{i}] proj=({p[0]:7.1f}, {p[1]:7.1f}){note}")

# 3D cuboid corners in cam frame
kp3d = make_pallet_keypoints_3d(*pose['dims'])
pts_cam = (R @ kp3d.T).T + t
print(f"\nAll 8 corners in cam frame (X,Y,Z):")
for i in range(8):
    print(f"  [{i}] ({pts_cam[i,0]:6.2f}, {pts_cam[i,1]:6.2f}, {pts_cam[i,2]:6.2f})")
print(f"\nDims width={pose['dims'][0]}m depth={pose['dims'][1]}m height={pose['dims'][2]}m")
print(f"Pallet sits on ground if Y_max-Y_min ~ height={pose['dims'][2]}m")
print(f"  Actual Y range cam-frame: {pts_cam[:,1].max()-pts_cam[:,1].min():.3f}m")

# Enumerate all candidates and check tilt distribution
def enumerate_candidates_tilt(kps_2d, K, dims):
    kp3d_loc = make_pallet_keypoints_3d(*dims)
    valid_idx = [i for i in range(min(9, len(kps_2d))) if kps_2d[i] is not None]
    obj = np.array([kp3d_loc[i] for i in valid_idx], dtype=np.float64)
    img_p = np.array([kps_2d[i] for i in valid_idx], dtype=np.float64)
    inits = []
    for _name, face in _CUBOID_FACES:
        inits.extend(_seed_from_ippe_face(kps_2d, K, kp3d_loc, list(face)))
    for flag in (cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_SQPNP):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img_p, K, None, flags=flag)
            if ok and tvec[2, 0] > 0:
                R0, _ = cv2.Rodrigues(rvec)
                inits.append((R0, tvec.flatten()))
        except cv2.error: pass
    try:
        ok_n, rl, tl, _ = cv2.solvePnPGeneric(obj, img_p, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok_n:
            for rv, tv in zip(rl, tl):
                if tv[2, 0] > 0:
                    R0, _ = cv2.Rodrigues(rv)
                    inits.append((R0, tv.flatten()))
    except cv2.error: pass
    cx_K, cy_K, fx_K = K[0,2], K[1,2], K[0,0]
    mean_u = np.mean([kps_2d[i][0] for i in valid_idx])
    mean_v = np.mean([kps_2d[i][1] for i in valid_idx])
    img_w_ = max(kps_2d[i][0] for i in valid_idx) - min(kps_2d[i][0] for i in valid_idx)
    z_guess = max(0.5, fx_K * dims[0] / max(img_w_, 50.0))
    t_man = np.array([(mean_u - cx_K) * z_guess / fx_K,
                      (mean_v - cy_K) * z_guess / fx_K, z_guess])
    Rx180 = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    inits.append((Rx180.copy(), t_man.copy()))
    inits.append((np.eye(3), t_man.copy()))
    flips = []
    for ax in _CUBE_FLIPS_DEG:
        rx_ = _rot_axis_angle((1,0,0), ax[0])
        ry_ = _rot_axis_angle((0,1,0), ax[1])
        rz_ = _rot_axis_angle((0,0,1), ax[2])
        flips.append(rz_ @ ry_ @ rx_)
    cps = np.array([kps_2d[i] for i in valid_idx])
    cspan = max(cps[:,0].max()-cps[:,0].min(), cps[:,1].max()-cps[:,1].min(), 50.0)
    z_far = 50.0 * fx_K * max(dims) / cspan
    img_area = 640*480
    min_bbox = 0.015 * img_area
    out = []
    for R0, t0 in inits:
        for F in flips:
            res = _refine_with_init(obj, img_p, K, R0 @ F, t0)
            if res is None: continue
            R_, t_ = res
            if t_[2] <= 0 or t_[2] > z_far: continue
            pts_c = (R_ @ kp3d_loc.T).T + t_
            if (pts_c[:,2] <= 0).any(): continue
            lrv, tbv, frv, proj_a, _ = _eval_pair_invariants(R_, t_, K, kp3d_loc)
            p8 = np.array(proj_a[:8])
            if (p8[:,0].max()-p8[:,0].min()) * (p8[:,1].max()-p8[:,1].min()) < min_bbox:
                continue
            err = _reproj_err_dict(proj_a, valid_idx, kps_2d)
            tilt = _eval_v8_tilt(R_)
            out.append({"R": R_, "t": t_, "err": err, "viol": lrv+tbv+frv,
                        "tilt": tilt, "proj_all": proj_a, "tz": t_[2]})
    return out

print(f"\n=== Tilt distribution analysis ===")
for dims_lbl, dims in [("110front", PALLET_DIMS),
                        ("130front", (PALLET_DIMS[1], PALLET_DIMS[0], PALLET_DIMS[2]))]:
    cands = enumerate_candidates_tilt(clicks, K, dims)
    strict = sorted([c for c in cands if c["viol"] == 0], key=lambda c: c["err"])
    print(f"\n[{dims_lbl}] total={len(cands)} strict-pass={len(strict)}")
    print(f"  Top 5 strict-pass (sorted by err) with tilt:")
    for i, c in enumerate(strict[:5]):
        flag = ""
        if c["tilt"] < V8_TILT_HARD_THR: flag = " [HARD-REJECT]"
        elif c["tilt"] < V8_TILT_SOFT_THR: flag = " [soft-penalty]"
        print(f"    [{i}] err={c['err']:6.2f}  tilt={c['tilt']:.3f}  tz={c['tz']:.2f}{flag}")
    print(f"  Top 5 strict-pass sorted by tilt (descending, most upright first):")
    strict_by_tilt = sorted(strict, key=lambda c: -c["tilt"])
    for i, c in enumerate(strict_by_tilt[:5]):
        print(f"    [{i}] tilt={c['tilt']:.3f}  err={c['err']:6.2f}  tz={c['tz']:.2f}")
