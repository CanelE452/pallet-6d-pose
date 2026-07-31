"""왜 line이 pose를 못 고쳤는지 시각화 (진단 전용, 학습 없음)."""
import sys, json, math
from pathlib import Path
import numpy as np, cv2, pandas as pd
ROOT = Path("/home/minjae/Documents/github/pallet-pose")
for p in ("Deep_Object_Pose/common","scripts/stage0","challenge/scripts","scripts/data_prep/eval"):
    sys.path.insert(0, str(ROOT/p))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_s2_mechanism_diagnostic as MD, paper_s2_frozen_diagnostic as FZ
import pallet_graph_geometry as PG, dimension_guided_graph_pose as DGP
import importlib.util
spec=importlib.util.spec_from_file_location("LS", ROOT/"scripts/stage0/paper_s2_palletgraph_line_screen.py")
LS=importlib.util.module_from_spec(spec); spec.loader.exec_module(LS)

OUT = LS.OUT_DIR / "why_line_failed"; OUT.mkdir(parents=True, exist_ok=True)
ev = LS.LineScreenEvaluator()
arms = {n: pd.read_parquet(LS.OUT_DIR/f"arm_{n}.parquet") for n in ("P0","P1","P3_f100")}

# ---- 그림 1: 에너지 지형 (yaw / tx 슬라이스) ----------------------------
merged = arms["P1"].merge(arms["P3_f100"], on="frame_id", suffixes=("_1","_3"))
cand = merged[merged.pose_success_1 & merged.pose_success_3].sort_values(
    "yaw_mod180_deg_1", ascending=False)
picks = [cand.iloc[0].frame_id, cand.iloc[len(cand)//2].frame_id]

fig, axes = plt.subplots(2, len(picks), figsize=(6*len(picks), 8))
for col, uid in enumerate(picks):
    spec_f = next(f for f in ev.frames if f["frame_id"]==uid)
    g = ev.geometry[uid]; size=(spec_f["image_width"], spec_f["image_height"])
    ref = g.solve(g.gt_points); base = g.solve(ev.decoded[uid]["D0"])
    obs, valid = ev.observations(uid)
    evi = ev.build_evidence("P3", uid, ref, size, LS.CANNY_SETTINGS[1])
    lam = 11.04  # f100 calibrated
    R0, t0 = base["R"], np.asarray(base["t"]).reshape(3)
    # yaw slice
    deg = np.linspace(-12, 12, 49)
    Ep, El = [], []
    for d in deg:
        a = math.radians(d)
        Rd = np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
        R = R0 @ Rd
        ep,_ = DGP.point_energy(R,t0,g.K,g.dims,obs,valid); Ep.append(ep)
        el,_ = DGP.line_energy(R,t0,g.K,g.dims,evi); El.append(el)
    Ep,El = np.array(Ep), np.array(El)
    gt_off = PG.yaw_from_rotation(ref["R"]) - PG.yaw_from_rotation(R0)
    gt_off = math.degrees(PG.wrap_half_pi(gt_off))
    ax = axes[0,col]
    ax.plot(deg, Ep/max(Ep.max(),1e-9), label="E_point (norm)", lw=2)
    ax.plot(deg, El/max(El.max(),1e-9), label="E_line (norm)", lw=2)
    ax.plot(deg, (Ep+lam*El)/max((Ep+lam*El).max(),1e-9), label="E_total", lw=2, ls="--")
    ax.axvline(0, color="gray", lw=1, label="current pose")
    ax.axvline(gt_off, color="red", lw=1.5, label=f"GT yaw ({gt_off:+.1f}°)")
    ax.set_xlabel("yaw offset from current pose (deg)"); ax.set_ylabel("normalised energy")
    ax.set_title(f"{uid.split(':')[-1][:12]}  yaw_err={cand[cand.frame_id==uid].yaw_mod180_deg_1.iloc[0]:.1f}°")
    ax.legend(fontsize=7)
    # 절대 스케일
    ax2 = axes[1,col]
    ax2.plot(deg, Ep, label=f"E_point (min {Ep.min():.1f})", lw=2)
    ax2.plot(deg, lam*El, label=f"λ·E_line (min {(lam*El).min():.1f})", lw=2)
    ax2.axvline(gt_off, color="red", lw=1.5)
    ax2.set_xlabel("yaw offset (deg)"); ax2.set_ylabel("absolute energy")
    ax2.set_title("절대 스케일 — 누가 최소를 결정하나")
    ax2.legend(fontsize=7)
fig.suptitle("왜 line이 pose를 못 옮겼나 — 에너지 지형 (P3 oracle, λ=11.04)", fontsize=13)
fig.tight_layout(); fig.savefig(OUT/"1_energy_landscape.png", dpi=140); plt.close(fig)
print("wrote 1_energy_landscape.png")

# ---- 그림 2: pose before/after wireframe ---------------------------------
def draw_pose(canvas, R, t, K, dims, color, thick=2):
    pts,_ = PG.project_points(PG.make_corners(*dims)[:8], R, t, K)
    for cls, pairs in PG.edge_sets(*dims).items():
        for i,j in pairs:
            a,b = pts[i], pts[j]
            if not (np.isfinite(a).all() and np.isfinite(b).all()): continue
            cv2.line(canvas, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                     color, thick, cv2.LINE_AA)

panels=[]
for uid in picks:
    spec_f = next(f for f in ev.frames if f["frame_id"]==uid)
    g = ev.geometry[uid]; img = ev.images[uid].copy()
    ref = g.solve(g.gt_points); base = g.solve(ev.decoded[uid]["D0"])
    r1 = arms["P1"][arms["P1"].frame_id==uid].iloc[0]
    r3 = arms["P3_f100"][arms["P3_f100"].frame_id==uid].iloc[0]
    draw_pose(img, ref["R"], np.asarray(ref["t"]).reshape(3), g.K, g.dims, (60,220,60), 3)   # GT green
    draw_pose(img, base["R"], np.asarray(base["t"]).reshape(3), g.K, g.dims, (60,60,235), 2) # P0 red
    panels.append(MD.banner(img, [
        f"{uid.split(':')[-1][:14]}  green=GT  red=P0(point PnP)",
        f"P1 yaw={r1.yaw_mod180_deg:.2f}°  ->  P3(oracle line) yaw={r3.yaw_mod180_deg:.2f}°  "
        f"(Δ={r3.yaw_mod180_deg-r1.yaw_mod180_deg:+.3f}°)"]))
h=min(p.shape[0] for p in panels)
cv2.imwrite(str(OUT/"2_pose_unchanged.png"), np.hstack([p[:h] for p in panels]))
print("wrote 2_pose_unchanged.png")

# ---- 그림 3: line이 개입조차 못한 17프레임 -------------------------------
failed = arms["P0"][~arms["P0"].pose_success].frame_id.tolist()
sel = failed[:6]
tiles=[]
for uid in sel:
    spec_f = next(f for f in ev.frames if f["frame_id"]==uid)
    g = ev.geometry[uid]; img = ev.images[uid].copy()
    ref = g.solve(g.gt_points)
    if ref is not None:
        draw_pose(img, ref["R"], np.asarray(ref["t"]).reshape(3), g.K, g.dims, (60,220,60), 3)
    MD.draw_points(img, ev.decoded[uid]["D0"], (60,60,235))
    n_det = sum(1 for p in ev.decoded[uid]["D0"][:8] if p is not None)
    tiles.append(MD.banner(cv2.resize(img,(480,360)), [
        f"{spec_f['domain']}  corners={n_det}/8  trunc={spec_f['is_truncated']}",
        "point PnP FAIL -> DGP 초기값 없음 -> line 미개입"]))
h=min(t.shape[0] for t in tiles)
rows=[np.hstack([t[:h] for t in tiles[i:i+3]]) for i in range(0,len(tiles),3)]
w=min(r.shape[1] for r in rows)
cv2.imwrite(str(OUT/"3_never_tested_frames.png"), np.vstack([r[:,:w] for r in rows]))
print("wrote 3_never_tested_frames.png")
