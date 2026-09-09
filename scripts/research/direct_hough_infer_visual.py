"""학습된 DOPE+Direct-Hough 체크포인트를 이 프레임에 실제로 돌려 12 개 line 을 그린다.

체인:  image -> FrozenA1(DOPE VGG trunk) -> F50 (128ch, 50x50)
       -> RoleQueryGlobal encoder -> 12 role descriptor
       -> DirectHoughHead: role @ f(theta,rho) -> (theta,rho) 격자 점수
       -> argmax -> role 마다 무한직선 하나

새 학습 0.  체크포인트 1 회 추론.  이 프레임은 stage0 기준 SEALED 세션이므로
**시각화 전용**이며 어떤 모델 선택·threshold 결정에도 쓰지 않는다.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/pallet/results/hough_line_visual"
C = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
FID = "eval_pallet07__1778652166837872128"
CKPT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation/"
        "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search/"
        "supporting_line_map/checkpoints/DH_full/step_08515.pth")

EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def main():
    DH = _load("DH", "scripts/stage0/line/direct_hough_role_heatmap.py")
    V2 = DH.V2
    dev = DH.DEV
    print(f"device {dev}  MAP {DH.MAP}  CANON {DH.CANON}  RHO_MAX {DH.RHO_MAX}")

    model = DH.DirectHoughModel().to(dev)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd["model"], strict=True), None
    model.eval()
    print(f"ckpt tag={sd['tag']} step={sd['step']} seed={sd['seed']} -> loaded strict=True")

    a1 = V2.load_a1()

    # ---- 이미지 준비: A1 이 기대하는 것과 같은 경로로
    man = {f["frame_id"]: f for f in json.load(open(C/"AXIS_REVIEW_MANIFEST.json"))["frames_list"]}
    fr = man[FID]
    bgr = cv2.imread(str(ROOT/fr["image"]))
    H0, W0 = bgr.shape[:2]
    side = DH.CANON * 8 if hasattr(DH, "CANON") else 400
    net_in = 400                                   # A1 belief 50 -> stride 8
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sq = cv2.resize(rgb, (net_in, net_in), interpolation=cv2.INTER_AREA)
    ten = torch.from_numpy(sq).float().permute(2,0,1)[None].to(dev) / 255.0
    mean = torch.tensor([0.485,0.456,0.406], device=dev).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225], device=dev).view(1,3,1,1)
    ten = (ten - mean) / std

    with torch.no_grad():
        f50, belief, _ = a1(ten)
        print(f"f50 {tuple(f50.shape)}  belief {tuple(belief.shape)}")
        gt_, gr_, valid = DH.lattice()
        feats = DH.hypothesis_features(gt_[valid], gr_[valid])
        scores = model(f50, feats)                      # (1, 12, Hyp_valid)
        th, rh = DH.decode(scores, gt_[valid], gr_[valid],
                           torch.ones_like(gt_[valid], dtype=torch.bool))
    th = th[0].cpu().numpy(); rh = rh[0].cpu().numpy()
    conf = torch.softmax(scores[0].float(), -1).max(-1).values.cpu().numpy()
    print("role theta/rho (centred MAP100):")
    for i,(t,r,c) in enumerate(zip(th, rh, conf)):
        print(f"  edge{i:2d} {EDGES[i]}  theta {t:7.2f} deg  rho {r:8.2f}  p {c:.4f}")

    # ---- centred MAP100 -> 원본 픽셀
    # MAP100 격자의 중심이 이미지 중심, 1 MAP100 픽셀 = W0/DH.MAP 원본 픽셀
    s = W0 / float(DH.MAP)
    cx, cy = W0/2.0, H0/2.0

    def draw(ax, t_deg, r_map, color, lw=1.6, alpha=.9):
        a = np.deg2rad(t_deg); n = np.array([np.cos(a), np.sin(a)])
        p0 = np.array([cx, cy]) + n * (r_map * s)
        d = np.array([-n[1], n[0]])
        pts = np.array([p0 - d*2000, p0 + d*2000])
        ax.plot(pts[:,0], pts[:,1], color=color, lw=lw, alpha=alpha)

    # GT / 예측 코너
    gt = json.load(open(C/"GEOMETRY_RESOLVED_POSE_GT.json"))["frames"][FID]
    Ki = json.load(open(ROOT/fr["annotation"]))["camera_data"]["intrinsics"]
    K = np.array([[Ki["fx"],0,Ki["cx"]],[0,Ki["fy"],Ki["cy"]],[0,0,1]])
    dm = gt["physical_dimensions_m"]
    ha,hh,hb = dm["across"]/2, dm["height"]/2, dm["along"]/2
    X = np.array([[-ha,-hh,-hb],[ha,-hh,-hb],[ha,hh,-hb],[-ha,hh,-hb],
                  [-ha,-hh,hb],[ha,-hh,hb],[ha,hh,hb],[-ha,hh,hb]])
    R = np.asarray(gt["R_gt_representative"]); t3 = np.asarray(gt["t_gt"])
    Q = X@R.T+t3
    G = np.stack([K[0,0]*Q[:,0]/Q[:,2]+K[0,2], K[1,1]*Q[:,1]/Q[:,2]+K[1,2]],1)
    P = np.asarray(json.load(open(C/"predictions/R0.json"))["frames"][FID]["keypoints_xy"])[:8]

    fig, ax = plt.subplots(1, 3, figsize=(21, 6.4))
    for a_ in ax: a_.set_xticks([]); a_.set_yticks([])

    a_ = ax[0]; a_.imshow(rgb)
    cmap = plt.get_cmap("turbo")
    for i in range(12): draw(a_, th[i], rh[i], cmap(i/11.0))
    a_.set_xlim(0, W0); a_.set_ylim(H0, 0)
    a_.set_title(f"(A) the trained DOPE+Direct-Hough model's 12 lines\n"
                 f"ckpt DH_full/step_{sd['step']:05d}  ·  one (theta,rho) argmax per edge role",
                 fontsize=11)

    a_ = ax[1]; a_.imshow(rgb, alpha=.55)
    for i in range(12): draw(a_, th[i], rh[i], cmap(i/11.0), lw=1.3, alpha=.75)
    for i,j in EDGES: a_.plot(*zip(G[i],G[j]), color="#00d000", lw=2.2)
    a_.set_xlim(0, W0); a_.set_ylim(H0, 0)
    a_.set_title("(B) same lines vs GT cuboid edges (green)\n"
                 "does the model put its lines on the pallet?", fontsize=11)

    a_ = ax[2]
    a_.scatter(th, rh, c=[cmap(i/11.0) for i in range(12)], s=90, marker="X",
               edgecolor="k", linewidth=.4, label="model argmax (12 roles)")
    def tr(p, q):
        d = q-p; n = np.array([-d[1], d[0]], float); n /= max(np.linalg.norm(n),1e-9)
        return np.degrees(np.arctan2(n[1], n[0])) % 180.0, float(n@(p-np.array([cx,cy])))/s
    gtr = np.array([tr(G[i],G[j]) for i,j in EDGES])
    a_.scatter(gtr[:,0], gtr[:,1], s=110, marker="*", c="#00a000",
               edgecolor="k", linewidth=.4, label="GT cuboid edges")
    a_.set_xlabel("theta (deg)"); a_.set_ylabel("rho (MAP100 px, centred)")
    a_.set_xticks(np.arange(0,181,30)); a_.grid(alpha=.3); a_.legend(fontsize=9)
    a_.set_title("(C) the lattice the model searches\n"
                 "argmax over the valid (theta, rho) grid", fontsize=11)

    fig.suptitle(f"{FID}  ·  trained Direct-Hough head on the DOPE A1 trunk", fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.94])
    p = OUT/"direct_hough_model_lines.png"
    fig.savefig(p, dpi=130); plt.close(fig)

    json.dump({"frame_id": FID, "checkpoint": str(CKPT.relative_to(ROOT)),
               "step": int(sd["step"]), "seed": int(sd["seed"]),
               "roles": [{"edge": list(EDGES[i]), "theta_deg": float(th[i]),
                          "rho_map100": float(rh[i]), "softmax_max": float(conf[i])}
                         for i in range(12)],
               "note": "visualization only; SEALED session, not used for any selection",
               "figure": str(p.relative_to(ROOT))},
              open(OUT/"DIRECT_HOUGH_MODEL_LINES.json","w"), indent=2)
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
