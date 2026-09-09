"""가림·잘림 없는 clean 프레임에서 A1+Direct-Hough 가 내는 선.

프레임 선택은 내 눈이 아니라 정본 조건 라벨(occlusion=none & truncation=none)로 한다.
세션이 겹치지 않게 앞에서부터 하나씩 뽑는다 — 결과를 보고 고르지 않는다.
새 학습 0.  체크포인트 1 회 추론.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/pallet/results/hough_line_visual"
C = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
CKPT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation/"
        "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search/"
        "supporting_line_map/checkpoints/DH_full/step_08515.pth")
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
N_SHOW = 4


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def main():
    DH = _load("DH", "scripts/stage0/line/direct_hough_role_heatmap.py")
    dev = DH.DEV
    model = DH.DirectHoughModel().to(dev)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"], strict=True); model.eval()
    a1 = DH.V2.load_a1()

    clean = json.load(open(OUT / "CLEAN_FRAMES.json"))
    man = {f["frame_id"]: f for f in json.load(open(C/"AXIS_REVIEW_MANIFEST.json"))["frames_list"]}
    gtall = json.load(open(C/"GEOMETRY_RESOLVED_POSE_GT.json"))["frames"]
    # 세션이 겹치지 않게 앞에서부터 (결과를 보고 고르지 않는다)
    picked, seen = [], set()
    for fid in clean:
        s = man[fid]["session_id"]
        if s in seen or fid not in gtall:
            continue
        seen.add(s); picked.append(fid)
        if len(picked) == N_SHOW:
            break
    print("선택된 프레임:", picked)

    gt_, gr_, valid = DH.lattice()
    feats = DH.hypothesis_features(gt_[valid], gr_[valid])
    mean = torch.tensor([0.485,0.456,0.406], device=dev).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225], device=dev).view(1,3,1,1)

    fig, ax = plt.subplots(2, N_SHOW, figsize=(5.0*N_SHOW, 8.2))
    rows = []
    for c_, fid in enumerate(picked):
        fr = man[fid]
        bgr = cv2.imread(str(ROOT/fr["image"])); H0, W0 = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sq = cv2.resize(rgb, (400,400), interpolation=cv2.INTER_AREA)
        ten = torch.from_numpy(sq).float().permute(2,0,1)[None].to(dev)/255.0
        ten = (ten-mean)/std
        with torch.no_grad():
            f50, _, _ = a1(ten)
            scores = model(f50, feats)
            th, rh = DH.decode(scores, gt_[valid], gr_[valid],
                               torch.ones_like(gt_[valid], dtype=torch.bool))
        th = th[0].cpu().numpy(); rh = rh[0].cpu().numpy()
        conf = torch.softmax(scores[0].float(), -1).max(-1).values.cpu().numpy()

        g = gtall[fid]
        Ki = json.load(open(ROOT/fr["annotation"]))["camera_data"]["intrinsics"]
        K = np.array([[Ki["fx"],0,Ki["cx"]],[0,Ki["fy"],Ki["cy"]],[0,0,1]])
        dm = g["physical_dimensions_m"]
        ha,hh,hb = dm["across"]/2, dm["height"]/2, dm["along"]/2
        X = np.array([[-ha,-hh,-hb],[ha,-hh,-hb],[ha,hh,-hb],[-ha,hh,-hb],
                      [-ha,-hh,hb],[ha,-hh,hb],[ha,hh,hb],[-ha,hh,hb]])
        Q = X@np.asarray(g["R_gt_representative"]).T + np.asarray(g["t_gt"])
        G = np.stack([K[0,0]*Q[:,0]/Q[:,2]+K[0,2], K[1,1]*Q[:,1]/Q[:,2]+K[1,2]],1)

        s = W0/float(DH.MAP); cx, cy = W0/2.0, H0/2.0
        cmap = plt.get_cmap("turbo")
        for r_, a_ in enumerate((ax[0][c_], ax[1][c_])):
            a_.imshow(rgb, alpha=1.0 if r_==0 else 0.55)
            for i in range(12):
                aa = np.deg2rad(th[i]); n = np.array([np.cos(aa), np.sin(aa)])
                p0 = np.array([cx,cy]) + n*(rh[i]*s); d = np.array([-n[1], n[0]])
                a_.plot(*np.array([p0-d*2000, p0+d*2000]).T, color=cmap(i/11.0),
                        lw=1.4, alpha=.85)
            if r_==1:
                for i,j in EDGES: a_.plot(*zip(G[i],G[j]), color="#00d000", lw=2.2)
            a_.set_xlim(0,W0); a_.set_ylim(H0,0); a_.set_xticks([]); a_.set_yticks([])
        ax[0][c_].set_title(f"{fr['session_id']}\nsoftmax max {conf.max():.4f}  "
                            f"mean {conf.mean():.4f}", fontsize=10)
        rows.append({"frame_id": fid, "session_id": fr["session_id"],
                     "softmax_max": float(conf.max()), "softmax_mean": float(conf.mean()),
                     "theta_deg": [round(float(x),2) for x in th],
                     "rho_map100": [round(float(x),2) for x in rh]})
        print(f"  {fid:42s} softmax max {conf.max():.4f} mean {conf.mean():.4f}")

    ax[0][0].set_ylabel("model lines only", fontsize=11)
    ax[1][0].set_ylabel("+ GT cuboid (green)", fontsize=11)
    fig.suptitle("A1 + Direct-Hough on CLEAN frames (occlusion=none & truncation=none)\n"
                 "frames chosen from the canonical condition labels, one per session — "
                 "not picked by looking at the result", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.93])
    p = OUT/"direct_hough_clean_frames.png"
    fig.savefig(p, dpi=125); plt.close(fig)
    json.dump({"checkpoint": str(CKPT.relative_to(ROOT)), "n_clean_pool": len(clean),
               "selection": "occlusion=none & truncation=none, first per session",
               "frames": rows, "figure": str(p.relative_to(ROOT))},
              open(OUT/"DIRECT_HOUGH_CLEAN.json","w"), indent=2)
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
