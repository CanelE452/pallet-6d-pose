"""셋별 오버레이 — 모델이 낸 12 선을 이미지 위에 얹는다.

한 셋당 2 프레임.  프레임은 결과를 보고 고르지 않고 정렬 순서 앞에서 가져온다.
angle median 오름차순으로 셋을 배치해 "통하는 셋 -> 안 통하는 셋" 이 보이게 한다.
새 학습 0.
"""
from __future__ import annotations
import importlib.util, json, sys, glob
from pathlib import Path
import numpy as np, torch, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/pallet/results/hough_line_visual"
C = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
CKPT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation/"
        "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search/"
        "supporting_line_map/checkpoints/DH_full/step_08515.pth")
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
GRID = 50
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
PER_SET = 2


def _load(n, r):
    sp = importlib.util.spec_from_file_location(n, ROOT/r); m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m; sp.loader.exec_module(m); return m


def prep(bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (400,400)), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32)/255.0 - MEAN)/STD).transpose(2,0,1), rgb


def main():
    DH = _load("DH", "scripts/stage0/line/direct_hough_role_heatmap.py")
    dev = DH.DEV
    model = DH.DirectHoughModel().to(dev)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"], strict=True); model.eval()
    a1 = DH.V2.load_a1()
    gt_, gr_, valid = DH.lattice()
    feats = DH.hypothesis_features(gt_[valid], gr_[valid])
    gtv, grv = gt_[valid], gr_[valid]
    wb = json.load(open(OUT / "WHERE_BROKEN.json"))["populations"]

    # angle median 오름차순 (통하는 셋 -> 안 통하는 셋)
    order = sorted([k for k in wb if k != "REAL_clean"],
                   key=lambda k: wb[k]["angle_median"])
    order = ["REAL_clean"] + order          # 실사를 맨 앞 참조로

    items = []   # (set, rgb400, grid50, angle_med)
    for name in order:
        if name == "REAL_clean":
            clean = json.load(open(OUT/"CLEAN_FRAMES.json"))
            man = {f["frame_id"]: f for f in json.load(open(C/"AXIS_REVIEW_MANIFEST.json"))["frames_list"]}
            gtall = json.load(open(C/"GEOMETRY_RESOLVED_POSE_GT.json"))["frames"]
            got = 0
            for fid in clean:
                if fid not in gtall or got >= PER_SET: continue
                fr = man[fid]; img = cv2.imread(str(ROOT/fr["image"]))
                H0, W0 = img.shape[:2]
                Ki = json.load(open(ROOT/fr["annotation"]))["camera_data"]["intrinsics"]
                K = np.array([[Ki["fx"],0,Ki["cx"]],[0,Ki["fy"],Ki["cy"]],[0,0,1]])
                g = gtall[fid]; dm = g["physical_dimensions_m"]
                ha,hh,hb = dm["across"]/2, dm["height"]/2, dm["along"]/2
                X = np.array([[-ha,-hh,-hb],[ha,-hh,-hb],[ha,hh,-hb],[-ha,hh,-hb],
                              [-ha,-hh,hb],[ha,-hh,hb],[ha,hh,hb],[-ha,hh,hb]])
                Q = X@np.asarray(g["R_gt_representative"]).T + np.asarray(g["t_gt"])
                P2 = np.stack([K[0,0]*Q[:,0]/Q[:,2]+K[0,2], K[1,1]*Q[:,1]/Q[:,2]+K[1,2]],1)
                n, rgb = prep(img)
                items.append((name, n, rgb, np.stack([P2[:,0]*GRID/W0, P2[:,1]*GRID/H0],1),
                              wb[name]["angle_median"])); got += 1
        else:
            files = sorted(glob.glob(str(ROOT/"data/pallet/training_data"/name/"**"/"*.json"),
                                     recursive=True))
            files = [f for f in files if not Path(f).name.startswith("_")]
            got = 0
            for f in files:
                if got >= PER_SET: break
                png = f.replace(".json", ".png")
                if not Path(png).is_file(): continue
                d = json.load(open(f)); cam = d["camera_data"]
                pc = np.asarray(d["objects"][0].get("projected_cuboid", []), float)
                if pc.shape[0] < 8: continue
                img = cv2.imread(png)
                if img is None: continue
                W, H = float(cam["width"]), float(cam["height"])
                n, rgb = prep(img)
                items.append((name, n, rgb, np.stack([pc[:8,0]*GRID/W, pc[:8,1]*GRID/H],1),
                              wb[name]["angle_median"])); got += 1

    cols = len(order); rows = PER_SET
    fig, ax = plt.subplots(rows, cols, figsize=(3.5*cols, 3.7*rows))
    cmap = plt.get_cmap("turbo")
    per_col = {n: 0 for n in order}
    for name, n, rgb, grid, amed in items:
        c = order.index(name); r = per_col[name]; per_col[name] += 1
        if r >= rows: continue
        a_ = ax[r][c]
        im = torch.from_numpy(n[None]).to(dev)
        pack = {"grid": grid[None]}
        tc, rc, sup = DH.batch_rows(pack, EDGES)
        with torch.no_grad():
            f50, _, _ = a1(im)
            sc = model(f50, feats)
        live = torch.nonzero(sup[0]).flatten()
        th, rh = DH.decode(sc[0][live], gtv, grv, torch.ones_like(gtv, dtype=torch.bool))
        th = th.cpu().numpy(); rh = rh.cpu().numpy(); lv = live.cpu().numpy()
        # 400px 캔버스, MAP100 중심 좌표 -> 400px
        s = 400.0/float(DH.MAP); cx = cy = 200.0
        a_.imshow(rgb)
        for k, role in enumerate(lv):
            aa = np.deg2rad(th[k]); nv = np.array([np.cos(aa), np.sin(aa)])
            p0 = np.array([cx,cy]) + nv*(rh[k]*s); dvec = np.array([-nv[1], nv[0]])
            a_.plot(*np.array([p0-dvec*900, p0+dvec*900]).T, color=cmap(role/11.0),
                    lw=1.5, alpha=.9)
        gp = grid*(400.0/GRID)
        for i,j in EDGES:
            a_.plot([gp[i,0],gp[j,0]],[gp[i,1],gp[j,1]], color="#00e000", lw=2.4)
        a_.set_xlim(0,400); a_.set_ylim(400,0); a_.set_xticks([]); a_.set_yticks([])
        if r == 0:
            a_.set_title(f"{name.replace('_',' ')}\nangle median {amed:.2f} deg",
                         fontsize=10.5,
                         color="#1e6fd9" if name=="REAL_clean" else
                               ("#2a9d3f" if amed < 20 else "#c0392b"))
    fig.suptitle("Direct-Hough model lines (colour) vs GT cuboid edges (green), per population\n"
                 "left to right: best to worst.  the model never puts its lines on the pallet, "
                 "in synthetic or real", fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.93])
    p = OUT/"hough_overlay_by_set.png"
    fig.savefig(p, dpi=115); plt.close(fig)
    print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
