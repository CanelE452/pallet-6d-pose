"""어디에서 끊기나 — 여러 합성셋 + role 별 + 조건별로 같은 척도로 잰다.

측정 경로는 정본(batch_rows / decode / measure / summarise)을 그대로 쓴다.
GT 배선은 오라클(격자 최근접)로 매 셋마다 함께 검증한다 — 오라클이 0.5도 안이 아니면
그 셋의 수치는 신뢰하지 않는다.
"""
from __future__ import annotations
import importlib.util, json, sys, glob
from pathlib import Path
import numpy as np, torch, cv2

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/pallet/results/hough_line_visual"
C = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
CKPT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation/"
        "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search/"
        "supporting_line_map/checkpoints/DH_full/step_08515.pth")
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
GRID, N = 50, 96
SYNTH_SETS = ["mixed_v8_train", "v4_split_base", "aug_squash_v2",
              "aug_trunc_v2", "aug_scale_v2", "paper_4pallet_mask_v1"]
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def _load(n, r):
    sp = importlib.util.spec_from_file_location(n, ROOT/r); m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m; sp.loader.exec_module(m); return m


def prep(bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (400,400)), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32)/255.0 - MEAN)/STD).transpose(2,0,1)


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

    def evaluate(images, grids, tag):
        A, O, R, ORA = [], [], [], []
        for i in range(0, len(images), 16):
            im = torch.from_numpy(np.stack(images[i:i+16])).to(dev)
            pack = {"grid": np.stack(grids[i:i+16])}
            tc, rc, sup = DH.batch_rows(pack, EDGES)
            with torch.no_grad():
                f50, _, _ = a1(im)
                sc = model(f50, feats)
            for b in range(sc.shape[0]):
                live = torch.nonzero(sup[b]).flatten()
                if live.numel() == 0:
                    continue
                th, rh = DH.decode(sc[b][live], gtv, grv,
                                   torch.ones_like(gtv, dtype=torch.bool))
                a, o = DH.measure(th, rh, tc[b][live], rc[b][live])
                A.append(a); O.append(o); R.append(live.cpu().numpy())
                # 오라클: 격자 최근접 — GT 배선 검증
                _, _, sq = DH.line_distance(gtv, grv, tc[b][live], rc[b][live])
                best = sq.argmin(0)
                oa, _ = DH.measure(gtv[best], grv[best], tc[b][live], rc[b][live])
                ORA.append(oa)
        A, O, R, ORA = map(np.concatenate, (A, O, R, ORA))
        per_role = {int(r): round(float(np.median(A[R == r])), 2)
                    for r in range(12) if (R == r).sum() >= 5}
        res = {"population": tag, "n_frames": len(images), "n_roles": int(A.size),
               "angle_median": round(float(np.median(A)), 2),
               "angle_p90": round(float(np.percentile(A, 90)), 2),
               "offset_median": round(float(np.median(O)), 2),
               "frac_angle_gt5": round(float((A > 5).mean()), 3),
               "oracle_angle_median": round(float(np.median(ORA)), 3),
               "per_role_angle_median": per_role}
        ok = res["oracle_angle_median"] <= 0.5
        print(f"{tag:24s} n={len(images):4d}  angle med {res['angle_median']:6.2f} "
              f"p90 {res['angle_p90']:6.2f}  >5deg {res['frac_angle_gt5']*100:5.1f}%   "
              f"oracle {res['oracle_angle_median']:.3f} {'OK' if ok else '★배선의심'}", flush=True)
        return res

    report = {"checkpoint": str(CKPT.relative_to(ROOT)), "step": int(sd["step"]),
              "recorded_on_deleted_trainset": {"D2_LINE_DEV512_angle_median": 4.431,
                                               "OVERFIT32_angle_median": 0.598},
              "populations": {}}

    for name in SYNTH_SETS:
        files = sorted(glob.glob(str(ROOT/"data/pallet/training_data"/name/"**"/"*.json"),
                                 recursive=True))
        files = [f for f in files if not Path(f).name.startswith("_")][:N]
        im, gr = [], []
        for f in files:
            png = f.replace(".json", ".png")
            if not Path(png).is_file():
                continue
            d = json.load(open(f)); cam = d["camera_data"]
            W, H = float(cam["width"]), float(cam["height"])
            pc = np.asarray(d["objects"][0].get("projected_cuboid", []), float)
            if pc.shape[0] < 8:
                continue
            img = cv2.imread(png)
            if img is None:
                continue
            im.append(prep(img))
            gr.append(np.stack([pc[:8,0]*GRID/W, pc[:8,1]*GRID/H], 1))
        if len(im) < 8:
            print(f"{name:24s} 이미지 부족 ({len(im)}) — 건너뜀"); continue
        report["populations"][name] = evaluate(im, gr, name)

    # 실사
    clean = json.load(open(OUT/"CLEAN_FRAMES.json"))[:N]
    man = {f["frame_id"]: f for f in json.load(open(C/"AXIS_REVIEW_MANIFEST.json"))["frames_list"]}
    gtall = json.load(open(C/"GEOMETRY_RESOLVED_POSE_GT.json"))["frames"]
    im, gr = [], []
    for fid in clean:
        if fid not in gtall: continue
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
        im.append(prep(img)); gr.append(np.stack([P2[:,0]*GRID/W0, P2[:,1]*GRID/H0],1))
    report["populations"]["REAL_clean"] = evaluate(im, gr, "REAL_clean")

    (OUT/"WHERE_BROKEN.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print("\nwrote", (OUT/"WHERE_BROKEN.json").relative_to(ROOT))


if __name__ == "__main__":
    main()
