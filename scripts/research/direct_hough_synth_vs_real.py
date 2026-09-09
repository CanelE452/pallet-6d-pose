"""판별: DH 의 실사 실패가 sim2real 인가 아키텍처인가.

같은 체크포인트·같은 전처리·같은 오차 정의로 두 모집단을 잰다.
  SYNTH  paper_4pallet_mask_v1 (살아 있는 합성)   ★DH 학습셋 아님 — 그건 삭제됐다
  REAL   PAPER_EVAL 319 중 clean

오차는 학습 때와 같은 정본 정의를 import 한다 (V2.gt_lines · DH.line_distance).
합성에서 잘 되면 sim2real 이 유력, 합성에서도 안 되면 아키텍처가 유력하다.
새 학습 0.
"""
from __future__ import annotations
import importlib.util, json, sys, glob
from pathlib import Path
import numpy as np
import torch, cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/pallet/results/hough_line_visual"
C = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
CKPT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation/"
        "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search/"
        "supporting_line_map/checkpoints/DH_full/step_08515.pth")
SYNTH = ROOT / "data/pallet/training_data/paper_4pallet_mask_v1"
N = 128
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
GRID = 50


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def prep(bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (400, 400)), cv2.COLOR_BGR2RGB)
    n = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return n.transpose(2, 0, 1)


def main():
    DH = _load("DH", "scripts/stage0/line/direct_hough_role_heatmap.py")
    V2, dev = DH.V2, DH.DEV
    model = DH.DirectHoughModel().to(dev)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"], strict=True); model.eval()
    a1 = V2.load_a1()
    gt_, gr_, valid = DH.lattice()
    feats = DH.hypothesis_features(gt_[valid], gr_[valid])
    edges = list(DH.EDGES) if hasattr(DH, "EDGES") else \
        [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

    def run(images, grids, tag):
        """정본 evaluate_network 과 **같은 경로**를 쓴다 — 내가 조립하지 않는다.

        내가 앞서 틀린 세 곳:
          (1) support 마스크 없이 12 role 을 전부 쟀다 (화면에 안 걸리는 role 포함)
          (2) offset 에 CANON/MAP 배율을 안 곱했다
          (3) theta % 180 · rho 부호 뒤집기를 안 했다  -> batch_rows 가 하는 일
        """
        A, O, P = [], [], []
        for i in range(0, len(images), 16):
            im = torch.from_numpy(np.stack(images[i:i+16])).to(dev)
            pack = {"grid": np.stack(grids[i:i+16])}
            theta_c, rho_c, support = DH.batch_rows(pack, edges)
            with torch.no_grad():
                f50, _, _ = a1(im)
                sc = model(f50, feats)
            P.append(torch.softmax(sc.float(), -1).max(-1).values.cpu().numpy())
            for b in range(sc.shape[0]):
                live = torch.nonzero(support[b]).flatten()
                if live.numel() == 0:
                    continue
                th, rh = DH.decode(sc[b][live], gt_[valid], gr_[valid],
                                   torch.ones_like(gt_[valid], dtype=torch.bool))
                a, o = DH.measure(th, rh, theta_c[b][live], rho_c[b][live])
                A.append(a); O.append(o)
        A = np.concatenate(A); O = np.concatenate(O); P = np.concatenate(P)
        r = {"population": tag, "n_frames": len(images), "n_supported_roles": int(A.size),
             **DH.summarise(A, O),
             "softmax_max": {f"p{q}": round(float(np.percentile(P, q)), 5) for q in (50, 90)}}
        print(f"{tag:6s} n={len(images):4d} roles={A.size:5d}  "
              f"angle med {r['angle_median']:6.2f} p90 {r['angle_p90']:7.2f}   "
              f"offset med {r['offset_median']:5.2f} p90 {r['offset_p90']:5.2f}   "
              f">5deg {r['frac_angle_gt5']*100:5.1f}%   softmax p50 {r['softmax_max']['p50']:.5f}",
              flush=True)
        return r

    # ---- SYNTH
    files = sorted(glob.glob(str(SYNTH / "*.json")))[:N]
    si, sg = [], []
    for f in files:
        d = json.load(open(f))
        img = cv2.imread(f.replace(".json", ".png"))
        if img is None:
            continue
        cam = d["camera_data"]; W, H = float(cam["width"]), float(cam["height"])
        pc = np.asarray(d["objects"][0]["projected_cuboid"], float)[:8]
        si.append(prep(img))
        sg.append(np.stack([pc[:,0]*GRID/W, pc[:,1]*GRID/H], 1))
    synth = run(si, sg, "SYNTH")

    # ---- REAL (clean)
    clean = json.load(open(OUT / "CLEAN_FRAMES.json"))[:N]
    man = {f["frame_id"]: f for f in json.load(open(C/"AXIS_REVIEW_MANIFEST.json"))["frames_list"]}
    gtall = json.load(open(C/"GEOMETRY_RESOLVED_POSE_GT.json"))["frames"]
    ri, rg = [], []
    for fid in clean:
        if fid not in gtall:
            continue
        fr = man[fid]
        img = cv2.imread(str(ROOT/fr["image"]))
        H0, W0 = img.shape[:2]
        Ki = json.load(open(ROOT/fr["annotation"]))["camera_data"]["intrinsics"]
        K = np.array([[Ki["fx"],0,Ki["cx"]],[0,Ki["fy"],Ki["cy"]],[0,0,1]])
        g = gtall[fid]; dm = g["physical_dimensions_m"]
        ha,hh,hb = dm["across"]/2, dm["height"]/2, dm["along"]/2
        X = np.array([[-ha,-hh,-hb],[ha,-hh,-hb],[ha,hh,-hb],[-ha,hh,-hb],
                      [-ha,-hh,hb],[ha,-hh,hb],[ha,hh,hb],[-ha,hh,hb]])
        Q = X@np.asarray(g["R_gt_representative"]).T + np.asarray(g["t_gt"])
        P2 = np.stack([K[0,0]*Q[:,0]/Q[:,2]+K[0,2], K[1,1]*Q[:,1]/Q[:,2]+K[1,2]],1)
        ri.append(prep(img))
        rg.append(np.stack([P2[:,0]*GRID/W0, P2[:,1]*GRID/H0], 1))
    real = run(ri, rg, "REAL")

    rep = {"checkpoint": str(CKPT.relative_to(ROOT)), "step": int(sd["step"]),
           "scope_note": ("SYNTH 은 paper_4pallet_mask_v1 이며 DH 학습셋이 아니다 "
                          "(학습셋 pallet6d_v2_10k 은 2026-08-14 삭제). "
                          "따라서 in-domain 재현이 아니라 '합성 일반' 대조다."),
           "recorded_training_dev": {"population": "D2_LINE_DEV512 (합성, 삭제된 학습셋)",
                                     "angle_median_deg": 3.736, "offset_median": 1.966},
           "SYNTH": synth, "REAL": real}
    (OUT / "SYNTH_VS_REAL.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    print("\nwrote", (OUT/"SYNTH_VS_REAL.json").relative_to(ROOT))


if __name__ == "__main__":
    main()
