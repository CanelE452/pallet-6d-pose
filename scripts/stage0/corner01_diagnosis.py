"""stage0/corner01_diagnosis.py — diagnose WHY corners 0,1 "spike" even when fully visible.

[목적] surface-point 데이터 재생성(비싼 작업) 전에 0,1 튐 원인을 A/B/C 로 가른다. 추론만(학습 X).
  (A) box corner 라 허공 모서리 → 시각 단서 약함 → 처방=surface point 키포인트
  (B) 180° yaw 대칭으로 코너 순서 꼬임(flip 오염) → 처방=symmetry permutation
  (C) 원근 정상인데 직사각형 아니라 오해 → 처방=0,1 은 맞음, 필터 직사각형 강제가 오류

★ 핵심: per-corner(channel-i) 비교. pred8[i] = belief channel i = GT projected_cuboid[i].
   convention camera_dynamic_0123_v4: 0,1=top-front, 2,3=bottom-front, 4,5=top-rear, 6,7=bottom-rear.
   0,1 이 진단 대상, 2,3 이 대조군(같은 front 면, bottom).
   진단 4(flip)만 별도로 order-free 가 아닌 symmetry-permutation 매칭.

재사용(새 기하 없음):
  padding_compare.infer_one : reflect-pad 추론 + 올바른 좌표 역매핑 (pred8 channel-order 보존)
  filter_pr_camfacing       : load_model, extract_keypoints_from_belief, canonical_kp3d
  filter_flip_consistency   : infer_flip_kp, FLIP_PAIRS  (180° yaw = L-R flip+swap)
  tau_calibrate             : collect_val_frames (filter-val, SEAL guard 내장)
  eval_pvnet_heads          : collect_manual, collect_syn, collect_v3

데이터:
  real    = filter-val(outside+night) + manual GT 36 + capturecad 22
  synth   = training_data/val (mixed_v8 도메인) + v3 batch_009 (B2 학습 제외 확인)
  전처리  = reflect-pad(pad=100), B2 cad eval 과 일치.

출력: data/pallet/eval_results/stage17_corner01_diagnosis/
  corner01_diagnosis.json / overlay_*.png / summary.md

CPU smoke: --list_only  (프레임 수집·SEAL 확인만, 모델 X)
GPU full :  conda run -n pallet-pose python scripts/stage0/corner01_diagnosis.py
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

sys.path[:0] = [os.path.join(ROOT, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
from tau_calibrate import collect_val_frames  # noqa: E402
from eval_pvnet_heads import collect_manual, collect_syn, collect_v3  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results",
                       "stage17_corner01_diagnosis")
CAD_DIR = os.path.join(ROOT, "challenge", "data", "capturepalletcad_manual_gt")
V3_HELDOUT = os.path.join(ROOT, "challenge", "data", "training", "v3", "batch_009")
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "stage11_16k_B2_maskaux",
                               "final_net_epoch_0084.pth")
PAD = 100
THRESHOLD = 0.3
SPIKE_PX = 15.0   # corner-error above this = "spike" (개별 코너 오차)
GOOD_PX = 10.0
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# convention camera_dynamic_0123_v4
TOP_FRONT = [0, 1]     # 진단 대상
BOT_FRONT = [2, 3]     # 대조군 (같은 front 면)
TOP_REAR = [4, 5]
BOT_REAR = [6, 7]
FLIP_PAIRS = [(0, 1), (3, 2), (4, 5), (7, 6)]   # 180° yaw = L-R mirror swap


def collect_cad():
    out = []
    for jp in sorted(glob.glob(os.path.join(CAD_DIR, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(CAD_DIR, fid + ".png")
        if os.path.exists(ip):
            out.append(("cad", fid, jp, ip))
    return out


def load_gt(jp):
    """Return (gt8 [8,2], dims dict or None, pose 4x4 or None, cam intrinsics or None)."""
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    dims = o.get("dimensions_m")
    pose = np.array(o["pose_transform"], float) if "pose_transform" in o else None
    cam = d.get("camera_data", {}).get("intrinsics")
    return gt8, dims, pose, cam


def per_corner_err(pred8, gt8):
    """Channel-i aligned per-corner distance (NaN for undetected). (8,) array."""
    e = np.full(8, np.nan)
    for i in range(8):
        if np.isfinite(pred8[i, 0]):
            e[i] = float(np.linalg.norm(pred8[i] - gt8[i]))
    return e


def symmetry_swap_err(pred8, gt8):
    """180° yaw symmetry: swap GT by FLIP_PAIRS, then per-corner err vs pred.
    If a frame's pose is 180°-ambiguous and the net predicts the *other* labeling,
    this swapped err < original err (B hypothesis signature)."""
    swap = list(range(8))
    for a, b in FLIP_PAIRS:
        swap[a], swap[b] = b, a
    gt_sw = gt8[swap]
    return per_corner_err(pred8, gt_sw)


def gt_is_trapezoid(gt8):
    """진단 C 정량: GT 자체가 (원근 때문에) 사다리꼴인가.
    front face 윗변(0-1) vs 아랫변(3-2) 길이비 + 좌변(0-3) vs 우변(1-2) 길이비.
    직사각형(정면)=1.0, 비스듬할수록 1 에서 멀어짐.  GT 가 이미 비사각형이면
    '0,1 이 GT 와 다르다'를 '깨짐'으로 오판하면 안 됨(C)."""
    def L(a, b):
        return float(np.linalg.norm(gt8[a] - gt8[b]))
    top_w, bot_w = L(0, 1), L(3, 2)
    left_h, right_h = L(0, 3), L(1, 2)
    wr = max(top_w, bot_w) / max(min(top_w, bot_w), 1e-6)
    hr = max(left_h, right_h) / max(min(left_h, right_h), 1e-6)
    return {"front_w_ratio": round(wr, 3), "front_h_ratio": round(hr, 3)}


def top_corner_floating(dims, pose, cam):
    """진단 3 정량: 0,1(top-front) 코너가 '표면서 먼 허공'인가.
    팔레트는 height 만큼 두꺼운 직육면체. top 코너는 윗면(solid deck) 위에 있다.
    '허공 모서리' = 같은 (x,z) 의 실제 표면(top deck)에서 멀리 떨어진 점.
    top 코너는 deck 표면의 모서리이므로 표면거리 ≈ 0 (떠있지 않음).
    여기선 height(두께)를 반환 — height 가 작을수록(=납작) top/bot 코너가
    이미지상 가까워 0,1↔2,3 구분 단서가 약함 (depth 붕괴 취약)."""
    if dims is None:
        return None
    return {"height_m": dims.get("height"), "width_m": dims.get("width"),
            "depth_m": dims.get("depth")}


def make_overlay(img, pred8, gt8, fid, dom, e_corner):
    """예측(빨강)+GT(초록) 겹침. 0,1 강조(굵게+라벨), 2,3 대조(파랑 라벨)."""
    import cv2
    vis = img.copy()
    for i in range(8):
        g = tuple(int(v) for v in gt8[i])
        cv2.circle(vis, g, 4, (0, 200, 0), -1)            # GT green
        if np.isfinite(pred8[i, 0]):
            p = tuple(int(v) for v in pred8[i])
            cv2.circle(vis, p, 4, (0, 0, 220), -1)        # pred red
            col = (0, 165, 255) if i in TOP_FRONT else (200, 100, 0)
            if i in TOP_FRONT or i in BOT_FRONT:
                cv2.line(vis, p, g, col, 1)
                cv2.putText(vis, f"{i}", (p[0] + 5, p[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    e01 = np.nanmean([e_corner[0], e_corner[1]])
    e23 = np.nanmean([e_corner[2], e_corner[3]])
    cv2.putText(vis, f"{dom}/{fid} 01={e01:.1f} 23={e23:.1f}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--pad", type=int, default=PAD)
    ap.add_argument("--n_overlay", type=int, default=30)
    ap.add_argument("--list_only", action="store_true",
                    help="CPU-safe: 프레임 수집·SEAL 확인만 (모델 로드 X)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 프레임 수집 (real + synthetic) ──────────────────────────────
    real = collect_val_frames() + collect_manual() + collect_cad()
    syn = [("syn_val", f, j, i) for _, f, j, i in collect_syn(300)]
    syn += [("syn_v3h", f, j, i) for _, f, j, i in collect_v3(V3_HELDOUT, 200)]
    frames = real + syn
    by_dom = {}
    for d, *_ in frames:
        by_dom[d] = by_dom.get(d, 0) + 1
    print(f"[frames] total={len(frames)} {by_dom}")
    print(f"[real={len(real)} synthetic={len(syn)}]")
    if args.list_only:
        # SEAL self-check: no final-test frame must appear
        seal = {"capturepallet09", "capturepallet07",
                "capturenight09", "capturenight08"}
        leaks = [f for d, f, j, i in frames
                 if any(s in (j or "") for s in seal)]
        print(f"[SEAL] final-test leaks in collected frames: {len(leaks)}")
        return

    import cv2
    import torch
    from filter_pr_camfacing import extract_keypoints_from_belief
    from eval_pvnet_heads import load_pvnet_model
    from dope_predict_mp4_pad import pad_frame

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] {args.weights} ({device})")
    # B2 has aux seg head (mask_aux) -> strict load fails; use PVNet loader.
    model, numVec, numSeg = load_pvnet_model(args.weights, device)
    print(f"[model] numVec={numVec} numSeg={numSeg}")

    def infer_one(img, gt8, pad, threshold):
        """reflect-pad inference, channel-order-preserving pred8 (orig px).
        Mapping identical to padding_compare.infer_one; handles seg-head arity."""
        h0, w0 = img.shape[:2]
        proc = pad_frame(img, pad, "reflect")
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        ph, pw = proc.shape[:2]
        _sc = 400.0 / min(ph, pw)
        _nw = max(8, int(round(pw * _sc)) & ~7)
        _nh = max(8, int(round(ph * _sc)) & ~7)
        t = ((cv2.resize(rgb, (_nw, _nh)).astype(np.float32) / 255.0
              - MEAN) / STD)
        tensor = (torch.from_numpy(t.transpose(2, 0, 1)).float()
                  .unsqueeze(0).to(device))
        with torch.no_grad():
            out = model(tensor)
        beliefs = out[0]
        belief = beliefs[-1][0].cpu().numpy()
        kps_bel = extract_keypoints_from_belief(belief, threshold)
        bh, bw = belief.shape[1], belief.shape[2]
        ux, uy = _nw / bw, _nh / bh
        pred8 = np.full((8, 2), np.nan)
        n_det = 0
        for i, k in enumerate(kps_bel[:8]):
            if k[0] < 0:
                continue
            cx = (k[0] * ux) / _sc
            cy = (k[1] * uy) / _sc
            ox = cx * (w0 + 2 * pad) / w0 - pad
            oy = cy * (h0 + 2 * pad) / h0 - pad
            pred8[i] = (ox, oy)
            n_det += 1
        return n_det, pred8

    records = []
    for n, (dom, fid, jp, ip) in enumerate(frames):
        img = cv2.imread(ip)
        if img is None:
            continue
        gt8, dims, pose, cam = load_gt(jp)
        # in-frame visibility (V): GT corner 가 이미지 안 → 보이는 코너
        h0, w0 = img.shape[:2]
        inb = ((gt8[:, 0] >= 0) & (gt8[:, 0] < w0)
               & (gt8[:, 1] >= 0) & (gt8[:, 1] < h0))
        # reflect-pad 추론 (pred8 = channel-order 보존)
        n_det, pred8 = infer_one(img, gt8, args.pad, args.threshold)
        if n_det < 6:
            continue
        ec = per_corner_err(pred8, gt8)          # channel-i per-corner err
        ec_sw = symmetry_swap_err(pred8, gt8)    # 진단4: 180° swap err
        trap = gt_is_trapezoid(gt8)              # 진단C
        flt = top_corner_floating(dims, pose, cam)  # 진단3

        # 0,1 이 보이는데도 튀는가: 두 코너 다 in-frame & detected & err>SPIKE
        vis01 = bool(inb[0] and inb[1]
                     and np.isfinite(ec[0]) and np.isfinite(ec[1]))
        spike01 = vis01 and (np.nanmax([ec[0], ec[1]]) > SPIKE_PX)
        # 대조군 2,3 도 동시에 튀나
        spike23 = (np.isfinite(ec[2]) and np.isfinite(ec[3])
                   and np.nanmax([ec[2], ec[3]]) > SPIKE_PX)
        is_real = dom in ("outside", "night", "manual", "cad")

        rec = {
            "dom": dom, "fid": str(fid), "is_real": is_real,
            "n_det": int(n_det), "V_inframe": int(inb.sum()),
            "e_corner": [None if not np.isfinite(x) else round(float(x), 2)
                         for x in ec],
            "e_corner_swap": [None if not np.isfinite(x) else round(float(x), 2)
                              for x in ec_sw],
            "e01_mean": (None if not np.isfinite(np.nanmean(ec[:2]))
                         else round(float(np.nanmean(ec[:2])), 2)),
            "e23_mean": (None if not np.isfinite(np.nanmean(ec[2:4]))
                         else round(float(np.nanmean(ec[2:4])), 2)),
            "vis01": bool(vis01), "spike01": bool(spike01),
            "spike23": bool(spike23),
            "trap": trap, "floating": flt,
            "pred8": [[None, None] if not np.isfinite(pred8[i, 0])
                      else [round(float(pred8[i, 0]), 1),
                            round(float(pred8[i, 1]), 1)] for i in range(8)],
            "gt8": [[round(float(gt8[i, 0]), 1), round(float(gt8[i, 1]), 1)]
                    for i in range(8)],
        }
        records.append(rec)
        if (n + 1) % 50 == 0:
            print(f"  [{n+1}/{len(frames)}] records={len(records)}")

    # ── overlay: spike01 케이스 위주로 n_overlay 장 ─────────────────
    spikes = [r for r in records if r["spike01"]]
    others = [r for r in records if not r["spike01"]]
    pick = spikes[:args.n_overlay]
    if len(pick) < args.n_overlay:
        pick += others[:args.n_overlay - len(pick)]
    # rebuild image lookup
    img_lut = {(d, str(f)): i for d, f, j, i in frames}
    for r in pick:
        ip = img_lut.get((r["dom"], r["fid"]))
        if ip is None:
            continue
        img = cv2.imread(ip)
        if img is None:
            continue
        pred8 = np.array([[np.nan, np.nan] if p[0] is None else p
                          for p in r["pred8"]], float)
        gt8 = np.array(r["gt8"], float)
        ec = np.array([np.nan if x is None else x for x in r["e_corner"]], float)
        tag = "SPIKE" if r["spike01"] else "ok"
        vis = make_overlay(img, pred8, gt8, r["fid"], r["dom"], ec)
        outp = os.path.join(OUT_DIR,
                            f"overlay_{tag}_{r['dom']}_{r['fid']}.png")
        cv2.imwrite(outp, vis)

    summarize_and_write(records)


def _stats(vals):
    a = np.array([v for v in vals if v is not None], float)
    if a.size == 0:
        return {"n": 0, "median": None, "mean": None, "p90": None}
    return {"n": int(a.size), "median": round(float(np.median(a)), 2),
            "mean": round(float(np.mean(a)), 2),
            "p90": round(float(np.percentile(a, 90)), 2)}


def summarize_and_write(records):
    def split(pred):
        return [r for r in records if pred(r)]
    real = split(lambda r: r["is_real"])
    syn = split(lambda r: not r["is_real"])

    def corner_summary(recs, idx):
        return _stats([r["e_corner"][idx] for r in recs
                       if r["e_corner"][idx] is not None])

    summary = {
        "n_records": len(records),
        "n_real": len(real), "n_syn": len(syn),
        "spike_px": SPIKE_PX, "good_px": GOOD_PX,
        # 진단1: per-corner err, 0,1 vs 2,3 대조 (real / syn 분리)
        "diag1_per_corner": {
            "real": {f"c{i}": corner_summary(real, i) for i in range(8)},
            "syn": {f"c{i}": corner_summary(syn, i) for i in range(8)},
        },
        # 진단1b: 0,1 이 보이는데(vis01) 튀는 비율
        "diag1_visible_spike01": {
            "real": {
                "n_vis01": sum(1 for r in real if r["vis01"]),
                "n_spike01": sum(1 for r in real if r["spike01"]),
                "n_spike23_ctrl": sum(1 for r in real if r["spike23"]),
            },
            "syn": {
                "n_vis01": sum(1 for r in syn if r["vis01"]),
                "n_spike01": sum(1 for r in syn if r["spike01"]),
                "n_spike23_ctrl": sum(1 for r in syn if r["spike23"]),
            },
        },
        # 진단2: 합성에서도 0,1 튀나
        "diag2_syn_vs_real_e01": {
            "real": _stats([r["e01_mean"] for r in real]),
            "syn": _stats([r["e01_mean"] for r in syn]),
            "real_e23_ctrl": _stats([r["e23_mean"] for r in real]),
            "syn_e23_ctrl": _stats([r["e23_mean"] for r in syn]),
        },
        # 진단4: spike01 케이스에서 symmetry-swap 으로 0,1 오차 주나 (B)
        "diag4_symmetry": diag4(records),
        # 진단C: spike01 vs non-spike 의 GT trapezoid 비율 차이
        "diagC_trapezoid": diagC(records),
        # 진단3: floating(height) 분포
        "diag3_floating": diag3(records),
    }
    jp = os.path.join(OUT_DIR, "corner01_diagnosis.json")
    json.dump({"summary": summary,
               "records": records}, open(jp, "w"), indent=2)
    print(f"[save] {jp}")
    write_summary_md(summary)


def diag4(records):
    """spike01 케이스에서 e01(orig) vs e01_swap(180° yaw swap).
    swap 이 더 작으면(개선) = pose 는 맞는데 라벨 순서가 180° 꼬임(B).
    ★ real / syn 분리 필수: syn pool 은 gross-broken 프레임(e01>100)이 많아
    그런 데선 어떤 relabel 도 무작위 개선 → pooled frac 이 B 를 과장한다."""
    def compute(recs):
        rows = []
        n_better = 0
        for r in recs:
            ec = r["e_corner"]; sw = r["e_corner_swap"]
            if ec[0] is None or ec[1] is None or sw[0] is None or sw[1] is None:
                continue
            orig = float(np.mean([ec[0], ec[1]]))
            swapv = float(np.mean([sw[0], sw[1]]))
            rows.append((r["dom"], r["fid"], round(orig, 1), round(swapv, 1)))
            if swapv < orig - 2.0:
                n_better += 1
        return rows, n_better
    spk_real = [r for r in records if r["spike01"] and r["is_real"]]
    spk_syn = [r for r in records if r["spike01"] and not r["is_real"]]
    rr, rb = compute(spk_real)
    sr, sb = compute(spk_syn)
    n_syn_gross = sum(1 for r in spk_syn
                      if r["e01_mean"] is not None and r["e01_mean"] > 100)
    return {
        "real": {"n_spike01": len(rr), "n_swap_improves_2px": rb,
                 "frac_swap_improves": round(rb / len(rr), 3) if rr else None,
                 "examples": rr[:12]},
        "syn": {"n_spike01": len(sr), "n_swap_improves_2px": sb,
                "frac_swap_improves": round(sb / len(sr), 3) if sr else None,
                "n_gross_broken_e01gt100": n_syn_gross,
                "note": "gross-broken 많음 → frac 신뢰 불가"},
    }


def diagC(records):
    """spike01 케이스의 GT front-face 비율(직사각형이면 1.0).
    spike01 인데 GT 가 강한 사다리꼴(비율 큼)이면, pred 가 사실 GT 와
    멀지만 '직사각형 기준'으로 보면 틀려보이는 게 아니라 진짜 멀단 뜻.
    핵심: spike01 의 e01 이 큰데 GT 가 이미 사다리꼴이면 'pred 가 사각형으로
    예측해서' 멀어진 건지 확인 — 여기선 GT 비율 분포만 보고, 판정은 md 에서."""
    spikes = [r for r in records if r["spike01"]]
    non = [r for r in records if not r["spike01"] and r["vis01"]]

    def wr(recs):
        return _stats([r["trap"]["front_w_ratio"] for r in recs
                       if r.get("trap")])
    return {"spike01_front_w_ratio": wr(spikes),
            "nonspike_front_w_ratio": wr(non),
            "note": "GT 자체 사다리꼴 비율. spike 군이 더 높으면 C(원근) 의심."}


def diag3(records):
    """height(두께) 분포 — real 만 dims 보유.  납작할수록 top/bot 코너 근접."""
    h = [r["floating"]["height_m"] for r in records
         if r.get("floating") and r["floating"].get("height_m") is not None]
    if not h:
        return {"n": 0, "note": "no dims (synthetic val 은 dims 필드 없음)"}
    return {"n": len(h), "height_m_set": sorted(set(round(x, 3) for x in h)),
            "note": "팔레트 납작(height≈0.11m)→top/bot 코너 이미지상 근접→"
                    "0,1↔2,3 구분 단서 약함(depth 붕괴 취약). top 코너는 deck "
                    "표면 위라 '허공'은 아님."}


def write_summary_md(s):
    L = []
    L.append("# Corner 0,1 Spike Diagnosis (B2, reflect-pad, per-channel)\n")
    L.append(f"records={s['n_records']} (real={s['n_real']} syn={s['n_syn']}), "
             f"spike threshold={s['spike_px']}px per-corner.\n")
    L.append("convention: 0,1=top-front (진단대상), 2,3=bottom-front (대조군), "
             "4,5=top-rear, 6,7=bottom-rear.\n")

    d1 = s["diag1_per_corner"]
    L.append("## 진단1 — per-corner err (channel-i, px)\n```")
    L.append(f"{'corner':<8}{'real_med':>10}{'real_p90':>10}{'real_n':>8}"
             f"{'syn_med':>10}{'syn_p90':>10}{'syn_n':>8}")
    L.append("-" * 64)
    names = {0: "0 TF", 1: "1 TF", 2: "2 BF*", 3: "3 BF*",
             4: "4 TR", 5: "5 TR", 6: "6 BR", 7: "7 BR"}
    for i in range(8):
        r = d1["real"][f"c{i}"]; y = d1["syn"][f"c{i}"]
        L.append(f"{names[i]:<8}{str(r['median']):>10}{str(r['p90']):>10}"
                 f"{r['n']:>8}{str(y['median']):>10}{str(y['p90']):>10}{y['n']:>8}")
    L.append("```")
    L.append("(TF=top-front 진단대상, BF*=bottom-front 대조군)\n")

    vs = s["diag1_visible_spike01"]
    L.append("## 진단1b — 0,1 보이는데 튀나 (vis01 & per-corner>spike)\n```")
    L.append(f"{'set':<8}{'n_vis01':>10}{'n_spike01':>12}{'n_spike23_ctrl':>16}")
    for k in ("real", "syn"):
        v = vs[k]
        L.append(f"{k:<8}{v['n_vis01']:>10}{v['n_spike01']:>12}"
                 f"{v['n_spike23_ctrl']:>16}")
    L.append("```\n")

    d2 = s["diag2_syn_vs_real_e01"]
    L.append("## 진단2 — 합성 vs real (e01 mean per frame, px)\n```")
    L.append(f"{'metric':<16}{'n':>6}{'median':>10}{'p90':>10}")
    for lab, key in [("real e01", "real"), ("syn e01", "syn"),
                     ("real e23 ctrl", "real_e23_ctrl"),
                     ("syn e23 ctrl", "syn_e23_ctrl")]:
        v = d2[key]
        L.append(f"{lab:<16}{v['n']:>6}{str(v['median']):>10}{str(v['p90']):>10}")
    L.append("```\n")

    d4 = s["diag4_symmetry"]
    L.append("## 진단4 — 180° yaw symmetry swap (B 검증) — real/syn 분리\n```")
    rr = d4["real"]; sr = d4["syn"]
    L.append(f"REAL: n_spike01={rr['n_spike01']} swap_improves_2px="
             f"{rr['n_swap_improves_2px']} frac={rr['frac_swap_improves']}")
    L.append(f"SYN : n_spike01={sr['n_spike01']} swap_improves_2px="
             f"{sr['n_swap_improves_2px']} frac={sr['frac_swap_improves']} "
             f"(gross_broken e01>100: {sr['n_gross_broken_e01gt100']} → 신뢰X)")
    L.append("REAL examples (dom fid e01_orig e01_swap):")
    for ex in rr["examples"][:10]:
        L.append(f"  {ex[0]:<12}{ex[1]:<22}{ex[2]:>7}{ex[3]:>9}")
    L.append("```\n")

    dC = s["diagC_trapezoid"]
    L.append("## 진단C — GT front-face 사다리꼴 비율 (1.0=직사각형)\n```")
    sp = dC["spike01_front_w_ratio"]; ns = dC["nonspike_front_w_ratio"]
    L.append(f"spike01    : median={sp['median']} p90={sp['p90']} n={sp['n']}")
    L.append(f"non-spike  : median={ns['median']} p90={ns['p90']} n={ns['n']}")
    L.append("```\n")

    d3 = s["diag3_floating"]
    L.append("## 진단3 — floating/두께\n```")
    L.append(json.dumps(d3, ensure_ascii=False))
    L.append("```\n")

    L.append("## 판정 (A/B/C) — 자동 가이드\n")
    L.append(_verdict(s))
    mp = os.path.join(OUT_DIR, "summary.md")
    open(mp, "w").write("\n".join(L))
    print(f"[save] {mp}")
    print("\n".join(L))


def _verdict(s):
    """수치 기반 자동 판정 가이드(휴리스틱). 최종은 사람이 overlay 로 확인."""
    out = []
    d1 = s["diag1_per_corner"]
    rm01 = np.nanmean([d1["real"]["c0"]["median"] or np.nan,
                       d1["real"]["c1"]["median"] or np.nan])
    rm23 = np.nanmean([d1["real"]["c2"]["median"] or np.nan,
                       d1["real"]["c3"]["median"] or np.nan])
    ym01 = np.nanmean([d1["syn"]["c0"]["median"] or np.nan,
                       d1["syn"]["c1"]["median"] or np.nan])
    out.append(f"- real 0,1 med≈{rm01:.1f} vs 2,3 med≈{rm23:.1f}; "
               f"syn 0,1 med≈{ym01:.1f}")
    # 진단2 게이트: 합성에서도 튀나
    if not np.isnan(ym01) and ym01 > 1.5 * (rm23 if not np.isnan(rm23) else 1):
        out.append("- 합성에서도 0,1 튐 → 구조/convention 문제(A·B 쪽). "
                   "sim2real 전이 단독 원인 아님.")
    else:
        out.append("- 합성 0,1 정상, real 만 튐 → sim2real 전이 성분 큼.")
    # 진단4 게이트: B — REAL 만으로 판정 (syn 은 gross-broken 오염)
    d4 = s["diag4_symmetry"]
    fr = d4["real"].get("frac_swap_improves")
    if fr is not None and fr >= 0.4:
        out.append(f"- (B) REAL 에서 180° swap 이 spike01 의 {fr*100:.0f}% 개선 → "
                   "대칭 순서 꼬임(flip 오염) 유력. 처방=symmetry permutation.")
    elif fr is not None:
        out.append(f"- (B) REAL swap 개선 {fr*100:.0f}% 뿐 (swap 후 오히려 악화) "
                   f"→ 180° 대칭 꼬임 아님. (syn frac={d4['syn']['frac_swap_improves']}"
                   f" 은 gross-broken 오염 artifact, 무시.)")
    # 진단C 게이트
    dC = s["diagC_trapezoid"]
    sp = dC["spike01_front_w_ratio"]["median"]
    ns = dC["nonspike_front_w_ratio"]["median"]
    if sp and ns and sp > ns * 1.3:
        out.append(f"- (C) spike 군 GT 사다리꼴 비율 {sp} >> non {ns} → "
                   "원근 강한 뷰에 spike 집중. 0,1 이 사실 맞고 필터 직사각형 "
                   "강제가 오류일 수 있음.")
    out.append("- (A) 위 B·C 가 약하고 0,1 이 보이는데도 per-corner 만 크면 "
               "→ box corner 시각단서 약함. surface-point 키포인트 검토.")
    out.append("\n★ 표본 작으면 예비. overlay 로 GT 정상/예측만 튐 여부 눈검증 필수.")
    return "\n".join(out)


if __name__ == "__main__":
    main()
