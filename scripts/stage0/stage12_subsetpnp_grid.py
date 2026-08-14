"""stage12_subsetpnp_grid.py — 무학습 subset-PnP decode grid (B3_final 위).

목적: B3_final ckpt 그대로(학습 X) postprocess(decoder/PnP 파라미터)만으로 V<8
(truncation) 프레임에서 pose 회복을 얼마나 끌어올리나. 추론은 프레임당 1회만
(belief→corner+score 캐시) 후 그 위에서 decode grid 만 반복.

핵심:
  - solve_pose(annotate_pnp) = order-free ITERATIVE/IPPE + 24-sym refine +
    positive-depth(t[2]>0, z_far_limit) + degenerate-reject. 이미 subset(None 포함)
    입력을 받고 len(valid)>=4 면 푼다. → 이게 곧 "subset-PnP solver".
  - decoder = "어느 corner subset 을 solve_pose 에 넣나" 의 정책:
      D0 = full-keypoint gate (n_det>=6 → 전부) [대조, current]
      D1/D2/D3 = subset-PnP, min_pts = 6/5/4 : score 상위 + in-frame corner 만
      D4 = D3 + centroid(9th) 포함
  - sweep: score_thr {0,.25τ,.5τ,.75τ,1τ}(τ=0.3) · reproj_reject {5,10,15,20}px ·
    pos-depth ON · refine ON. (RANSAC hyp 는 이 solver 에 없음 — N/A, 보고에 명시.)

"검출/성공" = subset-PnP 가 valid pose 반환(subset-reproj<reject & positive depth).

reproj 두 종류 분리:
  - subset_reproj = solve_pose 가 input subset 점에서 잰 reproj (= reject gate 용,
    GT 불필요 → 실제 배포서 쓸 수 있는 신호). subset 점 적으면 낙관적일 수 있음.
  - gt_reproj    = projected_all vs 전체 8 GT corner median (정직한 품질, GT 필요,
    eval 전용).

config 선택은 ★synthetic V<8 val(v3/batch_009 + addon_v1_val 의 num_corners_in_frame<8)
에서만. real 은 최종 보고용.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

import cv2, torch
from models import DopeNetwork
from filter_pr_camfacing import extract_keypoints_from_belief
from eval_pvnet_heads import preprocess, belief_to_orig, collect_manual
from tau_calibrate import collect_val_frames
import annotate_pnp as APNP

TAU = 0.3                       # belief threshold (extract_keypoints_from_belief)
B3 = os.path.join(ROOT, "weights/stage11_16k_B3_replay/net_epoch_0074.pth")
SYN_V3_VAL = os.path.join(ROOT, "challenge/data/02_synthetic/training/v3/batch_009")
SYN_ADDON_VAL = os.path.join(ROOT, "challenge/data/02_synthetic/training/addon_v1_val")
OUT = os.path.join(ROOT, "data/pallet/eval_results/stage12_subsetpnp")

# grid axes
SCORE_THRS = [0.0, 0.25 * TAU, 0.5 * TAU, 0.75 * TAU, 1.0 * TAU]
REPROJ_REJECTS = [5.0, 10.0, 15.0, 20.0]
DECODERS = ["D0", "D1", "D2", "D3", "D4"]   # D0=full gate; D1/2/3 min_pts 6/5/4; D4=D3+ctr
MIN_PTS = {"D1": 6, "D2": 5, "D3": 4, "D4": 4}


def load_model(wp, device):
    state = torch.load(wp, map_location=device)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    m = DopeNetwork(numVec=0, numSeg=0)
    m.load_state_dict(state, strict=False)
    return m.to(device).eval()


def K_from_json(d):
    it = d["camera_data"]["intrinsics"]
    return np.array([[it["fx"], 0, it["cx"]],
                     [0, it["fy"], it["cy"]], [0, 0, 1]], float)


def dims_from_json(d):
    o = d["objects"][0]
    dm = o.get("dimensions_m")
    if dm:
        return (float(dm["width"]), float(dm["depth"]), float(dm["height"]))
    cd = o.get("cuboid_dimensions_m")
    if cd:
        return (float(cd[0]), float(cd[1]), float(cd[2]))
    return APNP.PALLET_DIMS


def gt8_inframe(gt8, W, H):
    return np.array([(0 <= gt8[i, 0] < W) and (0 <= gt8[i, 1] < H)
                     for i in range(8)], bool)


# ─── per-frame inference (ONCE) → cache corners+score+gt ────────────────
def infer_cache(model, jp, ip, device):
    """프레임당 1회 추론. 반환: cache dict (corner px+score, gt, K, dims)."""
    img = cv2.imread(ip)
    if img is None:
        return None
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    ctr_gt = np.array(o["projected_cuboid_centroid"], float)
    H, W = img.shape[:2]
    K = K_from_json(d)
    dims = dims_from_json(d)
    # V = in-frame GT corner 수 (synthetic 은 num_corners_in_frame 사용, 일치 확인됨)
    ncif = o.get("num_corners_in_frame")
    if ncif is None or ncif < 0:
        V = int(gt8_inframe(gt8, W, H).sum())
    else:
        V = int(ncif)

    tensor, nw, nh, sc = preprocess(img)
    with torch.no_grad():
        beliefs, _ = model(tensor.to(device))
    belief = beliefs[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    kpb = extract_keypoints_from_belief(belief, TAU)  # 9 x (wx,wy,score), -1 if none

    # 9 corner px (orig) + score; centroid = idx 8
    corners = []   # list of (px or None, score)
    for i in range(9):
        bx, by, sco = kpb[i]
        if bx < 0:
            corners.append((None, float(sco)))
            continue
        ox, oy = belief_to_orig(bx, by, bw, bh, nw, nh, sc)
        inb = (0 <= ox < W) and (0 <= oy < H)
        corners.append(([float(ox), float(oy)] if inb else None, float(sco)))

    return {"H": H, "W": W, "K": K, "dims": dims, "gt8": gt8,
            "ctr_gt": ctr_gt, "V": V, "corners": corners}


# ─── FAST subset-PnP core (per-index correspondence known from belief channel) ──
# ★ DOPE belief channel i = corner i → correspondence 가 이미 결정됨. solve_pose 의
#   48-candidate(24-sym × 2-dim) symmetry search 불필요(그건 수동 annotation 용 — 거기선
#   클릭 순서가 모호). 여기선 단일 SOLVEPNP_ITERATIVE × 2-dim(110/130 front) 만.
#   → 호출당 ~100x 빠름. flat 물체 ITERATIVE (checklist 준수).
_DIMS_CACHE = {}


def _kp3d_for(dims):
    if dims not in _DIMS_CACHE:
        _DIMS_CACHE[dims] = APNP.make_pallet_keypoints_3d(*dims)
    return _DIMS_CACHE[dims]


def _solve_fast_single(kps9, K, dims):
    """단일 dim ITERATIVE PnP. kps9: length-9 list (None=missing). 반환 pose dict or None."""
    kp3d = _kp3d_for(dims)
    idx = [i for i in range(9) if kps9[i] is not None]
    if len(idx) < 4:
        return None
    obj = np.array([kp3d[i] for i in idx], np.float64)
    img = np.array([kps9[i] for i in idx], np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    # subset reproj (입력 점 기준, gate 용)
    proj = (kp3d[idx] @ R.T + t) @ K.T
    proj = proj[:, :2] / proj[:, 2:3]
    sub_rep = float(np.median(np.linalg.norm(proj - img, axis=1)))
    return {"R": R, "t": t, "dims": dims, "subset_reproj": sub_rep}


def solve_fast(kps9, K, dims):
    """110/130-front 두 dim ITERATIVE PnP 중 subset_reproj 최소 + positive-depth 채택."""
    dims_a = APNP.PALLET_DIMS
    dims_b = (dims_a[1], dims_a[0], dims_a[2])
    cands = []
    for dm in (dims_a, dims_b):
        p = _solve_fast_single(kps9, K, dm)
        if p is not None and p["t"][2] > 0:
            cands.append(p)
    if not cands:
        return None
    return min(cands, key=lambda p: p["subset_reproj"])


def gt_reproj_full(pose, gt8):
    """채택 pose 의 8 corner full projection vs 전체 8 GT corner median (정직한 품질)."""
    kp3d = _kp3d_for(tuple(pose["dims"]))[:8]
    pr = (kp3d @ pose["R"].T + pose["t"]) @ pose["_K"].T
    pr = pr[:, :2] / pr[:, 2:3]
    return float(np.median(np.linalg.norm(pr - gt8, axis=1)))


def posdepth_ok(pose):
    kp3d = _kp3d_for(tuple(pose["dims"]))
    z = (kp3d @ pose["R"].T + pose["t"])[:, 2]
    return bool((z > 0).all() and pose["t"][2] > 0)


def select_subset(corners, decoder, score_thr):
    """decoder 정책으로 solve_pose 입력 kps9 (length 9, None 포함) 구성.
    반환: (kps9, n_used_corners). 못 풀 조건이면 (None, n)."""
    # in-frame & score>=thr 인 corner index (0..7)
    cand = [(i, corners[i][0], corners[i][1]) for i in range(8)
            if corners[i][0] is not None and corners[i][1] >= score_thr]
    if decoder == "D0":
        # full gate: in-frame corner 전부, n>=6 이어야
        if len(cand) < 6:
            return None, len(cand)
        kps9 = [corners[i][0] if (corners[i][0] is not None
                and corners[i][1] >= score_thr) else None for i in range(8)]
        kps9.append(None)   # D0 은 centroid 미사용 (current eval 과 동일하게 corner gate)
        return kps9, len(cand)
    # D1/D2/D3/D4 : score 상위 + in-frame, min_pts gate
    mp = MIN_PTS[decoder]
    if len(cand) < mp:
        return None, len(cand)
    # score 상위로 정렬 (전부 통과해도 무방 — solver 가 다 쓰면 됨)
    cand_sorted = sorted(cand, key=lambda x: -x[2])
    use_idx = set(i for i, _, _ in cand_sorted)
    kps9 = [corners[i][0] if i in use_idx else None for i in range(8)]
    # centroid
    if decoder == "D4" and corners[8][0] is not None and corners[8][1] >= score_thr:
        kps9.append(corners[8][0])
    else:
        kps9.append(None)
    return kps9, len(cand)


def decode_one(cache, decoder, score_thr, reproj_reject):
    """캐시된 corner 로 1개 config 의 subset-PnP 시도 (FAST core).
    반환: dict(success, subset_reproj, gt_reproj, n_used). success=positive depth &
    subset_reproj<reject."""
    kps9, n_used = select_subset(cache["corners"], decoder, score_thr)
    if kps9 is None:
        return {"success": 0, "subset_reproj": np.inf, "gt_reproj": np.inf,
                "n_used": n_used}
    pose = solve_fast(kps9, cache["K"], cache["dims"])
    if pose is None:
        return {"success": 0, "subset_reproj": np.inf, "gt_reproj": np.inf,
                "n_used": n_used}
    pose["_K"] = cache["K"]
    sub_rep = pose["subset_reproj"]
    gt_rep = gt_reproj_full(pose, cache["gt8"])
    success = 1 if (posdepth_ok(pose) and sub_rep < reproj_reject) else 0
    return {"success": success, "subset_reproj": sub_rep, "gt_reproj": gt_rep,
            "n_used": n_used}


# ─── frame collectors ───────────────────────────────────────────────────
def collect_syn_vlt8():
    """synthetic V<8 held-out: v3/batch_009 + addon_v1_val 의 num_corners_in_frame<8."""
    out = []
    for tag, dd in [("v3b009", SYN_V3_VAL), ("addon_val", SYN_ADDON_VAL)]:
        for jp in sorted(glob.glob(os.path.join(dd, "*.json"))):
            fid = os.path.splitext(os.path.basename(jp))[0]
            ip = os.path.join(dd, fid + ".png")
            if not os.path.exists(ip):
                continue
            o = json.load(open(jp))["objects"][0]
            v = o.get("num_corners_in_frame", -1)
            if 0 <= v < 8:
                out.append((tag, fid, jp, ip))
    return out


def collect_syn_v8(n_cap=400):
    """synthetic V=8 held-out (보존 체크용) — v3/batch_009 V=8 subsample."""
    out = []
    for jp in sorted(glob.glob(os.path.join(SYN_V3_VAL, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(SYN_V3_VAL, fid + ".png")
        if not os.path.exists(ip):
            continue
        o = json.load(open(jp))["objects"][0]
        if o.get("num_corners_in_frame", -1) == 8:
            out.append(("v3b009", fid, jp, ip))
    if n_cap and len(out) > n_cap:
        idx = np.linspace(0, len(out) - 1, n_cap).round().astype(int)
        out = [out[int(i)] for i in sorted(set(idx))]
    return out


def collect_real():
    """real = filter-val(outside+night) + manual GT (eval_stage11 과 동일 소스)."""
    return collect_val_frames() + collect_manual()


# ─── caching loop ───────────────────────────────────────────────────────
def build_caches(model, frames, device, label):
    caches = []
    for i, (dom, fid, jp, ip) in enumerate(frames):
        c = infer_cache(model, jp, ip, device)
        if c is None:
            continue
        c["dom"] = dom
        c["fid"] = fid
        caches.append(c)
        if (i + 1) % 100 == 0:
            print(f"  [{label}] cached {i+1}/{len(frames)}")
    print(f"  [{label}] total cached = {len(caches)}")
    return caches


# honest quality bar: full-GT reproj 가 이 px 미만이면 진짜 회복 (false-accept 구분).
HONEST_PX = 15.0


def sweep_synthetic(caches_vlt8, caches_v8):
    """모든 (decoder, score_thr, reproj_reject) config 에 대해 V<8 회복 / V=8 보존 측정.

    두 종류 success 추적:
      gate_ok  = positive depth & subset_reproj<reject (배포 시 쓸 신호, GT 불필요)
      honest_ok= gate_ok AND full-GT reproj < HONEST_PX (진짜 옳은 pose, GT 필요)
    false_accept_pct = gate 통과인데 honest 실패 비율 (배포서 잘못 믿게 되는 위험).
    반환: list of config-result dicts."""
    def pct(lst):
        return round(100 * np.mean(lst), 1) if lst else 0.0

    def metrics_for(caches, dec, sthr, rej):
        per_v_gate = {}; per_v_honest = {}
        gate_all = []; honest_all = []; gtreps = []; false_acc = []
        for c in caches:
            r = decode_one(c, dec, sthr, rej)
            g = r["success"]
            h = 1 if (g and np.isfinite(r["gt_reproj"])
                      and r["gt_reproj"] < HONEST_PX) else 0
            per_v_gate.setdefault(c["V"], []).append(g)
            per_v_honest.setdefault(c["V"], []).append(h)
            gate_all.append(g); honest_all.append(h)
            if g:
                false_acc.append(0 if h else 1)
            if h:
                gtreps.append(r["gt_reproj"])
        return {"per_v_gate": per_v_gate, "per_v_honest": per_v_honest,
                "gate_all": gate_all, "honest_all": honest_all,
                "gtreps": gtreps, "false_acc": false_acc}

    results = []
    for dec in DECODERS:
        for sthr in SCORE_THRS:
            for rej in REPROJ_REJECTS:
                mv = metrics_for(caches_vlt8, dec, sthr, rej)
                m8 = metrics_for(caches_v8, dec, sthr, rej)

                def vp(d, v):  # honest per V
                    return pct(d["per_v_honest"].get(v, []))

                def vg(d, v):  # gate per V
                    return pct(d["per_v_gate"].get(v, []))
                results.append({
                    "decoder": dec, "score_thr": round(sthr, 4),
                    "reproj_reject": rej,
                    "vlt8_n": len(mv["gate_all"]),
                    # honest = 진짜 회복 (판단지표의 "회복")
                    "vlt8_recover_pct": pct(mv["honest_all"]),
                    "vlt8_gate_pct": pct(mv["gate_all"]),
                    "vlt8_false_accept_pct": pct(mv["false_acc"]),
                    "v7_pct": vp(mv, 7), "v7_n": len(mv["per_v_honest"].get(7, [])),
                    "v6_pct": vp(mv, 6), "v6_n": len(mv["per_v_honest"].get(6, [])),
                    "v5_pct": vp(mv, 5), "v5_n": len(mv["per_v_honest"].get(5, [])),
                    "v4_pct": vp(mv, 4), "v4_n": len(mv["per_v_honest"].get(4, [])),
                    "v7_gate": vg(mv, 7), "v6_gate": vg(mv, 6),
                    "v5_gate": vg(mv, 5), "v4_gate": vg(mv, 4),
                    "vlt8_gtreproj_med": (round(float(np.median(mv["gtreps"])), 1)
                                          if mv["gtreps"] else None),
                    # V=8 보존: honest success (옳은 pose 비율)
                    "v8_recover_pct": pct(m8["honest_all"]),
                    "v8_gate_pct": pct(m8["gate_all"]),
                    "v8_false_accept_pct": pct(m8["false_acc"]),
                    "v8_n": len(m8["gate_all"]),
                    "v8_gtreproj_med": (round(float(np.median(m8["gtreps"])), 1)
                                        if m8["gtreps"] else None),
                })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=B3)
    ap.add_argument("--stage", choices=["synthetic", "real", "both"],
                    default="synthetic")
    ap.add_argument("--config", default=None,
                    help="real eval 용 선택 config: 'DECODER,score_thr,reproj_reject'")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)

    if args.stage in ("synthetic", "both"):
        fr_vlt8 = collect_syn_vlt8()
        fr_v8 = collect_syn_v8(400)
        print(f"synthetic V<8 frames: {len(fr_vlt8)} | V=8 frames: {len(fr_v8)}")
        cv = build_caches(model, fr_vlt8, device, "syn-V<8")
        c8 = build_caches(model, fr_v8, device, "syn-V=8")
        res = sweep_synthetic(cv, c8)
        with open(os.path.join(OUT, "synthetic_grid.json"), "w") as f:
            json.dump({"grid": res,
                       "vlt8_total": len(cv), "v8_total": len(c8)},
                      f, indent=2)
        print(f"saved: {OUT}/synthetic_grid.json  ({len(res)} configs)")
        # 콘솔: D0 baseline + top configs by 판단지표
        d0 = [r for r in res if r["decoder"] == "D0"]
        d0_best = max(d0, key=lambda r: r["vlt8_recover_pct"]) if d0 else None
        print("\n[D0 reference] honest recover (full-GT reproj<15px) by reject:")
        for r in sorted(d0, key=lambda r: (r["score_thr"], r["reproj_reject"])):
            print(f"  thr={r['score_thr']:.3f} rej={r['reproj_reject']:>4} "
                  f"V<8={r['vlt8_recover_pct']:>5}%(gate{r['vlt8_gate_pct']:>5}% "
                  f"FA{r['vlt8_false_accept_pct']:>5}%) V8={r['v8_recover_pct']:>5}% "
                  f"v6={r['v6_pct']:>5}% v7={r['v7_pct']:>5}%")

    if args.stage in ("real", "both"):
        if not args.config:
            print("real stage needs --config DECODER,score_thr,reproj_reject")
            return
        dec, sthr, rej = args.config.split(",")
        sthr, rej = float(sthr), float(rej)
        fr = collect_real()
        print(f"real frames: {len(fr)}")
        cr = build_caches(model, fr, device, "real")
        eval_real(cr, dec, sthr, rej)


def eval_real(caches, dec, sthr, rej):
    """real: 선택 config(dec,sthr,rej) vs D0 비교. 분할 ALL/V8/V6-7/V4-5/large/near."""
    # depth/size edges from real GT
    sizes, depths = [], []
    for c in caches:
        gt8, W, H = c["gt8"], c["W"], c["H"]
        inb = gt8_inframe(gt8, W, H)
        if inb.sum() >= 2:
            sizes.append(float(np.hypot(gt8[inb, 0].ptp(), gt8[inb, 1].ptp())))
        # depth: pose_transform 없을 수 있음 → skip
    s_hi = np.percentile(sizes, 66) if sizes else np.inf

    def decode_for(c, decoder):
        if decoder == "D0":
            return decode_one(c, "D0", sthr, rej)
        return decode_one(c, dec, sthr, rej)

    rows = []
    for c in caches:
        gt8, W, H = c["gt8"], c["W"], c["H"]
        inb = gt8_inframe(gt8, W, H)
        psize = (float(np.hypot(gt8[inb, 0].ptp(), gt8[inb, 1].ptp()))
                 if inb.sum() >= 2 else np.nan)
        ndet = sum(1 for i in range(8) if c["corners"][i][0] is not None)
        # corner median (hungarian, order-free) over in-frame predicted
        pred8 = np.full((8, 2), np.nan)
        for i in range(8):
            if c["corners"][i][0] is not None:
                pred8[i] = c["corners"][i][0]
        cmed, worst2 = corner_stats(pred8, gt8)
        r_sel = decode_for(c, dec)
        r_d0 = decode_for(c, "D0")
        rows.append({"V": c["V"], "psize": psize, "ndet": ndet,
                     "cmed": cmed, "worst2": worst2,
                     "sel_ok": r_sel["success"], "sel_rep": r_sel["gt_reproj"],
                     "d0_ok": r_d0["success"], "d0_rep": r_d0["gt_reproj"]})

    def split(rows, kind):
        if kind == "ALL":
            return rows
        if kind == "V=8":
            return [r for r in rows if r["V"] == 8]
        if kind == "V6-7":
            return [r for r in rows if r["V"] in (6, 7)]
        if kind == "V4-5":
            return [r for r in rows if r["V"] in (4, 5)]
        if kind == "large":
            return [r for r in rows if np.isfinite(r["psize"]) and r["psize"] >= s_hi]
        return rows

    def agg(rs):
        n = len(rs)
        if n == 0:
            return {"n": 0}
        cm = [r["cmed"] for r in rs if np.isfinite(r["cmed"])]
        w2 = [r["worst2"] for r in rs if np.isfinite(r["worst2"])]
        sel_rep = [r["sel_rep"] for r in rs if r["sel_ok"] and np.isfinite(r["sel_rep"])]
        d0_rep = [r["d0_rep"] for r in rs if r["d0_ok"] and np.isfinite(r["d0_rep"])]
        return {"n": n,
                "sel_pnp": round(100 * np.mean([r["sel_ok"] for r in rs]), 1),
                "d0_pnp": round(100 * np.mean([r["d0_ok"] for r in rs]), 1),
                "cmed": round(float(np.median(cm)), 1) if cm else None,
                "worst2": round(float(np.median(w2)), 1) if w2 else None,
                "sel_rep": round(float(np.median(sel_rep)), 1) if sel_rep else None,
                "d0_rep": round(float(np.median(d0_rep)), 1) if d0_rep else None}

    L = [f"# stage12 subset-PnP REAL eval — config={dec},thr={sthr:.3f},rej={rej}",
         f"# N={len(rows)} real frames (filter-val + manual). D0=current full-gate.",
         f"# size_edge(large>= p66)={round(s_hi,1)}px",
         f"# sel_pnp/d0_pnp = subset-PnP success%; cmed=hungarian corner med px;",
         f"# worst2=top-2 corner med; sel_rep/d0_rep = full-GT reproj med px (honest)."]
    for k in ["ALL", "V=8", "V6-7", "V4-5", "large"]:
        s = agg(split(rows, k))
        L.append(f"\n## {k}  (n={s.get('n',0)})")
        if s.get("n", 0) == 0:
            continue
        L.append(f"  PnP success : sel={s['sel_pnp']}%  D0={s['d0_pnp']}%  "
                 f"(Δ={round(s['sel_pnp']-s['d0_pnp'],1)}p)")
        L.append(f"  corner med  : {s['cmed']}px   worst2 med : {s['worst2']}px")
        L.append(f"  reproj med  : sel={s['sel_rep']}  D0={s['d0_rep']} (full-GT, honest)")
    txt = "\n".join(L)
    print("\n" + txt)
    with open(os.path.join(OUT, "real_eval.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(OUT, "real_eval.json"), "w") as f:
        json.dump({"config": [dec, sthr, rej], "rows": rows,
                   "size_edge": float(s_hi)}, f, indent=2,
                  default=lambda x: None if isinstance(x, float)
                  and not np.isfinite(x) else x)
    print(f"\nsaved: {OUT}/real_eval.txt , real_eval.json")


def corner_stats(pred8, gt8):
    valid = ~np.isnan(pred8[:, 0])
    if valid.sum() < 6:
        return np.inf, np.inf
    from scipy.optimize import linear_sum_assignment
    P = pred8[valid]
    cost = np.linalg.norm(P[:, None, :] - gt8[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    d = cost[ri, ci]
    worst2 = float(np.mean(np.sort(d)[-2:])) if len(d) >= 2 else np.inf
    return float(np.median(d)), worst2


if __name__ == "__main__":
    main()
