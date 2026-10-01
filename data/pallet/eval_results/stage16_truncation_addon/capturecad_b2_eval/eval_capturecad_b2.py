"""eval_capturecad_b2.py — B2 on hand-labeled capturepalletcad (22 frames).

목적: 손라벨 capturecad 22장에서 best 모델 B2 의 정직한 성능 수치.
판단지표: det% / corner median(px) / worst2 / PnP% / honest full-8 reproj
           + V_geom(화면안 코너 수) 분포. squash vs reflect-pad 둘 다.

★ N=22 극소표본 + 극근접 frame-filling hard subset → 과결론 금지.

인프라 재사용 (eval_stage11 / eval_pvnet_heads / annotate_pnp):
  - order-free Hungarian corner 매칭 (convention order 무관)
  - per-frame K, solve_pose (order-free W/D swap)
  - honest full-8 reproj = solved pose 의 8코너 투영(GT dims) vs GT projected_cuboid
    (order-free Hungarian — solve 가 W/D swap 골랐을 수 있으므로)
  - 전처리: squash(aspect-only) vs reflect-pad(memory: 극근접/truncation 필수)
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path[:0] = [os.path.join(ROOT, "scripts", "data_prep", _s)
                for _s in ("plots", "filters")]
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

sys.path[:0] = [os.path.join(ROOT, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
import cv2, torch
from models import DopeNetwork
from filter_pr_camfacing import extract_keypoints_from_belief
from eval_pvnet_heads import preprocess, split_metrics
import annotate_pnp as APNP

GOOD_PX, GROSS_PX, N_DET_MIN = 10.0, 20.0, 6
FRONT, BACK = [0, 1, 2, 3], [4, 5, 6, 7]
CAD_DIR = os.path.join(ROOT, "challenge", "data", "capturepalletcad_manual_gt")
HERE = os.path.dirname(os.path.abspath(__file__))


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


def pad_frame(img, pad):
    """reflect-pad then resize back to original (dope_predict_mp4_pad 관례)."""
    if pad <= 0:
        return img
    h, w = img.shape[:2]
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    return cv2.resize(padded, (w, h), interpolation=cv2.INTER_LINEAR)


def belief_to_orig_pad(bx, by, bw, bh, nw, nh, sc, pad, W, H):
    """belief 격자 -> padded-resized canvas(원본 nw×nh aspect) -> 원본 px.

    eval preprocess 는 padded-resized 이미지(=원본 크기 W×H)를 받아 400/min 으로
    aspect resize 한다. 따라서 belief->padded-resized-canvas 매핑은 기존
    belief_to_orig 와 동일(좌표계 = padded-resized 640x480). 그 다음 padded canvas
    (W+2P)/(H+2P) 로 확대 후 pad 빼기 = 원본 좌표."""
    ux, uy = nw / bw, nh / bh
    cx = (bx * ux) / sc   # padded-resized canvas px (W×H)
    cy = (by * uy) / sc
    if pad > 0:
        px = cx * (W + 2 * pad) / W
        py = cy * (H + 2 * pad) / H
        return px - pad, py - pad
    return cx, cy


def hungarian(pred, gt):
    """pred (M,2 valid), gt (8,2) -> matched dists, gt-idx. None if <N_DET_MIN."""
    valid = ~np.isnan(pred[:, 0])
    if valid.sum() < N_DET_MIN:
        return None, None
    from scipy.optimize import linear_sum_assignment
    P = pred[valid]
    cost = np.linalg.norm(P[:, None, :] - gt[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    return cost[ri, ci], ci


def eval_frame(model, jp, ip, device, threshold, pad):
    img = cv2.imread(ip)
    if img is None:
        return None
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    gtc = np.array(o["projected_cuboid_centroid"], float)
    H, W = img.shape[:2]
    K = K_from_json(d)

    # V_geom = 화면안 GT 코너 수 (truncation/near 정도)
    inb = ((gt8[:, 0] >= 0) & (gt8[:, 0] < W)
           & (gt8[:, 1] >= 0) & (gt8[:, 1] < H))
    v_geom = int(inb.sum())

    proc = pad_frame(img, pad)
    tensor, nw, nh, sc = preprocess(proc)
    with torch.no_grad():
        beliefs, _ = model(tensor.to(device))
    belief = beliefs[-1][0].cpu().numpy()
    bh, bw = belief.shape[1], belief.shape[2]
    kps_bel = extract_keypoints_from_belief(belief, threshold)

    pred8 = np.full((8, 2), np.nan)
    for i, k in enumerate(kps_bel[:8]):
        if k[0] < 0:
            continue
        pred8[i] = belief_to_orig_pad(k[0], k[1], bw, bh, nw, nh, sc, pad, W, H)
    pred_c = None
    if kps_bel[8][0] >= 0:
        pred_c = list(belief_to_orig_pad(kps_bel[8][0], kps_bel[8][1],
                                         bw, bh, nw, nh, sc, pad, W, H))

    # order-free corner metrics
    m = split_metrics(pred8, gt8)
    dists, _ = hungarian(pred8, gt8)
    worst2 = (float(np.mean(np.sort(dists)[-2:]))
              if dists is not None and len(dists) >= 2 else np.inf)

    # PnP: 9kp, per-frame K, GT dims (solve_pose order-free W/D)
    kps9 = [None if np.isnan(pred8[i, 0]) else
            [float(pred8[i, 0]), float(pred8[i, 1])] for i in range(8)]
    kps9.append(pred_c)
    nvalid = sum(1 for k in kps9 if k is not None)
    pnp_reproj_click, pnp_honest8, pnp_ok = np.inf, np.inf, 0
    proj_all = None
    if nvalid >= N_DET_MIN:
        try:
            pose = APNP.solve_pose(kps9, K, dims=APNP.PALLET_DIMS,
                                   img_shape=img.shape)
            if pose is not None:
                pnp_ok = 1
                pnp_reproj_click = float(pose["reproj_error_px"])
                proj_all = np.array(pose["projected_all"], float)[:8]
                # honest full-8: solved-pose 8코너(GT dims) vs GT projected_cuboid,
                # order-free (solve 가 W/D swap 골랐을 수 있음)
                pa = proj_all.copy()
                # sentinel (-1,-1) -> nan 처리
                bad = (pa[:, 0] == -1.0) & (pa[:, 1] == -1.0)
                pa[bad] = np.nan
                hd, _ = hungarian(pa, gt8)
                if hd is not None:
                    pnp_honest8 = float(np.mean(hd))
        except Exception:
            pass

    return {"fid": os.path.splitext(os.path.basename(jp))[0],
            "v_geom": v_geom, "n_det": m["n_det"],
            "det": 1 if m["n_det"] >= N_DET_MIN else 0,
            "corner": m["overall"], "front": m["front"], "back": m["back"],
            "worst2": worst2, "pnp_ok": pnp_ok,
            "pnp_reproj_click": pnp_reproj_click, "pnp_honest8": pnp_honest8,
            "gt8": gt8.tolist(), "gtc": gtc.tolist(),
            "pred8": pred8.tolist(), "pred_c": pred_c,
            "proj_all": (proj_all.tolist() if proj_all is not None else None),
            "ip": ip}


def agg(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    cor = [r["corner"] for r in rows if np.isfinite(r["corner"])]
    w2 = [r["worst2"] for r in rows if np.isfinite(r["worst2"])]
    prc = [r["pnp_reproj_click"] for r in rows
           if r["pnp_ok"] and np.isfinite(r["pnp_reproj_click"])]
    ph8 = [r["pnp_honest8"] for r in rows
           if r["pnp_ok"] and np.isfinite(r["pnp_honest8"])]
    return {
        "n": n,
        "det_pct": round(100 * np.mean([r["det"] for r in rows]), 1),
        "corner_med": round(float(np.median(cor)), 1) if cor else None,
        "corner_good_pct": (round(100 * np.mean([c < GOOD_PX for c in cor]), 1)
                            if cor else None),
        "worst2_med": round(float(np.median(w2)), 1) if w2 else None,
        "pnp_ok_pct": round(100 * np.mean([r["pnp_ok"] for r in rows]), 1),
        "pnp_reproj_click_med": round(float(np.median(prc)), 1) if prc else None,
        "pnp_honest8_med": round(float(np.median(ph8)), 1) if ph8 else None,
    }


def overlay(row, out_path):
    img = cv2.imread(row["ip"])
    if img is None:
        return
    gt8 = np.array(row["gt8"], float)
    pr8 = np.array(row["pred8"], float)
    for p in gt8:
        cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 255, 0), 2)   # GT green
    for i, p in enumerate(pr8):
        if np.isnan(p[0]):
            continue
        cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)  # pred red
    if row.get("proj_all"):
        pa = np.array(row["proj_all"], float)
        order = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4]
        for a, b in zip(order[:-1], order[1:]):
            if (pa[a] == -1).all() or (pa[b] == -1).all():
                continue
            cv2.line(img, (int(pa[a][0]), int(pa[a][1])),
                     (int(pa[b][0]), int(pa[b][1])), (255, 200, 0), 1)
    txt = (f"V={row['v_geom']} det={row['n_det']}/8 "
           f"cor={row['corner']:.1f} h8={row['pnp_honest8']:.1f}")
    cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    cv2.imwrite(out_path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(
        ROOT, "weights/stage_screens/stage11_16k_B2_maskaux/final_net_epoch_0084.pth"))
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--pad", type=int, default=100)
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = []
    for jp in sorted(glob.glob(os.path.join(CAD_DIR, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(CAD_DIR, fid + ".png")
        if os.path.exists(ip):
            frames.append((jp, ip))
    print(f"[eval] capturecad frames = {len(frames)}")

    model = load_model(args.weights, device)

    variants = {"squash": 0, "pad100": args.pad}
    results = {}
    for vname, pad in variants.items():
        rows = []
        for jp, ip in frames:
            r = eval_frame(model, jp, ip, device, args.threshold, pad)
            if r is not None:
                rows.append(r)
        results[vname] = rows

    # V_geom distribution (from squash rows; V_geom is GT-only, same either way)
    vdist = {}
    for r in results["squash"]:
        vdist[r["v_geom"]] = vdist.get(r["v_geom"], 0) + 1

    txt = []
    txt.append("# B2 on hand-labeled capturepalletcad (N=22)")
    txt.append(f"# weights: {args.weights}")
    txt.append("# metrics order-free (Hungarian corner, solve_pose order-free W/D)")
    txt.append("# corner_med=hungarian overall median px | good%=corner<10px |")
    txt.append("# det%=n_det>=6 | worst2=top-2 corner med | pnp%=solve success |")
    txt.append("# pnp_reproj_click=reproj vs pred kps | honest8=solved 8corner(GTdim)")
    txt.append("#   vs GT projected_cuboid (order-free) <- the honest pose error")
    txt.append("")
    txt.append("# V_geom (in-frame GT corner count) distribution:")
    for v in sorted(vdist):
        txt.append(f"#   V={v}: {vdist[v]} frames")
    txt.append("")

    def line(label, s):
        return (f"{label:<14} n={s.get('n',0):<3} "
                f"det%={str(s.get('det_pct')):<6} "
                f"cor_med={str(s.get('corner_med')):<6} "
                f"good%={str(s.get('corner_good_pct')):<6} "
                f"worst2={str(s.get('worst2_med')):<6} "
                f"pnp%={str(s.get('pnp_ok_pct')):<6} "
                f"rep_click={str(s.get('pnp_reproj_click_med')):<6} "
                f"honest8={str(s.get('pnp_honest8_med')):<6}")

    out = {"weights": args.weights, "n": len(frames),
           "v_geom_dist": {str(k): v for k, v in sorted(vdist.items())},
           "variants": {}}
    txt.append("## ALL (N=22)")
    for vname in variants:
        s = agg(results[vname])
        out["variants"][vname] = {"all": s}
        txt.append(line(vname, s))

    # V_geom split (pad100 only — the recommended near-field path)
    txt.append("\n## by V_geom (pad100)")
    rows = results["pad100"]
    for vk, sel in [("V=8", [r for r in rows if r["v_geom"] == 8]),
                    ("V<8", [r for r in rows if r["v_geom"] < 8])]:
        s = agg(sel)
        out["variants"]["pad100"][vk] = s
        txt.append(line(vk, s))

    # per-frame dump (pad100)
    out["per_frame_pad100"] = [
        {k: r[k] for k in ("fid", "v_geom", "n_det", "corner",
                           "worst2", "pnp_ok", "pnp_reproj_click",
                           "pnp_honest8")}
        for r in results["pad100"]]

    report = "\n".join(txt)
    print("\n" + report)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics_table.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)

    # overlays: a few representative (best/median/worst by honest8, pad100)
    ovdir = os.path.join(args.out, "overlays")
    os.makedirs(ovdir, exist_ok=True)
    rr = [r for r in results["pad100"] if r["pnp_ok"]
          and np.isfinite(r["pnp_honest8"])]
    rr.sort(key=lambda r: r["pnp_honest8"])
    picks = []
    if rr:
        picks = [("best", rr[0]), ("median", rr[len(rr) // 2]),
                 ("worst", rr[-1])]
    # also a couple V<8 frames
    vlt = [r for r in results["pad100"] if r["v_geom"] < 8][:2]
    for i, r in enumerate(vlt):
        picks.append((f"Vlt8_{i}", r))
    for tag, r in picks:
        overlay(r, os.path.join(ovdir, f"{tag}_{r['fid']}.jpg"))

    print(f"\n[save] {args.out}/metrics_table.txt , metrics.json")
    print(f"[save] overlays: {ovdir} ({len(picks)} imgs)")


if __name__ == "__main__":
    main()
