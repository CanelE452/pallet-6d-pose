"""stage19_partB_eval.py — PART B paired eval: control vs mixup (A-PASS pilot).

Same-frame paired comparison on filter-val (outside+night, SEALED) + manual GT.
Only frames BOTH arms detect (n_det>=6) are paired.  reflect-pad inference
(matches B2 regime).  Reports, per arm on the paired set:
  corner median (FRONT 0-3 / REAR 4-7 split, channel-aligned)
  good(<10px) / gross(>20px) absolute counts (per-corner)
  honest full-8 SQPnP reprojection median (order-agnostic channel-aligned)
  V=8 full-view subset corner median (regression check)

Verdict (B5): mixup improves REAR / honest reproj vs control -> "signal, full
candidate"; else honest negative.  Small real GT -> paired + absolute counts.

Usage: conda run -n pallet-pose python scripts/stage0/stage19_partB_eval.py \
    --control weights/stage19_mixup_pilot/control/final_net.pth \
    --mixup   weights/stage19_mixup_pilot/mixup/final_net.pth
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path[:0] = [os.path.join(ROOT, "scripts", "data_prep", _s)
                for _s in ("plots", "filters")]
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

from tau_calibrate import collect_val_frames  # noqa: E402
from eval_pvnet_heads import collect_manual  # noqa: E402
from corner01_diagnosis import load_gt, per_corner_err  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "stage19_conf_mixup")
PAD = 100
THRESHOLD = 0.3
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
FRONT = [0, 1, 2, 3]
REAR = [4, 5, 6, 7]
GOOD, GROSS = 10.0, 20.0


def canonical_kp3d(w, d, h):
    W, H, D = w / 2, h / 2, d / 2
    c = np.array([[-W, -H, +D], [+W, -H, +D], [+W, +H, +D], [-W, +H, +D],
                  [-W, -H, -D], [+W, -H, -D], [+W, +H, -D], [-W, +H, -D]], float)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default=os.path.join(
        ROOT, "weights", "stage19_mixup_pilot", "control", "final_net.pth"))
    ap.add_argument("--mixup", default=os.path.join(
        ROOT, "weights", "stage19_mixup_pilot", "mixup", "final_net.pth"))
    args = ap.parse_args()

    import cv2
    import torch
    from filter_pr_camfacing import extract_keypoints_from_belief
    from eval_pvnet_heads import load_pvnet_model
    from dope_predict_mp4_pad import pad_frame

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = collect_val_frames() + collect_manual()
    print(f"[eval] {len(frames)} GT frames (filter-val + manual, final-test SEALED)")

    def build_infer(weights):
        model, nv, ns = load_pvnet_model(weights, device)
        model.eval()

        def infer(img):
            h0, w0 = img.shape[:2]
            proc = pad_frame(img, PAD, "reflect")
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            ph, pw = proc.shape[:2]
            sc = 400.0 / min(ph, pw)
            nw = max(8, int(round(pw * sc)) & ~7)
            nh = max(8, int(round(ph * sc)) & ~7)
            t = ((cv2.resize(rgb, (nw, nh)).astype(np.float32) / 255.0 - MEAN) / STD)
            tt = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tt)
            bel = out[0][-1][0].cpu().numpy()
            kps = extract_keypoints_from_belief(bel, THRESHOLD)
            bh, bw = bel.shape[1], bel.shape[2]
            ux, uy = nw / bw, nh / bh
            pred8 = np.full((8, 2), np.nan)
            for i, k in enumerate(kps[:8]):
                if k[0] < 0:
                    continue
                cx = (k[0] * ux) / sc; cy = (k[1] * uy) / sc
                pred8[i] = (cx * (w0 + 2 * PAD) / w0 - PAD,
                            cy * (h0 + 2 * PAD) / h0 - PAD)
            return pred8
        return infer

    inf_c = build_infer(args.control)
    inf_m = build_infer(args.mixup)

    def reproj8(pred8, gt8, dm, intr):
        det = [i for i in range(8) if np.isfinite(pred8[i, 0])]
        if len(det) < 6 or intr is None:
            return None
        kp3d = canonical_kp3d(dm["width"], dm["depth"], dm["height"])
        K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]],
                      [0, 0, 1]], float)
        obj = kp3d[det].reshape(-1, 1, 3)
        img = pred8[det].reshape(-1, 1, 2)
        ok, rv, tv = cv2.solvePnP(obj, img, K, np.zeros((5, 1)),
                                  flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        proj, _ = cv2.projectPoints(kp3d[det], rv, tv, K, np.zeros((5, 1)))
        return float(np.linalg.norm(proj.reshape(-1, 2) - pred8[det], axis=1).mean())

    rows = {"control": [], "mixup": []}
    npair = 0
    for dom, fid, jp, ip in frames:
        img = cv2.imread(ip)
        if img is None:
            continue
        gt8, dm, pose, intr = load_gt(jp)
        if dm is None:
            dm = {"width": 1.3, "depth": 1.1, "height": 0.11}
        pc = inf_c(img); pm = inf_m(img)
        nc = np.isfinite(pc[:, 0]).sum(); nm = np.isfinite(pm[:, 0]).sum()
        if nc < 6 or nm < 6:
            continue
        npair += 1
        h0, w0 = img.shape[:2]
        inb = ((gt8[:, 0] >= 0) & (gt8[:, 0] < w0) &
               (gt8[:, 1] >= 0) & (gt8[:, 1] < h0))
        v8 = bool(inb.sum() == 8)
        for name, pr in (("control", pc), ("mixup", pm)):
            e = per_corner_err(pr, gt8)
            rows[name].append({
                "dom": dom, "v8": v8,
                "front": [float(e[i]) for i in FRONT if np.isfinite(e[i])],
                "rear": [float(e[i]) for i in REAR if np.isfinite(e[i])],
                "reproj": reproj8(pr, gt8, dm, intr)})

    def agg(rs, v8only=False):
        fr, re, rp, good, gross = [], [], [], 0, 0
        for r in rs:
            if v8only and not r["v8"]:
                continue
            fr += r["front"]; re += r["rear"]
            if r["reproj"] is not None:
                rp.append(r["reproj"])
            good += sum(1 for x in r["front"] + r["rear"] if x < GOOD)
            gross += sum(1 for x in r["front"] + r["rear"] if x > GROSS)
        med = lambda a: round(float(np.median(a)), 2) if a else None
        return {"n": len([r for r in rs if not v8only or r["v8"]]),
                "front_med": med(fr), "rear_med": med(re),
                "reproj_med": med(rp), "good": good, "gross": gross}

    lines = []
    lines.append(f"PART B PAIRED EVAL (both-detect n={npair}; filter-val+manual, SEALED)")
    lines.append(f"control = {args.control}")
    lines.append(f"mixup   = {args.mixup}")
    lines.append("")
    hdr = f"{'arm':<9}{'n':>4}{'front_med':>10}{'rear_med':>10}{'reproj_med':>11}{'good<10':>8}{'gross>20':>9}"
    lines.append(hdr); lines.append("-" * len(hdr))
    res = {}
    for name in ("control", "mixup"):
        a = agg(rows[name]); res[name] = a
        lines.append(f"{name:<9}{a['n']:>4}{a['front_med']:>10}{a['rear_med']:>10}"
                     f"{a['reproj_med']:>11}{a['good']:>8}{a['gross']:>9}")
    lines.append("")
    lines.append("[V=8 full-view subset (regression check)]")
    lines.append(hdr); lines.append("-" * len(hdr))
    resv = {}
    for name in ("control", "mixup"):
        a = agg(rows[name], v8only=True); resv[name] = a
        lines.append(f"{name:<9}{a['n']:>4}{a['front_med']:>10}{a['rear_med']:>10}"
                     f"{a['reproj_med']:>11}{a['good']:>8}{a['gross']:>9}")

    # verdict
    def better(m, c):
        return (m is not None and c is not None and m < c)
    rear_imp = better(res["mixup"]["rear_med"], res["control"]["rear_med"])
    rep_imp = better(res["mixup"]["reproj_med"], res["control"]["reproj_med"])
    front_reg = (res["mixup"]["front_med"] is not None
                 and res["control"]["front_med"] is not None
                 and res["mixup"]["front_med"] > res["control"]["front_med"] + 1.0)
    v8_reg = (resv["mixup"]["front_med"] is not None
              and resv["control"]["front_med"] is not None
              and resv["mixup"]["front_med"] > resv["control"]["front_med"] + 1.0)
    if (rear_imp or rep_imp) and not front_reg and not v8_reg:
        verdict = "SIGNAL (mixup improves rear/honest w/o full-view regression) -> full candidate"
    elif front_reg or v8_reg:
        verdict = "NEGATIVE+REGRESSION (mixup hurts full-view/front) -> reject soft interpolation"
    else:
        verdict = "HONEST NEGATIVE (no rear/honest improvement) -> soft interpolation folds"
    lines.append("")
    lines.append(f"VERDICT: {verdict}")
    lines.append("[caveat] small real GT + sparse front-PL (36 real frames, 1-2 corners) "
                 "-> pilot signal; absolute counts + paired same-frame reported.")

    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT_DIR, "partB_pilot_paired.txt"), "w").write(txt)
    json.dump({"npair": npair, "control": res, "mixup": res["mixup"],
               "v8": resv, "verdict": verdict,
               "res_full": res, "res_v8": resv},
              open(os.path.join(OUT_DIR, "partB_eval.json"), "w"), indent=2)
    print(f"\n[save] {os.path.join(OUT_DIR, 'partB_pilot_paired.txt')}")


if __name__ == "__main__":
    main()
