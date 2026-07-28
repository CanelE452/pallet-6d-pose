"""paired_control_vs_unit.py — same-frame paired compare (control-heatmap vs
unit-heatmap). Removes the N-confound: only frames BOTH models detect (>=6
corners) are compared, per-frame. Reports paired median(overall/back), per-frame
win counts, good/gross on the paired set. Reuses eval_pvnet_heads (no new geom).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_pvnet_heads as E  # noqa: E402

CONTROL_W = "weights/stage4_pvnet/off/final_net_pvnet_off.pth"
UNIT_W = "weights/stage4_pvnet/unit/final_net_pvnet_unit.pth"


def frame_errs(model, numVec, frames, device, threshold=0.3):
    """per-frame heatmap split_metrics keyed by fid."""
    import cv2
    import torch
    out = {}
    has_vec = numVec > 0
    for dom, fid, jp, ip in frames:
        img = cv2.imread(ip)
        if img is None:
            continue
        gt8 = E.load_gt8(jp)
        tensor, nw, nh, sc = E.preprocess(img)
        tensor = tensor.to(device)
        with torch.no_grad():
            res = model(tensor)
        beliefs = res[0]
        belief = beliefs[-1][0].cpu().numpy()
        bh, bw = belief.shape[1], belief.shape[2]
        kps_bel = E.extract_keypoints_from_belief(belief, threshold)
        ph8 = E.heatmap_pred8(kps_bel, bw, bh, nw, nh, sc)
        out[fid] = E.split_metrics(ph8, gt8)
    return out


def paired_report(name, frames, mc, mu, device):
    ec = frame_errs(mc, 0, frames, device)
    eu = frame_errs(mu, 18, frames, device)
    fids = [f for f in ec if f in eu
            and np.isfinite(ec[f]["overall"]) and np.isfinite(eu[f]["overall"])]
    lines = [f"\n=== {name}  (paired N={len(fids)}; both detect >=6) ==="]
    if not fids:
        lines.append("  (no commonly-detected frames)")
        return "\n".join(lines)
    for key in ("overall", "back"):
        c = np.array([ec[f][key] for f in fids if np.isfinite(ec[f][key])])
        u = np.array([eu[f][key] for f in fids if np.isfinite(eu[f][key])])
        # per-frame win: unit strictly better (lower) on frames both finite
        both = [f for f in fids
                if np.isfinite(ec[f][key]) and np.isfinite(eu[f][key])]
        cu = np.array([ec[f][key] for f in both])
        uu = np.array([eu[f][key] for f in both])
        u_win = int((uu < cu - 1e-6).sum())
        c_win = int((cu < uu - 1e-6).sum())
        d = uu - cu  # >0 = unit worse
        lines.append(
            f"  [{key:<7}] control med={np.median(c):.1f} "
            f"good<10={int((c<10).sum())} gross>20={int((c>20).sum())}  |  "
            f"unit med={np.median(u):.1f} "
            f"good<10={int((u<10).sum())} gross>20={int((u>20).sum())}")
        lines.append(
            f"            paired Δmed(unit-ctrl)={np.median(d):+.1f}px  "
            f"unit_better={u_win}/{len(both)}  ctrl_better={c_win}  "
            f"mean Δ={d.mean():+.1f}")
    return "\n".join(lines)


def main():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    mc, _ = E.load_pvnet_model(CONTROL_W, device, numVec=0)
    mu, _ = E.load_pvnet_model(UNIT_W, device, numVec=18)

    out = ["PAIRED control-heatmap vs unit-heatmap (same-frame, N-confound 제거)",
           "Δmed>0 = unit WORSE; unit_better = unit이 더 낮은 프레임 수"]
    out.append(paired_report("manual GT 36", E.collect_manual(0), mc, mu, device))
    out.append(paired_report("filter-val", E.collect_val_frames(), mc, mu, device))
    out.append(paired_report("synthetic 200", E.collect_syn(200), mc, mu, device))
    txt = "\n".join(out)
    print(txt)
    fp = "data/pallet/eval_results/stage4_pvnet/paired_control_vs_unit.txt"
    open(fp, "w").write(txt)
    print(f"\n[save] {fp}")


if __name__ == "__main__":
    main()
