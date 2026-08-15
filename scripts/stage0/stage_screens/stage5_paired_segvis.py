"""stage5_paired_segvis.py — STEP5 voting 판독.
(1) same-frame 페어: control-heatmap vs unit-heatmap vs unit-voting-seg
    (N confound 제거, back 코너 강조). filter-val(real)+manual.
(2) seg 예측마스크 overlay 2 real + 1 synthetic → 박스투영 과포함 눈검증.
eval_pvnet_heads(E) 재사용. 학습 X.
"""
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import eval_pvnet_heads as E  # noqa: E402

OFF = "weights/stage_screens/stage4_pvnet/off/final_net_pvnet_off.pth"
NEW = "weights/stage_screens/stage5_voting/unit_seg/final_net_pvnet_unit.pth"
OUT = "data/pallet/eval_results/stage5_voting"


def per_frame(model, has_vec, has_seg, frames, device, want_voting, thr=0.5):
    """fid -> {hm:split, vseg:split}. hm=heatmap arm, vseg=seg-mask voting."""
    import cv2, torch
    out = {}
    for dom, fid, jp, ip in frames:
        img = cv2.imread(ip)
        if img is None:
            continue
        gt8 = E.load_gt8(jp)
        t, nw, nh, sc = E.preprocess(img); t = t.to(device)
        with torch.no_grad():
            res = model(t)
        beliefs = res[0]
        belief = beliefs[-1][0].cpu().numpy()
        bh, bw = belief.shape[1], belief.shape[2]
        kps_bel = E.extract_keypoints_from_belief(belief, 0.3)
        rec = {"hm": E.split_metrics(E.heatmap_pred8(kps_bel, bw, bh, nw, nh, sc), gt8)}
        if want_voting and has_vec and has_seg:
            vec = res[2][-1][0].cpu().numpy()
            seg = res[3][-1][0].cpu().numpy()
            mask = E.seg_pred_mask(seg, bw, thr=thr)
            kp_bel = E.vote_unit_ransac(vec, mask)
            rec["vseg"] = E.split_metrics(E.kp_to_pred8(kp_bel, bw, bh, nw, nh, sc), gt8)
        out[fid] = rec
    return out


def paired(name, frames, mc, mu, device):
    ec = per_frame(mc, False, False, frames, device, False)
    eu = per_frame(mu, True, True, frames, device, True)
    L = [f"\n=== {name} ==="]
    for key in ("overall", "back"):
        fids = [f for f in ec if f in eu
                and np.isfinite(ec[f]["hm"][key]) and np.isfinite(eu[f]["hm"][key])]
        if not fids:
            L.append(f"  [{key}] (no paired)"); continue
        ch = np.array([ec[f]["hm"][key] for f in fids])
        uh = np.array([eu[f]["hm"][key] for f in fids])
        # voting-seg on frames it produced finite + control detected
        vf = [f for f in fids if "vseg" in eu[f] and np.isfinite(eu[f]["vseg"][key])]
        vs = np.array([eu[f]["vseg"][key] for f in vf]) if vf else np.array([])
        L.append(f"  [{key:<7}] paired N={len(fids)}  "
                 f"control-hm med={np.median(ch):.1f}  unit-hm med={np.median(uh):.1f}  "
                 f"(unit-hm better={int((uh<ch-1e-6).sum())}/{len(fids)})")
        if vs.size:
            L.append(f"            voting-seg med={np.median(vs):.1f} "
                     f"(better than control-hm={int((vs<ch[[fids.index(f) for f in vf]]-1e-6).sum())}/{len(vf)})  "
                     f"good<10={int((vs<10).sum())} gross>20={int((vs>20).sum())}")
    return "\n".join(L)


def seg_overlay(mu, frames, device, tag, thr=0.5):
    import cv2, torch
    os.makedirs(f"{OUT}/seg_overlays", exist_ok=True)
    paths = []
    for dom, fid, jp, ip in frames:
        img = cv2.imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        gt8 = E.load_gt8(jp)
        t, nw, nh, sc = E.preprocess(img); t = t.to(device)
        with torch.no_grad():
            res = mu(t)
        seg = res[3][-1][0].cpu().numpy()
        bw = res[0][-1].shape[3]
        m = E.seg_pred_mask(seg, bw, thr=thr).astype(np.float32)  # 50x50
        mup = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        vis = img.copy()
        green = np.zeros_like(img); green[:, :, 1] = 255
        vis = np.where(mup[:, :, None] > 0.5, (0.5*vis + 0.5*green).astype(np.uint8), vis)
        for i, (x, y) in enumerate(gt8):
            cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
        cov = float(m.mean())
        cv2.putText(vis, f"{tag} segcov={cov:.2f}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, f"{tag} segcov={cov:.2f}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        p = f"{OUT}/seg_overlays/{tag}_{fid}.jpg"
        cv2.imwrite(p, vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append((p, cov))
    return paths


def main():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc, _, _ = E.load_pvnet_model(OFF, device, numVec=0)
    mu, _, _ = E.load_pvnet_model(NEW, device)  # auto-detect numVec/numSeg
    out = ["STEP5 paired (control-hm vs unit-hm vs voting-seg) + seg overlay"]
    out.append(paired("filter-val (real)", E.collect_val_frames(), mc, mu, device))
    out.append(paired("manual GT 36", E.collect_manual(0), mc, mu, device))
    txt = "\n".join(out)
    print(txt)
    open(f"{OUT}/paired_voting.txt", "w").write(txt)
    # seg overlay: 2 real (manual) + 1 synthetic
    mn = E.collect_manual(0)[:2]
    sy = E.collect_syn(1)[:1]
    pv = seg_overlay(mu, mn, device, "real") + seg_overlay(mu, sy, device, "synth")
    print("\n[seg overlays]")
    for p, c in pv:
        print(f"  cov={c:.2f}  {p}")


if __name__ == "__main__":
    main()
