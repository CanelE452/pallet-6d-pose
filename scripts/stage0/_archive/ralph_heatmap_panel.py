"""ralph_heatmap_panel.py — R0 vs self belief 히트맵 나란히 한 장 (측정 시각자료).

도메인별 대표 프레임 2개(결정론적, 매 K번째) × [R0 | self] belief 오버레이.
초록=GT, 파랑=예측점, 상단=det/peak. ralph_heatmap.heat_overlay 재사용.

Usage: python -u scripts/stage0/ralph_heatmap_panel.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
import paper_s2_filterval_9filters as F   # noqa: E402,F401
import paper_s2_testset17_9filters as T   # noqa: E402
T.THRESH = 0.3
import cv2                                # noqa: E402
import torch                              # noqa: E402
import ralph_heatmap as RH                # noqa: E402

RS = "data/pallet/results/ralph/ralph_selftrain"
R0 = "weights/paper_s2_stageB/net_epoch_0057_noseg.pth"
DOMS = ["outside", "night", "noapril"]
N_PER = 2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m0 = T.E.load_model(os.path.join(ROOT, R0), device)
    rows = []
    for dom in DOMS:
        self_path, gt_dirs = RH.CFG[dom]
        ms = T.E.load_model(os.path.join(ROOT, self_path), device)
        # 프레임 로드 (결정론적 매 K번째)
        seen = {}
        for fo in gt_dirs:
            for jf in sorted(glob.glob(os.path.join(ROOT, fo, "*.json"))):
                fid = os.path.splitext(os.path.basename(jf))[0]
                if fid in seen:
                    continue
                ip = jf[:-5] + ".png"
                if not os.path.isfile(ip):
                    continue
                try:
                    gt = np.array(json.load(open(jf))["objects"][0]["projected_cuboid"], float)[:8]
                except Exception:
                    continue
                seen[fid] = {"ip": ip, "gt": gt}
        vals = list(seen.values())
        picks = vals[:: max(1, len(vals) // N_PER)][:N_PER]
        for fr in picks:
            img = cv2.imread(fr["ip"])
            gt8 = [fr["gt"][i] for i in range(8)]
            a, d0, p0 = RH.heat_overlay(m0, img.copy(), device, gt8)
            b, ds, ps = RH.heat_overlay(ms, img.copy(), device, gt8)
            cv2.putText(a, f"{dom} R0  det{d0}/8 peak{p0:.2f}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            cv2.putText(b, f"{dom} self-train  det{ds}/8 peak{ps:.2f}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
            h = 300
            a = cv2.resize(a, (int(img.shape[1] * h / img.shape[0]), h))
            b = cv2.resize(b, (int(img.shape[1] * h / img.shape[0]), h))
            rows.append(cv2.hconcat([a, np.full((h, 5, 3), 255, np.uint8), b]))
        del ms
        if device == "cuda":
            torch.cuda.empty_cache()
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 5, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)) for r in rows]
    out = os.path.join(ROOT, RS, "heatmap_panel_R0_vs_self.png")
    cv2.imwrite(out, cv2.vconcat(rows))
    print("[save]", out, cv2.vconcat(rows).shape)


if __name__ == "__main__":
    main()
