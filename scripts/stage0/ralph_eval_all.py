"""ralph_eval_all.py — 여러 체크포인트를 eval GT(held-out)에서 동일 하네스로 평가.

지표: 도메인별 + 전체 corner_med(px, order-free Hungarian median), NN<20px %, det%.
동일 하네스(squash400/belief50/THRESH0.3/N_DET_MIN6)로 base+라운드 → delta 유효.

Usage:
  python -u scripts/stage0/ralph_eval_all.py <out_json> <label1>=<ckpt1> <label2>=<ckpt2> ...
  ckpt 경로는 ROOT 상대 또는 절대. label 에 '=' 금지.
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import paper_s2_filterval_9filters as F   # noqa: E402,F401
import paper_s2_testset17_9filters as T   # noqa: E402
import cv2                                # noqa: E402
import torch                              # noqa: E402

NMIN = T.N_DET_MIN
# weak 모델군은 belief peak 이 낮아 강한-모델용 THRESH(0.3)에서 인위적 미검출.
# RALPH_THRESH 로 base+라운드 동일 threshold 평가 → delta 유효.
if os.environ.get("RALPH_THRESH"):
    T.THRESH = float(os.environ["RALPH_THRESH"])
    print(f"[THRESH override] T.THRESH={T.THRESH}")
DOMAINS = {
    "outside": ["challenge/data/_outside_eval_manual_gt"]
    + [f"challenge/data/capturepallet0{i}_manual_gt" for i in range(2, 10)],
    "night": ["challenge/data/_night_eval_manual_gt"]
    + [f"challenge/data/capturenight0{i}_manual_gt" for i in range(5, 8)],
    "noapril": ["challenge/data/capture0403noapril_manual_gt"],
    "cad": ["challenge/data/capturepalletcad_manual_gt"],
}


def load_eval():
    frames = {}
    for dom, fos in DOMAINS.items():
        seen = {}
        for fo in fos:
            for jf in glob.glob(os.path.join(ROOT, fo, "*.json")):
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
        frames[dom] = list(seen.values())
    return frames


def eval_model(wp, frames, device):
    model = T.E.load_model(os.path.join(ROOT, wp) if not os.path.isabs(wp) else wp, device)
    out = {}
    for dom, fs in frames.items():
        cms, ok, det = [], 0, 0
        for fr in fs:
            img = cv2.imread(fr["ip"])
            _, pr, _, _, _ = T.infer_squash(model, img, device)
            if int(np.sum(~np.isnan(pr[:, 0]))) < NMIN:
                continue
            det += 1
            d, _ = T.E.hungarian(pr, fr["gt"])
            m = float(np.median(d))
            cms.append(m)
            if m < 20:
                ok += 1
        n = len(fs)
        out[dom] = {
            "n": n, "det": det,
            "det_pct": round(100 * det / n, 1) if n else None,
            "corner_med": round(float(np.median(cms)), 2) if cms else None,
            "nn20_pct": round(100 * ok / n, 1) if n else None,
        }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def overall(dom_res):
    # 전체 = 도메인 통합 (프레임 가중 corner_med는 도메인별 median 재median 대신
    # det 프레임 corner_med 를 통합해야 정확하나, 여기선 도메인 median 의 median 근사 + NN20 가중)
    tot_n = sum(v["n"] for v in dom_res.values())
    tot_ok = sum((v["nn20_pct"] or 0) / 100 * v["n"] for v in dom_res.values())
    cms = [v["corner_med"] for v in dom_res.values() if v["corner_med"] is not None]
    return {
        "nn20_pct": round(100 * tot_ok / tot_n, 1) if tot_n else None,
        "corner_med_domavg": round(float(np.median(cms)), 2) if cms else None,
    }


def main():
    out_json = sys.argv[1]
    specs = sys.argv[2:]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = load_eval()
    print("[eval GT sizes]", {d: len(v) for d, v in frames.items()})
    results = {}
    for spec in specs:
        label, wp = spec.split("=", 1)
        if not os.path.isfile(os.path.join(ROOT, wp) if not os.path.isabs(wp) else wp):
            print("[skip missing]", label, wp)
            continue
        dom_res = eval_model(wp, frames, device)
        ov = overall(dom_res)
        results[label] = {"ckpt": wp, "domains": dom_res, "overall": ov}
        line = " ".join(
            f"{d}:cm{dom_res[d]['corner_med']}/nn{dom_res[d]['nn20_pct']}/det{dom_res[d]['det_pct']}"
            for d in DOMAINS)
        print(f"  {label:<16} OVERALL nn20={ov['nn20_pct']} cmDomAvg={ov['corner_med_domavg']} | {line}")
    os.makedirs(os.path.dirname(os.path.join(ROOT, out_json)) if not os.path.isabs(out_json)
                else os.path.dirname(out_json), exist_ok=True)
    with open(os.path.join(ROOT, out_json) if not os.path.isabs(out_json) else out_json, "w") as f:
        json.dump({"eval_sizes": {d: len(v) for d, v in frames.items()}, "results": results}, f, indent=2)
    print("[save]", out_json)


if __name__ == "__main__":
    main()
