"""anchor 선택이 레버인가 — top-1 대신 top-k 중 최선을 고르면 얼마나 좋아지나.

`yolo26_explain.py` 가 그림으로 보여준 것을 정본에서 수치로 가른다.  상위 anchor 들의
score 는 거의 평평한데(0.93 vs 0.88) 답은 최대 12px 다르다.  그러면 두 가지 중 하나다.

    (가) 더 잘 고르면 이득이 있다        -> ranking 이 레버다
    (나) 오라클로 골라도 이득이 작다     -> 병목은 선택이 아니라 keypoint 자체다

오라클은 **GT 를 보고 고르는 것**이라 배포 가능한 방법이 아니다.  여기서 재는 것은
"다시 고르기(rerank)로 얻을 수 있는 최대 상금" 의 상한이다.  이 상한이 작으면 rerank
연구는 하지 않는 게 맞다.

평가 population 은 정본만 쓴다 (`challenge/data_paths.py`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/scripts/annotate")

PAD = 100
DEFAULT_W = ("challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
             "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")
OUT = f"{ROOT}/data/pallet/results/yolo26_explain"


def canonical_frames():
    """정본 평가셋 — data_paths 를 import 한다 (경로 문자열 금지)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dp", f"{ROOT}/challenge/data_paths.py")
    dp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dp)
    out = []
    for d in dp.EVAL_CANONICAL.values():        # dict 다 — 키가 아니라 값이 경로
        for jp in sorted(glob.glob(os.path.join(str(d), "*.json"))):
            ip = jp[:-5] + ".png"
            if os.path.exists(ip):
                out.append((jp, ip))
    return out


def gt8(jp):
    d = json.load(open(jp))
    o = d["objects"][0]
    if o.get("split") != "eval":
        return None
    kps = o.get("projected_cuboid")
    if not kps or len(kps) < 8:
        return None
    return np.asarray(kps[:8], float)


def hungarian_median(pred, gt):
    """order-free 대응 후 median — 평가 정의를 새로 만들지 않는다."""
    from scipy.optimize import linear_sum_assignment
    ok = np.isfinite(pred[:, 0])
    if ok.sum() < 4:
        return np.inf
    c = np.linalg.norm(pred[ok][:, None, :] - gt[None, :, :], axis=2)
    r, cc = linear_sum_assignment(c)
    return float(np.median(c[r, cc]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_W)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO

    net = YOLO(os.path.join(ROOT, a.weights), task="pose").model.float().eval()
    head = net.model[-1]
    frames = canonical_frames()
    if a.limit:
        frames = frames[:a.limit]
    print(f"정본 프레임 {len(frames)}장  top-{a.topk} 오라클", flush=True)

    rows = []
    for jp, ip in frames:
        g = gt8(jp)
        if g is None:
            continue
        img = cv2.imread(ip)
        if img is None:
            continue
        p = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        h0, w0 = p.shape[:2]
        x = torch.from_numpy(cv2.resize(p, (640, 640))[:, :, ::-1].copy()
                             ).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            out = net(x)
        d = out[1]["one2many"]
        sc = 1.0 / (1.0 + np.exp(-d["scores"][0, 0].numpy()))
        kp = d["kpts"][0].numpy()
        anc = head.anchors.detach().numpy()
        st = head.strides.detach().numpy().reshape(-1)
        sx, sy = w0 / 640.0, h0 / 640.0
        kx = (kp[0::3] + anc[0]) * st * sx - PAD
        ky = (kp[1::3] + anc[1]) * st * sy - PAD

        order = np.argsort(-sc)[:a.topk]
        errs = []
        for idx in order:
            pred = np.stack([kx[:8, idx], ky[:8, idx]], 1)
            errs.append(hungarian_median(pred, g))
        errs = np.array(errs, float)
        rows.append({"fid": os.path.basename(jp)[:-5],
                     "top1": float(errs[0]),
                     "oracle": float(np.nanmin(errs)),
                     "best_rank": int(np.nanargmin(errs)),
                     "score_top1": float(sc[order[0]]),
                     "score_spread": float(sc[order].max() - sc[order].min())})
        if len(rows) % 25 == 0:
            print(f"  {len(rows)}장…", flush=True)

    if not rows:
        print("프레임 0 — 중단")
        return 1

    t1 = np.array([r["top1"] for r in rows])
    orc = np.array([r["oracle"] for r in rows])
    gain = t1 - orc
    ranks = np.array([r["best_rank"] for r in rows])

    print()
    print(f"{'':22}{'median':>9}{'p90':>9}{'mean':>9}")
    print("-" * 49)
    print(f"{'top-1 (실제 동작)':22}{np.median(t1):>9.2f}{np.percentile(t1,90):>9.2f}{t1.mean():>9.2f}")
    print(f"{'oracle top-'+str(a.topk):22}{np.median(orc):>9.2f}{np.percentile(orc,90):>9.2f}{orc.mean():>9.2f}")
    print(f"{'상금 (top1-oracle)':22}{np.median(gain):>9.2f}{np.percentile(gain,90):>9.2f}{gain.mean():>9.2f}")
    print()
    print(f"  top-1 이 이미 최선인 프레임   {int((ranks==0).sum())}/{len(rows)}"
          f"  ({100*(ranks==0).mean():.1f}%)")
    print(f"  상금 > 2px 인 프레임          {int((gain>2).sum())}/{len(rows)}"
          f"  ({100*(gain>2).mean():.1f}%)")
    print(f"  상금 > 5px 인 프레임          {int((gain>5).sum())}/{len(rows)}"
          f"  ({100*(gain>5).mean():.1f}%)")
    print(f"  후보 score 폭 (median)        {np.median([r['score_spread'] for r in rows]):.3f}")

    dst = f"{OUT}/anchor_oracle_top{a.topk}.json"
    os.makedirs(OUT, exist_ok=True)
    json.dump({"weights": a.weights, "topk": a.topk, "n": len(rows),
               "top1_median": float(np.median(t1)),
               "oracle_median": float(np.median(orc)),
               "gain_median": float(np.median(gain)),
               "frac_top1_already_best": float((ranks == 0).mean()),
               "frac_gain_gt2px": float((gain > 2).mean()),
               "note": "oracle 은 GT 를 보고 고른 것 — 배포 불가. rerank 상한일 뿐이다.",
               "rows": rows}, open(dst, "w"), indent=1)
    print(f"\n-> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
