"""real manual GT 전수 평가 — 폴더별 검출률 + keypoint 오차.

eval_ft_fp.py 는 정본 161장(회귀 감시)만 본다. 이건 손으로 어노한 real 을 전부
돌려 도메인별로 어디가 약한지 보려는 참고용 진단이다.

★ 누수: 합성만으로 학습한 base(stage_a)는 real 을 하나도 안 봤으므로 아래 전부가
   out-of-sample 이다. 반면 runs_ft 의 finetune 모델은 pallet11_gt·night01~07·
   pallet02~05,08·forklift 를 학습에 썼으므로 그 폴더들은 in-sample 이다
   (runs_ft/PURPOSE.md 참조). ft 를 여기서 재면 부풀려진 수치가 나온다 — 비교하려면
   그 폴더를 빼고 봐야 한다. 스크립트가 in-sample 폴더를 [FT-SEEN] 으로 표시한다.

wood 는 나무 팔레트라 배포 대상(플라스틱)과 다른 물체다. dims 도 다르므로 따로 집계한다.
_night_eval_manual_gt 는 night05/06/07 과 같은 프레임이라 중복 표시만 하고 총계에서 뺀다.

사용:
  python challenge/yolo_pose_one_model/scripts/eval_real_all.py \
      --weights challenge/yolo_pose_one_model/final/pallet_yolo26n_pose_640_b32_final.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "challenge/yolo_pose_one_model"
PAD = 100

# runs_ft/PURPOSE.md 의 "포함" 목록 = finetune 이 학습에 쓴 real 세션
FT_SEEN = ("pallet11_gt", "capturenight01", "capturenight02", "capturenight03",
           "capturenight04", "capturenight05", "capturenight06", "capturenight07",
           "capturepallet02", "capturepallet03", "capturepallet04",
           "capturepallet05", "capturepallet08", "forklift_20260528")
DUP = ("_night_eval_manual_gt",)          # night05/06/07 과 중복 → 총계 제외
WOOD = ("wood_pallet",)                   # 나무 팔레트 → 별도 집계


def predict_batch(model, paths, conf, bs=32):
    """각 이미지의 (max_conf, kps(9,2) or None). 원본 좌표계로 되돌려 반환."""
    out = []
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        imgs, ok_idx = [], []
        for j, p in enumerate(chunk):
            im = cv2.imread(str(p))
            if im is None:
                continue
            imgs.append(cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD,
                                           cv2.BORDER_REFLECT_101))
            ok_idx.append(j)
        res = [(0.0, None)] * len(chunk)
        if imgs:
            for j, r in zip(ok_idx, model.predict(imgs, verbose=False,
                                                  conf=conf, imgsz=640)):
                if r.boxes is None or not len(r.boxes):
                    continue
                c = r.boxes.conf.cpu().numpy()
                b = int(np.argmax(c))
                res[j] = (float(c[b]), r.keypoints.xy.cpu().numpy()[b] - PAD)
        out.extend(res)
    return out


def gt_kps(ann):
    """9 keypoint GT (8 corner + centroid) + split 표시. 없으면 None."""
    try:
        o = json.load(open(ann, encoding="utf-8"))["objects"][0]
    except Exception:
        return None, None
    proj = o.get("projected_cuboid")
    if not proj or len(proj) < 8:
        return None, None
    cen = o.get("projected_cuboid_centroid") or [-1.0, -1.0]
    g = np.array([list(map(float, p)) for p in proj[:8]] + [list(map(float, cen))])
    return g, o.get("split")


def collect(root):
    """(png, gt(9,2), split) 리스트를 폴더별로."""
    items = []
    for a in sorted(glob.glob(os.path.join(root, "*.json"))):
        g, sp = gt_kps(a)
        p = os.path.splitext(a)[0] + ".png"
        if g is not None and os.path.exists(p):
            items.append((p, g, sp))
    return items


def evaluate(model, items, conf):
    """반환 (n, det, kp_med, kp_p90, err_list)."""
    if not items:
        return 0, 0, float("nan"), float("nan"), []
    preds = predict_batch(model, [it[0] for it in items], conf)
    det, errs = 0, []
    for (_, g, _), (c, k) in zip(items, preds):
        if c < conf or k is None:
            continue
        det += 1
        vis = [i for i in range(9) if not (g[i][0] == -1 and g[i][1] == -1)]
        if vis:
            errs.append(float(np.median(np.linalg.norm(k[vis] - g[vis], axis=1))))
    med = float(np.median(errs)) if errs else float("nan")
    p90 = float(np.percentile(errs, 90)) if errs else float("nan")
    return len(items), det, med, p90, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--out", default=str(OUT_ROOT / "runs_ft/_eval_real_all.json"))
    args = ap.parse_args()

    # 손 어노만. 접미사 _manual_gt 가 구분자다 — pallet11_gt(AprilTag 자동 GT)와
    # capturepallet07_augmented(증강본, GT 에 반쪽 -1 sentinel 혼입)는 이걸로 걸러진다.
    roots = sorted(
        d for d in glob.glob(str(REPO / "challenge/data/01_real/*/*_manual_gt"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.json")))

    model = YOLO(args.weights, task="pose")
    print(f"weights : {args.weights}")
    print(f"conf    : {args.conf}   pad {PAD}\n")

    hdr = f"{'폴더':<38}{'n':>5}{'det':>8}{'kp med':>9}{'kp p90':>9}  비고"
    print(hdr)
    print("-" * 84)

    groups = {"main": [], "wood": [], "dup": []}
    rows = {}
    for d in roots:
        name = os.path.basename(d)
        items = collect(d)
        if not items:
            continue
        n, det, med, p90, errs = evaluate(model, items, args.conf)
        tag = []
        if any(s in name for s in FT_SEEN):
            tag.append("[FT-SEEN]")
        if any(s in name for s in DUP):
            tag.append("[중복-총계제외]")
        if any(s in name for s in WOOD):
            tag.append("[나무팔레트]")
        n_eval = sum(1 for it in items if it[2] == "eval")
        if n_eval:
            tag.append(f"eval표시 {n_eval}")
        print(f"{name:<38}{n:>5}{det/n*100:>7.1f}%{med:>9.2f}{p90:>9.2f}  "
              + " ".join(tag))
        rows[name] = {"n": n, "det": det, "det_rate": det / n,
                      "kp_med": med, "kp_p90": p90, "n_eval_split": n_eval}
        key = ("wood" if any(s in name for s in WOOD)
               else "dup" if any(s in name for s in DUP) else "main")
        groups[key].append((n, det, errs))

    print("-" * 84)
    summary = {}
    for key, label in (("main", "플라스틱 팔레트 (총계)"),
                       ("wood", "나무 팔레트 (별도)"),
                       ("dup", "중복 폴더 (참고)")):
        g = groups[key]
        if not g:
            continue
        n = sum(x[0] for x in g)
        det = sum(x[1] for x in g)
        errs = [e for x in g for e in x[2]]
        med = float(np.median(errs)) if errs else float("nan")
        p90 = float(np.percentile(errs, 90)) if errs else float("nan")
        print(f"{label:<38}{n:>5}{det/n*100:>7.1f}%{med:>9.2f}{p90:>9.2f}")
        summary[key] = {"n": n, "det": det, "det_rate": det / n,
                        "kp_med": med, "kp_p90": p90}

    print("\ndet = conf>=%.2f 검출률 / kp med,p90 = 검출된 프레임의 9-keypoint "
          "중앙오차(px)" % args.conf)
    print("[FT-SEEN] = runs_ft finetune 이 학습에 쓴 세션 (ft 로 재면 in-sample)")
    json.dump({"weights": args.weights, "conf": args.conf,
               "per_folder": rows, "summary": summary},
              open(args.out, "w", encoding="utf-8"), indent=2, default=float)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
