"""PHASE A — 기존 NEGATIVE_SYNTH_V1 9,000 의 hardness 감사.  TRAINING = 0.

frozen 모델이 각 negative 를 얼마나 pallet 으로 오인하는지를 **raw one2one anchor**
수준에서 잰다.  post-NMS 출력만 보면 억제된 뒤의 그림이라 mining 근거가 안 된다.

접근점 (`ultralytics 8.4.60` `Detect.forward`, eval + end2end):

    yout, preds = model(x)
    preds["one2one"]["scores"]   (bs, nc, A)   ★sigmoid 이전 raw logit
    preds["one2one"]["feats"]    레벨별 feature -> anchor 개수로 P3/P4/P5 경계

anchor 는 레벨 concat 순서(P3 -> P4 -> P5)라 누적합으로 레벨이 정해진다.

전처리는 **학습 파이프라인과 같은 LetterBox(640)** 를 쓴다.  release 배포계약의
PAD=100 reflect 가 아니다 — 여기서 묻는 것은 "이 이미지를 학습에 넣었을 때 모델이
헷갈리는가" 이므로 학습 때 보는 형태로 재야 한다.

금지 준수: optimizer 0 / backward 0 / model.train() 0 / fuse 0.
BCE gradient 는 해석적으로 계산한다 — negative target y=0 에서 dL/dz = sigmoid(z).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
OUT = os.path.join(ROOT, "challenge/yolo_pose_one_model/hard_negative_v1/PHASE_A")
NEG9K = os.path.join(ROOT, "data/pallet/training_data/paper_release/negative/"
                           "extracted/negative_synth_v1_train/rgb")
REAL_NEG = os.path.join(ROOT, "data/pallet/raw_data/negative_real_20260823")
CKPT = {
    "Y0": "challenge/yolo_pose_one_model/runs_posecls_g38/"
          "Y26_G38_Y0_VANILLA_30EP_SEED42/weights/last.pt",
    "Y0E": "challenge/yolo_pose_one_model/runs_neg_g38/Y0E/weights/best.pt",
}
EXPECT_SHA = {"Y0": "37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1"}
IMGSZ, BATCH = 640, 16
N_HARD = 1900                       # round(38,002 x 0.05)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen(tag):
    from ultralytics import YOLO
    path = os.path.join(ROOT, CKPT[tag])
    got = sha256_file(path)
    if tag in EXPECT_SHA and got != EXPECT_SHA[tag]:
        raise SystemExit(f"CHECKPOINT SHA MISMATCH {tag}: {got}")
    model = YOLO(path, task="pose").model.float().eval()   # fuse 호출하지 않는다
    assert not model.training, "model.train() 금지"
    for p in model.parameters():
        p.requires_grad_(False)
    return model, got


def letterbox_batch(paths):
    from ultralytics.data.augment import LetterBox
    lb = LetterBox((IMGSZ, IMGSZ), auto=False, scale_fill=False)
    tensors, sizes = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            return None, None
        sizes.append(img.shape[:2])
        t = lb(image=img).transpose(2, 0, 1)[::-1].copy()
        tensors.append(torch.from_numpy(t).float() / 255.0)
    return torch.stack(tensors), sizes


def level_bounds(feats):
    counts = [f.shape[2] * f.shape[3] for f in feats]
    return np.cumsum([0] + counts), counts


@torch.no_grad()
def score_images(model, paths, device):
    """이미지당 hardness 레코드."""
    head = model.model[-1]
    rows = []
    for start in range(0, len(paths), BATCH):
        chunk = paths[start:start + BATCH]
        x, _ = letterbox_batch(chunk)
        if x is None:
            continue
        _yout, preds = model(x.to(device))
        o2o = preds["one2one"]
        logits = o2o["scores"][:, 0, :].float().cpu()          # (bs, A) raw
        dbox = head._get_decode_boxes(o2o).float().cpu()        # (bs, 4, A) xyxy px
        bounds, _counts = level_bounds(o2o["feats"])
        conf = torch.sigmoid(logits)

        for b, path in enumerate(chunk):
            z, p = logits[b], conf[b]
            order = torch.argsort(z, descending=True)
            top = order[:50]
            i = int(top[0])
            lvl = int(np.searchsorted(bounds, i, side="right") - 1)
            x1, y1, x2, y2 = dbox[b, :, i].tolist()
            w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
            mass = float(p.sum())                               # 해석적 gradient mass
            share = lambda k: float(p[top[:k]].sum()) / mass if mass > 0 else 0.0  # noqa: E731
            rows.append({
                "image_id": os.path.splitext(os.path.basename(path))[0],
                "path": os.path.relpath(path, ROOT),
                "max_raw_logit": float(z[i]),
                "max_raw_conf": float(p[i]),
                "top1_level": f"P{3 + lvl}",
                "top1_box_area_norm": (w * h) / (IMGSZ * IMGSZ),
                "top1_box_aspect": w / h,
                "top5_logits": [float(v) for v in z[order[:5]]],
                "top5_conf": [float(v) for v in p[order[:5]]],
                "grad_mass_total": mass,
                "grad_share_top1": share(1),
                "grad_share_top5": share(5),
                "grad_share_top10": share(10),
                "grad_share_top50": share(50),
                "n_anchors": int(z.numel()),
            })
        if (start // BATCH) % 40 == 0:
            print(f"    {start + len(chunk)}/{len(paths)}", flush=True)
    return rows


def distribution(rows):
    c = np.array([r["max_raw_conf"] for r in rows])
    q = lambda v: float(np.percentile(c, v))  # noqa: E731
    return {
        "n": len(rows), "mean": float(c.mean()), "median": float(np.median(c)),
        "p75": q(75), "p90": q(90), "p95": q(95), "p99": q(99), "max": float(c.max()),
        "frac_ge_0.01": float((c >= 0.01).mean()), "frac_ge_0.05": float((c >= 0.05).mean()),
        "frac_ge_0.10": float((c >= 0.10).mean()), "frac_ge_0.20": float((c >= 0.20).mean()),
        "frac_ge_0.40": float((c >= 0.40).mean()),
        "level_mix": {lv: int(sum(1 for r in rows if r["top1_level"] == lv))
                      for lv in ("P3", "P4", "P5")},
    }


def gradient_mass(rows):
    out = {}
    for k in ("top1", "top5", "top10", "top50"):
        v = np.array([r[f"grad_share_{k}"] for r in rows])
        out[k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                  "p90": float(np.percentile(v, 90))}
    out["grad_mass_total"] = {
        "mean": float(np.mean([r["grad_mass_total"] for r in rows])),
        "median": float(np.median([r["grad_mass_total"] for r in rows]))}
    out["note"] = ("negative target y=0 에서 dL/dz = sigmoid(z) 이므로 "
                   "gradient mass = sum(sigmoid). backward 하지 않았다.")
    return out


def rank_hard(rows):
    """H(x) = max raw logit 내림차순.  동률은 sha256(stem) 오름차순."""
    keyed = [(r, hashlib.sha256(r["image_id"].encode()).hexdigest()) for r in rows]
    keyed.sort(key=lambda t: (-t[0]["max_raw_logit"], t[1]))
    return [r for r, _ in keyed]


def write_csv(path, rows, fields):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], list) else r[k])
                        for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Y0", choices=list(CKPT))
    ap.add_argument("--real-neg", action="store_true",
                    help="real negative DEV 2,689 도 같은 자로 잰다 (진단 전용)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sha = load_frozen(args.ckpt)
    model.to(device)
    print(f"[{args.ckpt}] sha {sha[:16]}…  device {device}", flush=True)

    paths = sorted(os.path.join(NEG9K, n) for n in os.listdir(NEG9K)
                   if n.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"  negative pool {len(paths)}장", flush=True)
    rows = score_images(model, paths, device)

    fields = ["image_id", "path", "max_raw_logit", "max_raw_conf", "top1_level",
              "top1_box_area_norm", "top1_box_aspect", "top5_logits", "top5_conf",
              "grad_mass_total", "grad_share_top1", "grad_share_top5",
              "grad_share_top10", "grad_share_top50", "n_anchors"]
    write_csv(os.path.join(OUT, f"NEG9K_HARDNESS_PER_IMAGE__{args.ckpt}.csv"),
              rows, fields)

    ranked = rank_hard(rows)
    hard = ranked[:N_HARD]
    members = [r["image_id"] for r in hard]
    membership_path = os.path.join(OUT, f"HARD_NEG1900_MEMBERSHIP__{args.ckpt}.txt")
    with open(membership_path, "w") as f:
        f.write("\n".join(members) + "\n")
    write_csv(os.path.join(OUT, f"HARD_NEG1900_SCORES__{args.ckpt}.csv"), hard, fields)

    dist_pool, dist_hard = distribution(rows), distribution(hard)
    gate_a = (dist_hard["mean"] >= 0.10) or (dist_hard["frac_ge_0.10"] >= 0.25)
    payload = {
        "checkpoint": args.ckpt, "checkpoint_sha256": sha,
        "n_pool": len(rows), "n_hard": len(hard),
        "hardness_definition": "H(x) = max raw one2one pallet class logit",
        "tie_break": "sha256(image_id) ascending",
        "preprocess": f"LetterBox({IMGSZ}) — 학습 파이프라인과 동일, PAD=100 아님",
        "pool": dist_pool, "hard1900": dist_hard,
        "membership_sha256": hashlib.sha256(
            ("\n".join(members) + "\n").encode()).hexdigest(),
        "GATE_A": {
            "criterion_A_mean_ge_0.10": dist_hard["mean"] >= 0.10,
            "criterion_B_frac_ge_0.10_ge_0.25": dist_hard["frac_ge_0.10"] >= 0.25,
            "verdict": ("HARD_NEGATIVE_POOL_HAS_SIGNAL" if gate_a
                        else "EXISTING_NEGATIVE_POOL_TOO_EASY")},
    }
    json.dump(payload, open(os.path.join(
        OUT, f"NEG9K_HARDNESS_DISTRIBUTION__{args.ckpt}.json"), "w"), indent=1)
    json.dump({"pool": gradient_mass(rows), "hard1900": gradient_mass(hard)},
              open(os.path.join(OUT, f"ANCHOR_GRADIENT_MASS__{args.ckpt}.json"), "w"),
              indent=1)

    if args.real_neg:
        rn = []
        for base, dirs, names in os.walk(REAL_NEG):
            if os.path.basename(base) == "depth":
                dirs[:] = []
                continue
            rn += [os.path.join(base, n) for n in sorted(names)
                   if n.lower().endswith((".png", ".jpg", ".jpeg"))]
        rn = sorted(rn)
        print(f"  real negative DEV {len(rn)}장 (진단 전용 — membership 변경 없음)",
              flush=True)
        rrows = score_images(model, rn, device)
        json.dump({"note": "SECONDARY diagnosis only — 선택에 사용 금지",
                   "distribution": distribution(rrows)},
                  open(os.path.join(OUT, f"REALNEG_HARDNESS__{args.ckpt}.json"), "w"),
                  indent=1)

    print(f"\n[{args.ckpt}] pool  median {dist_pool['median']:.4f}  "
          f"p90 {dist_pool['p90']:.4f}  p95 {dist_pool['p95']:.4f}  "
          f"p99 {dist_pool['p99']:.4f}  >=.10 {dist_pool['frac_ge_0.10']:.3f}")
    print(f"[{args.ckpt}] hard  mean {dist_hard['mean']:.4f}  "
          f"median {dist_hard['median']:.4f}  "
          f"p10 {np.percentile([r['max_raw_conf'] for r in hard], 10):.4f}  "
          f">=.10 {dist_hard['frac_ge_0.10']:.3f}")
    print(f"[{args.ckpt}] GATE_A = {payload['GATE_A']['verdict']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
