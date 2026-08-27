"""PAIRWISE SIGNAL AUDIT — raw one2one anchor 수준에서 hard unassigned negative 존재 확인.

★ backward 0 · optimizer 0 · model.train() 0 · fuse 0 · 가중치 변경 0.
gradient 는 '가상' 해석식으로만 계산한다 (autograd 미사용).
"""
from __future__ import annotations
import csv, hashlib, json, os, sys, time

import numpy as np, cv2, torch

TA = "/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/p26_tal_target_audit"
sys.path.insert(0, TA)
import ta_core as TC                                                # noqa: E402
from ultralytics.utils.tal import make_anchors                      # noqa: E402

ROOT, Y, W = TC.ROOT, TC.Y, TC.W
NS = f"{Y}/p26_pairwise_signal_audit"
DS = f"{Y}/datasets/g38_generic_only"
QY = f"{Y}/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
MANI = ("/home/minjae/pallet_worker_transfer_20260821T105141Z/"
        "REAL_GT_QA_20260821T133405Z/REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")
NIGHT = {"eval_night08", "eval_night09"}
DUP_T, NEAR_T = 0.5, 0.1                     # 실행 전 고정


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gt_synth(lp, w, h):
    v = [float(x) for x in open(lp).read().strip().split("\n")[0].split()]
    cx, cy, bw, bh = v[1:5]
    k = np.array(v[5:]).reshape(-1, 3)
    return ([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h],
            np.stack([k[:, 0]*w, k[:, 1]*h], 1))


def gt_real(jp):
    o = json.load(open(jp))["objects"][0]
    g = np.array(o["projected_cuboid"], float)[:8]
    c = np.array(o["projected_cuboid_centroid"], float)
    return ([g[:, 0].min(), g[:, 1].min(), g[:, 0].max(), g[:, 1].max()], np.vstack([g, c]))


P = TC.TALProbe()
ROWS = []


def iou_all(boxes, gt):
    x1 = np.maximum(boxes[:, 0], gt[0]); y1 = np.maximum(boxes[:, 1], gt[1])
    x2 = np.minimum(boxes[:, 2], gt[2]); y2 = np.minimum(boxes[:, 3], gt[3])
    inter = np.clip(x2-x1, 0, None) * np.clip(y2-y1, 0, None)
    a = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
    b = (gt[2]-gt[0]) * (gt[3]-gt[1])
    return inter / np.maximum(a + b - inter, 1e-9)


@torch.no_grad()
def one_frame(image_id, im, gb_img, gk_img, split, domain, pad):
    r, padded = P.predict(im, pad=pad)
    if "preds" not in P.cap:
        return
    gb = [gb_img[0]+pad, gb_img[1]+pad, gb_img[2]+pad, gb_img[3]+pad]
    gk = gk_img + pad
    gb_in = P.gt_to_input(gb, padded.shape)
    A = P.assign(gb_in)
    c = P.criterion().one2one
    pd = P.cap["preds"]["one2one"]
    anchor_points, stride_tensor = make_anchors(pd["feats"], c.stride, 0.5)
    pb = (c.bbox_decode(anchor_points, pd["boxes"].permute(0, 2, 1).contiguous())
          * stride_tensor)[0].cpu().numpy()                      # (A,4) xyxy input space
    kp = P.head.kpts_decode(pd["kpts"])[0].cpu().numpy()          # (nk, A)
    nk = P.head.kpt_shape[0]
    ious = iou_all(pb, gb_in)
    logits = A["logits"]
    fg = A["fg_mask"] > 0
    if not fg.any():
        return
    ipos = int(np.argmax(np.where(fg, A["target_scores"], -1)))
    un = ~fg
    if not un.any():
        return
    ineg = int(np.argmax(np.where(un, logits, -1e9)))
    r_, dw, dh, _ = P.letterbox_params(padded.shape)
    gk_in = np.stack([gk[:, 0]*r_ + dw, gk[:, 1]*r_ + dh], 1)

    def kperr(a):
        k = kp[:, a].reshape(nk, -1)[:, :2]
        n = min(len(k), len(gk_in))
        return float(np.median(np.linalg.norm(k[:n] - gk_in[:n], axis=1)))

    s_pos, s_neg = float(logits[ipos]), float(logits[ineg])
    d = s_pos - s_neg
    io_n = float(ious[ineg])
    cat = ("DUPLICATE" if io_n >= DUP_T else
           "NEAR_DISTRACTOR" if io_n >= NEAR_T else "FAR_DISTRACTOR")
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    ROWS.append({
        "image_id": image_id, "split": split, "domain": domain, "gt_id": 0,
        "s_pos": s_pos, "s_neg": s_neg,
        "score_pos": float(sig(s_pos)), "score_neg": float(sig(s_neg)),
        "delta": d, "g_rank": float(sig(-d)),
        "target_pos": float(A["target_scores"][ipos]),
        "target_neg": float(A["target_scores"][ineg]),
        "g_bce_pos": float(sig(s_pos) - A["target_scores"][ipos]),
        "g_bce_neg": float(sig(s_neg) - A["target_scores"][ineg]),
        "pos_level": P.level_of(ipos)[0], "neg_level": P.level_of(ineg)[0],
        "pos_iou": float(ious[ipos]), "neg_iou": io_n, "neg_category": cat,
        "pos_kp_err": kperr(ipos), "neg_kp_err": kperr(ineg),
        "n_unassigned": int(un.sum()), "n_anchors": A["n_anchors"]})


# ---------------------------------------------------------------- membership
stems = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{DS}/images/train"))
key = lambda s: hashlib.sha256(s.encode()).hexdigest()
train5k = sorted(stems, key=key)[:5000]
open(f"{NS}/MEMBERSHIP_TRAIN5K.txt", "w").write("\n".join(train5k) + "\n")
log(f"train5k membership {len(train5k)}  sha16 "
    f"{hashlib.sha256(chr(10).join(train5k).encode()).hexdigest()[:16]}")

for n_, s in enumerate(train5k):
    im = cv2.imread(f"{DS}/images/train/{s}.png")
    lp = f"{DS}/labels/train/{s}.txt"
    if im is None or not os.path.exists(lp):
        continue
    h, w = im.shape[:2]
    gb, gk = gt_synth(lp, w, h)
    one_frame(s, im, gb, gk, "train5k", "SYNTH", 0)
    if (n_ + 1) % 1000 == 0:
        log(f"  train5k {n_+1}/5000")
log(f"train5k rows {sum(1 for r in ROWS if r['split']=='train5k')}")

vf = sorted(os.listdir(f"{DS}/images/val"))
for n_, f in enumerate(vf):
    im = cv2.imread(f"{DS}/images/val/{f}")
    lp = f"{DS}/labels/val/{os.path.splitext(f)[0]}.txt"
    if im is None or not os.path.exists(lp):
        continue
    h, w = im.shape[:2]
    gb, gk = gt_synth(lp, w, h)
    one_frame(os.path.splitext(f)[0], im, gb, gk, "val1998", "SYNTH", 0)
    if (n_ + 1) % 500 == 0:
        log(f"  val {n_+1}/{len(vf)}")
log(f"val1998 rows {sum(1 for r in ROWS if r['split']=='val1998')}")

LEAK = set(json.load(open(f"{QY}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
for it in json.load(open(MANI))["items"]:
    if it["frame_id"] in LEAK:
        continue
    ip, jp = os.path.join(ROOT, it["image"]), os.path.join(ROOT, it["label"])
    if not (os.path.exists(ip) and os.path.exists(jp)):
        continue
    im = cv2.imread(ip)
    gb, gk = gt_real(jp)
    one_frame(it["frame_id"], im, gb, gk, "realdev128",
              "NIGHT" if it.get("set") in NIGHT else "DAY", TC.PAD)
log(f"realdev rows {sum(1 for r in ROWS if r['split']=='realdev128')}")

keys = list(ROWS[0].keys())
with open(f"{NS}/PAIRWISE_SIGNAL_PER_FRAME.csv", "w", newline="") as fh:
    w_ = csv.DictWriter(fh, fieldnames=keys)
    w_.writeheader()
    w_.writerows(ROWS)
log(f"저장 완료 {len(ROWS)}")
