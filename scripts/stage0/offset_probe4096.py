"""offset_probe4096.py — §3 teacher-forced center-conditioned offset probe.

frozen base(challenge0123 heatmap) + offset head 만 학습(4096 synthetic). 그 뒤
3 decode 를 같은 프레임에서 비교:
  A heatmap        : belief peak corner (baseline)
  B GT-center off  : GT centroid 셀의 predicted offset 으로 corner 복원 (표현 학습가능성)
  C pred-center off: 예측 centroid peak 셀의 offset (실제 inference 조건)
지표: corner median(overall/front/back, order-free hungarian) + good<10/gross>20.
mask/edge/diag/fusion 없음. offset_overfit32 의 head/target/decode 재사용.

전처리는 overfit 과 동일 fixed-squash 400×400(3 arm 모두 동일 → 공정). heatmap baseline
도 같은 전처리라 eval_pvnet_heads(aspect) 수치와 절대값은 다를 수 있음 — 여기선 arm 간
상대비교가 목적.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))

import torch  # noqa
from offset_overfit32 import (  # noqa
    OffsetHead, load_frozen_base, preprocess_fixed, build_targets,
    decode_at_center, GRID, STRIDE, MEAN, STD, INPUT)
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa

sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
from eval_pvnet_heads import split_metrics  # noqa
from four_arm_pl_compare import collect_val_frames  # noqa

TRAIN_DIRS = [
    os.path.join(ROOT, "data/pallet/training_data/mixed_v8_train"),
    os.path.join(ROOT, "challenge/data/training/v1"),
    os.path.join(ROOT, "challenge/data/training/v2"),
]
VAL_DIR = os.path.join(ROOT, "data/pallet/training_data/val")
MANUAL = os.path.join(ROOT, "data/pallet/eval_results/stage0_gt_candidates/manual_gt")
OUT = os.path.join(ROOT, "data/pallet/eval_results/stage8_offset/probe4096")
GOOD, GROSS = 10.0, 20.0


def list_pairs(dirs, n):
    pairs = []
    for d in dirs:
        for jp in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            ip = jp[:-5] + ".png"
            if os.path.exists(ip):
                pairs.append((jp, ip))
    pairs.sort()
    if n and n < len(pairs):
        idx = np.linspace(0, len(pairs) - 1, n).round().astype(int)
        pairs = [pairs[int(i)] for i in sorted(set(idx))]
    return pairs


def load_train_sample(jp, ip):
    d = json.load(open(jp)); o = d["objects"][0]
    g8 = np.array(o["projected_cuboid"], float)[:8]
    ct = np.array(o["projected_cuboid_centroid"], float)
    img = cv2.imread(ip)
    if img is None:
        return None
    tensor, sx, sy = preprocess_fixed(img)
    corner_out = np.stack([g8[:, 0] * sx, g8[:, 1] * sy], 1) / STRIDE
    center_out = np.array([ct[0] * sx, ct[1] * sy]) / STRIDE
    if not (0 <= center_out[0] < GRID and 0 <= center_out[1] < GRID):
        return None
    tgt, w = build_targets(corner_out, center_out)
    return tensor, tgt, w


def train(base, head, device, pairs, iters, lr, bs=16, unfreeze=False):
    import torch.nn.functional as F
    groups = [{"params": head.parameters(), "lr": lr}]
    vgg_train = []
    if unfreeze:
        # §6: decoder 마지막 block(out1 생성 conv "23","24","25","26") unfreeze, 낮은 lr
        for name, m in base.vgg.named_children():
            if name in ("23", "24", "25", "26"):
                for p in m.parameters():
                    p.requires_grad_(True); vgg_train.append(p)
        groups.append({"params": vgg_train, "lr": lr * 0.1})
        print(f"[unfreeze] vgg last block params={len(vgg_train)} lr={lr*0.1}")
    opt = torch.optim.AdamW(groups)
    cache = []
    print(f"[train] caching {len(pairs)} samples ...")
    for jp, ip in pairs:
        s = load_train_sample(jp, ip)
        if s is not None:
            cache.append(s)
    print(f"[train] usable={len(cache)}  iters={iters} bs={bs} lr={lr}")
    rng = np.random.RandomState(0)
    head.train()
    for it in range(iters):
        idx = rng.randint(0, len(cache), bs)
        tens = torch.cat([cache[i][0] for i in idx]).to(device)
        tgt = torch.from_numpy(np.stack([cache[i][1] for i in idx])).to(device)
        wt = torch.from_numpy(np.stack([cache[i][2] for i in idx])).unsqueeze(1).to(device)
        if unfreeze:
            feat = base.vgg(tens)          # grad flows into last block
        else:
            with torch.no_grad():
                feat = base.vgg(tens)
        pred = head(feat)
        lm = F.smooth_l1_loss(pred, tgt, reduction="none")
        loss = (lm * wt).sum() / (wt.sum() * 16 + 1e-6)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 250 == 0 or it == iters - 1:
            print(f"  it{it:5d}  loss={loss.item():.5f}")
    head.eval()


def decode_pred_center(pred, kps_bel):
    """예측 centroid(kp index 8) peak 셀에서 offset 읽어 corner 복원 (output px)."""
    c = kps_bel[8]
    if c[0] < 0:
        return None
    return decode_at_center(pred, (c[0], c[1]))


def to_orig(pts_out, sx, sy):
    """output px -> input px(*STRIDE) -> orig px(/s)."""
    o = pts_out * STRIDE
    return np.stack([o[:, 0] / sx, o[:, 1] / sy], 1)


def evalset(base, head, device, frames, label, lines):
    import torch.nn.functional as F
    recs = {"A_heatmap": [], "B_gtcenter": [], "C_predcenter": []}
    for jp, ip in frames:
        d = json.load(open(jp)); o = d["objects"][0]
        g8 = np.array(o["projected_cuboid"], float)[:8]
        ct = np.array(o["projected_cuboid_centroid"], float)
        img = cv2.imread(ip)
        if img is None:
            continue
        tensor, sx, sy = preprocess_fixed(img)
        center_out = np.array([ct[0] * sx, ct[1] * sy]) / STRIDE
        with torch.no_grad():
            beliefs, _ = base(tensor.to(device))
            feat = base.vgg(tensor.to(device))
            pred = head(feat)[0].cpu().numpy()
        belief = beliefs[-1][0].cpu().numpy()
        kps_bel = extract_keypoints_from_belief(belief, 0.3)
        # A heatmap (belief peak corners 0-7 -> orig)
        ph = np.full((8, 2), np.nan)
        for i in range(8):
            if kps_bel[i][0] >= 0:
                ph[i] = (kps_bel[i][0] * STRIDE / sx, kps_bel[i][1] * STRIDE / sy)
        recs["A_heatmap"].append(split_metrics(ph, g8))
        # B GT-center offset
        if 0 <= center_out[0] < GRID and 0 <= center_out[1] < GRID:
            cb = to_orig(decode_at_center(pred, center_out), sx, sy)
            recs["B_gtcenter"].append(split_metrics(cb, g8))
        # C pred-center offset
        dc = decode_pred_center(pred, kps_bel)
        if dc is not None:
            recs["C_predcenter"].append(split_metrics(to_orig(dc, sx, sy), g8))
    lines.append(f"\n=== {label} ===")
    lines.append(f"  {'arm':<14}{'N':>4}{'ov_med':>8}{'fr_med':>8}{'bk_med':>8}{'good':>6}{'gross':>7}")
    for arm in ("A_heatmap", "B_gtcenter", "C_predcenter"):
        r = recs[arm]
        ov = [x["overall"] for x in r if np.isfinite(x["overall"])]
        fr = [x["front"] for x in r if np.isfinite(x["front"])]
        bk = [x["back"] for x in r if np.isfinite(x["back"])]
        if not ov:
            lines.append(f"  {arm:<14}{0:>4}{'--':>8}"); continue
        g = sum(1 for v in ov if v < GOOD); gr = sum(1 for v in ov if v > GROSS)
        lines.append(f"  {arm:<14}{len(ov):>4}{np.median(ov):>8.1f}"
                     f"{(np.median(fr) if fr else float('nan')):>8.1f}"
                     f"{(np.median(bk) if bk else float('nan')):>8.1f}{g:>6}{gr:>7}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(
        ROOT, "weights/challenge0123/final_net_epoch_0060.pth"))
    ap.add_argument("--n_train", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--unfreeze", action="store_true",
                    help="§6: decoder 마지막 block(out1 conv) unfreeze 재시험")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = "_unfrozen" if args.unfreeze else ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_frozen_base(args.weights, device)
    head = OffsetHead().to(device)

    train_pairs = list_pairs(TRAIN_DIRS, args.n_train)
    train(base, head, device, train_pairs, args.iters, args.lr, unfreeze=args.unfreeze)
    torch.save(head.state_dict(), os.path.join(OUT, f"offset_head{tag}.pth"))

    lines = [f"STAGE8 §{'6 unfrozen' if args.unfreeze else '3'} offset probe "
             f"({'decoder last block + ' if args.unfreeze else ''}offset head, 4096 train)",
             f"weights={args.weights} n_train={len(train_pairs)} iters={args.iters} unfreeze={args.unfreeze}",
             "A=heatmap baseline / B=GT-center offset / C=pred-center offset",
             "⚠ unfreeze 시 trunk 바뀌어 A도 §3와 다름 → 같은 런 A 기준 비교"]
    val_pairs = list_pairs([VAL_DIR], args.n_val)
    evalset(base, head, device, val_pairs, f"synthetic val ({len(val_pairs)})", lines)
    man_pairs = list_pairs([MANUAL], 0)
    evalset(base, head, device, man_pairs, f"manual GT real ({len(man_pairs)})", lines)
    fv = [(jp, ip) for _, _, jp, ip in collect_val_frames()]
    evalset(base, head, device, fv, f"filter-val real ({len(fv)})", lines)
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(OUT, f"probe4096{tag}.txt"), "w").write(txt)
    print(f"\n[save] {OUT}/probe4096{tag}.txt")


if __name__ == "__main__":
    main()
