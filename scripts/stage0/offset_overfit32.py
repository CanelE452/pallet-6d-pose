"""offset_overfit32.py — §2 STAGE8 center-conditioned offset head 구현 sanity 게이트.

NVIDIA식 center-conditioned offset(중심 heatmap peak 주변 support 에서 8 corner 까지
full 2D offset 회귀, ★마스크 없음)이 "구현이 옳은가"를 32장 암기로 검증.

구조:
  frozen base = challenge0123 heatmap ckpt(belief+affinity). 공유 VGG feature out1(128ch,
  50×50)에 offset head 부착:
    Sequential(Conv2d(128,128,3,pad1), ReLU, Conv2d(128,16,1))   # 16 = 8 corner×(dx,dy)
  ★ encoder+decoder+belief/affinity head 전부 freeze, offset head 만 학습.

좌표(output stride=8, 50×50 map):
  center_out = center_in/8, corner_out = corner_in/8.
  support = GT center output좌표 주변 radius 2 output px.
  support 내 위치 q 의 target: T_k(q) = ((x_k - q_x)/W, (y_k - q_y)/H), W=H=50.
  ★ full 2D offset(단위벡터 아님), output-map 크기(50)로 정규화(object-diagonal 금지).

Loss(Gaussian-weighted support, mask 없음):
  weight = center_gaussian(support 내) * valid_support;
  loss_map = smooth_l1(pred, target, reduction=none);
  loss = (loss_map*weight).sum() / (weight.sum()*16 + 1e-6). support 밖 미계산. linear out.

통과 기준: GT center 위치 predicted offset 으로 복원한 corner median err < 0.5 output cell,
p95 < 1 cell. 못하면 인덱싱/stride/부호/denom 버그(표현 부정 금지).

데이터: synthetic 32장(val), augmentation 없음, 반복.
출력: data/pallet/eval_results/stage8_offset/overfit32/
★ 4096/대규모 학습 실행 금지 — 32 overfit 게이트까지만.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from models import DopeNetwork  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
INPUT = 400
GRID = 50              # belief/feature map size
STRIDE = INPUT / GRID  # 8.0
RADIUS = 2             # support radius in output px
GAUSS_SIGMA = 1.0      # center gaussian for weighting
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results",
                       "stage8_offset", "overfit32")
VAL_DIR = os.path.join(ROOT, "data", "pallet", "training_data", "val")


class OffsetHead(nn.Module):
    """out1(128ch,50×50) -> 16ch(8 corner×dx,dy) linear offset field."""
    def __init__(self, c=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 16, 1),
        )

    def forward(self, feat):
        return self.net(feat)   # (B,16,H,W) linear


def load_frozen_base(weights, device):
    """heatmap ckpt strict 로드 후 전부 freeze, eval. forward 는 vgg(out1) 만 사용."""
    model = DopeNetwork()
    state = torch.load(weights, map_location=device)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)   # ★ strict
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def preprocess_fixed(img):
    """고정 400×400 (overfit 단순화 — aspect 왜곡 허용. corner 좌표도 동일 스케일로
    변환하므로 일관). 반환: tensor, sx, sy (orig->input scale)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]
    r = cv2.resize(rgb, (INPUT, INPUT))
    t = ((r.astype(np.float32) / 255.0 - MEAN) / STD)
    tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, INPUT / w0, INPUT / h0


def load_sample(jp, ip):
    """반환: tensor(1,3,400,400), corner_out(8,2 output px), center_out(2,)."""
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8_orig = np.array(o["projected_cuboid"], float)[:8]
    ctr_orig = np.array(o["projected_cuboid_centroid"], float)
    img = cv2.imread(ip)
    tensor, sx, sy = preprocess_fixed(img)
    # orig px -> input px -> output px (/STRIDE)
    corner_out = np.stack([gt8_orig[:, 0] * sx, gt8_orig[:, 1] * sy], 1) / STRIDE
    center_out = np.array([ctr_orig[0] * sx, ctr_orig[1] * sy]) / STRIDE
    return tensor, corner_out, center_out


def build_targets(corner_out, center_out):
    """support 내 각 셀 q 의 target offset (16,H,W) + weight (H,W).

    T_k(q) = ((x_k - q_x)/GRID, (y_k - q_y)/GRID).  full 2D offset.
    weight = center_gaussian * valid_support(=1 if |q-center|_inf <= RADIUS).
    """
    H = W = GRID
    target = np.zeros((16, H, W), np.float32)
    weight = np.zeros((H, W), np.float32)
    cx, cy = center_out
    ci, cj = int(round(cy)), int(round(cx))   # row, col
    for di in range(-RADIUS, RADIUS + 1):
        for dj in range(-RADIUS, RADIUS + 1):
            qi, qj = ci + di, cj + dj
            if not (0 <= qi < H and 0 <= qj < W):
                continue
            qx, qy = float(qj), float(qi)   # cell center (col=x, row=y)
            for k in range(8):
                target[2 * k, qi, qj] = (corner_out[k, 0] - qx) / GRID
                target[2 * k + 1, qi, qj] = (corner_out[k, 1] - qy) / GRID
            dist2 = (qx - cx) ** 2 + (qy - cy) ** 2
            weight[qi, qj] = float(np.exp(-dist2 / (2 * GAUSS_SIGMA ** 2)))
    return target, weight


def decode_at_center(pred, center_out):
    """GT center 셀의 predicted offset 으로 8 corner 복원 (output px).
    pred (16,H,W) numpy. 반환 (8,2) output px."""
    cx, cy = center_out
    ci, cj = int(round(cy)), int(round(cx))
    qx, qy = float(cj), float(ci)
    out = np.zeros((8, 2))
    for k in range(8):
        dx = pred[2 * k, ci, cj] * GRID
        dy = pred[2 * k + 1, ci, cj] * GRID
        out[k] = (qx + dx, qy + dy)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(
        ROOT, "weights", "challenge0123", "final_net_epoch_0060.pth"))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = load_frozen_base(args.weights, device)
    head = OffsetHead(128).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    sl1 = nn.SmoothL1Loss(reduction="none")

    files = sorted(glob.glob(os.path.join(VAL_DIR, "*.json")))[:args.n]
    samples = []
    for jp in files:
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(VAL_DIR, fid + ".png")
        if not os.path.exists(ip):
            continue
        tensor, corner_out, center_out = load_sample(jp, ip)
        # skip frames whose center falls outside the grid
        if not (0 <= center_out[0] < GRID and 0 <= center_out[1] < GRID):
            continue
        target, weight = build_targets(corner_out, center_out)
        samples.append({
            "fid": fid,
            "tensor": tensor.to(device),
            "target": torch.from_numpy(target).unsqueeze(0).to(device),
            "weight": torch.from_numpy(weight).unsqueeze(0).unsqueeze(0).to(device),
            "corner_out": corner_out,
            "center_out": center_out,
        })
    print(f"[data] {len(samples)} samples (requested {args.n})")

    # cache frozen features (base never changes)
    with torch.no_grad():
        for s in samples:
            s["feat"] = base.vgg(s["tensor"])   # (1,128,50,50)

    log = []
    for it in range(1, args.iters + 1):
        head.train()
        opt.zero_grad()
        total = 0.0
        for s in samples:
            pred = head(s["feat"])              # (1,16,50,50)
            w = s["weight"]                     # (1,1,50,50)
            lm = sl1(pred, s["target"])         # (1,16,50,50)
            num = (lm * w).sum()
            den = w.sum() * 16 + 1e-6
            loss = num / den
            loss.backward()
            total += float(loss)
        opt.step()
        if it % 250 == 0 or it == 1:
            print(f"[it {it}] loss={total/len(samples):.5f}")
            log.append({"iter": it, "loss": total / len(samples)})

    # eval: decode at GT center, corner err in output cells
    head.eval()
    all_err = []
    per_frame = []
    with torch.no_grad():
        for s in samples:
            pred = head(s["feat"])[0].cpu().numpy()   # (16,50,50)
            dec = decode_at_center(pred, s["center_out"])  # (8,2) out px
            err = np.linalg.norm(dec - s["corner_out"], axis=1)  # cells
            all_err.extend(err.tolist())
            per_frame.append({"fid": s["fid"],
                              "median_err_cell": float(np.median(err)),
                              "max_err_cell": float(err.max())})
    all_err = np.array(all_err)
    med = float(np.median(all_err))
    p95 = float(np.percentile(all_err, 95))
    passed = (med < 0.5) and (p95 < 1.0)

    lines = [
        "# STAGE8 offset overfit32 — center-conditioned mask-free offset head",
        f"samples={len(samples)} iters={args.iters} lr={args.lr}",
        f"corner err (output cells): median={med:.4f}  p95={p95:.4f}",
        f"pass criteria: median<0.5 AND p95<1.0  =>  {'PASS' if passed else 'FAIL'}",
        "",
        f"{'frame':<22}{'med_cell':>10}{'max_cell':>10}",
        "─" * 42,
    ]
    for pf in sorted(per_frame, key=lambda x: -x["median_err_cell"]):
        lines.append(f"{pf['fid']:<22}{pf['median_err_cell']:>10.3f}"
                     f"{pf['max_err_cell']:>10.3f}")
    table = "\n".join(lines)
    print("\n" + table)

    with open(os.path.join(OUT_DIR, "overfit32.txt"), "w") as f:
        f.write(table + "\n")
    json.dump({"n_samples": len(samples), "iters": args.iters, "lr": args.lr,
               "median_cell": med, "p95_cell": p95, "passed": bool(passed),
               "loss_log": log, "per_frame": per_frame},
              open(os.path.join(OUT_DIR, "overfit32.json"), "w"), indent=2)
    print(f"\n[save] {OUT_DIR}/overfit32.json  passed={passed}")


if __name__ == "__main__":
    main()
