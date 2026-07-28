"""measure_lambda_gradients.py — STEP 3 λ 시작값 근거 측정 (학습 X, 1 step).

PVNet 2헤드 DOPE 의 4개 보조 loss 각각의 gradient 크기를 heatmap(belief) loss
대비 측정해 λ_k 시작값을 근거 있게 정한다. ★ 1 batch, 1 forward + per-loss
backward 만 (학습 아님). 모델/로더/loss 동작 변경 없음 — 전부 재사용.

비교면(공정성): 4개 loss 가 흐르는 head 는 서로 다르다
  - hm   : belief head + 공유 VGG
  - diag : soft_argmax(belief) → belief head + 공유 VGG
  - edge : soft_argmax(belief) → belief head + 공유 VGG
  - vec  : vec head + 공유 VGG
따라서 *공유 VGG trunk* 파라미터의 grad L2 norm 을 공통 비교면으로 쓴다.
r_k = g_k / g_hm. 권장 λ_k = target_frac / r_k (λ_k·r_k ≈ target_frac).

L_diag (diag→centroid) 정식화: filt_diag 의 공간 대각선 (0-6,1-7,2-4,3-5) 가
centroid(8) 를 통과해야 한다는 동일 동기. 선-교점은 거의 평행 시 불안정하므로
대신 **각 대각선 직선에 대한 centroid 의 point-line distance** 합을 쓴다
(division 없음 → 미분 안정). object_diagonal 로 정규화 후 huber.
GT 불필요(예측 키포인트 자기일관 prior), 필터와 동일.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "train"))

from models import DopeNetwork  # noqa: E402
from utils_dataset import CleanVisiiDopeLoader  # noqa: E402
from geo_loss_bpnp import SpatialSoftArgmax2D  # noqa: E402
from geo_loss_struct import SparseEdgeLoss  # noqa: E402
from geo_loss_pose import object_diagonal, huber  # noqa: E402

# filt_diag 의 공간 대각선 정의 (camera-facing 0123, filter_pr_camfacing 와 동일)
SPACE_DIAG = [(0, 6), (1, 7), (2, 4), (3, 5)]


# ── L_diag: 공간 대각선이 centroid 통과 (point-line distance, 미분안정) ──
def diag_centroid_loss(pred_kp, delta=0.03):
    """pred_kp (B,9,2). 각 공간 대각선 직선에 대한 centroid(8) 의 수직거리를
    object_diagonal 로 정규화 후 huber 평균. 교점(division) 없이 안정·미분가능."""
    c = pred_kp[:, 8]                       # (B,2) centroid
    diag = object_diagonal(pred_kp)         # (B,) clamp(min=1.0)
    dists = []
    for (i, j) in SPACE_DIAG:
        p = pred_kp[:, i]                   # (B,2)
        q = pred_kp[:, j]
        d = q - p                           # 대각선 방향
        L = torch.norm(d, dim=-1).clamp_min(1e-6)   # (B,)
        # point-line distance = |cross(d, c-p)| / |d|
        w = c - p                           # (B,2)
        cross = d[:, 0] * w[:, 1] - d[:, 1] * w[:, 0]   # (B,)
        dist = cross.abs() / L              # (B,)
        dists.append(dist / diag)           # scale-free
    dn = torch.stack(dists, dim=-1)         # (B,4)
    return huber(dn, delta).mean()


# ── L_vec: dense vector head vs GT (masked smooth-L1) ──
def vec_loss(vec_stages, gt_vec, gt_mask):
    """vec_stages: list of (B,18,H,W) (2 stages). gt_vec (B,18,H,W), gt_mask (B,1,H,W).
    belief loss 와 동일하게 stage 합. 마스크 픽셀×18채널 정규화 smooth-L1."""
    total = torch.tensor(0.0, device=gt_vec.device)
    denom = gt_mask.sum().clamp_min(1.0) * gt_vec.shape[1]   # mask_px * 18
    for v in vec_stages:
        diff = (v - gt_vec) * gt_mask       # broadcast mask over 18 ch
        total = total + torch.nn.functional.smooth_l1_loss(
            diff, torch.zeros_like(diff), reduction="sum") / denom
    return total


# ── heatmap (belief) loss = train.py 와 동일 (6 stage MSE 합, 9ch) ──
def heatmap_loss(beliefs, target_belief):
    loss = torch.tensor(0.0, device=target_belief.device)
    for stage in range(len(beliefs)):
        d = beliefs[stage] - target_belief
        loss = loss + (d * d).mean()
    return loss


def vgg_grad_norm(net):
    """공유 VGG trunk 파라미터 전체의 grad L2 norm."""
    sq = 0.0
    n = 0
    for p in net.vgg.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
            n += p.grad.numel()
    return float(np.sqrt(sq)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default=os.path.join(ROOT, "weights", "challenge0123",
                                         "final_net_epoch_0060.pth"))
    ap.add_argument("--batchsize", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_frac", type=float, default=0.07,
                    help="aux gradient 가 hm gradient 의 이 비율이 되도록 λ 권장")
    ap.add_argument("--edge_target_frac", type=float, default=0.02,
                    help="edge 는 868%% 폭발 교훈 — 보수적 목표 비율")
    ap.add_argument("--struct_delta", type=float, default=0.03)
    ap.add_argument("--output_dir",
                    default=os.path.join(ROOT, "data", "pallet",
                                         "eval_results", "stage3_lambda"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dirs = [
        os.path.join(ROOT, "data", "pallet", "training_data", "mixed_v8_train"),
        os.path.join(ROOT, "challenge", "data", "training", "v1"),
        os.path.join(ROOT, "challenge", "data", "training", "v2"),
    ]
    print(f"[device] {device}")
    print(f"[weights] {args.weights}")

    # 로더: pvnet_vec=True (offset 모드), output_size=50, sigma=4.0 (config 기본)
    ds = CleanVisiiDopeLoader(data_dirs, objects=["pallet"], sigma=4.0,
                              output_size=50, pvnet_vec=True, pvnet_unit=False)
    # 1 배치 고정 (seed 고정 sampler 없이 앞에서 batchsize 개)
    samples = [ds[i] for i in range(args.batchsize)]
    batch = {}
    for k in ("img", "beliefs", "affinities", "pvnet_vec", "pvnet_mask"):
        batch[k] = torch.stack([s[k] for s in samples], dim=0).to(device)

    img = batch["img"].float()
    target_belief = batch["beliefs"].float()
    gt_vec = batch["pvnet_vec"].float()
    gt_mask = batch["pvnet_mask"].float()
    print(f"[batch] img {tuple(img.shape)} belief {tuple(target_belief.shape)} "
          f"vec {tuple(gt_vec.shape)} mask {tuple(gt_mask.shape)} "
          f"mask_px/img {gt_mask.sum().item()/args.batchsize:.0f}")

    # 모델: numVec=18, weight strict=False (vec head random)
    net = DopeNetwork(numVec=18).to(device)
    state = torch.load(args.weights, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    vec_missing = [m for m in missing if m.startswith("m_vec")]
    print(f"[load] strict=False  missing={len(missing)} "
          f"(vec-head {len(vec_missing)}) unexpected={len(unexpected)}")
    net.train()

    soft_argmax = SpatialSoftArgmax2D(temperature=1.0)
    edge_loss_mod = SparseEdgeLoss(delta=args.struct_delta).to(device)

    # ── 각 loss 별 독립 backward → 공유 VGG grad norm ──
    def measure(name, loss_fn):
        net.zero_grad(set_to_none=True)
        beliefs, affinities, vec_stages = net(img)
        L = loss_fn(beliefs, vec_stages)
        Lval = float(L.detach().item())
        L.backward()
        g, n = vgg_grad_norm(net)
        return name, Lval, g, n

    results = []

    # 1) heatmap (비교 기준)
    results.append(measure("heatmap",
                   lambda b, v: heatmap_loss(b, target_belief)))

    # 2) vec
    results.append(measure("vec",
                   lambda b, v: vec_loss(v, gt_vec, gt_mask)))

    # 3) diag→centroid (예측 키포인트 자기일관, GT 불필요)
    def _diag(b, v):
        pred_kp = soft_argmax(b[-1][:, :9])
        return diag_centroid_loss(pred_kp, delta=args.struct_delta)
    name, Lval, g, n = measure("diag", _diag)
    # 미분가능 확인: grad 유한·비영 assert
    diag_finite = np.isfinite(g)
    diag_nonzero = g > 0.0
    results.append((name, Lval, g, n))

    # 4) edge (pred vs gt soft_argmax)
    def _edge(b, v):
        pred_kp = soft_argmax(b[-1][:, :9])
        with torch.no_grad():
            gt_kp = soft_argmax(target_belief[:, :9])
        l, _ = edge_loss_mod(pred_kp, gt_kp)
        return l
    results.append(measure("edge", _edge))

    g_hm = results[0][2]

    # ── 표 작성 ──
    target = {"heatmap": 1.0, "vec": args.target_frac, "diag": args.target_frac,
              "edge": args.edge_target_frac}
    lines = []
    lines.append(f"# STEP3 λ gradient measurement (1 batch, seed={args.seed}, "
                 f"bs={args.batchsize})")
    lines.append(f"# weights: {args.weights}")
    lines.append(f"# 비교면 = 공유 VGG trunk grad L2 norm. r_k = g_k / g_hm.")
    lines.append(f"# 권장 λ_k = target_frac_k / r_k  (vec/diag frac="
                 f"{args.target_frac}, edge frac={args.edge_target_frac} 보수)")
    lines.append("")
    hdr = (f"{'loss':<10}{'raw_value':>14}{'g_vgg':>14}"
           f"{'r_k(vs hm)':>13}{'rec_lambda':>13}")
    lines.append(hdr)
    lines.append("─" * len(hdr))
    rec = {}
    for name, Lval, g, n in results:
        r = g / g_hm if g_hm > 0 else float("nan")
        if name == "heatmap":
            lam_str = "1.0 (base)"
        elif r > 0:
            lam = target[name] / r
            rec[name] = lam
            lam_str = f"{lam:.4g}"
        else:
            lam_str = "n/a (g=0)"
        lines.append(f"{name:<10}{Lval:>14.6g}{g:>14.6g}{r:>13.4g}{lam_str:>13}")
    lines.append("")
    lines.append(f"# VGG param count (grad!=None): {results[0][3]}")
    lines.append(f"# diag 미분가능: grad finite={bool(diag_finite)} "
                 f"nonzero={bool(diag_nonzero)} (g_diag={results[2][2]:.6g})")
    lines.append("# ⚠ 1 batch 신호 — 학습 중 ramp/도메인 따라 변동. 시작값 근거용.")

    txt = "\n".join(lines)
    print("\n" + txt + "\n")

    out_txt = os.path.join(args.output_dir, "lambda_gradients.txt")
    with open(out_txt, "w") as f:
        f.write(txt + "\n")

    # JSON 도 저장
    import json
    out_json = os.path.join(args.output_dir, "lambda_gradients.json")
    json.dump({
        "weights": args.weights, "seed": args.seed, "batchsize": args.batchsize,
        "target_frac": args.target_frac, "edge_target_frac": args.edge_target_frac,
        "g_hm": g_hm,
        "results": [{"loss": n, "raw_value": L, "g_vgg": g,
                     "r_k": (g / g_hm if g_hm > 0 else None)}
                    for (n, L, g, _) in results],
        "recommended_lambda": rec,
        "diag_differentiable": {"finite": bool(diag_finite),
                                "nonzero": bool(diag_nonzero),
                                "g_diag": results[2][2]},
    }, open(out_json, "w"), indent=2)

    assert diag_finite and diag_nonzero, \
        f"L_diag NOT differentiable: g={results[2][2]} (finite={diag_finite}, nonzero={diag_nonzero})"
    print(f"[save] {out_txt}\n[save] {out_json}")


if __name__ == "__main__":
    main()
