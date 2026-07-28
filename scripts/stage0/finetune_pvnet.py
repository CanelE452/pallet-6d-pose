"""finetune_pvnet.py — STEP 4 PVNet 2헤드 DOPE 짧은 finetune 하니스.

목적: 벡터 헤드(offset vs unit)가 heatmap 단독보다 키포인트(특히 안 보이는 back
코너) 정확도를 더 잘 찍는지 *학습으로* 검증하기 위한 finetune 러너.

설계 원칙 (제약 준수):
  - train.py / utils_dataset.py / models.py / loss 정의를 **변경하지 않는다**.
    이 스크립트는 별도 러너이며, loss 는 measure_lambda_gradients.py 와
    geo_loss_struct.py 에서 import 만 한다(재정의 금지).
  - net 로드는 strict=False (vec 헤드만 새 파라미터).
  - belief loss = train.py 의 symmetric swap min 로직을 그대로 복제(원본 미변경).
  - vec/diag/edge 보조 loss 는 ramp(=min(1, step/warmup)) 로 켠다(폭발 방지).
  - log_every 마다 각 보조 loss 의 공유 VGG grad norm 비율(measure_lambda_gradients
    방식) + raw loss + ramp 를 stdout + 파일로 출력.

vec_mode:
  off    → numVec=0, PVNet loss 없음 = control(heatmap 단독, 동일 스케줄·공정 비교).
  offset → numVec=18, 로더 pvnet_vec=True, pvnet_unit=False. vec=offset GT.
  unit   → numVec=18, 로더 pvnet_vec=True, pvnet_unit=True.  vec=단위벡터 GT.

⚠ 풀 finetune 은 이 스크립트로 사용자가 직접 돌린다. 스모크(소량·소 step)만 자동.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "train"))

from models import DopeNetwork  # noqa: E402
from utils_dataset import CleanVisiiDopeLoader  # noqa: E402
from geo_loss_bpnp import SpatialSoftArgmax2D  # noqa: E402
from geo_loss_struct import SparseEdgeLoss  # noqa: E402
# loss 재정의 금지 — measure_lambda_gradients 의 vec_loss / diag_centroid_loss 재사용
from measure_lambda_gradients import vec_loss, diag_centroid_loss  # noqa: E402

DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "challenge0123",
                               "final_net_epoch_0060.pth")
DEFAULT_DATA = [
    os.path.join(ROOT, "data", "pallet", "training_data", "mixed_v8_train"),
    os.path.join(ROOT, "challenge", "data", "training", "v1"),
    os.path.join(ROOT, "challenge", "data", "training", "v2"),
]
# 180° Y-rotation swap (train.py line 90 과 동일). belief loss symmetric.
SWAP_IDX = [5, 4, 7, 6, 1, 0, 3, 2, 8]


def belief_loss_symmetric(output_belief, target_belief):
    """train.py _runnetwork 의 belief loss (symmetric swap min, 6-stage 합) 복제.
    원본 동작 변경 금지 — 동일 수식을 여기서 재현(별 스크립트)."""
    loss_belief = torch.tensor(0.0, device=target_belief.device)
    target_swapped = target_belief[:, SWAP_IDX]
    for stage in range(len(output_belief)):
        d0 = output_belief[stage] - target_belief
        ds = output_belief[stage] - target_swapped
        loss_orig = (d0 * d0).mean()
        loss_swap = (ds * ds).mean()
        loss_belief = loss_belief + torch.min(loss_orig, loss_swap)
    return loss_belief


def affinity_loss(output_aff, target_aff):
    """train.py 의 affinity loss (6-stage MSE 합) 복제."""
    loss = torch.tensor(0.0, device=target_aff.device)
    for stage in range(len(output_aff)):
        d = output_aff[stage] - target_aff
        loss = loss + (d * d).mean()
    return loss


def seg_loss(seg_stages, gt_mask):
    """seg_stages: list of (B,1,H,W) logits (2 stages). gt_mask (B,1,H,W) in {0,1}.
    belief/vec loss 와 동일하게 stage 합. BCE-with-logits (수치 안정)."""
    total = torch.tensor(0.0, device=gt_mask.device)
    tgt = (gt_mask > 0.5).float()
    for s in seg_stages:
        total = total + torch.nn.functional.binary_cross_entropy_with_logits(
            s, tgt)
    return total


def vgg_grad_norm(net):
    """공유 VGG trunk grad L2 norm (measure_lambda_gradients 와 동일 비교면)."""
    sq = 0.0
    for p in net.vgg.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
    return float(np.sqrt(sq))


def unpack_forward(net, out, numVec, numSeg):
    """forward 반환을 (beliefs, affinities, vec_stages, seg_stages) 로 정규화.
    arity 규칙: numSeg>0 면 4-tuple(vec 자리 None 가능), 아니면 2/3-tuple."""
    if numSeg > 0:
        beliefs, affinities, vec_stages, seg_stages = out
    elif numVec > 0:
        beliefs, affinities, vec_stages = out
        seg_stages = None
    else:
        beliefs, affinities = out
        vec_stages = seg_stages = None
    return beliefs, affinities, vec_stages, seg_stages


def grad_ratios(net, img, target_belief, target_aff, gt_vec, gt_mask,
                soft_argmax, edge_mod, struct_delta, numVec, numSeg):
    """각 보조 loss 의 공유 VGG grad norm 을 belief grad 대비 측정 (measure_
    lambda_gradients 방식). ★ 진단 전용 — 옵티마이저 그래프와 분리된 자체 forward
    + per-loss backward 를 쓴다(메인 학습 step 에 영향 없음). 반환 r_k=g_k/g_belief."""
    def _g(loss_fn):
        net.zero_grad(set_to_none=True)
        out = net(img)
        beliefs, _, vec_stages, seg_stages = unpack_forward(
            net, out, numVec, numSeg)
        L = loss_fn(beliefs, vec_stages, seg_stages)
        L.backward()
        return vgg_grad_norm(net)

    g_bel = _g(lambda b, v, s: belief_loss_symmetric(b, target_belief))
    g_diag = _g(lambda b, v, s: diag_centroid_loss(
        soft_argmax(b[-1][:, :9]), delta=struct_delta))

    def _edge(b, v, s):
        pred_kp = soft_argmax(b[-1][:, :9])
        with torch.no_grad():
            gt_kp = soft_argmax(target_belief[:, :9])
        l, _ = edge_mod(pred_kp, gt_kp)
        return l
    g_edge = _g(_edge)

    r = {"vec": float("nan"), "diag": float("nan"), "edge": float("nan"),
         "seg": float("nan")}
    if g_bel > 0:
        r["diag"] = g_diag / g_bel
        r["edge"] = g_edge / g_bel
        if numVec > 0:
            g_vec = _g(lambda b, v, s: vec_loss(v, gt_vec, gt_mask))
            r["vec"] = g_vec / g_bel
        if numSeg > 0:
            g_seg = _g(lambda b, v, s: seg_loss(s, gt_mask))
            r["seg"] = g_seg / g_bel
    net.zero_grad(set_to_none=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec_mode", choices=["off", "offset", "unit"],
                    required=True)
    ap.add_argument("--pvnet_mask_rle", action="store_true",
                    help="v3 전용: seg/vec-support 마스크를 mask_rle(real per-pixel)로. "
                         "off 면 cuboid hull(기존 동작).")
    ap.add_argument("--seg", action="store_true",
                    help="seg(마스크) 헤드 추가 (numSeg=1, BCE vs pvnet_mask)")
    ap.add_argument("--lam_seg", type=float, default=1.0,
                    help="seg BCE loss 가중치 (ramp 적용)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--data", nargs="+", default=DEFAULT_DATA)
    ap.add_argument("--subsample", type=int, default=0,
                    help="균등 서브샘플 개수 (0=전체)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_iters", type=int, default=0,
                    help=">0 이면 그 step 수만 돌고 중단 (스모크용)")
    ap.add_argument("--batchsize", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--output_size", type=int, default=50)
    ap.add_argument("--imagesize", type=int, default=400)
    ap.add_argument("--lam_vec", type=float, default=0.5)
    ap.add_argument("--lam_diag", type=float, default=0.013)
    ap.add_argument("--lam_edge", type=float, default=0.0003)
    ap.add_argument("--warmup", type=int, default=200,
                    help="ramp warmup step 수 (ramp=min(1, step/warmup))")
    ap.add_argument("--struct_delta", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outf", required=True, help="체크포인트/로그 출력 폴더")
    ap.add_argument("--namefile", default="pvnet")
    ap.add_argument("--save_every", type=int, default=1,
                    help="N epoch 마다 ckpt 저장")
    ap.add_argument("--log_every", type=int, default=20,
                    help="N step 마다 raw loss + grad ratio 로깅")
    args = ap.parse_args()

    os.makedirs(args.outf, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    numVec = 0 if args.vec_mode == "off" else 18
    numSeg = 1 if args.seg else 0
    pvnet_vec = args.vec_mode != "off"
    pvnet_unit = args.vec_mode == "unit"
    # seg GT = pvnet_mask (박스투영 마스크). 로더는 pvnet_vec=True 일 때만 mask 를
    # 채우므로, vec_mode=off 인데 seg 만 켜는 경우에도 mask 가 필요하면 vec 로더를
    # 켠다(vec 헤드는 numVec=0 이라 안 만들어짐 → 동작 불변).
    need_mask = pvnet_vec or numSeg > 0
    loader_pvnet_vec = need_mask

    print(f"[cfg] vec_mode={args.vec_mode} numVec={numVec} numSeg={numSeg} "
          f"pvnet_vec={pvnet_vec} pvnet_unit={pvnet_unit}")
    print(f"[cfg] weights={args.weights}")
    print(f"[cfg] data={args.data}")
    print(f"[cfg] subsample={args.subsample} epochs={args.epochs} "
          f"max_iters={args.max_iters} bs={args.batchsize} lr={args.lr}")
    print(f"[cfg] lam_vec={args.lam_vec} lam_diag={args.lam_diag} "
          f"lam_edge={args.lam_edge} warmup={args.warmup}")

    # ── 데이터 ──
    ds = CleanVisiiDopeLoader(
        args.data, objects=["pallet"], sigma=args.sigma,
        output_size=args.output_size,
        pvnet_vec=loader_pvnet_vec, pvnet_unit=pvnet_unit,
        pvnet_mask_rle=args.pvnet_mask_rle)
    n_total = len(ds)
    if args.subsample > 0 and args.subsample < n_total:
        idx = np.linspace(0, n_total - 1, args.subsample).round().astype(int)
        idx = sorted(set(int(i) for i in idx))
        ds = torch.utils.data.Subset(ds, idx)
    print(f"[data] total={n_total} used={len(ds)}")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batchsize, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    # ── 모델 (strict=False: vec 헤드만 새 파라미터) ──
    net = DopeNetwork(numVec=numVec, numSeg=numSeg).to(device)
    state = torch.load(args.weights, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    vec_missing = [m for m in missing if m.startswith("m_vec")]
    seg_missing = [m for m in missing if m.startswith("m_seg")]
    print(f"[load] strict=False missing={len(missing)} "
          f"(vec-head {len(vec_missing)} seg-head {len(seg_missing)}) "
          f"unexpected={len(unexpected)}")
    net.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    soft_argmax = SpatialSoftArgmax2D(temperature=1.0)
    edge_mod = SparseEdgeLoss(delta=args.struct_delta).to(device)

    log_path = os.path.join(args.outf, f"train_log_{args.vec_mode}.txt")
    logf = open(log_path, "w")

    def logline(s):
        print(s)
        logf.write(s + "\n")
        logf.flush()

    logline(f"# finetune_pvnet vec_mode={args.vec_mode} "
            f"weights={args.weights}")
    logline(f"# lam_vec={args.lam_vec} lam_seg={args.lam_seg} "
            f"lam_diag={args.lam_diag} lam_edge={args.lam_edge} "
            f"warmup={args.warmup} numSeg={numSeg}")
    hdr = (f"{'step':>6}{'ep':>4}{'ramp':>7}{'L_total':>11}{'L_bel':>10}"
           f"{'L_aff':>10}{'L_vec':>10}{'L_seg':>10}{'L_diag':>10}{'L_edge':>10}"
           f"{'r_vec':>8}{'r_seg':>8}{'r_diag':>8}{'r_edge':>8}")
    logline(hdr)
    logline("-" * len(hdr))

    step = 0
    t0 = time.time()
    stop = False
    for epoch in range(args.epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            img = batch["img"].float().to(device)
            target_belief = batch["beliefs"].float().to(device)
            target_aff = batch["affinities"].float().to(device)

            out = net(img)
            beliefs, affinities, vec_stages, seg_stages = unpack_forward(
                net, out, numVec, numSeg)

            ramp = min(1.0, (step + 1) / max(1, args.warmup))

            L_bel = belief_loss_symmetric(beliefs, target_belief)
            L_aff = affinity_loss(affinities, target_aff)

            # 보조 loss
            L_vec = torch.tensor(0.0, device=device)
            if vec_stages is not None:
                gt_vec = batch["pvnet_vec"].float().to(device)
                gt_mask = batch["pvnet_mask"].float().to(device)
                L_vec = vec_loss(vec_stages, gt_vec, gt_mask)

            L_seg = torch.tensor(0.0, device=device)
            if seg_stages is not None:
                gt_mask_s = batch["pvnet_mask"].float().to(device)
                L_seg = seg_loss(seg_stages, gt_mask_s)

            pred_kp = soft_argmax(beliefs[-1][:, :9])
            L_diag = diag_centroid_loss(pred_kp, delta=args.struct_delta)

            with torch.no_grad():
                gt_kp = soft_argmax(target_belief[:, :9])
            L_edge, _ = edge_mod(pred_kp, gt_kp)

            loss = (L_bel + L_aff
                    + ramp * (args.lam_vec * L_vec
                              + args.lam_seg * L_seg
                              + args.lam_diag * L_diag
                              + args.lam_edge * L_edge))

            # ── NaN-abort 가드 (밤샘 안전) ──
            if not torch.isfinite(loss):
                msg = (f"[FATAL] step={step} non-finite loss={loss.item()} "
                       f"(L_bel={L_bel.item():.4g} L_aff={L_aff.item():.4g} "
                       f"L_vec={float(L_vec):.4g} L_seg={float(L_seg):.4g} "
                       f"L_diag={float(L_diag):.4g} L_edge={float(L_edge):.4g}) "
                       f"— abort")
                logline(msg)
                logf.close()
                sys.exit(2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # ── grad ratio 진단 (log_every 마다; 자체 forward, step 무영향) ──
            r_vec = r_diag = r_edge = r_seg = float("nan")
            if step % args.log_every == 0:
                _need = vec_stages is not None or seg_stages is not None
                _gv = batch["pvnet_vec"].float().to(device) \
                    if vec_stages is not None else None
                _gm = batch["pvnet_mask"].float().to(device) \
                    if _need else None
                r = grad_ratios(net, img, target_belief, target_aff,
                                _gv, _gm, soft_argmax, edge_mod,
                                args.struct_delta, numVec, numSeg)
                r_vec, r_diag, r_edge, r_seg = (
                    r["vec"], r["diag"], r["edge"], r["seg"])
                net.train()

            if step % args.log_every == 0:
                logline(f"{step:>6}{epoch:>4}{ramp:>7.3f}"
                        f"{loss.item():>11.5f}{L_bel.item():>10.5f}"
                        f"{L_aff.item():>10.5f}{float(L_vec):>10.5f}"
                        f"{float(L_seg):>10.5f}"
                        f"{float(L_diag):>10.5f}{float(L_edge):>10.5f}"
                        f"{r_vec:>8.3f}{r_seg:>8.3f}"
                        f"{r_diag:>8.3f}{r_edge:>8.3f}")

            step += 1
            if args.max_iters > 0 and step >= args.max_iters:
                logline(f"[smoke] reached max_iters={args.max_iters}, stop")
                stop = True
                break
        if stop:
            break

        # epoch ckpt
        if (epoch + 1) % args.save_every == 0:
            ck = os.path.join(
                args.outf, f"net_{args.namefile}_{args.vec_mode}_"
                           f"{str(epoch + 1).zfill(4)}.pth")
            torch.save(net.state_dict(), ck)
            logline(f"[ckpt] {ck}")

    # 항상 final ckpt 저장 (스모크에서도 검증 가능하게)
    final_ck = os.path.join(
        args.outf, f"final_net_{args.namefile}_{args.vec_mode}.pth")
    torch.save(net.state_dict(), final_ck)
    logline(f"[ckpt] {final_ck}")
    logline(f"[done] steps={step} elapsed={time.time() - t0:.1f}s "
            f"({(time.time() - t0) / max(1, step):.3f}s/step)")
    logf.close()
    # 메타 json (eval 스크립트가 numVec/vec_mode 알 수 있게)
    json.dump({"vec_mode": args.vec_mode, "numVec": numVec, "numSeg": numSeg,
               "pvnet_unit": pvnet_unit, "lam_seg": args.lam_seg,
               "weights": args.weights,
               "final_ckpt": final_ck, "steps": step},
              open(os.path.join(args.outf, f"meta_{args.vec_mode}.json"), "w"),
              indent=2)
    print(f"[save] log={log_path}")


if __name__ == "__main__":
    main()
