"""finetune_edgevec.py — STEP 7 sparse edge-vector (HybridPose식) finetune 하니스.

판별 질문: dense voting(STEP4/5)이 진 게 "픽셀 마스크(데이터) 문제"냐 "벡터 자체가
이 단일-팔레트 문제에 불필요"냐. sparse edge-vector 는 **픽셀 마스크 불필요**(꼭짓점→
꼭짓점 unit 방향만 예측)라 마스크 오염을 우회한다 → 깨끗한 판별.

표현 선택 (보고용 근거):
  - dense 필드(키포인트에서만 supervise) vs global-pool→FC. → **global-pool→FC** 채택.
    이유: (1) per-pixel target/mask 불필요(가장 mask-free), (2) refine 는 edge 당 단위
    방향 1개만 필요, (3) dense 필드의 "어느 픽셀이 supervise" 모호성 제거. 학습도 가볍다.
  - full 72(9×8) vs 구조 core. → **구조 core 16 edge**(CUBOID_EDGES 12 + CUBOID_DIAGONALS
    4) 채택. 72 는 대부분 기하적으로 redundant. 16 = 같은 면 인접변 + 깊이방향 대응 + 면
    대각선 = 박스 구조 핵심. (geo_loss_struct 의 기존 상수 재사용, 재정의 안 함.)

설계 원칙 (제약 준수):
  - train.py / utils_dataset.py / models.py / loss 정의를 의미적으로 변경하지 않는다.
    models.py 는 numEdgeVec=0 이면 byte-identical (flag-gated 헤드만 추가).
  - edge GT 는 **GT belief 의 soft-argmax 키포인트에서 계산**(JSON/loader 안 건드림).
    0-division 가드: ||kp_j - kp_i|| < eps 인 edge 는 무효(loss 제외).
  - belief/affinity loss = finetune_pvnet.py 의 train.py 복제본을 import (재정의 금지).
  - edge 보조 loss = ramp(min(1, step/warmup)) 로 켠다(random-init 헤드 워밍업 흡수).
  - NaN-abort 가드 + r_edgevec/L_bel 로깅 + strict=False + meta json.

control(--edge_head 없음) = heatmap 단독, 동일 스케줄. sparse(--edge_head) = heatmap + edge.
⚠ 풀 학습은 사용자가 직접 돌린다(스모크만 자동).
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
# 구조 edge 상수만 재사용(loss 재정의 금지). EDGE_WEIGHTS 는 edge 별 가중치.
from geo_loss_struct import CUBOID_EDGES, CUBOID_DIAGONALS  # noqa: E402
# belief/affinity loss 복제본 재사용(재정의 금지)
from finetune_pvnet import belief_loss_symmetric, affinity_loss  # noqa: E402

DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "challenge0123",
                               "final_net_epoch_0060.pth")
DEFAULT_DATA = [
    os.path.join(ROOT, "data", "pallet", "training_data", "mixed_v8_train"),
    os.path.join(ROOT, "challenge", "data", "training", "v1"),
    os.path.join(ROOT, "challenge", "data", "training", "v2"),
]

# 구조 core edge = 12 물리 edge + 4 면 대각선 = 16. (full 72 대신)
STRUCT_EDGES = list(CUBOID_EDGES) + list(CUBOID_DIAGONALS)
EPS_EDGE = 1e-3   # belief-grid(50px) 단위. ||kp_j-kp_i||<eps → 무효 edge.


def edge_dirs_from_kp(kp, edges, eps=EPS_EDGE):
    """kp (B,9,2) belief 격자 좌표 -> (unit_dirs (B,E,2), valid (B,E) bool).
    unit = (kp_j - kp_i)/||·||. 0-division 가드: ||·||<eps → 0벡터+invalid."""
    B = kp.shape[0]
    E = len(edges)
    dirs = torch.zeros(B, E, 2, device=kp.device, dtype=kp.dtype)
    valid = torch.zeros(B, E, device=kp.device, dtype=torch.bool)
    for e, (i, j) in enumerate(edges):
        d = kp[:, j] - kp[:, i]               # (B,2)
        n = torch.norm(d, dim=-1)             # (B,)
        ok = n > eps
        safe = n.clamp_min(eps).unsqueeze(-1)
        dirs[:, e] = torch.where(ok.unsqueeze(-1), d / safe,
                                 torch.zeros_like(d))
        valid[:, e] = ok
    return dirs, valid


def edge_vec_loss(pred_raw, gt_dirs, valid, edges):
    """pred_raw (B, 2E) FC 출력 -> pair 별 L2-normalize 한 단위방향 vs gt_dirs.
    loss = 유효 edge 에서 (1 - cos)  (방향만; 크기 무시). 무효 edge 제외."""
    B = pred_raw.shape[0]
    E = len(edges)
    pe = pred_raw.view(B, E, 2)
    pn = torch.norm(pe, dim=-1, keepdim=True).clamp_min(1e-6)
    pu = pe / pn                                # 예측 단위방향 (B,E,2)
    cos = (pu * gt_dirs).sum(dim=-1)           # (B,E)
    per_edge = 1.0 - cos                        # 0=정합, 2=반대
    m = valid.float()
    denom = m.sum().clamp_min(1.0)
    return (per_edge * m).sum() / denom


def vgg_grad_norm(net):
    sq = 0.0
    for p in net.vgg.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
    return float(np.sqrt(sq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_head", action="store_true",
                    help="sparse edge-vector 헤드 켬 (없으면 control=heatmap 단독)")
    ap.add_argument("--lam_edgevec", type=float, default=0.1)
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
    ap.add_argument("--warmup", type=int, default=2000,
                    help="ramp warmup step 수 (ramp=min(1, step/warmup))")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outf", required=True, help="체크포인트/로그 출력 폴더")
    ap.add_argument("--namefile", default="edgevec")
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outf, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = "sparse" if args.edge_head else "control"
    numEdgeVec = 2 * len(STRUCT_EDGES) if args.edge_head else 0

    print(f"[cfg] mode={mode} edge_head={args.edge_head} "
          f"numEdgeVec={numEdgeVec} n_edges={len(STRUCT_EDGES)}")
    print(f"[cfg] weights={args.weights}")
    print(f"[cfg] data={args.data}")
    print(f"[cfg] subsample={args.subsample} epochs={args.epochs} "
          f"max_iters={args.max_iters} bs={args.batchsize} lr={args.lr}")
    print(f"[cfg] lam_edgevec={args.lam_edgevec} warmup={args.warmup}")

    # ── 데이터 (loader 옵션 기본값 = vec/seg 미사용; JSON 안 건드림) ──
    ds = CleanVisiiDopeLoader(
        args.data, objects=["pallet"], sigma=args.sigma,
        output_size=args.output_size)
    n_total = len(ds)
    if args.subsample > 0 and args.subsample < n_total:
        idx = np.linspace(0, n_total - 1, args.subsample).round().astype(int)
        idx = sorted(set(int(i) for i in idx))
        ds = torch.utils.data.Subset(ds, idx)
    print(f"[data] total={n_total} used={len(ds)}")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batchsize, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    # ── 모델 (strict=False: edge 헤드만 새 파라미터) ──
    net = DopeNetwork(numEdgeVec=numEdgeVec).to(device)
    state = torch.load(args.weights, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    edge_missing = [m for m in missing if m.startswith("m_edge")]
    print(f"[load] strict=False missing={len(missing)} "
          f"(edge-head {len(edge_missing)}) unexpected={len(unexpected)}")
    net.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    soft_argmax = SpatialSoftArgmax2D(temperature=1.0)

    log_path = os.path.join(args.outf, f"train_log_{mode}.txt")
    logf = open(log_path, "w")

    def logline(s):
        print(s)
        logf.write(s + "\n")
        logf.flush()

    logline(f"# finetune_edgevec mode={mode} weights={args.weights}")
    logline(f"# lam_edgevec={args.lam_edgevec} warmup={args.warmup} "
            f"numEdgeVec={numEdgeVec} n_edges={len(STRUCT_EDGES)}")
    hdr = (f"{'step':>6}{'ep':>4}{'ramp':>7}{'L_total':>11}{'L_bel':>10}"
           f"{'L_aff':>10}{'L_edge':>10}{'r_edgevec':>11}{'valid_e':>9}")
    logline(hdr)
    logline("-" * len(hdr))

    def edge_grad_ratio(img, target_belief):
        """edge loss 의 공유 VGG grad norm / belief grad norm (진단; step 무영향)."""
        if numEdgeVec == 0:
            return float("nan")
        net.zero_grad(set_to_none=True)
        b, _, _ = net(img)
        L = belief_loss_symmetric(b, target_belief)
        L.backward()
        g_bel = vgg_grad_norm(net)
        net.zero_grad(set_to_none=True)
        _, _, edge_raw = net(img)
        with torch.no_grad():
            gt_kp = soft_argmax(target_belief[:, :9])
            gt_dirs, valid = edge_dirs_from_kp(gt_kp, STRUCT_EDGES)
        Le = edge_vec_loss(edge_raw, gt_dirs, valid, STRUCT_EDGES)
        Le.backward()
        g_edge = vgg_grad_norm(net)
        net.zero_grad(set_to_none=True)
        return g_edge / g_bel if g_bel > 0 else float("nan")

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
            if numEdgeVec > 0:
                beliefs, affinities, edge_raw = out
            else:
                beliefs, affinities = out
                edge_raw = None

            ramp = min(1.0, (step + 1) / max(1, args.warmup))

            L_bel = belief_loss_symmetric(beliefs, target_belief)
            L_aff = affinity_loss(affinities, target_aff)

            L_edge = torch.tensor(0.0, device=device)
            n_valid = 0
            if edge_raw is not None:
                with torch.no_grad():
                    gt_kp = soft_argmax(target_belief[:, :9])
                    gt_dirs, valid = edge_dirs_from_kp(gt_kp, STRUCT_EDGES)
                    n_valid = int(valid.sum().item())
                L_edge = edge_vec_loss(edge_raw, gt_dirs, valid, STRUCT_EDGES)

            loss = L_bel + L_aff + ramp * args.lam_edgevec * L_edge

            if not torch.isfinite(loss):
                msg = (f"[FATAL] step={step} non-finite loss={loss.item()} "
                       f"(L_bel={L_bel.item():.4g} L_aff={L_aff.item():.4g} "
                       f"L_edge={float(L_edge):.4g}) — abort")
                logline(msg)
                logf.close()
                sys.exit(2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            r_edgevec = float("nan")
            if step % args.log_every == 0:
                r_edgevec = edge_grad_ratio(img, target_belief)
                net.train()
                logline(f"{step:>6}{epoch:>4}{ramp:>7.3f}"
                        f"{loss.item():>11.5f}{L_bel.item():>10.5f}"
                        f"{L_aff.item():>10.5f}{float(L_edge):>10.5f}"
                        f"{r_edgevec:>11.4f}{n_valid:>9}")

            step += 1
            if args.max_iters > 0 and step >= args.max_iters:
                logline(f"[smoke] reached max_iters={args.max_iters}, stop")
                stop = True
                break
        if stop:
            break

        if (epoch + 1) % args.save_every == 0:
            ck = os.path.join(
                args.outf, f"net_{args.namefile}_{mode}_"
                           f"{str(epoch + 1).zfill(4)}.pth")
            torch.save(net.state_dict(), ck)
            logline(f"[ckpt] {ck}")

    final_ck = os.path.join(
        args.outf, f"final_net_{args.namefile}_{mode}.pth")
    torch.save(net.state_dict(), final_ck)
    logline(f"[ckpt] {final_ck}")
    logline(f"[done] steps={step} elapsed={time.time() - t0:.1f}s "
            f"({(time.time() - t0) / max(1, step):.3f}s/step)")
    logf.close()
    json.dump({"mode": mode, "edge_head": args.edge_head,
               "numEdgeVec": numEdgeVec, "n_edges": len(STRUCT_EDGES),
               "edges": STRUCT_EDGES, "lam_edgevec": args.lam_edgevec,
               "weights": args.weights, "final_ckpt": final_ck, "steps": step},
              open(os.path.join(args.outf, f"meta_{mode}.json"), "w"),
              indent=2)
    print(f"[save] log={log_path}")


if __name__ == "__main__":
    main()
