#!/usr/bin/env python
"""diffpnp3d_selftest.py — PAPER_S2 DiffPnP3D 자체 테스트 (P2 Q0 부품 검증).

pnp_valid_3d index 에서 valid∧V8 프레임 32개(+ gating 확인용 aug_squash 1개)를 로드,
GT projected_cuboid 에 sigma2 가우시안을 심은 가짜 belief → LocalSoftArgmax2D →
DiffPnP3DLoss → backward 로 다음을 verify:
  (1) L finite, GT-근사 belief 면 작은 값
  (2) belief 입력 grad + pred_xy grad 도달
  (3) pred_xy 를 흔들면 loss 증가 (민감도)
  (4) positive-depth 통과, NaN 없음
  (5) pnp grad-norm vs heatmap(belief MSE) grad-norm 비율
  (6) aug_squash(pnp_valid_3d=false) → gating 으로 loss 0
"""
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "train"))
from diffpnp3d_loss import (  # noqa: E402
    LocalSoftArgmax2D, DiffPnP3DLoss, build_diffpnp3d_targets)

IDX_DIR = os.path.join(
    ROOT, "data/pallet/results/paper_s2_scratch_diffpnp/pnp_valid_3d_index")
DATA_ROOT = os.path.join(ROOT, "data/pallet/training_data")
BEL = 50           # belief map size
ORIG = (640, 480)  # projected_cuboid / K pixel space
SIGMA = 2.0        # belief gaussian sigma (spec: sigma 2)


def make_belief(pc_orig, sigma=SIGMA, size=BEL, orig=ORIG, jitter=None):
    """pc_orig (9,2) orig px → belief (9,size,size). belief↔orig anisotropic scale.
    jitter (9,2) orig px 를 더하면 peak 를 흔든 belief 생성(grad-ratio/민감도용)."""
    sx, sy = size / orig[0], size / orig[1]
    pc = pc_orig.copy()
    if jitter is not None:
        pc = pc + jitter
    cx = pc[:, 0] * sx
    cy = pc[:, 1] * sy
    ys, xs = np.mgrid[0:size, 0:size]
    b = np.zeros((pc.shape[0], size, size), np.float32)
    for k in range(pc.shape[0]):
        b[k] = np.exp(-((xs - cx[k]) ** 2 + (ys - cy[k]) ** 2) / (2 * sigma ** 2))
    return b


def load_pool(name, want, need_valid):
    idx = json.load(open(os.path.join(IDX_DIR, f"{name}.json")))
    out = []
    for rel, e in idx.items():
        if need_valid and not (e["pnp_valid_3d"] and e["V8"]):
            continue
        if (not need_valid) and e["pnp_valid_3d"]:
            continue
        out.append((rel, e))
        if len(out) >= want:
            break
    return out


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- valid∧V8 32개 (dims 다양성 위해 mixed_v8 + v4 + paper_4pallet 혼합) ---
    picks = (load_pool("mixed_v8_train", 12, True)
             + load_pool("v4_split_base", 10, True)
             + load_pool("paper_4pallet_mask_v1", 10, True))
    gate = load_pool("aug_squash_v2", 1, False)   # pnp_valid_3d=false (gating)

    Xs, Ks, Rs, ts, diags, masks, pcs = [], [], [], [], [], [], []
    ds_of = lambda rel: rel.split("/")[0]
    all_items = [(r, e, True) for r, e in picks] + [(r, e, False) for r, e in gate]
    dims_seen = []
    for rel, e, is_valid in all_items:
        p = os.path.join(DATA_ROOT, rel)
        tg = build_diffpnp3d_targets(p, e)
        Xs.append(tg["X_i"]); Ks.append(tg["K"]); Rs.append(tg["R_gt"])
        ts.append(tg["t_gt"]); diags.append(tg["diag"]); pcs.append(tg["projected_cuboid"])
        masks.append(bool(tg["pnp_valid"] and tg["V8"]))
        dims_seen.append(e["dims"])

    B = len(all_items)
    to_t = lambda a: torch.tensor(np.array(a), dtype=torch.float32, device=dev)
    X = to_t(Xs)                       # (B,8,3)
    K = to_t(Ks)                       # (B,3,3)
    R_gt = to_t(Rs)                    # (B,3,3)
    t_gt = to_t(ts)                    # (B,3)
    diag = to_t(diags)                 # (B,)
    mask = torch.tensor(masks, device=dev)

    # projected_cuboid 8 corner (belief gen: 9점 심되 loss 는 8 corner) ------------
    pc9 = []
    for rel, e, _ in all_items:
        d = json.load(open(os.path.join(DATA_ROOT, rel)))
        o = d["objects"][0]
        arr = np.array(o["projected_cuboid"], np.float64)  # (9,2) or (8,2)
        if arr.shape[0] == 8:
            arr = np.vstack([arr, arr.mean(0)])
        pc9.append(arr[:9])
    pc9 = np.stack(pc9)                # (B,9,2)

    sa = LocalSoftArgmax2D(window=7, temperature=0.1,
                           orig_size=ORIG, belief_size=(BEL, BEL)).to(dev)
    loss_fn = DiffPnP3DLoss(n_gn=4, huber_delta=0.05).to(dev)

    # ================= (A) GT-근사 belief (jitter 없음) =================
    belief = torch.tensor(np.stack([make_belief(pc9[b]) for b in range(B)]),
                          device=dev, requires_grad=True)
    coords, conf = sa(belief)                       # (B,9,2) orig px
    coords8 = coords[:, :8]
    coords8.retain_grad()
    L, info = loss_fn(coords8, X, K, R_gt, t_gt, diag, mask)
    L.backward()

    softargmax_err = (coords[:, :8].detach() -
                      torch.tensor(pc9[:, :8], dtype=torch.float32, device=dev)
                      ).norm(dim=2)
    sa_err_valid = softargmax_err[mask].mean().item()

    belief_grad_norm = belief.grad.norm().item()
    predxy_grad_norm = coords8.grad.norm().item()

    # ================= (B) 민감도: pred_xy 를 직접 흔든다 =================
    with torch.no_grad():
        base_coords8 = coords8.detach().clone()
    sens = []
    for shift in [0.0, 2.0, 5.0, 10.0]:
        c = base_coords8.clone()
        # 4 rear corner(4~7) 에 +shift px (x 방향)
        c[:, 4:8, 0] = c[:, 4:8, 0] + shift
        Ls, _ = loss_fn(c, X, K, R_gt, t_gt, diag, mask)
        sens.append((shift, Ls.item()))

    # ================= (C) grad-norm 비율 (jitter 3px regime) =================
    jit = np.zeros((B, 9, 2), np.float32); jit[:, :, 0] = 3.0   # +3px x
    belief_j = torch.tensor(np.stack([make_belief(pc9[b], jitter=jit[b]) for b in range(B)]),
                            device=dev, requires_grad=True)
    coords_j, _ = sa(belief_j)
    Lp, _ = loss_fn(coords_j[:, :8], X, K, R_gt, t_gt, diag, mask)
    gp = torch.autograd.grad(Lp, belief_j, retain_graph=False)[0]
    pnp_gn = gp.norm().item()

    # heatmap MSE: target = GT-peak belief(무 jitter), pred = 흔든 belief
    belief_j2 = torch.tensor(np.stack([make_belief(pc9[b], jitter=jit[b]) for b in range(B)]),
                             device=dev, requires_grad=True)
    tgt = torch.tensor(np.stack([make_belief(pc9[b]) for b in range(B)]), device=dev)
    Lh = torch.nn.functional.mse_loss(belief_j2, tgt, reduction="mean")
    gh = torch.autograd.grad(Lh, belief_j2)[0]
    hm_gn = gh.norm().item()
    ratio = 100.0 * pnp_gn / (hm_gn + 1e-12)

    # ================= (D) gating: aug_squash per-frame L =================
    gate_L = info["per_frame_L"][~mask].abs().max().item() if (~mask).any() else 0.0

    # depth/NaN
    L_finite = bool(torch.isfinite(L).item())
    no_nan = bool(torch.isfinite(info["per_frame_L"]).all().item())

    # ---------------- 보고 (공백정렬) ----------------
    print("=" * 72)
    print("DiffPnP3D 자체 테스트 결과")
    print("=" * 72)
    print(f"frames: valid&V8={int(mask.sum())}  gated(aug_squash)={int((~mask).sum())}  total={B}")
    ws = sorted(set(round(d['W'],2) for d in dims_seen[:32]))
    ds = sorted(set(round(d['D'],2) for d in dims_seen[:32]))
    print(f"dims diversity  W(unique)={ws}")
    print(f"                D(unique)={ds}")
    print()
    print("check                          value            판정")
    print("-" * 72)
    print(f"(1) L_pnp3d finite             {L.item():.6e}   {'PASS' if L_finite else 'FAIL'}")
    print(f"    GT-근사 belief L 작은가     {L.item():.6e}   {'PASS(<1e-2)' if L.item()<1e-2 else 'CHECK'}")
    print(f"    softargmax err(px, valid)  {sa_err_valid:.3f}          {'PASS(<1px)' if sa_err_valid<1 else 'CHECK'}")
    print(f"(2) belief.grad norm           {belief_grad_norm:.4e}   {'PASS(>0)' if belief_grad_norm>0 else 'FAIL'}")
    print(f"    pred_xy.grad norm          {predxy_grad_norm:.4e}   {'PASS(>0)' if predxy_grad_norm>0 else 'FAIL'}")
    gate_ok = info['n_valid'] == int(mask.sum())
    print(f"(4) valid frames = mask.sum     {info['n_valid']}/{int(mask.sum())} (batch {B})   {'PASS' if gate_ok else 'FAIL'}")
    print(f"    skip depth/nan/cond        {info['skip_depth']}/{info['skip_nan']}/{info['skip_cond']}            {'PASS' if info['skip_depth']+info['skip_nan']+info['skip_cond']==0 else 'CHECK'}")
    print(f"    per_frame_L no NaN         {no_nan}           {'PASS' if no_nan else 'FAIL'}")
    print(f"    cond_max                   {info['cond_max']:.3e}")
    print(f"(6) gated aug_squash per-fr L  {gate_L:.6e}   {'PASS(=0)' if gate_L==0 else 'FAIL'}")
    print()
    print("(3) 민감도 (rear 4~7 +x px shift → L):")
    print("    shift_px      L_pnp3d")
    print("    " + "-" * 28)
    for s, lv in sens:
        print(f"    {s:5.1f}        {lv:.6e}")
    mono = all(sens[i][1] <= sens[i+1][1] + 1e-9 for i in range(len(sens)-1))
    print(f"    monotonic 증가: {'PASS' if mono else 'FAIL'}")
    print()
    print("(5) grad-norm 비율 (jitter 3px regime, λ=1 raw):")
    print(f"    pnp grad-norm     = {pnp_gn:.4e}")
    print(f"    heatmap grad-norm = {hm_gn:.4e}")
    print(f"    ratio pnp/heatmap (λ=1) = {ratio:.1f}%")
    lam_lo, lam_hi = 5.0 / (ratio + 1e-9), 30.0 / (ratio + 1e-9)
    print(f"    → 목표 5~30% 를 만드는 λ_pnp ∈ [{lam_lo:.4f}, {lam_hi:.4f}]")
    print(f"      (PLAN λ 후보 0.001~0.01 → 이 구간과 겹침: {'PASS' if lam_lo<=0.01 and lam_hi>=0.001 else 'CHECK'})")
    print()

    # ================= (7) guard 발화 검증 (주입) =================
    print("(7) guard 발화 (주입 케이스):")
    # 7a NaN guard: valid 프레임 하나 pred_xy 를 NaN 으로
    c_nan = base_coords8.clone(); c_nan[0] = float("nan")
    Ln, In = loss_fn(c_nan, X, K, R_gt, t_gt, diag, mask)
    # NaN pred_xy 는 depth check((z>0).all() on NaN=False) 에서도 걸려 depth 버킷으로
    # 분류될 수 있음 → 기능적 기준: 해당 프레임 제외 + 나머지 loss finite (poison 없음)
    excluded = (In["skip_nan"] + In["skip_depth"]) >= 1
    ok_nan = excluded and torch.isfinite(Ln).item() and \
        (In["per_frame_L"][0].item() == 0.0)
    print(f"    NaN 주입 → skip_nan={In['skip_nan']} skip_depth={In['skip_depth']} "
          f"(frame0 제외), L finite={bool(torch.isfinite(Ln).item())}, "
          f"per_frame_L[0]={In['per_frame_L'][0].item():.1e}  {'PASS' if ok_nan else 'FAIL'}")
    # 7b positive-depth guard: object 를 카메라 뒤로(z<0), pred_xy 를 그 pose 와
    # self-consistent 하게 주면 GN 이 안 움직여 P_pred z<0 유지 → depth guard 발화
    from diffpnp3d_loss import _project_batch
    t_bad = t_gt.clone(); t_bad[:, 2] = -t_bad[:, 2].abs()
    rvec_bad = DiffPnP3DLoss.rvec_from_R(R_gt)
    with torch.no_grad():
        uv_bad, _ = _project_batch(rvec_bad, t_bad, X, K)   # (B,8,2)
    Ld, Id = loss_fn(uv_bad, X, K, R_gt, t_bad, diag, mask)
    ok_depth = (Id["skip_depth"] >= 1) and torch.isfinite(Ld).item()
    print(f"    z<0(뒤) 주입 → skip_depth={Id['skip_depth']}, L finite={bool(torch.isfinite(Ld).item())}"
          f"  {'PASS' if ok_depth else 'FAIL'}")
    # 7c condition guard: cond_max 를 비현실적으로 낮게 → 전부 skip, crash 없음
    lf_strict = DiffPnP3DLoss(n_gn=4, huber_delta=0.05, cond_max=1.0).to(dev)
    Lc, Ic = lf_strict(base_coords8, X, K, R_gt, t_gt, diag, mask)
    ok_cond = (Ic["skip_cond"] >= 1) and torch.isfinite(Lc).item()
    print(f"    cond_max=1.0 → skip_cond={Ic['skip_cond']}, L={Lc.item():.1e} finite  "
          f"{'PASS' if ok_cond else 'FAIL'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
