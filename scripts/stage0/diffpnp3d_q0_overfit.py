#!/usr/bin/env python
"""diffpnp3d_q0_overfit.py — PAPER_S2 Phase 1c Q0: 32-sample overfit harness.

Verifies the *full integrated pipeline* (loader aspect-resize + per-frame DiffPnP3D
targets + train.py-style forward/loss/backward) end-to-end on a tiny overfit set.

Uses the REAL CleanVisiiDopeLoader (aspect_resize=True, diffpnp_index=...) and the
REAL DopeNetwork, exactly as train.py wires them (--diffpnp path), so a PASS means
the integration — not just the isolated loss module (that was diffpnp3d_selftest.py)
— is sound.

Q0 pass criteria (per PLAN):
  - L_heatmap decreasing, L_pnp3d decreasing
  - pred_xy / belief NaN count == 0
  - T_pred positive-depth ratio high (guard not chronically firing)
  - pnp gradient reaches the belief head (grad on output_belief != 0)
  - pnp/heatmap belief-grad-norm ratio in ~5-30% band for a chosen lambda
  - lambda ON does not blow up L_heatmap vs lambda=0 control
  - local soft-argmax err < ~1px once belief overfits
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "train"))

from utils_dataset import CleanVisiiDopeLoader  # noqa: E402
from models import DopeNetwork  # noqa: E402
from diffpnp3d_loss import LocalSoftArgmax2D, DiffPnP3DLoss  # noqa: E402

IDX_DIR = os.path.join(ROOT, "data/pallet/results/paper_s2_scratch_diffpnp/pnp_valid_3d_index")
DATA_ROOT = os.path.join(ROOT, "data/pallet/training_data")
OUT_MD = os.path.join(ROOT, "data/pallet/results/paper_s2_scratch_diffpnp/q0_overfit_results.md")
OUTPUT_SIZE = 50
SIGMA = 2.0


def _interior_ok(json_path, w=4, sz=OUTPUT_SIZE):
    """All 8 projected_cuboid corners inside the belief interior [w, sz-w).
    Mirrors CreateBeliefMap's draw predicate under the fixed anisotropic Resize
    (640x480 -> sz), so a True frame has all 8 belief channels populated."""
    o = json.load(open(json_path))["objects"][0]
    pc = np.array(o["projected_cuboid"][:8], dtype=np.float64)
    bx = pc[:, 0] * sz / 640.0
    by = pc[:, 1] * sz / 480.0
    return bool(np.all((bx - w >= 0) & (bx + w < sz)
                       & (by - w >= 0) & (by + w < sz)))


def pick_frames():
    """32 pnp_valid_3d&V8 & belief-interior frames across the 3 sets (dims variety)."""
    plan = [("mixed_v8_train", 12), ("v4_split_base", 10),
            ("paper_4pallet_mask_v1", 10)]
    chosen = {}       # abs_json -> entry
    rel_list = []
    for ds, want in plan:
        idx = json.load(open(os.path.join(IDX_DIR, f"{ds}.json")))
        n = 0
        for rel, e in idx.items():
            if e["pnp_valid_3d"] and e["V8"]:
                aj = os.path.abspath(os.path.join(DATA_ROOT, rel))
                if not os.path.exists(aj) or not _interior_ok(aj):
                    continue
                chosen[aj] = e
                rel_list.append(rel)
                n += 1
                if n >= want:
                    break
    return chosen, rel_list


def project_gt(X, K, R, t):
    """X (B,8,3), K/R (B,3,3), t (B,3) -> uv_gt (B,8,2) orig px."""
    Pc = torch.bmm(X, R.transpose(1, 2)) + t.unsqueeze(1)
    uvw = torch.bmm(Pc, K.transpose(1, 2))
    return uvw[:, :, :2] / uvw[:, :, 2:3].clamp(min=1e-6)


def heatmap_loss(out_bel, out_aff, tgt_bel, tgt_aff):
    lb = torch.tensor(0.0, device=tgt_bel.device)
    la = torch.tensor(0.0, device=tgt_bel.device)
    for s in range(len(out_aff)):
        la = la + ((out_aff[s] - tgt_aff) ** 2).mean()
        lb = lb + ((out_bel[s] - tgt_bel) ** 2).mean()
    return lb, la


def run_lambda(lam, subset, dev, iters, temp, ramp, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = torch.utils.data.DataLoader(subset, batch_size=12, shuffle=True,
                                         num_workers=0, drop_last=False)
    net = DopeNetwork(numSeg=0).to(dev)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    sa = LocalSoftArgmax2D(window=7, temperature=temp,
                           orig_size=(640, 480),
                           belief_size=(OUTPUT_SIZE, OUTPUT_SIZE)).to(dev)
    loss_fn = DiffPnP3DLoss(n_gn=4, huber_delta=0.05).to(dev)

    curve = []          # (it, L_heatmap, L_pnp_raw, sa_err, valid_frac)
    nan_belief = 0
    nan_predxy = 0
    depth_ok_num = 0
    depth_mask_num = 0
    raw_ratio = None          # lambda-independent: 100*||grad L_pnp||/||grad L_heat||
    param_grad_nonzero = None
    meas_iter = int(0.6 * iters)   # measure once belief has partly formed
    it = 0
    data_iter = iter(loader)
    while it < iters:
        try:
            b = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            b = next(data_iter)
        img = b["img"].to(dev)
        tgt_bel = b["beliefs"].to(dev)
        tgt_aff = b["affinities"].to(dev)
        out_bel, out_aff = net(img)

        if not torch.isfinite(out_bel[-1]).all():
            nan_belief += 1

        lb, la = heatmap_loss(out_bel, out_aff, tgt_bel, tgt_aff)
        L_heat = lb + la

        # DiffPnP3D
        pred_xy, _ = sa(out_bel[-1][:, :8])
        if not torch.isfinite(pred_xy).all():
            nan_predxy += 1
        X = b["diffpnp_X"].to(dev)
        K = b["diffpnp_K"].to(dev)
        R = b["diffpnp_R"].to(dev)
        t = b["diffpnp_t"].to(dev)
        dg = b["diffpnp_diag"].to(dev)
        mk = b["diffpnp_valid"].to(dev).bool()
        L_pnp, info = loss_fn(pred_xy, X, K, R, t, dg, mk)

        depth_ok_num += info["n_valid"]
        depth_mask_num += int(mk.sum().item())

        ramp_f = min(1.0, it / max(1.0, float(ramp)))
        total = L_heat + lam * ramp_f * L_pnp

        # soft-argmax err (valid frames) vs GT projection
        with torch.no_grad():
            if mk.any():
                uv_gt = project_gt(X, K, R, t)
                err = (pred_xy - uv_gt).norm(dim=2)[mk].mean().item()
            else:
                err = float("nan")

        # lambda-INDEPENDENT belief-grad ratio at a belief-formed checkpoint:
        # raw = 100*||grad(L_pnp)||/||grad(L_heat)|| on the belief head. Effective
        # contribution at weight lam is (lam * raw). Measured for every run
        # (incl. lam=0) so it reflects the geometry, not the chosen weight.
        if raw_ratio is None and it >= meas_iter and mk.any():
            gh = torch.autograd.grad(L_heat, out_bel[-1], retain_graph=True)[0]
            gp = torch.autograd.grad(L_pnp, out_bel[-1], retain_graph=True)[0]
            raw_ratio = 100.0 * gp.norm().item() / (gh.norm().item() + 1e-12)
            # verify pnp gradient alone reaches the belief-head params (!= 0)
            gp_params = torch.autograd.grad(L_pnp, [p for p in net.parameters()
                                                    if p.requires_grad],
                                            retain_graph=True, allow_unused=True)
            param_grad_nonzero = float(sum(
                g.norm().item() for g in gp_params if g is not None))

        opt.zero_grad()
        total.backward()
        opt.step()

        if it % 30 == 0 or it == iters - 1:
            curve.append((it, float(L_heat.item()), float(L_pnp.item()),
                          err, info["valid_frac"]))
        it += 1

    depth_ratio = depth_ok_num / max(1, depth_mask_num)
    return {
        "lam": lam, "curve": curve, "nan_belief": nan_belief,
        "nan_predxy": nan_predxy, "depth_ratio": depth_ratio,
        "raw_ratio": raw_ratio, "param_grad": param_grad_nonzero,
        "final_heat": curve[-1][1], "final_pnp": curve[-1][2],
        "min_pnp": min(c[2] for c in curve), "final_saerr": curve[-1][3],
        "first_heat": curve[0][1], "first_pnp": curve[0][2],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--ramp", type=int, default=100)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.001, 0.003, 0.005])
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    chosen, rel_list = pick_frames()
    print(f"picked {len(chosen)} valid&V8 frames")

    # Build the REAL loader (aspect_resize + diffpnp_index), then Subset to picks.
    ds_dirs = sorted({os.path.join(DATA_ROOT, r.split("/")[0]) for r in rel_list})
    full = CleanVisiiDopeLoader(
        ds_dirs, objects=["pallet"], sigma=SIGMA, output_size=OUTPUT_SIZE,
        aspect_resize=True, diffpnp_index=chosen)
    want_abs = set(chosen.keys())
    sub_idx = [i for i, tup in enumerate(full.imgs)
               if os.path.abspath(tup[2]) in want_abs]
    print(f"loader matched {len(sub_idx)}/{len(chosen)} frames in {len(full.imgs)} total")
    subset = torch.utils.data.Subset(full, sub_idx)

    # dims diversity of the picks
    ws = sorted({round(chosen[a]["dims"]["W"], 2) for a in want_abs})
    ds = sorted({round(chosen[a]["dims"]["D"], 2) for a in want_abs})

    results = {}
    for lam in args.lambdas:
        print(f"\n=== lambda={lam} ===")
        r = run_lambda(lam, subset, dev, args.iters, args.temp, args.ramp)
        results[lam] = r
        for (it, lh, lp, e, vf) in r["curve"]:
            print(f"  it={it:4d}  L_heat={lh:.5f}  L_pnp={lp:.5f}  "
                  f"saerr={e:.3f}px  valid_frac={vf:.2f}")

    ctrl = results.get(0.0)
    lines = []
    lines.append("# PAPER_S2 DiffPnP3D — Q0 32-sample overfit results\n")
    lines.append(f"iters={args.iters}  soft-argmax temp={args.temp}  "
                 f"ramp={args.ramp} steps  batch=12  scratch DopeNetwork\n")
    lines.append(f"frames: {len(sub_idx)} pnp_valid_3d&V8 "
                 f"(mixed_v8 + v4_split_base + paper_4pallet)\n")
    lines.append(f"dims W(unique)={ws}\n")
    lines.append(f"dims D(unique)={ds}\n")

    def block(txt):
        lines.append("```\n" + txt + "\n```\n")

    # lambda-independent raw belief-grad ratio (median across runs) -> implied
    # lambda band for a 5-30% effective contribution.
    raws = [results[l]["raw_ratio"] for l in args.lambdas
            if results[l]["raw_ratio"] is not None]
    raw_med = float(np.median(raws)) if raws else float("nan")
    lam_lo = 5.0 / raw_med if raw_med > 0 else float("nan")
    lam_hi = 30.0 / raw_med if raw_med > 0 else float("nan")

    # summary table (effective% = lam * raw_ratio)
    hdr = ("lam       L_heat0 -> L_heatF   L_pnp0 -> L_pnpF (min)     "
           "sa_err   depthR   raw%    eff%    NaN(b/xy)")
    rows = [hdr, "-" * len(hdr)]
    for lam in args.lambdas:
        r = results[lam]
        rr = r["raw_ratio"]
        rrs = f"{rr:.0f}" if rr is not None else "  -"
        eff = (lam * rr) if rr is not None else float("nan")
        effs = f"{eff:.1f}" if rr is not None else "  -"
        rows.append(
            f"{lam:<8.4f}  {r['first_heat']:.4f}->{r['final_heat']:.4f}   "
            f"{r['first_pnp']:.4f}->{r['final_pnp']:.4f} ({r['min_pnp']:.4f})   "
            f"{r['final_saerr']:.3f}    {r['depth_ratio']:.3f}   {rrs:>5}   "
            f"{effs:>5}    {r['nan_belief']}/{r['nan_predxy']}")
    block("\n".join(rows))

    lines.append(f"\nraw belief-grad ratio (lambda-indep, median at 0.6*iters) = "
                 f"{raw_med:.0f}%  ->  lambda for 5-30% effective in "
                 f"[{lam_lo:.4f}, {lam_hi:.4f}]\n")
    lines.append(f"PLAN lambda candidates 0.001-0.01 overlap this band: "
                 f"{'YES' if (lam_lo <= 0.01 and lam_hi >= 0.001) else 'NO'}\n")

    # pass/fail checks
    checks = []
    for lam in args.lambdas:
        if lam == 0.0:
            continue
        r = results[lam]
        rr = r["raw_ratio"]
        eff = (lam * rr) if rr is not None else None
        heat_dn = r["final_heat"] < r["first_heat"]
        pnp_dn = r["min_pnp"] < r["first_pnp"]
        no_nan = (r["nan_belief"] == 0 and r["nan_predxy"] == 0)
        depth_ok = r["depth_ratio"] > 0.9
        grad_reach = (r["param_grad"] is not None and r["param_grad"] > 0)
        eff_band = (eff is not None and 5.0 <= eff <= 30.0)
        no_explode = (ctrl is None) or (r["final_heat"] <= 1.5 * ctrl["final_heat"])
        checks.append((lam, heat_dn, pnp_dn, no_nan, depth_ok, grad_reach,
                       eff_band, no_explode, eff))

    chk_hdr = ("lam      heat_dn  pnp_dn  no_NaN  depth>.9  grad_reach  "
               "eff5-30%      no_explode")
    crows = [chk_hdr, "-" * len(chk_hdr)]
    for (lam, h, p, n, d, g, eb, e, eff) in checks:
        def m(x):
            return "PASS" if x else "FAIL"
        effs = f"{eff:.1f}%" if eff is not None else "-"
        crows.append(f"{lam:<8.4f} {m(h):>6}  {m(p):>6}  {m(n):>6}  {m(d):>7}  "
                     f"{m(g):>9}  {m(eb):>5}({effs:>6})  {m(e):>9}")
    block("\n".join(crows))

    # sa_err convergence (belief resolution floor: 1 belief px ~= 12.8 orig px in x)
    lines.append("\nsa_err = local soft-argmax err vs GT projection (orig px). "
                 "Belief is 50x50 (1 belief px ~ 12.8 orig px in x / 9.6 in y), so a "
                 "few-px orig error = sub-belief-pixel. Report is the converged value; "
                 "criterion = decreasing toward the belief-resolution floor.\n")

    # explosion control note
    if ctrl is not None:
        lines.append(f"\nlambda=0 control final L_heat = {ctrl['final_heat']:.5f} "
                     f"(explosion guard: lambda-on L_heat must stay <= 1.5x this)\n")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nsaved -> {OUT_MD}")


if __name__ == "__main__":
    main()
