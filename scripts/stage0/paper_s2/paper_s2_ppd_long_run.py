#!/usr/bin/env python3
"""PPD long-run: 3k/1k training, untouched test, real N87 one-shot, CGR.

Progression is gated on H3 (candidate polarity) because that is the direct
objective: the learned 5-class map only has to pick the upright candidate out
of a *fixed* oracle SAI-U set.  H1 (mask) and H2 (pixel line) keep their
historical FAIL verdicts and are reported as diagnostics — no threshold was
lowered and no loss was changed to make them pass.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import argparse, json, math, os, pathlib, sys, time, hashlib
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
for p in ("Deep_Object_Pose/common", "challenge/scripts", "scripts/stage0",
          "scripts/data_prep/eval", "Deep_Object_Pose/train"):
    sys.path.insert(0, str(ROOT / p))
import pallet_graph_geometry as PG, pallet_polarity_disambiguation as PPD
import semantic_axis_initialization as SAI, dimension_guided_graph_pose as DGP
import polarity_aware_line_head as PLH
from models import DopeNetwork

D = ROOT / "data/pallet/results/paper_s2_palletgraph_line_screen"
DATA = ROOT / "data/pallet/training_data/paper_4pallet_mask_v1"
WROOT = ROOT / "weights/paper_s2/paper_s2_ppd_t2_screen"
ALLOWED_ROOT = "paper_4pallet_mask_v1"
BANNED_ROOTS = ("mixed_v8_train", "v4_split_base", "aug_squash_v2",
                "aug_trunc_v2", "aug_scale_v2")
SEED, BATCH, LR, WD, EPOCHS, GRID = 1, 8, 3e-4, 1e-4, 20, PLH.TARGET_GRID
ARMS = ("L0", "M0", "M1")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(m): print(m, flush=True)
def seed_all(s):
    import random; random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def guard_root(path):
    text = str(path)
    if ALLOWED_ROOT not in text:
        raise RuntimeError(f"BLOCKED: training root violation: {text}")
    for banned in BANNED_ROOTS:
        if banned in text:
            raise RuntimeError(f"BLOCKED: banned root {banned}")
    return path


def support_maps(R, t, K, dims, size, keep):
    """Original-resolution T2 support; the 100x100 resize path is forbidden."""
    edges = PPD.polarity_edge_classes(dims)
    vis = PG.visible_edges(R, t, dims)
    proj, dep = PG.project_points(PG.make_corners(*dims)[:8], R, t, K)
    out = {c: np.zeros((size[1], size[0]), np.uint8) for c in PLH.CLASS_ORDER}
    for (i, j), cls in edges:
        if dep[i] <= 1e-6 or dep[j] <= 1e-6 or not vis[(i, j)]:
            continue
        cl = PG.clip_segment_to_image(proj[i], proj[j], size[0], size[1])
        if cl is None:
            continue
        for q in PG.sample_along(cl[0], cl[1], pixels_per_sample=1.0):
            x, y = int(round(q[0])), int(round(q[1]))
            if 0 <= x < size[0] and 0 <= y < size[1] and keep[y, x]:
                out[cls][y, x] = 1
    return out


def load_frame(fn, with_candidates=False):
    guard_root(DATA / fn)
    d = json.load(open(DATA / fn)); o = d["objects"][0]; c = d["camera_data"]
    K = np.array([[c["intrinsics"]["fx"], 0, c["intrinsics"]["cx"]],
                  [0, c["intrinsics"]["fy"], c["intrinsics"]["cy"]], [0, 0, 1.]])
    dims = (o["dimensions_m"]["width"], o["dimensions_m"]["depth"], o["dimensions_m"]["height"])
    T = np.asarray(o["pose_transform"], float); R, t = T[:3, :3], T[:3, 3]
    img = cv2.imread(str(DATA / fn.replace(".json", ".png"))); size = (c["width"], c["height"])
    mask = PLH.decode_mask_rle(o["mask_rle"], (c["height"], c["width"]))
    tg, _ = PLH.build_polarity_targets_v2(R, t, K, dims, size, "observed_fragment", image_bgr=img)
    rgb = cv2.cvtColor(cv2.resize(img, (400, 400)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    rec = {"file": fn, "x": torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)),
           "target": torch.from_numpy(tg),
           "mask": torch.from_numpy(cv2.resize(mask, (GRID, GRID), interpolation=cv2.INTER_NEAREST).astype(np.float32))[None],
           "K": K, "dims": dims, "R": R, "t": t, "size": size}
    if with_candidates:
        sup = support_maps(R, t, K, dims, size, PLH.gradient_association_mask(img))
        uns = {"width": sup["top_width"] | sup["base_width"],
               "depth": sup["top_depth"] | sup["base_depth"], "vertical": sup["vertical"]}
        rec["cands"] = [c_["R"] for c_ in SAI.semantic_axis_initialization(uns, K, dims, size)["candidates"]]
    return rec


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = DopeNetwork(numVec=0, numSeg=1)
        st = torch.load(str(ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"),
                        map_location="cpu", weights_only=True)
        self.net.load_state_dict({k.removeprefix("module."): v for k, v in st.items()}, strict=True)
        for p_ in self.net.parameters():
            p_.requires_grad_(False)
        self.net.eval(); self.feat = None
    def discover(self, sample):
        self.idx, self.ch = PLH.find_high_resolution_feature(self.net.vgg, sample, GRID)
        self.net.vgg[self.idx].register_forward_hook(lambda m, i, o: setattr(self, "feat", o))
        return self.idx, self.ch
    @torch.no_grad()
    def forward(self, x):
        self.net(x)
        assert self.feat.shape[-2:] == (GRID, GRID), self.feat.shape
        return self.feat.detach()


def line_metrics(logits, target, tol=2):
    p = (torch.sigmoid(logits) >= 0.5).float(); g = (target >= 0.5).float()
    k = 2 * tol + 1
    gd = F.max_pool2d(g, k, 1, tol); pd_ = F.max_pool2d(p, k, 1, tol)
    rec = (g * pd_).sum(dim=(0, 2, 3)) / g.sum(dim=(0, 2, 3)).clamp_min(1)
    pre = (p * gd).sum(dim=(0, 2, 3)) / p.sum(dim=(0, 2, 3)).clamp_min(1)
    f1 = 2 * pre * rec / (pre + rec).clamp_min(1e-9)
    return rec.cpu().numpy(), pre.cpu().numpy(), f1.cpu().numpy()


def polarity_select(prob, fr):
    """Score each fixed candidate by sampling the native 100x100 map."""
    W, H = fr["size"]; dims = fr["dims"]; K = fr["K"]; t = fr["t"]
    es = PPD.polarity_edge_classes(dims); best = None; second = None
    for Rc in fr["cands"]:
        proj, dep = PG.project_points(PG.make_corners(*dims)[:8], Rc, t, K)
        num = den = 0.0
        for (i, j), cls in es:
            if dep[i] <= 1e-6 or dep[j] <= 1e-6:
                continue
            cl = PG.clip_segment_to_image(proj[i], proj[j], W, H)
            if cl is None:
                continue
            s = PG.sample_along(cl[0], cl[1], pixels_per_sample=4.0)
            g = np.stack([s[:, 0] * (GRID - 1) / max(W - 1, 1),
                          s[:, 1] * (GRID - 1) / max(H - 1, 1)], 1)
            v, ins = DGP.bilinear_sample(prob[PLH.CLASS_ORDER.index(cls)], g)
            if not bool(ins.any()):
                continue
            num += float(np.sum(-np.log(np.clip(v[ins], 1e-6, 1.)))); den += float(ins.sum())
        if den <= 0:
            continue
        e = num / den
        if best is None or e < best[0]:
            second = best; best = (e, Rc)
        elif second is None or e < second[0]:
            second = (e, Rc)
    if best is None:
        return None
    return {"R": best[1], "energy": best[0],
            "margin": (second[0] - best[0]) if second else None}


@torch.no_grad()
def evaluate(line, mask, base, frames, arm, want_indexed=True):
    line.eval(); mask.eval()
    X = torch.stack([f["x"] for f in frames])
    Y = torch.stack([f["target"] for f in frames]).to(dev)
    M = torch.stack([f["mask"] for f in frames]).to(dev)
    probs, mious, mdice = [], [], []
    for i in range(0, len(X), 16):
        feat = base(X[i:i + 16].to(dev))
        ml = mask(feat)
        gate = PLH.soft_gate(ml) if arm == "M1" else torch.ones_like(ml)
        lo = line(feat, gate)
        probs.append(torch.sigmoid(lo).cpu())
        mp = (torch.sigmoid(ml) >= 0.5).float(); mt = M[i:i + 16]
        inter = (mp * mt).sum()
        mious.append(float(inter / (mp + mt).clamp(0, 1).sum().clamp_min(1)))
        mdice.append(float(2 * inter / (mp.sum() + mt.sum()).clamp_min(1)))
    P = torch.cat(probs)
    logits = torch.log(P.clamp(1e-6, 1 - 1e-6) / (1 - P).clamp_min(1e-6)).to(dev)
    rec, pre, f1 = line_metrics(logits, Y)
    n = ok = navail = 0
    inv_frames, reproj, margins = [], [], []
    for i, fr in enumerate(frames):
        cands = fr.get("cands", [])
        pol = {PPD.candidate_polarity(Rc, fr["dims"]) for Rc in cands}
        if len(cands) < 2 or pol != {"upright", "inverted"}:
            continue
        navail += 1
        sel = polarity_select(P[i].numpy(), fr)
        if sel is None:
            continue
        n += 1
        good = PPD.polarity_correct(sel["R"], fr["R"], fr["dims"])
        ok += int(good)
        if not good:
            inv_frames.append(fr["file"])
        if sel["margin"] is not None:
            margins.append(sel["margin"])
        if want_indexed:
            gt2d, _ = PG.project_points(PG.make_corners(*fr["dims"]), fr["R"], fr["t"], fr["K"])
            r = PPD.fixed_indexed_reprojection(
                {"R": sel["R"], "t": fr["t"]}, [list(p) for p in gt2d], fr["K"], fr["dims"])
            if r is not None:
                reproj.append(r)
    return {"arm": arm, "n_frames": len(frames), "n_candidate_pair": navail, "n_scored": n,
            "polarity_acc": ok / max(n, 1), "inversion": n - ok,
            "inversion_rate": (n - ok) / max(n, 1),
            "indexed_reproj_median": float(np.median(reproj)) if reproj else None,
            "margin_median": float(np.median(margins)) if margins else None,
            "macro_f1": float(np.mean(f1)),
            "line_recall": {c: float(rec[i]) for i, c in enumerate(PLH.CLASS_ORDER)},
            "line_precision": {c: float(pre[i]) for i, c in enumerate(PLH.CLASS_ORDER)},
            "mask_iou": float(np.mean(mious)), "mask_dice": float(np.mean(mdice)),
            "inversion_frames": inv_frames[:50]}


def atomic_save(obj, path):
    tmp = pathlib.Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_arm(arm, train_frames, val_frames, base, ch, init_line, init_mask, cal):
    out = WROOT / arm; out.mkdir(parents=True, exist_ok=True)
    state_path = out / "run_state.json"
    line = PLH.PolarityLineHead(ch).to(dev); mask = PLH.FreshMaskHead(ch).to(dev)
    params = list(line.parameters()) + (list(mask.parameters()) if arm in ("M0", "M1") else [])
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    start = 0; history = []
    if state_path.is_file():
        st = json.loads(state_path.read_text())
        if st.get("completed"):
            log(f"[{arm}] already completed (epoch {st['epoch']}) — skipping")
            return json.loads((out / "metrics_by_epoch.json").read_text())
        ck = torch.load(out / "last.pth", map_location=dev)
        line.load_state_dict(ck["line"]); mask.load_state_dict(ck["mask"])
        opt.load_state_dict(torch.load(out / "optimizer_last.pth", map_location=dev))
        torch.set_rng_state(torch.load(out / "rng_state.pt")["cpu"])
        start = st["epoch"]; history = json.loads((out / "metrics_by_epoch.json").read_text())
        log(f"[{arm}] resumed from epoch {start}")
    else:
        seed_all(SEED); line.load_state_dict(init_line); mask.load_state_dict(init_mask)

    X = torch.stack([f["x"] for f in train_frames])
    Y = torch.stack([f["target"] for f in train_frames])
    M = torch.stack([f["mask"] for f in train_frames])
    pw = torch.tensor(cal["pos_weight"], dtype=torch.float32)
    for epoch in range(start, EPOCHS):
        line.train(); mask.train()
        g = torch.Generator().manual_seed(SEED * 1000 + epoch)   # identical across arms
        order = torch.randperm(len(train_frames), generator=g)
        losses = []
        t0 = time.time()
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            feat = base(X[idx].to(dev))
            y_, m_ = Y[idx].to(dev), M[idx].to(dev)
            ml = mask(feat)
            gate = PLH.soft_gate(ml) if arm == "M1" else torch.ones_like(ml)
            lo = line(feat, gate)
            loss = PLH.line_map_loss(lo, y_, pw) + cal["lambda_pol"] * PLH.polarity_contrast_loss(lo, y_)
            if arm in ("M0", "M1"):
                loss = loss + cal["lambda_mask"] * PLH.mask_loss(ml, m_) \
                     + cal["lambda_out"] * PLH.outside_mask_penalty(lo, PLH.soft_gate(ml))
            if not torch.isfinite(loss):
                raise RuntimeError(f"BLOCKED: non-finite loss in {arm} epoch {epoch}")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss))
        val = evaluate(line, mask, base, val_frames, arm)
        val.update({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                    "seconds": time.time() - t0})
        history.append(val)
        atomic_save({"line": line.state_dict(), "mask": mask.state_dict()}, out / "last.pth")
        atomic_save({"line": line.state_dict(), "mask": mask.state_dict()},
                    out / f"epoch_{epoch + 1:03d}.pth")
        atomic_save(opt.state_dict(), out / "optimizer_last.pth")
        atomic_save({"cpu": torch.get_rng_state()}, out / "rng_state.pt")
        (out / "metrics_by_epoch.json").write_text(json.dumps(history, indent=1))
        state_path.write_text(json.dumps(
            {"arm": arm, "epoch": epoch + 1, "completed": epoch + 1 >= EPOCHS,
             "head": os.popen("git rev-parse HEAD").read().strip(),
             "timestamp": time.time()}, indent=1))
        log(f"[{arm}] epoch {epoch+1}/{EPOCHS} loss {np.mean(losses):.4f} "
            f"polAcc {val['polarity_acc']:.3f} inv {val['inversion']}/{val['n_scored']} "
            f"F1 {val['macro_f1']:.3f} IoU {val['mask_iou']:.3f} ({val['seconds']:.0f}s)")
    return history


def select_best(history):
    def key(h):
        return (-h["polarity_acc"], h["inversion_rate"],
                h["indexed_reproj_median"] if h["indexed_reproj_median"] is not None else 1e9,
                -h["macro_f1"], h["epoch"])
    return min(history, key=key)


VAL_GATE = {"polarity_acc": 0.95, "inversion_rate": 0.05, "reproj_reduction": 0.70}


def gate_check(metrics, unsigned_reproj):
    reduction = (None if metrics["indexed_reproj_median"] is None or not unsigned_reproj
                 else 1.0 - metrics["indexed_reproj_median"] / unsigned_reproj)
    checks = {
        "polarity_acc>=0.95": metrics["polarity_acc"] >= VAL_GATE["polarity_acc"],
        "inversion_rate<=0.05": metrics["inversion_rate"] <= VAL_GATE["inversion_rate"],
        "reproj_reduction>=0.70": (reduction or 0) >= VAL_GATE["reproj_reduction"],
        "nan_inf==0": all(np.isfinite(v) for v in metrics["line_recall"].values()),
    }
    return {"checks": checks, "reproj_reduction": reduction, "passed": all(checks.values())}


def unsigned_baseline(frames):
    """Indexed reprojection with NO polarity information (the S0 behaviour)."""
    values = []
    for fr in frames:
        cands = fr.get("cands", [])
        if len(cands) < 2:
            continue
        gt2d, _ = PG.project_points(PG.make_corners(*fr["dims"]), fr["R"], fr["t"], fr["K"])
        r = PPD.fixed_indexed_reprojection(
            {"R": cands[0], "t": fr["t"]}, [list(p) for p in gt2d], fr["K"], fr["dims"])
        if r is not None:
            values.append(r)
    return float(np.median(values)) if values else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-long-run", action="store_true")
    ap.add_argument("--arm", choices=ARMS, action="append", default=[])
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--output_dir", default=str(D), help="results folder; must hold PURPOSE.md")
    ap.add_argument("--evaluate-test", action="store_true")
    ap.add_argument("--evaluate-real", action="store_true")
    ap.add_argument("--limit-test", type=int, default=0)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.output_dir)
    if not (out_dir / "PURPOSE.md").is_file():
        raise RuntimeError(f"BLOCKED: {out_dir}/PURPOSE.md missing")
    WROOT.mkdir(parents=True, exist_ok=True)

    train_man = json.loads((D / "ppd_train_manifest.json").read_text())
    val_man = json.loads((D / "ppd_val_manifest.json").read_text())
    cal = json.loads((D / "ppd_t2_loss_calibration.json").read_text())
    provenance = {
        "split_source": "locked full split (no fixed 3000/1000 submanifest existed)",
        "train_n": train_man["n"], "val_n": val_man["n"],
        "train_hash": hashlib.sha256((D / "ppd_train_manifest.json").read_bytes()).hexdigest(),
        "val_hash": hashlib.sha256((D / "ppd_val_manifest.json").read_bytes()).hexdigest(),
        "calibration_hash": hashlib.sha256((D / "ppd_t2_loss_calibration.json").read_bytes()).hexdigest(),
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WD, "seed": SEED,
        "note": "H1/H2 keep their historical FAIL; progression uses H3 candidate polarity.",
    }
    (D / "PPD_LONG_RUN_PROVENANCE.md").write_text(
        "# PPD long-run provenance\n\n```\n" + json.dumps(provenance, indent=1) + "\n```\n",
        encoding="utf-8")

    files_tr = [f["file"] for f in train_man["frames"]]
    files_va = [f["file"] for f in val_man["frames"]]
    if args.limit_train:
        files_tr = files_tr[:args.limit_train]
    if args.limit_val:
        files_va = files_va[:args.limit_val]
    log(f"[data] train {len(files_tr)}  val {len(files_va)}  (locked full split)")

    t0 = time.time(); train_frames = [load_frame(f) for f in files_tr]
    log(f"[data] train loaded in {time.time()-t0:.0f}s")
    t0 = time.time(); val_frames = [load_frame(f, with_candidates=True) for f in files_va]
    pair = sum(1 for f in val_frames if len(f.get("cands", [])) >= 2
               and {PPD.candidate_polarity(R, f["dims"]) for R in f["cands"]} == {"upright", "inverted"})
    log(f"[data] val loaded in {time.time()-t0:.0f}s  candidate-pair {pair}/{len(val_frames)}")
    base_unsigned = unsigned_baseline(val_frames)
    log(f"[data] unsigned validation indexed-reproj baseline = "
        f"{base_unsigned if base_unsigned is None else round(base_unsigned,2)} px")
    (D / "ppd_val_candidate_set.json").write_text(json.dumps(
        {"n": len(val_frames), "candidate_pair": pair,
         "unsigned_indexed_reproj_median": base_unsigned,
         "hash": hashlib.sha256(json.dumps(files_va).encode()).hexdigest()}, indent=1))

    base = Base().to(dev)
    idx, ch = base.discover(train_frames[0]["x"][None].to(dev))
    log(f"[model] high-res feature vgg[{idx}] ch={ch} {GRID}x{GRID} asserted")
    seed_all(SEED)
    init_line = PLH.PolarityLineHead(ch).state_dict()
    init_mask = PLH.FreshMaskHead(ch).state_dict()

    arms = args.arm or (list(ARMS) if args.all_long_run else [])
    if not (args.evaluate_test or args.evaluate_real):
        for arm in arms:
            log(f"=== training {arm} ===")
            train_arm(arm, train_frames, val_frames, base, ch, init_line, init_mask, cal)

    summary = {}
    for arm in ARMS:
        path = WROOT / arm / "metrics_by_epoch.json"
        if not path.is_file():
            continue
        history = json.loads(path.read_text())
        best = select_best(history)
        gate = gate_check(best, base_unsigned)
        summary[arm] = {"best_epoch": best["epoch"], "best": best, "gate": gate}
        log(f"[{arm}] best epoch {best['epoch']}  polAcc {best['polarity_acc']:.3f}  "
            f"inv {best['inversion']}/{best['n_scored']}  gate "
            f"{'PASS' if gate['passed'] else 'FAIL'}")
    if summary:
        (D / "ppd_validation_metrics.json").write_text(json.dumps(summary, indent=1))
        rows = []
        for arm in summary:
            for h in json.loads((WROOT / arm / "metrics_by_epoch.json").read_text()):
                rows.append({"arm": arm, **{k: v for k, v in h.items()
                                            if not isinstance(v, (dict, list))}})
        pd.DataFrame(rows).to_csv(D / "ppd_metrics_by_epoch.csv", index=False)
    # --- untouched synthetic test (validation-PASS arms only) ---
    if args.evaluate_test:
        gate_summary = json.loads((D / "ppd_validation_metrics.json").read_text())
        passing = [a for a, v in gate_summary.items() if v["gate"]["passed"]]
        log(f"=== untouched test: validation-PASS arms {passing} ===")
        evaluate_untouched(passing, base, ch, limit=args.limit_test)

    # --- real N87 one-shot (untouched-PASS arms only) ---
    if args.evaluate_real:
        path = D / "ppd_untouched_metrics.json"
        if not path.is_file():
            raise RuntimeError("BLOCKED: run --evaluate-test before --evaluate-real")
        untouched = json.loads(path.read_text())["per_arm"]
        passing = [a for a, v in untouched.items() if v["gate"]["passed"]]
        if not passing:
            log("[real] no arm passed the untouched gate -> real evaluation forbidden")
        else:
            log(f"=== real N87 one-shot: untouched-PASS arms {passing} ===")
            evaluate_real(passing, base, ch)

    log(f"[done] {D}")
    return 0




# ============================================================================
# Untouched synthetic test / real N87 one-shot / conditional CGR
# ============================================================================
def load_best(arm, ch):
    """Best checkpoint chosen on synthetic validation only."""
    summary = json.loads((D / "ppd_validation_metrics.json").read_text())
    epoch = summary[arm]["best_epoch"]
    ck = torch.load(WROOT / arm / f"epoch_{epoch:03d}.pth", map_location=dev)
    line = PLH.PolarityLineHead(ch).to(dev); line.load_state_dict(ck["line"])
    mask = PLH.FreshMaskHead(ch).to(dev); mask.load_state_dict(ck["mask"])
    return line, mask, epoch


UNTOUCHED_CHUNK = 400


def _merge(parts):
    """Combine per-chunk evaluate() results into one record."""
    total = {"n_frames": 0, "n_candidate_pair": 0, "n_scored": 0, "inversion": 0}
    reproj, margins, f1s, ious, dices = [], [], [], [], []
    rec = {c: [] for c in PLH.CLASS_ORDER}; pre = {c: [] for c in PLH.CLASS_ORDER}
    inv_frames = []
    for p in parts:
        for k in total:
            total[k] += p[k]
        if p["indexed_reproj_median"] is not None:
            reproj.append((p["indexed_reproj_median"], p["n_scored"]))
        if p["margin_median"] is not None:
            margins.append(p["margin_median"])
        f1s.append(p["macro_f1"]); ious.append(p["mask_iou"]); dices.append(p["mask_dice"])
        for c in PLH.CLASS_ORDER:
            rec[c].append(p["line_recall"][c]); pre[c].append(p["line_precision"][c])
        inv_frames.extend(p["inversion_frames"])
    weight = sum(n for _, n in reproj) or 1
    return {
        **total,
        "polarity_acc": (total["n_scored"] - total["inversion"]) / max(total["n_scored"], 1),
        "inversion_rate": total["inversion"] / max(total["n_scored"], 1),
        "indexed_reproj_median": (sum(v * n for v, n in reproj) / weight) if reproj else None,
        "margin_median": float(np.median(margins)) if margins else None,
        "macro_f1": float(np.mean(f1s)), "mask_iou": float(np.mean(ious)),
        "mask_dice": float(np.mean(dices)),
        "line_recall": {c: float(np.mean(rec[c])) for c in PLH.CLASS_ORDER},
        "line_precision": {c: float(np.mean(pre[c])) for c in PLH.CLASS_ORDER},
        "inversion_frames": inv_frames[:50],
    }


def evaluate_untouched(arms, base, ch, limit=0):
    """Chunked: 5,916 frames do not fit in memory at once (an earlier attempt
    was OOM-killed).  Chunking changes no metric definition, only residency."""
    man = json.loads((D / "ppd_untouched_manifest.json").read_text())
    files = [f["file"] for f in man["frames"]]
    if limit:
        files = files[:limit]
    meta = {f["file"]: f for f in man["frames"]}
    heads = {arm: load_best(arm, ch) for arm in arms}
    parts = {arm: [] for arm in arms}
    slice_parts = {arm: {} for arm in arms}
    base_values = []
    t0 = time.time()
    for start in range(0, len(files), UNTOUCHED_CHUNK):
        chunk = [load_frame(f, with_candidates=True)
                 for f in files[start:start + UNTOUCHED_CHUNK]]
        b = unsigned_baseline(chunk)
        if b is not None:
            base_values.append(b)
        for arm in arms:
            line, mask, _ = heads[arm]
            parts[arm].append(evaluate(line, mask, base, chunk, arm))
            for key in ("asset", "mode"):
                for value in sorted({meta[f["file"]][key] for f in chunk if f["file"] in meta}):
                    subset = [f for f in chunk if meta.get(f["file"], {}).get(key) == value]
                    if len(subset) < 10:
                        continue
                    slice_parts[arm].setdefault(f"{key}={value}", []).append(
                        evaluate(line, mask, base, subset, arm, want_indexed=False))
        del chunk
        log(f"  [untouched] {min(start + UNTOUCHED_CHUNK, len(files))}/{len(files)} "
            f"{time.time()-t0:.0f}s")
    baseline = float(np.median(base_values)) if base_values else None
    log(f"[untouched] unsigned indexed reproj baseline = {baseline:.2f}px")
    out = {}
    for arm in arms:
        m = _merge(parts[arm])
        m["arm"] = arm
        m["best_epoch"] = heads[arm][2]
        m["gate"] = gate_check(m, baseline)
        m["slices"] = {k: {"n": _merge(v)["n_scored"],
                           "polarity_acc": _merge(v)["polarity_acc"],
                           "inversion_rate": _merge(v)["inversion_rate"]}
                       for k, v in slice_parts[arm].items()}
        out[arm] = m
        log(f"[untouched] {arm} polAcc {m['polarity_acc']:.3f} "
            f"inv {m['inversion']}/{m['n_scored']} reproj {m['indexed_reproj_median']:.2f}px "
            f"gate {'PASS' if m['gate']['passed'] else 'FAIL'}")
    (D / "ppd_untouched_metrics.json").write_text(
        json.dumps({"baseline_indexed_reproj": baseline, "per_arm": out}, indent=1))
    return out, baseline


def real_frames():
    """Build the real mechanism-val N87 frame records (shared by eval and overlays)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "LS", ROOT / "scripts/stage0/paper_s2/paper_s2_palletgraph_line_screen.py")
    LS = importlib.util.module_from_spec(spec); spec.loader.exec_module(LS)
    ev = LS.LineScreenEvaluator()
    raw = pd.read_parquet(D / "rotation_candidates_raw.parquet").set_index("frame_id")
    frames = []
    for spec_f in ev.frames:
        uid = spec_f["frame_id"]
        if uid not in raw.index:
            continue
        cands_raw = raw.loc[uid, "_candidates"]
        if cands_raw is None or len(cands_raw) == 0:
            continue
        g = ev.geometry[uid]
        ref = g.solve(g.gt_points)
        if ref is None:
            continue
        img = ev.images[uid]
        rgb = cv2.cvtColor(cv2.resize(img, (400, 400)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        frames.append({
            "file": uid, "x": torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)),
            "target": torch.zeros(5, GRID, GRID), "mask": torch.zeros(1, GRID, GRID),
            "K": g.K, "dims": g.dims, "R": ref["R"], "t": np.asarray(ref["t"]).reshape(3),
            "size": (spec_f["image_width"], spec_f["image_height"]),
            "cands": [np.asarray(m, float).reshape(3, 3) for m in cands_raw],
            "point_fail": bool(g.solve(ev.decoded[uid]["D0"]) is None),
            "is_truncated": spec_f["is_truncated"],
            "failure_class": ev.classes.loc[uid, "failure_class"],
            "session": spec_f["session_id"], "domain": spec_f["domain"]})
    log(f"[real] {len(frames)} frames with candidates")
    return frames


def evaluate_real(arms, base, ch):
    """One shot on the real mechanism-val N87.  Never used for selection."""
    frames = real_frames()
    out = {}
    for arm in arms:
        line, mask, epoch = load_best(arm, ch)
        m = evaluate(line, mask, base, frames, arm)
        m["best_epoch"] = epoch
        pf = [f for f in frames if f["point_fail"]]
        pm = evaluate(line, mask, base, pf, arm, want_indexed=False) if pf else {}
        m["point_fail_polarity_correct"] = int(round(
            pm.get("polarity_acc", 0) * pm.get("n_scored", 0)))
        m["point_fail_n"] = pm.get("n_scored", 0)
        for key in ("failure_class", "domain", "is_truncated"):
            slices = {}
            for value in sorted({str(f[key]) for f in frames}):
                subset = [f for f in frames if str(f[key]) == value]
                if len(subset) < 5:
                    continue
                sm = evaluate(line, mask, base, subset, arm, want_indexed=False)
                slices[f"{key}={value}"] = {"n": sm["n_scored"],
                                            "polarity_acc": sm["polarity_acc"],
                                            "inversion": sm["inversion"]}
            m[f"slices_{key}"] = slices
        out[arm] = m
        log(f"[real] {arm} polAcc {m['polarity_acc']:.3f} inv {m['inversion']}/{m['n_scored']} "
            f"reproj {m['indexed_reproj_median']:.2f}px "
            f"point-fail {m['point_fail_polarity_correct']}/{m['point_fail_n']}")
    (D / "ppd_real_metrics.json").write_text(json.dumps(out, indent=1))
    return out, frames


if __name__ == "__main__":
    raise SystemExit(main())
