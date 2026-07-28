"""STAGE24 vec impl-check — 6항목 텐서레벨 검산.

목적: L_vec가 seen조차 안 내려간(0.51->0.38) 게 target/loss 버그인지, 표현 자체
불가인지 가른다. train조차 실패면 target/loss 먼저 의심.

버그면 벡터 negative 결론 재검토. 정상이면 negative 확정 유지.
학습 재실행 없이 loaded weights + 실제 배치 forward로만 검산.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "train"))

from models import DopeNetwork  # noqa: E402
from utils_dataset import CleanVisiiDopeLoader  # noqa: E402
from measure_lambda_gradients import vec_loss  # noqa: E402

OUT = os.path.join(ROOT, "data", "pallet", "eval_results",
                   "stage24_vec_newdata", "impl_check")
os.makedirs(OUT, exist_ok=True)
WEIGHTS = os.path.join(ROOT, "weights", "stage24_vec_newdata", "voting",
                       "final_net_voting_unit.pth")
SEEN = os.path.join(ROOT, "challenge", "data", "training", "v3", "batch_000")
HELD = os.path.join(ROOT, "challenge", "data", "training", "v3", "batch_009")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)


def make_loader(path, n=64, bs=8):
    ds = CleanVisiiDopeLoader(
        [path], objects=["pallet"], sigma=4.0, output_size=50,
        pvnet_vec=True, pvnet_unit=True, pvnet_mask_rle=True)
    ntot = len(ds)
    idx = np.linspace(0, ntot - 1, min(n, ntot)).round().astype(int)
    idx = sorted(set(int(i) for i in idx))
    ds = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=False, num_workers=4), ntot


def load_net():
    net = DopeNetwork(numVec=18, numSeg=1).to(device)
    state = torch.load(WEIGHTS, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    net.eval()
    return net


def percorner_vec_loss(vpred, gt_vec, gt_mask):
    """last-stage per-corner masked smooth_l1 (denom=mask_px*2 per corner)."""
    out = []
    mpx = gt_mask.sum().clamp_min(1.0).item()
    for k in range(9):
        ch = [2 * k, 2 * k + 1]
        diff = (vpred[:, ch] - gt_vec[:, ch]) * gt_mask
        l = torch.nn.functional.smooth_l1_loss(
            diff, torch.zeros_like(diff), reduction="sum").item() / (mpx * 2)
        out.append(l)
    return out


def perimg_vec_loss(vpred, gt_vec, gt_mask):
    """per-image L_vec: 이미지별 masked smooth_l1 sum / (mask_px_i * 18)."""
    B = gt_vec.shape[0]
    res = []
    for i in range(B):
        m = gt_mask[i:i + 1]
        mpx = m.sum().clamp_min(1.0).item()
        diff = (vpred[i:i + 1] - gt_vec[i:i + 1]) * m
        l = torch.nn.functional.smooth_l1_loss(
            diff, torch.zeros_like(diff), reduction="sum").item() / (mpx * 18)
        res.append((l, mpx))
    return res


def analyze(net, loader, tag, nkps_report):
    """returns dict of aggregated metrics for a split."""
    gt_norm_all = []        # per masked pixel-corner GT vector norm (unit check)
    mask_uni = set()
    mask_holes = 0
    mask_frames = 0
    valid_counts = []       # per-image mask px count
    Lvec_batch = []         # global-denom L_vec (as trained) per batch
    Lvec_zero = []          # baseline: predict-zero L_vec per batch
    perimg = []             # (loss, mask_px)
    corner_acc = np.zeros(9)
    corner_n = 0
    # scale/condition buckets: (bucket -> [perimg loss])
    buckets = {}
    nan_ct = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].float().to(device)
            gt_vec = batch["pvnet_vec"].float().to(device)
            gt_mask = batch["pvnet_mask"].float().to(device)
            out = net(img)
            beliefs, aff, vec_stages, seg = out
            vpred = vec_stages[-1]

            # item2: GT unit norm within mask (last corner set, sample)
            for k in range(9):
                dx = gt_vec[:, 2 * k]
                dy = gt_vec[:, 2 * k + 1]
                nrm = torch.sqrt(dx * dx + dy * dy)
                m = gt_mask[:, 0] > 0.5
                vals = nrm[m]
                if vals.numel():
                    gt_norm_all.append(vals.cpu().numpy())
            # item3: mask binary + holes
            for i in range(gt_mask.shape[0]):
                mm = gt_mask[i, 0].cpu().numpy()
                mask_uni.update(np.unique(mm).tolist())
                mask_frames += 1
                # hole = interior 0 surrounded by 1 (rough: 0-px inside bbox of mask)
                ys, xs = np.where(mm > 0.5)
                if len(ys):
                    sub = mm[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                    if (sub < 0.5).any():
                        mask_holes += 1
                valid_counts.append(float((mm > 0.5).sum()))
            # item1/6: L_vec global-denom (as trained) + zero baseline
            Lg = vec_loss([vpred], gt_vec, gt_mask).item()  # 1-stage version
            Lvec_batch.append(Lg)
            zero = torch.zeros_like(vpred)
            Lz = vec_loss([zero], gt_vec, gt_mask).item()
            Lvec_zero.append(Lz)
            # per-image
            perimg.extend(perimg_vec_loss(vpred, gt_vec, gt_mask))
            # item5: per-corner
            pc = percorner_vec_loss(vpred, gt_vec, gt_mask)
            corner_acc += np.array(pc)
            corner_n += 1
            # item4: scale/condition buckets by per-image
            pil = perimg_vec_loss(vpred, gt_vec, gt_mask)
            for i, (l, mpx) in enumerate(pil):
                # scale bucket by mask px (of 50x50=2500)
                if mpx < 150:
                    sb = "small"
                elif mpx < 500:
                    sb = "mid"
                else:
                    sb = "large"
                buckets.setdefault(sb, []).append(l)
            if not np.isfinite(Lg):
                nan_ct += 1

    gt_norm_all = np.concatenate(gt_norm_all) if gt_norm_all else np.array([0.])
    vc = np.array(valid_counts)
    pil_l = np.array([p[0] for p in perimg])
    res = {
        "tag": tag,
        "n_frames": mask_frames,
        # item1
        "valid_count_mean": float(vc.mean()),
        "valid_count_std": float(vc.std()),
        "valid_count_min": float(vc.min()),
        "valid_count_max": float(vc.max()),
        "valid_count_cv": float(vc.std() / (vc.mean() + 1e-9)),
        "Lvec_globaldenom_mean": float(np.mean(Lvec_batch)),
        "Lvec_perimg_mean": float(pil_l.mean()),
        "Lvec_perimg_std": float(pil_l.std()),
        # item2
        "gt_norm_mean": float(gt_norm_all.mean()),
        "gt_norm_std": float(gt_norm_all.std()),
        "gt_norm_frac_near1": float(np.mean(np.abs(gt_norm_all - 1.0) < 0.05)),
        "gt_norm_has_nan": bool(np.isnan(gt_norm_all).any()),
        # item3
        "mask_unique_vals": sorted(mask_uni),
        "mask_frac_with_holes": float(mask_holes / max(1, mask_frames)),
        # item5
        "percorner_Lvec": {str(k): float(corner_acc[k] / max(1, corner_n))
                           for k in range(9)},
        # item6 baseline
        "Lvec_predictzero_mean": float(np.mean(Lvec_zero)),
        # item4 buckets
        "scale_buckets": {k: {"n": len(v), "Lvec": float(np.mean(v))}
                          for k, v in buckets.items()},
        "nan_batches": nan_ct,
    }
    return res


def main():
    net = load_net()
    seen_loader, seen_tot = make_loader(SEEN)
    held_loader, held_tot = make_loader(HELD)
    print(f"[data] seen(batch_000) tot={seen_tot}  held(batch_009) tot={held_tot}")
    seen = analyze(net, seen_loader, "seen_batch000", nkps_report=True)
    held = analyze(net, held_loader, "held_batch009", nkps_report=True)

    result = {
        "weights": WEIGHTS,
        "note": "unit-vec voting head. Lvec 1-stage (last) here; train logs 2-stage sum (~2x).",
        "seen": seen,
        "held": held,
    }
    with open(os.path.join(OUT, "impl_check.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[save] {OUT}/impl_check.json")


if __name__ == "__main__":
    main()
