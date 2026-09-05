"""GATE C — 국소 코너 전문가 파일럿 학습.

    C0  SYN_LOCAL            합성 정확 감독만
    C1  SYN_PLUS_REAL_SOFT   + high-consensus real soft distribution + photometric consistency

동일 init(seed) · 동일 synthetic 노출 · 동일 optimizer updates.
checkpoint 는 last only.  DEV_EVAL 로 epoch 을 고르지 않는다.

loss 가중치는 실행 전에 고정했고 결과를 보고 바꾸지 않는다 (아래 WEIGHTS).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M
from mtcd_specialist import (CROP, LocalCornerSpecialist, extract_crop,
                             gaussian_target, mixture_target)

SYN = M.REPO_ROOT / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
WEIGHTS = {"heat": 1.0, "visibility": 0.2, "uncertainty": 0.1, "edge": 0.1,
           "real_kl": 1.0, "consistency": 0.5}
SYN_CROPS_PER_STEP = 48
REAL_CROPS_PER_STEP = 16
SEED = 20260905


def load_residuals():
    a = np.load(M.AUDIT / "R0_SOURCE_COARSE_RESIDUALS.npy")
    return {k: a[a[:, 0] == k][:, 1:] for k in range(8)}


def syn_sample(stem, residuals, rng):
    img = cv2.imread(str(SYN / "images/train" / f"{stem}.png"))
    if img is None:
        return []
    h, w = img.shape[:2]
    line = (SYN / "labels/train" / f"{stem}.txt").read_text().split("\n")[0].split()
    if len(line) < 32:
        return []
    v = list(map(float, line[5:]))
    xy = np.array([[v[3 * i] * w, v[3 * i + 1] * h] for i in range(9)])
    vis = np.array([v[3 * i + 2] for i in range(9)])
    out = []
    for k in range(8):
        if vis[k] <= 0:
            continue
        pool = residuals[k]
        centre = xy[k] + pool[rng.integers(len(pool))]
        patch, origin = extract_crop(img, centre)
        local = xy[k] - origin
        inside = bool(0 <= local[0] < CROP and 0 <= local[1] < CROP)
        dirs = []
        for a, b in M.INCIDENT_EDGES[k]:
            other = b if a == k else a
            d = xy[other] - xy[k]
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                dirs.append((n, d / n))
        dirs.sort(key=lambda t: -t[0])
        edge = np.zeros((2, 2), np.float32)
        for i in range(min(2, len(dirs))):
            edge[i] = dirs[i][1]
        out.append({"patch": patch, "kp": k, "local": local.astype(np.float32),
                    "inside": inside, "edge": edge,
                    "has_edge": float(len(dirs) >= 2)})
    return out


def photometric(patch, rng, strong):
    """기하를 바꾸지 않는 증강만. flip·rotation·crop transform 금지."""
    x = patch.astype(np.float32)
    if strong:
        x = x * rng.uniform(0.55, 1.5) + rng.uniform(-45, 45)
        hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8),
                           cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-12, 12)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.5, 1.6), 0, 255)
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        x = x + rng.normal(0, rng.uniform(2, 14), x.shape)
        if rng.random() < 0.5:
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.4, 1.6))
    else:
        x = x * rng.uniform(0.9, 1.1) + rng.uniform(-8, 8)
    return np.clip(x, 0, 255)


def to_tensor(patches, device):
    a = np.stack(patches).astype(np.float32) / 255.0
    return torch.from_numpy(a.transpose(0, 3, 1, 2)).to(device)


def real_batch(entries, rng, n, cache):
    out = []
    for _ in range(n):
        e = entries[rng.integers(len(entries))]
        img = cache.get(e["image_path"])
        if img is None:
            img = cv2.imread(str(M.REPO_ROOT / e["image_path"]))
            if img is None:
                continue
            if len(cache) < 240:
                cache[e["image_path"]] = img
        patch, origin = extract_crop(img, np.asarray(e["r0_xy"], float))
        q = mixture_target(np.asarray(e["teacher_xy"], float) - origin)
        if q is None:
            continue
        out.append({"patch": patch, "kp": e["kp"], "q": q})
    return out


def train(arm, updates, cap_minutes, device="cuda:0"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    residuals = load_residuals()
    stems = sorted(p.stem for p in (SYN / "images/train").glob("*.png"))
    random.Random(SEED).shuffle(stems)

    real_entries, real_cache = [], {}
    if arm == "C1":
        cached = json.loads((M.GATE_C / "TARGET_TEACHER_CACHE.json").read_text())
        for f in cached["frames"].values():
            for kp in f["keypoints"]:
                if kp.get("usable"):
                    real_entries.append({"image_path": f["image_path"], "kp": kp["kp"],
                                         "r0_xy": f["r0_keypoints_xy"][kp["kp"]],
                                         "teacher_xy": kp["teacher_xy"]})
        print(f"  real usable keypoints: {len(real_entries)}")

    model = LocalCornerSpecialist().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates)
    started = time.time()
    cursor, history, done = 0, [], 0

    for step in range(updates):
        if (time.time() - started) / 60.0 > cap_minutes:
            print(f"  wall-clock cap reached at step {step}")
            break
        samples = []
        while len(samples) < SYN_CROPS_PER_STEP:
            samples += syn_sample(stems[cursor], residuals, rng)
            cursor = (cursor + 1) % len(stems)
        samples = samples[:SYN_CROPS_PER_STEP]

        crops = to_tensor([s["patch"] for s in samples], device)
        ids = torch.tensor([s["kp"] for s in samples], device=device)
        out = model(crops, ids)

        inside = torch.tensor([s["inside"] for s in samples], device=device)
        tgt = np.stack([gaussian_target(s["local"]) if s["inside"]
                        else np.zeros((CROP, CROP), np.float32) for s in samples])
        tgt_t = torch.from_numpy(tgt.reshape(len(samples), -1)).to(device)
        logp = F.log_softmax(out["heat_logits"], dim=1)
        heat_loss = (-(tgt_t * logp).sum(1)[inside].mean() if inside.any()
                     else torch.zeros((), device=device))

        vis_loss = F.binary_cross_entropy_with_logits(out["visibility_logit"],
                                                      inside.float())

        with torch.no_grad():
            idx = out["heat_logits"].argmax(1)
            pred_xy = torch.stack([(idx % CROP).float(), (idx // CROP).float()], 1)
            true_xy = torch.tensor(np.stack([s["local"] for s in samples]), device=device)
            actual = torch.linalg.norm(pred_xy - true_xy, dim=1)
        unc_loss = (F.l1_loss(F.softplus(out["log_uncertainty"])[inside], actual[inside])
                    if inside.any() else torch.zeros((), device=device))

        edge_t = torch.tensor(np.stack([s["edge"] for s in samples]), device=device)
        has_edge = torch.tensor([s["has_edge"] for s in samples], device=device)
        pe = F.normalize(out["edge_dirs"], dim=2)
        edge_loss = (((1 - (pe * edge_t).sum(2)).mean(1) * has_edge).sum()
                     / has_edge.sum().clamp_min(1))

        loss = (WEIGHTS["heat"] * heat_loss + WEIGHTS["visibility"] * vis_loss +
                WEIGHTS["uncertainty"] * unc_loss + WEIGHTS["edge"] * edge_loss)
        real_kl = cons = torch.zeros((), device=device)

        if arm == "C1" and real_entries:
            rb = real_batch(real_entries, rng, REAL_CROPS_PER_STEP, real_cache)
            if rb:
                weak = to_tensor([photometric(s["patch"], rng, False) for s in rb], device)
                strong = to_tensor([photometric(s["patch"], rng, True) for s in rb], device)
                rid = torch.tensor([s["kp"] for s in rb], device=device)
                ow, os_ = model(weak, rid), model(strong, rid)
                q = torch.from_numpy(np.stack([s["q"] for s in rb])
                                     .reshape(len(rb), -1)).to(device)
                real_kl = -(q * F.log_softmax(ow["heat_logits"], 1)).sum(1).mean()
                pw = F.log_softmax(ow["heat_logits"], 1)
                ps = F.log_softmax(os_["heat_logits"], 1)
                cons = 0.5 * (F.kl_div(ps, pw.exp(), reduction="batchmean") +
                              F.kl_div(pw, ps.exp(), reduction="batchmean"))
                loss = loss + WEIGHTS["real_kl"] * real_kl + WEIGHTS["consistency"] * cons

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        done = step + 1

        if step % 250 == 0 or step == updates - 1:
            med = float(actual[inside].median()) if inside.any() else None
            history.append({"step": step, "loss": float(loss), "heat": float(heat_loss),
                            "vis": float(vis_loss), "unc": float(unc_loss),
                            "edge": float(edge_loss), "real_kl": float(real_kl),
                            "consistency": float(cons), "median_px": med,
                            "minutes": round((time.time() - started) / 60.0, 2)})
            print(f"  step {step:5d}  loss {float(loss):7.4f}  heat {float(heat_loss):7.4f}  "
                  f"med {'-' if med is None else round(med, 2)} px  "
                  f"realKL {float(real_kl):7.4f}  {history[-1]['minutes']:5.1f} min",
                  flush=True)

    out_dir = M.REPO_ROOT / "weights/multiteacher_corner_distill_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"specialist_{arm}_last.pt"
    torch.save({"state_dict": model.state_dict(), "arm": arm, "weights": WEIGHTS,
                "seed": SEED, "updates_done": done,
                "n_parameters": sum(p.numel() for p in model.parameters()),
                "n_real_entries": len(real_entries)}, path)
    return path, history, len(real_entries), done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["C0", "C1"])
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--cap-minutes", type=float, default=60.0)
    args = parser.parse_args()
    print(f"=== {args.arm}  updates {args.updates}  cap {args.cap_minutes} min ===", flush=True)
    path, history, n_real, done = train(args.arm, args.updates, args.cap_minutes)
    rec = {"arm": args.arm, "checkpoint": str(path.relative_to(M.REPO_ROOT)),
           "checkpoint_sha256": M.sha256_file(path),
           "generated_utc": datetime.now(timezone.utc).isoformat(),
           "loss_weights": WEIGHTS, "seed": SEED,
           "syn_crops_per_step": SYN_CROPS_PER_STEP,
           "real_crops_per_step": REAL_CROPS_PER_STEP if args.arm == "C1" else 0,
           "requested_updates": args.updates, "updates_done": done,
           "cap_minutes": args.cap_minutes,
           "n_real_usable_keypoints": n_real,
           "checkpoint_selection": "last only",
           "history": history}
    out = M.GATE_C / f"TRAIN_{args.arm}.json"
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print(f"-> {out.relative_to(M.REPO_ROOT)}  ckpt {path.relative_to(M.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
