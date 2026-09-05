"""GATE C — 국소 코너 전문가 파일럿 학습.

    C0  SYN_LOCAL            합성 정확 감독만
    C1  SYN_PLUS_REAL_SOFT   + high-consensus real soft distribution + photometric consistency

동일 init(seed) · 동일 synthetic 노출 · 동일 optimizer updates.
checkpoint 는 last only.  DEV_EVAL 로 epoch 을 고르지 않는다.
loss 가중치는 실행 전에 고정했고 결과를 보고 바꾸지 않는다 (아래 WEIGHTS).

★ 2026-09-05 재작성 사유 (계약 변경 없음, 실행 방식만)
   최초판은 step 마다 840x680 PNG 를 6장 디코딩해 crop 을 뽑았다.  55,980 장을
   섞어 콜드로 읽으니 3.1 초/step 이었고, 77 분 뒤 CUDA 동기화 지점에서 멈췄다
   (GPU 0% · IO 0 · 유저공간 100% 스핀).  캡은 step **안**에서 멈춰 발동하지 못했다.
   그래서
     - crop 을 `build_crop_bank.py` 로 미리 뽑아 메모리에서 학습한다
     - target 을 numpy 대신 GPU 에서 만든다
     - 하트비트 파일과 주기 체크포인트를 남겨 진행을 산출물로 판정한다
     - 캡을 step 시작과 끝 양쪽에서 본다
   구조 · 5,000 update · 60 분 캡 · jitter 분포 · arm 구성 · last checkpoint 는 그대로다.
"""
from __future__ import annotations

import os

# ★ torch/numpy import 전에 CPU 스레드를 1 로 묶는다.  최초 실행에서 GPU 0% ·
#   IO 0 인데 메인 스레드가 100% 로 스핀하며 77 분을 태웠고, 매핑된 libgomp 가
#   유일한 단서였다.  이 트랙의 배치는 작아 intra-op 병렬이 필요 없다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
from mtcd_specialist import (CROP, HEATMAP_SIGMA, LocalCornerSpecialist,
                             extract_crop)

WEIGHTS = {"heat": 1.0, "visibility": 0.2, "uncertainty": 0.1, "edge": 0.1,
           "real_kl": 1.0, "consistency": 0.5}
SYN_CROPS_PER_STEP = 48
REAL_CROPS_PER_STEP = 16
MIXTURE_SIGMA = 3.0
SEED = 20260905
HEARTBEAT_EVERY = 100
CHECKPOINT_EVERY = 1000
STEP_STALL_SECONDS = 120


def gaussian_maps(xy: torch.Tensor, sigma: float) -> torch.Tensor:
    """(B,2) 좌표 -> (B, CROP*CROP) 정규화 가우시안. GPU 에서 만든다."""
    ax = torch.arange(CROP, device=xy.device, dtype=xy.dtype)
    gx = torch.exp(-((ax[None, :] - xy[:, 0:1]) ** 2) / (2 * sigma ** 2))
    gy = torch.exp(-((ax[None, :] - xy[:, 1:2]) ** 2) / (2 * sigma ** 2))
    g = gy[:, :, None] * gx[:, None, :]
    g = g.reshape(xy.shape[0], -1)
    return g / g.sum(1, keepdim=True).clamp_min(1e-12)


def mixture_maps(points: torch.Tensor, mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """(B,T,2) 교사 좌표들의 등가중 가우시안 혼합. hard coordinate 를 쓰지 않는다."""
    b, t, _ = points.shape
    flat = gaussian_maps(points.reshape(b * t, 2), sigma).reshape(b, t, -1)
    w = mask.to(flat.dtype).unsqueeze(-1)
    acc = (flat * w).sum(1)
    return acc / acc.sum(1, keepdim=True).clamp_min(1e-12)


def load_bank(split):
    z = np.load(M.AUDIT / f"CROP_BANK_{split}.npz")
    return {k: z[k] for k in ("patches", "kp", "local", "inside", "edge", "has_edge")}


def photometric_batch(x: torch.Tensor, rng: np.random.Generator, strong: bool):
    """기하를 바꾸지 않는 증강만. flip·rotation·crop transform 금지."""
    b = x.shape[0]
    dev = x.device
    if strong:
        gain = torch.tensor(rng.uniform(0.55, 1.5, (b, 1, 1, 1)), device=dev, dtype=x.dtype)
        bias = torch.tensor(rng.uniform(-0.18, 0.18, (b, 1, 1, 1)), device=dev, dtype=x.dtype)
        tint = torch.tensor(rng.uniform(0.85, 1.15, (b, 3, 1, 1)), device=dev, dtype=x.dtype)
        y = x * gain * tint + bias
        noise = torch.tensor(rng.uniform(0.005, 0.06, (b, 1, 1, 1)), device=dev, dtype=x.dtype)
        y = y + torch.randn_like(y) * noise
    else:
        gain = torch.tensor(rng.uniform(0.9, 1.1, (b, 1, 1, 1)), device=dev, dtype=x.dtype)
        bias = torch.tensor(rng.uniform(-0.03, 0.03, (b, 1, 1, 1)), device=dev, dtype=x.dtype)
        y = x * gain + bias
    return y.clamp(0.0, 1.0)


def build_real_bank(device):
    """usable real keypoint 를 한 번만 crop 으로 뽑아 둔다."""
    cached = json.loads((M.GATE_C / "TARGET_TEACHER_CACHE.json").read_text())
    patches, kps, teacher_pts, masks = [], [], [], []
    max_t = 0
    entries = []
    for f in cached["frames"].values():
        for kp in f["keypoints"]:
            if kp.get("usable"):
                entries.append((f["image_path"], kp["kp"],
                                f["r0_keypoints_xy"][kp["kp"]], kp["teacher_xy"]))
                max_t = max(max_t, len(kp["teacher_xy"]))
    image_cache = {}
    for path, k, r0_xy, tpts in entries:
        img = image_cache.get(path)
        if img is None:
            img = cv2.imread(str(M.REPO_ROOT / path))
            if img is None:
                continue
            if len(image_cache) < 400:
                image_cache[path] = img
        patch, origin = extract_crop(img, np.asarray(r0_xy, float))
        local = np.asarray(tpts, float) - origin
        keep = np.isfinite(local).all(1) & (local[:, 0] > -2) & (local[:, 0] < CROP + 2) \
            & (local[:, 1] > -2) & (local[:, 1] < CROP + 2)
        if keep.sum() == 0:
            continue
        padded = np.zeros((max_t, 2), np.float32)
        mask = np.zeros(max_t, bool)
        sel = local[keep][:max_t]
        padded[:len(sel)] = sel
        mask[:len(sel)] = True
        patches.append(patch)
        kps.append(k)
        teacher_pts.append(padded)
        masks.append(mask)
    if not patches:
        return None
    return {
        "patches": torch.from_numpy(np.stack(patches).astype(np.float32) / 255.0
                                    ).permute(0, 3, 1, 2).to(device),
        "kp": torch.tensor(kps, device=device),
        "points": torch.from_numpy(np.stack(teacher_pts)).to(device),
        "mask": torch.from_numpy(np.stack(masks)).to(device),
        "n": len(patches),
    }


def train(arm, updates, cap_minutes, device="cuda:0"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)

    bank = load_bank("train")
    n_bank = len(bank["kp"])
    # crop 은 CPU 에 uint8 로 둔다 — float32 로 GPU 에 다 올리면 4 GB 가 넘는다.
    patches_cpu = torch.from_numpy(bank["patches"])            # (N, 64, 64, 3) uint8
    kp_all = torch.from_numpy(bank["kp"]).to(device)
    local_all = torch.from_numpy(bank["local"]).to(device)
    inside_all = torch.from_numpy(bank["inside"]).to(device)
    edge_all = torch.from_numpy(bank["edge"]).to(device)
    hasedge_all = torch.from_numpy(bank["has_edge"]).to(device)
    print(f"  crop bank: {n_bank} crops, "
          f"{patches_cpu.element_size()*patches_cpu.nelement()/1e9:.2f} GB (CPU uint8)",
          flush=True)

    real = build_real_bank(device) if arm == "C1" else None
    if arm == "C1":
        print(f"  real usable crops: {0 if real is None else real['n']}", flush=True)

    model = LocalCornerSpecialist().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates)
    ckpt_dir = M.REPO_ROOT / "weights/multiteacher_corner_distill_v1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"specialist_{arm}_last.pt"
    beat = M.GATE_C / f"HEARTBEAT_{arm}.json"

    def save(step):
        torch.save({"state_dict": model.state_dict(), "arm": arm, "weights": WEIGHTS,
                    "seed": SEED, "updates_done": step,
                    "n_parameters": sum(p.numel() for p in model.parameters()),
                    "n_real_entries": 0 if real is None else real["n"]}, path)

    started = time.time()
    history, done, stalled = [], 0, None
    for step in range(updates):
        step_started = time.time()
        if (step_started - started) / 60.0 > cap_minutes:
            print(f"  wall-clock cap reached at step {step}", flush=True)
            break
        idx_cpu = torch.from_numpy(rng.integers(0, n_bank, SYN_CROPS_PER_STEP))
        idx = idx_cpu.to(device)
        crops = (patches_cpu[idx_cpu].to(device, non_blocking=True)
                 .permute(0, 3, 1, 2).float().div_(255.0))
        ids = kp_all[idx]
        local, inside = local_all[idx], inside_all[idx]
        out = model(crops, ids)

        tgt = gaussian_maps(local, HEATMAP_SIGMA)
        logp = F.log_softmax(out["heat_logits"], dim=1)
        heat_loss = (-(tgt * logp).sum(1)[inside].mean() if inside.any()
                     else torch.zeros((), device=device))
        vis_loss = F.binary_cross_entropy_with_logits(out["visibility_logit"],
                                                      inside.float())
        with torch.no_grad():
            am = out["heat_logits"].argmax(1)
            pred_xy = torch.stack([(am % CROP).float(), (am // CROP).float()], 1)
            actual = torch.linalg.norm(pred_xy - local, dim=1)
        unc_loss = (F.l1_loss(F.softplus(out["log_uncertainty"])[inside], actual[inside])
                    if inside.any() else torch.zeros((), device=device))
        pe = F.normalize(out["edge_dirs"], dim=2)
        he = hasedge_all[idx]
        edge_loss = (((1 - (pe * edge_all[idx]).sum(2)).mean(1) * he).sum()
                     / he.sum().clamp_min(1))

        loss = (WEIGHTS["heat"] * heat_loss + WEIGHTS["visibility"] * vis_loss +
                WEIGHTS["uncertainty"] * unc_loss + WEIGHTS["edge"] * edge_loss)
        real_kl = cons = torch.zeros((), device=device)

        if real is not None:
            ridx = torch.from_numpy(rng.integers(0, real["n"], REAL_CROPS_PER_STEP)).to(device)
            rc, rid = real["patches"][ridx], real["kp"][ridx]
            weak = photometric_batch(rc, rng, False)
            strong = photometric_batch(rc, rng, True)
            ow, os_ = model(weak, rid), model(strong, rid)
            q = mixture_maps(real["points"][ridx], real["mask"][ridx], MIXTURE_SIGMA)
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

        step_seconds = time.time() - step_started
        if step_seconds > STEP_STALL_SECONDS:
            stalled = {"step": step, "seconds": step_seconds}
            print(f"  step {step} took {step_seconds:.0f}s — 정지로 보고 중단한다", flush=True)
            break
        if (time.time() - started) / 60.0 > cap_minutes:
            print(f"  wall-clock cap reached after step {step}", flush=True)
            break

        if step % HEARTBEAT_EVERY == 0 or step == updates - 1:
            beat.write_text(json.dumps({
                "arm": arm, "step": step, "updates": updates,
                "minutes": round((time.time() - started) / 60.0, 2),
                "steps_per_second": round((step + 1) / max(time.time() - started, 1e-6), 2),
                "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        if step % CHECKPOINT_EVERY == 0 and step > 0:
            save(done)
        if step % 250 == 0 or step == updates - 1:
            med = float(actual[inside].median()) if inside.any() else None
            history.append({"step": step, "loss": float(loss), "heat": float(heat_loss),
                            "vis": float(vis_loss), "unc": float(unc_loss),
                            "edge": float(edge_loss), "real_kl": float(real_kl),
                            "consistency": float(cons), "median_px": med,
                            "minutes": round((time.time() - started) / 60.0, 2)})
            print(f"  step {step:5d}  loss {float(loss):7.4f}  heat {float(heat_loss):7.4f}  "
                  f"med {'-' if med is None else round(med, 2)} px  "
                  f"realKL {float(real_kl):7.4f}  {history[-1]['minutes']:5.2f} min", flush=True)

    save(done)
    return path, history, (0 if real is None else real["n"]), done, stalled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["C0", "C1"])
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--cap-minutes", type=float, default=60.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(f"=== {args.arm}  updates {args.updates}  cap {args.cap_minutes} min "
          f"device {args.device} ===", flush=True)
    path, history, n_real, done, stalled = train(args.arm, args.updates,
                                                 args.cap_minutes, args.device)
    rec = {"arm": args.arm, "checkpoint": str(path.relative_to(M.REPO_ROOT)),
           "checkpoint_sha256": M.sha256_file(path),
           "generated_utc": datetime.now(timezone.utc).isoformat(),
           "loss_weights": WEIGHTS, "seed": SEED, "device": args.device,
           "crop_bank": "audit/CROP_BANK_train.npz",
           "syn_crops_per_step": SYN_CROPS_PER_STEP,
           "real_crops_per_step": REAL_CROPS_PER_STEP if args.arm == "C1" else 0,
           "requested_updates": args.updates, "updates_done": done,
           "cap_minutes": args.cap_minutes, "stalled": stalled,
           "n_real_usable_keypoints": n_real,
           "checkpoint_selection": "last only",
           "history": history}
    out = M.GATE_C / f"TRAIN_{args.arm}.json"
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print(f"-> {out.relative_to(M.REPO_ROOT)}  ckpt {path.relative_to(M.REPO_ROOT)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
