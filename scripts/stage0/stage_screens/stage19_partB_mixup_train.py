"""stage19_partB_mixup_train.py — domain-mixup pilot (2-arm), A-PASS branch.

DM-ADA core only (domain linear interpolation + soft supervision; NO adversarial
discriminator).  Continues from B2 (stage11_16k_B2_maskaux/final_ep0084).

  x_tilde = lam * x_sim + (1-lam) * x_real ,  lam ~ U(lambda_lo, 1.0)  (sim-heavy)
  belief target:
    accepted FRONT channels (GATE-A operating point peak>=0.9 & flip<=10):
        tgt = lam*H_sim + (1-lam)*pl_weight*H_realPL     (soft PL supervision)
    non-accepted / REAR channels:
        tgt = lam*H_sim   (real contributes nothing)
        AND belief loss is SPATIALLY MASKED to 0 inside the real pallet bbox for
        those channels only (rear is confidently-wrong -> no signal there; sim
        supervision elsewhere kept -> NO whole-channel masking).
  affinity target = sim affinity (unmixed); plain MSE (sim-supervised field).

Arms (identical sim data / order / schedule / lr; only the real blend differs):
  --mixup off  : control  (pure sim finetune from B2)
  --mixup on   : ~50% steps draw a real mixup partner

Reuses: CleanVisiiDopeLoader (sim beliefs+affinities, full fidelity),
CreateBeliefMap (real-PL belief gen, identical convention),
eval_pvnet_heads.load_pvnet_model (B2 seg-head weights), PART A records
(accepted-front PL per real frame — NO new teacher inference).

Safety: 2-iter smoke (--smoke) checks shape/NaN/OOM; NAN-GUARD kills on non-finite
loss; ckpt every --save_every steps + final.  Run via nohup DIRECT redirect (no tee).

Usage (GPU):
  conda run -n pallet-pose python scripts/stage0/stage19_partB_mixup_train.py \
      --mixup --out weights/stage_screens/stage19_mixup_pilot/mixup --steps 600
  (control: drop --mixup, --out .../control)
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import argparse
import glob
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path[:0] = [os.path.join(ROOT, "scripts", "data_prep", _s)
                for _s in ("plots", "filters")]
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

REC_FP = os.path.join(ROOT, "data", "pallet", "eval_results",
                      "stage19_conf_mixup", "partA_records.json")
NOAPRIL_RGB = os.path.join(ROOT, "data", "pallet", "raw_data",
                           "capture0403noapril", "rgb")
B2 = os.path.join(ROOT, "weights", "stage11_16k_B2_maskaux",
                  "final_net_epoch_0084.pth")
SIM_DIRS = [
    os.path.join(ROOT, "data", "pallet", "training_data", "mixed_v8_train"),
    os.path.join(ROOT, "challenge", "data", "training", "addon_v1_train"),
    os.path.join(ROOT, "challenge", "data", "training", "v3", "batch_000"),
    os.path.join(ROOT, "data", "pallet", "training_data", "aug_trunc"),
]
OUTPUT_SIZE = 50
IMG_SIZE = 400
SIGMA = 4.0
FRONT = [0, 1, 2, 3]
PEAK_ACCEPT = 0.9
FLIP_ACCEPT = 10.0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ── real-PL dataset (front-accepted PL from PART A records; rear-mask hull) ─
class RealPLDataset:
    """Deterministic real branch: resize 400, brightness jitter (NO flip/rotate),
    beliefs from accepted FRONT corners, rear-suppress hull mask (all corners)."""
    def __init__(self):
        import cv2  # noqa
        recs = json.load(open(REC_FP))["records"]
        self.items = []
        for r in recs:
            if r["is_gt"] or r["dom"] != "noapril":
                continue
            acc = {}
            for i in FRONT:
                if (r["kp9"][i] is not None and r["peak9"][i] >= PEAK_ACCEPT
                        and r["flip9"][i] is not None and r["flip9"][i] <= FLIP_ACCEPT):
                    acc[i] = r["kp9"][i]
            if not acc:
                continue
            det = [r["kp9"][i] for i in range(8) if r["kp9"][i] is not None]
            self.items.append({"fid": r["fid"], "acc": acc, "det": det})
        print(f"[RealPL] {len(self.items)} real frames with >=1 accepted front corner")

    def __len__(self):
        return len(self.items)

    def get(self, idx, rng):
        import cv2
        from utils_belief import CreateBeliefMap
        it = self.items[idx % len(self.items)]
        ip = os.path.join(NOAPRIL_RGB, it["fid"] + ".png")
        img = cv2.imread(ip)
        h0, w0 = img.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
        # brightness jitter (mild, no geometry)
        g = 1.0 + (rng.random() - 0.5) * 0.4
        rgb = np.clip(rgb.astype(np.float32) * g, 0, 255)
        t = ((rgb / 255.0 - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)
        sx, sy = IMG_SIZE / w0, IMG_SIZE / h0
        bs = IMG_SIZE / OUTPUT_SIZE   # 8: image->belief downscale
        # accepted-front kp in belief(50) coords
        pts9 = [[-1.0, -1.0]] * 9
        accept = np.zeros(9, np.float32)
        for i, (x, y) in it["acc"].items():
            pts9[i] = [x * sx / bs, y * sy / bs]
            accept[i] = 1.0
        bel = np.array(CreateBeliefMap(size=OUTPUT_SIZE, pointsBelief=[pts9],
                                       nbpoints=9, sigma=SIGMA), np.float32)
        # rear-suppress region: convex hull of ALL detected corners (belief coords)
        rear = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), np.float32)
        det = np.array([[x * sx / bs, y * sy / bs] for x, y in it["det"]], np.float32)
        if len(det) >= 3:
            hull = cv2.convexHull(det.astype(np.int32))
            cv2.fillConvexPoly(rear, hull, 1.0)
            rear = cv2.dilate(rear, np.ones((3, 3), np.uint8))
        return t, bel, accept, rear


def load_batch_real(ds, bs, rng):
    import torch
    T, B, A, R = [], [], [], []
    for _ in range(bs):
        t, bel, acc, rear = ds.get(rng.randrange(len(ds)), rng)
        T.append(t); B.append(bel); A.append(acc); R.append(rear)
    return (torch.from_numpy(np.stack(T)), torch.from_numpy(np.stack(B)),
            torch.from_numpy(np.stack(A)), torch.from_numpy(np.stack(R)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixup", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=B2)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pl_weight", type=float, default=0.3)
    ap.add_argument("--lambda_lo", type=float, default=0.5)
    ap.add_argument("--mixup_prob", type=float, default=0.5)
    ap.add_argument("--save_every", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2657)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from torch.utils.data import DataLoader
    from utils_dataset import CleanVisiiDopeLoader
    from eval_pvnet_heads import load_pvnet_model

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim_ds = CleanVisiiDopeLoader([d for d in SIM_DIRS if os.path.isdir(d)],
                                  sigma=SIGMA, output_size=OUTPUT_SIZE,
                                  objects=["pallet"])
    sim_loader = DataLoader(sim_ds, batch_size=args.batch, shuffle=True,
                            num_workers=4, drop_last=True)
    print(f"[sim] {len(sim_ds)} images from {len([d for d in SIM_DIRS if os.path.isdir(d)])} dirs")
    real_ds = RealPLDataset() if args.mixup else None

    model, numVec, numSeg = load_pvnet_model(args.init, device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[init] {args.init}  mixup={args.mixup}  steps={args.steps}")

    def masked_belief_loss(out_bel_list, tgt, wmap):
        L = 0.0
        for stage in out_bel_list:
            d = (stage - tgt) ** 2
            L = L + (wmap * d).sum() / wmap.sum().clamp(min=1.0)
        return L

    def aff_loss(out_aff_list, tgt):
        L = 0.0
        for stage in out_aff_list:
            L = L + ((stage - tgt) ** 2).mean()
        return L

    step = 0
    log = []
    done = False
    while not done:
        for targets in sim_loader:
            img_sim = targets["img"].to(device)
            bel_sim = targets["beliefs"].to(device).float()
            aff_sim = targets["affinities"].to(device).float()
            B = img_sim.shape[0]

            use_mix = args.mixup and (rng.random() < args.mixup_prob)
            if use_mix:
                ir, br, ar, rr = load_batch_real(real_ds, B, rng)
                ir = ir.to(device); br = br.to(device); ar = ar.to(device)
                rr = rr.to(device)
                lam = (args.lambda_lo + (1 - args.lambda_lo)
                       * torch.rand(B, 1, 1, 1, device=device))
                data = lam * img_sim + (1 - lam) * ir
                lam_c = lam.view(B, 1, 1, 1)          # broadcast to (B,9,H,W)
                acc_c = ar.view(B, 9, 1, 1)
                tgt_bel = (lam_c * bel_sim
                           + (1 - lam_c) * args.pl_weight * br * acc_c)
                # weight: 0 inside real pallet bbox for NON-accepted channels only
                rear_b = rr.view(B, 1, OUTPUT_SIZE, OUTPUT_SIZE)
                wmap = 1.0 - (1.0 - acc_c) * rear_b      # (B,9,H,W)
                tgt_aff = aff_sim
            else:
                data = img_sim
                tgt_bel = bel_sim
                wmap = torch.ones_like(bel_sim)
                tgt_aff = aff_sim

            opt.zero_grad()
            out = model(data)
            out_bel, out_aff = out[0], out[1]
            lb = masked_belief_loss(out_bel, tgt_bel, wmap)
            la = aff_loss(out_aff, tgt_aff)
            loss = lb + la
            if not torch.isfinite(loss):
                raise RuntimeError(f"[NAN-GUARD] non-finite loss at step {step}: {loss}")
            loss.backward()
            opt.step()

            if step % 25 == 0:
                pk = out_bel[-1][:, :9].reshape(B, 9, -1).max(-1).values.mean().item()
                print(f"[step {step}] loss={loss.item():.4f} bel={lb.item():.4f} "
                      f"aff={la.item():.4f} mix={int(use_mix)} peak={pk:.3f}", flush=True)
                log.append({"step": step, "loss": float(loss.item()),
                            "bel": float(lb.item()), "aff": float(la.item()),
                            "mix": int(use_mix), "peak": float(pk)})
            step += 1
            if args.smoke and step >= 2:
                print("[SMOKE] 2 iters finite, shapes ok:",
                      tuple(out_bel[-1].shape), tuple(tgt_bel.shape))
                return
            if step % args.save_every == 0:
                torch.save(model.state_dict(),
                           os.path.join(args.out, f"net_step_{step:05d}.pth"))
                print(f"[ckpt] step {step}", flush=True)
            if step >= args.steps:
                done = True
                break

    fp = os.path.join(args.out, "final_net.pth")
    torch.save(model.state_dict(), fp)
    json.dump({"args": vars(args), "log": log}, open(
        os.path.join(args.out, "train_log.json"), "w"), indent=2)
    print(f"[done] {fp}")


if __name__ == "__main__":
    main()
