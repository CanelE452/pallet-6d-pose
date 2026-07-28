"""stage20_cutpaste_train.py — cut-paste domain-adaptation pilot (2-arm).

Replaces STAGE19-B's PIXEL domain-mixup (which ghosted -> real regression) with
CUT-PASTE: a synthetic pallet is segmented (mask_rle, or projected-cuboid hull
fallback) and composited onto a REAL background.  Foreground pallet keeps its
SYNTHETIC GT verbatim (heatmap + affinity), so label noise = 0; only the
BACKGROUND appearance becomes real (real floor / clutter / lighting).  This
isolates the sim2real APPEARANCE gap with zero ghosting.

Continues from B2 (stage11_16k_B2_maskaux/final_ep0084).

Arms (identical sim source / order / schedule / lr / seed; only treatment differs):
  --cutpaste off : control  (pure sim finetune from B2; bit-identical to STAGE19
                   control -- cutpaste off => rng untouched => same torch order)
  --cutpaste on  : ~50% of steps use a full cut-paste batch (sim pallet on real bg)

CUT-PASTE batch:  data = alpha*sim + (1-alpha)*real_bg
  alpha = feathered pallet mask (interior=1, ~2px soft edge)
  belief/affinity target = synthetic GT (built with the SAME utils_belief
  CreateBeliefMap / GenerateMapAffinity the main loader uses -> no convention drift)
  wmap = ones (no PL, no rear-masking; the whole label is clean synthetic GT)

Safety: 2-iter smoke (--smoke) checks shape/NaN/OOM; NAN-GUARD kills on non-finite
loss; ckpt every --save_every steps + final.  Run via nohup DIRECT redirect (no tee).

Usage (GPU):
  conda run -n pallet-pose python scripts/stage0/stage20_cutpaste_train.py \
      --cutpaste --out weights/stage20_cutpaste_pilot/cutpaste --steps 600
  (control: drop --cutpaste, --out .../control)
  (examples: --dump_examples 5 --out data/pallet/eval_results/stage20_cutpaste)
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

B2 = os.path.join(ROOT, "weights", "stage11_16k_B2_maskaux",
                  "final_net_epoch_0084.pth")
# real background pools (real floor/clutter/lighting; pallets present -> covered
# by the pasted synthetic pallet at its original position where they overlap)
REAL_BG_DIRS = [
    os.path.join(ROOT, "data", "pallet", "raw_data", "capture0403noapril", "rgb"),
    os.path.join(ROOT, "data", "pallet", "pl", "c0403_paper_base"),
]
SIM_DIRS = [
    os.path.join(ROOT, "data", "pallet", "training_data", "mixed_v8_train"),
    os.path.join(ROOT, "challenge", "data", "training", "addon_v1_train"),
    os.path.join(ROOT, "challenge", "data", "training", "v3", "batch_000"),
    os.path.join(ROOT, "data", "pallet", "training_data", "aug_trunc"),
]
# Cut source = ONLY dirs whose every object carries mask_rle (precise pallet
# silhouette -> no sim-background bleed through fork gaps).  mixed_v8 / aug_trunc
# lack mask_rle (hull fallback bleeds sim bg) so they are excluded from cutting
# (still used by control's full-sim finetune, unchanged).
CUT_SIM_DIRS = [
    os.path.join(ROOT, "challenge", "data", "training", "addon_v1_train"),
    os.path.join(ROOT, "challenge", "data", "training", "v3", "batch_000"),
]
OUTPUT_SIZE = 50
IMG_SIZE = 400
SIGMA = 4.0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _list_sim_frames(dirs=CUT_SIM_DIRS):
    frames = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for jp in sorted(glob.glob(os.path.join(d, "*.json"))):
            if jp.endswith(".orig") or ".face_centers." in jp:
                continue
            ip = jp[:-5] + ".png"
            if os.path.exists(ip):
                frames.append((ip, jp))
    return frames


def _list_real_bg():
    bg = []
    for d in REAL_BG_DIRS:
        if os.path.isdir(d):
            bg += sorted(glob.glob(os.path.join(d, "*.png")))
    return bg


class CutPasteDataset:
    """Self-contained: sim pallet (squash-400) cut by mask_rle/hull, composited
    onto a real background (squash-400).  Belief/affinity from synthetic GT via
    the exact utils_belief generators (no convention drift).  No albumentations
    geometry (keeps mask<->image<->GT alignment trivially exact)."""

    def __init__(self):
        import cv2  # noqa
        self.sim = _list_sim_frames()
        self.bg = _list_real_bg()
        if not self.sim:
            raise RuntimeError("no sim frames found")
        if not self.bg:
            raise RuntimeError("no real background frames found")
        print(f"[CutPaste] {len(self.sim)} sim frames, {len(self.bg)} real bg frames")

    def __len__(self):
        return len(self.sim)

    def _load_mask400(self, obj0, kp9_400, w0, h0):
        import cv2
        from utils_pvnet import decode_mask_rle, make_cuboid_mask
        if "mask_rle" in obj0:
            m = decode_mask_rle(obj0["mask_rle"]).astype(np.uint8)
            if m.shape[:2] != (h0, w0):
                m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_NEAREST)
            m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            if m.sum() < 20:  # degenerate/empty rle -> hull fallback
                m = make_cuboid_mask(kp9_400, IMG_SIZE)
            return m.astype(np.float32)
        return make_cuboid_mask(kp9_400, IMG_SIZE).astype(np.float32)

    def get(self, idx, rng, return_vis=False):
        import cv2
        from utils_belief import CreateBeliefMap, GenerateMapAffinity
        ip, jp = self.sim[idx % len(self.sim)]
        img = cv2.imread(ip)
        h0, w0 = img.shape[:2]
        sim = cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
        data_json = json.load(open(jp))
        obj0 = data_json["objects"][0]
        cub = obj0["projected_cuboid"]                       # 8 x [x,y] orig px
        ctr = obj0["projected_cuboid_centroid"]              # [x,y] orig px
        kp9_orig = np.array(cub + [ctr], np.float64)         # (9,2)
        sx, sy = IMG_SIZE / w0, IMG_SIZE / h0
        kp9_400 = kp9_orig * np.array([sx, sy])              # image(400) coords
        bx, by = OUTPUT_SIZE / w0, OUTPUT_SIZE / h0
        kp9_50 = (kp9_orig * np.array([bx, by])).tolist()    # belief(50) coords

        mask = self._load_mask400(obj0, kp9_400, w0, h0)     # (400,400) {0,1}
        # feather: soft ~2px edge, interior stays ~1
        alpha = cv2.GaussianBlur(mask, (5, 5), 0)
        alpha = np.clip(np.maximum(alpha, mask), 0, 1)[..., None].astype(np.float32)

        bgp = self.bg[rng.randrange(len(self.bg))]
        bgimg = cv2.imread(bgp)
        bg = cv2.cvtColor(cv2.resize(bgimg, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)

        comp = (alpha * sim.astype(np.float32)
                + (1 - alpha) * bg.astype(np.float32))       # (400,400,3) 0-255
        # mild brightness jitter (match real branch spirit; no geometry)
        g = 1.0 + (rng.random() - 0.5) * 0.3
        comp = np.clip(comp * g, 0, 255)

        t = ((comp / 255.0 - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)
        bel = np.array(CreateBeliefMap(size=OUTPUT_SIZE, pointsBelief=[kp9_50],
                                       nbpoints=9, sigma=SIGMA), np.float32)
        aff = GenerateMapAffinity(size=OUTPUT_SIZE, nb_vertex=8,
                                  pointsInterest=[kp9_50],
                                  objects_centroid=[kp9_50[-1]], scale=1)
        aff = np.clip(aff.numpy().astype(np.float32), -1, 1)
        bel = np.clip(bel, 0, 1)
        if return_vis:
            return t, bel, aff, comp.astype(np.uint8), kp9_400, os.path.basename(ip)
        return t, bel, aff


def load_batch_cut(ds, bs, rng):
    import torch
    T, B, A = [], [], []
    for _ in range(bs):
        t, bel, aff = ds.get(rng.randrange(len(ds)), rng)
        T.append(t); B.append(bel); A.append(aff)
    return (torch.from_numpy(np.stack(T)), torch.from_numpy(np.stack(B)),
            torch.from_numpy(np.stack(A)))


def dump_examples(n, out_dir):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    ds = CutPasteDataset()
    rng = random.Random(7)
    E = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
         (255, 0, 255), (0, 255, 255), (128, 128, 255), (255, 128, 0)]
    for k in range(n):
        idx = rng.randrange(len(ds))
        _, _, _, comp, kp9_400, name = ds.get(idx, rng, return_vis=True)
        vis = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR).copy()
        for i in range(8):
            x, y = int(kp9_400[i, 0]), int(kp9_400[i, 1])
            cv2.circle(vis, (x, y), 4, E[i], -1)
            cv2.putText(vis, str(i), (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, E[i], 1)
        cx, cy = int(kp9_400[8, 0]), int(kp9_400[8, 1])
        cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 10, 2)
        fp = os.path.join(out_dir, f"cutpaste_example_{k:02d}.jpg")
        cv2.imwrite(fp, vis)
        print(f"[dump] {fp}  (src {name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutpaste", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=B2)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cutpaste_prob", type=float, default=0.5)
    ap.add_argument("--save_every", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2657)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dump_examples", type=int, default=0)
    args = ap.parse_args()

    if args.dump_examples > 0:
        dump_examples(args.dump_examples, args.out)
        return

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
    print(f"[sim] {len(sim_ds)} images from "
          f"{len([d for d in SIM_DIRS if os.path.isdir(d)])} dirs")
    cut_ds = CutPasteDataset() if args.cutpaste else None

    model, numVec, numSeg = load_pvnet_model(args.init, device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[init] {args.init}  cutpaste={args.cutpaste}  steps={args.steps}")

    def belief_loss(out_bel_list, tgt):
        L = 0.0
        for stage in out_bel_list:
            L = L + ((stage - tgt) ** 2).mean()
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
            use_cut = args.cutpaste and (rng.random() < args.cutpaste_prob)
            if use_cut:
                B = args.batch
                data, tgt_bel, tgt_aff = load_batch_cut(cut_ds, B, rng)
                data = data.to(device); tgt_bel = tgt_bel.to(device)
                tgt_aff = tgt_aff.to(device)
            else:
                data = targets["img"].to(device)
                tgt_bel = targets["beliefs"].to(device).float()
                tgt_aff = targets["affinities"].to(device).float()
                B = data.shape[0]

            opt.zero_grad()
            out = model(data)
            out_bel, out_aff = out[0], out[1]
            lb = belief_loss(out_bel, tgt_bel)
            la = aff_loss(out_aff, tgt_aff)
            loss = lb + la
            if not torch.isfinite(loss):
                raise RuntimeError(f"[NAN-GUARD] non-finite loss at step {step}: {loss}")
            loss.backward()
            opt.step()

            if step % 25 == 0:
                pk = out_bel[-1][:, :9].reshape(B, 9, -1).max(-1).values.mean().item()
                print(f"[step {step}] loss={loss.item():.4f} bel={lb.item():.4f} "
                      f"aff={la.item():.4f} cut={int(use_cut)} peak={pk:.3f}", flush=True)
                log.append({"step": step, "loss": float(loss.item()),
                            "bel": float(lb.item()), "aff": float(la.item()),
                            "cut": int(use_cut), "peak": float(pk)})
            step += 1
            if args.smoke and step >= 2:
                print("[SMOKE] 2 iters finite, shapes ok:",
                      tuple(out_bel[-1].shape), tuple(tgt_bel.shape),
                      tuple(out_aff[-1].shape), tuple(tgt_aff.shape))
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
