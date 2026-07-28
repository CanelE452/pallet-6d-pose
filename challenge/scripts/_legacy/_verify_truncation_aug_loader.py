#!/usr/bin/env python3
"""Verify on-the-fly truncation augmentation in CleanVisiiDopeLoader.

Checks:
  (a) clean:truncation ratio over N samples ~= 40:60 at prob=0.6
  (b) truncation samples: belief(50x50) supervises all 9 corners (incl. the
      corners that were off-image before crop/pad)
  (c) saves a few overlay images
  (d) prob=0 sanity: behaves like the original clean loader

Run: conda activate pallet-pose
  python challenge/scripts/_verify_truncation_aug_loader.py
"""
import os
import sys
import glob
import random

import numpy as np
import cv2
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

from utils_dataset import CleanVisiiDopeLoader  # noqa: E402

OUT_DIR = os.path.join(ROOT, "challenge", "data", "_verify_truncation_aug")
N = 200


def find_src_dirs():
    dirs = []
    for v in ("v1", "v2"):
        dirs += sorted(glob.glob(os.path.join(
            ROOT, "challenge", "data", "training", v,
            "part_*", f"train_palletobj_{v}")))
    return dirs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    src_dirs = find_src_dirs()
    print(f"source dirs: {len(src_dirs)}")

    # --- prob=0.6 loader ---
    ds = CleanVisiiDopeLoader(
        src_dirs, objects=["pallet"], sigma=4.0, output_size=50,
        truncation_aug_prob=0.6,
    )
    print(f"dataset size: {len(ds)}")

    rng = random.Random(0)
    idxs = [rng.randrange(len(ds)) for _ in range(N)]

    # The loader does not expose the per-sample applied_truncation flag in its
    # return dict, so we monkey-patch the module-level apply_truncation_aug to
    # record whether each __getitem__ actually performed a truncation.
    _run_instrumented(ds, idxs)


def _run_instrumented(ds, idxs):
    """Re-run sampling but capture the applied_truncation flag by patching the
    module-level apply_truncation_aug to record calls per index."""
    import utils_dataset as U

    calls = {"applied_idx": set()}
    orig = U.apply_truncation_aug

    def wrapped(img, kps9, rng):
        res = orig(img, kps9, rng)
        wrapped._last = res is not None
        return res
    wrapped._last = False

    # We need to know which index triggered an *attempted* aug and whether it
    # succeeded. Patch __getitem__ indirectly is complex; instead we replicate
    # the loader's decision logic here deterministically is not possible without
    # worker_info. Simplest robust method: set truncation_aug_prob=1.0 to force
    # an attempt on every eligible frame, measure success rate + supervision;
    # then separately trust the Bernoulli(0.6) gate for the ratio.
    print("  forcing prob=1.0 to measure truncation success + belief coverage")
    ds.truncation_aug_prob = 1.0
    U.apply_truncation_aug = wrapped

    n_attempt_success = 0
    n_attempt_fail = 0
    sup_counts = []
    saved = 0
    os.makedirs(OUT_DIR, exist_ok=True)
    for idx in idxs:
        wrapped._last = False
        sample = ds[idx]
        if wrapped._last:
            n_attempt_success += 1
            bel = sample["beliefs"].numpy()
            sup = int(np.sum([bel[c].max() > 1e-4 for c in range(9)]))
            sup_counts.append(sup)
            if saved < 8:
                _save_overlay(sample, idx, saved)
                saved += 1
        else:
            n_attempt_fail += 1

    U.apply_truncation_aug = orig
    sc = np.array(sup_counts)
    print(f"\n=== truncation (forced prob=1.0) ===")
    print(f"  attempts: {len(idxs)}  success: {n_attempt_success}  "
          f"fail(retry-exhausted): {n_attempt_fail}")
    print(f"  belief supervised channels: min={sc.min()} median={int(np.median(sc))} "
          f"max={sc.max()} (of 9)")
    print(f"  fraction with 9/9 supervised: "
          f"{100.0*np.mean(sc==9):.1f}%")

    # --- ratio check at prob=0.6 (Bernoulli gate) ---
    print(f"\n=== clean:truncation ratio (prob=0.6) ===")
    ds.truncation_aug_prob = 0.6
    n_trunc = 0
    for idx in idxs:
        wrapped._last = False
        U.apply_truncation_aug = wrapped
        _ = ds[idx]
        if wrapped._last:
            n_trunc += 1
    U.apply_truncation_aug = orig
    n = len(idxs)
    print(f"  N={n}  truncation={n_trunc} ({100.0*n_trunc/n:.1f}%)  "
          f"clean={n-n_trunc} ({100.0*(n-n_trunc)/n:.1f}%)  target=60:40")

    # --- prob=0 sanity ---
    print(f"\n=== prob=0 sanity ===")
    ds.truncation_aug_prob = 0.0
    n_trunc0 = 0
    for idx in idxs:
        wrapped._last = False
        U.apply_truncation_aug = wrapped
        _ = ds[idx]
        if wrapped._last:
            n_trunc0 += 1
    U.apply_truncation_aug = orig
    print(f"  truncation applied at prob=0: {n_trunc0} (must be 0)")
    print(f"\noverlays -> {OUT_DIR}")


def _save_overlay(sample, idx, k):
    """Overlay belief peaks + input (50x50 belief upscaled, 400x400 input)."""
    img = sample["img_original"].numpy().transpose(1, 2, 0)  # 400x400x3 RGB [0,1]
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bel = sample["beliefs"].numpy()  # (9,50,50)
    H, W = img.shape[:2]
    colors = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 128),
              (0, 255, 0), (255, 255, 0), (255, 128, 0), (255, 0, 0),
              (255, 0, 255)]
    pts = []
    for c in range(9):
        m = bel[c]
        if m.max() <= 1e-4:
            pts.append(None)
            continue
        yx = np.unravel_index(np.argmax(m), m.shape)
        px = int(yx[1] / 50.0 * W)
        py = int(yx[0] / 50.0 * H)
        pts.append((px, py))
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], (0, 200, 0), 1)
    sup = 0
    for c, p in enumerate(pts):
        if p:
            sup += 1
            cv2.circle(img, p, 4, colors[c], -1)
            cv2.putText(img, str(c), (p[0] + 3, p[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[c], 1)
    cv2.putText(img, f"idx={idx} sup={sup}/9", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    fn = os.path.join(OUT_DIR, f"trunc_{k:02d}_idx{idx}.png")
    cv2.imwrite(fn, img)


if __name__ == "__main__":
    main()
