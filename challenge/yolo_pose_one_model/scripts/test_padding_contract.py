"""Unit tests for the padding contract. Training must not start unless these pass.

1. Round trip
     original -> +100 shift -> normalise by padded size -> denormalise -> -100 shift
   must return the original coordinate within 1e-4 px.

2. PnP coordinate-system equivalence
     (original coords, original K)  vs  (padded coords, padded K with cx+100, cy+100)
   must give the same pose. Mixing the two systems must NOT give the same pose - that
   negative control is checked too, otherwise the test could pass vacuously.

Run:
  python challenge/yolo_pose_one_model/scripts/test_padding_contract.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "challenge/yolo_pose_one_model/scripts"))
from verify_kp_contract import model_points  # noqa: E402

PAD = 100
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAILURES.append(name)


def test_roundtrip():
    print("\n[1] padding round trip (1e-4 px)")
    rng = np.random.default_rng(42)
    worst = 0.0
    for (w, h) in [(640, 480), (960, 540), (720, 480), (560, 560), (1280, 720)]:
        pts = rng.uniform(0, [w, h], size=(500, 2))
        pw, ph = w + 2 * PAD, h + 2 * PAD
        shifted = pts + PAD
        norm = shifted / [pw, ph]                      # what the label stores
        back = norm * [pw, ph] - PAD                   # what inference undoes
        err = float(np.abs(back - pts).max())
        worst = max(worst, err)
        check(f"{w}x{h}", err <= 1e-4, f"max err {err:.2e} px")
    check("overall", worst <= 1e-4, f"worst {worst:.2e} px")


def test_label_precision():
    """The label is written with %.6f, so quantisation is the real limit, not the maths."""
    print("\n[2] label text precision (%.6f) — informational")
    for (w, h) in [(640, 480), (960, 540)]:
        pw, ph = w + 2 * PAD, h + 2 * PAD
        step_x, step_y = pw * 1e-6, ph * 1e-6
        print(f"  {w}x{h}: 1 LSB of %.6f = {step_x:.5f} px (x), {step_y:.5f} px (y)")


def test_pnp_equivalence():
    print("\n[3] PnP: (original, K) vs (padded, K_pad)")
    files = sorted(glob.glob(str(REPO / "data/pallet/training_data/paper_release/"
                                  "v2_prod40k_clean_merged/labels/*.json")))[:200]
    diffs_t, diffs_r, mixed_t = [], [], []
    used = 0
    for f in files:
        o = json.load(open(f, encoding="utf-8"))["objects"][0]
        cd = json.load(open(f, encoding="utf-8"))["camera_data"]["intrinsics"]
        dm = o["dimensions_m"]
        kp = np.array([*o["projected_cuboid"][:8], o["projected_cuboid_centroid"]], dtype=float)
        vis = (kp[:, 0] >= 0) & (kp[:, 1] >= 0) & ~np.isnan(kp).any(axis=1)
        if vis.sum() < 6:
            continue
        obj = model_points(dm["width"], dm["height"], dm["depth"])[vis].reshape(-1, 1, 3)
        K = np.array([[cd["fx"], 0, cd["cx"]], [0, cd["fy"], cd["cy"]], [0, 0, 1]])
        Kp = K.copy()
        Kp[0, 2] += PAD
        Kp[1, 2] += PAD
        D = np.zeros((5, 1))
        a = kp[vis].reshape(-1, 1, 2)
        b = a + PAD

        def solve(img, k):
            ok, rv, tv = cv2.solvePnP(obj, img, k, D, flags=cv2.SOLVEPNP_EPNP)
            if not ok:
                return None
            rv, tv = cv2.solvePnPRefineLM(obj, img, k, D, rv, tv)
            return rv.ravel(), tv.ravel()

        r1 = solve(a, K)
        r2 = solve(b, Kp)
        r3 = solve(b, K)          # negative control: padded coords with original K
        if r1 is None or r2 is None:
            continue
        used += 1
        diffs_t.append(float(np.abs(r1[1] - r2[1]).max()))
        diffs_r.append(float(np.abs(r1[0] - r2[0]).max()))
        if r3 is not None:
            mixed_t.append(float(np.abs(r1[1] - r3[1]).max()))

    t = np.array(diffs_t); r = np.array(diffs_r); m = np.array(mixed_t)
    check("translation identical", t.max() < 1e-6, f"max |dt| {t.max():.2e} m over n={used}")
    check("rotation identical", r.max() < 1e-6, f"max |drvec| {r.max():.2e} rad")
    check("negative control differs", m.max() > 1e-3,
          f"mixing padded coords with original K shifts t by up to {m.max():.4f} m "
          f"(median {np.median(m):.4f})")


if __name__ == "__main__":
    print("padding contract tests  (pad=%d px)" % PAD)
    test_roundtrip()
    test_label_precision()
    test_pnp_equivalence()
    print("\n" + ("ALL PASS" if not FAILURES else f"FAILED: {FAILURES}"))
    sys.exit(1 if FAILURES else 0)
