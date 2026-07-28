#!/usr/bin/env python
"""diffpnp3d_q1_build_split.py — PAPER_S2 Phase 3 (Q1) fixed train/val split.

Builds ONE fixed split reused by every lambda run (only variable = lambda):
  - train: 3,000 frames from Arm A (mixed_v8 / v4 / aug_*_v2) as a SYMLINK dir
           + a matching DiffPnP index (keyed to the symlink abs paths) so the
           loader emits per-frame DiffPnP targets exactly as on the real dirs.
           Composition = high DiffPnP-eligible fraction so lambda actually bites:
             mixed_v8_train interior-valid&V8   : 1400
             v4_split_base  interior-valid&V8   : 1000
             aug_{squash,trunc,scale}_v2 (2D-aug, DiffPnP-ineligible): 600
           => 2400/3000 = 80% eligible.
  - val: 500 frames sampled from training_data/val (1500), fixed seed. Eval only.

Eligible = pnp_valid_3d & V8 & belief-interior (mirrors the loader's runtime gate:
all 8 projected corners inside [2*sigma, 50-2*sigma) in the 50x50 belief grid under
the fixed anisotropic 640x480->50 scale). sigma=2 => w=4.

Outputs under data/pallet/results/paper_s2_scratch_diffpnp/q1_split/:
  train/000000.{png,json} ... (symlinks)
  train_index/train.json          (DiffPnP index for the symlink dir)
  val_list.json                   ({fid, json, png, entry} x500)
  split_manifest.json             (provenance: source frame per subset id)
"""
import json
import os
import random

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
DATA = os.path.join(ROOT, "data/pallet/training_data")
IDX_DIR = os.path.join(ROOT, "data/pallet/results/paper_s2_scratch_diffpnp/pnp_valid_3d_index")
OUT = os.path.join(ROOT, "data/pallet/results/paper_s2_scratch_diffpnp/q1_split")
SEED = 20260708
SIGMA = 2.0
SZ = 50
W = int(SIGMA * 2)  # 4

ELIGIBLE_PLAN = [("mixed_v8_train", 1400), ("v4_split_base", 1000)]
AUG_PLAN = [("aug_squash_v2", 200), ("aug_trunc_v2", 200), ("aug_scale_v2", 200)]


def interior_ok(pc):
    """all 8 projected corners inside belief interior under 640x480->50 aniso scale."""
    pc = np.asarray(pc[:8], float)
    bx = pc[:, 0] * SZ / 640.0
    by = pc[:, 1] * SZ / 480.0
    return bool(np.all((bx - W >= 0) & (bx + W < SZ)
                       & (by - W >= 0) & (by + W < SZ)))


def load_pc(abs_json):
    o = json.load(open(abs_json))["objects"][0]
    return o["projected_cuboid"][:8]


def main():
    rng = random.Random(SEED)
    os.makedirs(os.path.join(OUT, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "train_index"), exist_ok=True)

    # ---- pick eligible train frames (interior-valid & V8) ----
    train_entries = {}   # subset_rel_json -> entry
    manifest = {}        # subset_id -> source rel
    sid = 0

    def link(src_rel, entry=None):
        nonlocal sid
        stem = f"{sid:06d}"
        for ext in (".png", ".json"):
            src = os.path.join(DATA, src_rel.replace(".json", ext))
            dst = os.path.join(OUT, "train", stem + ext)
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
        if entry is not None:
            train_entries[f"train/{stem}.json"] = entry
        manifest[stem] = src_rel
        sid += 1

    for ds, want in ELIGIBLE_PLAN:
        idx = json.load(open(os.path.join(IDX_DIR, f"{ds}.json")))
        cand = [(rel, e) for rel, e in idx.items()
                if e["pnp_valid_3d"] and e["V8"]]
        rng.shuffle(cand)
        n = 0
        for rel, e in cand:
            aj = os.path.join(DATA, rel)
            if not os.path.exists(aj):
                continue
            try:
                if not interior_ok(load_pc(aj)):
                    continue
            except Exception:
                continue
            link(rel, e)
            n += 1
            if n >= want:
                break
        print(f"[eligible] {ds}: linked {n}/{want} interior-valid&V8")

    n_eligible = sid

    # ---- pick aug (DiffPnP-ineligible) train frames for diversity ----
    for ds, want in AUG_PLAN:
        ddir = os.path.join(DATA, ds)
        if not os.path.isdir(ddir):
            print(f"[aug] {ds}: MISSING dir, skip")
            continue
        jsons = sorted(f for f in os.listdir(ddir) if f.endswith(".json"))
        rng.shuffle(jsons)
        n = 0
        for jf in jsons:
            stem = jf[:-5]
            if not os.path.exists(os.path.join(ddir, stem + ".png")):
                continue
            link(f"{ds}/{jf}", None)   # no index entry => loader valid=0
            n += 1
            if n >= want:
                break
        print(f"[aug] {ds}: linked {n}/{want}")

    with open(os.path.join(OUT, "train_index", "train.json"), "w") as f:
        json.dump(train_entries, f)
    with open(os.path.join(OUT, "split_manifest.json"), "w") as f:
        json.dump({"seed": SEED, "n_train": sid, "n_eligible_linked": n_eligible,
                   "eligible_plan": ELIGIBLE_PLAN, "aug_plan": AUG_PLAN,
                   "source": manifest}, f)
    print(f"[train] total linked = {sid} ({n_eligible} eligible, "
          f"{sid - n_eligible} aug). index entries = {len(train_entries)}")

    # ---- val 500 (fixed seed) ----
    vidx = json.load(open(os.path.join(IDX_DIR, "val.json")))
    vrels = sorted(vidx.keys())
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(vrels)
    val_list = []
    for rel in vrels:
        stem = os.path.basename(rel)[:-5]
        jp = os.path.join(DATA, rel)
        ip = jp.replace(".json", ".png")
        if not (os.path.exists(jp) and os.path.exists(ip)):
            continue
        val_list.append({"fid": stem, "json": jp, "png": ip, "entry": vidx[rel]})
        if len(val_list) >= 500:
            break
    with open(os.path.join(OUT, "val_list.json"), "w") as f:
        json.dump(val_list, f)
    v8 = sum(1 for v in val_list if v["entry"]["V8"])
    print(f"[val] {len(val_list)} frames (V8={v8}, V<8={len(val_list)-v8})")
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
