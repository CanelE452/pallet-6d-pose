#!/usr/bin/env python
"""Read-only audit of paper_base training labels for camera-facing 0123 convention.

Outputs numbers only (no clean/contaminated verdict). See module docstring in task spec.
Writes results under data/pallet/eval_results/v8_audit/.
"""
import os, glob, json, random
import numpy as np
import cv2

random.seed(0)
ROOT = "data/pallet/training_data"
OUT = "data/pallet/eval_results/v8_audit"
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "audit_squash"), exist_ok=True)

SAMPLE_N = 400  # per-source random sample target


def load_obj(path):
    d = json.load(open(path))
    o = d["objects"][0]
    K = d["camera_data"]["intrinsics"]
    Km = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1.0]])
    proj = np.array(o["projected_cuboid"], float)
    cub = np.array(o["cuboid"], float)
    return o, Km, proj, cub


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def invariants(proj):
    """camera-facing 0123 invariants on projected_cuboid order.
    inv1: front area(0,1,2,3) >= back area(4,5,6,7)
    inv2: mean_y{0,1,4,5} (top) < mean_y{2,3,6,7} (bottom)
    """
    front = poly_area(proj[[0, 1, 2, 3]])
    back = poly_area(proj[[4, 5, 6, 7]])
    inv1 = front >= back
    top_y = proj[[0, 1, 4, 5], 1].mean()
    bot_y = proj[[2, 3, 6, 7], 1].mean()
    inv2 = top_y < bot_y
    return inv1, inv2


def n_visible(proj, w=640, h=480):
    return int(((proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)).sum())


def pnp_residual(cub, proj, Km):
    """Rigid-set consistency: fit PnP(cuboid->projected), return mean reproj px err.

    NOTE: requires cuboid[i] to correspond to proj[i]. For mixed_v8 with-orig the
    reorder script permuted projected_cuboid but NOT cuboid, so the caller must
    reorder cuboid by the recovered orig->curr permutation first (see audit_with_orig).
    """
    try:
        ok, rv, tv = cv2.solvePnP(cub, proj, Km, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        uv, _ = cv2.projectPoints(cub, rv, tv, Km, None)
        uv = uv.reshape(-1, 2)
        return float(np.linalg.norm(uv - proj, axis=1).mean())
    except cv2.error:
        return None


def recovered_perm(stem):
    """For a with-orig stem, find permutation s.t. curr_proj[i] == orig_proj[perm[i]]."""
    pc = np.array(json.load(open(os.path.join(mv, stem + ".json")))["objects"][0]["projected_cuboid"])
    po = np.array(json.load(open(os.path.join(mv, stem + ".json.orig")))["objects"][0]["projected_cuboid"])
    return [int(np.linalg.norm(po - pc[i], axis=1).argmin()) for i in range(8)]


def list_files(d):
    js = [f for f in glob.glob(os.path.join(d, "*.json")) if not f.endswith(".orig")]
    return sorted(js)


# ---------- 1. mixed_v8 with-orig vs no-orig split ----------
mv = os.path.join(ROOT, "mixed_v8_train")
all_json = list_files(mv)
stems = [os.path.basename(f)[:-5] for f in all_json]
orig_stems = set(os.path.basename(f)[:-10] for f in glob.glob(os.path.join(mv, "*.json.orig")))
with_orig = [s for s in stems if s in orig_stems]
no_orig = [s for s in stems if s not in orig_stems]

no_orig_path = os.path.join(OUT, "no_orig_list.txt")
with open(no_orig_path, "w") as f:
    for s in sorted(no_orig):
        f.write(s + "\n")
print(f"mixed_v8: total={len(stems)} with_orig={len(with_orig)} no_orig={len(no_orig)} -> {no_orig_path}")


# ---------- helper: audit a sample list ----------
def audit_group(name, stem_dir, stems, tag=".json", trunc_filter=False,
                reproj_mode="direct"):
    """reproj_mode: 'direct' (cuboid[i]<->proj[i]) | 'perm' (with-orig: reorder cuboid by
    recovered permutation) | 'none' (squash: skip)."""
    if len(stems) > SAMPLE_N:
        stems = random.sample(stems, SAMPLE_N)
    n = 0
    inv1_pass = inv2_pass = 0
    reproj_errs = []
    skipped_trunc = 0
    for s in stems:
        p = os.path.join(stem_dir, s + tag)
        try:
            o, Km, proj, cub = load_obj(p)
        except Exception:
            continue
        if trunc_filter and n_visible(proj) < 6:
            skipped_trunc += 1
            continue
        n += 1
        i1, i2 = invariants(proj)
        inv1_pass += int(i1)
        inv2_pass += int(i2)
        if reproj_mode != "none":
            cub_use = cub
            if reproj_mode == "perm":
                cub_use = cub[recovered_perm(s)]
            e = pnp_residual(cub_use, proj, Km)
            if e is not None:
                reproj_errs.append(e)
    res = {
        "n": n,
        "inv1_area_%": 100.0 * inv1_pass / n if n else 0,
        "inv2_yorder_%": 100.0 * inv2_pass / n if n else 0,
        "reproj_n": len(reproj_errs),
        "reproj_mean_px": float(np.mean(reproj_errs)) if reproj_errs else None,
        "reproj_le2px_%": 100.0 * np.mean([e <= 2.0 for e in reproj_errs]) if reproj_errs else None,
        "skipped_trunc": skipped_trunc,
    }
    return res


results = {}
results["(a) mixed_v8 with-orig"] = audit_group("a", mv, with_orig, ".json", reproj_mode="perm")
results["(b) mixed_v8 no-orig"] = audit_group("b", mv, no_orig, ".json", reproj_mode="direct")
results["(c) aug_squash"] = audit_group("c", os.path.join(ROOT, "aug_squash"),
                                        [os.path.basename(f)[:-5] for f in list_files(os.path.join(ROOT, "aug_squash"))],
                                        ".json", reproj_mode="none")  # squash: reproj invalid by design
results["(d) aug_trunc"] = audit_group("d", os.path.join(ROOT, "aug_trunc"),
                                       [os.path.basename(f)[:-5] for f in list_files(os.path.join(ROOT, "aug_trunc"))],
                                       ".json", trunc_filter=True, reproj_mode="direct")  # no .json.orig
results["(e) aug_scale"] = audit_group("e", os.path.join(ROOT, "aug_scale"),
                                       [os.path.basename(f)[:-5] for f in list_files(os.path.join(ROOT, "aug_scale"))],
                                       ".json", reproj_mode="direct")  # no .json.orig


# ---------- 3. permutation check: .json vs .json.orig ----------
def permutation_check(stems, k=200):
    if len(stems) > k:
        stems = random.sample(stems, k)
    same_set = 0  # Hungarian min-cost matching ~0 (same 8 points)
    is_perm = 0   # but index order differs (>0 displacement)
    n = 0
    from scipy.optimize import linear_sum_assignment
    for s in stems:
        cur = json.load(open(os.path.join(mv, s + ".json")))["objects"][0]
        org = json.load(open(os.path.join(mv, s + ".json.orig")))["objects"][0]
        pc = np.array(cur["projected_cuboid"]); po = np.array(org["projected_cuboid"])
        C = np.linalg.norm(pc[:, None, :] - po[None, :, :], axis=2)
        r, c = linear_sum_assignment(C)
        matched_cost = C[r, c].mean()
        n += 1
        if matched_cost < 1.0:
            same_set += 1
        # index permutation distance: how many points moved index
        moved = int((c != np.arange(8)).sum())
        if moved > 0:
            is_perm += 1
    return {"n": n, "same_set_%": 100.0 * same_set / n, "reordered_%": 100.0 * is_perm / n}


perm_res = permutation_check(with_orig, k=200)
print("permutation check:", perm_res)


# ---------- 5. squash overlay dump ----------
def dump_squash_overlays(k=25):
    sq = os.path.join(ROOT, "aug_squash")
    stems = [os.path.basename(f)[:-5] for f in list_files(sq)]
    stems = random.sample(stems, min(k, len(stems)))
    colors = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0),
              (255, 255, 0), (255, 128, 0), (255, 0, 0), (255, 0, 255)]
    done = 0
    for s in stems:
        imgp = os.path.join(sq, s + ".png")
        if not os.path.exists(imgp):
            continue
        img = cv2.imread(imgp)
        if img is None:
            continue
        o, Km, proj, cub = load_obj(os.path.join(sq, s + ".json"))
        for i, (x, y) in enumerate(proj):
            pt = (int(round(x)), int(round(y)))
            cv2.circle(img, pt, 4, colors[i], -1)
            cv2.putText(img, str(i), (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
        # front face edges
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            cv2.line(img, tuple(proj[a].astype(int)), tuple(proj[b].astype(int)), (255, 255, 255), 1)
        cv2.imwrite(os.path.join(OUT, "audit_squash", f"{s}_overlay.png"), img)
        done += 1
    return done


nsq = dump_squash_overlays(25)
print(f"squash overlays dumped: {nsq} -> {os.path.join(OUT, 'audit_squash')}")


# ---------- 6. print table ----------
print("\n==== SOURCE PASS-RATE TABLE ====")
hdr = f"{'source':<26}{'n':>5}{'inv1_area%':>11}{'inv2_yorder%':>13}{'reproj_n':>9}{'reproj_mean_px':>15}{'reproj<=2px%':>13}"
print(hdr)
print("-" * len(hdr))
for k, v in results.items():
    rm = f"{v['reproj_mean_px']:.2f}" if v["reproj_mean_px"] is not None else "n/a"
    rp = f"{v['reproj_le2px_%']:.1f}" if v["reproj_le2px_%"] is not None else "n/a"
    print(f"{k:<26}{v['n']:>5}{v['inv1_area_%']:>11.1f}{v['inv2_yorder_%']:>13.1f}{v['reproj_n']:>9}{rm:>15}{rp:>13}")
print(f"\n(d) aug_trunc skipped (<6 visible corners): {results['(d) aug_trunc']['skipped_trunc']}")
print(f"\npermutation (.json vs .json.orig, n={perm_res['n']}): "
      f"same_set(Hungarian<1px)={perm_res['same_set_%']:.1f}%  reordered(index moved)={perm_res['reordered_%']:.1f}%")

# save json summary
summary = {"sources": results, "permutation": perm_res,
           "no_orig_list": no_orig_path, "squash_overlay_dir": os.path.join(OUT, "audit_squash"),
           "mixed_v8_counts": {"total": len(all_json), "with_orig": len(with_orig), "no_orig": len(no_orig)}}
json.dump(summary, open(os.path.join(OUT, "audit_summary.json"), "w"), indent=2)
print(f"\nsummary -> {os.path.join(OUT, 'audit_summary.json')}")
