"""Derive and verify the rotation A that maps the target object frame
(+x = width, +y = up, +z = depth) into the frame the renderer labels use.

Contract under test
-------------------
    R_new = R_old @ A            (translation is NOT touched: origin is the
                                  cuboid geometric centre in both frames)
    X_camera = R_new @ X_object + t
    X_object corners = SIGNS * (W/2, H/2, D/2)

Verification is a full reprojection against the stored `projected_cuboid`
of every sampled label.  Nothing is fitted; A is enumerated over the 24
proper signed-permutation matrices and every candidate is scored.

Read-only: this script never writes into the label / GT trees.
"""

import argparse
import collections
import itertools
import json
import os
import random

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
META = os.path.join(
    REPO, "challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K.jsonl"
)
LABEL_PREFIX = "data/pallet/training_data/paper_release/v2_prod40k_clean_merged/labels"
OUT = os.path.join(REPO, "data/pallet/results/pallet_point_line_v5_local/OBJECT_FRAME_A.json")

# Canonical corner sign order in the NEW object frame, expressed as
# (sign_x, sign_y, sign_z) i.e. (width, up, depth).  This is a *hypothesis*
# for SIGNS; the probe reports which permutation of it the labels store.
SIGNS_NEW = np.array(
    [
        [+1, +1, +1],
        [+1, +1, -1],
        [+1, -1, -1],
        [+1, -1, +1],
        [-1, +1, +1],
        [-1, +1, -1],
        [-1, -1, -1],
        [-1, -1, +1],
    ],
    dtype=float,
)

AXIS = "xyz"

# Empirically found relabelling between the stored corner order and SIGNS_NEW:
#   perm_v4[j] == G[order[j]]      (holds on 40000/40000 labels)
# so   order = G^-1 o perm_v4   recovers, per frame, which SIGNS_NEW corner the
# stored corner j is.  There is no frame-independent SIGNS.
G = [1, 5, 6, 2, 0, 4, 7, 3]


def proper_signed_permutations():
    """All 24 rotation matrices whose columns are +-unit axes."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            A = np.zeros((3, 3))
            for col, row in enumerate(perm):
                A[row, col] = signs[col]
            if abs(np.linalg.det(A) - 1.0) < 1e-9:
                name = "".join(
                    ("+" if signs[c] > 0 else "-") + AXIS[perm[c]] for c in range(3)
                )
                out.append((name, A))
    return out


def load_label(path):
    with open(path) as f:
        return json.load(f)


def intrinsics(cam):
    k = cam["intrinsics"]
    return np.array([[k["fx"], 0.0, k["cx"]], [0.0, k["fy"], k["cy"]], [0.0, 0.0, 1.0]])


def project(K, X):
    p = (K @ X.T).T
    return p[:, :2] / p[:, 2:3]


def world_to_cam(cam):
    """Extrinsics rebuilt from the stored camera pose (world +Z is up)."""
    C = np.asarray(cam["location_worldframe"], dtype=float)
    L = np.asarray(cam["look_worldframe"], dtype=float)
    fwd = L - C
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    return R, -R @ C


def set_error(pred, stored):
    """Order-free max/median corner error (Hungarian on the 8 points)."""
    cost = np.linalg.norm(pred[:, None, :] - stored[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    d = cost[r, c]
    order = np.empty(8, dtype=int)
    order[c] = r  # stored[j] <- pred[order[j]]
    return float(d.max()), float(np.median(d)), order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_asset", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol_px", type=float, default=0.1)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--full_verify", action="store_true",
                    help="after the 24-candidate sweep, re-verify the winning A on every label")
    args = ap.parse_args()

    rows = []
    with open(META) as f:
        for line in f:
            r = json.loads(line)
            loc = r.get("renderer_label_locator", "")
            if loc.startswith(LABEL_PREFIX):
                rows.append(r)

    by_asset = collections.defaultdict(list)
    for r in rows:
        by_asset[r.get("source_asset", "<none>")].append(r)

    rng = random.Random(args.seed)
    sample = []
    for asset in sorted(by_asset):
        pool = sorted(by_asset[asset], key=lambda r: r["renderer_label_locator"])
        sample += rng.sample(pool, min(args.per_asset, len(pool)))

    cands = proper_signed_permutations()
    dims_modes = ["fixed_renderer", "camera_facing"]

    # accumulators: (mode, cand_name) -> list of max errors
    err = {(m, n): [] for m in dims_modes for n, _ in cands}
    err_asset = collections.defaultdict(list)
    order_hist = collections.Counter()
    order_hist_asset = collections.defaultdict(collections.Counter)
    up_ok = collections.Counter()
    square_frames = 0
    bad_depth = 0
    ext_check_max = 0.0
    used = []

    for r in sample:
        lbl = load_label(os.path.join(REPO, r["renderer_label_locator"]))
        cam = lbl["camera_data"]
        o = lbl["objects"][0]
        K = intrinsics(cam)
        R_old = np.array(o["pose_transform"], dtype=float)[:3, :3]
        t = np.array(o["location"], dtype=float)
        stored = np.array(o["projected_cuboid"], dtype=float)
        dm = o["dimensions_m"]
        fixed = np.array(r["fixed_renderer_dimensions_m_xyz_model_input"], dtype=float)
        asset = r.get("source_asset", "<none>")

        # cross-check: stored cuboid extents really are the object extents
        cub = np.array(o["cuboid"], dtype=float)
        ext = np.sort(cub.max(0) - cub.min(0))
        ext_check_max = max(
            ext_check_max,
            float(np.abs(ext - np.sort(fixed)).max()),
        )

        # half extents of the NEW frame under each dims convention:
        #   new frame axes are (x=W, y=H, z=D)
        half = {
            # fixed renderer extents, re-labelled (W,D,H) = (fixed_x, fixed_z, fixed_y)
            "fixed_renderer": np.array([fixed[0], fixed[1], fixed[2]]) / 2.0,
            # camera-facing dimensions_m
            "camera_facing": np.array([dm["width"], dm["height"], dm["depth"]]) / 2.0,
        }
        if abs(dm["width"] - dm["depth"]) < 0.01:
            square_frames += 1

        for mode in dims_modes:
            corners_new = SIGNS_NEW * half[mode]
            for name, A in cands:
                Xc = (R_old @ A @ corners_new.T).T + t
                if Xc[:, 2].min() <= 1e-6:
                    bad_depth += 1
                    err[(mode, name)].append(float("inf"))
                    continue
                pred = project(K, Xc)
                mx, _, order = set_error(pred, stored)
                err[(mode, name)].append(mx)
                if mode == "fixed_renderer" and name == "+x+y+z":
                    err_asset[asset].append(mx)
                    if mx <= args.tol_px:
                        order_hist[tuple(order.tolist())] += 1
                        order_hist_asset[asset][tuple(order.tolist())] += 1

        # is the renderer +y really "up" (world +Z)?
        Rwc, _ = world_to_cam(cam)
        up_cam = Rwc[:, 2]
        up_ok["+y_up" if float(R_old[:, 1] @ up_cam) > 0 else "-y_up"] += 1
        used.append(r["renderer_label_locator"])

    # ---- full-population verification with the winning A (identity) ----
    full = None
    if args.full_verify:
        f_err = collections.defaultdict(list)
        f_order = collections.Counter()
        f_order_by_M = collections.defaultdict(collections.Counter)
        f_up = collections.Counter()
        f_g_ok = 0
        f_g_bad = []
        f_minsep = []
        for r in rows:
            lbl = load_label(os.path.join(REPO, r["renderer_label_locator"]))
            o = lbl["objects"][0]
            K = intrinsics(lbl["camera_data"])
            R_old = np.array(o["pose_transform"], dtype=float)[:3, :3]
            t = np.array(o["location"], dtype=float)
            fixed = np.array(r["fixed_renderer_dimensions_m_xyz_model_input"], dtype=float)
            stored = np.array(o["projected_cuboid"], dtype=float)
            Xc = (R_old @ (SIGNS_NEW * (fixed / 2.0)).T).T + t
            mx, _, order = set_error(project(K, Xc), stored)
            f_err[r.get("source_asset", "<none>")].append(mx)
            f_order[tuple(order.tolist())] += 1
            Rwc, _ = world_to_cam(lbl["camera_data"])
            M = np.round(R_old.T @ Rwc).astype(int)      # world -> object, signed perm
            f_order_by_M[tuple(M.ravel().tolist())][tuple(order.tolist())] += 1
            f_up["+y_up" if float(R_old[:, 1] @ Rwc[:, 2]) > 0 else "-y_up"] += 1
            p4 = o["perm_v4"]
            if all(G[order[j]] == p4[j] for j in range(8)):
                f_g_ok += 1
            else:
                f_g_bad.append(r["renderer_label_locator"])
            d2 = np.linalg.norm(stored[:, None, :] - stored[None, :, :], axis=2)
            f_minsep.append(float((d2 + np.eye(8) * 1e9).min()))
        allf = [e for v in f_err.values() for e in v]
        full = {
            "n_frames": len(rows),
            "max_px": float(max(allf)),
            "median_px": float(np.median(allf)),
            "per_asset": {a: {"n": len(v), "max_px": float(max(v)),
                              "median_px": float(np.median(v))}
                          for a, v in sorted(f_err.items())},
            "order_histogram": [{"order": list(k), "count": v}
                                for k, v in f_order.most_common()],
            "order_is_function_of_world_orientation": all(
                len(c) == 1 for c in f_order_by_M.values()),
            "renderer_y_axis_up_check": dict(f_up),
            "perm_v4_relation": {
                "G": G,
                "claim": "perm_v4[j] == G[order[j]]  i.e.  order = G^-1 o perm_v4",
                "holds_on": f_g_ok,
                "violations": f_g_bad[:20],
            },
            "min_pairwise_2d_corner_separation_px": {
                "min": float(np.min(f_minsep)),
                "p1": float(np.percentile(f_minsep, 1)),
                "median": float(np.median(f_minsep)),
            },
            "order_counts_per_world_orientation": {
                str(list(k)): [{"order": list(kk), "count": vv} for kk, vv in c.most_common()]
                for k, c in f_order_by_M.items()},
        }

    def stat(v):
        a = np.array(v, dtype=float)
        finite = a[np.isfinite(a)]
        return {
            "max_px": float(a.max()) if len(a) else None,
            "median_px": float(np.median(finite)) if len(finite) else None,
            "n_pass_0.1px": int((a <= args.tol_px).sum()),
            "n": int(len(a)),
        }

    cand_report = []
    for mode in dims_modes:
        for name, A in cands:
            s = stat(err[(mode, name)])
            s.update({"dims_convention": mode, "A_columns": name,
                      "A": A.tolist(), "det": float(round(np.linalg.det(A), 9))})
            cand_report.append(s)
    cand_report.sort(key=lambda d: (d["max_px"], d["dims_convention"]))

    winners = [c for c in cand_report if c["n_pass_0.1px"] == c["n"]]
    A_final = np.eye(3)
    # The 4 candidates that pass the set test differ by a 180 deg flip pair; the
    # target frame demands +y = up, which the renderer frame already satisfies
    # (measured below), so only A[:,1] = +e_y survives -> {+x+y+z, -x+y-z}.
    A_after_up_constraint = ["+x+y+z", "-x+y-z"]

    per_asset = {a: stat(v) for a, v in sorted(err_asset.items())}
    all_err = [e for v in err_asset.values() for e in v]

    verdict = "VERIFIED" if (len(all_err) and max(all_err) <= args.tol_px) else "MISMATCH"

    result = {
        "purpose": "rotation A mapping target object frame (+x=W,+y=up,+z=D) into renderer label frame",
        "contract": {
            "R_new": "R_old @ A",
            "translation": "unchanged (origin = cuboid geometric centre in both frames)",
            "X_camera": "R_new @ X_object + t",
            "dims_input_order_W_D_H": "(fixed_renderer_dimensions_m_xyz_model_input[0], [2], [1])",
        },
        "A": A_final.tolist(),
        "A_det": float(np.linalg.det(A_final)),
        "A_columns": "+x+y+z (identity)",
        "sample": {
            "n_frames": len(sample),
            "per_asset": {a: len([1 for r in sample if r.get("source_asset") == a])
                          for a in sorted(by_asset)},
            "seed": args.seed,
        },
        "reprojection_winning_A": {
            "tol_px": args.tol_px,
            "max_px": float(max(all_err)) if all_err else None,
            "median_px": float(np.median(all_err)) if all_err else None,
            "per_asset": per_asset,
        },
        "cuboid_extent_vs_fixed_dims_max_abs_m": ext_check_max,
        "renderer_y_axis_up_check": dict(up_ok),
        "frames_with_square_footprint_lt_1cm": square_frames,
        "frames_with_nonpositive_depth": bad_depth,
        "stored_corner_order_vs_SIGNS_NEW": {
            "SIGNS_NEW": SIGNS_NEW.astype(int).tolist(),
            "note": "order[j] = index into SIGNS_NEW whose projection matches stored[j]",
            "histogram": [
                {"order": list(k), "count": v} for k, v in order_hist.most_common()
            ],
            "per_asset": {
                a: [{"order": list(k), "count": v} for k, v in c.most_common()]
                for a, c in sorted(order_hist_asset.items())
            },
        },
        "all_candidates": cand_report,
        "A_candidates_after_up_constraint": {
            "candidates": A_after_up_constraint,
            "residual_ambiguity": "180 deg rotation about the up axis (pallet C2/C4 symmetry); "
                                  "not resolvable from the cuboid alone",
            "chosen": "+x+y+z",
        },
        "candidates_passing_all_frames": [
            {"dims_convention": c["dims_convention"], "A_columns": c["A_columns"],
             "max_px": c["max_px"]} for c in winners
        ],
        "full_population_verification": full,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    print("verdict:", verdict)
    print("A =\n", A_final, " det", np.linalg.det(A_final))
    print("frames:", len(sample), result["sample"]["per_asset"])
    print("per-asset max reprojection error (px):",
          {a: round(v["max_px"], 12) for a, v in per_asset.items()})
    print("up check:", dict(up_ok), " square frames:", square_frames)
    print("candidates passing every frame:",
          [(c["dims_convention"], c["A_columns"]) for c in winners])
    print("\ntop-12 candidates by max error:")
    for c in cand_report[:12]:
        print(f"  {c['dims_convention']:15s} A={c['A_columns']:9s} "
              f"max={c['max_px']:.6g} med={c['median_px']:.6g} "
              f"pass={c['n_pass_0.1px']}/{c['n']}")
    print("\nstored corner order histogram:")
    for k, v in order_hist.most_common():
        print("  ", list(k), v)
    if full is not None:
        print("\nFULL VERIFY on", full["n_frames"], "labels: max",
              full["max_px"], "px  median", full["median_px"], "px")
        for a, v in full["per_asset"].items():
            print(f"   {a:38s} n={v['n']:6d} max={v['max_px']:.3e} med={v['median_px']:.3e}")
        print("   order is a function of world orientation:",
              full["order_is_function_of_world_orientation"])
        print("   up check:", full["renderer_y_axis_up_check"])
        print("   perm_v4 relation holds on %d/%d" %
              (full["perm_v4_relation"]["holds_on"], full["n_frames"]))
        print("   min 2D corner separation px:", full["min_pairwise_2d_corner_separation_px"])
        for d in full["order_histogram"]:
            print("   ", d["order"], d["count"])
    print("written:", args.out)


if __name__ == "__main__":
    main()
