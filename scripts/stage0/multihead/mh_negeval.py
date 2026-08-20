"""PHASE 8-10 -- presence, pose safety and the operating point for N0 vs N1.

Presence uses `score_4kp` only; no objectness head exists.  Pose uses the F3
solver that the fusion screen selected, and it is evaluated on the *whole*
positive population, not on accepted frames only -- conditioning pose error on
acceptance would be survivorship.
"""
from __future__ import annotations

import argparse, json, pathlib, sys
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_cigm as CG        # noqa: E402
import mh_curriculum as CU  # noqa: E402
import mh_data as MD        # noqa: E402
import mh_fusion as FU      # noqa: E402
import mh_negative as NG    # noqa: E402
import mh_negscreen as NS   # noqa: E402
import mh_screen as MS      # noqa: E402
import mh_splitlate as SL   # noqa: E402

OUT = MD.OUT
ARMS = ("N0", "N1")
SEEDS = (1, 2)
POS_EVAL = "D2_MH_DEV512"
RECALL_TARGET = 0.95


def log(m):
    print(m, flush=True)


def load(arm, seed):
    state = torch.load(MS.CKPT / f"curriculum_{arm}_seed{seed}" / "step_03000.pth",
                       map_location=MD.DEV, weights_only=False)
    model = SL.SplitLate("A1_CORNER_LINE")
    model.load_state_dict(state["model"])
    return model.to(MD.DEV).eval()


def pose_f3(model, stems, weight):
    """F3: corners -> PnP, theta refines rotation only, corners refit t."""
    import cv2
    grid_theta, grid_rho, valid, features = MS.lattice()
    from mh_arms import DH
    rows = []
    with torch.no_grad():
        for start in range(0, len(stems), MS.BATCH):
            chunk = stems[start:start + MS.BATCH]
            pack = CU.load_pack_items([(MD.DATA, s) for s in chunk])
            f50 = model.line_late(model.early(pack["images"]).detach())
            scores = model.line(f50, features)
            theta, rho = DH.decode(scores, grid_theta, grid_rho, valid)
            theta, rho = DH.canonical_from_centred(theta, rho)
            beliefs = CU.corner_forward(model, pack["images"])
            peaks = MS._decode_peaks(beliefs[-1][:, :9])
            presence = NG.presence_score(beliefs).cpu().numpy()
            _, _, support = DH.batch_rows(pack, CG.EDGES)
            for i, stem in enumerate(chunk):
                label = MD.read_label(stem)
                w, h = pack["resolution"][i]
                X, K = CG.object_points(label), CG.intrinsics(label)
                R_gt, t_gt = CG.gt_pose(label)
                px = CG.grid_to_pixels(peaks[i][:8], w, h)
                lines = __import__("mh_diagnose")._line_in_pixels(
                    theta[i].cpu().numpy(), rho[i].cpu().numpy(), w, h)
                base = CG.solve(X, px, K)
                row = {"stem": stem, "score": float(presence[i])}
                if base is None:
                    row.update({"R": np.nan, "t": np.nan, "add": np.nan,
                                "adds": np.nan})
                else:
                    rvec0, _ = cv2.Rodrigues(base[0])
                    try:
                        rvec = FU.rotation_only(rvec0.reshape(3), base[1], X, K, px,
                                                lines, CG.EDGES,
                                                support[i].cpu().numpy().astype(bool),
                                                weight)
                        R_star, _ = cv2.Rodrigues(rvec)
                        t_star = FU.translation_refit(R_star, base[1], X, K, px)
                        pose = (R_star, t_star)
                    except Exception:
                        pose = base
                    err = CG.pose_error(pose, R_gt, t_gt)
                    add, adds = FU.add_metrics(pose, R_gt, t_gt, X)
                    row.update({"R": err[0] if err else np.nan,
                                "t": err[1] if err else np.nan,
                                "add": add, "adds": adds})
                rows.append(row)
    return rows


def summarise_pose(rows):
    R = np.array([r["R"] for r in rows], float)
    t = np.array([r["t"] for r in rows], float)
    add = np.array([r["add"] for r in rows], float)
    adds = np.array([r["adds"] for r in rows], float)
    g = np.isfinite(R) & np.isfinite(t)
    return {"n": len(rows),
            "R_median": round(float(np.median(R[g])), 4),
            "R_p90": round(float(np.percentile(R[g], 90)), 4),
            "t_median": round(float(np.median(t[g])), 5),
            "t_p90": round(float(np.percentile(t[g], 90)), 5),
            "success_5cm5deg": round(float(((R <= 5) & (t <= 0.05) & g).sum()
                                           / max(len(rows), 1)), 4),
            "ADD_median": round(float(np.nanmedian(add)), 5),
            "ADDS_median": round(float(np.nanmedian(adds)), 5)}


def run(_a):
    MS.deterministic()
    d0 = json.loads((OUT / "theta_posealigned_d0.json").read_text())
    pos_stems = json.loads(
        (OUT / f"{POS_EVAL.lower()}_manifest.json").read_text())["stems"]
    neg_items = NG.negative_pool("dev")
    neg_type = {r["stem"]: r["negative_type"] for r in json.loads(
        (OUT / "negative_filtered_manifest_dev.json").read_text())["items"]}
    result = {"positive_eval": POS_EVAL, "n_pos": len(pos_stems),
              "negative_eval": "negative_synth_v1_dev", "n_neg": len(neg_items),
              "solver": "F3 rotation-only fusion + point-only t-refit",
              "recall_target": RECALL_TARGET, "seeds": {}}
    for seed in SEEDS:
        weight = d0["seeds"][f"seed{seed}"]["selected_lambda_theta"]
        block = {}
        for arm in ARMS:
            model = load(arm, seed)
            rows = pose_f3(model, pos_stems, weight)
            pos = np.array([r["score"] for r in rows])
            neg = NS.scores_negative(model, neg_items)
            entry = {"pose": summarise_pose(rows),
                     **NS.presence_metrics(pos, neg)}
            entry["operating_point"] = NS.operating_point(pos, neg)
            by = {}
            for (root, stem), s in zip(neg_items, neg):
                by.setdefault(neg_type[stem], []).append(float(s))
            entry["neg_by_category"] = {
                k: {"n": len(v), "median": round(float(np.median(v)), 5),
                    "p90": round(float(np.percentile(v, 90)), 5)}
                for k, v in sorted(by.items())}
            np.savez_compressed(OUT / f"neg_eval_scores_{arm}_seed{seed}.npz",
                                pos=pos, neg=neg,
                                neg_type=np.array([neg_type[s] for _, s in neg_items]),
                                neg_stem=np.array([s for _, s in neg_items]),
                                pos_R=np.array([r["R"] for r in rows]),
                                pos_t=np.array([r["t"] for r in rows]))
            block[arm] = entry
            p, o = entry["pose"], entry["operating_point"]
            log(f"seed{seed} {arm}  R {p['R_median']:6.3f} t {p['t_median']:.4f} "
                f"5cm5 {p['success_5cm5deg']:.4f} ADDS {p['ADDS_median']:.4f} | "
                f"AUROC {entry['AUROC']:.4f} AUPRC {entry['AUPRC']:.4f} "
                f"recall {o['recall']:.3f} FP/img {o['fp_per_image']:.4f}")
        result["seeds"][f"seed{seed}"] = block
    (OUT / "negative_presence_curves.json").write_text(json.dumps(result, indent=1))
    log("-> negative_presence_curves.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.parse_args()
    run(None)
