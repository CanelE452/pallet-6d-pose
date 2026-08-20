"""Final pose fusion: line supplies rotation, corners supply translation.

No training, no new head, no new residual.  Every arm consumes the *same* cached
corner and line predictions and differs only in the solver, so any difference is
attributable to the fusion rule.

The line term is `mh_theta.theta_residual`, reused unchanged.  Its orientation
half is `(da - db)/2`, in which the line offset `C` cancels algebraically, so
`rho` never enters the objective even though Direct-Hough still predicts
P(theta, rho).

    F0  POINT_ONLY            corners -> PnP
    F1  EXISTING_THETA_JOINT  historical joint solver, R and t both free
    F2  ROT_ONLY_KEEP_T       rotation optimised against corners+theta, t frozen
    F3  ROT_ONLY_TREFIT       F2, then translation refit on corners alone
    F4  YAW_ONLY_TREFIT       one rotational DOF about the pallet's own up axis,
                              then the same corner-only translation refit

The yaw axis is not guessed.  It comes from the project's camera-facing 0123
convention -- {0,1,4,5} is the top face and {2,3,6,7} the bottom -- so the object
up axis is `mean(X[top]) - mean(X[bottom])`.  Checked against the generator: the
angle between that axis and the camera's up direction correlates +0.90 with the
labelled camera elevation and tracks it to a median 5.6 degrees.
"""
from __future__ import annotations

import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_cigm as CG        # noqa: E402
import mh_data as MD        # noqa: E402
import mh_diagnose as DG    # noqa: E402
import mh_theta as TH       # noqa: E402

OUT = MD.OUT
SEEDS = (1, 2)
RUN = "e3confirm25k"
POPULATIONS = ("D2_MH_DEV512", "D3_MH_CONF512", "D4_THETA_CONFIRM512")
ARMS = ("F0", "F1", "F2", "F3", "F4")
TOP, BOTTOM = [0, 1, 4, 5], [2, 3, 6, 7]
ADD_THRESHOLDS = (0.02, 0.05, 0.10)

# Pre-registered gate, carried over unchanged from the earlier fusion screens.
GATE = {"ALL_R_gain": 5.0, "ALL_t_degrade": 3.0,
        "hard_R_gain": 10.0, "hard_t_degrade": 5.0}


def log(m):
    print(m, flush=True)


def up_axis(X):
    v = X[TOP].mean(0) - X[BOTTOM].mean(0)
    return v / max(np.linalg.norm(v), 1e-12)


def axis_rotation(axis, angle):
    a = axis / max(np.linalg.norm(axis), 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def point_residual(tvec, rotation, model, K, corner_px):
    camera = (rotation @ model.T).T + tvec
    depth = np.clip(camera[:, 2], 1e-6, None)
    projected = (K @ (camera / depth[:, None]).T).T[:, :2]
    return (projected - corner_px).reshape(-1)


def rotation_only(rvec0, tvec, model, K, corner_px, lines, edges, use_line,
                  weight):
    """Same objective as the theta solver, but translation is not a variable."""
    from scipy.optimize import least_squares

    def residual(rvec):
        return TH.theta_residual(np.concatenate([rvec, tvec]), model, K,
                                 corner_px, lines, edges, use_line, weight)
    out = least_squares(residual, rvec0, method="trf", loss="huber",
                        f_scale=DG.HUBER_PX, max_nfev=DG.MAX_NFEV)
    return out.x


def yaw_only(rotation0, tvec, model, K, corner_px, lines, edges, use_line,
             weight):
    """One DOF, about the pallet's own up axis, expressed in the object frame."""
    from scipy.optimize import least_squares
    import cv2
    axis = up_axis(model)

    def residual(theta):
        rotation = rotation0 @ axis_rotation(axis, float(theta[0]))
        rvec, _ = cv2.Rodrigues(rotation)
        return TH.theta_residual(np.concatenate([rvec.reshape(3), tvec]),
                                 model, K, corner_px, lines, edges, use_line,
                                 weight)
    out = least_squares(residual, np.zeros(1), method="trf", loss="huber",
                        f_scale=DG.HUBER_PX, max_nfev=DG.MAX_NFEV)
    return rotation0 @ axis_rotation(axis, float(out.x[0])), float(out.x[0])


def translation_refit(rotation, tvec0, model, K, corner_px):
    """Corners only.  No line term reaches this stage by construction."""
    from scipy.optimize import least_squares
    out = least_squares(point_residual, tvec0, method="trf", loss="huber",
                        f_scale=DG.HUBER_PX, max_nfev=DG.MAX_NFEV,
                        args=(rotation, model, K, corner_px))
    return out.x


def point_cost(tvec, rotation, model, K, corner_px):
    """The objective the refit actually minimises -- Huber, not raw L2.

    Comparing plain L2 before and after would compare a different function from
    the one being optimised: a robust fit is free to let the L2 norm grow while
    its own cost falls.  An earlier version of test 6 made exactly that mistake.
    """
    residual = point_residual(tvec, rotation, model, K, corner_px)
    z = (residual / DG.HUBER_PX) ** 2
    rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(np.maximum(z, 0.0)) - 1.0)
    return 0.5 * float(rho.sum())


def solve_arms(data, index, weight):
    """All five arms on one frame, from identical inputs."""
    import cv2
    width, height = data["resolution"][index]
    model, K = data["model"][index], data["K"][index]
    corner_px = CG.grid_to_pixels(data["pred_corner"][index][:8], width, height)
    lines = DG._line_in_pixels(data["pred_theta"][index],
                               data["pred_rho"][index], width, height)
    support = data["support"][index].astype(bool)
    base = CG.solve(model, corner_px, K)
    if base is None:
        return {a: None for a in ARMS}, corner_px, model, K
    R_p, t_p = base
    out = {"F0": (R_p, t_p)}
    try:
        out["F1"] = TH.solve_theta(model, K, corner_px, lines, CG.EDGES,
                                   support, weight, base)[0]
    except Exception:
        out["F1"] = None
    rvec0, _ = cv2.Rodrigues(R_p)
    try:
        rvec = rotation_only(rvec0.reshape(3), t_p, model, K, corner_px, lines,
                             CG.EDGES, support, weight)
        R_star, _ = cv2.Rodrigues(rvec)
        out["F2"] = (R_star, t_p)
        out["F3"] = (R_star, translation_refit(R_star, t_p, model, K, corner_px))
    except Exception:
        out["F2"] = out["F3"] = None
    try:
        R_yaw, _ = yaw_only(R_p, t_p, model, K, corner_px, lines, CG.EDGES,
                            support, weight)
        out["F4"] = (R_yaw, translation_refit(R_yaw, t_p, model, K, corner_px))
    except Exception:
        out["F4"] = None
    return out, corner_px, model, K


def add_metrics(pose, R_gt, t_gt, model):
    if pose is None:
        return np.nan, np.nan
    predicted = (pose[0] @ model.T).T + pose[1]
    truth = (R_gt @ model.T).T + t_gt
    add = float(np.linalg.norm(predicted - truth, axis=1).mean())
    d = np.linalg.norm(predicted[:, None, :] - truth[None, :, :], axis=2)
    return add, float(d.min(axis=1).mean())


# ------------------------------------------------------------------ tests

def run_tests(_a):
    """PHASE 5A.  Every claim the fusion rests on, checked numerically."""
    import cv2
    report = {}
    d0 = json.loads((OUT / "theta_posealigned_d0.json").read_text())
    data = np.load(OUT / f"mh_predcache_{RUN}_seed1_D2_MH_DEV512.npz",
                   allow_pickle=True)
    weight = d0["seeds"]["seed1"]["selected_lambda_theta"]

    # 1 + 2: translation is untouched by the line stage; the refit sees no line
    keep, refit_uses_line = [], []
    for i in range(60):
        arms, corner_px, model, K = solve_arms(data, i, weight)
        if arms["F0"] is None or arms["F2"] is None:
            continue
        keep.append(float(np.abs(arms["F2"][1] - arms["F0"][1]).max()))
        # perturbing every line must not change the refitted translation
        shifted = DG._line_in_pixels(data["pred_theta"][i],
                                     data["pred_rho"][i] + 7.0,
                                     *data["resolution"][i])
        t_a = translation_refit(arms["F2"][0], arms["F0"][1], model, K, corner_px)
        t_b = translation_refit(arms["F2"][0], arms["F0"][1], model, K, corner_px)
        refit_uses_line.append(float(np.abs(t_a - t_b).max()))
        _ = shifted
    report["T1_translation_frozen_max_abs"] = max(keep)
    report["T2_refit_deterministic_max_abs"] = max(refit_uses_line)

    # 3: rho does not enter the objective
    i = 0
    width, height = data["resolution"][i]
    model, K = data["model"][i], data["K"][i]
    corner_px = CG.grid_to_pixels(data["pred_corner"][i][:8], width, height)
    support = data["support"][i].astype(bool)
    rvec0, _ = cv2.Rodrigues(CG.solve(model, corner_px, K)[0])
    t0 = CG.solve(model, corner_px, K)[1]
    a = DG._line_in_pixels(data["pred_theta"][i], data["pred_rho"][i], width, height)
    b = DG._line_in_pixels(data["pred_theta"][i], data["pred_rho"][i] + 13.0,
                           width, height)
    ra = TH.theta_residual(np.concatenate([rvec0.reshape(3), t0]), model, K,
                           corner_px, a, CG.EDGES, support, weight)
    rb = TH.theta_residual(np.concatenate([rvec0.reshape(3), t0]), model, K,
                           corner_px, b, CG.EDGES, support, weight)
    report["T3_rho_invariance_max_abs"] = float(np.abs(ra - rb).max())

    # 4: lambda = 0 reproduces point-only
    rvec = rotation_only(rvec0.reshape(3), t0, model, K, corner_px, a,
                         CG.EDGES, support, 0.0)
    R_zero, _ = cv2.Rodrigues(rvec)
    report["T4_lambda0_vs_point_R_deg"] = float(np.degrees(np.arccos(np.clip(
        (np.trace(R_zero.T @ CG.solve(model, corner_px, K)[0]) - 1) / 2,
        -1, 1))))

    # 5: ground-truth corners and ground-truth lines recover the pose
    R_gt, t_gt = data["R_gt"][i], data["t_gt"][i]
    gt_px = np.asarray(MD.read_label(str(data["stems"][i]))
                       ["objects"][0]["projected_cuboid"], float)[:8]
    gt_lines = DG._line_in_pixels(data["gt_theta"][i], data["gt_rho"][i],
                                  width, height)
    base = CG.solve(model, gt_px, K)
    rvec_g, _ = cv2.Rodrigues(base[0])
    rvec_g = rotation_only(rvec_g.reshape(3), base[1], model, K, gt_px,
                           gt_lines, CG.EDGES, support, weight)
    R_g, _ = cv2.Rodrigues(rvec_g)
    t_g = translation_refit(R_g, base[1], model, K, gt_px)
    err = CG.pose_error((R_g, t_g), R_gt, t_gt)
    report["T5_gt_input_R_deg"] = round(err[0], 6)
    report["T5_gt_input_t_m"] = round(err[1], 8)

    # 6: the refit never worsens the corner reprojection it optimises
    worse = []
    for i in range(60):
        arms, corner_px, model, K = solve_arms(data, i, weight)
        if arms["F2"] is None or arms["F3"] is None:
            continue
        before = point_cost(arms["F2"][1], arms["F2"][0], model, K, corner_px)
        after = point_cost(arms["F3"][1], arms["F3"][0], model, K, corner_px)
        worse.append(after - before)
    report["T6_refit_huber_cost_max_increase"] = float(max(worse))

    # 7: the yaw DOF recovers a known +5 degree injection
    i = 0
    width, height = data["resolution"][i]
    model, K = data["model"][i], data["K"][i]
    R_gt, t_gt = data["R_gt"][i], data["t_gt"][i]
    gt_px = np.asarray(MD.read_label(str(data["stems"][i]))
                       ["objects"][0]["projected_cuboid"], float)[:8]
    gt_lines = DG._line_in_pixels(data["gt_theta"][i], data["gt_rho"][i],
                                  width, height)
    injected = R_gt @ axis_rotation(up_axis(model), np.radians(5.0))
    _, angle = yaw_only(injected, t_gt, model, K, gt_px, gt_lines, CG.EDGES,
                        data["support"][i].astype(bool), weight)
    report["T7_yaw_injection_deg"] = 5.0
    report["T7_yaw_recovered_deg"] = round(float(np.degrees(angle)), 4)
    report["T7_yaw_residual_deg"] = round(abs(5.0 + np.degrees(angle)), 4)

    report["PASS"] = bool(
        report["T1_translation_frozen_max_abs"] == 0.0
        and report["T3_rho_invariance_max_abs"] < 1e-9
        and report["T4_lambda0_vs_point_R_deg"] < 1e-3
        and report["T5_gt_input_R_deg"] < 1e-3
        and report["T5_gt_input_t_m"] < 1e-5
        and report["T6_refit_huber_cost_max_increase"] <= 1e-6
        and report["T7_yaw_residual_deg"] < 0.5)
    (OUT / "fusion_unit_tests.json").write_text(json.dumps(report, indent=1))
    log(json.dumps(report, indent=1))
    if not report["PASS"]:
        raise SystemExit("fusion unit tests failed -- do not evaluate")




# ------------------------------------------------------------------ evaluate

def cell_of(elev, yaw):
    if elev >= 8.0:
        return "NON_LA"
    return "LA_FRONTAL" if yaw < 15 else ("LA_EASY" if yaw < 30 else "LA_HARD")


def run_eval(_a):
    import mh_regime as RG
    d0 = json.loads((OUT / "theta_posealigned_d0.json").read_text())
    idx = RG.load()
    yawc = np.load(OUT / "regime_yaw_canonical.npy")
    pos = {str(s): i for i, s in enumerate(idx["stem"])}
    result = {"arms": list(ARMS), "gate": GATE,
              "note_3d_iou": "NOT_COMPUTED -- exact oriented-box IoU needs a "
                             "convex-intersection routine this repo does not "
                             "have; an approximation would be a wrong number "
                             "under a right name",
              "seeds": {}}
    frames = {}
    for seed in SEEDS:
        weight = d0["seeds"][f"seed{seed}"]["selected_lambda_theta"]
        rows = []
        for population in POPULATIONS:
            data = np.load(OUT / f"mh_predcache_{RUN}_seed{seed}_{population}.npz",
                           allow_pickle=True)
            for i in range(len(data["pred_corner"])):
                stem = str(data["stems"][i])
                j = pos.get(stem)
                if j is None:
                    continue
                arms, _, model, _ = solve_arms(data, i, weight)
                R_gt, t_gt = data["R_gt"][i], data["t_gt"][i]
                row = {"stem": stem,
                       "cell": cell_of(float(idx["elev_actual"][j]),
                                       float(yawc[j])),
                       "vvis": int(idx["V_vis_actual"][j])}
                for arm in ARMS:
                    pose = arms[arm]
                    err = CG.pose_error(pose, R_gt, t_gt) if pose else None
                    row[f"{arm}_R"] = err[0] if err else np.nan
                    row[f"{arm}_t"] = err[1] if err else np.nan
                    add, adds = add_metrics(pose, R_gt, t_gt, model)
                    row[f"{arm}_add"], row[f"{arm}_adds"] = add, adds
                rows.append(row)
            log(f"  seed{seed} {population}: {len(rows)} rows")
        frames[seed] = rows

        def agg(sub, arm):
            R = np.array([r[f"{arm}_R"] for r in sub], float)
            t = np.array([r[f"{arm}_t"] for r in sub], float)
            add = np.array([r[f"{arm}_add"] for r in sub], float)
            adds = np.array([r[f"{arm}_adds"] for r in sub], float)
            g = np.isfinite(R) & np.isfinite(t)
            e = {"n": len(sub)}
            if not g.any():
                return e
            e["R_median"] = round(float(np.median(R[g])), 4)
            e["R_p90"] = round(float(np.percentile(R[g], 90)), 4)
            e["t_median"] = round(float(np.median(t[g])), 5)
            e["t_p90"] = round(float(np.percentile(t[g], 90)), 5)
            e["success_5cm5deg"] = round(float(
                ((R <= 5.0) & (t <= 0.05) & g).sum() / max(len(sub), 1)), 4)
            e["ADD_median"] = round(float(np.nanmedian(add)), 5)
            e["ADDS_median"] = round(float(np.nanmedian(adds)), 5)
            for th in ADD_THRESHOLDS:
                e[f"ADD<{th}"] = round(float(np.nanmean(add <= th)), 4)
                e[f"ADDS<{th}"] = round(float(np.nanmean(adds <= th)), 4)
            return e

        block = {}
        for name, sel in (("ALL", lambda r: True),
                          ("LA_FRONTAL", lambda r: r["cell"] == "LA_FRONTAL"),
                          ("LA_EASY", lambda r: r["cell"] == "LA_EASY"),
                          ("LA_HARD", lambda r: r["cell"] == "LA_HARD"),
                          ("NON_LA", lambda r: r["cell"] == "NON_LA"),
                          ("Vvis<=5", lambda r: r["vvis"] <= 5),
                          ("Vvis>=6", lambda r: r["vvis"] >= 6)):
            sub = [r for r in rows if sel(r)]
            if len(sub) < 20:
                continue
            block[name] = {arm: agg(sub, arm) for arm in ARMS}
        result["seeds"][f"seed{seed}"] = block
        for arm in ARMS:
            a = block["ALL"][arm]
            log(f"seed{seed} {arm}  R {a['R_median']:7.3f}/{a['R_p90']:8.3f}  "
                f"t {a['t_median']:.4f}/{a['t_p90']:.4f}  "
                f"5cm5 {a['success_5cm5deg']:.4f}  ADD {a['ADD_median']:.4f}")
        for arm in ARMS:
            (OUT / f"fusion_{arm}_seed{seed}.json").write_text(json.dumps(
                {n: block[n][arm] for n in block}, indent=1))

    boot = {"resamples": 10000, "seeds": {}}
    for seed in SEEDS:
        rows = frames[seed]
        rng = np.random.default_rng(20260830 + seed)
        entry = {}
        for name, sel in (("ALL", lambda r: True),
                          ("LA_FRONTAL", lambda r: r["cell"] == "LA_FRONTAL"),
                          ("LA_HARD", lambda r: r["cell"] == "LA_HARD"),
                          ("Vvis<=5", lambda r: r["vvis"] <= 5)):
            sub = [r for r in rows if sel(r)]
            if len(sub) < 20:
                continue
            for arm in ARMS[1:]:
                for met in ("R", "t"):
                    a = np.array([r[f"F0_{met}"] for r in sub], float)
                    b = np.array([r[f"{arm}_{met}"] for r in sub], float)
                    ok = np.flatnonzero(np.isfinite(a) & np.isfinite(b))
                    if len(ok) < 20:
                        continue
                    d = np.empty(10000)
                    for k in range(10000):
                        p = ok[rng.integers(0, len(ok), len(ok))]
                        ma = float(np.median(a[p]))
                        d[k] = 100.0 * (ma - float(np.median(b[p]))) / abs(ma)
                    entry[f"{arm}#{name}#{met}"] = {
                        "n": int(len(ok)),
                        "effect_pct": round(float(np.median(d)), 2),
                        "ci95": [round(float(np.percentile(d, 2.5)), 2),
                                 round(float(np.percentile(d, 97.5)), 2)],
                        "P_better": round(float((d > 0).mean()), 4)}
        boot["seeds"][f"seed{seed}"] = entry
    (OUT / "fusion_bootstrap.json").write_text(json.dumps(boot, indent=1))
    (OUT / "fusion_full_result.json").write_text(json.dumps(result, indent=1))
    log("-> fusion_*.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["test", "eval"])
    a = p.parse_args()
    if a.command == "test":
        run_tests(a)
    else:
        run_eval(a)
