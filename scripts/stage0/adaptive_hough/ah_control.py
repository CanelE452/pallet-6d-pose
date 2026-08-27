"""TRAINING-0 CONTROL — F0 / FP / FH / FO on identical predictions and solver.

새 학습 0.  lambda 는 synthetic GATE_DEV 에서만 고른다 (real 튜닝 금지).
GT support 는 inference 에 쓰지 않는다 (support 는 예측 코너에서 유도).
rho 는 pose objective 에 들어가지 않는다 (0.5*(da-db) 에서 정확히 소거).
"""
from __future__ import annotations
import hashlib, json, os, sys, time

import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0", "scripts/stage0/paper_s2", "scripts/stage0/multihead",
            "scripts/stage0/line", "scripts/stage0/real_eval", "challenge",
            "scripts/annotate"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import cv2                                        # noqa: E402
import paper_s2_real_eval as PRE                  # noqa: E402
import mh_data as MD                              # noqa: E402
import mh_screen as MS                            # noqa: E402
import mh_splitlate as SL                         # noqa: E402
import mh_fusion as FU                            # noqa: E402
import mh_diagnose as DG                          # noqa: E402
import mh_cigm as CG                              # noqa: E402
import line_feature_capacity_v2 as V2             # noqa: E402
import re_metrics as RM                           # noqa: E402
import annotate_pnp as APNP                       # noqa: E402
from mh_arms import DH                            # noqa: E402

NS = f"{ROOT}/data/pallet/results/adaptive_hough_g38"
CKROOT = f"{ROOT}/weights/paper_s2/paper_s2_multihead"
POOL = f"{ROOT}/data/pallet/training_data/paper_release/v2_prod40k_clean_merged"
YQ = (f"{ROOT}/challenge/yolo_pose_one_model/runs_camera_facing_loss/"
      "ubuntu_cf_loss_queue_20260823T0930")
MANI = ("/home/minjae/pallet_worker_transfer_20260821T105141Z/"
        "REAL_GT_QA_20260821T133405Z/REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")
LAMBDA_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]          # 기존 theta-only solver 의 grid 그대로
NIGHT = {"eval_night08", "eval_night09"}
OPEN_SETS = {"eval_outside", "eval_noapril", "eval_cad"}
GATE_NS = "G38_GATE_SPLIT_V1"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ gate split
def gate_split():
    p = f"{NS}/gate_split.json"
    if os.path.exists(p):
        return json.load(open(p))
    man = json.load(open(f"{YQ}/G38_GENERIC_ONLY_MANIFEST.json"))
    stems = sorted(s.replace("G__", "") for s in man["val"])
    key = lambda s: hashlib.sha1(f"{GATE_NS}|{s}".encode()).hexdigest()
    ordered = sorted(stems, key=key)
    half = len(ordered) // 2
    out = {"namespace": GATE_NS, "source": "G38 generic val (manifest 실측)",
           "n_val": len(stems), "GATE_FIT": sorted(ordered[:half]),
           "GATE_DEV": sorted(ordered[half:]),
           "rule": "sha1(namespace|stem) 오름차순 후 50:50, 한 번만 나눈다",
           "disjoint": len(set(ordered[:half]) & set(ordered[half:])) == 0,
           "★seen_by_FINAL40K": ("FINAL40K 는 BROAD 40,000 (train 38,002 + val 1,998) 을 "
                                 "전부 학습했다. 이 split 은 FINAL40K 에게 unseen 이 아니다 — "
                                 "lambda 절대값은 낙관 편향이나 FP/FH 는 같은 split·같은 "
                                 "grid·같은 rule 을 쓰므로 대조 자체는 공정하다.")}
    out["sha256"] = hashlib.sha256(
        ("\n".join(out["GATE_FIT"]) + "|" + "\n".join(out["GATE_DEV"])).encode()).hexdigest()
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    with open(f"{NS}/gate_split.jsonl", "w") as fh:
        for s in out["GATE_FIT"]:
            fh.write(json.dumps({"stem": s, "split": "GATE_FIT"}) + "\n")
        for s in out["GATE_DEV"]:
            fh.write(json.dumps({"stem": s, "split": "GATE_DEV"}) + "\n")
    return out


# ------------------------------------------------------------------ prediction
def px_to_grid(px, w, h, g=50):
    a = np.asarray(px, float)
    return np.stack([a[:, 0] * g / w, a[:, 1] * g / h], 1)


def undirected_err(t1, t2):
    d = np.abs(np.asarray(t1) - np.asarray(t2)) % np.pi
    return np.degrees(np.minimum(d, np.pi - d))


def load_model(seed):
    p = f"{CKROOT}/screen_A1_CORNER_LINE_FINAL40K_seed{seed}/step_25000.pth"
    st = torch.load(p, map_location=MD.DEV, weights_only=False)
    m = SL.SplitLate(st["arm"])
    m.load_state_dict(st["model"])
    m.to(MD.DEV).eval()
    return m, os.path.relpath(p, ROOT)


@torch.no_grad()
def predict(model, features, img_path, label):
    image = cv2.imread(img_path)
    if image is None:
        return None
    h, w = image.shape[:2]
    obj = label["objects"][0]
    dims = obj["dimensions_m"]
    model_pts = APNP.make_pallet_keypoints_3d_diagram(
        width=dims["width"], depth=dims["depth"], height=dims["height"])[:8]
    K = CG.intrinsics(label)
    R_gt, t_gt = CG.gt_pose(label)
    gt8 = np.asarray(obj["projected_cuboid"], float)[:8]

    out = model(PRE.preprocess_squash(image).to(MD.DEV), features)
    gth, grh, valid = MS.lattice()[0], MS.lattice()[1], MS.lattice()[2]
    th_hat, rh_hat = DH.decode(out["line_scores"], gth, grh, valid)
    th_H, rh_H = DH.canonical_from_centred(th_hat, rh_hat)
    beliefs = out["beliefs"]
    peaks = MS._decode_peaks(beliefs[-1][:, :9])[0]
    belief = beliefs[-1][0].detach().cpu().numpy()

    gt_grid9 = px_to_grid(np.vstack([gt8, np.asarray(
        obj["projected_cuboid_centroid"], float)]), w, h)
    th_P, rh_P, p0, p1, ln = V2.gt_lines(np.asarray(peaks, float)[None], CG.EDGES)
    th_G, _, _, _, _ = V2.gt_lines(gt_grid9[None], CG.EDGES)
    support = V2.visible_segments(p0, p1, ln)["hit"][0]     # 예측 코너 기반, GT 아님
    return {"w": w, "h": h, "model_pts": model_pts, "K": K, "R_gt": R_gt, "t_gt": t_gt,
            "extents": (dims["width"], dims["height"], dims["depth"]),
            "peaks": np.asarray(peaks, float),
            "th_P": th_P[0], "rh_P": rh_P[0],
            "th_H": th_H[0].cpu().numpy(), "rh_H": rh_H[0].cpu().numpy(),
            "th_G": th_G[0], "support": support,
            "corner_px": float(np.median(np.linalg.norm(
                CG.grid_to_pixels(np.asarray(peaks)[:8], w, h) - gt8, axis=1))),
            "peak_conf": np.max(belief[:9].reshape(9, -1), axis=1)}


# ------------------------------------------------------------------ solver
def pose_from(pred, theta, rho, lam):
    """F0 기반 rotation-only + translation refit. rho 는 objective 에서 소거된다."""
    w, h = pred["w"], pred["h"]
    corner_px = CG.grid_to_pixels(pred["peaks"][:8], w, h)
    base = CG.solve(pred["model_pts"], corner_px, pred["K"])
    if base is None:
        return None, None
    R_p, t_p = base
    if lam is None:
        return (R_p, t_p), (R_p, t_p)
    lines = DG._line_in_pixels(np.asarray(theta, float), np.asarray(rho, float), w, h)
    sup = np.asarray(pred["support"], bool)
    if not sup.any():
        return (R_p, t_p), (R_p, t_p)
    import cv2 as _cv
    rvec0, _ = _cv.Rodrigues(R_p)
    try:
        rvec = FU.rotation_only(rvec0.reshape(3), t_p, pred["model_pts"], pred["K"],
                                corner_px, lines, CG.EDGES, sup, lam)
        R_s, _ = _cv.Rodrigues(rvec)
        t_s = FU.translation_refit(R_s, t_p, pred["model_pts"], pred["K"], corner_px)
        return (R_p, t_p), (R_s, t_s)
    except Exception:
        return (R_p, t_p), None


def metrics(pose, pred):
    if pose is None:
        return {"ok": 0, "R": np.nan, "t": np.nan, "5cm5": 0, "adds": np.nan}
    R, t = pose
    deg, met = RM.pose_error(R, t, pred["R_gt"], pred["t_gt"])
    return {"ok": 1, "R": deg, "t": met,
            "5cm5": int(RM.success_5cm5deg(R, t, pred["R_gt"], pred["t_gt"])),
            "adds": RM.add_s(pred["model_pts"], R, t, pred["R_gt"], pred["t_gt"])}


def agg(rows, key, total):
    R = np.array([r[f"{key}_R"] for r in rows], float)
    t = np.array([r[f"{key}_t"] for r in rows], float)
    R, t = R[np.isfinite(R)], t[np.isfinite(t)]
    f = lambda a, q: (float(np.percentile(a, q)) if a.size else None)
    return {"n": len(rows),
            "solve_rate": round(sum(r.get(f"{key}_ok", 0) for r in rows) / max(total, 1), 4),
            "R_median": f(R, 50), "R_p90": f(R, 90),
            "t_median": f(t, 50), "t_p90": f(t, 90),
            "success_5cm5deg": round(sum(r.get(f"{key}_5cm5", 0) for r in rows)
                                     / max(total, 1), 4)}


# ------------------------------------------------------------------ lambda 선택
SAFETY = {"t_median_degrade_max": 0.03, "success_5cm5_drop_max": 0.0,
          "solve_rate_drop_max": 0.01,
          "rule": "생존자 중 R median 최소, 동률이면 작은 lambda",
          "source": "기존 theta-only solver 의 safety rule 을 그대로 재사용",
          "selected_on": "synthetic GATE_DEV only — real 128 은 튜닝에 쓰지 않는다"}


def sweep(preds, arm_key):
    """arm_key: 'P' or 'H'.  같은 grid, 같은 safety rule."""
    base = []
    for p in preds:
        f0, _ = pose_from(p, None, None, None)
        base.append(metrics(f0, p))
    b = {"solve_rate": np.mean([m["ok"] for m in base]),
         "R_median": float(np.median([m["R"] for m in base if np.isfinite(m["R"])])),
         "t_median": float(np.median([m["t"] for m in base if np.isfinite(m["t"])])),
         "success_5cm5deg": float(np.mean([m["5cm5"] for m in base]))}
    table = {}
    for lam in LAMBDA_GRID:
        ms = []
        for p in preds:
            th = p[f"th_{arm_key}"]
            rh = p[f"rh_{arm_key}"]
            _, arm = pose_from(p, th, rh, lam)
            ms.append(metrics(arm, p))
        R = np.array([m["R"] for m in ms], float)
        t = np.array([m["t"] for m in ms], float)
        e = {"solve_rate": float(np.mean([m["ok"] for m in ms])),
             "R_median": float(np.median(R[np.isfinite(R)])),
             "t_median": float(np.median(t[np.isfinite(t)])),
             "success_5cm5deg": float(np.mean([m["5cm5"] for m in ms]))}
        e["t_degrade"] = (e["t_median"] - b["t_median"]) / max(b["t_median"], 1e-12)
        e["d_5cm5"] = e["success_5cm5deg"] - b["success_5cm5deg"]
        e["d_solve"] = e["solve_rate"] - b["solve_rate"]
        e["SURVIVES"] = bool(e["t_degrade"] <= SAFETY["t_median_degrade_max"]
                             and e["d_5cm5"] >= -1e-12
                             and e["d_solve"] >= -SAFETY["solve_rate_drop_max"])
        table[str(lam)] = e
    surv = [(v["R_median"], float(k)) for k, v in table.items() if v["SURVIVES"]]
    chosen = (sorted(surv)[0][1] if surv else None)
    return {"baseline_F0": b, "grid": table, "survivors": sorted(surv),
            "chosen_lambda": chosen}


# ------------------------------------------------------------------ 프레임 평가
def evaluate(preds, meta, lam_P, lam_H):
    rows = []
    for p, m in zip(preds, meta):
        f0, _ = pose_from(p, None, None, None)
        r = dict(m)
        r.update({f"F0_{k}": v for k, v in metrics(f0, p).items()})
        for tag, key, lam in (("FP", "P", lam_P), ("FH", "H", lam_H)):
            _, arm = pose_from(p, p[f"th_{key}"], p[f"rh_{key}"], lam)
            r.update({f"{tag}_{k}": v for k, v in metrics(arm, p).items()})
        # FO — edge 별로 GT angle error 가 작은 쪽 theta 선택 (diagnostic only)
        eP = undirected_err(p["th_P"], p["th_G"])
        eH = undirected_err(p["th_H"], p["th_G"])
        win = eH < eP
        th_O = np.where(win, p["th_H"], p["th_P"])
        rh_O = np.where(win, p["rh_H"], p["rh_P"])
        _, arm = pose_from(p, th_O, rh_O, lam_H)
        r.update({f"FO_{k}": v for k, v in metrics(arm, p).items()})
        sup = np.asarray(p["support"], bool)
        r.update({"corner_px": p["corner_px"],
                  "e_pointtheta": float(np.median(eP[sup])) if sup.any() else np.nan,
                  "e_houghtheta": float(np.median(eH[sup])) if sup.any() else np.nan,
                  "hough_edge_win": float(win[sup].mean()) if sup.any() else np.nan,
                  "n_support": int(sup.sum())})
        r["d_FP_F0_R"] = r["F0_R"] - r["FP_R"]
        r["d_FH_F0_R"] = r["F0_R"] - r["FH_R"]
        r["d_FH_FP_R"] = r["FP_R"] - r["FH_R"]
        rows.append(r)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-max", type=int, default=500,
                    help="GATE_DEV 에서 lambda sweep 에 쓸 프레임 수 (결정론적 앞에서부터)")
    A = ap.parse_args()

    gs = gate_split()
    log(f"gate split  FIT {len(gs['GATE_FIT'])}  DEV {len(gs['GATE_DEV'])}  "
        f"disjoint {gs['disjoint']}  sha {gs['sha256'][:16]}")

    dev_stems = gs["GATE_DEV"][:A.dev_max]
    leak = set(json.load(open(f"{YQ}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
    real_items = [it for it in json.load(open(MANI))["items"]
                  if it["frame_id"] not in leak]
    features = MS.lattice()[3]

    OUT = {"gate_split_sha256": gs["sha256"], "lambda_grid": LAMBDA_GRID,
           "safety": SAFETY, "seeds": {}}
    PERF = {}
    for seed in (1, 2):
        model, ckpt = load_model(seed)
        log(f"seed{seed} {ckpt}")

        dev = []
        for i, s in enumerate(dev_stems):
            lab = json.load(open(f"{POOL}/labels/{s}_label.json"))
            p = predict(model, features, f"{POOL}/rgb/{s}_rgb.png", lab)
            if p:
                dev.append(p)
            if (i + 1) % 100 == 0:
                log(f"  DEV predict {i+1}/{len(dev_stems)}")
        log(f"  GATE_DEV n={len(dev)}  lambda sweep")
        selP, selH = sweep(dev, "P"), sweep(dev, "H")
        log(f"  lambda  FP={selP['chosen_lambda']}  FH={selH['chosen_lambda']}")

        preds, meta = [], []
        for it in real_items:
            jp, ip = os.path.join(ROOT, it["label"]), os.path.join(ROOT, it["image"])
            if not (os.path.exists(jp) and os.path.exists(ip)):
                continue
            lab = json.load(open(jp))
            p = predict(model, features, ip, lab)
            if p is None:
                continue
            st = it.get("set", "?")
            preds.append(p)
            meta.append({"fid": it["frame_id"], "set": st,
                         "domain": "NIGHT" if st in NIGHT else "DAY",
                         "group": "OPEN40" if st in OPEN_SETS else "NEW88"})
        log(f"  REAL n={len(preds)}  solve")
        rows = evaluate(preds, meta, selP["chosen_lambda"], selH["chosen_lambda"])

        entry = {"checkpoint": ckpt, "n_real": len(rows),
                 "lambda_selection": {"FP": selP, "FH": selH,
                                      "n_gate_dev_used": len(dev)},
                 "scopes": {}}
        for g, sub in (("ALL128", rows),
                       ("OPEN40", [r for r in rows if r["group"] == "OPEN40"]),
                       ("NEW88", [r for r in rows if r["group"] == "NEW88"])):
            entry["scopes"][g] = {a: agg(sub, a, len(sub))
                                  for a in ("F0", "FP", "FH", "FO")}
            cp = np.array([r["corner_px"] for r in sub], float)
            entry["scopes"][g]["corner_px"] = {
                "median": float(np.median(cp)), "p90": float(np.percentile(cp, 90))}
            ep = np.array([r["e_pointtheta"] for r in sub], float)
            eh = np.array([r["e_houghtheta"] for r in sub], float)
            entry["scopes"][g]["e_pointtheta_median"] = float(np.nanmedian(ep))
            entry["scopes"][g]["e_houghtheta_median"] = float(np.nanmedian(eh))
            entry["scopes"][g]["hough_edge_win_rate"] = float(
                np.nanmean([r["hough_edge_win"] for r in sub]))
            d = np.array([r["d_FH_FP_R"] for r in sub], float)
            d = d[np.isfinite(d)]
            entry["scopes"][g]["FH_minus_FP"] = {
                "R_median_delta": float(np.median(d)) if d.size else None,
                "frac_FH_better": float((d > 0).mean()) if d.size else None,
                "n": int(d.size)}
        OUT["seeds"][f"seed{seed}"] = entry
        PERF[f"seed{seed}"] = rows
        del model
        torch.cuda.empty_cache()

    # -------------------------------------------------- VERDICT (FH vs FP)
    checks = {}
    for s, e in OUT["seeds"].items():
        a = e["scopes"]["ALL128"]
        FP, FH = a["FP"], a["FH"]
        t_deg = (FH["t_median"] - FP["t_median"]) / max(FP["t_median"], 1e-12)
        checks[s] = {
            "R_median_FP": FP["R_median"], "R_median_FH": FH["R_median"],
            "R_p90_FP": FP["R_p90"], "R_p90_FH": FH["R_p90"],
            "R_median_gain": (FP["R_median"] - FH["R_median"]) / max(FP["R_median"], 1e-12),
            "t_degrade": t_deg,
            "d_5cm5": FH["success_5cm5deg"] - FP["success_5cm5deg"],
            "frac_FH_better": a["FH_minus_FP"]["frac_FH_better"],
        }
        checks[s]["IMPROVES"] = bool(checks[s]["R_median_gain"] > 0
                                     and t_deg <= 0.03
                                     and checks[s]["d_5cm5"] >= 0.0)
    consistent = all(v["IMPROVES"] for v in checks.values())
    VERDICT = ("HOUGH_ADDS_INFORMATION_BEYOND_REWEIGHTING" if consistent
               else "HOUGH_INCREMENTAL_VALUE_NOT_ESTABLISHED")
    OUT["FH_vs_FP"] = checks
    OUT["VERDICT"] = VERDICT
    OUT["NEXT"] = ("Hough-private FPN/PAN + 4-feature gate 구현/학습 계속"
                   if consistent else "Hough-private FPN/PAN training STOP")
    OUT["HARD_RESCUE_HYPOTHESIS"] = {
        "status": "FAIL",
        "why": ("HARD_CRITERION_UNREACHABLE_BY_ORACLE — per-frame ARM oracle 로도 "
                "hard 25% R p90 개선이 0.0% (seed1) / 4.6% (seed2) 로 문턱 10% 미달. "
                "hard 정의는 결과를 보고 수정하지 않았다."),
        "recorded_at": "STEP A, training 전"}
    OUT["NEW88_FAILURE_MODE"] = {
        "name": "POINT_LOCALIZATION_COLLAPSE",
        "evidence": "NEW88 corner px median ~89 (OPEN40 ~8)",
        "meaning": ("orientation correction 으로 구할 수 없는 subset. adaptive-Hough 의 "
                    "'hard rescue' 성능으로 해석하지 말 것."),
        "affects": "current same-real n=128 의 88/128 (69%)"}
    OUT["★caveat"] = ("lambda 는 GATE_DEV(G38 val)에서 골랐는데 FINAL40K 는 val 을 이미 "
                      "학습했다 — 절대값은 낙관 편향. FP/FH 는 같은 split·grid·rule 이라 "
                      "대조는 공정. real 128 은 튜닝에 쓰지 않았다.")
    json.dump(OUT, open(f"{NS}/TRAINING0_CONTROL.json", "w"), indent=2, ensure_ascii=False)

    import csv
    with open(f"{NS}/TRAINING0_CONTROL_PER_FRAME.csv", "w", newline="") as fh:
        keys = sorted({k for rows in PERF.values() for r in rows for k in r})
        wr = csv.DictWriter(fh, fieldnames=["seed"] + keys, extrasaction="ignore")
        wr.writeheader()
        for s, rows in PERF.items():
            for r in rows:
                wr.writerow({"seed": s, **r})
    log(f"VERDICT = {VERDICT}")
    print(json.dumps({"VERDICT": VERDICT, "FH_vs_FP": checks}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
