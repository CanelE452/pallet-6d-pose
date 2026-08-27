"""STEP A (5-7) — FEATURE_TAP / current-real n=128 / point-vs-Hough ORACLE headroom.

새 학습 0.  기존 파일 overwrite 0.  GT 는 진단에만 쓰고 어떤 arm 의 inference 에도
들어가지 않는다 (oracle 은 diagnostic 전용이라고 명시).
"""
from __future__ import annotations
import hashlib, json, os, sys

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
import mh_cigm as CG                              # noqa: E402
import line_feature_capacity_v2 as V2             # noqa: E402
import re_metrics as RM                           # noqa: E402
import annotate_pnp as APNP                       # noqa: E402
import mh_arms as MH                              # noqa: E402
from mh_arms import DH                            # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief   # noqa: E402

NS = f"{ROOT}/data/pallet/results/adaptive_hough_g38"
CKROOT = f"{ROOT}/weights/paper_s2/paper_s2_multihead"
MANI = ("/home/minjae/pallet_worker_transfer_20260821T105141Z/"
        "REAL_GT_QA_20260821T133405Z/REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")
YQ = (f"{ROOT}/challenge/yolo_pose_one_model/runs_camera_facing_loss/"
      "ubuntu_cf_loss_queue_20260823T0930")
LAMBDA_THETA = 3.0                    # 기존 REAL_DEV_F0_F3_EVAL 이 쓴 값, 새로 만들지 않음
NIGHT_SETS = {"eval_night08", "eval_night09"}
THRESH, N_DET_MIN = 0.3, 6

# ---------------------------------------------------------------- FEATURE TAP
def feature_tap_audit():
    model = SL.SplitLate("A1_CORNER_LINE")
    model.eval()
    dummy = torch.zeros(1, 3, 400, 400, device=MD.DEV)
    rows, x = [], dummy
    vgg_children = list(model.a1.vgg.named_children())
    loaded = {id(p) for p in model.a1.vgg.parameters()}
    with torch.no_grad():
        for name, mod in vgg_children:
            x = mod(x)
            ps = list(mod.parameters())
            rows.append({"index": int(name), "module": type(mod).__name__,
                         "shape": list(x.shape),
                         "spatial": int(x.shape[-1]),
                         "channels": int(x.shape[1]),
                         "requires_grad": bool(any(p.requires_grad for p in ps)),
                         "has_params": bool(ps),
                         "checkpoint_loaded": bool(all(id(p) in loaded for p in ps))})
    # 목표 scale 후보
    by_spatial = {}
    for r in rows:
        by_spatial.setdefault(r["spatial"], []).append(r["index"])
    out = {"input": [1, 3, 400, 400],
           "first_trainable_vgg": MH.FIRST_TRAINABLE_VGG,
           "children": rows,
           "spatial_levels": {str(k): v for k, v in sorted(by_spatial.items(),
                                                           reverse=True)},
           "f50_shape": list(model.line_late(model.early(dummy)).shape)
           if True else None}
    have = set(by_spatial)
    out["target_scales"] = {
        "high_100": (max(by_spatial[100]) if 100 in have else None),
        "middle_50": (max(by_spatial[50]) if 50 in have else None),
        "low_25": (max(by_spatial[25]) if 25 in have else None)}
    out["needs_synthetic_low_level"] = out["target_scales"]["low_25"] is None
    out["note"] = ("low 25x25 이 trunk 에 없으면 Hough neck 내부에서 "
                   "middle 50 -> stride2 3x3 conv 로 만든다. backbone 은 건드리지 않는다.")
    json.dump(out, open(f"{NS}/FEATURE_TAP_AUDIT.json", "w"), indent=2, ensure_ascii=False)
    del model
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- membership
def membership():
    leak = set(json.load(open(f"{YQ}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
    items = json.load(open(MANI))["items"]
    rows = []
    for it in items:
        if it["frame_id"] in leak:
            continue
        jp, ip = os.path.join(ROOT, it["label"]), os.path.join(ROOT, it["image"])
        if not (os.path.exists(jp) and os.path.exists(ip)):
            continue
        s = it.get("set", "?")
        rows.append({"fid": it["frame_id"], "json": jp, "img": ip, "set": s,
                     "domain": "NIGHT" if s in NIGHT_SETS else "DAY"})
    return rows, len(items), len(leak)


def px_to_grid(px, w, h, g=50):
    a = np.asarray(px, float)
    return np.stack([a[:, 0] * g / w, a[:, 1] * g / h], 1)


def undirected_angle_err(t1, t2):
    """두 undirected line 각(라디안) 사이 오차(도). wrap-aware."""
    d = np.abs(t1 - t2) % np.pi
    return np.degrees(np.minimum(d, np.pi - d))


# ---------------------------------------------------------------- per frame
def run_frame(model, features, r):
    image = cv2.imread(r["img"])
    if image is None:
        return None
    h, w = image.shape[:2]
    label = json.load(open(r["json"]))
    obj = label["objects"][0]
    dims = obj["dimensions_m"]
    extents = (dims["width"], dims["height"], dims["depth"])
    model_pts = APNP.make_pallet_keypoints_3d_diagram(
        width=dims["width"], depth=dims["depth"], height=dims["height"])[:8]
    K = CG.intrinsics(label)
    R_gt, t_gt = CG.gt_pose(label)
    gt8 = np.asarray(obj["projected_cuboid"], float)[:8]

    with torch.no_grad():
        out = model(PRE.preprocess_squash(image).to(MD.DEV), features)
        beliefs = out["beliefs"]
        scores = out["line_scores"]
        gt_th, gt_rh, valid = DH.lattice()
        theta_hat, rho_hat = DH.decode(scores, gt_th, gt_rh, valid)
        theta_H, rho_H = DH.canonical_from_centred(theta_hat, rho_hat)
        # gate feature 재료 (GT 아님)
        logp = torch.log_softmax(scores.masked_fill(~valid[None, None], -1e9), -1)
        p = logp.exp()
        ent = -(p * logp).sum(-1)[0].cpu().numpy()
        n_valid = int(valid.sum())
        ent_norm = ent / np.log(max(n_valid, 2))
        top2 = torch.topk(p[0], 2, dim=-1).values.cpu().numpy()
        margin = top2[:, 0] - top2[:, 1]
    belief = beliefs[-1][0].detach().cpu().numpy()
    peaks = MS._decode_peaks(beliefs[-1][:, :9])[0]
    peak_conf = np.max(belief[:9].reshape(9, -1), axis=1)
    n_det = int(sum(1 for k in extract_keypoints_from_belief(belief, THRESH)[:8]
                    if k[0] >= 0))

    gt_grid9 = px_to_grid(np.vstack([gt8, np.asarray(
        obj["projected_cuboid_centroid"], float)]), w, h)

    th_P, rh_P, p0, p1, ln = V2.gt_lines(np.asarray(peaks, float)[None], CG.EDGES)
    th_G, rh_G, _, _, _ = V2.gt_lines(gt_grid9[None], CG.EDGES)
    th_P, rh_P, th_G = th_P[0], rh_P[0], th_G[0]
    th_H = theta_H[0].cpu().numpy()
    rh_Hn = rho_H[0].cpu().numpy()

    e_point = undirected_angle_err(th_P, th_G)
    e_hough = undirected_angle_err(th_H, th_G)
    hough_wins = e_hough < e_point

    support = V2.visible_segments(*V2.gt_lines(np.asarray(peaks, float)[None],
                                               CG.EDGES)[2:])["hit"][0]

    th_O = np.where(hough_wins, th_H, th_P)
    rh_O = np.where(hough_wins, rh_Hn, rh_P)

    data = {"resolution": np.array([[w, h]]), "model": np.array([model_pts]),
            "K": np.array([K]), "pred_corner": np.array([peaks]),
            "support": np.array([support])}
    row = {"fid": r["fid"], "set": r["set"], "domain": r["domain"],
           "n_det": n_det, "det": int(n_det >= N_DET_MIN),
           "e_point": e_point.tolist(), "e_hough": e_hough.tolist(),
           "hough_wins": hough_wins.tolist(), "support": support.tolist(),
           "hough_entropy": ent_norm.tolist(), "hough_margin": margin.tolist(),
           "peak_conf": peak_conf.tolist()}

    for arm_tag, (th, rh) in (("HOUGH", (th_H, rh_Hn)),
                              ("POINTTHETA", (th_P, rh_P)),
                              ("ORACLE", (th_O, rh_O))):
        data["pred_theta"] = np.asarray(th, float)[None]
        data["pred_rho"] = np.asarray(rh, float)[None]
        arms, corner_px, _, _ = FU.solve_arms(data, 0, LAMBDA_THETA)
        for arm in ("F0", "F3"):
            if arm == "F0" and arm_tag != "HOUGH":
                continue                       # F0 는 line 을 안 쓰므로 한 번만
            key = "F0" if arm == "F0" else f"F3_{arm_tag}"
            pose = arms.get(arm)
            if pose is None:
                row.update({f"{key}_ok": 0, f"{key}_R": np.nan, f"{key}_t": np.nan,
                            f"{key}_5cm5": 0, f"{key}_adds": np.nan})
                continue
            Rp, tp = pose
            deg, met = RM.pose_error(Rp, tp, R_gt, t_gt)
            row.update({f"{key}_ok": 1, f"{key}_R": deg, f"{key}_t": met,
                        f"{key}_adds": RM.add_s(model_pts, Rp, tp, R_gt, t_gt),
                        f"{key}_5cm5": int(RM.success_5cm5deg(Rp, tp, R_gt, t_gt))})
    # corner px error
    cpx = CG.grid_to_pixels(np.asarray(peaks)[:8], w, h)
    row["corner_px"] = float(np.median(np.linalg.norm(cpx - gt8, axis=1)))
    return row


def stats(v):
    a = np.asarray([x for x in v if np.isfinite(x)], float)
    if a.size == 0:
        return {"n": 0, "median": None, "p90": None}
    return {"n": int(a.size), "median": round(float(np.median(a)), 4),
            "p90": round(float(np.percentile(a, 90)), 4)}


def summarise(rows, key, total):
    return {"solve_rate": round(sum(r.get(f"{key}_ok", 0) for r in rows) / max(total, 1), 4),
            "R_deg": stats([r.get(f"{key}_R", np.nan) for r in rows]),
            "t_m": stats([r.get(f"{key}_t", np.nan) for r in rows]),
            "ADD_S": stats([r.get(f"{key}_adds", np.nan) for r in rows]),
            "success_5cm5deg": round(sum(r.get(f"{key}_5cm5", 0) for r in rows)
                                     / max(total, 1), 4)}


def main():
    print("[1/3] FEATURE TAP AUDIT", flush=True)
    tap = feature_tap_audit()
    print("  levels:", tap["spatial_levels"], " targets:", tap["target_scales"], flush=True)

    rows_m, n_manifest, n_leak = membership()
    print(f"[2/3] REAL membership {n_manifest} - leak {n_leak} -> n={len(rows_m)}", flush=True)

    _, _, valid = DH.lattice()
    features = DH.lattice_features() if hasattr(DH, "lattice_features") else None
    if features is None:
        gt_th, gt_rh, valid = MS.lattice()[0], MS.lattice()[1], MS.lattice()[2]
        features = MS.lattice()[3]
    RES, PER = {}, {}
    for seed in (1, 2):
        d = f"{CKROOT}/screen_A1_CORNER_LINE_FINAL40K_seed{seed}/step_25000.pth"
        st = torch.load(d, map_location=MD.DEV, weights_only=False)
        model = SL.SplitLate(st["arm"])
        model.load_state_dict(st["model"])
        model.to(MD.DEV).eval()
        rows = []
        for i, r in enumerate(rows_m):
            out = run_frame(model, features, r)
            if out:
                rows.append(out)
            if (i + 1) % 32 == 0:
                print(f"    seed{seed} {i+1}/{len(rows_m)}", flush=True)
        PER[f"seed{seed}"] = rows
        n = len(rows)
        entry = {"checkpoint": os.path.relpath(d, ROOT), "n_frames": n,
                 "lambda_theta": LAMBDA_THETA}
        for scope, sub in (("ALL", rows),
                           ("DAY", [r for r in rows if r["domain"] == "DAY"]),
                           ("NIGHT", [r for r in rows if r["domain"] == "NIGHT"])):
            entry[scope] = {k: summarise(sub, k, len(sub))
                            for k in ("F0", "F3_HOUGH", "F3_POINTTHETA", "F3_ORACLE")}
            entry[scope]["corner_px"] = stats([r["corner_px"] for r in sub])
            entry[scope]["n"] = len(sub)
        # edge level
        ep = np.concatenate([np.array(r["e_point"])[np.array(r["support"], bool)]
                             for r in rows if any(r["support"])])
        eh = np.concatenate([np.array(r["e_hough"])[np.array(r["support"], bool)]
                             for r in rows if any(r["support"])])
        entry["edge_level"] = {
            "n_supported_edges": int(ep.size),
            "e_point": stats(ep), "e_hough": stats(eh),
            "hough_win_rate": round(float((eh < ep).mean()), 4),
            "oracle_e": stats(np.minimum(ep, eh))}
        RES[f"seed{seed}"] = entry
        del model
        torch.cuda.empty_cache()
        print(f"  seed{seed} done  n={n}", flush=True)

    # ------------------------------------------------ ORACLE STOP GATE
    GATE = {"R_p90_gain_min": 0.10, "5cm5_gain_min_pp": 0.03,
            "rule": "둘 중 하나라도 넘으면 PROCEED, 둘 다 미달이면 STOP",
            "frozen_before_looking": True}
    verdicts = {}
    for s, e in RES.items():
        f0, orc = e["ALL"]["F0"], e["ALL"]["F3_ORACLE"]
        rp0, rpo = f0["R_deg"]["p90"], orc["R_deg"]["p90"]
        gain_r = (rp0 - rpo) / rp0 if (rp0 and rpo is not None) else None
        gain_5 = orc["success_5cm5deg"] - f0["success_5cm5deg"]
        verdicts[s] = {"R_p90_point": rp0, "R_p90_oracle": rpo,
                       "R_p90_gain": None if gain_r is None else round(gain_r, 4),
                       "5cm5_point": f0["success_5cm5deg"],
                       "5cm5_oracle": orc["success_5cm5deg"],
                       "5cm5_gain_pp": round(gain_5, 4),
                       "PASS_R": bool(gain_r is not None and gain_r >= GATE["R_p90_gain_min"]),
                       "PASS_5cm5": bool(gain_5 >= GATE["5cm5_gain_min_pp"])}
        verdicts[s]["PROCEED"] = verdicts[s]["PASS_R"] or verdicts[s]["PASS_5cm5"]
    DECISION = ("PROCEED" if any(v["PROCEED"] for v in verdicts.values())
                else "ADAPTIVE_HOUGH_HEADROOM_TOO_SMALL")

    payload = {"gate": GATE, "per_seed": verdicts, "DECISION": DECISION,
               "results": RES,
               "membership": {"manifest": MANI, "n_manifest": n_manifest,
                              "leak_removed": n_leak, "n_used": len(rows_m)},
               "★scope": ("oracle 은 GT 를 보는 진단 상한이며 deployment 결과가 아니다. "
                          "real 은 development evaluation 전용 — gate/lambda 학습에 "
                          "사용하지 않는다. n=128 중 88 장이 ★final-test 세션이다.")}
    json.dump(payload, open(f"{NS}/ORACLE_HEADROOM.json", "w"), indent=2, ensure_ascii=False)
    np.savez_compressed(f"{NS}/stepA_per_frame.npz",
                        **{f"{s}_json": np.array([json.dumps(PER[s])]) for s in PER})
    print(json.dumps({"DECISION": DECISION, "per_seed": verdicts}, indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
