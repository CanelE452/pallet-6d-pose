"""PHASE 5 — 기전 분석.  새 ablation 없이 현재 출력만으로 실패를 분해한다.

5A  bbox 실패와 keypoint 실패를 가른다 (섞으면 원인을 못 찾는다)
5B  keypoint 별 오차·신뢰도·결측
5C  domain / 기하 축별
5D  확신에 찬 큰 오차 프레임 목록 (contact sheet 용)
"""
from __future__ import annotations

import csv, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0/model_compare", "scripts/stage0/real_eval",
            "scripts/stage0/stage_screens", "scripts/annotate", "challenge"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import cv2                    # noqa: E402
import mc_geom as MG          # noqa: E402
import re_metrics as RM       # noqa: E402
from stage25_paperbase_eval import elev_from_pose  # noqa: E402

PIPE = os.path.join(ROOT, "challenge/yolo_pose_one_model/paper_generic_pipeline")
ANA = os.path.join(ROOT, "challenge/yolo_pose_one_model/analysis")
DUMP = os.path.join(ROOT, "data/pallet/results/model_compare")
GROSS_PX, GROSS_R, GROSS_T = 30.0, 10.0, 0.2
SMALL_CELL = 20


def classify(row):
    if not row["box_detected"]:
        return "NO_BOX"
    if not np.isfinite(row["corner_med"]) or row["corner_med"] > GROSS_PX:
        return "BOX_OK_KP_BAD"
    if not row["pnp_ok"] or row["R"] > GROSS_R:
        return "POSE_BAD_DESPITE_KP"
    return "BOX_OK_KP_GOOD"


def main(model="yolo26n_paper_generic_v1"):
    os.makedirs(ANA, exist_ok=True)
    man = {i["frame_id"]: i for i in
           json.load(open(os.path.join(PIPE, "eval_manifest.json")))["items"]}
    dump = json.load(open(os.path.join(DUMP, f"kps_{model}.json")))

    rows = []
    for e in dump["frames"]:
        m = man[e["fid"]]
        gt8 = np.asarray(m["gt_corners_2d"], float)
        truth = {"R": np.asarray(m["R_gt"], float),
                 "t": np.asarray(m["t_gt"], float),
                 "K": np.asarray(m["K"], float),
                 "model": np.asarray(m["object_points"], float),
                 "gt8": gt8,
                 "extents": (m["dimensions_m"]["width"],
                             m["dimensions_m"]["height"],
                             m["dimensions_m"]["depth"])}
        px = MG.points_of(e, model)
        ok = np.isfinite(px).all(1)
        row = {"fid": e["fid"], "set": e["set"],
               "population": m["population"],
               "box_detected": int(e["kps"] is not None),
               "box_conf": e.get("box_conf") or np.nan,
               "n_points": int(ok.sum()),
               "corner_med": float(np.median(np.linalg.norm(
                   px[ok] - gt8[ok], axis=1))) if ok.any() else np.nan,
               "corner_p90": float(np.percentile(np.linalg.norm(
                   px[ok] - gt8[ok], axis=1), 90)) if ok.any() else np.nan}
        pose = MG.solve(px, truth) if ok.sum() >= 4 else None
        if pose is None:
            row.update({"pnp_ok": 0, "R": np.nan, "t": np.nan,
                        "adds": np.nan, "iou": np.nan, "5cm5": 0})
        else:
            R_p, t_p = pose
            deg, met = RM.pose_error(R_p, t_p, truth["R"], truth["t"])
            row.update({"pnp_ok": 1, "R": deg, "t": met,
                        "adds": RM.add_s(truth["model"], R_p, t_p,
                                         truth["R"], truth["t"]),
                        "iou": RM.iou_3d(R_p, t_p, truth["extents"],
                                         truth["R"], truth["t"],
                                         truth["extents"]),
                        "5cm5": int(RM.success_5cm5deg(R_p, t_p,
                                                       truth["R"], truth["t"]))})
        # 기하 축
        image = cv2.imread(os.path.join(ROOT, m["image"]))
        h, w = image.shape[:2]
        inside = ((gt8[:, 0] >= 0) & (gt8[:, 0] < w)
                  & (gt8[:, 1] >= 0) & (gt8[:, 1] < h))
        span = gt8.max(0) - gt8.min(0)
        row.update({"elev": elev_from_pose(np.eye(4).tolist()) if False else
                    elev_from_pose([[*r, 0] for r in truth["R"].tolist()]
                                   if False else
                                   np.vstack([np.hstack([truth["R"],
                                                         truth["t"][:, None]]),
                                              [0, 0, 0, 1]]).tolist()),
                    "n_outside": int((~inside).sum()),
                    "truncated": int((~inside).any()),
                    "size_ratio": float(np.hypot(*span) / np.hypot(w, h)),
                    "luma": float(np.median(cv2.cvtColor(
                        image, cv2.COLOR_BGR2GRAY)))})
        row["class"] = classify(row)
        # per-keypoint
        for i in range(8):
            row[f"kp{i}_err"] = (float(np.linalg.norm(px[i] - gt8[i]))
                                 if np.isfinite(px[i]).all() else np.nan)
            kc = e.get("kp_conf")
            row[f"kp{i}_conf"] = (float(kc[i]) if kc and i < len(kc)
                                  and kc[i] is not None else np.nan)
        rows.append(row)

    with open(os.path.join(ANA, "per_frame_analysis.csv"), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        wtr.writeheader(); wtr.writerows(rows)

    # 5A
    bbox = {}
    for pop in ("REAL_DEV_OPEN_56", "REAL_CHALLENGE_DEV_105"):
        sub = [r for r in rows if r["population"] == pop]
        cnt = {}
        for r in sub:
            cnt[r["class"]] = cnt.get(r["class"], 0) + 1
        bbox[pop] = {"n": len(sub), "classes": cnt,
                     "rates": {k: round(v / len(sub), 4) for k, v in cnt.items()}}
    json.dump(bbox, open(os.path.join(ANA, "bbox_vs_keypoint.json"), "w"), indent=1)

    # 5B
    kp = []
    for pop in ("REAL_DEV_OPEN_56", "REAL_CHALLENGE_DEV_105"):
        sub = [r for r in rows if r["population"] == pop]
        for i in range(8):
            e = np.array([r[f"kp{i}_err"] for r in sub], float)
            c = np.array([r[f"kp{i}_conf"] for r in sub], float)
            good = np.isfinite(e)
            kp.append({"population": pop, "keypoint": i,
                       "group_near_far": "near" if i < 4 else "far",
                       "group_top_bottom": "top" if i in (0, 1, 4, 5) else "bottom",
                       "n": int(good.sum()),
                       "err_median": round(float(np.median(e[good])), 3)
                       if good.any() else None,
                       "err_p90": round(float(np.percentile(e[good], 90)), 3)
                       if good.any() else None,
                       "conf_median": round(float(np.nanmedian(c)), 3)
                       if np.isfinite(c).any() else None,
                       "missing_rate": round(float(1 - good.mean()), 4)})
    with open(os.path.join(ANA, "per_keypoint_error.csv"), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(kp[0])); wtr.writeheader()
        wtr.writerows(kp)

    # 5C
    dom = []
    for pop in ("REAL_DEV_OPEN_56", "REAL_CHALLENGE_DEV_105"):
        for s in sorted({r["set"] for r in rows if r["population"] == pop}):
            sub = [r for r in rows if r["set"] == s]
            R = np.array([r["R"] for r in sub], float)
            dom.append({"population": pop, "set": s, "n": len(sub),
                        "exploratory": len(sub) < SMALL_CELL,
                        "availability": round(float(np.mean(
                            [r["box_detected"] for r in sub])), 4),
                        "corner_med": round(float(np.nanmedian(
                            [r["corner_med"] for r in sub])), 3),
                        "R_med": round(float(np.nanmedian(R)), 3)
                        if np.isfinite(R).any() else None,
                        "success_5cm5": round(float(np.mean(
                            [r["5cm5"] for r in sub])), 4)})
    with open(os.path.join(ANA, "domain_breakdown.csv"), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(dom[0])); wtr.writeheader()
        wtr.writerows(dom)

    # 5D
    tail = [r for r in rows
            if (np.isfinite(r["corner_med"]) and r["corner_med"] > GROSS_PX)
            or (np.isfinite(r["R"]) and r["R"] > GROSS_R)
            or (np.isfinite(r["t"]) and r["t"] > GROSS_T)]
    tail.sort(key=lambda r: -(r["R"] if np.isfinite(r["R"]) else 1e9))
    json.dump({"thresholds": {"corner_px": GROSS_PX, "R_deg": GROSS_R,
                              "t_m": GROSS_T},
               "n_tail": len(tail), "n_total": len(rows),
               "frames": [{k: r[k] for k in
                           ("fid", "set", "population", "box_conf", "corner_med",
                            "R", "t", "elev", "n_outside", "size_ratio",
                            "luma", "class")} for r in tail[:40]]},
              open(os.path.join(ANA, "gross_error_frames.json"), "w"),
              indent=1, default=str)

    # 5F confidence vs pose (real negative 없으므로 precision/AP 계산 안 함)
    conf = []
    for r in rows:
        if not r["box_detected"]:
            continue
        conf.append({"box_conf": r["box_conf"], "R": r["R"],
                     "correct_5cm5": r["5cm5"]})
    json.dump({"note": "real negative 가 없어 Precision/AP/FPR 은 계산하지 않는다",
               "n": len(conf), "rows": conf},
              open(os.path.join(ANA, "confidence_vs_pose.json"), "w"),
              indent=1, default=str)

    for pop, b in bbox.items():
        print(f"  {pop:24} {b['rates']}")
    print(f"  gross-error frames {len(tail)}/{len(rows)}")
    print("-> analysis/{per_frame_analysis.csv, bbox_vs_keypoint.json, "
          "per_keypoint_error.csv, domain_breakdown.csv, gross_error_frames.json, "
          "confidence_vs_pose.json}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "yolo26n_paper_generic_v1")
