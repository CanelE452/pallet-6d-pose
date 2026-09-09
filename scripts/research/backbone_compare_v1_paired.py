"""백본 비교 paired 재집계 — 두 arm 모두 예측이 있는 공통 프레임에서만.

앞선 표는 YOLO 319 대 DOPE 177 로 **모집단이 달랐다**.  여기서는 공통 프레임으로
잘라 같은 모집단에서 비교한다.  이 잘라내기는 DOPE 가 검출한 쉬운 프레임만 남기므로
**DOPE 에 유리한 쪽으로 기운다** — 그래도 밀리면 그 결론은 더 강해진다.

정본 evaluator 와 같은 selector·GT·metric 함수를 import 한다.  새 추론 0.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[2]
for sub in ("scripts/paper/pose_metric_closure_v1", "scripts/evaluation",
            "scripts/self_training_yolo", ""):
    sys.path.insert(0, str(ROOT / sub) if sub else str(ROOT))
C = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
OUT = ROOT / "data/pallet/results/backbone_compare_v1"
ARMS = {"YOLO26n_R0": "R0", "DOPE_G38": "BB_DOPE_G38"}
CF_W, CF_D = "CF_WIDTH", "CF_DEPTH"


def cuboid(a, h, b):
    ha, hh, hb = a/2, h/2, b/2
    return np.array([[-ha,-hh,-hb],[ha,-hh,-hb],[ha,hh,-hb],[-ha,hh,-hb],
                     [-ha,-hh,hb],[ha,-hh,hb],[ha,hh,hb],[-ha,hh,hb]], float)


def solve(model, pts, cam, usable):
    ok, rv, tv = cv2.solvePnP(model[usable], pts[usable], cam, None,
                              flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(model[usable], pts[usable], cam, None, rv, tv)
    R, _ = cv2.Rodrigues(rv)
    return R, tv.reshape(-1)


def main():
    from pose_evaluation_paths import load_pose_object_contract, object_spec
    from symmetry_aware_pose_metrics import (cuboid_model_points, model_diameter_m,
                                             pose_auc, rotation_error_degrees,
                                             symmetry_aware_add_m,
                                             translation_components_m, yaw_error_degrees)
    from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses

    contract = load_pose_object_contract(str(C/"POSE_EVAL_OBJECT_CONTRACT.json"))
    gt = json.loads((C/"GEOMETRY_RESOLVED_POSE_GT.json").read_text())["frames"]
    man = {f["frame_id"]: f for f in json.loads((C/"AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]}

    per = {}
    for label, arm in ARMS.items():
        payload = json.loads((C/"predictions"/f"{arm}.json").read_text())["frames"]
        preds = {k: v for k, v in payload.items()
                 if v.get("status") == "OK" and v.get("keypoints_xy")}
        rows = {}
        for fid, truth in gt.items():
            pr = preds.get(fid)
            if not pr:
                continue
            fr = man[fid]
            raw = json.loads((ROOT/fr["annotation"]).read_text())["camera_data"]["intrinsics"]
            cam = np.array([[raw["fx"],0,raw["cx"]],[0,raw["fy"],raw["cy"]],[0,0,1]], float)
            spec = object_spec(contract, fr["object_type"])
            L, S, H = spec["long_m"], spec["short_m"], spec["height_m"]
            models = {CF_W: cuboid(L,H,S), CF_D: cuboid(S,H,L)}
            pts = np.asarray(pr["keypoints_xy"], float)[:8]
            usable = np.isfinite(pts).all(axis=1)
            if usable.sum() < 6:
                continue
            chosen = None
            try:
                res = select_pnp_hypotheses(np.nan_to_num(np.asarray(pr["keypoints_xy"], float)),
                                            cam, {"x": L, "y": H, "z": S}, None)
                for hyp in res.hypotheses:
                    if hyp.name == res.selected_hypothesis and hyp.success:
                        dd = hyp.camera_facing_dimensions.as_dict()
                        chosen = CF_W if abs(float(dd["width"]) - L) < 1e-6 else CF_D
            except Exception:
                chosen = None
            if chosen is None:
                continue
            fit = solve(models[chosen], pts, cam, usable)
            if fit is None:
                continue
            R, t = fit
            gR = np.asarray(truth["R_gt_representative"], float)
            gt_t = np.asarray(truth["t_gt"], float)
            dm = truth["physical_dimensions_m"]
            ext = (dm["across"], dm["height"], dm["along"])
            mp = cuboid_model_points(ext)
            parts = translation_components_m(t, gt_t)
            rows[fid] = {"object_type": fr["object_type"], "session_id": fr["session_id"],
                         "axis_correct": chosen == truth["physical_long_axis"],
                         "rot": rotation_error_degrees(R, gR),
                         "yaw": yaw_error_degrees(R, gR),
                         "t_cm": parts["total_m"]*100, "depth_cm": parts["depth_m"]*100,
                         "lat_cm": parts["lateral_m"]*100,
                         "iou3d": oriented_iou_3d(R, t, ext, gR, gt_t, ext),
                         "add": symmetry_aware_add_m(mp, R, t, gR, gt_t),
                         "dia": model_diameter_m(mp)}
        per[label] = rows
        print(f"  {label:12s} usable {len(rows)}")

    common = sorted(set(per["YOLO26n_R0"]) & set(per["DOPE_G38"]))
    print(f"\n공통 프레임 {len(common)}")

    def summ(rows, keys):
        a = lambda k: np.array([rows[f][k] for f in keys], float)
        return {"n": len(keys),
                "axis_accuracy": float(np.mean([rows[f]["axis_correct"] for f in keys])),
                "t_median_cm": float(np.median(a("t_cm"))),
                "t_p90_cm": float(np.percentile(a("t_cm"), 90)),
                "depth_median_cm": float(np.median(a("depth_cm"))),
                "depth_p90_cm": float(np.percentile(a("depth_cm"), 90)),
                "lateral_median_cm": float(np.median(a("lat_cm"))),
                "rot_median_deg": float(np.median(a("rot"))),
                "yaw_median_deg": float(np.median(a("yaw"))),
                "iou3d_median": float(np.median(a("iou3d"))),
                "add_sym_auc": pose_auc(a("add"), float(np.median(a("dia"))))}

    rep = {"schema_version": "backbone_compare_paired_v1", "new_inference": 0,
           "n_common": len(common),
           "note": ("공통 프레임만. DOPE 가 검출한 쉬운 프레임으로 좁히는 것이므로 "
                    "DOPE 에 유리한 쪽으로 기운 비교다."),
           "paired": {k: summ(per[k], common) for k in ARMS},
           "coverage": {k: {"usable": len(per[k]), "of": len(gt)} for k in ARMS}}
    for m in ("plastic", "wood"):
        sel = [f for f in common if per["YOLO26n_R0"][f]["object_type"].startswith(m)]
        rep[f"paired_{m}"] = {k: summ(per[k], sel) for k in ARMS} if len(sel) >= 5 else {}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"BACKBONE_PAIRED.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False)+"\n")

    y, d = rep["paired"]["YOLO26n_R0"], rep["paired"]["DOPE_G38"]
    print(f"\n{'metric':18s}{'YOLO26n':>12s}{'DOPE':>12s}{'ratio':>10s}")
    print("-"*52)
    for lab, k in (("axis acc","axis_accuracy"),("t median cm","t_median_cm"),
                   ("t p90 cm","t_p90_cm"),("depth median cm","depth_median_cm"),
                   ("depth p90 cm","depth_p90_cm"),("lateral cm","lateral_median_cm"),
                   ("R median deg","rot_median_deg"),("yaw median deg","yaw_median_deg"),
                   ("IoU3D","iou3d_median"),("ADD-S AUC","add_sym_auc")):
        print(f"{lab:18s}{y[k]:>12.4f}{d[k]:>12.4f}{d[k]/y[k] if y[k] else float('nan'):>9.2f}x")
    for m in ("plastic","wood"):
        r = rep.get(f"paired_{m}")
        if r:
            print(f"{m:8s} n {r['YOLO26n_R0']['n']:3d}   t {r['YOLO26n_R0']['t_median_cm']:7.3f} / "
                  f"{r['DOPE_G38']['t_median_cm']:7.3f}   depth {r['YOLO26n_R0']['depth_median_cm']:7.3f} / "
                  f"{r['DOPE_G38']['depth_median_cm']:7.3f}")
    print(f"\nwrote {(OUT/'BACKBONE_PAIRED.json').relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
