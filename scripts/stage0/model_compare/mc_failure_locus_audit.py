"""남은 실패가 다섯 축 중 어디인가 — 기존 산출물만으로 가른다.  학습 0, 추론 0.

    (1) box threshold  (2) top-1 selection  (3) metric scale / dimensions
    (4) symmetry / near-far  (5) topology / appearance coverage

새 추론은 하지 않는다.  `_cc_raw_dump.json`(conf=0.001 전수 후보) 의 키포인트를
그대로 읽어 PnP 만 다시 푼다.  기존 파일은 하나도 건드리지 않고 timestamp 디렉터리에만
쓴다.

이미 있는 것을 다시 계산하는 이유는 하나뿐이다.  기존 `YOLO_CONF_SWEEP.json` 의
`success_5cm5` 는 **분모가 threshold 마다 바뀐다**(available 프레임 수).  그래서
threshold 를 낮추면 값이 내려가는 것처럼 보인다.  브리프가 요구하는 것은 전체
positive 를 분모로 하는 unconditional 이고, 그 둘은 다른 것을 말한다.

Phase E 의 nominal 은 결과를 보고 고르지 않는다.  `annotate_pnp.PALLET_DIMS`
(1.1, 1.3, 0.11) 를 그대로 쓴다 — 코드에 이미 있던 상수다.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0/model_compare", "scripts/stage0/real_eval",
            "scripts/annotate", "challenge"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import cv2                                        # noqa: E402
import re_metrics as RM                           # noqa: E402
import annotate_pnp as APNP                       # noqa: E402

A2 = os.path.join(ROOT, "challenge/yolo_pose_one_model/analysis_pre_v2")
PIPE = os.path.join(ROOT, "challenge/yolo_pose_one_model/paper_generic_pipeline")
V2DIR = os.path.join(ROOT, "challenge/yolo_pose_one_model/broad_family_v2")
STAMP = os.environ.get("AUDIT_STAMP", "20260821T1449")
OUT = os.path.join(ROOT, f"challenge/yolo_pose_one_model/audit_{STAMP}")

IOU_MATCH = 0.5              # 기존 계약과 동일
GROSS_R = 10.0               # 기존 계약과 동일
RMAX_SLACK = 0.005           # 브리프: top1_recall >= Rmax - 0.005
NOMINAL = APNP.PALLET_DIMS   # (1.1, 1.3, 0.11) = (width, depth, height)


def log(message):
    print(message, flush=True)


def bbox_of(points, width, height):
    p = np.asarray(points, float)
    return [max(0.0, float(p[:, 0].min())), max(0.0, float(p[:, 1].min())),
            min(float(width), float(p[:, 0].max())),
            min(float(height), float(p[:, 1].max()))]


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
             + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def solve(points2d, model3d, K):
    px = np.asarray(points2d, float)[:8]
    if not np.isfinite(px).all():
        return None
    ok, rvec, tvec = cv2.solvePnP(model3d[:8], px.reshape(-1, 1, 2), K, None,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(model3d[:8], px.reshape(-1, 1, 2), K,
                                      None, rvec, tvec)
    return cv2.Rodrigues(rvec)[0], tvec.reshape(3)


def load():
    raw = json.load(open(os.path.join(A2, "_cc_raw_dump.json")))
    manifest = json.load(open(os.path.join(PIPE, "eval_manifest.json")))
    items = {i["frame_id"]: i for i in manifest["items"]}
    return raw, manifest, items


# --------------------------------------------------------------- phase B
def contract(raw, manifest):
    return {
        "generated": STAMP,
        "positive_denominator": {"n": manifest["n_total"], "source":
                                 "paper_generic_pipeline/eval_manifest.json",
                                 "populations": manifest["counts"]},
        "negative_denominator": {"n": len(raw["negative"]),
                                 "source": "analysis_pre_v2/REAL_NEG_DEV_AUDIT.json",
                                 "★bias": "prepare_real_ft.py 가 max_conf<0.20 인 프레임만 "
                                          "채택(911 중 259). 이전 모델이 오검출한 프레임이 "
                                          "체계적으로 빠졌으므로 FP/image 는 하한이다."},
        "box_correctness": {"metric": "axis-aligned bbox IoU", "threshold": IOU_MATCH,
                            "bbox_from": "8 corner min/max, pred 와 GT 에 같은 이미지 clip"},
        "nms_and_top1": {"nms": "ultralytics 기본(모델 내부)",
                         "top1_rule": "survivor 중 box confidence 최대",
                         "dump_conf": raw.get("recipe", {}).get("dump_conf")},
        "pnp": {"min_points": 8, "solver": "cv2.SOLVEPNP_SQPNP",
                "refinement": "cv2.solvePnPRefineLM",
                "points_used": "corner 0..7 (centroid 제외)"},
        "K_source": "eval_manifest item['K'] — per-frame GT intrinsics",
        "dimensions_source": {
            "current_eval": "eval_manifest item['dimensions_m'] = per-frame **exact label**",
            "object_points": manifest["object_points_source"],
            "★distribution": "라벨이 2종뿐이고 서로 W<->D 스왑이다 "
                             "(1.1x0.11x1.3: 89 frames / 1.3x0.11x1.1: 72 frames). "
                             "배포에서는 어느 쪽인지 알 수 없다.",
            "nominal_used_here": {"value": list(NOMINAL),
                                  "source": "annotate_pnp.PALLET_DIMS (사전 고정 상수)"}},
        "rotation_metric": {
            "raw": "re_metrics.pose_error — permutation 없음. main metric.",
            "symmetry_aware": "KEYPOINT_PERMUTATION_AUDIT 의 permutation 중 최소. "
                              "진단 전용이며 GT 를 보고 고르므로 배포 수치가 아니다."},
        "metric_denominator": {
            "conditional": "available 프레임(=survivor 있음) 기준. 기존 "
                           "YOLO_CONF_SWEEP.json 의 success_5cm5 가 이것이다.",
            "unconditional": "positive 161 전체 기준. 이 감사의 1차 지표."},
        "checkpoint_selection": {
            "rule": "last.pt (val 500 이 같은 synthetic pool 이라 선택 금지)",
            "source": "datasets/paper_generic_v1_manifest.json val_role",
            "★note": "같은 디렉터리에 best.pt 가 존재하지만 계약상 쓰지 않는다."},
    }


# --------------------------------------------------------------- phase C/D
def build_candidates(raw, items):
    """후보마다 IoU 와 pose 를 한 번만 푼다 (exact label dims)."""
    frames = []
    for entry in raw["positive"]:
        meta = items[entry["fid"]]
        width, height = meta["width"], meta["height"]
        K = np.asarray(meta["K"], float)
        model = np.asarray(meta["object_points"], float)
        R_gt = np.asarray(meta["R_gt"], float)
        t_gt = np.asarray(meta["t_gt"], float)
        gt8 = np.asarray(meta["gt_corners_2d"], float)[:8]
        gt_box = bbox_of(gt8, width, height)

        candidates = []
        for box in entry["boxes"]:
            pixels = [max(0.0, box["xyxy"][0]), max(0.0, box["xyxy"][1]),
                      min(float(width), box["xyxy"][2]),
                      min(float(height), box["xyxy"][3])]
            pose = solve(box["kps"], model, K)
            if pose is None:
                metrics = {"R": np.nan, "t": np.nan, "corner": np.nan, "s5": 0}
            else:
                R_err, t_err = RM.pose_error(pose[0], pose[1], R_gt, t_gt)
                metrics = {"R": R_err, "t": t_err,
                           "corner": float(np.median(np.linalg.norm(
                               np.asarray(box["kps"], float)[:8] - gt8, axis=1))),
                           "s5": int(RM.success_5cm5deg(pose[0], pose[1], R_gt, t_gt))}
            candidates.append({"conf": float(box["conf"]),
                               "iou": iou(pixels, gt_box), **metrics})
        candidates.sort(key=lambda c: -c["conf"])
        frames.append({"fid": entry["fid"], "set": entry["set"],
                       "population": entry["population"], "cands": candidates,
                       "dims": (meta["dimensions_m"]["width"],
                                meta["dimensions_m"]["depth"],
                                meta["dimensions_m"]["height"])})
    negatives = [{"frame": e["frame"],
                  "confs": sorted((b["conf"] for b in e["boxes"]), reverse=True)}
                 for e in raw["negative"]]
    return frames, negatives


def sweep_at(frames, negatives, tau):
    total = len(frames)
    present = top1_correct = any_correct = s5_uncond = 0
    corner, R, t = [], [], []
    for frame in frames:
        survivors = [c for c in frame["cands"] if c["conf"] >= tau]
        if not survivors:
            continue
        present += 1
        top = survivors[0]                     # 이미 conf 내림차순
        if top["iou"] >= IOU_MATCH:
            top1_correct += 1
        if any(c["iou"] >= IOU_MATCH for c in survivors):
            any_correct += 1
        s5_uncond += top["s5"]
        if np.isfinite(top["R"]):
            R.append(top["R"]); t.append(top["t"]); corner.append(top["corner"])

    neg_top1 = sum(1 for n in negatives if n["confs"] and n["confs"][0] >= tau)
    neg_boxes = sum(sum(1 for c in n["confs"] if c >= tau) for n in negatives)
    median = lambda v: float(np.median(v)) if v else None    # noqa: E731
    return {
        "conf": tau,
        "presence_recall": present / total,
        "top1_correct_recall": top1_correct / total,
        "oracle_any_candidate_recall": any_correct / total,
        "negative_frame_fpr_top1": neg_top1 / len(negatives),
        "fp_detections_per_image": neg_boxes / len(negatives),
        "uncond_success_5cm5": s5_uncond / total,
        "cond_success_5cm5": (s5_uncond / present) if present else None,
        "corner_median": median(corner), "R_median": median(R),
        "t_median": median(t),
    }


def phase_c(frames, negatives):
    scores = sorted({round(c["conf"], 4) for f in frames for c in f["cands"]}
                    | {round(c, 4) for n in negatives for c in n["confs"]})
    grid = sorted(set([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
                      + scores[::max(1, len(scores) // 200)]))
    curve = [sweep_at(frames, negatives, tau) for tau in grid]

    rmax = max(p["top1_correct_recall"] for p in curve)
    eligible = [p for p in curve if p["top1_correct_recall"] >= rmax - RMAX_SLACK]
    tau_star = min(eligible, key=lambda p: p["fp_detections_per_image"])

    by_session = defaultdict(lambda: {"n": 0, "top1": 0, "any": 0, "s5": 0})
    for frame in frames:
        survivors = [c for c in frame["cands"] if c["conf"] >= tau_star["conf"]]
        row = by_session[frame["set"]]
        row["n"] += 1
        if survivors:
            top = survivors[0]
            row["top1"] += int(top["iou"] >= IOU_MATCH)
            row["any"] += int(any(c["iou"] >= IOU_MATCH for c in survivors))
            row["s5"] += top["s5"]

    ranks = Counter()
    for frame in frames:
        survivors = [c for c in frame["cands"] if c["conf"] >= tau_star["conf"]]
        hit = [i for i, c in enumerate(survivors) if c["iou"] >= IOU_MATCH]
        ranks[(hit[0] + 1) if hit else 0] += 1

    return {"grid_size": len(grid), "curve": curve,
            "Rmax_top1": rmax, "rmax_slack": RMAX_SLACK,
            "tau_star": tau_star,
            "tau_star_rule": "top1_correct_recall >= Rmax_top1 - 0.005 중 "
                             "fp_detections_per_image 최소",
            "recall_098_reachable": bool(rmax >= 0.98),
            "session_table": {k: dict(v) for k, v in by_session.items()},
            "candidate_rank_histogram": {str(k): v for k, v in sorted(ranks.items())}}


def phase_d(frames, tau_star, baseline=0.40):
    groups = {"detected_at_0.40": [], "recovered_below_0.40": []}
    for frame in frames:
        base = [c for c in frame["cands"] if c["conf"] >= baseline]
        low = [c for c in frame["cands"] if c["conf"] >= tau_star]
        if base:
            groups["detected_at_0.40"].append(base[0])
        elif low:
            groups["recovered_below_0.40"].append(low[0])

    def describe(rows):
        if not rows:
            return {"n": 0}
        finite = [r for r in rows if np.isfinite(r["R"])]
        q = lambda key, p: (float(np.percentile([r[key] for r in finite], p))   # noqa: E731
                            if finite else None)
        return {"n": len(rows),
                "bbox_iou_ge_0.5_frac": float(np.mean([r["iou"] >= IOU_MATCH
                                                       for r in rows])),
                "bbox_iou_median": float(np.median([r["iou"] for r in rows])),
                "corner_median": q("corner", 50), "corner_p90": q("corner", 90),
                "R_median": q("R", 50), "R_p90": q("R", 90),
                "t_median": q("t", 50), "t_p90": q("t", 90),
                "s5_hits": int(sum(r["s5"] for r in rows))}

    total = len(frames)
    base = describe(groups["detected_at_0.40"])
    recovered = describe(groups["recovered_below_0.40"])
    gain = recovered.get("s5_hits", 0) / total
    if recovered["n"] == 0:
        verdict = "NO_RECOVERY_AT_TAU_STAR"
    elif gain <= 0.01:
        verdict = "THRESHOLD_ONLY_NOT_HELPFUL"
    else:
        verdict = "THRESHOLD_ONLY_PARTIALLY_HELPFUL"
    return {"tau_star": tau_star, "baseline": baseline,
            "detected_at_0.40": base, "recovered_below_0.40": recovered,
            "uncond_5cm5_contribution_of_recovered": gain,
            "positive_denominator": total, "VERDICT": verdict}


# --------------------------------------------------------------- phase E
def phase_e(raw, items, tau_star):
    """같은 keypoint 예측에 3D 치수만 갈아끼워 PnP 를 다시 푼다."""
    conditions = {
        "exact_label": None,
        "nominal_fixed": NOMINAL,
        "nominal_minus2pct": tuple(d * 0.98 for d in NOMINAL),
        "nominal_plus2pct": tuple(d * 1.02 for d in NOMINAL),
        "nominal_minus5pct": tuple(d * 0.95 for d in NOMINAL),
        "nominal_plus5pct": tuple(d * 1.05 for d in NOMINAL),
    }
    results = {name: {"R": [], "t": [], "s5": 0, "n": 0}
               for name in conditions}
    framewise = []

    for entry in raw["positive"]:
        meta = items[entry["fid"]]
        boxes = [b for b in entry["boxes"] if b["conf"] >= tau_star]
        if not boxes:
            continue
        top = max(boxes, key=lambda b: b["conf"])
        K = np.asarray(meta["K"], float)
        R_gt = np.asarray(meta["R_gt"], float)
        t_gt = np.asarray(meta["t_gt"], float)
        row = {"fid": entry["fid"], "population": entry["population"],
               "set": entry["set"]}
        for name, dims in conditions.items():
            model = (np.asarray(meta["object_points"], float) if dims is None
                     else APNP.make_pallet_keypoints_3d_diagram(
                         width=dims[0], depth=dims[1], height=dims[2])[:8])
            pose = solve(top["kps"], model, K)
            if pose is None:
                row[f"{name}_R"] = row[f"{name}_t"] = None
                continue
            R_err, t_err = RM.pose_error(pose[0], pose[1], R_gt, t_gt)
            hit = int(RM.success_5cm5deg(pose[0], pose[1], R_gt, t_gt))
            bucket = results[name]
            bucket["R"].append(R_err); bucket["t"].append(t_err)
            bucket["s5"] += hit; bucket["n"] += 1
            row[f"{name}_R"] = R_err; row[f"{name}_t"] = t_err
            row[f"{name}_s5"] = hit
        framewise.append(row)

    total = len(raw["positive"])
    summary = {}
    for name, bucket in results.items():
        if not bucket["n"]:
            summary[name] = {"n": 0}
            continue
        summary[name] = {
            "n": bucket["n"],
            "R_median": float(np.median(bucket["R"])),
            "t_median": float(np.median(bucket["t"])),
            "s5_hits": bucket["s5"],
            "uncond_5cm5": bucket["s5"] / total}

    exact = summary["exact_label"]["uncond_5cm5"]
    nominal = summary["nominal_fixed"]["uncond_5cm5"]
    drop = exact - nominal
    return {"conditions": {k: (list(v) if v else "per-frame label")
                           for k, v in conditions.items()},
            "positive_denominator": total, "summary": summary,
            "exact_minus_nominal_pp": drop * 100,
            "★label_is_two_swapped_variants":
                "라벨 치수는 (1.1,1.3) 과 (1.3,1.1) 두 종뿐이고 W<->D 가 바뀐 것이다. "
                "고정 nominal 을 쓰면 그중 한 무리는 18% 틀린 3D 모델로 PnP 를 푼다. "
                "이는 +-5% 섭동과 다른 축이다.",
            "VERDICT": ("KNOWN_SIZE_ASSUMPTION_REQUIRED" if drop > 0.05
                        else "SIZE_ASSUMPTION_TOLERABLE"),
            "framewise": framewise}


# --------------------------------------------------------------- phase F
def phase_f():
    audit = json.load(open(os.path.join(A2, "KEYPOINT_PERMUTATION_AUDIT.json")))
    attributes = {}
    path = os.path.join(V2DIR, "REAL_DEV_FAILURE_ATTRIBUTE.csv")
    if os.path.exists(path):
        for row in csv.DictReader(open(path)):
            attributes[row["fid"]] = row

    permutations = ("identity", "lr_swap", "near_far_swap", "near_far_lr",
                    "top_bottom", "rot180_face")
    rows, counts = [], Counter()
    for row in audit["rows"]:
        if row.get("identity_R") is None or row["identity_R"] <= GROSS_R:
            continue
        scores = {p: row.get(f"{p}_R") for p in permutations
                  if row.get(f"{p}_R") is not None}
        best = min(scores, key=scores.get)
        counts[best] += 1
        extra = attributes.get(row["fid"], {})
        rows.append({
            "fid": row["fid"], "set": row["set"],
            "population": row["population"],
            "identity_R": row["identity_R"],
            "identity_corner": row.get("identity_corner"),
            "near_far_swap_R": row.get("near_far_swap_R"),
            "near_far_swap_corner": row.get("near_far_swap_corner"),
            "best_permutation": best, "best_R": scores[best],
            "improved_by_near_far": bool(
                row.get("near_far_swap_R") is not None
                and row["near_far_swap_R"] < row["identity_R"]),
            "elevation_deg": extra.get("elev"), "distance_m": extra.get("distance_m"),
            "luma": extra.get("luma"), "truncated": extra.get("truncated"),
            "obj_diag_frac": extra.get("obj_diag_frac"),
            "failure_type": extra.get("failure_type"),
        })

    near_far_wins = sum(1 for r in rows if r["best_permutation"] == "near_far_swap")
    identity_wins = sum(1 for r in rows if r["best_permutation"] == "identity")
    # near_far_swap 이 이겼을 때 실제로 180 도 부근으로 뒤집혔는지
    flipped = [r for r in rows if r["best_permutation"] == "near_far_swap"
               and r["identity_R"] is not None and r["identity_R"] > 90.0]
    return {
        "gross_threshold_R_deg": GROSS_R,
        "n_gross": len(rows),
        "best_permutation_counts": dict(counts),
        "near_far_swap_win_frac": near_far_wins / len(rows) if rows else None,
        "identity_win_frac": identity_wins / len(rows) if rows else None,
        "near_far_wins_that_were_180ish": len(flipped),
        "★physical_symmetry": "팔레트는 180도 yaw 에 대해 외형이 거의 같다. "
                              "camera-facing 0123 convention 에서 near/far 는 "
                              "카메라 기준으로만 정의되므로, 저앙각에서 두 해가 "
                              "이미지상 거의 구분되지 않는다.",
        "★metric_rule": "raw(permutation 없음) 를 main 으로 쓴다. "
                        "symmetry-aware 는 GT 를 보고 고르므로 배포 불가.",
        "rows": rows}


# --------------------------------------------------------------- phase G
def phase_g():
    if not os.path.isdir(V2DIR):
        return {"status": "V2_DIR_MISSING"}
    images = [f for _, _, fs in os.walk(V2DIR) for f in fs
              if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files = Counter(os.path.splitext(f)[1].lower()
                    for _, _, fs in os.walk(V2DIR) for f in fs)
    manifest_like = [f for _, _, fs in os.walk(V2DIR) for f in fs
                     if "manifest" in f.lower()]
    return {"status": "V2_NOT_RENDERED" if not images else "V2_HAS_SAMPLES",
            "rendered_images": len(images),
            "manifest_files": manifest_like,
            "file_extension_counts": dict(files),
            "note": "사양·계획 문서만 존재하면 샘플 수/토폴로지 클러스터 감사는 "
                    "성립하지 않는다. 생성하지 않는다."}


# --------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    raw, manifest, items = load()
    log(f"입력: positive {len(raw['positive'])} / negative {len(raw['negative'])}")

    log("PHASE B — evaluation contract")
    contract_data = contract(raw, manifest)
    json.dump(contract_data, open(os.path.join(OUT, "EVAL_CONTRACT_AUDIT.json"), "w"),
              indent=1, ensure_ascii=False)

    log("PHASE C — confidence sweep (PnP 재계산)")
    frames, negatives = build_candidates(raw, items)
    curves = phase_c(frames, negatives)
    tau_star = curves["tau_star"]["conf"]
    log(f"  Rmax_top1 = {curves['Rmax_top1']:.4f}  "
        f"recall>=0.98 도달가능 = {curves['recall_098_reachable']}")
    log(f"  tau* = {tau_star}  FP/img {curves['tau_star']['fp_detections_per_image']:.3f}"
        f"  uncond 5cm5 {curves['tau_star']['uncond_success_5cm5']:.4f}")
    json.dump(curves, open(os.path.join(OUT, "CONF_SWEEP_CURVES.json"), "w"),
              indent=1, ensure_ascii=False)

    with open(os.path.join(OUT, "CONF_SWEEP_FRAMEWISE.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fid", "set", "population", "n_candidates",
                         "top1_conf", "top1_iou", "top1_R", "top1_t",
                         "top1_corner", "top1_s5", "best_iou_conf",
                         "best_iou_rank", "detected_at_0.40",
                         "detected_at_tau_star", "recovered"])
        for frame in frames:
            survivors = [c for c in frame["cands"] if c["conf"] >= tau_star]
            top = survivors[0] if survivors else None
            hits = [(i, c) for i, c in enumerate(frame["cands"])
                    if c["iou"] >= IOU_MATCH]
            at_040 = bool(frame["cands"] and frame["cands"][0]["conf"] >= 0.40)
            writer.writerow([
                frame["fid"], frame["set"], frame["population"],
                len(frame["cands"]),
                top["conf"] if top else "", top["iou"] if top else "",
                top["R"] if top else "", top["t"] if top else "",
                top["corner"] if top else "", top["s5"] if top else "",
                hits[0][1]["conf"] if hits else "",
                hits[0][0] + 1 if hits else "",
                int(at_040), int(bool(survivors)),
                int(bool(survivors) and not at_040)])

    log("PHASE D — recovered pose quality")
    recovered = phase_d(frames, tau_star)
    log(f"  {recovered['VERDICT']}  recovered n={recovered['recovered_below_0.40']['n']}"
        f"  5cm5 기여 +{recovered['uncond_5cm5_contribution_of_recovered']*100:.2f}pp")

    log("PHASE E — dimensions sensitivity (PnP 재계산)")
    dims = phase_e(raw, items, tau_star)
    log(f"  {dims['VERDICT']}  exact {dims['summary']['exact_label']['uncond_5cm5']:.4f}"
        f" -> nominal {dims['summary']['nominal_fixed']['uncond_5cm5']:.4f}"
        f"  ({dims['exact_minus_nominal_pp']:+.2f}pp)")
    framewise = dims.pop("framewise")
    json.dump(dims, open(os.path.join(OUT, "DIMS_SENSITIVITY.json"), "w"),
              indent=1, ensure_ascii=False)
    if framewise:
        with open(os.path.join(OUT, "DIMS_FRAMEWISE.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted(framewise[0]))
            writer.writeheader(); writer.writerows(framewise)

    log("PHASE F — near/far and symmetry")
    symmetry = phase_f()
    log(f"  gross n={symmetry['n_gross']}  best perm {symmetry['best_permutation_counts']}")
    rows = symmetry.pop("rows")
    json.dump(symmetry, open(os.path.join(OUT, "SYMMETRY_METRIC_COMPARE.json"), "w"),
              indent=1, ensure_ascii=False)
    if rows:
        with open(os.path.join(OUT, "NEAR_FAR_AUDIT.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    log("PHASE G — V2 readiness")
    v2 = phase_g()
    log(f"  {v2['status']}  rendered images {v2.get('rendered_images')}")
    json.dump(v2, open(os.path.join(OUT, "V2_READINESS.json"), "w"),
              indent=1, ensure_ascii=False)

    summary = {"stamp": STAMP, "phase_C": {k: curves[k] for k in
                                           ("Rmax_top1", "recall_098_reachable",
                                            "tau_star", "candidate_rank_histogram",
                                            "session_table")},
               "phase_D": recovered, "phase_E_verdict": dims["VERDICT"],
               "phase_E_summary": dims["summary"],
               "phase_F": symmetry, "phase_G": v2}
    json.dump(summary, open(os.path.join(OUT, "FAILURE_LOCUS_SUMMARY.json"), "w"),
              indent=1, ensure_ascii=False)
    written = sorted(os.listdir(OUT))
    log(f"완료: {len(written)}개 파일 -> {os.path.relpath(OUT, ROOT)}")
    log("  " + "  ".join(written))


if __name__ == "__main__":
    main()
