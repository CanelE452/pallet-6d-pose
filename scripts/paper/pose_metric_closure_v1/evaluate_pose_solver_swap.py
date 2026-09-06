"""pose read-out solver 만 바꿔 다시 평가한다 — 학습 0, 새 추론 0.

    conda run -n pallet-pose python -u \
        scripts/paper/pose_metric_closure_v1/evaluate_pose_solver_swap.py

`evaluate_pose_by_session.py` 와 **같은 GT·같은 선택기·같은 metric·같은 모집단** 을 쓴다.
유일한 변수는 pose 를 읽는 solver 다.

게이트는 `solver_swap_v1/SOLVER_SWAP_METHOD_LOCK.json` 에 실행 전에 고정돼 있다.
GATE 0(S0 이 정본 수치를 재현) 이 깨지면 하네스가 무효이므로 D arm 수치를 읽지 않고
비정상 종료한다.

출력  solver_swap_v1/SOLVER_SWAP_RESULTS.json
      solver_swap_v1/SOLVER_SWAP_REPORT.md
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))

from diffpnp_eval_solver import solve_pnp_gn  # noqa: E402

OUT_DIR = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
SWAP_DIR = OUT_DIR / "solver_swap_v1"
GT_PATH = OUT_DIR / "GEOMETRY_RESOLVED_POSE_GT.json"
MANIFEST = OUT_DIR / "AXIS_REVIEW_MANIFEST.json"
PREDICTIONS = OUT_DIR / "predictions"
CONTRACT = OUT_DIR / "POSE_EVAL_OBJECT_CONTRACT.json"
LOCK = SWAP_DIR / "SOLVER_SWAP_METHOD_LOCK.json"

ARMS = ["R0", "R0_CONT", "R1_NAIVE", "R2_CONF", "R3_CONF_REPROJ",
        "R4_CONF_REMOVE", "R5_PROPOSED"]
CF_WIDTH, CF_DEPTH = "CF_WIDTH", "CF_DEPTH"
TOLERANCE = 1e-9

SOLVER_ARMS = ["S0_SQPNP_LM", "D1_GN_LS", "D2_GN_HUBER",
               "D3_SQPNP_GN", "D4_GN_HUBER_CONF"]
PRIMARY_METRICS = {"rotation_median_deg": "lower", "translation_median_cm": "lower",
                   "iou3d_median": "higher", "add_sym_auc": "higher"}


def cuboid(across, height, along):
    ha, hh, hb = across / 2.0, height / 2.0, along / 2.0
    return np.array([
        [-ha, -hh, -hb], [+ha, -hh, -hb], [+ha, +hh, -hb], [-ha, +hh, -hb],
        [-ha, -hh, +hb], [+ha, -hh, +hb], [+ha, +hh, +hb], [-ha, +hh, +hb],
    ], dtype=np.float64)


def solve_canonical(model, points, camera, usable):
    """정본 — SQPnP + RefineLM. evaluate_pose_by_session.solve() 와 동일해야 한다."""
    ok, rvec, tvec = cv2.solvePnP(model[usable], points[usable], camera, None,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(model[usable], points[usable], camera, None,
                                      rvec, tvec)
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation, tvec.reshape(-1)


def solve_variant(name, model, points, camera, usable, conf):
    """D arm — 미분가능 GN. 대응점·모집단은 정본과 완전히 같고 solver 만 다르다."""
    X, uv = model[usable], points[usable]
    if name == "D1_GN_LS":
        R, t, info = solve_pnp_gn(X, uv, camera, init="epnp")
    elif name == "D2_GN_HUBER":
        R, t, info = solve_pnp_gn(X, uv, camera, init="epnp", huber_delta=12.0)
    elif name == "D3_SQPNP_GN":
        R, t, info = solve_pnp_gn(X, uv, camera, init="sqpnp")
    elif name == "D4_GN_HUBER_CONF":
        w = None if conf is None else np.asarray(conf, np.float64)[:8][usable]
        R, t, info = solve_pnp_gn(X, uv, camera, init="epnp", huber_delta=12.0,
                                  weights=w)
    else:
        raise ValueError(name)
    if R is None:
        return None, info
    return (R, t), info


def summarize(rows):
    if not rows:
        return {"n": 0}
    arr = lambda k: np.array([r[k] for r in rows], float)  # noqa: E731
    from symmetry_aware_pose_metrics import pose_auc
    return {
        "n": len(rows),
        "axis_accuracy": float(np.mean([r["axis_correct"] for r in rows])),
        "rotation_median_deg": float(np.median(arr("rotation_error_deg"))),
        "yaw_median_deg": float(np.median(arr("yaw_error_deg"))),
        "translation_median_cm": float(np.median(arr("translation_error_cm"))),
        "iou3d_median": float(np.median(arr("iou3d"))),
        "add_sym_auc": pose_auc(arr("add_sym_m"), float(np.median(arr("diameter_m")))),
    }


def main() -> int:
    from pose_evaluation_paths import load_pose_object_contract, object_spec
    from symmetry_aware_pose_metrics import (cuboid_model_points, model_diameter_m,
                                             rotation_error_degrees,
                                             symmetry_aware_add_m,
                                             translation_components_m,
                                             yaw_error_degrees)
    from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses

    lock = json.loads(LOCK.read_text())
    contract = load_pose_object_contract(str(CONTRACT))
    gt = json.loads(GT_PATH.read_text())["frames"]
    frames = {f["frame_id"]: f for f in json.loads(MANIFEST.read_text())["frames_list"]}

    annotation_cache: dict[str, dict] = {}

    def annotation_of(frame):
        path = frame["annotation"]
        if path not in annotation_cache:
            annotation_cache[path] = json.loads((REPO_ROOT / path).read_text())
        return annotation_cache[path]

    started = time.time()
    per_arm: dict[str, dict[str, list[dict]]] = {}
    health = {s: {"fallback": 0, "solved": 0} for s in SOLVER_ARMS}

    for arm in ARMS:
        payload = json.loads((PREDICTIONS / f"{arm}.json").read_text())["frames"]
        predictions = {k: v for k, v in payload.items()
                       if v.get("status") == "OK" and v.get("keypoints_xy")}
        rows_by_solver: dict[str, list[dict]] = {s: [] for s in SOLVER_ARMS}

        for frame_id, truth in gt.items():
            pred = predictions.get(frame_id)
            if not pred:
                continue
            frame = frames[frame_id]
            raw = annotation_of(frame)["camera_data"]["intrinsics"]
            camera = np.array([[raw["fx"], 0.0, raw["cx"]],
                               [0.0, raw["fy"], raw["cy"]], [0.0, 0.0, 1.0]], np.float64)
            spec = object_spec(contract, frame["object_type"])
            long_m, short_m, height_m = spec["long_m"], spec["short_m"], spec["height_m"]
            models = {CF_WIDTH: cuboid(long_m, height_m, short_m),
                      CF_DEPTH: cuboid(short_m, height_m, long_m)}

            points = np.asarray(pred["keypoints_xy"], np.float64)[:8]
            usable = np.isfinite(points).all(axis=1)
            if usable.sum() < 6:
                continue

            # 축 가설 선택기 — 모든 solver arm 에서 동일하게 고정한다
            chosen = None
            try:
                result = select_pnp_hypotheses(
                    np.asarray(pred["keypoints_xy"], np.float64), camera,
                    {"x": long_m, "y": height_m, "z": short_m}, None)
                for hypothesis in result.hypotheses:
                    if hypothesis.name == result.selected_hypothesis and hypothesis.success:
                        dims = hypothesis.camera_facing_dimensions.as_dict()
                        chosen = (CF_WIDTH if abs(float(dims["width"]) - long_m) < 1e-6
                                  else CF_DEPTH)
            except Exception:
                chosen = None
            if chosen is None:
                continue

            canonical = {k: solve_canonical(m, points, camera, usable)
                         for k, m in models.items()}
            if any(v is None for v in canonical.values()):
                continue

            gt_R = np.asarray(truth["R_gt_representative"], np.float64)
            gt_t = np.asarray(truth["t_gt"], np.float64)
            dims = truth["physical_dimensions_m"]
            extents = (dims["across"], dims["height"], dims["along"])
            model_points = cuboid_model_points(extents)
            conf = pred.get("keypoints_conf")

            for solver in SOLVER_ARMS:
                if solver == "S0_SQPNP_LM":
                    fit = canonical[chosen]
                else:
                    fit, info = solve_variant(solver, models[chosen], points,
                                              camera, usable, conf)
                    health[solver]["solved"] += 1
                    if info.get("fallback"):
                        health[solver]["fallback"] += 1
                if fit is None:
                    continue
                rotation, translation = fit
                parts = translation_components_m(translation, gt_t)
                rows_by_solver[solver].append({
                    "frame_id": frame_id,
                    "session_id": frame["session_id"],
                    "axis_correct": chosen == truth["physical_long_axis"],
                    "rotation_error_deg": rotation_error_degrees(rotation, gt_R),
                    "yaw_error_deg": yaw_error_degrees(rotation, gt_R),
                    "translation_error_cm": parts["total_m"] * 100.0,
                    "iou3d": oriented_iou_3d(rotation, translation, extents,
                                             gt_R, gt_t, extents),
                    "add_sym_m": symmetry_aware_add_m(model_points, rotation,
                                                      translation, gt_R, gt_t),
                    "diameter_m": model_diameter_m(model_points),
                })
        per_arm[arm] = rows_by_solver
        print(f"  {arm:16}" + "  ".join(
            f"{s}={len(rows_by_solver[s])}" for s in SOLVER_ARMS), flush=True)

    # ── GATE 0 — S0 이 정본을 재현하지 못하면 전부 무효
    mismatches = []
    for arm in ARMS:
        recomputed = summarize(per_arm[arm]["S0_SQPNP_LM"])
        existing = json.loads(
            (OUT_DIR / f"POSE_EVALUATION_{arm}.json").read_text())["paths"]["MAIN"]["ALL"]
        for key in lock["gate_0_harness_validity"]["keys"]:
            if abs(float(recomputed[key]) - float(existing[key])) > TOLERANCE:
                mismatches.append(f"{arm}.{key}: {recomputed[key]} != {existing[key]}")
    gate0 = not mismatches
    print(f"\nGATE 0 harness validity: {'PASS' if gate0 else 'FAIL'}", flush=True)
    if not gate0:
        for line in mismatches[:10]:
            print(f"  {line}")
        print("\nVOID — S0 이 정본을 재현하지 못했다. D arm 수치를 읽지 않는다.")
        return 1

    # ── 짝지은 모집단 = 모든 solver arm 이 pose 를 반환한 프레임 교집합
    paired: dict[str, dict] = {}
    for arm in ARMS:
        common = set.intersection(*[{r["frame_id"] for r in per_arm[arm][s]}
                                    for s in SOLVER_ARMS])
        paired[arm] = {s: summarize([r for r in per_arm[arm][s]
                                     if r["frame_id"] in common])
                       for s in SOLVER_ARMS}
        paired[arm]["_n_common"] = len(common)

    # ── 사전등록 판정
    rel_tol = lock["verdict_rule"]["relative_tolerance"]
    primary = lock["primary_arms"]
    verdicts = {}
    for solver in SOLVER_ARMS:
        if solver == "S0_SQPNP_LM":
            continue
        detail, ok_all, improved_all = {}, True, 0
        for arm in primary:
            base, cand = paired[arm]["S0_SQPNP_LM"], paired[arm][solver]
            per_metric = {}
            for metric, direction in PRIMARY_METRICS.items():
                b, c = float(base[metric]), float(cand[metric])
                rel = (c - b) / abs(b) if b else 0.0
                better = (c < b) if direction == "lower" else (c > b)
                worse_beyond_tol = (rel > rel_tol) if direction == "lower" \
                    else (-rel > rel_tol)
                per_metric[metric] = {"baseline": b, "variant": c,
                                      "relative_change": rel,
                                      "better": bool(better),
                                      "worse_beyond_tolerance": bool(worse_beyond_tol)}
                if worse_beyond_tol:
                    ok_all = False
                if better:
                    improved_all += 1
            detail[arm] = per_metric
        accept = ok_all and improved_all >= 2 * len(primary)
        # 사전등록 규칙은 'better' 를 부등호로만 정의해서 1e-9 차이도 개선으로 센다.
        # 규칙은 그대로 두고, 그 판정이 실체가 있는지를 별도 필드로 드러낸다.
        biggest = max(abs(x["relative_change"])
                      for m in detail.values() for x in m.values())
        indistinguishable = biggest < 1e-6
        verdicts[solver] = {"verdict": "ACCEPT_SOLVER_SWAP" if accept else "REJECT",
                            "no_metric_worse_beyond_tolerance": ok_all,
                            "improved_metric_count": improved_all,
                            "largest_absolute_relative_change": biggest,
                            "indistinguishable_from_baseline": bool(indistinguishable),
                            "effective_reading": ("NO_CHANGE — 사전등록 부등호가 "
                                                  "1e-9 차이를 개선으로 셌다"
                                                  if indistinguishable else
                                                  "REAL_DIFFERENCE"),
                            "by_arm": detail}

    # ── 짝별 개선/악화 (R0)
    paired_counts = {}
    r0 = per_arm["R0"]
    base_by_id = {r["frame_id"]: r for r in r0["S0_SQPNP_LM"]}
    for solver in SOLVER_ARMS:
        if solver == "S0_SQPNP_LM":
            continue
        counts = {}
        for metric in ("rotation_error_deg", "translation_error_cm"):
            imp = wor = 0
            for r in r0[solver]:
                b = base_by_id.get(r["frame_id"])
                if not b:
                    continue
                if r[metric] < b[metric]:
                    imp += 1
                elif r[metric] > b[metric]:
                    wor += 1
            counts[metric] = {"improved": imp, "worsened": wor}
        paired_counts[solver] = counts

    report = {
        "schema_version": "solver_swap_results_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - started, 1),
        "new_training": 0,
        "new_inference": 0,
        "method_lock_sha_note": "SOLVER_SWAP_METHOD_LOCK.json 은 실행 전에 고정됐다",
        "gate_0_harness_validity": "PASS",
        "solver_health": health,
        "paired": paired,
        "verdicts": verdicts,
        "paired_counts_R0": paired_counts,
        "leakage_note": lock["leakage_note"],
    }
    (SWAP_DIR / "SOLVER_SWAP_RESULTS.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    # ── 보고서
    lines = ["# SOLVER SWAP v1 — SQPnP+RefineLM vs 미분가능 Gauss-Newton PnP", "",
             f"생성 {report['generated_utc']} · 학습 0 step · 새 추론 0 회 · "
             f"{report['elapsed_sec']}s", "",
             "GATE 0 (S0 이 정본 POSE_EVALUATION_{ARM}.json 재현) = **PASS** — "
             "따라서 아래 D arm 수치는 solver 만의 차이다.", "",
             "축 가설 선택기는 모든 arm 에서 SQPnP 기반으로 고정했다. 바뀐 것은 "
             "pose read-out 뿐이다.", ""]
    for arm in ARMS:
        lines += [f"## {arm}  (짝지은 프레임 n={paired[arm]['_n_common']})", "",
                  "| solver | rot med° | yaw med° | trans med cm | IoU3D | ADD AUC |",
                  "|---|---:|---:|---:|---:|---:|"]
        for s in SOLVER_ARMS:
            v = paired[arm][s]
            lines.append(f"| {s} | {v['rotation_median_deg']:.3f} | "
                         f"{v['yaw_median_deg']:.3f} | {v['translation_median_cm']:.3f} | "
                         f"{v['iou3d_median']:.4f} | {v['add_sym_auc']:.4f} |")
        lines.append("")
    lines += ["## 사전등록 판정", "",
              "사전등록 규칙은 개선을 부등호로만 정의한다. 그래서 baseline 과 사실상 같은 "
              "solver 도 ACCEPT 가 될 수 있다. 마지막 두 칸이 그 구분이다.", "",
              "| solver | 판정 | 허용범위 밖 악화 없음 | 개선 지표 수 (R0+R5, 8칸) "
              "| 최대 상대변화 | 실질 |",
              "|---|---|---|---:|---:|---|"]
    for s, v in verdicts.items():
        lines.append(f"| {s} | **{v['verdict']}** | "
                     f"{v['no_metric_worse_beyond_tolerance']} | "
                     f"{v['improved_metric_count']} | "
                     f"{v['largest_absolute_relative_change']:.2e} | "
                     f"{v['effective_reading'].split(' —')[0]} |")
    lines += ["", "## R0 짝별 개선/악화", "",
              "| solver | rotation 개선/악화 | translation 개선/악화 |", "|---|---|---|"]
    for s, c in paired_counts.items():
        lines.append(f"| {s} | {c['rotation_error_deg']['improved']}/"
                     f"{c['rotation_error_deg']['worsened']} | "
                     f"{c['translation_error_cm']['improved']}/"
                     f"{c['translation_error_cm']['worsened']} |")
    lines += ["", "## solver health", "",
              "| solver | 풀린 프레임 | init/guard fallback |", "|---|---:|---:|"]
    for s in SOLVER_ARMS[1:]:
        lines.append(f"| {s} | {health[s]['solved']} | {health[s]['fallback']} |")
    lines += ["", f"> {lock['leakage_note']}", ""]
    (SWAP_DIR / "SOLVER_SWAP_REPORT.md").write_text("\n".join(lines))

    print("\n판정:")
    for s, v in verdicts.items():
        print(f"  {s:20s} {v['verdict']:20s} "
              f"개선 {v['improved_metric_count']}/8  "
              f"악화없음={v['no_metric_worse_beyond_tolerance']}")
    print(f"\n산출  {SWAP_DIR / 'SOLVER_SWAP_RESULTS.json'}")
    print(f"      {SWAP_DIR / 'SOLVER_SWAP_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
