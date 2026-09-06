"""DiffPnP 스크린 드라이버 — 학습에서 끝내지 않는다.

    STEP 1 두 arm 학습 (lambda_dp 만 다르다)
    STEP 2 PAPER_EVAL 319 추론 (정본 recipe 그대로)
    STEP 3 pose 평가 (정본 selector · GT · metric)
    STEP 4 사전등록 게이트로 판정
    STEP 5 판정과 핵심 수치를 남긴다

완료 판정은 exit code 가 아니라 **산출물**로 한다.

    conda run -n pallet-yolo26 python -u \
        scripts/research/diffpnp_yolo_v1/run_screen.py --epochs 10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/paper/pose_metric_closure_v1"))

OUT = REPO / "data/pallet/results/diffpnp_yolo_v1"
RUNS = OUT / "screen_runs"
PMC = REPO / "data/pallet/results/paper_pose_metric_closure_v1"
LOCK = OUT / "DIFFPNP_SCREEN_METHOD_LOCK.json"

ARMS = {"C_LAMBDA0": 0.0, "T_LAMBDA1": 1.0}
CF_WIDTH, CF_DEPTH = "CF_WIDTH", "CF_DEPTH"


def cuboid(across, height, along):
    ha, hh, hb = across / 2.0, height / 2.0, along / 2.0
    return np.array([
        [-ha, -hh, -hb], [+ha, -hh, -hb], [+ha, +hh, -hb], [-ha, +hh, -hb],
        [-ha, -hh, +hb], [+ha, -hh, +hb], [+ha, +hh, +hb], [-ha, +hh, +hb],
    ], dtype=np.float64)


# ---------------------------------------------------------------- STEP 1 ----
def train_arm(arm: str, lam: float, epochs: int, batch: int) -> Path:
    weights = RUNS / arm / "weights" / "last.pt"
    if weights.exists():
        print(f"[{arm}] 이미 학습됨 — 건너뜀 ({weights})", flush=True)
        return weights
    cfg = OUT / f"diffpnp_config_{arm}.json"
    cfg.write_text(json.dumps({
        "enabled": lam != 0.0, "lambda_dp": lam, "gn_steps": 5,
        "huber_delta_norm": 0.10, "damping": 1e-3, "delta_clip": 0.5,
        "min_visible": 6, "affine_residual_max_px": 0.5, "index_dir": str(OUT),
    }, indent=1))
    env = dict(os.environ, DIFFPNP_CONFIG=str(cfg))
    env.pop("DIFFPNP_PROBE", None)
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from pallet_yolo_loss.trainer import DiffPnPTrainer\n"
        "t = DiffPnPTrainer(overrides=dict(\n"
        "    model=%r, data=%r, epochs=%d, batch=%d, imgsz=640, device='0',\n"
        "    workers=2, seed=42, deterministic=True, val=False, plots=False,\n"
        "    save=True, save_period=-1, patience=0, optimizer='SGD', cos_lr=True,\n"
        "    project=%r, name=%r, exist_ok=True, verbose=False))\n"
        "t.train()\n"
    ) % (str(REPO),
         str(REPO / "challenge/weights/pretrained_yolo/yolo26n-pose.pt"),
         str(REPO / "challenge/yolo_pose_one_model/datasets/"
                    "g38_legacy_v1v2_p0_tex20k/data.yaml"),
         epochs, batch, str(RUNS), arm)
    print(f"[{arm}] 학습 시작 lambda_dp={lam}", flush=True)
    started = time.time()
    proc = subprocess.run([sys.executable, "-u", "-c", code], env=env,
                          cwd=str(REPO))
    print(f"[{arm}] 학습 종료 {time.time() - started:.0f}s rc={proc.returncode}",
          flush=True)
    if not weights.exists():
        raise SystemExit(f"[{arm}] 산출물 없음 — 학습이 실제로 끝나지 않았다")
    return weights


# ---------------------------------------------------------------- STEP 2 ----
def infer_arm(arm: str, weights: Path) -> Path:
    out = OUT / "predictions" / f"{arm}.json"
    if out.exists():
        print(f"[{arm}] 추론 결과 있음 — 건너뜀", flush=True)
        return out
    import cv2
    from ultralytics import YOLO

    spec = json.loads((PMC / "INFERENCE_REPLAY_LOCK.json").read_text())
    spec = spec.get("recipe", spec)
    frames = json.loads((PMC / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]
    pad = int(spec["pad_px"])
    imgsz = int(spec["input_size"])
    conf = float(spec["confidence_floor"])

    model = YOLO(str(weights), task="pose")
    predictions, no_det = {}, 0
    for frame in frames:
        image = cv2.imread(str(REPO / frame["image"]))
        if image is None:
            predictions[frame["frame_id"]] = {"status": "IMAGE_MISSING"}
            continue
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        r = model.predict(padded, conf=conf, imgsz=imgsz, augment=False,
                          half=False, device="0", verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            predictions[frame["frame_id"]] = {"status": "NO_DETECTION"}
            no_det += 1
            continue
        scores = r.boxes.conf.detach().cpu().numpy()
        best = int(np.argmax(scores))
        box = r.boxes.xyxy.detach().cpu().numpy()[best] - pad
        kp = kc = None
        if r.keypoints is not None:
            kp = r.keypoints.xy.detach().cpu().numpy()[best] - pad
            if r.keypoints.conf is not None:
                kc = r.keypoints.conf.detach().cpu().numpy()[best]
        predictions[frame["frame_id"]] = {
            "status": "OK", "box_xyxy": box.tolist(),
            "box_conf": float(scores[best]),
            "keypoints_xy": kp.tolist() if kp is not None else None,
            "keypoints_conf": kc.tolist() if kc is not None else None,
            "detections": int(len(scores))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema_version": "diffpnp_screen_prediction_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arm": arm, "checkpoint": str(weights.relative_to(REPO)),
        "recipe": spec, "n_frames": len(frames), "no_detection": no_det,
        "frames": predictions}, indent=1))
    print(f"[{arm}] 추론 완료 no_detection={no_det}/{len(frames)}", flush=True)
    return out


# ---------------------------------------------------------------- STEP 3 ----
def evaluate_arm(pred_path: Path) -> dict:
    import cv2
    from pose_evaluation_paths import load_pose_object_contract, object_spec
    from symmetry_aware_pose_metrics import (cuboid_model_points, model_diameter_m,
                                             pose_auc, rotation_error_degrees,
                                             symmetry_aware_add_m,
                                             translation_components_m,
                                             yaw_error_degrees)
    from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses

    contract = load_pose_object_contract(str(PMC / "POSE_EVAL_OBJECT_CONTRACT.json"))
    gt = json.loads((PMC / "GEOMETRY_RESOLVED_POSE_GT.json").read_text())["frames"]
    frames = {f["frame_id"]: f for f in
              json.loads((PMC / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]}
    payload = json.loads(pred_path.read_text())["frames"]
    predictions = {k: v for k, v in payload.items()
                   if v.get("status") == "OK" and v.get("keypoints_xy")}

    rows = []
    for frame_id, truth in gt.items():
        pred = predictions.get(frame_id)
        if not pred:
            continue
        frame = frames[frame_id]
        raw = json.loads((REPO / frame["annotation"]).read_text())["camera_data"]["intrinsics"]
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
        chosen = None
        try:
            result = select_pnp_hypotheses(
                np.asarray(pred["keypoints_xy"], np.float64), camera,
                {"x": long_m, "y": height_m, "z": short_m}, None)
            for h in result.hypotheses:
                if h.name == result.selected_hypothesis and h.success:
                    d = h.camera_facing_dimensions.as_dict()
                    chosen = (CF_WIDTH if abs(float(d["width"]) - long_m) < 1e-6
                              else CF_DEPTH)
        except Exception:
            chosen = None
        if chosen is None:
            continue
        model = models[chosen]
        ok, rvec, tvec = cv2.solvePnP(model[usable], points[usable], camera, None,
                                      flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(model[usable], points[usable], camera,
                                          None, rvec, tvec)
        rotation, _ = cv2.Rodrigues(rvec)
        translation = tvec.reshape(-1)
        dims = truth["physical_dimensions_m"]
        extents = (dims["across"], dims["height"], dims["along"])
        mp = cuboid_model_points(extents)
        gt_R = np.asarray(truth["R_gt_representative"], np.float64)
        gt_t = np.asarray(truth["t_gt"], np.float64)
        rows.append({
            "axis_correct": chosen == truth["physical_long_axis"],
            "rotation_error_deg": rotation_error_degrees(rotation, gt_R),
            "yaw_error_deg": yaw_error_degrees(rotation, gt_R),
            "translation_error_cm": translation_components_m(translation, gt_t)["total_m"] * 100.0,
            "iou3d": oriented_iou_3d(rotation, translation, extents, gt_R, gt_t, extents),
            "add_sym_m": symmetry_aware_add_m(mp, rotation, translation, gt_R, gt_t),
            "diameter_m": model_diameter_m(mp)})
    if not rows:
        return {"n": 0}
    arr = lambda k: np.array([r[k] for r in rows], float)  # noqa: E731
    return {"n": len(rows),
            "axis_accuracy": float(np.mean([r["axis_correct"] for r in rows])),
            "rotation_median_deg": float(np.median(arr("rotation_error_deg"))),
            "yaw_median_deg": float(np.median(arr("yaw_error_deg"))),
            "translation_median_cm": float(np.median(arr("translation_error_cm"))),
            "iou3d_median": float(np.median(arr("iou3d"))),
            "add_sym_auc": pose_auc(arr("add_sym_m"),
                                    float(np.median(arr("diameter_m"))))}


# ---------------------------------------------------------------- STEP 4 ----
def verdict(control: dict, treatment: dict, lock: dict) -> dict:
    metrics = lock["primary_metrics"]
    tol = lock["verdict_rule"]["relative_tolerance"]
    detail, ok_all, improved = {}, True, 0
    for m, direction in metrics.items():
        b, c = float(control[m]), float(treatment[m])
        rel = (c - b) / abs(b) if b else 0.0
        better = (c < b) if direction == "lower_is_better" else (c > b)
        worse = (rel > tol) if direction == "lower_is_better" else (-rel > tol)
        detail[m] = {"control": b, "treatment": c, "relative_change": rel,
                     "better": bool(better), "worse_beyond_tolerance": bool(worse)}
        ok_all = ok_all and not worse
        improved += int(better)
    biggest = max(abs(v["relative_change"]) for v in detail.values())
    accept = ok_all and improved >= 2
    return {"verdict": "ACCEPT_DIFFPNP" if accept else "REJECT",
            "no_metric_worse_beyond_tolerance": ok_all,
            "improved_metric_count": improved,
            "largest_absolute_relative_change": biggest,
            "indistinguishable_from_control": bool(biggest < 1e-6),
            "effective_reading": ("NO_CHANGE" if biggest < 1e-6 else
                                  ("IMPROVED" if accept else "NOT_IMPROVED")),
            "by_metric": detail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK.read_text())
    started = time.time()
    summaries, preds = {}, {}
    for arm, lam in ARMS.items():
        w = train_arm(arm, lam, args.epochs, args.batch)
        preds[arm] = infer_arm(arm, w)
    for arm in ARMS:
        summaries[arm] = evaluate_arm(preds[arm])
        print(f"[{arm}] {json.dumps(summaries[arm])}", flush=True)

    v = verdict(summaries["C_LAMBDA0"], summaries["T_LAMBDA1"], lock)
    report = {"schema_version": "diffpnp_screen_result_v1",
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_sec": round(time.time() - started, 1),
              "epochs": args.epochs, "batch": args.batch,
              "arms": summaries, "verdict": v,
              "leakage_note": lock["leakage_note"]}
    (OUT / "DIFFPNP_SCREEN_RESULT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    lines = ["# DiffPnP 학습 스크린 — lambda_dp 만 다르다", "",
             f"{args.epochs} epochs · batch {args.batch} · seed 42 · "
             f"{report['elapsed_sec'] / 60:.0f}분", "",
             "| 지표 | 대조 lambda=0 | 처치 lambda=1 | 상대변화 |", "|---|---:|---:|---:|"]
    for m, d in v["by_metric"].items():
        lines.append(f"| {m} | {d['control']:.4f} | {d['treatment']:.4f} | "
                     f"{d['relative_change']:+.2%} |")
    lines += ["", f"판정 **{v['verdict']}** · 실질 {v['effective_reading']} · "
                  f"개선 {v['improved_metric_count']}/4", "",
              "| 보조 | 대조 | 처치 |", "|---|---:|---:|"]
    for m in ("n", "axis_accuracy", "yaw_median_deg"):
        lines.append(f"| {m} | {summaries['C_LAMBDA0'][m]} | {summaries['T_LAMBDA1'][m]} |")
    lines += ["", f"> {lock['leakage_note']}", ""]
    (OUT / "DIFFPNP_SCREEN_REPORT.md").write_text("\n".join(lines))

    print("\n" + "=" * 60)
    print(f"판정: {v['verdict']}  (실질 {v['effective_reading']}, "
          f"개선 {v['improved_metric_count']}/4)")
    for m, d in v["by_metric"].items():
        print(f"  {m:26s} {d['control']:.4f} -> {d['treatment']:.4f}  "
              f"{d['relative_change']:+.2%}")
    print(f"\n산출 {OUT / 'DIFFPNP_SCREEN_RESULT.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
