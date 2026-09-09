"""A2b screen 드라이버 — 학습에서 끝내지 않는다.

  CONTROL 5ep -> A2b 5ep -> 319 추론 x2 -> 정본 evaluator x2 -> paired 분석 -> 판정

판정 기준은 이 파일 안에 하드코딩돼 있고 METHOD_LOCK_A2B §10 과 같다.
결과를 보고 threshold 를 고칠 수 없다.  마지막 출력은 '완료' 가 아니라 **판정**이다.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "challenge/yolo_pose_one_model/pallet_translation_loss_v1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))
OUT = REPO / "data/pallet/results/pallet_translation_loss_v1"
RUNS = PKG / "runs"
CLOSURE = REPO / "data/pallet/results/paper_pose_metric_closure_v1"
PY_ENV = "/home/minjae/anaconda3/envs/pallet-yolo26/bin/python"

RUN_TAG = os.environ.get("A2B_RUN_TAG", "r1")
ARMS = {"A2B_CONTROL": 0.0, "A2B_LC": None}     # None 이면 calibration 값을 넣는다
EPOCHS = 5

# ---- 사전등록 gate (METHOD_LOCK_A2B §10).  결과를 보고 고치지 않는다. ----
GATE = {"depth_improve_min": 0.05, "translation_improve_min": 0.05,
        "rotation_degrade_max": 0.05, "corner_degrade_max": 0.05}


PROGRESS = None


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    if PROGRESS is not None:
        with open(PROGRESS, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def train_arm(arm: str, lam: float) -> Path:
    from a2b_preflight import screen_args, R0_CKPT, DATA_YAML
    import a2b_model  # noqa: F401  (trainer 등록)
    from a2b_model import A2BTrainer

    cfg = {"enabled": True, "lambda_geo": float(lam), "calibration": False,
           "train_stem_list": str(PKG / "TRAIN_STEMS.txt")}
    cfg_path = PKG / f"_a2b_config_{arm}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    os.environ["A2B_CONFIG"] = str(cfg_path)

    a = screen_args()
    ov = {k: v for k, v in vars(a).items() if not k.startswith("_")}
    ov.update(dict(model=str(R0_CKPT), data=str(DATA_YAML), epochs=EPOCHS,
                   project=str(RUNS), name=f"{arm}_{RUN_TAG}", exist_ok=True,
                   patience=0, save_period=-1, val=True, plots=False,
                   seed=42, deterministic=True, workers=2, batch=32, imgsz=640))
    log(f"=== TRAIN {arm}  lambda_geo={lam} ===")
    A2BTrainer(overrides=ov).train()
    last = RUNS / f"{arm}_{RUN_TAG}" / "weights/last.pt"
    if not last.is_file():
        raise RuntimeError(f"{arm}: last.pt not produced")
    log(f"{arm} done -> {last}")
    return last


def infer_arm(arm: str, weights: Path):
    """정본 recipe 로 319 장 추론.  lock 파일은 수정하지 않고 읽기만 한다."""
    import cv2
    from ultralytics import YOLO

    lock = json.loads((CLOSURE / "INFERENCE_REPLAY_LOCK.json").read_text())
    assert lock["status"] == "FROZEN"
    spec = lock["recipe"]
    frames = json.loads((CLOSURE / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]
    model = YOLO(str(weights), task="pose")
    pad, imgsz, conf = int(spec["pad_px"]), int(spec["input_size"]), float(spec["confidence_floor"])
    preds, nod = {}, 0
    for fr in frames:
        img = cv2.imread(str(REPO / fr["image"]))
        if img is None:
            preds[fr["frame_id"]] = {"status": "IMAGE_MISSING"}
            continue
        p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        r = model.predict(p, conf=conf, imgsz=imgsz, augment=False, half=False,
                          device=spec["device"], verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            preds[fr["frame_id"]] = {"status": "NO_DETECTION"}
            nod += 1
            continue
        sc = r.boxes.conf.detach().cpu().numpy()
        b = int(np.argmax(sc))
        kp = r.keypoints.xy.detach().cpu().numpy()[b] - pad if r.keypoints is not None else None
        kc = (r.keypoints.conf.detach().cpu().numpy()[b]
              if r.keypoints is not None and r.keypoints.conf is not None else None)
        preds[fr["frame_id"]] = {
            "status": "OK",
            "box_xyxy": (r.boxes.xyxy.detach().cpu().numpy()[b] - pad).tolist(),
            "box_conf": float(sc[b]),
            "keypoints_xy": kp.tolist() if kp is not None else None,
            "keypoints_conf": kc.tolist() if kc is not None else None,
            "detections": int(len(sc))}
    payload = {"schema_version": "frozen_arm_prediction_v1", "arm": arm,
               "checkpoint": str(weights.relative_to(REPO)),
               "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
               "recipe": spec,
               "recipe_lock_sha256": hashlib.sha256(
                   (CLOSURE / "INFERENCE_REPLAY_LOCK.json").read_bytes()).hexdigest(),
               "population_frame_order_sha256": lock["population"]["frame_order_sha256"],
               "n_frames": len(frames), "no_detection": nod,
               "new_training": 0, "checkpoint_reselection": 0, "frames": preds}
    (CLOSURE / "predictions" / f"{arm}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    shutil.copy(CLOSURE / "predictions" / f"{arm}.json", OUT / "predictions" / f"{arm}.json")
    log(f"{arm} inference: {len(frames)} frames, no_detection {nod}")


def evaluate_arm(arm: str) -> dict:
    cmd = [PY_ENV, str(REPO / "scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py"),
           "--pose-object-contract", str(CLOSURE / "POSE_EVAL_OBJECT_CONTRACT.json"),
           "--arm", arm]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"evaluation failed for {arm}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    res = CLOSURE / f"POSE_EVALUATION_{arm}.json"
    d = json.loads(res.read_text())
    shutil.copy(res, OUT / f"POSE_EVALUATION_{arm}.json")
    return d


def main() -> int:
    global PROGRESS
    OUT.mkdir(parents=True, exist_ok=True)
    PROGRESS = OUT / f"A2B_SCREEN_PROGRESS_{RUN_TAG}.log"
    log(f"driver start  tag={RUN_TAG}  pid={os.getpid()}")
    cal = json.loads((OUT / "A2B_LAMBDA_CALIBRATION.json").read_text())
    if cal["sanity_gate"] != "PASS":
        log("lambda calibration gate FAILED — stop")
        return 1
    pre = json.loads((OUT / "A2B_PREFLIGHT.json").read_text())
    if pre["geometry_batch_smoke"]["verdict"] != "PASS":
        log("geometry wiring gate FAILED — stop")
        return 1
    ARMS["A2B_LC"] = float(cal["lambda_lc"])
    log(f"lambda_lc = {ARMS['A2B_LC']:.8g}")

    weights = {}
    for arm, lam in ARMS.items():
        weights[arm] = train_arm(arm, lam)

    hashes = {}
    for arm in ARMS:
        p = RUNS / f"{arm}_{RUN_TAG}" / "A2B_BATCH_ORDER.json"
        hashes[arm] = json.loads(p.read_text()) if p.is_file() else None
    equal = (hashes["A2B_CONTROL"] and hashes["A2B_LC"]
             and hashes["A2B_CONTROL"]["first_100_batch_sha256"]
             == hashes["A2B_LC"]["first_100_batch_sha256"])
    log(f"FIRST_100_BATCH_HASH_EQUAL = {'YES' if equal else 'NO'}")

    evals = {}
    for arm in ARMS:
        infer_arm(arm, weights[arm])
        evals[arm] = evaluate_arm(arm)
        log(f"{arm} evaluated")

    r0 = json.loads((CLOSURE / "POSE_EVALUATION_R0.json").read_text())
    summary = {"schema_version": "a2b_screen_result_v1",
               "lambda_lc": ARMS["A2B_LC"], "epochs": EPOCHS,
               "first_100_batch_hash_equal": bool(equal),
               "batch_order": hashes,
               "gate": GATE,
               "paths": {a: evals[a]["paths"] for a in ARMS},
               "R0_reference": r0["paths"]}
    (OUT / "A2B_SCREEN_RESULT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    c = evals["A2B_CONTROL"]["paths"]["MAIN"]["ALL"]
    a = evals["A2B_LC"]["paths"]["MAIN"]["ALL"]
    def imp(k):
        return (c[k] - a[k]) / c[k] if c[k] else 0.0
    checks = {
        "depth_improve": imp("depth_median_cm"),
        "translation_improve": imp("translation_median_cm"),
        "rotation_degrade": -imp("rotation_median_deg"),
    }
    ok = (checks["depth_improve"] >= GATE["depth_improve_min"]
          and checks["translation_improve"] >= GATE["translation_improve_min"]
          and checks["rotation_degrade"] < GATE["rotation_degrade_max"])
    verdict = ("POSE_SENSITIVITY_MECHANISM_SUPPORTED" if ok
               else "POSE_SENSITIVITY_LOSS_NO_SCREEN_SIGNAL")
    if not equal:
        verdict = "SCREEN_INVALID (batch order mismatch)"
    summary["preliminary_checks"] = checks
    summary["preliminary_verdict"] = verdict
    (OUT / "A2B_SCREEN_RESULT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    log("=" * 62)
    log(f"SCREEN VERDICT (preliminary) = {verdict}")
    log(f"  t  median  CONTROL {c['translation_median_cm']:.4f} -> A2b {a['translation_median_cm']:.4f} cm"
        f"  ({checks['translation_improve']*100:+.2f}%)")
    log(f"  depth med CONTROL {c['depth_median_cm']:.4f} -> A2b {a['depth_median_cm']:.4f} cm"
        f"  ({checks['depth_improve']*100:+.2f}%)")
    log(f"  R median  CONTROL {c['rotation_median_deg']:.4f} -> A2b {a['rotation_median_deg']:.4f} deg")
    log(f"  R0 reference: t {r0['paths']['MAIN']['ALL']['translation_median_cm']:.4f}  "
        f"depth {r0['paths']['MAIN']['ALL']['depth_median_cm']:.4f} cm")
    log("=" * 62)
    (OUT / f"A2B_SCREEN_DONE_{RUN_TAG}.mark").write_text(verdict + "\n")
    log(f"DONE mark written: A2B_SCREEN_DONE_{RUN_TAG}.mark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
