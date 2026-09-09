"""§13~§22 — cheap continuation screen.  학습에서 끝내지 않고 판정까지 간다.

  C0 = R0 + 5ep on CONTROL(g38_legacy_v1v2_p0_tex20k)
  D1 = R0 + 5ep on DIVERSE_SWAP
유일 변수는 train manifest 다.  loss·augmentation·optimizer·init 전부 동일.
"""
from __future__ import annotations
import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
DSROOT = REPO / "challenge/yolo_pose_one_model/datasets"
OUT = REPO / "data/pallet/results/low_angle_diversity_v1"
RUNS = REPO / "challenge/yolo_pose_one_model/low_angle_diversity_v1/runs"
CLOSURE = REPO / "data/pallet/results/paper_pose_metric_closure_v1"
R0_CKPT = (REPO / "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
           "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")
PY_ENV = "/home/minjae/anaconda3/envs/pallet-yolo26/bin/python"
TAG = "r1"
EPOCHS = 5
ARMS = {"C0": DSROOT / "g38_legacy_v1v2_p0_tex20k",
        "D1": DSROOT / "low_angle_diversity_v1_swap"}
PROGRESS = OUT / f"SCREEN_PROGRESS_{TAG}.log"

# §20 사전등록 gate — 결과 보기 전 고정
GATE = {"depth_improve_min": 0.05, "translation_improve_min": 0.05,
        "p90_degrade_max": 0.05, "rotation_degrade_max": 0.05, "corner_degrade_max": 0.05}


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def assert_no_custom_loss():
    """§14 — A2b/LC 가 섞이면 즉시 STOP."""
    import os
    bad = [k for k in ("A2B_CONFIG", "PSPC_CONFIG", "C4_CONFIG", "DIFFPNP_CONFIG") if os.environ.get(k)]
    if bad:
        raise RuntimeError(f"custom loss env 가 설정돼 있다: {bad}")
    if "a2b_loss" in sys.modules or "pallet_yolo_loss" in sys.modules:
        raise RuntimeError("custom loss 모듈이 import 돼 있다")
    log("LC_TERM_ACTIVE = FALSE")
    log("CUSTOM_LOSS_ACTIVE = FALSE")


def r0_recipe():
    """§15 — R0 의 실제 args.yaml 을 그대로 쓴다 (A2b 의 geometry-off 설정 재사용 금지)."""
    import yaml
    a = yaml.safe_load((R0_CKPT.parents[1] / "args.yaml").read_text())
    drop = {"model", "data", "epochs", "project", "name", "save_dir", "exist_ok",
            "resume", "mode", "task", "time", "cfg", "tracker"}
    return {k: v for k, v in a.items() if k not in drop and v is not None or k in
            ("degrees", "translate", "scale", "shear", "perspective", "fliplr", "flipud",
             "mosaic", "mixup", "cutmix", "copy_paste", "hsv_h", "hsv_s", "hsv_v", "erasing",
             "warmup_epochs", "lr0", "lrf", "momentum", "weight_decay", "box", "cls", "dfl",
             "pose", "kobj", "rle", "angle", "patience", "close_mosaic", "seed", "batch",
             "imgsz", "workers", "device", "optimizer", "cos_lr", "single_cls", "amp",
             "deterministic", "rect", "val")}


def train_arm(arm, data_dir, recipe):
    from ultralytics.models.yolo.pose import PoseTrainer
    ov = dict(recipe)
    ov.update(dict(model=str(R0_CKPT), data=str(data_dir / "data.yaml"), epochs=EPOCHS,
                   project=str(RUNS), name=f"{arm}_{TAG}", exist_ok=True,
                   patience=0, save_period=-1, plots=False, val=True))
    log(f"=== TRAIN {arm}  data={data_dir.name} ===")
    PoseTrainer(overrides=ov).train()
    last = RUNS / f"{arm}_{TAG}" / "weights/last.pt"
    if not last.is_file():
        raise RuntimeError(f"{arm}: last.pt 없음")
    log(f"{arm} done -> {last}")
    return last


def infer(arm, weights):
    import cv2
    from ultralytics import YOLO
    lock = json.loads((CLOSURE / "INFERENCE_REPLAY_LOCK.json").read_text())
    assert lock["status"] == "FROZEN"
    spec = lock["recipe"]
    frames = json.loads((CLOSURE / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]
    m = YOLO(str(weights), task="pose")
    pad, imgsz, conf = int(spec["pad_px"]), int(spec["input_size"]), float(spec["confidence_floor"])
    preds, nod = {}, 0
    for fr in frames:
        img = cv2.imread(str(REPO / fr["image"]))
        p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        r = m.predict(p, conf=conf, imgsz=imgsz, augment=False, half=False,
                      device=spec["device"], verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            preds[fr["frame_id"]] = {"status": "NO_DETECTION"}; nod += 1; continue
        sc = r.boxes.conf.detach().cpu().numpy(); b = int(np.argmax(sc))
        kp = r.keypoints.xy.detach().cpu().numpy()[b] - pad if r.keypoints is not None else None
        kc = (r.keypoints.conf.detach().cpu().numpy()[b]
              if r.keypoints is not None and r.keypoints.conf is not None else None)
        preds[fr["frame_id"]] = {"status": "OK",
                                 "box_xyxy": (r.boxes.xyxy.detach().cpu().numpy()[b] - pad).tolist(),
                                 "box_conf": float(sc[b]),
                                 "keypoints_xy": kp.tolist() if kp is not None else None,
                                 "keypoints_conf": kc.tolist() if kc is not None else None,
                                 "detections": int(len(sc))}
    name = f"LAD_{arm}"
    (CLOSURE / "predictions" / f"{name}.json").write_text(json.dumps({
        "schema_version": "frozen_arm_prediction_v1", "arm": name,
        "checkpoint": str(weights.relative_to(REPO)),
        "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "recipe": spec, "n_frames": len(frames), "no_detection": nod,
        "new_training": 0, "checkpoint_reselection": 0, "frames": preds}, indent=2) + "\n")
    log(f"{arm} inference: {len(frames)} frames, no_detection {nod}")
    return name


def evaluate(name):
    r = subprocess.run([PY_ENV, str(REPO / "scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py"),
                        "--pose-object-contract", str(CLOSURE / "POSE_EVAL_OBJECT_CONTRACT.json"),
                        "--arm", name], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"eval 실패 {name}:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    src = CLOSURE / f"POSE_EVALUATION_{name}.json"
    shutil.copy(src, OUT / f"POSE_EVALUATION_{name}.json")
    return json.loads(src.read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"driver start tag={TAG}")
    assert_no_custom_loss()
    sw = json.loads((OUT / "SWAP_DESIGN.json").read_text())
    td = json.loads((OUT / "TREATMENT_DATASET.json").read_text())
    lc = json.loads((OUT / "POOL_LABEL_CONTRACT.json").read_text())
    if lc["verdict"] != "PASS" or not td["total_N_equal"]:
        log("preflight FAIL — STOP"); return 1
    log(f"swap_N={sw['swap_N']}  identification={sw['identification']}")

    recipe = r0_recipe()
    (OUT / "SCREEN_RECIPE.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    w = {a: train_arm(a, d, recipe) for a, d in ARMS.items()}
    ev = {}
    for a in ARMS:
        ev[a] = evaluate(infer(a, w[a]))
        log(f"{a} evaluated")

    r0 = json.loads((CLOSURE / "POSE_EVALUATION_R0.json").read_text())["paths"]["MAIN"]
    c, d = ev["C0"]["paths"]["MAIN"]["ALL"], ev["D1"]["paths"]["MAIN"]["ALL"]
    imp = lambda k: (c[k] - d[k]) / c[k] if c[k] else 0.0
    pl_c, pl_d = ev["C0"]["paths"]["MAIN"]["plastic"], ev["D1"]["paths"]["MAIN"]["plastic"]
    checks = {"depth_improve": imp("depth_median_cm"),
              "translation_improve": imp("translation_median_cm"),
              "rotation_degrade": -imp("rotation_median_deg"),
              "plastic_t_improve": (pl_c["translation_median_cm"] - pl_d["translation_median_cm"]) / pl_c["translation_median_cm"],
              "plastic_depth_improve": (pl_c["depth_median_cm"] - pl_d["depth_median_cm"]) / pl_c["depth_median_cm"]}
    passed = (checks["depth_improve"] >= GATE["depth_improve_min"]
              and checks["translation_improve"] >= GATE["translation_improve_min"]
              and checks["plastic_t_improve"] > 0 and checks["plastic_depth_improve"] > 0
              and checks["rotation_degrade"] < GATE["rotation_degrade_max"])
    verdict = "STRONG_PASS" if passed else (
        "PARTIAL_SIGNAL" if (checks["depth_improve"] > 0 and checks["translation_improve"] > 0)
        else "NO_SIGNAL")
    summary = {"schema_version": "low_angle_diversity_v1_screen_v1", "tag": TAG,
               "gate": GATE, "checks": checks, "verdict": verdict,
               "swap": {k: sw[k] for k in ("swap_N", "identification", "interpretation_scope")},
               "paths": {a: ev[a]["paths"] for a in ARMS}, "R0_reference": r0}
    (OUT / "SCREEN_RESULT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log("=" * 60)
    log(f"SCREEN VERDICT = {verdict}")
    log(f"  t     C0 {c['translation_median_cm']:.4f} -> D1 {d['translation_median_cm']:.4f} cm ({checks['translation_improve']*100:+.2f}%)")
    log(f"  depth C0 {c['depth_median_cm']:.4f} -> D1 {d['depth_median_cm']:.4f} cm ({checks['depth_improve']*100:+.2f}%)")
    log(f"  plastic t {checks['plastic_t_improve']*100:+.2f}%  depth {checks['plastic_depth_improve']*100:+.2f}%")
    log(f"  R0 ref: t {r0['ALL']['translation_median_cm']:.4f}  depth {r0['ALL']['depth_median_cm']:.4f}")
    log("=" * 60)
    (OUT / f"SCREEN_DONE_{TAG}.mark").write_text(verdict + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
