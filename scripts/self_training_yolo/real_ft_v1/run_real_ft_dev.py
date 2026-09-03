"""REAL_FT_V1 드라이버 — 학습에서 끝내지 않는다.

    STEP 1  학습 (stock pose trainer)   완료 판정은 results.csv epoch 수로만
    STEP 2  PAPER_EVAL 평가             V1~V5 와 같은 evaluator·같은 manifest
    STEP 3  지표 + G1~G6 gate           REAL_FT_V1_METHOD_LOCK 에 사전등록된 기준
    STEP 4  알림 = '완료' 가 아니라 판정

지표 함수는 `v3_dev_metrics` 에서 **import** 한다.  복사하면 조건이 뒤집힌 이력이 있다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo" / "v3"))

import v3_dev_metrics as M  # noqa: E402

RESULTS = REPO_ROOT / "data/pallet/results/paper_real_ft_v1"
METHOD_LOCK = RESULTS / "REAL_FT_V1_METHOD_LOCK.json"
DATASETS = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/paper_real_ft_v1"
RUNS = REPO_ROOT / "challenge/yolo_pose_one_model/paper_real_ft_v1"
EVAL_OUT = REPO_ROOT / "data/pallet/results/paper_eval_real_ft_v1/arms"
V3_EVAL = REPO_ROOT / "data/pallet/results/paper_eval_v3/arms"
EVALUATOR = REPO_ROOT / "challenge/evaluation_v2/paper_real_eval.py"
MANIFESTS = REPO_ROOT / "challenge/real_gt_v2/manifests"
R0 = REPO_ROOT / (
    "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
    "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")

ARM = "REAL_FT"
GROSS_PX = 20.0

# V1/V2/V3 와 동일.  단 loss 는 stock (true-ignore 아님).
TRAIN_ARGS = dict(
    imgsz=640, batch=32, optimizer="SGD", lr0=0.002, lrf=0.01, cos_lr=True,
    momentum=0.937, weight_decay=0.0005, warmup_epochs=1.0, patience=0,
    box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, degrees=0.0, translate=0.1, scale=0.25,
    shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
    mosaic=0.15, close_mosaic=3, mixup=0.0, copy_paste=0.0, erasing=0.4,
    seed=42, deterministic=True, val=False, plots=False, save_period=-1,
    workers=2, cache=False,
)


def notify(message: str) -> None:
    hook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not hook:
        print(f"[notify skipped] {message}", flush=True)
        return
    try:
        import urllib.request

        request = urllib.request.Request(
            hook, data=json.dumps({"content": message[:1900]}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(request, timeout=20).read()
    except Exception as exc:  # noqa: BLE001
        print(f"[notify failed] {exc}", flush=True)


def free_memory_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / (1024 ** 2)
    return float("inf")


def completed_epochs(name: str) -> int:
    results = RUNS / name / "results.csv"
    if not results.exists():
        return 0
    return max(len(results.read_text().strip().splitlines()) - 1, 0)


def preflight(dataset: Path) -> dict:
    """P1~P3.  선언이 아니라 확인한다."""

    report = json.loads((dataset / "BUILD_REPORT.json").read_text())

    # P2 — replay 멤버십이 V1~V5 와 같은가
    reference = sorted(
        p.name for p in
        (REPO_ROOT / "challenge/yolo_pose_one_model/datasets/paper_selftrain_v3"
         / "V3B_TRUE_IGNORE_AMBIG" / "images" / "train").iterdir()
        if p.name.startswith("replay__"))
    import hashlib
    expected = hashlib.sha256("\n".join(reference).encode()).hexdigest()
    p2 = report["replay_membership_sha256"] == expected

    # P1 — loader 가 라벨을 실제로 읽는가, 그리고 좌표가 보존되는가.
    #   심링크를 resolve 한 train.txt 는 ultralytics 가 라벨을 못 찾아 전 프레임을
    #   조용히 background 로 넣는다.  '배경 0' 을 먼저 확인한다.
    from ultralytics.data.dataset import YOLODataset
    import yaml
    config = yaml.safe_load((dataset / "data.yaml").read_text())
    loader = YOLODataset(
        img_path=str(dataset / "train.txt"), data=config, task="pose",
        augment=False, rect=False, imgsz=640)
    labels = loader.labels
    backgrounds = sum(1 for label in labels
                      if np.asarray(label["keypoints"]).reshape(-1, 9, 3).shape[0] == 0)
    real_seen = sum(1 for label in labels
                    if Path(label["im_file"]).name.startswith("real__"))
    checked = 0
    max_drift = 0.0
    for label in labels:
        name = Path(label["im_file"]).name
        if not name.startswith("real__"):
            continue
        loaded = np.asarray(label["keypoints"], dtype=float).reshape(-1, 9, 3)
        if loaded.shape[0] == 0:
            continue
        text = (dataset / "labels" / "train" / name).with_suffix(".txt").read_text()
        raw = np.asarray(text.split()[5:], dtype=float).reshape(9, 3)
        supervised = raw[:, 2] > 0
        if not supervised.any():
            continue
        checked += 1
        # normalized=True 이므로 loader 좌표도 정규화 상태다.
        max_drift = max(max_drift, float(
            np.abs(loaded[0][supervised, :2] - raw[supervised, :2]).max()))
        if checked >= 300:
            break
    p1 = backgrounds == 0 and real_seen > 0 and checked > 0 and max_drift < 1e-3

    # P3 — 학습 이미지가 PAPER_EVAL 과 픽셀 동일하지 않은가
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
    from eval_workspace import load_frames, evaluation_population_views
    workspace = REPO_ROOT / "data/evaluation/pallet_eval_v1"
    population = evaluation_population_views(
        load_frames(workspace))["PAPER_EVAL_POSITIVE"]
    eval_hashes = {hashlib.sha256((workspace / row["image_path"]).read_bytes()).hexdigest()
                   for row in population}
    duplicates = 0
    for image in (dataset / "images" / "train").iterdir():
        if not image.name.startswith("real__"):
            continue
        if hashlib.sha256(image.resolve().read_bytes()).hexdigest() in eval_hashes:
            duplicates += 1
    p3 = duplicates == 0

    return {"P1_loader_preserves_coordinates": p1, "P1_max_drift": max_drift,
            "P1_labels_checked": checked, "P1_backgrounds": backgrounds,
            "P1_real_entries_seen": real_seen,
            "P2_replay_membership_identical": p2,
            "P3_no_eval_image_in_training": p3, "P3_duplicates": duplicates,
            "status": "PASS" if (p1 and p2 and p3) else "FAIL"}


def train_arm(epochs: int, tag: str) -> dict:
    from ultralytics import YOLO

    name = f"{ARM}__{tag}"
    weights = RUNS / name / "weights" / "last.pt"
    done = completed_epochs(name)
    if weights.exists() and done >= epochs:
        return {"status": "ALREADY_DONE", "epochs_completed": done,
                "weights": str(weights)}
    available = free_memory_gb()
    if available < 6.0:
        raise SystemExit(f"REFUSING_TO_START: 가용 메모리 {available:.1f} GB")
    print(f"\n{'=' * 70}\n{name}  epochs={epochs}  free={available:.1f}GB\n"
          f"{'=' * 70}", flush=True)
    model = YOLO(str(R0), task="pose")
    model.train(data=str(DATASETS / ARM / "data.yaml"), epochs=epochs,
                project=str(RUNS), name=name, exist_ok=True, **TRAIN_ARGS)
    if not weights.exists() or weights.stat().st_size == 0:
        return {"status": "FAILED_NO_CHECKPOINT", "expected": str(weights)}
    done = completed_epochs(name)
    if done < epochs:
        return {"status": "FAILED_INCOMPLETE_EPOCHS",
                "epochs_completed": done, "epochs_expected": epochs}
    return {"status": "OK", "epochs_completed": done, "weights": str(weights)}


def evaluate(name: str, weights: Path) -> tuple[Path, Path]:
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    report = EVAL_OUT / f"{name}.json"
    per_frame = EVAL_OUT / f"{name}_per_frame.csv"
    if report.exists() and per_frame.exists():
        print(f"  {name}: 평가 이미 있음 — 건너뜀", flush=True)
        return report, per_frame
    command = [
        sys.executable, str(EVALUATOR),
        "--positive-manifest", str(MANIFESTS / "PAPER_EVAL_ALL_POS.json"),
        "--negative-manifest", str(MANIFESTS / "DEV_NEG2689.json"),
        "--population-role", "DEV",
        "--weights", str(weights),
        "--migration-gate", str(REPO_ROOT / "challenge/real_gt_v2/MIGRATION_GATE.json"),
        "--symmetry-contract",
        str(REPO_ROOT / "challenge/real_gt_v2/SYMMETRY_CONTRACT.json"),
        "--object-migration-gate",
        f"wood_small_80x59x14={REPO_ROOT}/challenge/real_gt_v2/wood_audit/"
        "migration/MIGRATION_GATE.json",
        "--out", str(report),
        "--per-frame-out", str(per_frame),
        "--report-out", str(EVAL_OUT / f"{name}_report.md"),
        "--device", "0",
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)
    if not report.exists() or not per_frame.exists():
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit(f"EVALUATION_PRODUCED_NO_ARTIFACT: {name}")
    return report, per_frame


def tail_stats(base_rows: dict, arm_rows: dict, context: dict, subset=None) -> dict:
    """paired 집합에서 p90 과 gross20 — G3/G4 용."""

    base_all: list[float] = []
    arm_all: list[float] = []
    base_px: list[float] = []
    arm_px: list[float] = []
    for frame, item in context.items():
        if subset is not None and not subset(item):
            continue
        base_row, arm_row = base_rows.get(frame), arm_rows.get(frame)
        if base_row is None or arm_row is None:
            continue
        if not (M.detected(base_row) and M.detected(arm_row)):
            continue
        a = M.parse(base_row["top_keypoint_supervised_errors_px"])
        b = M.parse(arm_row["top_keypoint_supervised_errors_px"])
        if len(a) != len(b) or not a:
            continue
        diagonal = item["diagonal"]
        if not np.isfinite(diagonal) or diagonal <= 1e-6:
            continue
        base_px += a
        arm_px += b
        base_all += [v / diagonal for v in a]
        arm_all += [v / diagonal for v in b]
    if not base_all:
        return {}
    return {
        "n_keypoints": len(base_all),
        "base_nme_p90": float(np.percentile(base_all, 90)),
        "arm_nme_p90": float(np.percentile(arm_all, 90)),
        "base_gross20": float(np.mean(np.asarray(base_px) > GROSS_PX)),
        "arm_gross20": float(np.mean(np.asarray(arm_px) > GROSS_PX)),
        "base_px_median": float(np.median(base_px)),
        "arm_px_median": float(np.median(arm_px)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    args = parser.parse_args()

    lock = json.loads(METHOD_LOCK.read_text())
    if lock["training_labels"]["eligibility_gate"] != "VERIFIED":
        raise SystemExit(
            "ELIGIBILITY_GATE_NOT_VERIFIED — 3d-expert 규약 판정 전에는 학습 금지")
    epochs = 1 if args.stage == "smoke" else int(lock["recipe"]["epochs"])
    tag = args.stage.upper()
    RESULTS.mkdir(parents=True, exist_ok=True)

    checks = preflight(DATASETS / ARM)
    (RESULTS / f"REAL_FT_PREFLIGHT_{tag}.json").write_text(
        json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2), flush=True)
    if checks["status"] != "PASS":
        notify(f"REAL_FT_V1 {tag} PREFLIGHT FAIL — 학습하지 않음\n{checks}")
        return 1

    state = {"schema_version": "real_ft_v1_driver_state_v1", "stage": args.stage,
             "started": datetime.now(timezone.utc).isoformat(), "epochs": epochs,
             "preflight": checks}
    state_path = RESULTS / f"REAL_FT_DRIVER_STATE_{tag}.json"

    outcome = train_arm(epochs, tag)
    state["training"] = outcome
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    if outcome["status"] not in ("OK", "ALREADY_DONE"):
        notify(f"REAL_FT_V1 {tag} 학습 실패: {outcome['status']}")
        return 1
    if args.stage == "smoke":
        print("smoke 통과 — full 로 진행 가능", flush=True)
        return 0

    name = f"{ARM}__{tag}"
    evaluate(name, Path(outcome["weights"]))

    context = M.gt_context()
    rows = {
        "R0": M.load_per_frame("R0", V3_EVAL),
        ARM: M.load_per_frame(name, EVAL_OUT),
    }
    detection = {key: M.detection_rates(value, context) for key, value in rows.items()}
    day = lambda item: item["paper_domain"] == "daytime"          # noqa: E731
    night = lambda item: item["paper_domain"] == "nighttime"      # noqa: E731
    paired = {
        "ALL": M.paired_nme(rows["R0"], rows[ARM], context),
        "daytime": M.paired_nme(rows["R0"], rows[ARM], context, day),
        "nighttime": M.paired_nme(rows["R0"], rows[ARM], context, night),
    }
    tails = {
        "ALL": tail_stats(rows["R0"], rows[ARM], context),
        "daytime": tail_stats(rows["R0"], rows[ARM], context, day),
        "nighttime": tail_stats(rows["R0"], rows[ARM], context, night),
    }

    base_nme = paired["ALL"]["base_nme_median"]
    arm_nme = paired["ALL"]["arm_nme_median"]
    night_base = paired["nighttime"]["base_nme_median"]
    night_arm = paired["nighttime"]["arm_nme_median"]
    gates = [
        {"id": "G1", "primary": True, "rule": "ALL paired NME < R0",
         "pass": bool(arm_nme < base_nme), "base": base_nme, "arm": arm_nme},
        {"id": "G2", "rule": "Night paired NME <= R0",
         "pass": bool(night_arm <= night_base), "base": night_base, "arm": night_arm},
        {"id": "G3", "rule": "NME p90 <= R0",
         "pass": bool(tails["ALL"]["arm_nme_p90"] <= tails["ALL"]["base_nme_p90"]),
         "base": tails["ALL"]["base_nme_p90"], "arm": tails["ALL"]["arm_nme_p90"]},
        {"id": "G4", "rule": "gross20 <= R0",
         "pass": bool(tails["ALL"]["arm_gross20"] <= tails["ALL"]["base_gross20"]),
         "base": tails["ALL"]["base_gross20"], "arm": tails["ALL"]["arm_gross20"]},
        {"id": "G5", "rule": "Day detection >= 0.95",
         "pass": bool(detection[ARM].get("daytime", {}).get("rate", 0) >= 0.95),
         "arm": detection[ARM].get("daytime", {}).get("rate")},
        {"id": "G6", "rule": "Night detection >= R0",
         "pass": bool(detection[ARM].get("nighttime", {}).get("rate", 0)
                      >= detection["R0"].get("nighttime", {}).get("rate", 0)),
         "base": detection["R0"].get("nighttime", {}).get("rate"),
         "arm": detection[ARM].get("nighttime", {}).get("rate")},
    ]
    primary = next(g for g in gates if g.get("primary"))
    verdict = "PASSED" if primary["pass"] else "FAILED"
    report = {"schema_version": "real_ft_v1_dev_metrics_v1",
              "generated": datetime.now(timezone.utc).isoformat(),
              "detection": detection, "paired_nme": paired, "tails": tails,
              "gates": gates, "REAL_FT_V1_METHOD_STATUS": verdict}
    (RESULTS / "REAL_FT_V1_DEV_METRICS.json").write_text(
        json.dumps(report, indent=2) + "\n")

    delta = paired["ALL"]["frame_delta"]
    lines = [f"REAL_FT_V1 = {verdict}",
             f"ALL NME   R0 {base_nme:.4f} -> REAL_FT {arm_nme:.4f}",
             f"Night NME R0 {night_base:.4f} -> {night_arm:.4f}",
             f"p90       {tails['ALL']['base_nme_p90']:.4f} -> "
             f"{tails['ALL']['arm_nme_p90']:.4f}",
             f"gross20   {tails['ALL']['base_gross20']:.3f} -> "
             f"{tails['ALL']['arm_gross20']:.3f}",
             f"frame delta {delta['median']:+.5f} "
             f"[{delta['low']:+.5f}, {delta['high']:+.5f}]"]
    lines += [f"{g['id']} {'PASS' if g['pass'] else 'FAIL'}  {g['rule']}"
              for g in gates]
    summary = "\n".join(lines)
    print("\n" + summary, flush=True)
    state["verdict"] = verdict
    state["finished"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    notify(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
