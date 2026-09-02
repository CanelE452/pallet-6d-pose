"""§15 — 복원이 실제로 좌표를 개선하는지 GT 로 채점한다.  학습 **전에** 한다.

teacher 는 R0.  PAPER_EVAL 프레임에서 같은 candidate keypoint 를 두 가지로 놓고 잰다.

    RAW        teacher 가 낸 좌표
    REPAIRED   신뢰 코너 anchor 와 등록 기하로 복원한 좌표

같은 keypoint 에 대한 paired 비교다.  metric 은 NME(cuboid diagonal 정규화)와
gross(>20 px) · catastrophic(>40 px).

이 단계가 실패하면 학습하지 않는다 (§16).  threshold 를 고쳐 다시 맞추지 않는다.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo" / "v2"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

from eval_workspace import load_frames, evaluation_population_views  # noqa: E402
from keypoint_scores import per_keypoint_scores  # noqa: E402
from geometry_repair import (  # noqa: E402
    N_CORNERS, REPAIR_OK, REPAIR_CANDIDATE, repair_keypoints,
)
from pseudo_label_filters import projected_diagonal  # noqa: E402

WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
V4_RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v4"
LOCK = V4_RESULTS / "SELFTRAIN_V4_METHOD_LOCK.json"
CACHE = V4_RESULTS / "V4_PROXY_TEACHER_CACHE.json"
OUT_JSON = V4_RESULTS / "V4_REPAIR_PROXY.json"
GATE_JSON = V4_RESULTS / "V4_REPAIR_PROXY_GATE.json"
OUT_MD = REPO_ROOT / "_docs/paper/V4_REPAIR_PROXY.md"
REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"

R0 = REPO_ROOT / (
    "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
    "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")
PAD, IMGSZ, CONF_FLOOR = 100, 640, 0.001
GROSS_PX, CATASTROPHIC_PX = 20.0, 40.0
FLIP_IDX = (1, 0, 3, 2, 5, 4, 7, 6, 8)


# population view 의 `object_type` 은 짧은 이름("plastic"/"wood")이고 registry 는
# 긴 이름을 쓴다.  registry 가 정본이므로 여기서 잇는다 — 어노테이션의
# `dimensions_m` 를 직접 쓰지 않는다 (그건 프레임마다 다를 수 있다).
REGISTRY_NAME = {
    "plastic": "plastic_standard_110x130x11",
    "wood": "wood_small_80x59x14",
}


def registry_dimensions(object_type: str) -> dict:
    name = REGISTRY_NAME.get(object_type, object_type)
    for entry in json.loads(REGISTRY.read_text())["objects"]:
        if entry["object_type"] == name:
            dims = entry["physical_dimensions_m"]
            return {axis: float(dims[axis]) for axis in ("x", "y", "z")}
    raise SystemExit(f"OBJECT_TYPE_NOT_IN_REGISTRY: {object_type} -> {name}")


def canonical(frame_id: str) -> str:
    return frame_id.replace("__", ":")


def teacher_predictions(rows) -> dict:
    """R0 를 원본 + 수평반전으로 한 번씩 돌려 캐시한다 (배포 recipe 그대로)."""

    if CACHE.exists():
        return json.loads(CACHE.read_text())
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(R0), task="pose")
    cache: dict = {}
    for index, row in enumerate(rows):
        image = cv2.imread(str(WORKSPACE / row["image_path"]))
        if image is None:
            raise SystemExit(f"UNREADABLE_IMAGE: {row['image_path']}")
        height, width = image.shape[:2]
        entry = {"image_width": width, "image_height": height}
        for tag, source in (("top1", image), ("flip_top1", cv2.flip(image, 1))):
            padded = cv2.copyMakeBorder(source, PAD, PAD, PAD, PAD,
                                        cv2.BORDER_REFLECT_101)
            result = model.predict(padded, imgsz=IMGSZ, conf=CONF_FLOOR,
                                   verbose=False)[0]
            if result.boxes is None or not len(result.boxes):
                entry[tag] = None
                continue
            best = int(np.argmax(result.boxes.conf.cpu().numpy()))
            keypoints = result.keypoints.xy.cpu().numpy()[best] - PAD
            confidence = (result.keypoints.conf.cpu().numpy()[best]
                          if result.keypoints.conf is not None else np.zeros(9))
            confidence = np.nan_to_num(confidence, nan=0.0)
            if tag == "flip_top1":
                keypoints = np.stack([width - 1 - keypoints[:, 0], keypoints[:, 1]],
                                     axis=1)[list(FLIP_IDX)]
                confidence = confidence[list(FLIP_IDX)]
            entry[tag] = {
                "keypoints_xy": keypoints.tolist(),
                "keypoints_conf": confidence.tolist(),
                "box_conf": float(result.boxes.conf.cpu().numpy()[best]),
            }
        cache[canonical(row["frame_id"])] = entry
        if (index + 1) % 50 == 0:
            print(f"  teacher {index + 1}/{len(rows)}", flush=True)
    CACHE.write_text(json.dumps(cache) + "\n")
    return cache


def bootstrap_median_delta(values: np.ndarray, seed=20260902) -> dict:
    if values.size == 0:
        return {"n": 0, "median": None, "ci95": None}
    rng = np.random.default_rng(seed)
    samples = np.median(values[rng.integers(0, values.size,
                                            size=(10000, values.size))], axis=1)
    return {"n": int(values.size), "median": float(np.median(values)),
            "ci95": [float(np.percentile(samples, 2.5)),
                     float(np.percentile(samples, 97.5))]}


def summarise(pairs, key) -> dict:
    raw = np.asarray([p["raw_nme"] for p in pairs])
    repaired = np.asarray([p["repaired_nme"] for p in pairs])
    if raw.size == 0:
        return {"n": 0}
    return {
        "n": int(raw.size),
        "raw_median_nme": float(np.median(raw)),
        "repaired_median_nme": float(np.median(repaired)),
        "raw_p90_nme": float(np.percentile(raw, 90)),
        "repaired_p90_nme": float(np.percentile(repaired, 90)),
        "raw_gross_rate": float(np.mean([p["raw_px"] > GROSS_PX for p in pairs])),
        "repaired_gross_rate": float(
            np.mean([p["repaired_px"] > GROSS_PX for p in pairs])),
        "raw_catastrophic_rate": float(
            np.mean([p["raw_px"] > CATASTROPHIC_PX for p in pairs])),
        "repaired_catastrophic_rate": float(
            np.mean([p["repaired_px"] > CATASTROPHIC_PX for p in pairs])),
        "paired_delta_nme": bootstrap_median_delta(repaired - raw),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    lock = json.loads(LOCK.read_text())
    reused = lock["thresholds_reused_unchanged"]
    kp_high = float(lock["new_threshold_this_track"]["KP_HIGH_CONF"])

    rows = evaluation_population_views(load_frames(WORKSPACE))["PAPER_EVAL_POSITIVE"]
    cache = teacher_predictions(rows)

    pairs: list[dict] = []
    status_counter: collections.Counter = collections.Counter()
    for row in rows:
        frame = canonical(row["frame_id"])
        entry = cache.get(frame)
        if not entry or not entry.get("top1"):
            continue
        top = entry["top1"]
        if float(top["box_conf"]) < float(reused["box_conf"]):
            continue
        payload = json.loads((WORKSPACE / row["annotation_path"]).read_text())
        points = payload["objects"][0]["keypoint_annotations"]
        gt = np.array([p["xy"] if p.get("xy") else [np.nan, np.nan] for p in points],
                      dtype=float)
        supervised = np.array(
            [bool(p.get("visibility", 0)) and p.get("xy") is not None for p in points])
        if not np.isfinite(gt[:8]).all():
            continue
        diagonal = projected_diagonal(gt[:8])
        if not np.isfinite(diagonal) or diagonal <= 1e-6:
            continue

        dimensions = registry_dimensions(row["object_type"])
        intrinsics = payload["camera_data"]["intrinsics"]
        camera = np.array([[intrinsics["fx"], 0.0, intrinsics["cx"]],
                           [0.0, intrinsics["fy"], intrinsics["cy"]],
                           [0.0, 0.0, 1.0]], dtype=float)
        keypoints = np.asarray(top["keypoints_xy"], dtype=float)
        confidence = np.asarray(top["keypoints_conf"], dtype=float)
        flip = entry.get("flip_top1") or {}
        scores = per_keypoint_scores(
            keypoints, confidence, camera, dimensions,
            flip_keypoints_2d=(np.asarray(flip["keypoints_xy"], dtype=float)
                               if flip else None),
            flip_conf=(np.asarray(flip["keypoints_conf"], dtype=float)
                       if flip else None),
            kp_conf_threshold=float(reused["kp_conf_floor"]),
            remove_threshold=float(reused["tau_remove"]),
            flip_threshold=float(reused["tau_flip"]),
            ambiguity_threshold=float(reused["ambiguity_q"]))
        repair = repair_keypoints(
            keypoints, confidence, camera, dimensions, scores,
            (float(entry["image_width"]), float(entry["image_height"])),
            kp_high_conf=kp_high, kp_floor=float(reused["kp_conf_floor"]),
            tau_reproj=float(reused["tau_reproj"]))

        for corner in range(N_CORNERS):
            status = repair["repair_status"][corner]
            if status is not None:
                status_counter[status] += 1
            if status != REPAIR_OK or not supervised[corner]:
                continue
            raw_px = float(np.linalg.norm(keypoints[corner] - gt[corner]))
            repaired_px = float(np.linalg.norm(
                np.asarray(repair["repaired_xy"][corner]) - gt[corner]))
            pairs.append({
                "frame": frame, "kp": corner,
                "paper_domain": row.get("paper_domain"),
                "teacher_conf": float(confidence[corner]),
                "occlusion": row.get("occlusion"),
                "raw_px": raw_px, "repaired_px": repaired_px,
                "raw_nme": raw_px / diagonal,
                "repaired_nme": repaired_px / diagonal,
            })

    night = [p for p in pairs if p["paper_domain"] == "nighttime"]
    report = {
        "schema_version": "v4_repair_proxy_v1",
        "teacher": "R0",
        "population": "PAPER_EVAL_319 (V4 development set)",
        "repair_status_counts": dict(status_counter),
        "groups": {
            "ALL": summarise(pairs, "ALL"),
            "nighttime": summarise(night, "night"),
            "daytime": summarise(
                [p for p in pairs if p["paper_domain"] == "daytime"], "day"),
            "night_occlusion_medium": summarise(
                [p for p in night if p["occlusion"] == "medium"], "night_occ"),
            "teacher_conf_0.50_0.80": summarise(
                [p for p in pairs if p["teacher_conf"] < 0.80], "c0"),
            "teacher_conf_0.80_0.95": summarise(
                [p for p in pairs if 0.80 <= p["teacher_conf"] < 0.95], "c1"),
        },
        "pairs": pairs,
    }

    # ── §16 proxy gate ────────────────────────────────────────────────
    night_block = report["groups"]["nighttime"]
    all_block = report["groups"]["ALL"]
    gates: dict = {}
    if night_block.get("n", 0) == 0:
        gates["P1_night_median_improves"] = {
            "pass": False, "detail": "야간 복원 표본이 0 개 — 비교 불가"}
        gates["P2_night_gross_not_worse"] = {
            "pass": False, "detail": "야간 복원 표본이 0 개 — 비교 불가"}
    else:
        gates["P1_night_median_improves"] = {
            "pass": bool(night_block["repaired_median_nme"]
                         < night_block["raw_median_nme"]),
            "detail": f"raw {night_block['raw_median_nme']:.5f} -> "
                      f"repaired {night_block['repaired_median_nme']:.5f} "
                      f"(n={night_block['n']})"}
        gates["P2_night_gross_not_worse"] = {
            "pass": bool(night_block["repaired_gross_rate"]
                         <= night_block["raw_gross_rate"]),
            "detail": f"raw {night_block['raw_gross_rate']:.3f} -> "
                      f"repaired {night_block['repaired_gross_rate']:.3f}"}
    low_power = all_block.get("n", 0) < 20 or night_block.get("n", 0) < 10
    gates["P3_power"] = {
        "pass": not low_power,
        "detail": f"복원 성공 keypoint  ALL n={all_block.get('n', 0)}  "
                  f"Night n={night_block.get('n', 0)}"
                  + ("  LOW_POWER" if low_power else "")}

    status = "PASS" if all(item["pass"] for item in gates.values()) else "FAIL"
    verdict = {
        "schema_version": "v4_repair_proxy_gate_v1",
        "status": status,
        "GEOMETRY_REPAIR_MECHANISM": "OK" if status == "PASS"
        else "GEOMETRY_REPAIR_MECHANISM_FAIL",
        "gates": gates,
        "note": ("P3 는 최소 N 임계를 새로 만든 것이 아니라, 표본이 비교를 지탱하지 "
                 "못할 때 LOW_POWER 로 표시하기 위한 것이다."),
    }

    V4_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    GATE_JSON.write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n")
    render(report, verdict)

    print(f"\n{'group':26} {'n':>5} {'raw NME':>9} {'rep NME':>9} {'Δ':>10} "
          f"{'raw gross':>10} {'rep gross':>10}")
    print("-" * 84)
    for name, block in report["groups"].items():
        if not block.get("n"):
            print(f"{name:26} {0:5d}  (표본 없음)")
            continue
        print(f"{name:26} {block['n']:5d} {block['raw_median_nme']:9.5f} "
              f"{block['repaired_median_nme']:9.5f} "
              f"{block['paired_delta_nme']['median']:+10.5f} "
              f"{block['raw_gross_rate']:10.3f} {block['repaired_gross_rate']:10.3f}")
    print(f"\n복원 상태: {dict(status_counter)}")
    print(f"\nPROXY GATE {status}")
    for key, value in gates.items():
        print(f"  {'PASS' if value['pass'] else 'FAIL'}  {key}: {value['detail']}")
    return 0


def render(report: dict, verdict: dict) -> None:
    lines = [
        "# V4 geometry repair — 학습 전 proxy test",
        "",
        "복원된 좌표가 teacher 원본보다 GT 에 가까운지 **학습 전에** 채점한다.",
        "teacher 는 R0, 모집단은 PAPER_EVAL 319 (V4 development set).",
        "",
        "같은 keypoint 에 대한 paired 비교다.  NME 는 GT projected cuboid diagonal 로",
        "정규화했고, gross 는 20 px, catastrophic 은 40 px 다.",
        "",
        "```text",
        f"{'group':26} {'n':>5} {'raw NME':>9} {'rep NME':>9} {'Δ':>10} "
        f"{'raw gross':>10} {'rep gross':>10}",
        "-" * 84,
    ]
    for name, block in report["groups"].items():
        if not block.get("n"):
            lines.append(f"{name:26} {0:5d}  (표본 없음)")
            continue
        lines.append(
            f"{name:26} {block['n']:5d} {block['raw_median_nme']:9.5f} "
            f"{block['repaired_median_nme']:9.5f} "
            f"{block['paired_delta_nme']['median']:+10.5f} "
            f"{block['raw_gross_rate']:10.3f} {block['repaired_gross_rate']:10.3f}")
    lines += ["```", "", "## 복원 상태 분포", "", "```text"]
    for key, value in sorted(report["repair_status_counts"].items()):
        lines.append(f"{key:26} {value}")
    lines += ["```", "", "## Proxy gate", "", "```text"]
    for key, value in verdict["gates"].items():
        lines.append(f"{'PASS' if value['pass'] else 'FAIL'}  {key}: {value['detail']}")
    lines += ["```", "", f"**{verdict['status']}** — "
              f"`{verdict['GEOMETRY_REPAIR_MECHANISM']}`", "", verdict["note"], ""]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
