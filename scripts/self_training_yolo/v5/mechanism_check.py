"""§18 — 고정된 R_total 이 가중치를 옳은 방향으로 주는지 학습 **전에** 한 번만 잰다.

여기서 처음 GT 를 연다.  score 는 이미 `RELIABILITY_SCORE_LOCK.json` 으로 동결됐고,
이 측정을 보고 바꾸지 않는다.

두 가지를 본다.

    1  R_total 이 gross 프레임을 가르는가 (AUC)
    2  V3-B 의 균등 노출 대신 V5 의 가중 노출을 썼을 때 학생이 **기대상** 보게 되는
       라벨 품질이 나아지는가

2 는 노출 횟수를 가중치로 한 기대값이다.  같은 프레임이 3 번 나오면 그 품질이 3 배로
계산된다 — 학생이 실제로 겪는 분포가 그렇기 때문이다.

pool 프레임에는 GT 가 없으므로, 품질은 **같은 teacher·같은 규칙을 GT 가 있는
PAPER_EVAL 에 적용했을 때의 관계**로 추정한다.  이 대리 관계의 한계를 문서에 적는다.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo" / "v2"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

from eval_workspace import load_frames, evaluation_population_views  # noqa: E402
from keypoint_scores import per_keypoint_scores  # noqa: E402
from pseudo_label_filters import geometry_scores  # noqa: E402
from reliability_score import N_CORNERS, score_pool  # noqa: E402

WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
V4 = REPO_ROOT / "data/pallet/results/paper_selftrain_v4"
V5 = REPO_ROOT / "data/pallet/results/paper_selftrain_v5"
TEACHER_CACHE = V4 / "V4_PROXY_TEACHER_CACHE.json"
LOCK = REPO_ROOT / "data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json"
REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
SCORES_CSV = V5 / "RELIABILITY_POOL_SCORES.csv"
OUT_JSON = V5 / "V5_MECHANISM_CHECK.json"
OUT_MD = REPO_ROOT / "_docs/archive/paper_pre_final_20260903/diagnostics/V5_MECHANISM_CHECK.md"

GROSS_PX = 20.0
REGISTRY_NAME = {"plastic": "plastic_standard_110x130x11",
                 "wood": "wood_small_80x59x14"}


def auc(scores, labels, higher_is_bad=True) -> float | None:
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    scores, labels = scores[finite], np.asarray(labels)[finite]
    if scores.size < 10 or len(set(labels.tolist())) < 2:
        return None
    values = scores if higher_is_bad else -scores
    ranks = values.argsort().argsort() + 1
    positives, negatives = labels.sum(), (1 - labels).sum()
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def registry_dimensions(object_type: str) -> dict:
    name = REGISTRY_NAME.get(object_type, object_type)
    for entry in json.loads(REGISTRY.read_text())["objects"]:
        if entry["object_type"] == name:
            dims = entry["physical_dimensions_m"]
            return {axis: float(dims[axis]) for axis in ("x", "y", "z")}
    raise SystemExit(f"OBJECT_TYPE_NOT_IN_REGISTRY: {object_type}")


def evaluation_records() -> list[dict]:
    """PAPER_EVAL 에 **같은 teacher·같은 신호**를 적용하고 GT 로 채점한다."""

    lock = json.loads(LOCK.read_text())
    validity = lock["keypoint_validity"]
    thresholds = lock["geometry_thresholds"]
    tau_box = float(lock["TAU_BOX"])
    cache = json.loads(TEACHER_CACHE.read_text())

    raw: list[dict] = []
    for row in evaluation_population_views(load_frames(WORKSPACE))["PAPER_EVAL_POSITIVE"]:
        frame = row["frame_id"].replace("__", ":")
        entry = cache.get(frame)
        if not entry or not entry.get("top1"):
            continue
        top = entry["top1"]
        if float(top["box_conf"]) < tau_box:
            continue
        condition = row.get("paper_domain")
        if condition not in ("daytime", "nighttime"):
            continue          # pool 은 Day/Night 뿐이므로 같은 조건만 쓴다
        payload = json.loads((WORKSPACE / row["annotation_path"]).read_text())
        points = payload["objects"][0]["keypoint_annotations"]
        gt = np.array([p["xy"] if p.get("xy") else [np.nan, np.nan] for p in points],
                      dtype=float)
        supervised = np.array(
            [bool(p.get("visibility", 0)) and p.get("xy") is not None for p in points])
        if not np.isfinite(gt[:8]).all() or not supervised[:8].any():
            continue
        dimensions = registry_dimensions(row["object_type"])
        intrinsics = payload["camera_data"]["intrinsics"]
        camera = np.array([[intrinsics["fx"], 0.0, intrinsics["cx"]],
                           [0.0, intrinsics["fy"], intrinsics["cy"]],
                           [0.0, 0.0, 1.0]], dtype=float)
        keypoints = np.asarray(top["keypoints_xy"], dtype=float)
        confidence = np.nan_to_num(
            np.asarray(top["keypoints_conf"], dtype=float), nan=0.0)
        flip = entry.get("flip_top1") or {}
        flip_xy = np.asarray(flip["keypoints_xy"], dtype=float) if flip else None
        flip_conf = np.asarray(flip["keypoints_conf"], dtype=float) if flip else None
        corner = per_keypoint_scores(
            keypoints, confidence, camera, dimensions,
            flip_keypoints_2d=flip_xy, flip_conf=flip_conf,
            kp_conf_threshold=float(validity["kp_conf_threshold"]),
            remove_threshold=float(thresholds["tau_remove"]),
            flip_threshold=float(thresholds["tau_flip"]))
        frame_scores = geometry_scores(
            keypoints, confidence >= float(validity["kp_conf_threshold"]),
            camera, dimensions, flip_xy,
            None if flip_conf is None
            else flip_conf >= float(validity["kp_conf_threshold"]))
        errors = np.linalg.norm(keypoints - gt, axis=1)[:8]
        mask = supervised[:8] & np.isfinite(errors)
        raw.append({
            "frame_id": frame, "condition": condition,
            "box_conf": float(top["box_conf"]),
            "s_reproj": float(frame_scores["s_reproj"]),
            "s_remove": float(frame_scores["s_remove"]),
            "s_flip": float(frame_scores["s_flip"])
            if frame_scores["s_flip"] is not None else float("inf"),
            "kp_conf": [float(confidence[i]) for i in range(N_CORNERS)],
            "r_remove": [float(v) for v in corner["r_remove"]],
            "r_flip": [float(v) for v in corner["r_flip"]],
            "errors_px": errors[mask].tolist(),
        })
    return score_pool(raw)


def weighted(values, weights) -> float:
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    return float((values * weights).sum() / weights.sum())


def allocate_like_v5(records: list[dict]) -> np.ndarray:
    """PAPER_EVAL 프레임에 V5 규칙을 그대로 적용했을 때의 노출 수.

    pool 의 노출을 그대로 옮길 수 없으므로(프레임이 다르다) **규칙**을 옮긴다.
    condition 별 총 슬롯은 V3-B 와 같은 비율(프레임당 평균 2.637)을 쓴다.
    """

    from reliability_score import largest_remainder_allocation

    receipt = json.loads((V5 / "V5_EXPOSURE_RECEIPT.json").read_text())
    ratio = receipt["pseudo_slots"] / receipt["n_pseudo_labels"]
    counts = np.zeros(len(records), dtype=int)
    for condition in sorted({r["condition"] for r in records}):
        index = [i for i, r in enumerate(records) if r["condition"] == condition]
        slots = int(round(len(index) * ratio))
        allocation = largest_remainder_allocation(
            [records[i]["R_total"] for i in index], slots)
        for position, i in enumerate(index):
            counts[i] = allocation[position]
    return counts


def main() -> int:
    records = evaluation_records()
    labels = np.array([1 if any(e > GROSS_PX for e in r["errors_px"]) else 0
                       for r in records])
    scores = np.array([r["R_total"] for r in records])
    corner_gross = np.array([np.mean([e > GROSS_PX for e in r["errors_px"]])
                             for r in records])
    median_error = np.array([np.median(r["errors_px"]) for r in records])
    p90_error = np.array([np.percentile(r["errors_px"], 90) for r in records])

    uniform = np.ones(len(records))
    v5_counts = allocate_like_v5(records)

    report = {
        "schema_version": "v5_mechanism_check_v1",
        "population": "PAPER_EVAL positive, daytime+nighttime, box_conf >= 0.85",
        "n_frames": len(records),
        "frame_gross_rate": float(labels.mean()),
        "auc_R_total": auc(-scores, labels, higher_is_bad=True),
        "auc_single": {
            "box_conf": auc([r["box_conf"] for r in records], labels, False),
            "s_reproj": auc([r["s_reproj"] for r in records], labels, True),
            "s_remove": auc([r["s_remove"] for r in records], labels, True),
            "s_flip": auc([r["s_flip"] for r in records], labels, True),
        },
        "expected_quality": {
            "uniform_v3b": {
                "frame_gross": weighted(labels, uniform),
                "corner_gross": weighted(corner_gross, uniform),
                "median_error_px": weighted(median_error, uniform),
                "p90_error_px": weighted(p90_error, uniform),
            },
            "reliability_weighted_v5": {
                "frame_gross": weighted(labels, v5_counts),
                "corner_gross": weighted(corner_gross, v5_counts),
                "median_error_px": weighted(median_error, v5_counts),
                "p90_error_px": weighted(p90_error, v5_counts),
            },
        },
        "exposure_on_eval": {
            "histogram": {int(k): int(v) for k, v in
                          zip(*np.unique(v5_counts, return_counts=True))},
            "note": ("pool 의 노출을 옮길 수 없으므로 V5 **규칙**을 PAPER_EVAL 에 "
                     "적용했다.  프레임당 평균 노출은 pool 과 같게 맞췄다."),
        },
        "caveat": ("pool 프레임에는 GT 가 없다.  여기 수치는 같은 teacher·같은 규칙을 "
                   "GT 가 있는 PAPER_EVAL 에 적용한 대리 측정이다."),
    }

    expected = report["expected_quality"]
    gates = {
        "M1_score_discriminates": {
            "pass": bool(report["auc_R_total"] is not None
                         and report["auc_R_total"] > 0.5),
            "detail": f"AUC(R_total) = {report['auc_R_total']:.3f}",
        },
        "M2_corner_gross_improves": {
            "pass": bool(expected["reliability_weighted_v5"]["corner_gross"]
                         < expected["uniform_v3b"]["corner_gross"]),
            "detail": f"uniform {expected['uniform_v3b']['corner_gross']:.4f} -> "
                      f"weighted {expected['reliability_weighted_v5']['corner_gross']:.4f}",
        },
        "M3_frame_gross_improves": {
            "pass": bool(expected["reliability_weighted_v5"]["frame_gross"]
                         < expected["uniform_v3b"]["frame_gross"]),
            "detail": f"uniform {expected['uniform_v3b']['frame_gross']:.4f} -> "
                      f"weighted {expected['reliability_weighted_v5']['frame_gross']:.4f}",
        },
    }
    status = "PASS" if all(g["pass"] for g in gates.values()) else "FAIL"
    report["gates"] = gates
    report["status"] = status
    report["V5_MECHANISM"] = ("OK" if status == "PASS"
                              else "V5_RELIABILITY_MECHANISM_FAIL")

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    render(report)

    print(f"  frames {report['n_frames']}  frame gross {report['frame_gross_rate']:.1%}")
    print(f"  AUC(R_total) {report['auc_R_total']:.3f}   "
          + "  ".join(f"{k} {v:.3f}" for k, v in report["auc_single"].items()))
    print(f"\n  {'metric':18} {'uniform V3-B':>13} {'weighted V5':>12}")
    for key in ("frame_gross", "corner_gross", "median_error_px", "p90_error_px"):
        print(f"  {key:18} {expected['uniform_v3b'][key]:13.4f} "
              f"{expected['reliability_weighted_v5'][key]:12.4f}")
    print(f"\n  MECHANISM {status}")
    for name, gate in gates.items():
        print(f"  {'PASS' if gate['pass'] else 'FAIL'}  {name}: {gate['detail']}")
    return 0 if status == "PASS" else 1


def render(report: dict) -> None:
    expected = report["expected_quality"]
    lines = [
        "# V5 mechanism check — 학습 전에 한 번만",
        "",
        "고정된 `R_total` 이 가중치를 옳은 방향으로 주는지 본다.  score 는 이미",
        "`RELIABILITY_SCORE_LOCK.json` 으로 동결됐고 이 결과를 보고 바꾸지 않는다.",
        "",
        f"모집단 {report['population']}, n={report['n_frames']}, "
        f"frame gross {report['frame_gross_rate']:.1%}",
        "",
        "## 분리력",
        "",
        "```text",
        f"{'signal':14} {'AUC':>7}",
        "-" * 22,
        f"{'R_total':14} {report['auc_R_total']:7.3f}",
    ]
    for name, value in report["auc_single"].items():
        lines.append(f"{name:14} {value:7.3f}")
    lines += ["```", "",
              "## 학생이 기대상 보게 되는 라벨 품질", "",
              "노출 횟수를 가중치로 한 기대값이다 — 같은 프레임이 3 번 나오면 3 배로 센다.",
              "", "```text",
              f"{'metric':18} {'uniform V3-B':>13} {'weighted V5':>12} {'변화':>9}",
              "-" * 56]
    for key in ("frame_gross", "corner_gross", "median_error_px", "p90_error_px"):
        a = expected["uniform_v3b"][key]
        b = expected["reliability_weighted_v5"][key]
        lines.append(f"{key:18} {a:13.4f} {b:12.4f} {b - a:+9.4f}")
    lines += ["```", "", "## Gate", "", "```text"]
    for name, gate in report["gates"].items():
        lines.append(f"{'PASS' if gate['pass'] else 'FAIL'}  {name}: {gate['detail']}")
    lines += ["```", "", f"**{report['status']}** — `{report['V5_MECHANISM']}`", "",
              f"> {report['caveat']}", ""]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
