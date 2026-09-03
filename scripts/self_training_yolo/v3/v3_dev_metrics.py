"""V2 development 지표와 gate.  기준은 여기 하드코딩되어 있고 결과를 보고 고치지 않는다.

지표 (§16, §17)

    Detection rate                 coverage endpoint (별도 primary)
    Common-detected paired NME     geometry endpoint (localisation primary)
                                   R0 와 비교 모델이 **둘 다** IoU>=0.5 로 검출한
                                   프레임의 같은 supervised keypoint 만
    Axis permutation rate          yaw90 + yaw270 비율.  판정은 **최대** 코너 오차로
                                   한다 — median 은 이봉분포에 속는다
    kp_conf 분포                   감시 지표.  mask 가 kobj 를 통해 kp_conf 를
                                   누르는지 본다 (KEYPOINT_MASK_CONTRACT.json)

두 축을 한 숫자로 합치지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
# checkpoint 가 TrueIgnorePoseModel 을 pickle 하므로 unpickle 경로가 필요하다.
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo" / "v3"))

from eval_workspace import load_frames, evaluation_population_views  # noqa: E402

WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
EVAL_OUT = REPO_ROOT / "data/pallet/results/paper_eval_v3/arms"
V2_RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v2"
V3_RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v3"
OUT_JSON = V3_RESULTS / "V3_DEV_METRICS.json"
GATE_JSON = V3_RESULTS / "V3_DEV_GATE.json"
OUT_MD = REPO_ROOT / "_docs/archive/paper_pre_final_20260903/legacy_paper_outputs/generated/V3_DEV_RESULTS.md"

# V2 결과는 읽기만 한다 (V2 평가 폴더에서).  V3 는 자기 폴더에 쓴다.
V2_EVAL = REPO_ROOT / "data/pallet/results/paper_eval_v2/arms"
V2_ARMS = ("V2B_KP_MASK__FULL",)
ARMS = ("V3A_TRUE_IGNORE", "V3B_TRUE_IGNORE_AMBIG")
BASE = "R0"
PROPOSED = "V3B_TRUE_IGNORE_AMBIG"
CONTROL = "V3A_TRUE_IGNORE"
V2_REFERENCE = "V2B_KP_MASK__FULL"

AXIS_ABSOLUTE_PX = 25.0
AXIS_RATIO = 0.5
AMBIGUITY_THRESHOLD = 0.75
BOOTSTRAP = 10_000
SEED = 20260902

YAW90 = (1, 5, 6, 2, 0, 4, 7, 3, 8)
FLIP_IDX = (1, 0, 3, 2, 5, 4, 7, 6, 8)
# Day detection 이 이보다 더 떨어지면 파국적 열화로 본다 (§19-2).
DAY_DETECTION_TOLERANCE = 0.02


def compose(outer, inner):
    return tuple(inner[index] for index in outer)


PERMUTATIONS = {
    "identity": tuple(range(9)),
    "yaw90": YAW90,
    "yaw180": compose(YAW90, YAW90),
    "yaw270": compose(compose(YAW90, YAW90), YAW90),
    "mirror": FLIP_IDX,
}


def canonical(frame_id: str) -> str:
    return frame_id.replace("__", ":")


def parse(value: str) -> list[float]:
    return [float(part) for part in value.split(";") if part] if value else []


def load_per_frame(name: str, directory: Path | None = None) -> dict[str, dict]:
    path = (directory or EVAL_OUT) / f"{name}_per_frame.csv"
    if not path.exists():
        raise SystemExit(f"PER_FRAME_MISSING: {path}")
    return {canonical(row["frame_id"]): row
            for row in csv.DictReader(path.open(encoding="utf-8"))
            if row["kind"] == "POSITIVE"}


def detected(row: dict) -> bool:
    return row.get("top_iou50_match") == "True"


def gt_context() -> dict[str, dict]:
    """GT 좌표·정규화 분모·시점 모호성.  평가 전용이며 학습에 쓰이지 않는다."""

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo" / "v2"))
    from keypoint_scores import ambiguity_q
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo"))
    from pseudo_label_filters import projected_diagonal

    context: dict[str, dict] = {}
    for row in evaluation_population_views(load_frames(WORKSPACE))["PAPER_EVAL_POSITIVE"]:
        payload = json.loads((WORKSPACE / row["annotation_path"]).read_text())
        points = payload["objects"][0]["keypoint_annotations"]
        xy = np.array([p["xy"] if p.get("xy") else [np.nan, np.nan] for p in points],
                      dtype=float)
        supervised = np.array(
            [bool(p.get("visibility", 0)) and p.get("xy") is not None for p in points],
            dtype=bool)
        if not np.isfinite(xy[:8]).all():
            continue
        context[canonical(row["frame_id"])] = {
            "gt_xy": xy,
            "supervised": supervised,
            "diagonal": projected_diagonal(xy[:8]),
            "q": ambiguity_q(xy),
            "paper_domain": row.get("paper_domain"),
            "object_type": row.get("object_type"),
            "image_path": row["image_path"],
        }
    return context


def bootstrap_median(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0, "median": None, "ci95": None}
    rng = np.random.default_rng(SEED)
    samples = np.median(values[rng.integers(0, values.size,
                                            size=(BOOTSTRAP, values.size))], axis=1)
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "ci95": [float(np.percentile(samples, 2.5)),
                 float(np.percentile(samples, 97.5))],
    }


def detection_rates(rows: dict, context: dict) -> dict:
    result = {}
    for domain in ("daytime", "nighttime", None):
        keys = [f for f, item in context.items()
                if domain is None or item["paper_domain"] == domain]
        keys = [f for f in keys if f in rows]
        if not keys:
            continue
        rate = float(np.mean([detected(rows[f]) for f in keys]))
        result["ALL" if domain is None else domain] = {"n": len(keys), "rate": rate}
    return result


def paired_nme(base_rows: dict, arm_rows: dict, context: dict,
               subset=None) -> dict:
    """둘 다 검출한 프레임의 같은 supervised keypoint 만, cuboid diagonal 로 정규화."""

    base_values: list[float] = []
    arm_values: list[float] = []
    per_frame_delta: list[float] = []
    for frame, item in context.items():
        if subset is not None and not subset(item):
            continue
        base_row, arm_row = base_rows.get(frame), arm_rows.get(frame)
        if base_row is None or arm_row is None:
            continue
        if not (detected(base_row) and detected(arm_row)):
            continue
        a = parse(base_row["top_keypoint_supervised_errors_px"])
        b = parse(arm_row["top_keypoint_supervised_errors_px"])
        if len(a) != len(b) or not a:
            continue
        diagonal = item["diagonal"]
        if not np.isfinite(diagonal) or diagonal <= 1e-6:
            continue
        a_n = [value / diagonal for value in a]
        b_n = [value / diagonal for value in b]
        base_values += a_n
        arm_values += b_n
        per_frame_delta.append(float(np.median(b_n)) - float(np.median(a_n)))
    return {
        "n_frames": len(per_frame_delta),
        "base_nme_median": (float(np.median(base_values)) if base_values else None),
        "arm_nme_median": (float(np.median(arm_values)) if arm_values else None),
        "frame_delta": bootstrap_median(np.asarray(per_frame_delta)),
    }


def axis_verdict(keypoints: np.ndarray, gt: np.ndarray, supervised: np.ndarray) -> str:
    def maximum(perm) -> float:
        errors = np.linalg.norm(keypoints[list(perm)] - gt, axis=1)[supervised]
        return float(np.max(errors)) if errors.size else float("nan")

    identity = maximum(PERMUTATIONS["identity"])
    if not np.isfinite(identity):
        return "NO_SUPERVISION"
    if identity <= AXIS_ABSOLUTE_PX:
        return "OK"
    best_name, best = None, float("inf")
    for name, perm in PERMUTATIONS.items():
        if name == "identity":
            continue
        value = maximum(perm)
        if value < best:
            best_name, best = name, value
    if best < AXIS_ABSOLUTE_PX and best < AXIS_RATIO * identity:
        return best_name.upper()
    return "MISLOCATED"


def axis_rates(name: str, context: dict, predictions: dict) -> dict:
    counts = {"all": {"n": 0, "permuted": 0}, "ambiguous": {"n": 0, "permuted": 0}}
    for frame, item in context.items():
        blob = predictions.get(f"{name}|{frame}")
        if blob is None:
            continue
        verdict = axis_verdict(np.asarray(blob["keypoints"], dtype=float),
                               item["gt_xy"], item["supervised"])
        if verdict == "NO_SUPERVISION":
            continue
        permuted = verdict in ("YAW90", "YAW270")
        counts["all"]["n"] += 1
        counts["all"]["permuted"] += int(permuted)
        if np.isfinite(item["q"]) and item["q"] >= AMBIGUITY_THRESHOLD:
            counts["ambiguous"]["n"] += 1
            counts["ambiguous"]["permuted"] += int(permuted)
    for block in counts.values():
        block["rate"] = (block["permuted"] / block["n"]) if block["n"] else None
    return counts


def dump_predictions(names: dict[str, Path], context: dict, cache_path: Path) -> dict:
    """축 판정을 위해 keypoint 좌표가 필요하다.  per-frame CSV 에는 오차만 있다."""

    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = [name for name in names
               if not all(f"{name}|{frame}" in cache for frame in context)]
    if not missing:
        return cache

    import cv2
    from ultralytics import YOLO

    for name in missing:
        model = YOLO(str(names[name]), task="pose")
        print(f"  축 판정용 추론: {name}", flush=True)
        for frame, item in context.items():
            image = cv2.imread(str(WORKSPACE / item["image_path"]))
            if image is None:
                raise SystemExit(f"UNREADABLE_IMAGE: {item['image_path']}")
            padded = cv2.copyMakeBorder(image, 100, 100, 100, 100,
                                        cv2.BORDER_REFLECT_101)
            result = model.predict(padded, imgsz=640, conf=0.001, verbose=False)[0]
            blob = None
            if result.boxes is not None and len(result.boxes):
                best = int(np.argmax(result.boxes.conf.cpu().numpy()))
                confidence = (result.keypoints.conf.cpu().numpy()[best]
                              if result.keypoints.conf is not None
                              else np.zeros(9))
                blob = {
                    "keypoints": (result.keypoints.xy.cpu().numpy()[best] - 100).tolist(),
                    "kp_conf": np.nan_to_num(confidence, nan=0.0).tolist(),
                    "box_conf": float(result.boxes.conf.cpu().numpy()[best]),
                }
            cache[f"{name}|{frame}"] = blob
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache) + "\n")
    return cache


def kp_conf_summary(name: str, context: dict, predictions: dict) -> dict:
    values: list[float] = []
    for frame in context:
        blob = predictions.get(f"{name}|{frame}")
        if blob:
            values += [float(v) for v in blob["kp_conf"][:8]]
    if not values:
        return {"n": 0, "median": None, "p05": None, "below_0.5_rate": None}
    array = np.asarray(values)
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "below_0.5_rate": float(np.mean(array < 0.5)),
    }


def number(value, spec=".4f") -> str:
    return "—" if value is None else format(value, spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="FULL")
    args = parser.parse_args()

    context = gt_context()
    names = {BASE: EVAL_OUT.parent}      # placeholder, filled below
    runs = REPO_ROOT / "challenge/yolo_pose_one_model/paper_selftrain_v3"
    checkpoints = {BASE: REPO_ROOT / (
        "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
        "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")}
    for arm in ARMS:
        checkpoints[arm] = runs / f"{arm}__{args.tag}" / "weights" / "last.pt"

    rows = {BASE: load_per_frame(BASE)}
    for arm in ARMS:
        rows[arm] = load_per_frame(f"{arm}__{args.tag}")
    # V2 는 읽기만 한다 — G5 대조군.  같은 evaluator·같은 manifest 로 이미 재어 뒀다.
    rows[V2_REFERENCE] = load_per_frame(V2_REFERENCE, V2_EVAL)
    checkpoints[V2_REFERENCE] = (
        REPO_ROOT / "challenge/yolo_pose_one_model/paper_selftrain_v2"
        / V2_REFERENCE / "weights" / "last.pt")

    predictions = dump_predictions(
        checkpoints, context, V3_RESULTS / "V3_AXIS_PREDICTIONS.json")

    report: dict = {
        "schema_version": "v3_dev_metrics_v1",
        "tag": args.tag,
        "n_frames": len(context),
        "models": {},
    }
    for name in (BASE, V2_REFERENCE, *ARMS):
        block = {
            "detection": detection_rates(rows[name], context),
            "axis": axis_rates(name, context, predictions),
            "kp_conf": kp_conf_summary(name, context, predictions),
        }
        if name != BASE:
            block["paired_nme"] = {
                "ALL": paired_nme(rows[BASE], rows[name], context),
                "daytime": paired_nme(rows[BASE], rows[name], context,
                                      lambda item: item["paper_domain"] == "daytime"),
                "nighttime": paired_nme(rows[BASE], rows[name], context,
                                        lambda item: item["paper_domain"] == "nighttime"),
            }
        report["models"][name] = block

    # ── gate G1~G8.  기준은 method lock 과 이 블록에 고정되어 있다 ──────
    proposed = report["models"][PROPOSED]
    control = report["models"][CONTROL]
    base = report["models"][BASE]
    v2ref = report["models"][V2_REFERENCE]

    def rate(block, key):
        entry = block["detection"].get(key)
        return None if entry is None else entry["rate"]

    gates: dict[str, dict] = {}
    night_base, night_prop = rate(base, "nighttime"), rate(proposed, "nighttime")
    gates["G1_night_detection_above_r0"] = {
        "pass": bool(night_prop is not None and night_base is not None
                     and night_prop > night_base),
        "detail": f"R0 {number(night_base, '.3f')} -> V3-B {number(night_prop, '.3f')}",
    }
    day_base, day_prop = rate(base, "daytime"), rate(proposed, "daytime")
    gates["G2_day_detection_no_collapse"] = {
        "pass": bool(day_prop is not None and day_base is not None
                     and day_prop >= day_base - DAY_DETECTION_TOLERANCE),
        "detail": f"R0 {number(day_base, '.3f')} -> V3-B {number(day_prop, '.3f')} "
                  f"(허용 -{DAY_DETECTION_TOLERANCE})",
    }
    nme_all = proposed["paired_nme"]["ALL"]["frame_delta"]["median"]
    gates["G3_all_nme_below_r0"] = {
        "pass": bool(nme_all is not None and nme_all < 0),
        "detail": f"Δframe median {number(nme_all, '+.5f')} (음수여야 한다)",
    }
    nme_night = proposed["paired_nme"]["nighttime"]["frame_delta"]["median"]
    gates["G4_night_nme_not_worse_than_r0"] = {
        "pass": bool(nme_night is not None and nme_night <= 0),
        "detail": f"Night Δframe median {number(nme_night, '+.5f')} (0 이하여야 한다)",
    }
    v3a_nme = control["paired_nme"]["ALL"]["arm_nme_median"]
    v2b_nme = v2ref["paired_nme"]["ALL"]["arm_nme_median"]
    gates["G5_true_ignore_beats_v2_masking"] = {
        "pass": bool(v3a_nme is not None and v2b_nme is not None and v3a_nme < v2b_nme),
        "detail": f"V2B {number(v2b_nme)} -> V3-A {number(v3a_nme)}",
    }
    amb_prop = proposed["axis"]["ambiguous"]["rate"]
    amb_ctrl = control["axis"]["ambiguous"]["rate"]
    gates["G6_ambiguity_helps"] = {
        "pass": bool(amb_prop is not None and amb_ctrl is not None
                     and amb_prop < amb_ctrl),
        "detail": f"q>=0.75  V3-A {number(amb_ctrl, '.3f')} -> V3-B {number(amb_prop, '.3f')}",
    }
    axis_prop, axis_base = proposed["axis"]["all"]["rate"], base["axis"]["all"]["rate"]
    gates["G7_axis_at_most_r0"] = {
        "pass": bool(axis_prop is not None and axis_base is not None
                     and axis_prop <= axis_base),
        "detail": f"R0 {number(axis_base, '.3f')} -> V3-B {number(axis_prop, '.3f')}",
    }
    gates["G8_not_detection_only"] = {
        "pass": bool(gates["G3_all_nme_below_r0"]["pass"]
                     and gates["G4_night_nme_not_worse_than_r0"]["pass"]),
        "detail": "detection 이 좋아져도 G3/G4 가 실패하면 overall FAIL",
    }

    status = "PASS" if all(item["pass"] for item in gates.values()) else "FAIL"
    verdict = {
        "schema_version": "v3_dev_gate_v1",
        "V3_METHOD_STATUS": "DEV_PASS" if status == "PASS" else "FAILED",
        "status": status,
        "tag": args.tag,
        "gates": gates,
        "note": ("DEV 는 방법 개발 모집단이다.  CI 는 따로 보고하며 점추정 gate 를 "
                 "결과를 보고 바꾸지 않는다.  이것을 paper confirmation 이라 부르지 "
                 "않는다."),
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    GATE_JSON.write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n")
    render(report, verdict)

    print(f"\n{'model':14} {'det ALL':>8} {'det day':>8} {'det night':>10} "
          f"{'NME':>9} {'axis all':>9} {'axis q>=.75':>12} {'kp_conf med':>12}")
    print("-" * 88)
    for name in (BASE, V2_REFERENCE, *ARMS):
        block = report["models"][name]
        nme_value = (block.get("paired_nme", {}).get("ALL", {}).get("arm_nme_median")
                     if name != BASE else
                     report["models"][PROPOSED]["paired_nme"]["ALL"]["base_nme_median"])
        print(f"{name:14} {number(rate(block, 'ALL'), '.3f'):>8} "
              f"{number(rate(block, 'daytime'), '.3f'):>8} "
              f"{number(rate(block, 'nighttime'), '.3f'):>10} "
              f"{number(nme_value):>9} "
              f"{number(block['axis']['all']['rate'], '.3f'):>9} "
              f"{number(block['axis']['ambiguous']['rate'], '.3f'):>12} "
              f"{number(block['kp_conf']['median'], '.3f'):>12}")
    print(f"\nGATE {status}")
    for key, value in gates.items():
        print(f"  {'PASS' if value['pass'] else 'FAIL'}  {key}: {value['detail']}")
    return 0


def render(report: dict, verdict: dict) -> None:
    lines = [
        "# V2 development results",
        "",
        "**개발 모집단은 PAPER_EVAL 319 다.**  V2 방법이 이 모집단의 진단을 보고",
        "설계됐으므로 여기서 최종 성능을 주장하지 않는다",
        "(`_docs/archive/paper_pre_final_20260903/diagnostics/SELFTRAIN_V2_PROTOCOL.md`).",
        "",
        "## Coverage 와 Geometry — 두 축을 합치지 않는다",
        "",
        "```text",
        f"{'model':14} {'det ALL':>8} {'det day':>8} {'det night':>10} "
        f"{'NME':>9} {'axis all':>9} {'axis q>=.75':>12} {'kp_conf med':>12}",
        "-" * 88,
    ]
    for name in (BASE, V2_REFERENCE, *ARMS):
        block = report["models"][name]
        nme_value = (block.get("paired_nme", {}).get("ALL", {}).get("arm_nme_median")
                     if name != BASE else
                     report["models"][PROPOSED]["paired_nme"]["ALL"]["base_nme_median"])

        def rate(key):
            entry = block["detection"].get(key)
            return None if entry is None else entry["rate"]

        lines.append(
            f"{name:14} {number(rate('ALL'), '.3f'):>8} "
            f"{number(rate('daytime'), '.3f'):>8} "
            f"{number(rate('nighttime'), '.3f'):>10} "
            f"{number(nme_value):>9} "
            f"{number(block['axis']['all']['rate'], '.3f'):>9} "
            f"{number(block['axis']['ambiguous']['rate'], '.3f'):>12} "
            f"{number(block['kp_conf']['median'], '.3f'):>12}")
    lines += ["```", "",
              "`NME` 는 R0 와 그 arm 이 둘 다 검출한 프레임의 같은 supervised keypoint 를",
              "cuboid diagonal 로 나눈 값이다.  R0 열은 V2-D 와의 공통 프레임 기준이다.",
              "",
              "`kp_conf med` 는 감시 지표다 — per-keypoint mask 가 keypoint objectness 를",
              "통해 kp_conf 를 누르는지 본다 (배포는 kp_conf >= 0.5 를 쓴다).",
              "", "## DEV gate", "", "```text"]
    for key, value in verdict["gates"].items():
        lines.append(f"{'PASS' if value['pass'] else 'FAIL'}  {key}: {value['detail']}")
    lines += ["```", "", f"**{verdict['status']}** — "
              f"`V3_METHOD_STATUS = {verdict['V3_METHOD_STATUS']}`", "",
              verdict["note"], ""]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
