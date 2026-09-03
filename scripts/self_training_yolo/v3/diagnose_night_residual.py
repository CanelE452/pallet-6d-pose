"""V2 에 남은 야간 localisation 열화의 성격을 기록한다.  방법을 고르기 위한 sweep 이 아니다.

V2 DEV 는 야간에서만 악화가 남았다 (Δframe +0.00297, CI [+0.00052, +0.00564]).
주간은 오히려 개선(-0.00092)이다.  그 잔차가 **어디에 몰려 있는지**를 본다.

분해 축

    3-1  keypoint index 별 (kp0..kp8)
    3-2  GT visibility 별 (직접 보임 / 가림)
    3-3  frame condition 별 (clean / occlusion / truncation / far / elevation)
    3-4  teacher confidence bin 별
    3-5  V2 의 pseudo-supervision status (KEEP / MASK) 에 대응하는 위치별
    3-6  축 순열 프레임을 제외한 뒤의 잔차

이 문서를 근거로 threshold·q·pseudo fraction 을 새로 고르지 않는다.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo" / "v2"))

from eval_workspace import load_frames, evaluation_population_views  # noqa: E402
from pseudo_label_filters import projected_diagonal  # noqa: E402
from keypoint_scores import ambiguity_q, per_keypoint_scores  # noqa: E402

WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
V2_EVAL = REPO_ROOT / "data/pallet/results/paper_eval_v2/arms"
V2_RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v2"
V3_RESULTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v3"
OUT_JSON = V3_RESULTS / "NIGHT_RESIDUAL_DIAGNOSIS.json"
OUT_MD = REPO_ROOT / "_docs/archive/paper_pre_final_20260903/diagnostics/V3_NIGHT_RESIDUAL_DIAGNOSIS.md"

BASE = "R0"
ARMS = ("V2A_CONF25__FULL", "V2B_KP_MASK__FULL", "V2C_AMBIG__FULL", "V2D_FULL__FULL")
PROPOSED = "V2D_FULL__FULL"
BOOTSTRAP = 10_000
SEED = 20260902

AXIS_ABSOLUTE_PX = 25.0
AXIS_RATIO = 0.5
YAW90 = (1, 5, 6, 2, 0, 4, 7, 3, 8)
FLIP_IDX = (1, 0, 3, 2, 5, 4, 7, 6, 8)


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


def load_per_frame(name: str) -> dict[str, dict]:
    path = V2_EVAL / f"{name}_per_frame.csv"
    if not path.exists():
        raise SystemExit(f"PER_FRAME_MISSING: {path}")
    return {canonical(row["frame_id"]): row
            for row in csv.DictReader(path.open(encoding="utf-8"))
            if row["kind"] == "POSITIVE"}


def detected(row: dict) -> bool:
    return row.get("top_iou50_match") == "True"


def workspace_tags() -> dict[str, dict]:
    return {canonical(row["frame_id"]): row
            for row in evaluation_population_views(
                load_frames(WORKSPACE))["PAPER_EVAL_POSITIVE"]}


def night_context(tags: dict) -> dict[str, dict]:
    """야간 프레임의 GT 좌표·감독 마스크·정규화 분모·조건 태그."""

    context: dict[str, dict] = {}
    for frame, row in tags.items():
        if row.get("paper_domain") != "nighttime":
            continue
        payload = json.loads((WORKSPACE / row["annotation_path"]).read_text())
        points = payload["objects"][0]["keypoint_annotations"]
        xy = np.array([p["xy"] if p.get("xy") else [np.nan, np.nan] for p in points],
                      dtype=float)
        visibility = np.array([int(p.get("visibility", 0)) for p in points])
        supervised = np.array(
            [bool(p.get("visibility", 0)) and p.get("xy") is not None for p in points])
        if not np.isfinite(xy[:8]).all():
            continue
        context[frame] = {
            "gt_xy": xy,
            "visibility": visibility,
            "supervised": supervised,
            "diagonal": projected_diagonal(xy[:8]),
            "q": ambiguity_q(xy),
            "tags": row,
            "image_path": row["image_path"],
        }
    return context


def bootstrap_median(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0, "median": None, "p90": None, "ci95": None}
    rng = np.random.default_rng(SEED)
    samples = np.median(values[rng.integers(0, values.size,
                                            size=(BOOTSTRAP, values.size))], axis=1)
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "ci95": [float(np.percentile(samples, 2.5)),
                 float(np.percentile(samples, 97.5))],
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


def per_keypoint_pairs(base_rows, arm_rows, context, frames):
    """(frame, supervised 순서상의 위치) 별 NME 쌍.

    evaluator 는 supervised keypoint 의 오차만 순서대로 흘리므로, GT 의 supervised
    인덱스와 zip 하면 keypoint index 를 되찾을 수 있다.
    """

    pairs = []
    for frame in frames:
        base_row, arm_row = base_rows.get(frame), arm_rows.get(frame)
        if base_row is None or arm_row is None:
            continue
        if not (detected(base_row) and detected(arm_row)):
            continue
        a = parse(base_row["top_keypoint_supervised_errors_px"])
        b = parse(arm_row["top_keypoint_supervised_errors_px"])
        item = context[frame]
        indices = np.flatnonzero(item["supervised"])
        if len(a) != len(b) or len(a) != len(indices) or not a:
            continue
        diagonal = item["diagonal"]
        if not np.isfinite(diagonal) or diagonal <= 1e-6:
            continue
        for position, index in enumerate(indices):
            pairs.append({
                "frame": frame,
                "kp": int(index),
                "base": a[position] / diagonal,
                "arm": b[position] / diagonal,
                "visibility": int(item["visibility"][index]),
            })
    return pairs


def group_delta(pairs, key) -> dict:
    groups: dict = {}
    for pair in pairs:
        groups.setdefault(key(pair), []).append(pair["arm"] - pair["base"])
    return {str(name): bootstrap_median(np.asarray(values))
            for name, values in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def main() -> int:
    tags = workspace_tags()
    context = night_context(tags)
    base_rows = load_per_frame(BASE)
    arm_rows = {arm: load_per_frame(arm) for arm in ARMS}
    frames = sorted(context)

    predictions = json.loads((V2_RESULTS / "V2_AXIS_PREDICTIONS.json").read_text())

    report: dict = {
        "schema_version": "v3_night_residual_diagnosis_v1",
        "purpose": ("V2 에 남은 야간 localisation 잔차의 성격 기록.  이 결과로 "
                    "threshold·q·pseudo fraction 을 고르지 않는다."),
        "n_night_frames": len(frames),
        "arms": {},
    }

    # teacher 신뢰도와 V2 의 KEEP/MASK 대응 위치 (§3-4, §3-5)
    lock = json.loads((V2_RESULTS / "SELFTRAIN_V2_METHOD_LOCK.json").read_text())
    r0_key = f"{BASE}|"
    teacher_conf: dict[tuple[str, int], float] = {}
    for frame in frames:
        blob = predictions.get(r0_key + frame)
        if blob:
            for index, value in enumerate(blob["kp_conf"]):
                teacher_conf[(frame, index)] = float(value)

    for arm in ARMS:
        pairs = per_keypoint_pairs(base_rows, arm_rows[arm], context, frames)
        if not pairs:
            continue
        deltas = np.asarray([p["arm"] - p["base"] for p in pairs])

        def confidence_bin(pair):
            value = teacher_conf.get((pair["frame"], pair["kp"]))
            if value is None:
                return "unknown"
            if value < 0.5:
                return "a <0.50"
            if value < 0.8:
                return "b 0.50-0.80"
            if value < 0.95:
                return "c 0.80-0.95"
            return "d >=0.95"

        def condition_bin(pair):
            row = context[pair["frame"]]["tags"]
            for key in ("occlusion", "truncation"):
                value = row.get(key)
                if value and value not in ("none", ""):
                    return f"{key}={value}"
            return "clean"

        block = {
            "overall": bootstrap_median(deltas),
            "by_keypoint": group_delta(pairs, lambda p: f"kp{p['kp']}"),
            "by_visibility": group_delta(
                pairs, lambda p: "visible" if p["visibility"] == 2 else "occluded"),
            "by_condition": group_delta(pairs, condition_bin),
            "by_teacher_confidence": group_delta(pairs, confidence_bin),
            "by_distance": group_delta(
                pairs, lambda p: f"distance={context[p['frame']]['tags'].get('distance_bin')}"),
            "by_elevation": group_delta(
                pairs, lambda p: f"elevation={context[p['frame']]['tags'].get('elevation_bin')}"),
        }

        # §3-6 축 순열 프레임 제외
        permuted = set()
        for frame in frames:
            for name in (BASE, arm.split("__")[0]):
                blob = predictions.get(f"{name}|{frame}")
                if blob is None:
                    continue
                verdict = axis_verdict(np.asarray(blob["keypoints"], dtype=float),
                                       context[frame]["gt_xy"],
                                       context[frame]["supervised"])
                if verdict in ("YAW90", "YAW270", "MIRROR", "YAW180"):
                    permuted.add(frame)
        clean_pairs = [p for p in pairs if p["frame"] not in permuted]
        block["axis_permuted_frames_excluded"] = sorted(permuted)
        block["excluding_axis_permutation"] = bootstrap_median(
            np.asarray([p["arm"] - p["base"] for p in clean_pairs]))
        report["arms"][arm] = block

    # §3-5 V2 의 pseudo KEEP/MASK 는 pool 통계다.  eval keypoint 와 직접 대응하지
    # 않으므로, pool 에서 어떤 semantic index 가 얼마나 masked 됐는지만 기록한다.
    dataset_report = json.loads(
        (V2_RESULTS / "V2_PSEUDO_DATASET_REPORT.json").read_text())
    report["pool_mask_context"] = {
        "note": ("pool 의 KEEP/MASK 는 eval frame 과 1:1 대응하지 않는다.  "
                 "semantic index 별 mask 비율만 참고로 둔다."),
        "arms": {name: {k: v for k, v in block.items()
                        if k in ("kept_corners", "masked_corners", "ambiguous")}
                 for name, block in dataset_report["arms"].items()},
    }

    V3_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    render(report)
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_MD.relative_to(REPO_ROOT)}")
    return 0


def number(value, spec="+.5f") -> str:
    return "—" if value is None else format(value, spec)


def table(title: str, block: dict, lines: list) -> None:
    lines += [f"### {title}", "", "```text",
              f"{'group':22} {'n_kp':>6} {'Δ median':>11} {'CI95':>24}",
              "-" * 66]
    for name, stats in block.items():
        ci = ("—" if stats["ci95"] is None
              else f"[{stats['ci95'][0]:+.5f}, {stats['ci95'][1]:+.5f}]")
        lines.append(f"{name:22} {stats['n']:6d} {number(stats['median']):>11} {ci:>24}")
    lines += ["```", ""]


def render(report: dict) -> None:
    lines = [
        "# V2 에 남은 야간 localisation 잔차 — 성격 진단",
        "",
        "진단 전용이다.  이 문서를 근거로 threshold·q·pseudo fraction 을 고르지 않는다.",
        "",
        "Δ 는 **NME 차이**(arm − R0)다.  양수면 그 그룹에서 arm 이 더 나쁘다.",
        f"야간 프레임 {report['n_night_frames']} 장, 공통 검출만.",
        "",
    ]
    for arm, block in report["arms"].items():
        overall = block["overall"]
        excluded = block["excluding_axis_permutation"]
        lines += [f"## {arm}", "", "```text",
                  f"{'전체':30} n={overall['n']:5d}  Δ {number(overall['median'])}  "
                  f"[{overall['ci95'][0]:+.5f}, {overall['ci95'][1]:+.5f}]",
                  f"{'축 순열 프레임 제외':30} n={excluded['n']:5d}  "
                  f"Δ {number(excluded['median'])}  "
                  + ("—" if excluded["ci95"] is None
                     else f"[{excluded['ci95'][0]:+.5f}, {excluded['ci95'][1]:+.5f}]"),
                  f"{'제외된 프레임':30} {len(block['axis_permuted_frames_excluded'])} 장",
                  "```", ""]
        if arm.startswith("V2D"):
            for title, key in (("keypoint index 별", "by_keypoint"),
                               ("GT visibility 별", "by_visibility"),
                               ("frame condition 별", "by_condition"),
                               ("teacher confidence 별", "by_teacher_confidence"),
                               ("거리 별", "by_distance"),
                               ("앙각 별", "by_elevation")):
                table(title, block[key], lines)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
