"""파국적 corner 오차가 위치 실패인지 축 배정 실패인지 가른다.

`A_WORSE_TOP20` 의 극단 사례에서 예측 코너는 GT 와 거의 같은 자리에 있었고 라벨만
어긋나 있었다.  Hungarian 최적 배정을 풀면 세 프레임이 **똑같은 순열**을 냈다:

    gt <- pred  [1, 5, 6, 2, 0, 4, 7, 3]

규약(0~3 근면 · {0,1,4,5} 위 · flip pair (0,1)(3,2)(4,5)(7,6))으로 풀면 수직 모서리가
near-left -> far-left -> far-right -> near-right -> near-left 로 한 칸 돈다.
**수직축 90도 회전**이다.

중앙값을 쓰면 안 된다.  90도 순열은 8 코너 중 절반만 맞히므로 오차가 이봉분포가 되고
median 이 작은 쪽 봉우리에 앉는다 — 한때 이걸 "좌우 뒤집힘" 으로 잘못 읽었다.
그래서 판정은 **최대 오차**로 한다.

판정 (프레임 단위, supervised keypoint 만)
    identity            라벨 그대로
    YAW90/180/270       수직축 회전 재배정
    MIRROR              flip_idx [1,0,3,2,5,4,7,6,8]
    hungarian           8 코너 자유 배정 (자리가 맞는지)

    AXIS_PERMUTED   identity 가 크고, 어떤 회전/거울 재배정에서 max 오차가 작다
    MISLOCATED      자유 배정으로도 max 오차가 크다 = 자리 자체가 틀렸다
    OK              identity 의 max 오차가 작다

출력:
    data/pallet/results/paper_eval_v1/AXIS_FAILURES.json
    _docs/paper/generated/AXIS_FAILURES.md
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
from eval_workspace import load_frames, evaluation_population_views  # noqa: E402

WORKSPACE = REPO_ROOT / "data" / "evaluation" / "pallet_eval_v1"
RESULTS = REPO_ROOT / "data" / "pallet" / "results" / "paper_eval_v1"
OUT_JSON = RESULTS / "AXIS_FAILURES.json"
OUT_MD = REPO_ROOT / "_docs" / "paper" / "generated" / "AXIS_FAILURES.md"
CACHE = RESULTS / "visual_audit" / "MIRROR_PREDICTIONS.json"

PAD, IMGSZ, CONF_FLOOR = 100, 640, 0.001
FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6, 8]

# gt[i] <- pred[YAW90[i]].  Hungarian 이 세 파국 프레임에서 독립적으로 낸 순열이다.
YAW90 = [1, 5, 6, 2, 0, 4, 7, 3, 8]


def compose(outer: list[int], inner: list[int]) -> list[int]:
    return [inner[index] for index in outer]


YAW180 = compose(YAW90, YAW90)
YAW270 = compose(YAW180, YAW90)
PERMUTATIONS = {
    "identity": list(range(9)),
    "yaw90": YAW90,
    "yaw180": YAW180,
    "yaw270": YAW270,
    "mirror": FLIP_IDX,
    "mirror_yaw90": compose(FLIP_IDX, YAW90),
    "mirror_yaw180": compose(FLIP_IDX, YAW180),
    "mirror_yaw270": compose(FLIP_IDX, YAW270),
}

# 자리는 맞는데 라벨만 어긋났다고 부르려면, 재배정 후 **모든** 코너가 가까워야 한다.
AXIS_ABSOLUTE_PX = 25.0   # gross 경계 20 px 보다 약간 느슨 — 순열 후 잔차를 허용한다
AXIS_RATIO = 0.5          # identity 의 절반 밑으로 떨어져야 한다

MODELS = {
    "R0": REPO_ROOT / (
        "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
        "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt"),
    "R2_CONF": REPO_ROOT / ("challenge/yolo_pose_one_model/paper_selftrain_v1/"
                            "R2_CONF__FULL/weights/last.pt"),
    "R5_PROPOSED": REPO_ROOT / ("challenge/yolo_pose_one_model/paper_selftrain_v1/"
                                "R5_PROPOSED__FULL/weights/last.pt"),
}


def targets() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in evaluation_population_views(load_frames(WORKSPACE))["PAPER_EVAL_POSITIVE"]:
        payload = json.loads((WORKSPACE / row["annotation_path"]).read_text())
        points = payload["objects"][0]["keypoint_annotations"]
        xy = np.array([p["xy"] if p.get("xy") else [np.nan, np.nan] for p in points],
                      dtype=float)
        supervised = np.array(
            [bool(p.get("visibility", 0)) and p.get("xy") is not None for p in points],
            dtype=bool)
        out[row["frame_id"].replace("__", ":")] = {
            "image_path": row["image_path"],
            "paper_domain": row.get("paper_domain"),
            "object_type": row.get("object_type"),
            "gt_xy": xy,
            "supervised": supervised,
        }
    return out


def classify(keypoints: np.ndarray, gt: np.ndarray, supervised: np.ndarray) -> dict:
    from scipy.optimize import linear_sum_assignment

    def stats(perm) -> tuple[float, float]:
        errors = np.linalg.norm(keypoints[perm] - gt, axis=1)[supervised]
        if not errors.size:
            return float("nan"), float("nan")
        return float(np.median(errors)), float(np.max(errors))

    per_permutation = {name: stats(perm) for name, perm in PERMUTATIONS.items()}
    identity_med, identity_max = per_permutation["identity"]

    cost = np.linalg.norm(keypoints[:8, None, :] - gt[None, :8, :], axis=2)
    rows, cols = linear_sum_assignment(np.nan_to_num(cost, nan=1e6))
    assignment = np.empty(8, dtype=int)
    assignment[cols] = rows
    hungarian_errors = cost[assignment, np.arange(8)]
    hungarian_max = float(np.max(hungarian_errors))
    centroid = float(np.linalg.norm(
        np.nanmean(keypoints[:8], axis=0) - np.nanmean(gt[:8], axis=0)))

    best_name, best_max = None, float("inf")
    for name, (_, maximum) in per_permutation.items():
        if name != "identity" and maximum < best_max:
            best_name, best_max = name, maximum

    if identity_max <= AXIS_ABSOLUTE_PX:
        verdict = "OK"
    elif best_max < AXIS_ABSOLUTE_PX and best_max < AXIS_RATIO * identity_max:
        verdict = "AXIS_PERMUTED"
    elif hungarian_max < AXIS_ABSOLUTE_PX and hungarian_max < AXIS_RATIO * identity_max:
        verdict = "OTHER_PERMUTATION"
    else:
        verdict = "MISLOCATED"
    return {
        "identity_median_px": identity_med,
        "identity_max_px": identity_max,
        "permutation_max_px": {name: value[1] for name, value in per_permutation.items()},
        "best_permutation": best_name,
        "best_permutation_max_px": best_max,
        "hungarian_max_px": hungarian_max,
        "hungarian_assignment_gt_from_pred": assignment.tolist(),
        "centroid_delta_px": centroid,
        "verdict": verdict,
    }


def main() -> int:
    import cv2
    from ultralytics import YOLO

    frames = targets()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    need = [name for name in MODELS
            if not all(f"{name}|{fid}" in cache for fid in frames)]

    for name in need:
        path = MODELS[name]
        if not path.exists():
            raise SystemExit(f"CHECKPOINT_MISSING: {path}")
        model = YOLO(str(path), task="pose")
        print(f"{name}: {len(frames)} 프레임 추론", flush=True)
        for index, (frame_id, item) in enumerate(frames.items()):
            image = cv2.imread(str(WORKSPACE / item["image_path"]))
            if image is None:
                raise SystemExit(f"UNREADABLE_IMAGE: {WORKSPACE / item['image_path']}")
            padded = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD,
                                        cv2.BORDER_REFLECT_101)
            result = model.predict(padded, imgsz=IMGSZ, conf=CONF_FLOOR,
                                   verbose=False)[0]
            blob = None
            if result.boxes is not None and len(result.boxes):
                best = int(np.argmax(result.boxes.conf.cpu().numpy()))
                blob = {
                    "keypoints": (result.keypoints.xy.cpu().numpy()[best] - PAD).tolist(),
                    "box_conf": float(result.boxes.conf.cpu().numpy()[best]),
                }
            cache[f"{name}|{frame_id}"] = blob
            if (index + 1) % 100 == 0:
                print(f"  {index + 1}/{len(frames)}", flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache) + "\n")

    report: dict = {
        "schema_version": "mirror_failure_diagnosis_v1",
        "criterion": {
            "axis_absolute_px": AXIS_ABSOLUTE_PX,
            "axis_ratio": AXIS_RATIO,
            "statistic": "max corner error over supervised keypoints",
            "permutations": PERMUTATIONS,
            "note": ("AXIS_PERMUTED = identity 의 최대 코너 오차가 경계를 넘고, 어떤 "
                     "회전/거울 재배정에서 최대 오차가 경계 밑이면서 identity 의 절반 "
                     "아래로 떨어지는 프레임.  median 은 90 도 순열의 이봉분포에 속는다."),
        },
        "models": {},
    }

    for name in MODELS:
        per_frame = {}
        for frame_id, item in frames.items():
            blob = cache.get(f"{name}|{frame_id}")
            if blob is None:
                per_frame[frame_id] = {"verdict": "NO_DETECTION"}
                continue
            if not item["supervised"].any():
                per_frame[frame_id] = {"verdict": "NO_SUPERVISION"}
                continue
            entry = classify(np.asarray(blob["keypoints"], dtype=float),
                             item["gt_xy"], item["supervised"])
            entry["box_conf"] = blob["box_conf"]
            entry["paper_domain"] = item["paper_domain"]
            entry["object_type"] = item["object_type"]
            per_frame[frame_id] = entry
        report["models"][name] = per_frame

    render(report, frames)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_MD.relative_to(REPO_ROOT)}")
    return 0


def render(report: dict, frames: dict) -> None:
    import collections

    verdicts = ("OK", "AXIS_PERMUTED", "OTHER_PERMUTATION", "MISLOCATED",
                "NO_DETECTION", "NO_SUPERVISION")
    lines = [
        "# 파국적 corner 오차는 위치 실패가 아니라 축 배정 실패다",
        "",
        "진단 전용이다.  threshold·pool·model 을 바꾸지 않는다.",
        "",
        "`AXIS_PERMUTED` = identity 의 **최대** 코너 오차가 25 px 를 넘는데, 수직축 회전",
        "또는 거울 재배정 중 하나에서 최대 오차가 25 px 밑이면서 identity 의 절반 밑으로",
        "떨어지는 프레임.  코너가 제자리에 있고 라벨만 돌아갔다는 뜻이다.",
        "",
        "판정에 중앙값을 쓰지 않는다.  90 도 순열은 8 코너 중 절반만 맞히므로 오차가",
        "이봉분포가 되고 median 이 작은 쪽 봉우리에 앉는다 — 초판에서 이걸 \"좌우 뒤집힘\"",
        "으로 잘못 읽었다.",
        "",
        "## 프레임 판정",
        "",
        "```text",
        f"{'model':14} {'domain':11} " + " ".join(f"{v:>18}" for v in verdicts[:4]),
        "─" * 96,
    ]
    summary: dict = {}
    for name, per_frame in report["models"].items():
        for domain in ("daytime", "nighttime", "none"):
            counts = collections.Counter(
                entry["verdict"] for fid, entry in per_frame.items()
                if (frames[fid]["paper_domain"] or "none") == domain)
            if not counts:
                continue
            summary[(name, domain)] = counts
            lines.append(f"{name:14} {domain:11} " +
                         " ".join(f"{counts.get(v, 0):18d}" for v in verdicts[:4]))
    lines += ["```", "",
              "## 전체 (도메인 합)", "", "```text",
              f"{'model':14} {'frames':>7} " + " ".join(f"{v:>18}" for v in verdicts[:4]),
              "─" * 92]
    for name, per_frame in report["models"].items():
        counts = collections.Counter(e["verdict"] for e in per_frame.values())
        lines.append(f"{name:14} {len(per_frame):7d} " +
                     " ".join(f"{counts.get(v, 0):18d}" for v in verdicts[:4]))
    lines += ["```", "", "## 어떤 순열이었나", "", "```text",
              f"{'model':14} " + " ".join(f"{k:>14}" for k in
                                          ("yaw90", "yaw180", "yaw270", "mirror", "기타")),
              "─" * 90]
    for name, per_frame in report["models"].items():
        counts = collections.Counter(
            entry.get("best_permutation") for entry in per_frame.values()
            if entry.get("verdict") in ("AXIS_PERMUTED", "OTHER_PERMUTATION"))
        other = sum(v for k, v in counts.items()
                    if k not in ("yaw90", "yaw180", "yaw270", "mirror"))
        lines.append(f"{name:14} " + " ".join(
            f"{counts.get(k, 0):14d}" for k in ("yaw90", "yaw180", "yaw270", "mirror"))
            + f" {other:14d}")
    lines += ["```", "",
              "## R0 는 맞았는데 R5 가 어긋난 프레임", "", "```text"]
    base = report["models"].get("R0", {})
    for name in ("R2_CONF", "R5_PROPOSED"):
        other = report["models"].get(name, {})
        PERMUTED = ("AXIS_PERMUTED", "OTHER_PERMUTATION")
        gained = [fid for fid in other
                  if other[fid]["verdict"] in PERMUTED
                  and base.get(fid, {}).get("verdict") == "OK"]
        fixed = [fid for fid in other
                 if other[fid]["verdict"] == "OK"
                 and base.get(fid, {}).get("verdict") in PERMUTED]
        lines.append(f"{name:14} 새로 어긋남 {len(gained):3d}   고쳐짐 {len(fixed):3d}")
        for fid in sorted(gained):
            entry = other[fid]
            lines.append(
                f"    {fid:42} identity max {entry['identity_max_px']:7.1f} -> "
                f"{entry['best_permutation'] or '—':>13} max "
                f"{entry['best_permutation_max_px']:6.1f}  "
                f"centroidΔ {entry['centroid_delta_px']:5.1f}")
    lines += ["```", ""]
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
