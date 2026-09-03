"""학습 없이 R0 를 여러 view 로 돌려 R0 단독보다 나은 2D teacher 가 되는지 본다.

teacher 단계에서만 판정한다.  teacher 가 R0 를 못 이기면 student 를 시작하지 않는다.

    FAST-A   R0 @640 original + horizontal flip 의 **단순 평균**
    FAST-B   R0 @640 orig/flip + R0 @960 orig/flip 의 좌표별 median
    FAST-C   호환되는 다른 source-only checkpoint 를 추가한 median

새 학습 0 회.  새 가중치 0 개.  confidence 로 좌표를 가중하지 않는다 — tuning 을
만들지 않기 위해서다.

## 비교는 반드시 같은 keypoint 집합에서

candidate 가 어려운 keypoint 를 버리고 쉬운 것만 남기면 지표가 저절로 좋아진다.
V1~V5 를 다섯 번 속인 것이 정확히 그 selection 효과다.  그래서 **candidate 가 값을
낸 keypoint 에서 R0 도 같이 재는 paired 비교**만 gate 에 쓴다.  coverage 는 따로 적는다.

## flip cache

`dump_teacher_predictions` 계열 캐시는 이미 `x -> width-1-x` 와 flip_idx 재배정을
마치고 저장한다.  여기서 다시 mirror 하면 좌표가 반대편으로 날아간다(과거 실측
1.9 px -> 127 px).  실행 시 계약을 데이터로 확인하고, 어긋나면 멈춘다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
for relative in ("scripts/self_training_yolo/v2", "scripts/self_training_yolo",
                 "scripts/evaluation"):
    sys.path.insert(0, str(REPO_ROOT / relative))

from eval_workspace import load_frames, evaluation_population_views  # noqa: E402
from keypoint_scores import ambiguity_q  # noqa: E402
from pseudo_label_filters import projected_diagonal  # noqa: E402

WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
OUT_DIR = REPO_ROOT / "data/pallet/results/paper_fast_teacher_v1"
CACHE_640 = REPO_ROOT / "data/pallet/results/paper_selftrain_v4/V4_PROXY_TEACHER_CACHE.json"
CACHE_960 = OUT_DIR / "R0_TTA960_CACHE.json"
CACHE_C = OUT_DIR / "G38_GENERIC_CACHE.json"
FREEZE = OUT_DIR / "FAST_C_MEMBERSHIP_FREEZE.json"
R0 = REPO_ROOT / (
    "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
    "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")

PAD, CONF_FLOOR = 100, 0.001
KP_CONF_FLOOR = 0.5
BOX_CONF = 0.85
GROSS_PX, CATASTROPHIC_PX = 20.0, 40.0
AMBIGUITY_Q = 0.75
N_CORNERS = 8
FLIP_IDX = (1, 0, 3, 2, 5, 4, 7, 6, 8)
YAW90 = (1, 5, 6, 2, 0, 4, 7, 3, 8)
AXIS_ABSOLUTE_PX, AXIS_RATIO = 25.0, 0.5


def compose(outer, inner):
    return tuple(inner[i] for i in outer)


PERMUTATIONS = {
    "yaw90": YAW90,
    "yaw180": compose(YAW90, YAW90),
    "yaw270": compose(compose(YAW90, YAW90), YAW90),
    "mirror": FLIP_IDX,
}


def canonical(frame_id: str) -> str:
    return frame_id.replace("__", ":")


def assert_cache_already_unflipped(cache: dict) -> float:
    residuals = []
    for entry in cache.values():
        top, flip = entry.get("top1"), entry.get("flip_top1")
        if not top or not flip:
            continue
        a = np.asarray(top["keypoints_xy"], dtype=float)
        b = np.asarray(flip["keypoints_xy"], dtype=float)
        residuals.append(float(np.median(np.linalg.norm(a - b, axis=1))))
    if not residuals:
        raise SystemExit("NO_FLIP_PREDICTIONS_IN_CACHE")
    median = float(np.median(residuals))
    if median > 20.0:
        raise SystemExit(
            f"FLIP_CACHE_LOOKS_STILL_FLIPPED: median {median:.1f} px — 재mirror 금지")
    return median


def build_context() -> dict:
    context = {}
    for row in evaluation_population_views(load_frames(WORKSPACE))["PAPER_EVAL_POSITIVE"]:
        frame = canonical(row["frame_id"])
        payload = json.loads((WORKSPACE / row["annotation_path"]).read_text())
        points = payload["objects"][0]["keypoint_annotations"]
        gt = np.array([p["xy"] if p.get("xy") else [np.nan, np.nan] for p in points],
                      dtype=float)
        supervised = np.array(
            [bool(p.get("visibility", 0)) and p.get("xy") is not None for p in points])
        if not np.isfinite(gt[:N_CORNERS]).all():
            continue
        context[frame] = {
            "gt": gt, "supervised": supervised,
            "diagonal": projected_diagonal(gt[:N_CORNERS]),
            "domain": row.get("paper_domain"),
            "image_path": row["image_path"],
        }
    return context


def views_from(cache: dict, tag: str) -> dict[str, dict]:
    out = {f"{tag}_orig": {}, f"{tag}_flip": {}}
    for frame, entry in cache.items():
        for suffix, key in (("orig", "top1"), ("flip", "flip_top1")):
            blob = entry.get(key)
            if not blob or float(blob["box_conf"]) < BOX_CONF:
                continue
            out[f"{tag}_{suffix}"][frame] = {
                "xy": np.asarray(blob["keypoints_xy"], dtype=float),
                "conf": np.nan_to_num(
                    np.asarray(blob["keypoints_conf"], dtype=float), nan=0.0),
            }
    return out


def run_inference(context: dict, weights: Path, imgsz: int, cache_path: Path) -> dict:
    """checkpoint 하나를 original + flip 으로 한 번씩.  결과는 원본 좌표계다."""

    if cache_path.exists():
        return json.loads(cache_path.read_text())
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(weights), task="pose")
    cache: dict = {}
    for index, (frame, item) in enumerate(context.items()):
        image = cv2.imread(str(WORKSPACE / item["image_path"]))
        if image is None:
            raise SystemExit(f"UNREADABLE_IMAGE: {item['image_path']}")
        height, width = image.shape[:2]
        entry = {"image_width": width, "image_height": height}
        for tag, source in (("top1", image), ("flip_top1", cv2.flip(image, 1))):
            padded = cv2.copyMakeBorder(source, PAD, PAD, PAD, PAD,
                                        cv2.BORDER_REFLECT_101)
            try:
                result = model.predict(padded, imgsz=imgsz, conf=CONF_FLOOR,
                                       verbose=False)[0]
            except RuntimeError as exc:            # OOM 등
                raise SystemExit(f"INFERENCE_UNAVAILABLE imgsz={imgsz}: {exc}") from exc
            if result.boxes is None or not len(result.boxes):
                entry[tag] = None
                continue
            best = int(np.argmax(result.boxes.conf.cpu().numpy()))
            keypoints = result.keypoints.xy.cpu().numpy()[best] - PAD
            confidence = (result.keypoints.conf.cpu().numpy()[best]
                          if result.keypoints.conf is not None else np.zeros(9))
            confidence = np.nan_to_num(confidence, nan=0.0)
            if tag == "flip_top1":
                keypoints = np.stack(
                    [width - 1 - keypoints[:, 0], keypoints[:, 1]], axis=1)[list(FLIP_IDX)]
                confidence = confidence[list(FLIP_IDX)]
            entry[tag] = {"keypoints_xy": keypoints.tolist(),
                          "keypoints_conf": confidence.tolist(),
                          "box_conf": float(result.boxes.conf.cpu().numpy()[best])}
        cache[frame] = entry
        if (index + 1) % 50 == 0:
            print(f"  imgsz={imgsz} {index + 1}/{len(context)}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache) + "\n")
    return cache


def merge(views: dict[str, dict], frame: str, mode: str, minimum: int):
    """mode 'mean' 은 FAST-A(두 view 평균), 'median' 은 FAST-B/C."""

    stacks, confidences = [], []
    for view in views.values():
        blob = view.get(frame)
        if blob is None:
            continue
        stacks.append(blob["xy"])
        confidences.append(blob["conf"])
    if len(stacks) < minimum:
        return None, None
    points = np.stack(stacks)
    valid = np.stack(confidences) >= KP_CONF_FLOOR
    merged = np.full((9, 2), np.nan)
    for index in range(9):
        usable = points[valid[:, index], index, :]
        if len(usable) < minimum:
            continue
        merged[index] = (usable.mean(axis=0) if mode == "mean"
                         else np.median(usable, axis=0))
    return merged, np.isfinite(merged).all(axis=1)


def axis_rate(points_by_frame: dict, context: dict, subset=None) -> dict:
    total = permuted = ambiguous_total = ambiguous_permuted = 0
    for frame, item in context.items():
        if subset is not None and item["domain"] != subset:
            continue
        points = points_by_frame.get(frame)
        if points is None or not np.isfinite(points[:N_CORNERS]).all():
            continue
        mask = item["supervised"][:N_CORNERS]
        if not mask.any():
            continue
        gt = item["gt"]
        identity = float(np.max(
            np.linalg.norm(points[:N_CORNERS] - gt[:N_CORNERS], axis=1)[mask]))
        best = float("inf")
        best_name = None
        for name, perm in PERMUTATIONS.items():
            value = float(np.max(np.linalg.norm(
                points[list(perm)][:N_CORNERS] - gt[:N_CORNERS], axis=1)[mask]))
            if value < best:
                best, best_name = value, name
        is_permuted = (identity > AXIS_ABSOLUTE_PX and best < AXIS_ABSOLUTE_PX
                       and best < AXIS_RATIO * identity
                       and best_name in ("yaw90", "yaw270"))
        total += 1
        permuted += int(is_permuted)
        q = ambiguity_q(points)
        if np.isfinite(q) and q >= AMBIGUITY_Q:
            ambiguous_total += 1
            ambiguous_permuted += int(is_permuted)
    return {
        "n": total, "rate": permuted / total if total else None,
        "n_ambiguous": ambiguous_total,
        "ambiguous_rate": ambiguous_permuted / ambiguous_total
        if ambiguous_total else None,
    }


def paired_stats(candidate: dict, availability: dict, baseline: dict,
                 context: dict, domain=None) -> dict:
    """candidate 가 값을 낸 keypoint 에서만, R0 와 **같은 집합**으로 잰다."""

    cand_px, base_px, cand_nme, base_nme = [], [], [], []
    for frame, item in context.items():
        if domain is not None and item["domain"] != domain:
            continue
        points = candidate.get(frame)
        available = availability.get(frame)
        reference = baseline.get(frame)
        if points is None or available is None or reference is None:
            continue
        mask = item["supervised"][:N_CORNERS] & available[:N_CORNERS]
        mask &= np.isfinite(reference["xy"][:N_CORNERS]).all(axis=1)
        if not mask.any():
            continue
        gt = item["gt"][:N_CORNERS]
        c = np.linalg.norm(points[:N_CORNERS] - gt, axis=1)[mask]
        b = np.linalg.norm(reference["xy"][:N_CORNERS] - gt, axis=1)[mask]
        keep = np.isfinite(c) & np.isfinite(b)
        cand_px += c[keep].tolist()
        base_px += b[keep].tolist()
        cand_nme += (c[keep] / item["diagonal"]).tolist()
        base_nme += (b[keep] / item["diagonal"]).tolist()

    def block(px, nme):
        if not px:
            return {"n_keypoints": 0}
        px, nme = np.asarray(px), np.asarray(nme)
        return {"n_keypoints": int(px.size),
                "nme_median": float(np.median(nme)),
                "nme_p90": float(np.percentile(nme, 90)),
                "px_median": float(np.median(px)),
                "px_p90": float(np.percentile(px, 90)),
                "gross20_rate": float(np.mean(px > GROSS_PX)),
                "catastrophic40_rate": float(np.mean(px > CATASTROPHIC_PX))}

    return {"candidate": block(cand_px, cand_nme), "R0": block(base_px, base_nme)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("A", "B", "C"), default="A")
    args = parser.parse_args()

    cache640 = json.loads(CACHE_640.read_text())
    residual = assert_cache_already_unflipped(cache640)
    context = build_context()
    views640 = views_from(cache640, "R0_640")
    baseline = views640["R0_640_orig"]

    if args.stage == "A":
        views, mode, minimum, tag = views640, "mean", 2, "FAST_A"
    elif args.stage == "B":
        cache960 = run_inference(context, R0, 960, CACHE_960)
        views = {**views640, **views_from(cache960, "R0_960")}
        mode, minimum, tag = "median", 3, "FAST_B"
    else:
        freeze = json.loads(FREEZE.read_text())
        name = "OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42"
        weights = REPO_ROOT / freeze["audited_candidates"][name]["path"]
        if not weights.exists():
            raise SystemExit(f"FAST_C_CHECKPOINT_MISSING: {weights}")
        cache_c = run_inference(context, weights, 640, CACHE_C)
        views = {**views640, **views_from(cache_c, "G38_640")}
        mode, minimum, tag = "median", 3, "FAST_C"

    merged, available = {}, {}
    for frame in context:
        points, mask = merge(views, frame, mode, minimum)
        if points is None:
            continue
        merged[frame], available[frame] = points, mask

    report = {
        "schema_version": "fast_teacher_v1",
        "tag": tag,
        "views": sorted(views),
        "merge": mode,
        "min_views_per_keypoint": minimum,
        "flip_cache_residual_px": residual,
        "new_training_runs": 0,
        "new_inference_passes": 0 if args.stage == "A" else 2 * len(context),
        "coverage": {
            "frames_with_candidate": len(merged),
            "frames_total": len(context),
        },
        "paired": {
            "ALL": paired_stats(merged, available, baseline, context),
            "daytime": paired_stats(merged, available, baseline, context, "daytime"),
            "nighttime": paired_stats(merged, available, baseline, context, "nighttime"),
        },
        "axis": {
            "candidate": axis_rate(merged, context),
            "R0": axis_rate({f: b["xy"] for f, b in baseline.items()}, context),
        },
    }

    all_block = report["paired"]["ALL"]
    night_block = report["paired"]["nighttime"]
    candidate, reference = all_block["candidate"], all_block["R0"]
    gates = {
        f"{tag}_1_all_nme": {
            "pass": bool(candidate.get("nme_median") is not None
                         and candidate["nme_median"] < reference["nme_median"]),
            "detail": f"R0 {reference.get('nme_median')} -> "
                      f"{tag} {candidate.get('nme_median')}"},
        f"{tag}_2_night_nme": {
            "pass": bool(night_block["candidate"].get("nme_median") is not None
                         and night_block["candidate"]["nme_median"]
                         <= night_block["R0"]["nme_median"]),
            "detail": f"Night R0 {night_block['R0'].get('nme_median')} -> "
                      f"{night_block['candidate'].get('nme_median')}"},
        f"{tag}_3_p90": {
            "pass": bool(candidate.get("nme_p90") is not None
                         and candidate["nme_p90"] < reference["nme_p90"]),
            "detail": f"p90 R0 {reference.get('nme_p90')} -> {candidate.get('nme_p90')}"},
        f"{tag}_4_gross20": {
            "pass": bool(candidate.get("gross20_rate") is not None
                         and candidate["gross20_rate"] < reference["gross20_rate"]),
            "detail": f"gross20 R0 {reference.get('gross20_rate')} -> "
                      f"{candidate.get('gross20_rate')}"},
    }
    if args.stage in ("B", "C"):
        gates[f"{tag}_5_catastrophic40"] = {
            "pass": bool(candidate.get("catastrophic40_rate") is not None
                         and candidate["catastrophic40_rate"]
                         <= reference["catastrophic40_rate"]),
            "detail": f"cat40 R0 {reference.get('catastrophic40_rate')} -> "
                      f"{candidate.get('catastrophic40_rate')}"}
    status = "PASS" if all(g["pass"] for g in gates.values()) else "FAIL"
    report["gates"] = gates
    report["status"] = status

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{tag}_TEACHER.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"{tag}  views {sorted(views)}  merge={mode}  min={minimum}")
    print(f"  flip cache 잔차 {residual:.2f} px (재mirror 안 함)")
    print(f"  coverage {report['coverage']['frames_with_candidate']}"
          f"/{report['coverage']['frames_total']} frames"
          f"   새 추론 {report['new_inference_passes']} 회")
    print(f"\n{'set':22} {'n_kp':>6} {'NME med':>9} {'NME p90':>9} {'px med':>8} "
          f"{'px p90':>8} {'gross20':>8} {'cat40':>7}")
    print("-" * 82)
    for domain in ("ALL", "daytime", "nighttime"):
        for who in ("R0", "candidate"):
            item = report["paired"][domain][who]
            if not item.get("n_keypoints"):
                continue
            label = f"{domain} / {'R0' if who == 'R0' else tag}"
            print(f"{label:22} {item['n_keypoints']:6d} {item['nme_median']:9.4f} "
                  f"{item['nme_p90']:9.4f} {item['px_median']:8.2f} "
                  f"{item['px_p90']:8.2f} {item['gross20_rate']:8.3f} "
                  f"{item['catastrophic40_rate']:7.3f}")
    axis = report["axis"]
    print(f"\naxis permutation   R0 {axis['R0']['rate']:.3f} "
          f"(q>=.75 {axis['R0']['ambiguous_rate']})   "
          f"{tag} {axis['candidate']['rate']:.3f} "
          f"(q>=.75 {axis['candidate']['ambiguous_rate']})")
    print(f"\n{tag} GATE {status}")
    for name, gate in gates.items():
        print(f"  {'PASS' if gate['pass'] else 'FAIL'}  {name}: {gate['detail']}")
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
