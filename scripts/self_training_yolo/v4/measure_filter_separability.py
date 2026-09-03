"""필터 신호가 나쁜 pseudo-label 을 **얼마나** 가려낼 수 있는지 잰다.

지금까지는 신호를 임계로만 썼다 (`s_* <= 0.05`).  그래서 "안 걸린다" 는 결론이
**방식의 한계**인지 **신호의 한계**인지 구분되지 않았다.  여기서는 임계를 떼고
AUC 로 분리력 자체를 잰다.

두 층위를 따로 본다.

    frame level     프레임이 gross 코너를 하나라도 갖는가
    keypoint level  이 코너가 gross 인가

gross 는 `metric_split_lock` §2.2 의 20 px 다.

**중요**: 여기서 나온 수치는 PAPER_EVAL 에서 나왔다.  이 값으로 결합 가중치를
고정하면 개발셋을 또 소비하는 것이다 — 측정은 측정으로만 쓰고, 학습된 필터를
쓰려면 별도 데이터가 필요하다.

실행: `pallet-pose` 환경에서, `paper_selftrain_v4` 결과 폴더를 CWD 로 두고 돌린다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
for relative in ("scripts/self_training_yolo/v2", "scripts/self_training_yolo",
                 "scripts/evaluation"):
    sys.path.insert(0, str(REPO_ROOT / relative))

from eval_workspace import load_frames, evaluation_population_views  # noqa: E402
from keypoint_scores import per_keypoint_scores  # noqa: E402
from pseudo_label_filters import geometry_scores, projected_diagonal  # noqa: E402

WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
V4 = REPO_ROOT / "data/pallet/results/paper_selftrain_v4"
TEACHER_CACHE = V4 / "V4_PROXY_TEACHER_CACHE.json"
M4_RECORDS = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/M4_FRAME_RECORDS.json"
REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
OUT_JSON = V4 / "FILTER_SEPARABILITY.json"
OUT_MD = REPO_ROOT / "_docs/archive/paper_pre_final_20260903/diagnostics/FILTER_SEPARABILITY.md"

GROSS_PX = 20.0
BOX_CONF = 0.85
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


def cross_validated(features: np.ndarray, labels: np.ndarray, groups) -> tuple:
    """GroupKFold — 같은 프레임의 코너가 train/test 에 갈라지지 않게 한다."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out_of_fold = np.zeros(len(labels))
    for train, test in GroupKFold(5).split(features, labels, groups=groups):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=3000)).fit(
            features[train], labels[train])
        out_of_fold[test] = model.predict_proba(features[test])[:, 1]
    return auc(out_of_fold, labels), out_of_fold


def retention_curve(score: np.ndarray, labels: np.ndarray) -> list[dict]:
    order = np.argsort(score)
    curve = []
    for keep in (0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
        count = int(len(labels) * keep)
        selected = order[:count]
        curve.append({
            "keep_fraction": keep,
            "kept": count,
            "gross_rate": float(labels[selected].mean()),
            "gross_removed": int(labels.sum() - labels[selected].sum()),
            "gross_total": int(labels.sum()),
            "clean_lost": int((1 - labels).sum() - (1 - labels[selected]).sum()),
            "clean_total": int((1 - labels).sum()),
        })
    return curve


def frame_level() -> dict:
    records = [r for r in json.loads(M4_RECORDS.read_text())["frames"]
               if r["detected"] and r.get("errors_px")
               and all(r[k] is not None for k in
                       ("s_reproj", "s_remove", "s_flip", "box_conf"))]
    labels = np.array([1 if (r.get("gross_keypoints") or 0) > 0 else 0
                       for r in records])
    single = {
        "s_reproj": auc([r["s_reproj"] for r in records], labels, True),
        "s_remove": auc([r["s_remove"] for r in records], labels, True),
        "s_flip": auc([r["s_flip"] for r in records], labels, True),
        "box_conf": auc([r["box_conf"] for r in records], labels, False),
        "valid_corners": auc([r["valid_corners"] for r in records], labels, False),
    }
    features = np.array([[r["s_reproj"], r["s_remove"], r["s_flip"], -r["box_conf"]]
                         for r in records])
    groups = np.arange(len(records))       # 프레임이 곧 표본이다
    combined, out_of_fold = cross_validated(features, labels, groups)
    proposed = np.array([bool(r["verdict"].get("F4_PROPOSED")) for r in records])
    return {
        "population": "PAPER_EVAL_PLASTIC_POS (M4 records)",
        "n": len(records),
        "gross_rate": float(labels.mean()),
        "single_signal_auc": single,
        "combined_auc_cv": combined,
        "retention_curve": retention_curve(out_of_fold, labels),
        "current_proposed_filter": {
            "kept": int(proposed.sum()),
            "keep_fraction": float(proposed.mean()),
            "gross_rate": float(labels[proposed].mean()),
        },
    }


def keypoint_level() -> dict:
    cache = json.loads(TEACHER_CACHE.read_text())
    registry = {entry["object_type"]: entry["physical_dimensions_m"]
                for entry in json.loads(REGISTRY.read_text())["objects"]}
    rows: list[dict] = []
    for row in evaluation_population_views(load_frames(WORKSPACE))["PAPER_EVAL_POSITIVE"]:
        frame = row["frame_id"].replace("__", ":")
        entry = cache.get(frame)
        if not entry or not entry.get("top1"):
            continue
        top = entry["top1"]
        if float(top["box_conf"]) < BOX_CONF:
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
        dimensions = {axis: float(registry[REGISTRY_NAME[row["object_type"]]][axis])
                      for axis in ("x", "y", "z")}
        intrinsics = payload["camera_data"]["intrinsics"]
        camera = np.array([[intrinsics["fx"], 0.0, intrinsics["cx"]],
                           [0.0, intrinsics["fy"], intrinsics["cy"]],
                           [0.0, 0.0, 1.0]], dtype=float)
        keypoints = np.asarray(top["keypoints_xy"], dtype=float)
        confidence = np.asarray(top["keypoints_conf"], dtype=float)
        flip = entry.get("flip_top1") or {}
        flip_xy = np.asarray(flip["keypoints_xy"], dtype=float) if flip else None
        flip_conf = np.asarray(flip["keypoints_conf"], dtype=float) if flip else None
        corner = per_keypoint_scores(keypoints, confidence, camera, dimensions,
                                     flip_keypoints_2d=flip_xy, flip_conf=flip_conf)
        frame_scores = geometry_scores(keypoints, confidence >= 0.5, camera, dimensions,
                                       flip_xy,
                                       None if flip_conf is None else flip_conf >= 0.5)
        errors = np.linalg.norm(keypoints - gt, axis=1)

        def cap(value):
            return min(float(value), 1.0) if np.isfinite(value) else 1.0

        for index in range(8):
            if not supervised[index] or not np.isfinite(errors[index]):
                continue
            rows.append({
                "frame": frame,
                "label": int(errors[index] > GROSS_PX),
                "r_remove": cap(corner["r_remove"][index]),
                "r_flip": cap(corner["r_flip"][index]),
                "kp_conf": float(confidence[index]),
                "f_reproj": cap(frame_scores["s_reproj"]),
                "f_remove": cap(frame_scores["s_remove"]),
                "f_flip": cap(frame_scores["s_flip"]
                              if frame_scores["s_flip"] is not None else 1.0),
                "box_conf": float(top["box_conf"]),
                "q": float(corner["q"] or 0.0),
                "nme": float(errors[index] / diagonal),
            })

    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["frame"] for r in rows])
    single = {
        "r_remove": auc([r["r_remove"] for r in rows], labels, True),
        "r_flip": auc([r["r_flip"] for r in rows], labels, True),
        "kp_conf": auc([r["kp_conf"] for r in rows], labels, False),
    }
    blocks = {
        "corner_only": ["r_remove", "r_flip", "kp_conf"],
        "frame_only": ["f_reproj", "f_remove", "f_flip", "box_conf", "q"],
        "corner_plus_frame": ["r_remove", "r_flip", "kp_conf", "f_reproj", "f_remove",
                              "f_flip", "box_conf", "q"],
    }
    combined = {}
    curve = None
    for name, keys in blocks.items():
        features = np.array([[r[k] for k in keys] for r in rows])
        value, out_of_fold = cross_validated(features, labels, groups)
        combined[name] = value
        if name == "corner_plus_frame":
            curve = retention_curve(out_of_fold, labels)

    current = np.array([(r["r_remove"] <= 0.05 and r["r_flip"] <= 0.05
                         and r["kp_conf"] >= 0.5) for r in rows])
    below_floor = int(np.sum([r["kp_conf"] < 0.5 for r in rows]))
    return {
        "population": "PAPER_EVAL_POSITIVE, box_conf >= 0.85, supervised corners",
        "n": len(rows),
        "gross_rate": float(labels.mean()),
        "single_signal_auc": single,
        "combined_auc_cv": combined,
        "retention_curve": curve,
        "current_v3_rule": {
            "kept": int(current.sum()),
            "keep_fraction": float(current.mean()),
            "gross_rate": float(labels[current].mean()),
        },
        "keypoints_below_conf_floor": below_floor,
        "keypoint_conf_floor_note": (
            "kp_conf >= 0.5 removes nothing at the corner level; every supervised "
            "corner clears it."),
    }


def number(value, spec=".3f") -> str:
    return "—" if value is None else format(value, spec)


def render(report: dict) -> None:
    frame, keypoint = report["frame_level"], report["keypoint_level"]
    lines = [
        "# 필터 신호는 나쁜 pseudo-label 을 얼마나 가려내는가",
        "",
        "임계(`s_* <= 0.05`)를 떼고 **분리력 자체**를 AUC 로 쟀다.  그래야 '안 걸린다' 가",
        "방식의 한계인지 신호의 한계인지 갈린다.",
        "",
        "`gross` 는 `metric_split_lock` §2.2 의 20 px 다.",
        "",
        "> **이 수치로 결합 가중치를 고정하면 PAPER_EVAL 을 또 소비하는 것이다.**",
        "> 측정은 측정으로만 쓴다.  학습된 필터를 실제로 쓰려면 별도 데이터가 필요하다.",
        "",
        "## 프레임 단위 — 이 프레임이 gross 코너를 갖는가",
        "",
        f"모집단 {frame['population']}, n={frame['n']}, "
        f"gross 보유 {frame['gross_rate']:.1%}",
        "",
        "```text",
        f"{'signal':16} {'AUC':>7}",
        "-" * 24,
    ]
    for name, value in frame["single_signal_auc"].items():
        lines.append(f"{name:16} {number(value):>7}")
    lines += [f"{'4개 조합 (CV)':16} {number(frame['combined_auc_cv']):>7}", "```", "",
              "```text",
              f"{'유지율':>7} {'남은 n':>7} {'gross':>8} {'제거 gross':>12} {'버린 clean':>12}",
              "-" * 50]
    for point in frame["retention_curve"]:
        lines.append(f"{point['keep_fraction']:6.0%} {point['kept']:7d} "
                     f"{point['gross_rate']:8.1%} "
                     f"{point['gross_removed']:5d}/{point['gross_total']:<6d} "
                     f"{point['clean_lost']:5d}/{point['clean_total']:<6d}")
    current = frame["current_proposed_filter"]
    lines += ["```", "",
              f"현재 Proposed 필터: {current['kept']} 유지 "
              f"({current['keep_fraction']:.0%}), gross {current['gross_rate']:.1%}",
              "", "## Keypoint 단위 — 이 코너가 gross 인가", "",
              f"모집단 {keypoint['population']}, n={keypoint['n']}, "
              f"gross {keypoint['gross_rate']:.1%}", "", "```text",
              f"{'signal':22} {'AUC':>7}", "-" * 30]
    for name, value in keypoint["single_signal_auc"].items():
        lines.append(f"{name:22} {number(value):>7}")
    for name, value in keypoint["combined_auc_cv"].items():
        lines.append(f"{name:22} {number(value):>7}")
    lines += ["```", "",
              f"`kp_conf < 0.5` 인 supervised 코너: "
              f"**{keypoint['keypoints_below_conf_floor']} 개**.  "
              "코너 단위에서 confidence floor 는 아무것도 거르지 않는다.",
              "", "```text",
              f"{'유지율':>7} {'남은 n':>7} {'gross':>8} {'제거 gross':>12} {'버린 clean':>12}",
              "-" * 50]
    for point in keypoint["retention_curve"]:
        lines.append(f"{point['keep_fraction']:6.0%} {point['kept']:7d} "
                     f"{point['gross_rate']:8.1%} "
                     f"{point['gross_removed']:5d}/{point['gross_total']:<6d} "
                     f"{point['clean_lost']:5d}/{point['clean_total']:<6d}")
    rule = keypoint["current_v3_rule"]
    lines += ["```", "",
              f"현재 V3 규칙: {rule['kept']} 유지 ({rule['keep_fraction']:.0%}), "
              f"gross {rule['gross_rate']:.1%}", "",
              "## 읽는 법", "",
              "- 프레임 단위 분리력이 코너 단위보다 **높다**.  코너 하나가 틀리는 것은",
              "  대부분 프레임 전체가 틀린 것의 일부이기 때문으로 보인다.",
              "- 현재의 임계 AND 방식은 같은 신호를 결합했을 때의 곡선 위에 거의 그대로",
              "  얹혀 있다 — 방식이 정보를 버리고 있었다기보다, **신호 자체가 약하다**.",
              "- gross 를 0 으로 만들 수는 없다.  절반으로 줄이려면 30~40% 를 버려야 하고",
              "  그중 상당수는 멀쩡한 라벨이다.",
              ""]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    report = {
        "schema_version": "filter_separability_v1",
        "gross_px": GROSS_PX,
        "warning": ("Measured on PAPER_EVAL. Fixing combination weights from these "
                    "numbers would consume the development population again."),
        "frame_level": frame_level(),
        "keypoint_level": keypoint_level(),
    }
    V4.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    render(report)
    print(f"frame level   n={report['frame_level']['n']}  "
          f"combined AUC {number(report['frame_level']['combined_auc_cv'])}")
    for name, value in report["keypoint_level"]["combined_auc_cv"].items():
        print(f"keypoint {name:20} AUC {number(value)}")
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
