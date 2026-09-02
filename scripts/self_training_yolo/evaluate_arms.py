"""R0 와 self-training arm 을 PAPER_EVAL 에서 같은 계약으로 평가하고 subgroup 을 집계한다.

evaluator 자체는 object subgroup(ALL/PLASTIC/WOOD)만 낸다.  M2/M5 가 요구하는
Daytime/Nighttime/Clean/Occlusion/Truncation/Far/Low/Mid/High 는 여기서 **한 번의
추론 결과**를 workspace tag 와 조인해 오프라인으로 집계한다.  subgroup 마다 추론을
다시 돌리지 않는다.

keypoint 통계는 evaluator 와 **같은 정의**를 쓴다.  per-frame CSV 가 감독 keypoint
오차를 원시값으로 싣고 있으므로, 프레임별 median 이 아니라 전 프레임의 keypoint 를
풀링해 median/p90 을 낸다 (프레임별 median 을 다시 median 하면 다른 통계가 된다).

box AP 는 evaluator 가 모든 candidate 로 계산하므로 object subgroup 에서만 가져온다.
frame-level ranking(AUROC / FPR95)은 top-1 score 로 정의되므로 subgroup 에서도
정확히 계산할 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluation.eval_workspace import (  # noqa: E402
    evaluation_population_views,
    load_frames,
)

WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
WORKSPACE_REL = "data/evaluation/pallet_eval_v1"
MANIFESTS = REPO_ROOT / "challenge/real_gt_v2/manifests"
EVALUATOR = REPO_ROOT / "challenge/evaluation_v2/paper_real_eval.py"
OUT_DIR = REPO_ROOT / "data/pallet/results/paper_eval_v1/arms"
RUNS = REPO_ROOT / "challenge/yolo_pose_one_model/paper_selftrain_v1"

GROSS_PX = 20.0  # metric_split_lock.md §2.2 [LOCKED]
# metric_split_lock.md §2.2 [LOCKED] — Proj@5px / @10px / @20px.
# 여기서 새 threshold 를 만들지 않고 동결된 값을 그대로 쓴다.
PROJ_PX = (5.0, 10.0, 20.0)

# 값은 checkpoint 경로이거나 (checkpoint, cached-predictions) 쌍이다.
# Ultralytics 가 아닌 baseline 은 미리 덤프한 예측으로 **같은 evaluator** 를 태운다.
BASELINE_DIR = REPO_ROOT / "data/pallet/results/paper_eval_v1/baselines"

MODELS = {
    "DOPE": (
        REPO_ROOT / "weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth",
        BASELINE_DIR / "DOPE_R0_PREDICTIONS.json",
    ),
    "R0": REPO_ROOT / (
        "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
        "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt"
    ),
    "R0_CONT": RUNS / "R0_CONT__FULL/weights/last.pt",
    "R1_NAIVE": RUNS / "R1_NAIVE__FULL/weights/last.pt",
    "R2_CONF": RUNS / "R2_CONF__FULL/weights/last.pt",
    "R3_CONF_REPROJ": RUNS / "R3_CONF_REPROJ__FULL/weights/last.pt",
    "R4_CONF_REMOVE": RUNS / "R4_CONF_REMOVE__FULL/weights/last.pt",
    "R5_PROPOSED": RUNS / "R5_PROPOSED__FULL/weights/last.pt",
}
# Repeatability (§29).  ultralytics `seed` 는 dataloader 에 도달하지 않아
# seed 42/43/44 의 가중치가 **비트 동일**했다 (max|Δw| = 0, 텐서 비교로 확인).
# 그래서 seed override run 은 독립 반복이 아니고 여기서 평가하지 않는다.
# 진짜 replicate 는 pseudo 샘플링을 바꾼 것이다 — 노출 총량은 그대로다.
for _seed in (43, 44):
    for _arm in ("R1_NAIVE", "R5_PROPOSED"):
        MODELS[f"{_arm}_P{_seed}"] = RUNS / f"{_arm}_P{_seed}__FULL/weights/last.pt"
# A2 — UNIQUE-QUANTITY-MATCHED control (§30).  Proposed 와 unique PL 개수를 맞춘
# Naive 무작위 표본.  MAIN 의 EXPOSURE-MATCHED 와 다른 실험이므로 섞지 않는다.
for _seed in (1, 2, 3):
    MODELS[f"A2_MATCHED_S{_seed}"] = (
        RUNS / f"A2_NAIVE_MATCHED_S{_seed}__FULL/weights/last.pt"
    )
# A12 — self-training strength sensitivity.  Proposed manifest 그대로, pseudo:synthetic
# 비율만 바꾼다.  총 optimizer update 는 동일하다.
# 두 제안 필터의 2x2 ablation.  각 칸을 replicate 3 회로 채운다.
#   neither = R2_CONF · removal only = R4 · flip only = R6 · both = R5
MODELS["R6_CONF_FLIP"] = RUNS / "R6_CONF_FLIP__FULL/weights/last.pt"
for _seed in (43, 44):
    for _arm in ("R2_CONF", "R3_CONF_REPROJ", "R4_CONF_REMOVE", "R6_CONF_FLIP"):
        MODELS[f"{_arm}_P{_seed}"] = RUNS / f"{_arm}_P{_seed}__FULL/weights/last.pt"
# A8 cross-domain transfer.  한 도메인 pool 로만 적응한 모델.
# §18-B geometry incremental control.  confidence 를 이미 통과한 pool 에서 뽑는다.
for _seed in (1, 2, 3):
    MODELS[f"B_CONF_RANDOM_S{_seed}"] = (
        RUNS / f"B_CONF_RANDOM_S{_seed}__FULL/weights/last.pt")
MODELS["B_CONF_TOPN"] = RUNS / "B_CONF_TOPN__FULL/weights/last.pt"
MODELS["B_CONF_DECILE"] = RUNS / "B_CONF_DECILE__FULL/weights/last.pt"
MODELS["A8_DAY_ONLY"] = RUNS / "A8_DAY_ONLY__FULL/weights/last.pt"
MODELS["A8_NIGHT_ONLY"] = RUNS / "A8_NIGHT_ONLY__FULL/weights/last.pt"
MODELS["A12_PSEUDO25"] = RUNS / "R5_PROPOSED_W25__FULL/weights/last.pt"
MODELS["A12_PSEUDO75"] = RUNS / "R5_PROPOSED_S75__FULL/weights/last.pt"
# Real-FT supervised upper bound.  controlled comparison 이 아니다 — 다른 base 에서
# real GT 로 학습했다.  PAPER_EVAL 과의 중복은 SHA·파일명 stem 둘 다 0 으로 확인.
_YOLO_ROOT = REPO_ROOT / "challenge/yolo_pose_one_model"
MODELS["REALFT_A"] = _YOLO_ROOT / "runs_ft/ft_a_real157_neg259_synth12k/weights/best.pt"
MODELS["REALFT_B"] = _YOLO_ROOT / "runs_ft/ft_b_patience0_ep40/weights/best.pt"
MODELS["REALFT_LV1V2"] = (
    _YOLO_ROOT / "legacy_v1v2_ft/runs/LV1V2_FT_15EP_SEED42/weights/best.pt"
)

SUBGROUPS = {
    "ALL": lambda row: True,
    "Plastic": lambda row: row["object_type"] == "plastic",
    "Wood": lambda row: row["object_type"] == "wood",
    # MAIN 의 Daytime/Nighttime 은 lighting 이 아니라 paper_domain 이다.
    # lighting 은 모든 object·모든 acquisition 을 섞어 168/106 이 되지만, MAIN 은
    # morphology 를 lighting 효과와 섞지 않으려고 plastic 만 센다 (70/50).
    # reports/PAPER_DOMAIN_COVERAGE.md 가 원천이고 §33 불변식이 이 수를 검사한다.
    "Daytime": lambda row: row["paper_domain"] == "daytime",
    "Nighttime": lambda row: row["paper_domain"] == "nighttime",
    # M5 의 넓은 조명 분할.  MAIN 조건과 이름이 겹치지 않게 따로 둔다.
    "Lighting_day": lambda row: row["lighting"] == "day",
    "Lighting_night": lambda row: row["lighting"] == "night",
    "Clean": lambda row: row["occlusion"] == "none",
    "Occlusion": lambda row: row["occlusion"] == "medium",
    "Truncation": lambda row: row["truncation"] == "mild",
    "Far": lambda row: row["distance_bin"] == "far",
    "Low": lambda row: row["elevation_bin"] == "low",
    "Mid": lambda row: row["elevation_bin"] == "mid",
    "High": lambda row: row["elevation_bin"] == "high",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_evaluator(name: str, weights: Path,
                  predictions: Path | None = None) -> tuple[Path, Path]:
    report = OUT_DIR / f"{name}.json"
    per_frame = OUT_DIR / f"{name}_per_frame.csv"
    if report.exists() and per_frame.exists():
        print(f"  {name}: 이미 있음 — 건너뜀", flush=True)
        return report, per_frame
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(EVALUATOR),
        "--positive-manifest", str(MANIFESTS / "PAPER_EVAL_ALL_POS.json"),
        "--negative-manifest", str(MANIFESTS / "DEV_NEG2689.json"),
        "--population-role", "DEV",
        "--weights", str(weights),
        "--migration-gate", str(REPO_ROOT / "challenge/real_gt_v2/MIGRATION_GATE.json"),
        "--symmetry-contract", str(REPO_ROOT / "challenge/real_gt_v2/SYMMETRY_CONTRACT.json"),
        "--object-migration-gate",
        f"wood_small_80x59x14={REPO_ROOT}/challenge/real_gt_v2/wood_audit/migration/MIGRATION_GATE.json",
        "--out", str(report),
        "--per-frame-out", str(per_frame),
        "--report-out", str(OUT_DIR / f"{name}_report.md"),
        "--device", "0",
    ]
    if predictions is not None:
        command += ["--predictions", str(predictions)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)
    if not report.exists() or not per_frame.exists():
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit(f"EVALUATION_PRODUCED_NO_ARTIFACT: {name}")
    return report, per_frame


def workspace_metadata() -> dict[str, dict]:
    rows = evaluation_population_views(load_frames(WORKSPACE))["PAPER_EVAL_POSITIVE"]
    return {f"{WORKSPACE_REL}/{row['image_path']}": row for row in rows}


def ranking(positive_scores: np.ndarray, negative_scores: np.ndarray) -> dict:
    """frame-level AUROC 와 FPR@95TPR.  top-1 score 로만 정의된다."""

    if positive_scores.size == 0 or negative_scores.size == 0:
        return {"auroc": None, "fpr95": None}
    labels = np.concatenate([np.ones(positive_scores.size), np.zeros(negative_scores.size)])
    scores = np.concatenate([positive_scores, negative_scores])
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = tps / positive_scores.size
    fpr = fps / negative_scores.size
    auroc = float(np.trapz(tpr, fpr))
    index = int(np.searchsorted(tpr, 0.95, side="left"))
    index = min(index, fpr.size - 1)
    return {"auroc": auroc, "fpr95": float(fpr[index])}


def subgroup_table(per_frame: Path, metadata: dict[str, dict]) -> dict:
    rows = list(csv.DictReader(per_frame.open(encoding="utf-8")))
    positives = [row for row in rows if row["kind"] == "POSITIVE"]
    negatives = [row for row in rows if row["kind"] == "NEGATIVE"]
    negative_scores = np.array(
        [float(row["top_score"]) if row["top_score"] else 0.0 for row in negatives]
    )

    table: dict[str, dict] = {}
    for name, predicate in SUBGROUPS.items():
        selected = []
        for row in positives:
            meta = metadata.get(row["image"])
            if meta is None:
                raise SystemExit(f"UNJOINED_FRAME: {row['image']}")
            if predicate(meta):
                selected.append(row)
        if not selected:
            table[name] = {"N": 0}
            continue

        detected = sum(row["top_iou50_match"] == "True" for row in selected)
        errors: list[float] = []
        annotated: list[float] = []
        for row in selected:
            raw = row.get("top_keypoint_supervised_errors_px") or ""
            if raw:
                errors += [float(value) for value in raw.split(";")]
            raw_all = row.get("top_keypoint_all_annotated_errors_px") or ""
            if raw_all:
                annotated += [float(value) for value in raw_all.split(";")]
        scores = np.array(
            [float(row["top_score"]) if row["top_score"] else 0.0 for row in selected]
        )
        stats = ranking(scores, negative_scores)
        table[name] = {
            "N": len(selected),
            "detection_rate_iou50": detected / len(selected),
            "n_keypoints": len(errors),
            "corner_median_px": float(np.median(errors)) if errors else None,
            "corner_p90_px": float(np.percentile(errors, 90)) if errors else None,
            "gross_rate": float(np.mean(np.array(errors) > GROSS_PX)) if errors else None,
            **{
                f"proj_at_{int(t)}px": (
                    float(np.mean(np.array(errors) <= t)) if errors else None
                )
                for t in PROJ_PX
            },
            # 2026-09-02 이전에는 legacy 프레임의 visibility 가 전부 unknown 이라
            # supervision mask 가 비고 위 값이 None 이 됐다.  visibility 확정 이후
            # 319 장 전부 strict 를 낸다.  아래 all-annotated 는 visibility 를 무시한
            # 진단값으로 남긴다 — visible/occluded 주장이 아니다.
            "n_keypoints_annotated": len(annotated),
            "corner_median_px_all_annotated": (
                float(np.median(annotated)) if annotated else None
            ),
            "corner_p90_px_all_annotated": (
                float(np.percentile(annotated, 90)) if annotated else None
            ),
            "gross_rate_all_annotated": (
                float(np.mean(np.array(annotated) > GROSS_PX)) if annotated else None
            ),
            **{
                f"proj_at_{int(t)}px_all_annotated": (
                    float(np.mean(np.array(annotated) <= t)) if annotated else None
                )
                for t in PROJ_PX
            },
            "auroc": stats["auroc"],
            "fpr95": stats["fpr95"],
        }
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    args = parser.parse_args()

    metadata = workspace_metadata()
    results: dict[str, dict] = {}
    for name in args.models:
        entry = MODELS[name]
        weights, predictions = entry if isinstance(entry, tuple) else (entry, None)
        if not weights.exists():
            print(f"  {name}: checkpoint 없음 — 건너뜀 ({weights})", flush=True)
            continue
        if predictions is not None and not predictions.exists():
            print(f"  {name}: predictions 없음 — 건너뜀 ({predictions})", flush=True)
            continue
        print(f"evaluating {name}", flush=True)
        report, per_frame = run_evaluator(name, weights, predictions)
        payload = json.loads(report.read_text())
        metrics = payload["metrics"]["box_and_keypoint_2d"]
        results[name] = {
            "checkpoint": str(weights.relative_to(REPO_ROOT)),
            "checkpoint_sha256": sha256_file(weights),
            "scored_from": (
                "OFFLINE_CACHED_PREDICTIONS" if predictions is not None
                else "LIVE_ULTRALYTICS"
            ),
            "box_ap50_95": metrics["box_ap50_95"],
            "box_ap50": metrics["box_ap50"],
            "object_subgroup_ap": {
                key: {"box_ap50_95": value["box_ap50_95"], "box_ap50": value["box_ap50"]}
                for key, value in metrics["subgroups"].items()
            },
            "pose_status": payload["metrics"]["pose"]["status"],
            "subgroups": subgroup_table(per_frame, metadata),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ARM_RESULTS.json").write_text(
        json.dumps({
            "schema_version": "paper_arm_results_v1",
            "population": "PAPER_EVAL_ALL_POS + DEV_NEG2689",
            "gross_px": GROSS_PX,
            "keypoint_definition": (
                "supervised keypoints pooled across frames, identical to the "
                "evaluator's keypoint_location_* definition"
            ),
            "models": results,
        }, indent=2, ensure_ascii=False) + "\n"
    )

    def show(value, spec=".3f"):
        return "—" if value is None else format(value, spec)

    print(f"\n{'model':16} {'AP50-95':>8} {'corner~':>8} {'det':>6} "
          f"{'Day~*':>8} {'Night~*':>8} {'AUROC':>7} {'FPR95':>7}")
    print("─" * 78)
    for name, value in results.items():
        allg = value["subgroups"]["ALL"]
        day = value["subgroups"]["Daytime"]
        night = value["subgroups"]["Nighttime"]
        print(f"{name:16} {show(value['box_ap50_95'], '.4f'):>8} "
              f"{show(allg.get('corner_median_px')):>8} "
              f"{show(allg.get('detection_rate_iou50')):>6} "
              f"{show(day.get('corner_median_px')):>8} "
              f"{show(night.get('corner_median_px')):>8} "
              f"{show(allg.get('auroc'), '.4f'):>7} "
              f"{show(allg.get('fpr95'), '.4f'):>7}")
    print("* Day/Night 는 MAIN paper_domain (plastic 70 / 50) strict 값이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
