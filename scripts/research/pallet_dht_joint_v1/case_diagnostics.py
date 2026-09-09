"""Saved-output-only diagnosis of the user's preselected wrong-index case.

The reindexing below was already chosen using GT in the earlier case probe.
It is fixed here for every model/seed, never selected again or applied to model
outputs. This secondary diagnostic changes no canonical metric or verdict.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CASE = "eval_pallet07:1778652166837872128"
ARMS = ("point_only", "hough_features", "hough_joint")
SEEDS = (1, 2, 3)
PRIOR_REINDEX = (1, 5, 6, 2, 0, 4, 7, 3, 8)
POSE = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
PRIOR = ROOT / "data/pallet/results/hough_line_visual/case_recheck_20260908/RESULTS.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None:
            require(digest == expected, f"Changed saved input: {path}")
        self.hashes[str(path)] = digest
        return path

    def read(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text())


def summarize(points, gt, mask):
    if points is None:
        return dict(n_scored=0, median_px=None, mean_px=None, p90_px=None, max_px=None,
                    supervised_corner_median_px=None, centroid_error_px=None, per_point_error_px=[None] * 9)
    points = np.asarray(points, dtype=float)
    require(points.shape == (9, 2) and np.isfinite(points).all(), "Saved top1 must contain finite nine-point coordinates")
    error = np.linalg.norm(points - gt, axis=1)
    values = error[mask]
    corner_mask = mask.copy()
    corner_mask[8] = False
    return dict(n_scored=int(mask.sum()), median_px=float(np.median(values)), mean_px=float(np.mean(values)),
        p90_px=float(np.quantile(values, .9)), max_px=float(np.max(values)),
        supervised_corner_median_px=float(np.median(error[corner_mask])), centroid_error_px=float(error[8]) if mask[8] else None,
        per_point_error_px=[float(value) if valid else None for value, valid in zip(error, mask)])


def row_from_points(points, gt, mask):
    indexed = summarize(points, gt, mask)
    oracle = summarize(np.asarray(points)[list(PRIOR_REINDEX)] if points is not None else None, gt, mask)
    nearest = None
    if points is not None:
        ids = np.flatnonzero(mask[:8])
        distances = np.linalg.norm(np.asarray(points)[:8, None] - gt[None, ids], axis=-1)
        nearest = ids[np.argmin(distances, axis=1)].tolist()
    return dict(points_xy=points, indexed_top1_diagnostic=indexed,
        prior_GT_only_reindex=oracle, nearest_supervised_GT_corner_ids_diagnostic=nearest,
        nearest_corner_limit="Only seven supervised corner candidates; GT3 is omitted. This is not a bijective or deployed assignment.")


def csv_case(inputs, path):
    rows = [row for row in csv.DictReader(inputs.bind(path).open()) if row["frame_id"] == CASE]
    require(len(rows) == 1, "Exactly one canonical case CSV row is required")
    return rows[0]


def official_case(row, diagnostic):
    matched = row["top_iou50_match"] == "True"
    values = [float(x) for x in row["top_keypoint_supervised_errors_px"].split(";") if x]
    if matched and diagnostic["n_scored"]:
        calculated = [x for x in diagnostic["per_point_error_px"] if x is not None]
        require(len(values) == len(calculated) == 8, "The canonical case must score seven corners plus centroid")
        require(np.max(np.abs(np.asarray(values) - calculated)) <= 5.1e-7, "Case error vector differs beyond CSV rounding")
        require(abs(float(row["top_keypoint_supervised_error_median_px"]) - diagnostic["median_px"]) <= 1e-10,
                "Canonical full-precision case median differs")
        result = dict(diagnostic)
    else:
        require(not values, "Unmatched case unexpectedly has canonical point errors")
        result = dict(n_scored=0, median_px=None, mean_px=None, p90_px=None, max_px=None,
                      supervised_corner_median_px=None, centroid_error_px=None, per_point_error_px=[None] * 9)
    result.update(matched=matched, top_target_iou=float(row["top_target_iou"]) if row["top_target_iou"] else None,
                  n_gt_supervised=8, metric_scope="Canonical nine-point contract; this case has seven supervised corners plus centroid.")
    return result


def stats(values):
    return dict(values=values, mean=float(np.mean(values)) if all(x is not None for x in values) else None,
                sample_std_ddof1=float(np.std(values, ddof=1)) if all(x is not None for x in values) else None,
                meaning="Three trained seeds on the same single case; not three independent real images.")


def markdown(result):
    f = lambda x: "—" if x is None else f"{x:.3f}"
    lines = ["# 지정 사례의 점 번호·선 증거 진단", "",
        f"대상: `{CASE}`. 저장된 실제 결과만 읽은 단일 재사용 DEV 사례 분석입니다.", "",
        "공식 9점 계약 중 감독되는 좌표는 **0,1,2,4,5,6,7,8**입니다. Corner3은 화면 밖 visibility0으로 제외합니다. Corner5·6과 centroid8은 occluded이지만 감독되는 좌표입니다.", "",
        "GT를 보고 이전 실험에서 이미 고정한 순열 `[1,5,6,2,0,4,7,3,8]`을 모든 모델에 동일하게 대입했습니다. 이번 결과로 순열을 다시 고르지 않았고, 실제 출력·공식 지표·주 판정을 변경하지 않았습니다. 아래 oracle 값은 자동 보정 성능이 아닙니다.", "",
        "| 모델 | 공식 감독8점 median(px) | 공식 mean(px) | Max(px) | 7corner median(px) | Centroid error(px) | 고정 GT-only 순열 median(px) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in [result["historical_R0"], *result["runs"]]:
        m, o = row["canonical_case"], row["prior_GT_only_reindex"]
        lines.append('| ' + ' | '.join([row["model"], *[f(m.get(key)) for key in ("median_px", "mean_px", "max_px", "supervised_corner_median_px", "centroid_error_px")], f(o["median_px"])]) + ' |')
    lines += ["", "각 구조의 3 seed 평균 ± 표본 SD(ddof=1):", "",
        "| 구조 | 공식 median(px) | Centroid error(px) | 고정 GT-only 순열 median(px) |",
        "|---|---:|---:|---:|"]
    for arm, row in result["by_arm"].items():
        cells = [f(row[key]["mean"]) + ' ± ' + f(row[key]["sample_std_ddof1"]) for key in ("canonical_median_px", "centroid_error_px", "prior_GT_only_median_px")]
        lines.append('| ' + ' | '.join([arm, *cells]) + ' |')
    control = result["by_arm"]["point_only"]["canonical_median_px"]["mean"]
    joint = result["by_arm"]["hough_joint"]["canonical_median_px"]["mean"]
    lines += ["", f"이 사례의 공식 median을 seed별 계산한 뒤 평균하면 point_only {f(control)}px, hough_joint {f(joint)}px입니다. 이는 한 이미지의 결과이며 전체 평가 성능을 대신하지 않습니다."]
    mappings = [row["nearest_supervised_GT_corner_ids_diagnostic"] for row in result["runs"]]
    if all(mapping == [4, 0, 0, 7, 5, 1, 2, 6] for mapping in mappings):
        lines += ["", "9개 모델 모두 pred0이 GT4, pred1이 GT0 쪽에 놓이는 동일한 큰 번호 대응 오류가 남았습니다. 가장 가까운 감독 GT corner의 번호는 모두 `[4,0,0,7,5,1,2,6]`입니다. 이 최근접 진단은 미감독 GT3을 제외하며 일대일 대응이나 자동 보정 규칙이 아닙니다."]
    lines += ["", "점 좌표·번호별 오차·box score/IoU·입력 SHA와 실제 NPZ 정보는 [CASE_DIAGNOSIS.json](CASE_DIAGNOSIS.json)에 있습니다.", "",
        "영상 맨 왼쪽의 짧은 높이선은 GT4–GT7인 **role7 / rear_left_height**입니다. Role3 / front_left_height는 GT0–GT3이며 GT3이 미감독이므로 같은 가시 경계로 해석하지 않습니다.", "",
        "Hough arm의 저장 logits는 12×90×113, 이 사례의 역투표 map은 12×34×40입니다. `hough_features` 채널은 선 역할로 감독하지 않았고 `hough_joint`만 역할 손실을 사용했습니다. 배열 존재·유한성이나 sigmoid 크기는 올바른 edge 검출, calibrated confidence, attention 또는 인과성을 입증하지 않습니다.", "",
        "고정 순열을 대입했을 때 오차가 작아지는 현상은 번호 대응 문제를 설명하는 근거이며 localization 오차가 없다는 뜻은 아닙니다. 단일 사용자 선택 사례에서 전체 성능·일반화·필터 성능을 판정하지 않습니다. 전체 비교와 판정은 기존 SUMMARY/VERDICT 그대로입니다."]
    return '\n'.join(lines) + '\n'


def diagnose(run_dir):
    run_dir = Path(run_dir).resolve()
    require((run_dir / "PURPOSE.md").is_file(), "Purpose-declared result root required")
    # Check all nine completions before creating any diagnostic output.
    require(all((run_dir / "evaluation" / f"{arm}_seed{seed}" / "COMPLETION.json").is_file()
                for arm in ARMS for seed in SEEDS), "All nine actual evaluations must complete first")
    inputs = Inputs()
    population = inputs.read(ROOT / "challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json")
    item = next(row for row in population["items"] if row["frame_id"] == CASE)
    image = inputs.bind(ROOT / item["image_path"])
    annotation = inputs.read(ROOT / item["gt_v2_path"])["objects"][0]["keypoint_annotations"]
    gt = np.asarray([point["xy"] for point in annotation], dtype=float)
    mask = np.asarray([point["visibility"] > 0 for point in annotation])
    require(np.flatnonzero(mask).tolist() == [0, 1, 2, 4, 5, 6, 7, 8], "Canonical case GT mask changed")
    prior = inputs.read(PRIOR)
    require(prior["oracle_reindex"] == list(PRIOR_REINDEX), "Previously fixed GT-only reindex differs")
    require(np.array_equal(prior["gt_xy"], gt) and prior["gt_supervised"] == mask.tolist(), "Prior/canonical GT differs")
    axis = inputs.read(POSE / "AXIS_REVIEW_MANIFEST.json")
    frame = next(row for row in axis["frames_list"] if (ROOT / row["image"]).resolve() == image)
    r0 = inputs.read(POSE / "predictions/R0.json")["frames"][frame["frame_id"]]
    require(r0["keypoints_xy"] == prior["baseline"]["points_xy"], "Historical case/R0 predictions differ")
    baseline = dict(model="R0_historical", **row_from_points(r0["keypoints_xy"], gt, mask))
    baseline["canonical_case"] = official_case(csv_case(inputs, ROOT / "data/pallet/results/paper_eval_v1/arms/R0_per_frame.csv"), baseline["indexed_top1_diagnostic"])
    runs = []
    for arm in ARMS:
        for seed in SEEDS:
            label = f"{arm}_seed{seed}"
            directory = run_dir / "evaluation" / label
            completed = inputs.read(directory / "COMPLETION.json")
            require(completed.get("complete") and completed.get("PASS") and completed["arm"] == arm and completed["seed"] == seed
                    and completed["actual_positive_forwards"] == 319 and completed["actual_negative_forwards"] == 2689
                    and completed["baseline_candidate_copying"] is False, f"Actual evaluation incomplete: {label}")
            payload = inputs.read(directory / "PREDICTIONS.json", completed["output_sha256"]["PREDICTIONS.json"])
            csv_path = inputs.bind(directory / "PAPER_2D_per_frame.csv", completed["output_sha256"]["PAPER_2D_per_frame.csv"])
            key = next(key for key, metadata in payload["frame_metadata"].items() if metadata["kind"] == "positive" and metadata["frame_id"] == CASE)
            require(payload["frame_metadata"][key]["image_sha256"] == inputs.hashes[str(image)], "Actual case image differs")
            candidates = payload["frames"][key]
            top = max(candidates, key=lambda row: row["score"]) if candidates else {}
            row = dict(model=label, arm=arm, seed=seed, checkpoint_sha256=completed["checkpoint_sha256"],
                       n_candidates=len(candidates), box_score=top.get("score"), box_xyxy=top.get("box_xyxy"),
                       **row_from_points(top.get("keypoints_xy"), gt, mask))
            row["canonical_case"] = official_case(csv_case(inputs, csv_path), row["indexed_top1_diagnostic"])
            evidence = inputs.read(directory / "LINE_EVIDENCE.json", completed["output_sha256"]["LINE_EVIDENCE.json"])
            sample = next((sample for sample in evidence["samples"] if sample["frame_id"] == CASE), None)
            row["line_evidence"] = None
            if arm == "point_only":
                require(sample is None, "Point-only has unexpected line evidence")
            else:
                require(sample is not None, "Required case actual Hough evidence is missing")
                npz = inputs.bind(sample["path"], sample["sha256"])
                with np.load(npz, allow_pickle=False) as value:
                    require(value["logits"].shape == (12, 90, 113) and value["normalized_backprojection"].shape == (12, 34, 40), "Case Hough lattice differs")
                    require(all(np.isfinite(value[k]).all() for k in ("logits", "sigmoid_probability", "normalized_backprojection")), "Nonfinite saved line evidence")
                    row["line_evidence"] = dict(npz=str(npz), sha256=sample["sha256"], logits_shape=list(value["logits"].shape),
                        backprojection_shape=list(value["normalized_backprojection"].shape), finite=True,
                        sigmoid_min=float(value["sigmoid_probability"].min()), sigmoid_max=float(value["sigmoid_probability"].max()),
                        role_supervised=arm == "hough_joint", physical_image_left_height_role=7, endpoints=[4, 7])
            runs.append(row)
    by_arm = {}
    for arm in ARMS:
        rows = [row for row in runs if row["arm"] == arm]
        by_arm[arm] = dict(canonical_median_px=stats([row["canonical_case"]["median_px"] for row in rows]),
            centroid_error_px=stats([row["canonical_case"]["centroid_error_px"] for row in rows]),
            prior_GT_only_median_px=stats([row["prior_GT_only_reindex"]["median_px"] for row in rows]))
    for path, digest in inputs.hashes.items():
        require(sha(path) == digest, f"Input changed during diagnosis: {path}")
    result = dict(schema="pallet_dht_joint_case_diagnosis_v1", complete=True, PASS=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(), frame_id=CASE, image=str(image), image_sha256=inputs.hashes[str(image)],
        gt_xy=gt.tolist(), gt_supervised=mask.tolist(), gt_visibility=[point["visibility"] for point in annotation],
        n_gt_supervised_corners=7, n_gt_supervised_with_centroid=8, historical_R0=baseline, runs=runs, by_arm=by_arm,
        prior_GT_only_reindex=list(PRIOR_REINDEX), permutation_semantics="diagnostic[j] = saved_prediction[fixed_prior_reindex[j]]; GT mask remains canonical; centroid8 stays8",
        permutation_selected_again=False, permutation_applied_to_predictions=False,
        official_metrics_changed=False, primary_verdict_changed=False, model_forwards=0, GPU_used=False,
        scope="One preselected reused DEV case; complete/PASS means saved-input/arithmetic validity, not model improvement or automatic correction.",
        input_sha256=inputs.hashes, source_sha256={str(Path(__file__).resolve()): sha(__file__)})
    write(run_dir / "CASE_DIAGNOSIS.json", result)
    (run_dir / "CASE_DIAGNOSIS.md").write_text(markdown(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    output = diagnose(args.run_dir)
    print(json.dumps(dict(complete=True, n_runs=len(output["runs"]), by_arm=output["by_arm"], primary_verdict_changed=False), ensure_ascii=False, indent=2))
