"""Saved full-training results and native offline canvas gallery; no inference."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
HELPERS = ROOT / "scripts/research/pallet_line_pose_v1/report.py"
spec = importlib.util.spec_from_file_location("offline_report_helpers", HELPERS)
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
from line_targets import EDGES, ROLE_NAMES
ARMS = ("point_only", "hough_features", "hough_joint")
LABELS = dict(point_only="점 전용 · 동일 예산", hough_features="허프 특징 · 점 손실", hough_joint="허프 특징 + 점·선 공동학습", R0="YOLO26n R0 · 과거 기준")
TITLE = "Pallet DHT Joint · 점과 선 공동 학습"
METRICS = H.METRICS
sha, table, number = H.sha, H.table, H.number
POSE = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
REQUIRED_CASE = "eval_pallet07:1778652166837872128"


def read_csv(inputs, path):
    inputs.bind(path)
    return {row["frame_id"]: row for row in csv.DictReader(Path(path).open()) if row["kind"] == "POSITIVE"}


def frame_error(row):
    if not row or row.get("top_iou50_match") != "True":
        return None
    values = [float(v) for v in row["top_keypoint_supervised_errors_px"].split(";") if v]
    return float(np.mean(values)) if values else None


def quantize_maps(array):
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite saved line evidence")
    lo, hi = array.min(axis=(-2, -1)), array.max(axis=(-2, -1))
    norm = (array - lo[:, None, None]) / np.maximum(hi - lo, 1e-12)[:, None, None]
    return dict(shape=list(array.shape), bytes=base64.b64encode(np.rint(norm * 255).astype(np.uint8).tobytes()).decode(),
                minimum=lo.tolist(), maximum=hi.tolist(), display="Per-role linear min/max display only; exact arrays remain in bound NPZ.")


def evidence_data(inputs, directory):
    payload = inputs.read(directory / "LINE_EVIDENCE.json", {})
    result = {}
    for row in payload.get("samples", []):
        path = Path(row["path"])
        inputs.bind(path, row["sha256"])
        with np.load(path, allow_pickle=False) as a:
            result[row["image_key"]] = dict(spatial=quantize_maps(a["normalized_backprojection"]),
                hough=quantize_maps(a["sigmoid_probability"]), affine=a["raw_to_input_affine"].tolist(),
                input_shape_hw=a["input_shape_hw"].tolist(), theta_degrees=np.rad2deg(a["theta_radians"]).tolist(),
                rho_values=a["rho_values"].tolist(), npz=str(path), meaning=row["meaning"])
    return result


def collect(inputs):
    run_dir = inputs.root
    summary = inputs.read(run_dir / "SUMMARY.json", {})
    verdict = inputs.read(run_dir / "VERDICT.json", {})
    protocol = inputs.read(run_dir / "TRAIN_PROTOCOL.json", {})
    runtime = inputs.read(run_dir / "RUNTIME.json", {})
    progress, runs, prediction_files = [], [], []
    for arm in ARMS:
        for seed in (1, 2, 3):
            label = f"{arm}_seed{seed}"
            trained = inputs.read(run_dir / "runs" / label / "COMPLETION.json", {})
            failure = inputs.read(run_dir / "runs" / label / "FAILURE.json", {})
            history = inputs.read(run_dir / "runs" / label / "history.json", [])
            result = inputs.read(run_dir / "evaluation" / label / "RESULTS.json", {})
            complete = trained.get("complete") and trained.get("PASS") and trained.get("stage") == "main"
            state = "EVALUATED" if complete and result.get("complete") else "TRAINED" if complete else "FAILED" if failure else "TRAINING" if history else "NOT_RUN"
            progress.append(dict(arm=arm, seed=seed, state=state, epochs_completed=trained.get("epochs_completed"), optimizer_steps=trained.get("optimizer_steps")))
            if state == "EVALUATED":
                runs.append(result)
                prediction_files.append(run_dir / "evaluation" / label / "PREDICTIONS.json")
    frames = []
    if prediction_files:
        manifest = inputs.read(ROOT / "challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json")
        if manifest["role"] != "DEV" or manifest["expected_count"] != 319:
            raise ValueError("Only frozen positive DEV319 gallery is permitted")
        r0_rows = read_csv(inputs, ROOT / "data/pallet/results/paper_eval_v1/arms/R0_per_frame.csv")
        r0_preds = inputs.read(POSE / "predictions/R0.json")["frames"]
        r0_by_image = {str((ROOT / row["image"]).resolve()): r0_preds[row["frame_id"]] for row in inputs.read(POSE / "AXIS_REVIEW_MANIFEST.json")["frames_list"]}
        by_key = {}
        for item in manifest["items"]:
            image_path = (ROOT / item["image_path"]).resolve()
            gt_path = (ROOT / item["gt_v2_path"]).resolve()
            annotation = inputs.read(gt_path)["objects"][0]["keypoint_annotations"]
            gt = [point["xy"] if point["visibility"] > 0 else None for point in annotation]
            if len(gt) != 9:
                raise ValueError("Expected nine canonical GT entries")
            inputs.bind(image_path)
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot decode positive gallery image {image_path}")
            before = r0_by_image[str(image_path)]
            r0_error = frame_error(r0_rows[item["frame_id"]])
            frame = dict(id=item["frame_id"], session=item["session_id"], domain=item.get("domain", "UNKNOWN"), object_type=item.get("object_type"),
                width=image.shape[1], height=image.shape[0], image=H.data_uri(image), gt=gt,
                r0=before.get("keypoints_xy"), r0_error=r0_error,
                difficulty="missing" if r0_error is None else "easy_le10" if r0_error <= 10 else "moderate_10_20" if r0_error <= 20 else "hard_gt20", runs={})
            by_key[str(image_path)] = frame
            frames.append(frame)
        for path in prediction_files:
            payload = inputs.read(path)
            if not payload.get("complete") or payload.get("actual_positive_forwards") != 319 or payload.get("actual_negative_forwards") != 2689:
                raise ValueError("Report requires actual complete full-model inference")
            directory = path.parent
            label = payload["model"]
            rows = read_csv(inputs, directory / "PAPER_2D_per_frame.csv")
            evidence = evidence_data(inputs, directory)
            for image_key, metadata in payload["frame_metadata"].items():
                if metadata["kind"] != "positive":
                    continue
                image_path = (ROOT / image_key).resolve()
                if str(image_path) not in by_key:
                    raise ValueError("Non-DEV319 image in gallery")
                inputs.bind(image_path, metadata["image_sha256"])
                frame = by_key[str(image_path)]
                candidates = payload["frames"][image_key]
                top = max(candidates, key=lambda row: row["score"]) if candidates else {}
                frame["runs"][label] = dict(points=top.get("keypoints_xy"), box=top.get("box_xyxy"), score=top.get("score"),
                    n_candidates=len(candidates), error=frame_error(rows[frame["id"]]), evidence=evidence.get(image_key))
    complete = bool(summary.get("complete") and summary.get("PASS") and verdict.get("complete") and len(runs) == 9)
    return dict(summary=summary, verdict=verdict, protocol=protocol, runtime=runtime, progress=progress, runs=runs, frames=frames, complete=complete)


def sections(data):
    s = data["summary"]
    baseline = s.get("baseline", {})
    rows = [[LABELS["R0"], "과거 기준", *[number(H.metric(baseline, key), unit) for key, _, unit in METRICS]]]
    for arm in ARMS:
        family = s.get("arms", {}).get(arm, {})
        cells = []
        for key, _, unit in METRICS:
            values = family.get("metrics", {}).get(key, {})
            cells.append(number(values.get("mean"), unit) + f'<small>± {number(values.get("std"), unit)}</small>' if values else "미완료")
        rows.append([LABELS[arm], f"{sum(r['arm'] == arm for r in data['runs'])}/3 seeds", *cells])
    results = table(["모델", "완료", *[label for _, label, _ in METRICS]], rows)
    seeds = table(["모델", "Seed", *[label for _, label, _ in METRICS], "점 매칭/319", "Pose coverage"],
        [[LABELS[r["arm"]], r["seed"], *[number(H.metric(r, key), unit) for key, _, unit in METRICS],
          str(r["two_d"].get("keypoint_matched_frame_count_iou50")), number(100 * r["main_6d_coverage"], "%")] for r in data["runs"]])
    cirows = []
    for name, comparison in s.get("comparisons", {}).items():
        for metric, label, unit in METRICS:
            result = comparison["metrics"][metric]
            ci = result["session_cluster"]
            cirows.append([html.escape(name), label, number(result.get("difference"), unit),
                f"[{number(ci.get('low'), unit)}, {number(ci.get('high'), unit)}]", str(result.get("paired_frames", 0)),
                "개선 지지" if ci.get("confirmed_benefit") else "미확인"])
    cis = table(["비교 (왼쪽−오른쪽)", "지표", "대응 차이", "세션 95% CI", "공통 프레임", "구간"], cirows)
    negatives = []
    for r in data["runs"]:
        negatives.append([LABELS[r["arm"]], r["seed"], number(r["two_d"].get("box_ap50")), number(r["two_d"].get("box_ap50_95")),
            *[f"{r['negative']['by_threshold'][t]['n_frames_with_detection']}/2689 ({100*r['negative']['by_threshold'][t]['fraction_frames_with_detection']:.2f}%)" for t in ("0.001", "0.25", "0.5", "0.85")]])
    negative = table(["모델", "Seed", "Box AP50", "AP50:95", "FP @.001", "FP @.25", "FP @.5", "FP @.85"], negatives)
    sessions = []
    for arm, family in s.get("arms", {}).items():
        for sid, row in family.get("per_session", {}).items():
            sessions.append([LABELS[arm], html.escape(sid), *[number(row["metrics"][key]["mean"], unit) for key, _, unit in METRICS]])
    sessiontable = table(["모델", "촬영 세션", *[label for _, label, _ in METRICS]], sessions)
    runtime = data["runtime"]
    runtime_html = '<p class="muted">통제한 실제 runtime 비교가 아직 저장되지 않았습니다. 순차 추론 telemetry를 속도 개선 주장으로 사용하지 않습니다.</p>'
    if runtime.get("complete"):
        strict_failure = (runtime.get("complete") is True and runtime.get("timing_collection_complete") is True
            and runtime.get("PASS") is False and runtime.get("parity_PASS") is False
            and runtime.get("status") == "COMPLETE_WITH_STRICT_PARITY_FAILURE"
            and runtime.get("parity_policy") == dict(atol=1e-4, rtol=0, criterion_changed=False)
            and isinstance(runtime.get("strict_failures"), list) and bool(runtime["strict_failures"]))
        verified_pass = (runtime.get("PASS") is True and runtime.get("parity_PASS") is True
            and not runtime.get("strict_failures") and runtime.get("status") in (None, "COMPLETE"))
        if not strict_failure and not verified_pass:
            raise ValueError("Completed runtime failed execution/accuracy parity checks")
        runtime_html = ''
        if strict_failure:
            observations = [observation for row in runtime["runs"] for observation in row["observations"]]
            point_deltas = [observation["max_abs_delta_by_field"]["keypoints_xy"] for observation in observations]
            if (not point_deltas or not np.isfinite(point_deltas).all() or min(point_deltas) < 0
                    or len(observations) != sum(row["n"] for row in runtime["runs"])):
                raise ValueError("Invalid completed runtime observations")
            runtime_html = ('<div class="notice"><b>엄격한 예측 일치 검사 실패 · PASS=false / parity_PASS=false</b>'
                f'<p>{len(observations)}회 실측 중 {len(runtime["strict_failures"])}회가 기존 검사에 실패했습니다. '
                f'정확도 캐시 대비 관측된 최대 점 좌표 차이는 <b>{max(point_deltas):.9f}px</b>입니다. '
                '기존 atol=0.0001, rtol=0 기준을 유지했으며, 허용치를 완화하거나 실패를 PASS로 바꾸지 않았습니다. '
                '실사 정확도 캐시와 정확도 결과는 변경하지 않았습니다.</p>'
                '<p>CUDA sparse 허프 투표·역투표의 고정 입력 반복에서 수치적 비결정성이 확인됐습니다. '
                '이 관측 최대값은 다른 입력에서도 성립하는 오차 상한이 아닙니다. '
                '<a href="provenance/runtime_parity_diagnostic/FINDINGS.md">원인 진단 FINDINGS.md</a>.</p></div>'
                '<p><b>엄격한 예측 일치 검사 실패가 있는 참고 실측 시간</b></p>')
        runtime_html += table(["모델", "Seed", "Median", "P90", "Mean"],
            [[LABELS[row["arm"]], row["seed"], number(row.get("median_ms"), "ms"),
              number(row.get("p90_ms"), "ms"), number(row.get("mean_ms"), "ms")] for row in runtime.get("runs", [])])
        parity_text = '시간 측정 수집은 완료했으나 엄격한 예측 일치는 실패했습니다.' if strict_failure else 'Accuracy 예측과의 parity를 검증했습니다.'
        runtime_html += '<p class="foot">같은 26장(13세션×2) × 3반복, batch1. 원본 BGR 이후 실제 전체 predictor의 동기화 wall time이며 파일 decode·PnP는 제외합니다. ' + parity_text + ' 세부 측정 순서와 정의: <a href="RUNTIME.json">RUNTIME.json</a>.</p>'
    return dict(results=results, seeds=seeds, cis=cis, negative=negative, sessions=sessiontable, runtime=runtime_html)


ARCHITECTURE = '''<svg viewBox="0 0 1120 245" role="img" aria-label="영상의 YOLO 특징을 global Deep Hough Transform으로 모은 뒤 정규화 역투표로 점 예측에 되돌리는 full train 구조"><defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#89abc9"/></marker></defs><g fill="#142a40" stroke="#43627e"><rect x="10" y="75" width="165" height="95" rx="10"/><rect x="214" y="75" width="190" height="95" rx="10"/><rect x="445" y="20" width="240" height="195" rx="10"/><rect x="726" y="75" width="190" height="95" rx="10"/><rect x="957" y="75" width="153" height="95" rx="10"/></g><g stroke="#89abc9" stroke-width="2" marker-end="url(#arrow)"><path d="M175 122H210M404 122H441M685 122H722M916 122H953"/></g><g fill="#e7f1ff" text-anchor="middle" font-family="Arial,sans-serif" font-size="16"><text x="92" y="110">원본 BGR</text><text x="92" y="140">reflect100 → 640</text><text x="309" y="107">YOLO backbone / neck</text><text x="309" y="139">전체 가중치 학습</text><text x="565" y="48">P4 → Global DHT</text><text x="565" y="80">90 θ × 113 ρ</text><text x="565" y="113">Hough conv + 12선 logits</text><text x="565" y="147">정규화 transpose voting</text><text x="565" y="182">P3 / P4 / P5 residual</text><text x="821" y="106">기존 YOLO pose head</text><text x="821" y="139">검출 + 9점 공동학습</text><text x="1033" y="107">새 box / score / 9점</text><text x="1033" y="139">기존 MAIN 6D</text></g></svg>'''


TEMPLATE = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@@TITLE@@</title><style>
:root{font-family:Arial,"Noto Sans KR",sans-serif;color-scheme:dark;background:#09131e;color:#e3edf7}*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:30px 28px 70px}h1{font-size:34px}h2{font-size:23px}h3{font-size:15px}p,li{line-height:1.7}.muted,small{color:#a8bacb}small{display:block;font-size:11px;margin-top:4px}section,.notice{border:1px solid #34485c;border-radius:12px;background:#112131;padding:22px;margin:20px 0}.notice{border-left:4px solid #ffc078}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.cards div{background:#15283c;padding:20px;border-radius:10px}.cards b{display:block;font-size:27px}.cards span{font-size:12px;color:#abc0d2}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:right;white-space:nowrap;border-bottom:1px solid #34485c;padding:12px 9px}th:first-child,td:first-child{text-align:left}details{margin:16px 0}summary{cursor:pointer;color:#8fd1ff}a{color:#8fd1ff}svg{width:100%}.controls{display:flex;flex-wrap:wrap;gap:10px;margin:15px 0}.controls label{font-size:12px;color:#a6bdd0}select{display:block;margin-top:4px;padding:8px;max-width:340px;background:#0c1b29;color:#e1f1ff;border:1px solid #496178;border-radius:6px}.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.panel{min-width:0;background:#091724;padding:12px;border-radius:9px}canvas{display:block;width:100%;height:auto;background:#030b12}.legend{display:flex;flex-wrap:wrap;gap:20px;font-size:12px;margin:14px 0}.legend b{margin-right:5px}.foot{font-size:12px;color:#a4b8ca;line-height:1.7}pre{white-space:pre-wrap;max-height:350px;overflow:auto;font-size:11px}.wide{grid-column:1/-1}.hough-wrap{max-width:720px;margin:20px auto}.tag{color:#82c5ef;font-size:13px}.matrix{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.matrix div{padding:10px;background:#0b1927;border-radius:6px;font-size:12px}button{background:#254462;color:#fff;border:1px solid #6687a8;padding:8px;border-radius:6px;cursor:pointer}.zoom .panels{grid-template-columns:1fr}.zoom canvas{max-width:none}input[type=checkbox]{vertical-align:middle}@media(max-width:750px){main{padding:18px 10px}.panels,.cards{grid-template-columns:1fr}.matrix{grid-template-columns:1fr}h1{font-size:27px}}
</style></head><body><main><div class="tag">실제 end-to-end 점·선 아키텍처 · 동일 예산 대조실험</div><h1>@@TITLE@@</h1><div class="notice"><b>@@HEADLINE@@</b><p>@@REASONS@@</p></div><div class="cards"><div><b>55,980 / 4,020</b><span>동일 합성 train / validation</span></div><div><b>@@TRAINED@@ / 9</b><span>3 arms × 3 seeds 학습 완료</span></div><div><b>319 + 2,689</b><span>실제 새 모델 positive + negative forward</span></div><div><b>@@EVALUATED@@ / 9</b><span>canonical 2D · MAIN 6D 평가 완료</span></div></div>
<section><h2>실제 full train 구조</h2>@@ARCHITECTURE@@<p>주 비교는 <b>점·선 공동학습 vs 같은 예산 점 전용</b>입니다. 허프 특징 대조군은 동일한 global DHT 구조를 점·검출 손실로만 학습합니다. 공동학습군은 12개 역할의 선 손실을 추가합니다. Backbone, neck, pose head와 허프 분기를 함께 학습하며, 추론 시 GT·point proposal·후처리 WLS를 입력하지 않습니다.</p><p class="foot">정규화 역투표는 HT의 수학적 역함수가 아닙니다. 학습과 추론의 입력·증강 및 BN 규칙은 TRAIN_PROTOCOL을 따릅니다. R0는 추가 학습 전 과거 기준입니다.</p><details><summary>실제 예산과 진행 상태</summary><pre>@@BUDGET@@</pre><div class="matrix">@@MATRIX@@</div></details></section>
<section><h2>실사 2D · 6D 결과</h2><p class="muted">3개 학습 seed의 평균 ± 표본 표준편차(ddof=1). 실행 완료와 성능 향상 판정은 구분합니다.</p>@@RESULTS@@<p class="foot">2D: IoU≥0.5로 매칭된 최고 점수 검출의 감독되는 9점. 6D: 기존 MAIN selector와 geometry-reconstructed GT. 이전 52장·8점 frozen-DHT 실험의 수치는 섞지 않습니다.</p><details><summary>개별 seed와 전체 모집단 coverage</summary>@@SEEDS@@</details><details><summary>같은 세션 10,000회 paired bootstrap · 95% CI</summary><p class="foot">같은 프레임/세션 draw를 모든 모델과 3 seed에 적용한 뒤 seed별 통계 차이를 평균합니다. 표의 차이는 비교 모델 모두에서 관측 가능한 공통 프레임 기준이므로 위 전체 모집단 평균 차이와 다를 수 있습니다. 엄격한 향상 판정에는 전체 평균 개선, 올바른 방향의 세션 CI, 각 seed의 coverage 보존이 함께 필요합니다.</p>@@CIS@@</details><details><summary>13개 촬영 세션별 결과</summary>@@SESSIONS@@</details></section>
<section><h2>상자 AP와 실제 negative 결과</h2><p class="muted">Negative 2,689장도 각 새 체크포인트로 모두 추론했습니다. FP는 해당 score 이상 후보가 하나 이상 있는 이미지 비율이며, 임계값은 실사 결과로 고르지 않았습니다.</p>@@NEGATIVE@@</section>
<section id="gallery"><h2>같은 원본에서 점과 실제 선 증거 확인</h2><p class="muted">기본은 사용자가 지적한 번호 대응 실패 사례입니다. 도움/악화 정렬과 쉬움/어려움은 canonical GT를 사용한 사후 진단이며 운영 중 선택 규칙이 아닙니다. 원본 좌표와 점 번호를 유지합니다.</p><div class="controls"><label>새 모델<select id="arm"></select></label><label>Seed<select id="seed"><option>1</option><option>2</option><option>3</option></select></label><label>비교 기준<select id="reference"><option value="point_only">같은 seed 점 전용</option><option value="R0">R0 과거 기준</option></select></label><label>정렬<select id="rank"><option value="case">지정 사례 / 전체</option><option value="best">가장 도움</option><option value="worst">가장 악화</option><option value="neutral">변화 적음</option><option value="evidence">선 증거 저장된 고정 예제</option></select></label><label>R0 사후 난이도<select id="difficulty"><option value="all">전체</option><option value="easy_le10">≤10px</option><option value="moderate_10_20">10–20px</option><option value="hard_gt20">&gt;20px</option><option value="missing">미매칭</option></select></label><label>촬영 세션<select id="session"><option value="all">전체</option></select></label><label>이미지<select id="frame"></select></label><label>선 역할<select id="role"></select></label></div><div class="controls"><label><input id="showgt" type="checkbox" checked> GT 겹침</label><label><input id="showpoints" type="checkbox" checked> 점·구조선</label><label><input id="showheat" type="checkbox" checked> 선 증거 heatmap</label><button id="zoom">크게 보기</button></div><div id="frameinfo" class="notice"></div><div class="legend"><span><b style="color:#68e0a0">●</b>GT</span><span><b style="color:#56d8ed">●</b>같은 예산 점 / R0</span><span><b style="color:#f38cdd">●</b>새 예측</span><span><b style="color:#ffd18c">→</b>같은 ID의 점 이동</span></div><div class="panels"><div class="panel"><h3>① Original + supervised GT</h3><canvas id="gt"></canvas></div><div class="panel"><h3 id="beforetitle">② Matched-budget point model</h3><canvas id="before"></canvas></div><div class="panel"><h3>③ New model · original coordinates</h3><canvas id="after"></canvas></div><div class="panel"><h3>④ Learned line evidence · normalized backprojection</h3><canvas id="evidence"></canvas></div></div><div id="evidenceinfo" class="foot"></div><div class="hough-wrap"><canvas id="hough"></canvas></div><p class="foot">④는 실제 forward에서 포착한 12-role sigmoid logits의 정규화 역투표입니다. Attention·인과 설명·물리적 edge 가시성·보정된 정답 확률이 아닙니다. 각 역할의 최솟값/최댓값으로 색 범위만 조정합니다. 정확한 배열은 NPZ에 보존합니다. Feature grid의 셀을 실제 letterbox 입력에서 원본으로 되돌렸으며 padding 영역은 화면 밖에 놓입니다. 고정 예제 이외에는 가상 선 증거를 만들지 않습니다.</p></section>
<section><h2>속도 측정과 해석 범위</h2>@@RUNTIME@@<p class="foot">허프 특징 대조군(hough_features)은 선 역할 정답으로 학습하지 않습니다. 이 대조군의 채널 번호·역할명은 공동학습군과 같은 인덱스 표기이며, 채널이 그 물리적 역할을 배웠다는 보장은 없습니다.</p><ul>@@LIMITS@@</ul><p>@@LINKS@@</p><p class="foot">저장된 모델 결과·GT·학습 영수증만 읽는 보고서입니다. HTML 생성은 추론·학습·선택을 수행하지 않습니다. 모든 입력 SHA는 REPORT_RENDER.json에 기록됩니다.</p></section></main><script id="report-data" type="application/json">@@DATA@@</script><script>
const DATA=JSON.parse(document.getElementById('report-data').textContent),$=id=>document.getElementById(id),E=DATA.edges,images=new Map(),decoded=new WeakMap();let available=[],renderToken=0;
const finite=p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite),f=(v,n=2)=>v==null?'—':Number(v).toFixed(n);
function option(el,value,text){const o=document.createElement('option');o.value=value;o.textContent=text;el.append(o)}
DATA.arms.forEach(a=>option($('arm'),a,DATA.labels[a]));$('arm').value='hough_joint';DATA.roles.forEach((r,i)=>option($('role'),i,i+' · '+r+' ['+E[i].join('–')+']'));[...new Set(DATA.frames.map(f=>f.session))].sort().forEach(s=>option($('session'),s,s));
function ref(frame){return $('reference').value==='R0'?{points:frame.r0,error:frame.r0_error}:frame.runs['point_only_seed'+$('seed').value]||{}}
function current(frame){return frame.runs[$('arm').value+'_seed'+$('seed').value]||{}}
function delta(frame){const a=current(frame).error,b=ref(frame).error;return a==null||b==null?null:a-b}
function reset(){available=DATA.frames.filter(x=>x.runs[$('arm').value+'_seed'+$('seed').value]&&($('session').value==='all'||x.session===$('session').value)&&($('difficulty').value==='all'||x.difficulty===$('difficulty').value));const rank=$('rank').value;if(['best','worst','neutral'].includes(rank))available=available.filter(x=>delta(x)!=null);if(rank==='evidence')available=available.filter(x=>current(x).evidence);available.sort((a,b)=>rank==='best'?delta(a)-delta(b):rank==='worst'?delta(b)-delta(a):rank==='neutral'?Math.abs(delta(a))-Math.abs(delta(b)):a.id===DATA.required_case?-1:b.id===DATA.required_case?1:a.id.localeCompare(b.id));$('frame').replaceChildren();available.forEach((x,i)=>option($('frame'),i,x.id));render()}
function loadImage(src){if(!images.has(src))images.set(src,new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=()=>reject(new Error('embedded image decode'));i.src=src}));return images.get(src)}
function line(ctx,a,b,c,w=2){if(!finite(a)||!finite(b))return;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.strokeStyle=c;ctx.lineWidth=w;ctx.stroke()}
function points(ctx,p,c,alpha=1){if(!p)return;ctx.save();ctx.globalAlpha=alpha;E.forEach(([a,b])=>line(ctx,p[a],p[b],c,1.5));p.forEach((q,i)=>{if(!finite(q))return;ctx.beginPath();ctx.arc(...q,i===8?5:4,0,Math.PI*2);ctx.fillStyle=c;ctx.fill();ctx.font='bold 12px Arial';ctx.lineWidth=3;ctx.strokeStyle='#081320';ctx.strokeText(i,q[0]+6,q[1]-5);ctx.fillText(i,q[0]+6,q[1]-5)});ctx.restore()}
function arrow(ctx,a,b){if(!finite(a)||!finite(b)||Math.hypot(a[0]-b[0],a[1]-b[1])<.2)return;line(ctx,a,b,'#ffd18c',1.6);const t=Math.atan2(b[1]-a[1],b[0]-a[0]);for(const s of [-.5,.5])line(ctx,b,[b[0]-6*Math.cos(t+s),b[1]-6*Math.sin(t+s)],'#ffd18c',1.6)}
function canvas(id,img,w,h){const c=$(id);c.width=w;c.height=h;const ctx=c.getContext('2d');ctx.drawImage(img,0,0,w,h);return ctx}
function bytes(map){if(!decoded.has(map))decoded.set(map,Uint8Array.from(atob(map.bytes),c=>c.charCodeAt(0)));return decoded.get(map)}
function colorMap(map,role,alpha=255){const [n,h,w]=map.shape,c=document.createElement('canvas');c.width=w;c.height=h;const ctx=c.getContext('2d'),out=ctx.createImageData(w,h),b=bytes(map),offset=role*h*w;for(let i=0;i<h*w;i++){const v=b[offset+i]/255;out.data[4*i]=Math.round(255*v);out.data[4*i+1]=Math.round(185*Math.sqrt(v));out.data[4*i+2]=Math.round(160*(1-v));out.data[4*i+3]=alpha}ctx.putImageData(out,0,0);return c}
function parameterPlot(e,role){const c=$('hough');c.width=720;c.height=330;const ctx=c.getContext('2d');ctx.fillStyle='#081725';ctx.fillRect(0,0,720,330);const x=75,y=25,w=610,h=245;ctx.imageSmoothingEnabled=false;ctx.drawImage(colorMap(e.hough,role),x,y,w,h);ctx.fillStyle='#d8e8f7';ctx.font='13px Arial';ctx.textAlign='center';ctx.fillText('Signed rho · P4 feature cells',x+w/2,318);ctx.fillText(f(e.rho_values[0],1),x,291);ctx.fillText('0',x+w/2,291);ctx.fillText(f(e.rho_values.at(-1),1),x+w,291);ctx.textAlign='right';ctx.fillText(f(e.theta_degrees[0],0)+'°',66,y+10);ctx.fillText(f(e.theta_degrees.at(-1),0)+'°',66,y+h);ctx.save();ctx.translate(19,150);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText('Normal theta · degrees',0,0);ctx.restore()}
async function render(){window.REPORT_READY=false;const token=++renderToken,frame=available[Number($('frame').value)||0];if(!frame){$('frameinfo').textContent='완료된 실제 모델 예측이 없거나 선택 조건에 맞는 프레임이 없습니다.';for(const id of ['gt','before','after','evidence','hough']){$(id).width=640;$(id).height=360}$('evidenceinfo').textContent='';window.REPORT_READY=true;return}const run=current(frame),before=ref(frame),raw=await loadImage(frame.image);if(token!==renderToken)return;$('beforetitle').textContent=$('reference').value==='R0'?'② Historical YOLO26n R0':'② Matched-budget point model';$('frameinfo').textContent=frame.id+' · '+frame.session+' / '+frame.domain+' · R0 GT-bin='+frame.difficulty+' · 매칭 프레임 평균 점 오차 '+f(before.error)+' → '+f(run.error)+'px (Δ '+f(delta(frame))+'px) · 새 후보 '+run.n_candidates+'개, score '+f(run.score,3);
let ctx=canvas('gt',raw,frame.width,frame.height);if($('showgt').checked)points(ctx,frame.gt,'#68e0a0');ctx=canvas('before',raw,frame.width,frame.height);if($('showgt').checked)points(ctx,frame.gt,'#68e0a0',.38);if($('showpoints').checked)points(ctx,before.points,'#56d8ed');ctx=canvas('after',raw,frame.width,frame.height);if($('showgt').checked)points(ctx,frame.gt,'#68e0a0',.38);if($('showpoints').checked){points(ctx,run.points,'#f38cdd');if(before.points&&run.points)before.points.forEach((p,i)=>arrow(ctx,p,run.points[i]))}ctx=canvas('evidence',raw,frame.width,frame.height);const e=run.evidence,role=Number($('role').value);if(e){if($('showheat').checked){const a=e.affine;ctx.save();ctx.globalAlpha=.57;ctx.setTransform(16/a[0][0],0,0,16/a[1][1],-a[0][2]/a[0][0],-a[1][2]/a[1][1]);ctx.drawImage(colorMap(e.spatial,role),0,0);ctx.restore()}if($('showgt').checked)points(ctx,frame.gt,'#68e0a0',.55);if($('showpoints').checked)points(ctx,run.points,'#f38cdd');$('evidenceinfo').textContent=($('arm').value==='hough_features'?'선 역할로 감독하지 않은 채널: 역할 이름은 비교용 채널 번호이며 해당 경계를 검출했다는 뜻이 아닙니다. Nominal role ':'선 역할 손실로 공동학습한 실제 선 증거 지도 · role ')+role+' '+DATA.roles[role]+' · sigmoid 범위 '+f(e.hough.minimum[role],5)+'–'+f(e.hough.maximum[role],5)+' · 역투표 범위 '+f(e.spatial.minimum[role],5)+'–'+f(e.spatial.maximum[role],5)+' · 각 역할 min/max 색 스케일(확률 보정 아님). 아래 θ×ρ는 정규화 합이 1인 분포가 아니라 독립 sigmoid 값입니다.';parameterPlot(e,role)}else{$('evidenceinfo').textContent='이 모델/프레임에는 저장된 실제 선 증거가 없습니다. 지정 사례 또는 “선 증거 저장된 고정 예제”를 선택하세요. 점 전용 대조군에는 허프 분기가 없습니다.';$('hough').width=720;$('hough').height=1}window.REPORT_READY=true;window.REPORT_CURRENT_FRAME=frame.id}
['arm','seed','reference','rank','difficulty','session'].forEach(id=>$(id).addEventListener('change',reset));['frame','role','showgt','showpoints','showheat'].forEach(id=>$(id).addEventListener('change',render));$('zoom').addEventListener('click',()=>{$('gallery').classList.toggle('zoom')});window.REPORT_FRAME_COUNT=DATA.frames.length;reset();
</script></body></html>'''


def render(run_dir, screenshot=True):
    if not (run_dir / "PURPOSE.md").is_file():
        raise ValueError("Purpose-declared result directory required")
    inputs = H.Inputs(run_dir)
    inputs.bind(HELPERS)
    inputs.bind(HERE / "line_targets.py")
    data = collect(inputs)
    sections_html = sections(data)
    summary, verdict = data["summary"], data["verdict"]
    headline = verdict.get("headline_ko") if data["complete"] else "실험 미완료 · 실제 공동학습 성능 향상은 아직 확인되지 않았습니다"
    reasons = verdict.get("reasons", []) if data["complete"] else ["9개 학습·실제 평가와 동일 예산 대조군 통계가 모두 끝나야 개선 여부를 판정합니다."]
    matrix = ''.join(f'<div>{LABELS[r["arm"]]} · seed {r["seed"]}<br>{r["state"]} · epochs {r["epochs_completed"] or "—"} / optimizer steps {r["optimizer_steps"] or "—"}</div>' for r in data["progress"])
    limits = summary.get("limitations", ["재사용 실사 DEV319+2689 결과입니다. 독립 final 데이터는 접근하지 않습니다.", "아직 실제 성능 결과가 완성되지 않았습니다."])
    links = [f'<a href="{name}">{name}</a>' for name in ("TRAIN_PROTOCOL.json", "SUMMARY.json", "VERDICT.json", "AGGREGATE_COMPLETE.json", "RUNTIME.json") if (run_dir / name).is_file()]
    values = {**{k.upper(): v for k,v in sections_html.items()}, "TITLE": html.escape(TITLE), "HEADLINE": html.escape(headline),
        "REASONS": '<br>'.join(html.escape(str(r)) for r in reasons), "ARCHITECTURE": ARCHITECTURE,
        "TRAINED": str(sum(r["state"] in ("TRAINED", "EVALUATED") for r in data["progress"])), "EVALUATED": str(len(data["runs"])),
        "BUDGET": html.escape(json.dumps(summary.get("training_budget") or data["protocol"], ensure_ascii=False, indent=2)),
        "MATRIX": matrix, "LIMITS": ''.join('<li>'+html.escape(x)+'</li>' for x in limits), "LINKS": ' · '.join(links),
        "DATA": json.dumps(H.clean(dict(frames=data["frames"], arms=ARMS, labels=LABELS, edges=EDGES, roles=ROLE_NAMES, required_case=REQUIRED_CASE)), ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')}
    document = TEMPLATE
    for key, value in values.items():
        document = document.replace('@@'+key+'@@', value)
    if '@@' in document or '<script src=' in document or '<link ' in document:
        raise ValueError("Unresolved template/external rendering dependency")
    if any(not frame['image'].startswith('data:image/') for frame in data['frames']):
        raise ValueError("All gallery images must be embedded")
    destination = run_dir / 'index.html'
    destination.write_text(document)
    digest = sha(destination)
    qa = dict(complete=True, PASS=True, html_sha256=digest, experiment_complete=data['complete'],
        scope='Automatic offline document/input validation; human visual and interaction review is separate.',
        broken_local_images=0, unresolved_placeholders=0, external_render_dependencies=0,
        n_gallery_frames=len(data['frames']), browser_screenshot='not_attempted')
    chrome = shutil.which('google-chrome') or shutil.which('chromium')
    if screenshot and chrome:
        try:
            with tempfile.TemporaryDirectory(prefix='dht-joint-report-chrome-') as profile:
                result = subprocess.run([chrome, '--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
                    f'--user-data-dir={profile}', '--window-size=1560,1100', '--virtual-time-budget=3500',
                    f'--screenshot={run_dir / "report.png"}', destination.as_uri()], capture_output=True, text=True, timeout=30)
                qa['browser_screenshot'] = 'saved' if result.returncode == 0 and (run_dir / 'report.png').is_file() else 'unavailable'
                qa['browser_returncode'] = result.returncode
        except (subprocess.TimeoutExpired, OSError) as exc:
            qa.update(browser_screenshot='unavailable', browser_note=str(exc))
    (run_dir / 'VISUAL_QA.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2)+'\n')
    receipt = dict(complete=True, PASS=True, html=str(destination), html_sha256=digest, title=TITLE,
        input_sha256=inputs.hashes, renderer_sha256=sha(__file__), experiment_complete=data['complete'],
        n_completed_evaluations=len(data['runs']), n_gallery_frames=len(data['frames']),
        screenshot=str(run_dir / 'report.png') if qa['browser_screenshot']=='saved' else None,
        no_inference=True, no_auto_open=True)
    (run_dir / 'REPORT_RENDER.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='input_sha256'}, ensure_ascii=False, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--no-screenshot', action='store_true')
    args = parser.parse_args()
    render(args.run_dir.resolve(), screenshot=not args.no_screenshot)
