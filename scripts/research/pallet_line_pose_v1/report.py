"""Offline report of saved paper point-line results; never performs inference."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TITLE = "Pallet Line Pose · YOLO26n 검증"
ARMS = ("image_joint", "geometry_joint", "image_line_only")
LABELS = {"R0": "YOLO26n R0", "image_joint": "영상 + 점·선 공동학습",
          "geometry_joint": "기하 입력 대조군", "image_line_only": "선 손실 대조군"}
METRICS = [("keypoint_location_median_px", "점 중앙값 ↓", "px"),
           ("keypoint_location_p90_px", "점 P90 ↓", "px"),
           ("rotation_median_deg", "회전 중앙값 ↓", "°"),
           ("translation_median_cm", "이동 중앙값 ↓", "cm"),
           ("iou3d_median", "3D IoU ↑", ""), ("add_sym_auc", "ADDsym AUC ↑", "")]
EDGES = [[1, 2], [3, 0], [5, 6], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def number(value, unit=""):
    if value is None:
        return "—"
    try:
        return f"{float(value):.{4 if not unit else 2}f}{unit}"
    except (TypeError, ValueError):
        return "—"


class Inputs:
    def __init__(self, run_dir):
        self.root, self.hashes = run_dir, {}

    def read(self, path, default=None):
        path = Path(path)
        if not path.is_file():
            return default
        self.hashes[str(path.resolve())] = sha(path)
        return json.loads(path.read_text())

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"report source changed: {path}")
        self.hashes[str(path)] = digest


def metric(row, name):
    for block in (row, row.get("two_d", {}), row.get("main_6d", {}), row.get("metrics", {})):
        if name in block:
            value = block[name]
            return value.get("mean") if isinstance(value, dict) else value
    return None


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def data_uri(image, max_side=1000):
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        image = cv2.resize(image, (round(w * max_side / max(h, w)), round(h * max_side / max(h, w))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("report image encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode()


def gallery(inputs, prediction_files):
    if not prediction_files:
        return []
    # Read only the explicitly bound positive DEV population, never final-test data.
    baseline = inputs.read(inputs.root / "BASELINE_PROTOCOL.json")
    pos_path = REPO / "challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json"
    inputs.bind(pos_path, baseline["source_sha256"][str(pos_path)])
    population = inputs.read(pos_path)
    if population["expected_count"] != 319 or population["role"] != "DEV":
        raise ValueError("gallery requires the frozen 319-positive DEV population")
    items = {str((REPO / row["image_path"]).resolve()): row for row in population["items"]}
    matched = {}
    csv_path = inputs.root / "evaluation/R0_PASSTHROUGH/PAPER_2D_per_frame.csv"
    if csv_path.is_file():
        inputs.bind(csv_path)
        matched = {str((REPO / row["image"]).resolve()): row["top_iou50_match"] == "True"
                   for row in csv.DictReader(csv_path.open())}
    frames = {}
    for path in prediction_files:
        payload = inputs.read(path)
        if not payload.get("complete"):
            continue
        key = f"{payload['arm']}_seed{payload['seed']}"
        for record in payload["records"]:
            image_path = (REPO / record["image_key"]).resolve()
            if str(image_path) not in items:
                raise ValueError("prediction gallery contains a non-DEV319 image")
            item = items[str(image_path)]
            if str(image_path) not in frames:
                inputs.bind(image_path, record["image_sha256"])
                annotation_path = (REPO / item["gt_v2_path"]).resolve()
                inputs.bind(annotation_path, baseline["source_sha256"][str(annotation_path)])
                annotation = inputs.read(annotation_path)["objects"][0]["keypoint_annotations"]
                if len(annotation) != 9:
                    raise ValueError("nine GT entries required")
                gt = [a["xy"] if a["visibility"] > 0 else None for a in annotation]
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"report image decode failed: {image_path}")
                from source_data import prepare_image
                padded = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
                chw, tf = prepare_image(padded)
                input_bgr = np.rint(chw.transpose(1, 2, 0)[..., ::-1] * 255).astype(np.uint8)
                frames[str(image_path)] = dict(id=item["frame_id"], session=item["session_id"],
                    domain=item.get("domain", "UNKNOWN"), object_type=item.get("object_type", ""),
                    width=image.shape[1], height=image.shape[0], image=data_uri(image),
                    input_image=data_uri(input_bgr), transform=tf.to_dict(),
                    gt=gt, matched=matched.get(str(image_path), False), runs={})
            frame = frames[str(image_path)]
            pred = record["prediction"]
            selected = pred["selected_index"]
            before = pred.get("baseline_selected")
            before = before["keypoints_xy"] if before else None
            after = pred["candidates"][selected]["keypoints_xy"] if selected is not None else None
            errors_before, errors_after = [], []
            if frame["matched"] and before is not None and after is not None:
                for g, b, a in zip(frame["gt"], before, after):
                    if g is not None and b is not None and a is not None and np.isfinite([*b, *a]).all():
                        errors_before.append(float(np.linalg.norm(np.asarray(b) - g)))
                        errors_after.append(float(np.linalg.norm(np.asarray(a) - g)))
            bmean = float(np.mean(errors_before)) if errors_before else None
            amean = float(np.mean(errors_after)) if errors_after else None
            frame["runs"][key] = dict(before=before, after=after, baseline_error=bmean,
                new_error=amean, delta=None if bmean is None else amean - bmean,
                diagnostics=pred.get("diagnostics"), lam=pred["lam"], status=pred["status"],
                n_supervised=len(errors_before))
    return list(frames.values())


def collect(inputs):
    root = inputs.root
    protocol = inputs.read(root / "TRAIN_PROTOCOL.json", {})
    source = inputs.read(root / "SOURCE_DATA_AUDIT.json", {})
    selection = inputs.read(root / "SELECTION.json", {})
    synthetic = inputs.read(root / "SYNTHETIC_EVALUATION.json", {})
    summary = inputs.read(root / "SUMMARY.json", {})
    verdict = inputs.read(root / "VERDICT.json", {})
    runtime = inputs.read(root / "RUNTIME.json", {})
    baseline = summary.get("baseline") or inputs.read(root / "evaluation/R0_PASSTHROUGH/RESULTS.json", {})
    runs, progress, prediction_files = [], [], []
    for arm in ARMS:
        for seed in (1, 2, 3):
            key = f"{arm}_seed{seed}"
            trained = inputs.read(root / "runs" / key / "COMPLETION.json", {})
            underway = inputs.read(root / "runs" / key / "PROGRESS.json", {})
            result = inputs.read(root / "evaluation" / key / "RESULTS.json", {})
            complete = trained.get("complete") and trained.get("PASS") and trained.get("step") == 6000 and not trained.get("smoke")
            state = "EVALUATED" if result.get("complete") else "TRAINED" if complete else "TRAINING" if underway else "NOT_RUN"
            progress.append(dict(arm=arm, seed=seed, state=state, step=6000 if complete else underway.get("step", 0)))
            if result.get("complete"):
                runs.append(dict(**result, model_arm=arm, seed=seed))
                path = root / "evaluation" / key / "IMAGE_PREDICTIONS.json"
                if path.is_file():
                    prediction_files.append(path)
    done = bool(summary.get("complete") and verdict.get("complete") and len(runs) == 9)
    return dict(protocol=protocol, source=source, selection=selection, synthetic=synthetic,
                summary=summary, verdict=verdict, runtime=runtime, baseline=baseline,
                runs=runs, progress=progress, complete=done, frames=gallery(inputs, prediction_files))


def sections(data):
    baseline = data["baseline"]
    rows = [[LABELS["R0"], "기준", *[number(metric(baseline, key), unit) for key, _, unit in METRICS]]]
    for arm in ARMS:
        runs = [r for r in data["runs"] if r["model_arm"] == arm]
        cells = []
        for key, _, unit in METRICS:
            values = [metric(r, key) for r in runs]
            values = [v for v in values if v is not None]
            deviation = float(np.std(values, ddof=1)) if len(values) > 1 else None
            cells.append(number(np.mean(values), unit) + f'<small>± {number(deviation, unit)}</small>' if values else "미실행")
        rows.append([LABELS[arm], f"{len(runs)}/3 seeds", *cells])
    main_table = table(["모델", "완료", *[label for _, label, _ in METRICS]], rows)
    detail_rows = [[LABELS[r["model_arm"]], r["seed"], *[number(metric(r, k), u) for k, _, u in METRICS]] for r in data["runs"]]
    seeds = table(["모델", "Seed", *[label for _, label, _ in METRICS]], detail_rows) if detail_rows else '<p class="muted">완료된 새 모델 평가가 없습니다.</p>'
    rules = []
    for arm in ARMS:
        rule = data["selection"].get("selected_rules", {}).get(arm)
        if rule:
            temperatures = data["selection"].get("temperatures", {}).get(arm, {})
            rules.append([LABELS[arm], str(rule["lam"]), "없음" if rule["max_move_image_diagonal_fraction"] is None else f"원본 대각선 × {rule['max_move_image_diagonal_fraction']}",
                          " / ".join(str(temperatures.get(str(s), {}).get("temperature", "—")) for s in (1, 2, 3))])
    selection_table = table(["모델", "λ", "이동 상한", "Temperature: seed1 / 2 / 3"], rules) if rules else '<p class="muted">합성 calibration/selection 완료 전입니다. 선택값을 만들지 않았습니다.</p>'
    synth_rows = [[LABELS[r["arm"]], r["seed"], r["readout"], r["frames"], number(r["observed_median_px"], "px"), number(r["observed_p90_px"], "px"), number(r["frame_score_mean"]) ] for r in data["synthetic"].get("summaries", [])]
    synthetic_table = table(["모델", "Seed", "판독", "프레임", "점 중앙값", "P90", "선택 기준 점수 ↓"], synth_rows) if synth_rows else '<p class="muted">선택 동결 후 합성 heldout을 평가합니다.</p>'
    runtime_rows = []
    for key, row in data["runtime"].get("by_run", {}).items():
        runtime_rows.append([LABELS.get(row["arm"], row["arm"]), row["seed"], str(row.get("lam")),
                            number(row["baseline_ms"]["median"], "ms"), number(row["integrated_ms"]["median"], "ms"),
                            number(row["paired_added_ms"]["median"], "ms"), row["integrated_ms"]["n"]])
    runtime_table = table(["모델", "Seed", "λ", "R0 중앙값", "통합 중앙값", "쌍별 추가시간", "호출"], runtime_rows) if runtime_rows else '<p class="muted">실제 전체 추론시간은 아직 측정되지 않았습니다.</p>'
    comparisons = []
    for name, comparison in data["summary"].get("comparisons", {}).items():
        for key, label, unit in METRICS:
            stat = comparison.get("metrics", {}).get(key)
            if stat:
                ci = stat.get("session_cluster", {})
                comparisons.append([html.escape(name), label, number(stat.get("difference"), unit),
                                    f"[{number(ci.get('low'), unit)}, {number(ci.get('high'), unit)}]"])
    ci_table = table(["대조", "지표", "차이: 앞 모델 − 뒤 모델", "세션 bootstrap 95% CI"], comparisons) if comparisons else '<p class="muted">세션 단위 쌍별 통계가 준비되면 표시됩니다.</p>'
    session_rows = []
    for arm, arm_data in data["summary"].get("arms", {}).items():
        for session, block in arm_data.get("per_session", {}).items():
            session_rows.append([LABELS.get(arm, arm), html.escape(session), *[number(metric(block, key), unit) for key, _, unit in METRICS]])
    sessions = table(["모델", "세션", *[label for _, label, _ in METRICS]], session_rows) if session_rows else '<p class="muted">완료된 세션별 집계가 없습니다. 검증된 촬영각도 메타데이터 없이 각도별 성능을 추정하지 않습니다.</p>'
    return dict(main_table=main_table, seeds=seeds, selection_table=selection_table,
                synthetic_table=synthetic_table, runtime_table=runtime_table, ci_table=ci_table, sessions=sessions)


ARCHITECTURE = '''<svg viewBox="0 0 1100 205" role="img" aria-label="원본영상에서 YOLO 특징과 예측점으로 선 후보를 평가하고 점을 보정하는 구조"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#8ba7c5"/></marker></defs><g fill="#142a42" stroke="#385573"><rect x="8" y="50" width="170" height="100" rx="12"/><rect x="222" y="50" width="190" height="100" rx="12"/><rect x="456" y="30" width="210" height="140" rx="12"/><rect x="710" y="50" width="180" height="100" rx="12"/><rect x="934" y="50" width="158" height="100" rx="12"/></g><g stroke="#8ba7c5" stroke-width="2" marker-end="url(#arrow)"><path d="M178 100H218M412 100H452M666 100H706M890 100H930"/></g><g fill="#e7f2ff" text-anchor="middle" font-size="16" font-family="sans-serif"><text x="93" y="82">원본 BGR</text><text x="93" y="112">reflect100</text><text x="93" y="135">기존 YOLO 입력</text><text x="317" y="80">Frozen YOLO26n</text><text x="317" y="110">P3/P4 + 예측 9점</text><text x="317" y="135">단일 forward</text><text x="561" y="60">학습하는 선 분기</text><text x="561" y="89">13×17 후보 + null</text><text x="561" y="118">선당 32개 특징 샘플</text><text x="561" y="149">역할별 후보 점수</text><text x="800" y="81">합성에서 고정한 T</text><text x="800" y="111">선 분포 + 점 anchor</text><text x="800" y="135">가중 2×2 solve</text><text x="1013" y="81">보정된 8점</text><text x="1013" y="111">centroid 보존</text><text x="1013" y="135">box·score 보존</text></g></svg>'''


TEMPLATE = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@@TITLE@@</title><style>
:root{color-scheme:dark;font-family:Arial,"Noto Sans KR",sans-serif;background:#07111d;color:#d9e7f6}*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:38px 32px 80px}h1{font-size:36px;margin:12px 0}h2{font-size:24px;margin:0 0 16px}h3{font-size:16px}p{line-height:1.7}a{color:#8dcfff}small{display:block;font-size:11px;color:#99abc0;margin-top:4px}.muted{color:#9bb0c7}.tag{display:inline-block;padding:7px 12px;background:#1d3b58;border:1px solid #386186;border-radius:30px;font-size:12px}.notice{border-left:4px solid #ffbd6a;padding:12px 18px;background:#1c2735}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.card,section{background:#101f30;border:1px solid #293e55;border-radius:14px;padding:22px}.card b{font-size:28px;display:block}.card span{font-size:13px;color:#a9bdd2}section{margin-top:20px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:13px 10px;text-align:right;border-bottom:1px solid #293d52;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:#a9bed5;font-weight:500}td{font-variant-numeric:tabular-nums}details{margin-top:18px}summary{cursor:pointer;color:#9aceff;padding:8px 0}.controls{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}.controls label{font-size:12px;color:#9fb6cc}.controls select{display:block;background:#091725;color:#e3f0ff;border:1px solid #3c5772;padding:9px;border-radius:6px;max-width:330px;margin-top:4px}.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.panel{background:#091624;border:1px solid #2b4057;border-radius:10px;padding:12px;min-width:0}.panel h3{margin:3px 0 10px}canvas{width:100%;height:auto;display:block;background:#03080f}.legend{display:flex;gap:22px;font-size:13px;margin:12px 0}.legend i{display:inline-block;width:12px;height:12px;margin-right:6px;border-radius:50%}.evidencebar{height:5px;background:#294661;border-radius:4px;min-width:70px}.evidencebar i{display:block;height:5px;background:#eda85e;border-radius:4px}.matrix{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.matrix div{padding:12px;background:#0b1929;border-radius:7px;font-size:12px}.matrix strong{display:block;margin-bottom:7px}.foot{font-size:12px;color:#96aec7;line-height:1.8}svg{width:100%;height:auto}li{line-height:1.7;margin-bottom:8px}.empty{padding:30px;border:1px dashed #415773;text-align:center;color:#a3bad0}.good{color:#70deb2}.bad{color:#ffbd83}@media(max-width:800px){main{padding:20px 12px}.cards,.panels{grid-template-columns:1fr 1fr}.matrix{grid-template-columns:1fr}.card{padding:12px}.card b{font-size:22px}h1{font-size:29px}}@media(max-width:550px){.panels,.cards{grid-template-columns:1fr}}
</style></head><body><main><span class="tag">논문 평가 계약 · 새 학습 분기</span><h1>@@TITLE@@</h1><p class="muted">기존 YOLO26n의 공간 특징과 예측 점을 함께 쓰는 팔레트 측면선 모델. 실제 저장된 결과만 표시합니다.</p><div class="notice"><b>@@HEADLINE@@</b><p>@@REASONS@@</p></div><div class="cards"><div class="card"><b>55,980</b><span>기존 R0와 동일한 합성 학습 이미지</span></div><div class="card"><b>@@TRAINED@@ / 9</b><span>3 대조군 × 3 seeds · 각 6,000 steps</span></div><div class="card"><b>319 + 2,689</b><span>실사 positive + negative · 재사용 DEV</span></div><div class="card"><b>@@EVALUATED@@ / 9</b><span>새 모델의 논문 계약 평가 완료</span></div></div>
<section><h2>실사 2D · 6D 결과</h2><p class="muted">새 모델은 완료된 seed의 평균 ± 표본 표준편차(ddof=1)입니다. 같은 프레임의 반복 예측이므로 seed를 독립 이미지로 세지 않습니다.</p>@@MAIN_TABLE@@<p class="foot">2D: IoU≥0.5로 매칭된 최고 점수 검출의 감독되는 9점. 점 confidence 임계값 없음. 6D: 기존 MAIN 선택기와 geometry-reconstructed GT reference를 그대로 사용합니다. 이전 52장·8점 실험 수치는 이 표에 포함하지 않습니다.</p><details><summary>개별 seed 및 세션 bootstrap</summary>@@SEEDS@@@@CI_TABLE@@</details><details><summary>촬영 세션별 결과</summary>@@SESSIONS@@</details></section>
<section><h2>무엇을 학습하고 합치는가</h2>@@ARCHITECTURE@@<p class="foot">기하 입력 대조군은 영상 특징을 0으로 두고 같은 예산으로 학습합니다. 선 손실 대조군은 corner 손실을 제외합니다. R0는 frozen이며 새 분기 학습은 R0의 기존 60epoch 학습 외에 추가한 계산입니다. GT는 학습 loss target과 사후 평가에만 사용합니다.</p><details><summary>학습 진행 상황</summary><div class="matrix">@@MATRIX@@</div></details></section>
<section><h2>합성 데이터에서 고정한 선택</h2><p class="muted">Calibration 1,004장 → selection 1,031장 → heldout 1,985장. P0/TEX 같은 시나리오는 함께 분할했습니다. 실사 결과로 T·λ·이동 상한을 다시 맞추지 않습니다.</p>@@SELECTION_TABLE@@<details><summary>선택 동결 후 합성 heldout 결과</summary>@@SYNTHETIC_TABLE@@<p class="foot">새 분기의 heldout입니다. R0와 이전 probe가 사용한 기존 validation에서 분리했으므로 완전히 새로운 독립 검증 데이터는 아닙니다.</p></details></section>
<section id="gallery"><h2>같은 이미지에서 바뀐 점과 선</h2><p class="muted">도움·악화·중립 순위는 정답을 사용한 사후 설명입니다. 최고 점수 검출의 IoU≥0.5 매칭 프레임에서 감독되는 9점의 이미지별 평균 오차 차이로 정렬합니다. 운영 중 사용할 gate가 아닙니다.</p><div class="controls"><label>모델<select id="arm"></select></label><label>Seed<select id="seed"><option>1</option><option>2</option><option>3</option></select></label><label>순위<select id="rank"><option value="best">가장 도움된 순서</option><option value="worst">가장 악화된 순서</option><option value="neutral">변화가 작은 순서</option><option value="all">전체 프레임 ID순</option></select></label><label>세션<select id="session"><option value="all">전체</option></select></label><label>이미지<select id="frame"></select></label><label>선 역할<select id="role"></select></label></div><div id="frameinfo" class="notice"></div><div class="legend"><span><i style="background:#63e6a0"></i>GT</span><span><i style="background:#59d6ee"></i>R0 점</span><span><i style="background:#f48cda"></i>새 점</span><span><i style="background:#ffb45c"></i>실제 최고확률 선 후보의 샘플</span></div><div class="panels"><div class="panel"><h3>① 감독되는 GT · 측면선</h3><canvas id="gt"></canvas></div><div class="panel"><h3>② YOLO26n R0</h3><canvas id="before"></canvas></div><div class="panel"><h3>③ 새 모델 · 이동 화살표</h3><canvas id="after"></canvas></div><div class="panel"><h3>④ 실제 입력 좌표 · 선택한 선 역할</h3><canvas id="evidence"></canvas></div></div><div id="evidenceinfo"></div><p class="foot">④는 반사 패딩·letterbox를 포함한 실제 입력입니다. 저장된 최고확률 후보선과 예측 끝점으로 32개 샘플 위치를 재구성합니다. 분포·coverage는 실제 모델 출력의 요약이며 attention, Grad-CAM 또는 인과적 중요도 지도가 아닙니다. Amodal 측면선은 물리적으로 보이는 edge라는 뜻이 아닙니다. λ=0이면 분기를 우회하므로 선 증거를 만들지 않습니다.</p></section>
<section><h2>실제 추론시간과 보존 범위</h2>@@RUNTIME_TABLE@@<p class="foot">원본 BGR 이후 실제 전체 경로의 동기화 wall time입니다. 공통 26장(13세션×2)×3반복, batch1이며 이미지 파일 decode·PnP는 제외합니다. λ=0은 분기를 우회합니다. 상자·점수·다른 후보·centroid는 보존하며 negative 평가는 기존 candidate 복사와 해시 검증으로 유지합니다. 새 negative forward 성능이나 처리속도를 측정한 것은 아닙니다.</p></section>
<section><h2>해석 범위와 원본 근거</h2><ul><li>319 positive / 2,689 negative는 반복 사용한 DEV입니다. 독립 final test나 다른 현장으로의 일반화를 입증하지 않습니다.</li><li>6D GT는 기존 geometry reconstruction 계약을 사용합니다. 실제 물리 측정 정확도로 확대 해석하지 않습니다.</li><li>합성 데이터는 G38 + P0/TEX이며 새 head의 추가 학습 예산을 포함합니다. 이미지 특징의 효과는 같은 예산의 기하 입력·선 손실 대조군과 함께 판단합니다.</li><li>기존 DHT 데이터 계보 중복은 SOURCE_DATA_AUDIT에 공개했습니다. 보이지 않는 선의 오분류, 초기 점 후보 범위와 점·선 오류의 상관은 여전히 한계입니다.</li></ul><p>@@LINKS@@</p><p class="foot">보고서는 저장된 JSON·예측·GT만 읽으며 모델 추론, 선택 규칙 변경, 새 학습을 수행하지 않습니다. @@INPUT_COUNT@@개 입력 파일의 해시는 REPORT_RENDER.json에 기록됩니다.</p></section></main>
<script id="report-data" type="application/json">@@DATA@@</script><script>
const DATA=JSON.parse(document.getElementById('report-data').textContent), E=DATA.edges, labels=DATA.labels;
const $=id=>document.getElementById(id), finite=p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite), f=(x,n=2)=>x==null?'—':Number(x).toFixed(n), imgCache=new Map();let available=[];
function option(el,value,text){const o=document.createElement('option');o.value=value;o.textContent=text;el.append(o)}
Object.keys(labels).filter(x=>x!=='R0').forEach(k=>option($('arm'),k,labels[k]));
[...new Set(DATA.frames.map(x=>x.session))].sort().forEach(x=>option($('session'),x,x));E.forEach((e,i)=>option($('role'),i,(i<4?'높이':'깊이')+' '+i+' · '+e.join('–')));
function image(src){if(!imgCache.has(src)){imgCache.set(src,new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=reject;i.src=src}));}return imgCache.get(src)}
function line(ctx,a,b,color,width=2){if(!finite(a)||!finite(b))return;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
function points(ctx,ps,color,edges=true,alpha=1){if(!ps)return;ctx.save();ctx.globalAlpha=alpha;if(edges)E.forEach(([a,b])=>line(ctx,ps[a],ps[b],color,1.6));ps.forEach((p,i)=>{if(!finite(p))return;ctx.beginPath();ctx.arc(...p,i===8?5:4,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.font='bold 12px Arial';ctx.lineWidth=3;ctx.strokeStyle='#081321';ctx.strokeText(String(i),p[0]+6,p[1]-6);ctx.fillText(String(i),p[0]+6,p[1]-6)});ctx.restore()}
function arrow(ctx,a,b){if(!finite(a)||!finite(b)||Math.hypot(a[0]-b[0],a[1]-b[1])<.1)return;line(ctx,a,b,'#ffdd97',2);const t=Math.atan2(b[1]-a[1],b[0]-a[0]);line(ctx,b,[b[0]-7*Math.cos(t-.5),b[1]-7*Math.sin(t-.5)],'#ffdd97',2);line(ctx,b,[b[0]-7*Math.cos(t+.5),b[1]-7*Math.sin(t+.5)],'#ffdd97',2)}
function baseCanvas(id,im,w,h){const c=$(id);c.width=w;c.height=h;const ctx=c.getContext('2d');ctx.drawImage(im,0,0,w,h);return ctx}
function infinite(ctx,h,w,height){const [a,b,c]=h,out=[];if(Math.abs(b)>1e-10){for(const x of [0,w]){const y=-(a*x+c)/b;if(y>=0&&y<=height)out.push([x,y])}}if(Math.abs(a)>1e-10){for(const y of [0,height]){const x=-(b*y+c)/a;if(x>=0&&x<=w)out.push([x,y])}}if(out.length>=2)line(ctx,out[0],out[out.length-1],'#ffb45c',2)}
function reset(){const key=$('arm').value+'_seed'+$('seed').value;available=DATA.frames.filter(x=>x.runs[key]&&($('session').value==='all'||x.session===$('session').value));const mode=$('rank').value;if(mode!=='all')available=available.filter(x=>x.runs[key].delta!=null);available.sort((a,b)=>mode==='all'?a.id.localeCompare(b.id):mode==='best'?a.runs[key].delta-b.runs[key].delta:mode==='worst'?b.runs[key].delta-a.runs[key].delta:Math.abs(a.runs[key].delta)-Math.abs(b.runs[key].delta));$('frame').replaceChildren();available.forEach((x,i)=>option($('frame'),i,x.id));render()}
async function render(){window.REPORT_READY=false;const frame=available[Number($('frame').value)||0],key=$('arm').value+'_seed'+$('seed').value;if(!frame){$('frameinfo').textContent='완료된 새 모델 예측이 없습니다. 이 공간은 실제 평가 결과가 생성된 뒤 채워집니다.';$('evidenceinfo').replaceChildren();['gt','before','after','evidence'].forEach(id=>{const c=$(id);c.width=640;c.height=360;c.getContext('2d').clearRect(0,0,640,360)});window.REPORT_READY=true;return}
const run=frame.runs[key],[raw,input]=await Promise.all([image(frame.image),image(frame.input_image)]);$('frameinfo').textContent=frame.id+' · '+frame.session+' / '+frame.domain+' · 감독 '+run.n_supervised+'점 · 평균오차 '+f(run.baseline_error)+' → '+f(run.new_error)+'px · Δ '+f(run.delta)+'px · λ='+run.lam+' · '+run.status;
let ctx=baseCanvas('gt',raw,frame.width,frame.height);points(ctx,frame.gt,'#63e6a0');ctx=baseCanvas('before',raw,frame.width,frame.height);points(ctx,frame.gt,'#63e6a0',true,.42);points(ctx,run.before,'#59d6ee');ctx=baseCanvas('after',raw,frame.width,frame.height);points(ctx,frame.gt,'#63e6a0',true,.42);points(ctx,run.after,'#f48cda');if(run.before&&run.after)run.before.forEach((p,i)=>arrow(ctx,p,run.after[i]));
const tf=frame.transform,shape=tf.input_shape_hw;ctx=baseCanvas('evidence',input,shape[1],shape[0]);const d=run.diagnostics,role=Number($('role').value);$('evidenceinfo').replaceChildren();if(d&&d.peak_line_h_supplied_image){const r=tf.gain,pad=tf.pad_ltrb,p=(run.before||[]).map(q=>finite(q)?[(q[0]+100)*r+pad[0],(q[1]+100)*r+pad[1]]:null);points(ctx,p,'#59d6ee',false,.6);if(d.line_valid[role]){const h=d.peak_line_h_supplied_image[role],hi=[h[0],h[1],r*h[2]-h[0]*(100*r+pad[0])-h[1]*(100*r+pad[1])];infinite(ctx,hi,shape[1],shape[0]);const [ai,bi]=E[role],a=p[ai],b=p[bi];if(finite(a)&&finite(b)){const mid=[(a[0]+b[0])/2,(a[1]+b[1])/2],res=hi[0]*mid[0]+hi[1]*mid[1]+hi[2],base=[mid[0]-res*hi[0],mid[1]-res*hi[1]],len=Math.hypot(a[0]-b[0],a[1]-b[1]);for(let j=0;j<32;j++){const t=j/31-.5,x=base[0]-hi[1]*len*t,y=base[1]+hi[0]*len*t;ctx.beginPath();ctx.arc(x,y,2.6,0,2*Math.PI);ctx.fillStyle='#ffb45c';ctx.fill()}}}
const table=document.createElement('table'),head=document.createElement('tr');['역할','후보 peak 확률','null 확률','조건부 entropy','가중 샘플 coverage','peak coverage'].forEach(t=>{const th=document.createElement('th');th.textContent=t;head.append(th)});table.append(head);E.forEach((edge,i)=>{const tr=document.createElement('tr');[i+' · '+edge.join('–'),f(100*d.peak_probability_conditional[i],2)+'%',f(100*d.null_probability[i],2)+'%',f(d.entropy_normalized[i],3),f(100*d.expected_sample_coverage[i],1)+'%',f(100*d.peak_sample_coverage[i],1)+'%'].forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.append(td)});if(i===role)tr.style.background='#273b4d';table.append(tr)});const scroll=document.createElement('div');scroll.className='scroll';scroll.append(table);$('evidenceinfo').append(scroll)}else{const p=document.createElement('p');p.className='muted';p.textContent='이 프레임에서는 선 분기를 사용하지 않았거나 저장된 선 진단이 없습니다. 가상의 evidence를 표시하지 않습니다.';$('evidenceinfo').append(p)}window.REPORT_READY=true}
['arm','seed','rank','session'].forEach(id=>$(id).addEventListener('change',reset));['frame','role'].forEach(id=>$(id).addEventListener('change',render));reset();window.REPORT_FRAME_COUNT=DATA.frames.length;
</script></body></html>'''


def render(run_dir, screenshot=True):
    if not (run_dir / "PURPOSE.md").is_file():
        raise ValueError("purpose-declared result directory required")
    inputs = Inputs(run_dir)
    data = collect(inputs)
    values = sections(data)
    verdict = data["verdict"]
    headline = verdict.get("headline_ko") if data["complete"] else "실험 진행 중 · 새 모델의 개선 여부는 아직 검증되지 않았습니다"
    reasons = verdict.get("reasons", []) if data["complete"] else ["학습, 합성 선택, 실사 평가와 통계가 모두 끝난 뒤 판정합니다."]
    if isinstance(reasons, str):
        reasons = [reasons]
    matrix = ''.join(f'<div><strong>{LABELS[r["arm"]]} · seed {r["seed"]}</strong>{r["state"]} · {r["step"]:,}/6,000 steps</div>' for r in data["progress"])
    links = []
    for name in ("SUMMARY.json", "VERDICT.json", "SELECTION.json", "SYNTHETIC_EVALUATION.json", "RUNTIME.json", "TRAIN_PROTOCOL.json", "SOURCE_DATA_AUDIT.json", "INFERENCE_CONTRACT_QA.json"):
        path = run_dir / name
        if path.is_file():
            links.append(f'<a href="{html.escape(name)}">{html.escape(name)}</a>')
    bindings_before = len(inputs.hashes)
    replacements = {**{k.upper(): v for k, v in values.items()}, "TITLE": html.escape(TITLE),
        "HEADLINE": html.escape(headline or "평가 완료"), "REASONS": "<br>".join(html.escape(str(r)) for r in reasons),
        "TRAINED": str(sum(r["state"] in ("TRAINED", "EVALUATED") for r in data["progress"])),
        "EVALUATED": str(len(data["runs"])), "ARCHITECTURE": ARCHITECTURE, "MATRIX": matrix,
        "LINKS": " · ".join(links), "INPUT_COUNT": str(bindings_before),
        "DATA": json.dumps(clean(dict(frames=data["frames"], edges=EDGES, labels=LABELS)),
                           ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")}
    document = TEMPLATE
    for key, value in replacements.items():
        document = document.replace(f"@@{key}@@", value)
    if "@@" in document or '<script src=' in document or '<link ' in document:
        raise ValueError("unresolved template or external rendering dependency")
    # All gallery image sources are embedded; source links are local artifacts.
    for frame in data["frames"]:
        if not frame["image"].startswith("data:image/") or not frame["input_image"].startswith("data:image/"):
            raise ValueError("nonembedded gallery image")
    destination = run_dir / "index.html"
    destination.write_text(document)
    render_sha = sha(destination)
    qa = dict(PASS=True, html_sha256=render_sha, scope="automatic document/input checks; not human visual review",
              title_matches=True, unresolved_placeholders=0, external_render_dependencies=0,
              broken_local_images=0, embedded_gallery_frames=len(data["frames"]),
              browser_screenshot="not_attempted", experiment_complete=data["complete"])
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if screenshot and chrome:
        try:
            with tempfile.TemporaryDirectory(prefix="pallet-line-report-chrome-") as profile:
                result = subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu",
                    "--disable-dev-shm-usage", f"--user-data-dir={profile}", "--window-size=1560,1100",
                    "--virtual-time-budget=3500", f"--screenshot={run_dir / 'report.png'}", destination.as_uri()],
                    capture_output=True, text=True, timeout=25)
                qa["browser_screenshot"] = "saved" if result.returncode == 0 and (run_dir / "report.png").is_file() else "unavailable"
                qa["browser_returncode"] = result.returncode
        except (subprocess.TimeoutExpired, OSError) as exc:
            qa["browser_screenshot"] = "unavailable"
            qa["browser_note"] = str(exc)
    (run_dir / "VISUAL_QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n")
    receipt = dict(complete=True, PASS=True, html=str(destination), html_sha256=render_sha,
                   input_sha256=inputs.hashes, title=TITLE, experiment_complete=data["complete"],
                   renderer_sha256=sha(Path(__file__)), n_completed_evaluations=len(data["runs"]),
                   n_gallery_frames=len(data["frames"]), automatic_qa=str(run_dir / "VISUAL_QA.json"),
                   screenshot=str(run_dir / "report.png") if qa["browser_screenshot"] == "saved" else None,
                   no_inference=True, no_auto_open=True)
    (run_dir / "REPORT_RENDER.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "input_sha256"}, ensure_ascii=False, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--no-screenshot", action="store_true")
    args = parser.parse_args()
    render(args.run_dir.resolve(), screenshot=not args.no_screenshot)


if __name__ == "__main__":
    main()
