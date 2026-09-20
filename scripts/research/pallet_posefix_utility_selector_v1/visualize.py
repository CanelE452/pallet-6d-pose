"""Offline inspection of the learned selector after all real outputs freeze."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import core as P
from .model import PAIRS
from scripts.research.pallet_posefix_corner_gate_v1.visualize import (
    BG, CASES, draw, limits, table, top, valid_points, verify_metric,
)


ARMS = ("A_N2", "REPLAY", "CORNER_FULL_PAIR", "LEARNED_PAIR")
PLOT_ARMS = ("A_N2", "REPLAY", "LEARNED_PAIR")
LABELS = {"A_N2": "현재 최종 N2", "REPLAY": "Replay 전체 보정",
          "CORNER_FULL_PAIR": "이전 수동 규칙 필터", "LEARNED_PAIR": "학습된 보정 선택기"}
PLOT_LABELS = {"A_N2": "Current N2", "REPLAY": "Replay: all corrections", "LEARNED_PAIR": "Learned utility selector"}


def corner_accept(decision):
    result = [False] * 8
    for accept, pair in zip(decision["pair_accept"], PAIRS):
        for corner in pair:
            result[corner] = bool(accept)
    return result


def figure(case, row, scores, decision, record, groups, edges):
    tag, dataset, ident, canonical, description = case
    green = dataset == "GREEN150"
    for key in ("image", "annotation"):
        P.verify_binding(row[key])
    image = np.asarray(Image.open(P.ROOT / row["image"]["path"]).convert("RGB"))
    assert list(image.shape[:2]) == row["raw_hw"]
    entries = P.read(P.ROOT / row["annotation"]["path"])["objects"][0]["keypoint_annotations"]
    gt = np.asarray([r["xy"] if r.get("xy") is not None else [np.nan, np.nan] for r in entries], float)
    valid = np.asarray(scores["A_N2"]["canonical_valid"], bool)
    if green:
        manual = np.array([r.get("source") == "manual_click" and r.get("visibility", 0) != 0 for r in entries])
        manual &= valid_points(gt)
        np.testing.assert_array_equal(manual[:8], valid)
    else:
        assert all(r.get("source", "unknown") == "unknown" for r in entries)
    kind = "plastic_standard_110x110x15" if green else record["object_type"]
    permutations = np.asarray(groups[kind], int)
    points = {arm: np.asarray(top(row["predictions"][arm])["keypoints_xy"], float) for arm in PLOT_ARMS}
    models = {}
    for arm in PLOT_ARMS:
        metric = scores[arm]
        np.testing.assert_array_equal(metric["canonical_valid"], valid)
        verify_metric(points[arm], gt, valid, permutations, metric, row["raw_hw"])
        permutation = permutations[metric["branch"]]
        native = int(np.flatnonzero(permutation == canonical)[0])
        models[arm] = dict(native_corner=native, canonical_corner=canonical,
            whole_object_branch=metric["branch"], whole_object_permutation=permutation.tolist(),
            predicted_xy=points[arm][native].tolist(), focus_error_px=metric["canonical_errors"][canonical],
            frame_mean_px=metric["frame_mean_px"])
    object_extent = limits(np.concatenate([gt[:8][valid], *[points[a][:8] for a in PLOT_ARMS]]), row["raw_hw"])
    zoom_extent = limits([gt[canonical], *[points[a][models[a]["native_corner"]] for a in PLOT_ARMS]], row["raw_hw"], 26, 135)
    accepted = corner_accept(decision)
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 10), facecolor=BG)
    for col, arm in enumerate(PLOT_ARMS):
        info = models[arm]
        for view, extent in enumerate((object_extent, zoom_extent)):
            draw(axes[view, col], image, points[arm], points["A_N2"], gt, valid, edges, extent,
                 info["native_corner"], canonical, green, accepted if arm == "LEARNED_PAIR" else None)
            title = (f"{PLOT_LABELS[arm]}\nframe reference error: {info['frame_mean_px']:.2f} px" if view == 0
                     else f"{'G' if green else 'L'}{canonical} / P{info['native_corner']} / whole-object b{info['whole_object_branch']}\nfocus error: {info['focus_error_px']:.2f} px")
            axes[view, col].set_title(title, color="white", fontsize=11, pad=8)
    fig.suptitle(f"{'GREEN manual reference' if green else 'NON-GREEN unknown-origin reference'}\n{ident}",
                 color="white", fontsize=14, y=.985)
    legend = ("Magenta x = manual GT only; no PnP-derived target is drawn." if green
              else "Gray hollow circle = UNKNOWN-origin legacy reference, NOT verified manual GT.")
    fig.text(.5, .015, legend + "\nYellow lines = model output; white arrows = same native point from N2.\n"
             "Learned panel: lime = accepted Replay, cyan = exact N2 fallback; no hand-written geometry/flip gate.\n"
             "Identical RGB/crop in each row. GT displayed/scored only after all 222 learned selections were frozen.",
             ha="center", va="bottom", color="white", fontsize=9)
    fig.subplots_adjust(left=.015, right=.985, top=.875, bottom=.125, wspace=.035, hspace=.23)
    path = P.OUT / f"{tag}.png"
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)
    focus_native = models["LEARNED_PAIR"]["native_corner"]
    pair_index = next(i for i, pair in enumerate(PAIRS) if focus_native in pair)
    return dict(tag=tag, dataset=dataset, id=ident, description=description,
        reference_kind="manual_only" if green else "legacy_unknown_origin", canonical_focus=canonical,
        reference_xy=gt[canonical].tolist(), image=row["image"], annotation=row["annotation"],
        object_crop=object_extent, zoom_crop=zoom_extent, models=models,
        utility=decision["utility"], valid=decision["valid"], pair_accept=decision["pair_accept"],
        corner_accept=accepted, focus_native=focus_native, focus_pair_index=pair_index, output=P.bound(path))


def number(value, decimals=3):
    return "계산 불가" if value is None else f"{value:.{decimals}f}"


def source_section(source, fit):
    rows = []
    performance = []
    for key, title in (("train", "합성 TRAIN"), ("validation", "합성 VALIDATION")):
        summary = source["summary"][key]
        rows.append([title, summary["samples"], summary["supported_corners"],
            f"{100*summary['sign_accuracy']:.2f}%", f"{100*summary['majority_sign_accuracy']:.2f}%",
            number(summary["ROC_AUC"]), number(summary["utility_MAE"]), summary["target_positive"], summary["predicted_positive"]])
        for variant in ("clean", "stress"):
            values = summary[variant]
            for arm in ("N2", "Replay", "Learned"):
                metric = values[arm]
                performance.append([title, "원래 입력" if variant == "clean" else "입력 좌표 교란", arm,
                    f"{100*metric['PCK10']:.3f}%", number(metric['mean_px']), number(metric['P90_px']), metric["corners"]])
    return ('<section><h2>선택기 학습·합성 검증 진단</h2>' + (
        f'<p>새 선택기만 1,500 step 학습했습니다. 고정된 학습 입력 확인용 MAE: '
        f'{fit["training_probe_MAE_before"]:.3f} → {fit["training_probe_MAE_after"]:.3f} (% box diagonal).</p>') + table(
        ["분할", "입력 수", "유효 코너", "부호 정확도", "다수 부호 기준", "ROC AUC", "Utility MAE", "실제 양수", "예측 양수"], rows)
        + '<p class="muted">부호는 보정 이득이 0보다 큰지입니다. AUC는 이득·악화의 순위 구분 진단이며 실제 보정 정책은 고정된 0 기준을 사용합니다. '
          'VALIDATION은 시나리오 단위로 분리됐지만 기존 후보 모델의 과거 개발 데이터입니다. 실사 학습 0장, C4 학습 0장입니다.</p>'
        + table(["분할", "입력", "후보/선택", "PCK10", "평균 px", "P90 px", "코너 수"], performance)
        + '<p class="muted">합성 표는 N2 기준으로 한 번 고른 전체 대칭 분기를 모든 후보에 공통 적용한 진단입니다. 실사 표의 모델별 전체 대칭 평가와 조건이 다릅니다. '
          '합성 데이터에 이전 수동 필터는 다시 적용하지 않았습니다.</p></section>')


def html_report(results, source, fit, figures, path):
    status, sections = [], []
    for dataset, title in (("GREEN150_MANUAL", "초록 150장 · 수동 정답 681점"), ("DEV72", "비초록 72장 · 출처 미확인 참조점 555점")):
        summaries, comparisons = results["summary"][dataset], results["comparisons"][dataset]
        selected = summaries["LEARNED_PAIR"]
        versus_n2 = 100 * (selected["PCK"]["10"] - summaries["A_N2"]["PCK"]["10"])
        versus_gate = 100 * (selected["PCK"]["10"] - summaries["CORNER_FULL_PAIR"]["PCK"]["10"])
        comparison = comparisons["LEARNED_PAIR"]
        status.append(f'<li><b>{html.escape(title)}</b>: 학습 선택기 PCK10 {100*selected["PCK"]["10"]:.3f}%, '
            f'현재 N2 대비 {versus_n2:+.3f}%p, 이전 수동 필터 대비 {versus_gate:+.3f}%p. '
            f'Replay 훼손 {comparison["raw_Replay_damages"]}개 중 {comparison["raw_Replay_damages_prevented"]}개 차단, '
            f'좋은 보정 {comparison["raw_Replay_gains"]}개 중 {comparison["raw_Replay_gains_retained"]}개 유지.</li>')
        rows = []
        for arm in ARMS:
            s = summaries[arm]
            rows.append([LABELS[arm], f"{100*s['PCK']['10']:.3f}%", f"{s['correct10']}/{s['corners']}",
                         number(s["matched_mean_px"]), number(s["matched_pooled_corner8_median_px"]), number(s["matched_pooled_corner8_P90_px"])])
        preservation = []
        for arm in ("CORNER_FULL_PAIR", "LEARNED_PAIR"):
            c = comparisons[arm]
            preservation.append([LABELS[arm], f"{c['raw_Replay_damages_prevented']}/{c['raw_Replay_damages']}",
                c["raw_Replay_damages_remaining"], f"{c['raw_Replay_gains_retained']}/{c['raw_Replay_gains']}",
                c["raw_Replay_gains_lost"], c["gain_over_N2"], c["damage_over_N2"], c["symmetry_branch_changes"]])
        sections.append(f'<section><h2>{html.escape(title)}</h2>'
            + table(["고정 비교군", "PCK10", "정확점 / 전체", "매칭 평균 px", "매칭 중앙값 px", "매칭 P90 px"], rows)
            + '<p class="muted">정확도는 원래 전체 분모와 검출 실패 벌점을 유지합니다. 평균·중앙값·P90은 매칭된 점 기준입니다.</p>'
            + table(["선택 방법", "Replay 훼손 차단", "훼손 잔존", "좋은 보정 유지", "좋은 보정 손실", "N2 대비 복구", "N2 대비 훼손", "평가 대칭 분기 변경"], preservation)
            + '</section>')
    cards = []
    for item in figures:
        image_uri = "data:image/png;base64," + base64.b64encode((P.ROOT / item["output"]["path"]).read_bytes()).decode("ascii")
        model_rows = []
        for arm in PLOT_ARMS:
            value = item["models"][arm]
            model_rows.append([LABELS[arm], f"P{value['native_corner']} ↔ {'G' if item['dataset']=='GREEN150' else 'L'}{item['canonical_focus']}",
                               f"b{value['whole_object_branch']}", f"{value['focus_error_px']:.2f}px", f"{value['frame_mean_px']:.2f}px"])
        pair_rows = []
        for index, pair in enumerate(PAIRS):
            a, b = pair
            pair_rows.append([f"P{a}–P{b}" + (" · 초점" if index == item["focus_pair_index"] else ""),
                f"{item['utility'][a]:+.4f}", f"{item['utility'][b]:+.4f}",
                "둘 다 유효" if item["valid"][a] and item["valid"][b] else "무효 코너 있음",
                "Replay 적용" if item["pair_accept"][index] else "N2 그대로 유지"])
        cards.append(f'<section id="{item["tag"]}" class="case"><h2>{html.escape(item["description"])}</h2>'
            f'<p class="mono">{html.escape(item["id"])}</p><img src="{image_uri}" alt="학습 선택기 비교 {html.escape(item["description"])}" loading="lazy">'
            + table(["모델", "동일 참조점의 native 대응", "전체 대칭 분기", "선택 코너 오차", "사진 평균 오차"], model_rows)
            + '<h3>모델이 예측한 보정 이득</h3>' + table(["연결 코너쌍", "첫 점 utility", "둘째 점 utility", "입력 유효성", "실제 적용"], pair_rows)
            + '<p class="muted">Utility는 예측된 오차 감소량(% box diagonal)이며 확률이 아닙니다. 두 끝점이 모두 유효하고 두 utility가 모두 0보다 클 때만 '
              'Replay 좌표를 복사합니다. 그 외에는 N2를 유지합니다. 이 결정에는 기하·반전 수동 필터나 평가 GT를 사용하지 않았습니다.</p></section>')
    navigation = ''.join(f'<a href="#{item["tag"]}">{html.escape(item["description"])}</a>' for item in figures)
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>팔레트 · 학습된 보정 선택기 결과</title><style>
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#0a121a;color:#e7edf3;font-family:system-ui,"Noto Sans CJK KR",sans-serif;line-height:1.6}main{max-width:1500px;margin:auto;padding:28px}h1{font-size:30px;line-height:1.3}h2{margin:0 0 12px;font-size:22px}h3{font-size:17px}section{background:#121e2a;border:1px solid #2a3c4f;border-radius:14px;padding:23px;margin:24px 0}p{margin:10px 0}.muted{color:#aab8c7;font-size:14px}.warning{background:#332819;border:1px solid #795728;padding:17px;border-radius:12px}.status{border-left:4px solid #a6ff4a;padding-left:17px}.mono{font-family:ui-monospace,monospace;overflow-wrap:anywhere;font-size:13px}nav{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0}nav a{color:#cceaff;text-decoration:none;border:1px solid #365777;background:#172b3f;border-radius:20px;padding:7px 13px;font-size:14px}table{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}td,th{text-align:right;padding:10px 12px;border-bottom:1px solid #304153}td:first-child,th:first-child{text-align:left}th{color:#aecbdf;background:#192a3a}.tablewrap{overflow:auto;margin:12px 0}.case img{display:block;width:100%;height:auto;border-radius:8px}.status ul{margin:8px 0;padding-left:23px}footer{color:#93a7b8;font-size:13px;margin:28px 0}b{font-weight:650}@media(max-width:700px){main{padding:13px}section{padding:14px}h1{font-size:25px}}
</style><main><h1>학습으로 어떤 코너를 보정할지 골랐습니다</h1>
<p>두 후보는 그대로 두고, 작은 이미지 기반 모델만 새로 학습했습니다. 출력은 기존 N2 또는 Replay 좌표 중 하나이며 새로운 좌표를 생성하지 않습니다.</p>
<div class="warning"><b>최종 모델·라벨 변경 없음</b><br>학습에는 분리된 합성 데이터의 정답만 사용했습니다. 실사 평가 GT는 선택기에 입력하지 않았고, 222장 선택을 모두 저장한 뒤 평가·그림에만 사용했습니다.
<br><b>한계:</b>실사 학습 0장, 정사각 C4 학습 0장. 과거 개발 데이터 재사용으로 독립 확인이 아닙니다. 시드 1개·고정 마지막 1,500 step·양 끝점 utility &gt; 0 정책입니다. 결과를 보고 임계값이나 모델을 다시 고르지 않았습니다.
<br>전체 영상 대신 검출 상자 기반 crop을 보므로 잘못되거나 작은 검출 상자는 여전히 한계입니다. 이 페이지는 승격 또는 최종 모델 교체 선언이 아닙니다.</div>
<section class="status"><h2>자동 집계 요약</h2><ul>''' + ''.join(status) + '''</ul></section><nav>''' + navigation + '</nav>' + ''.join(sections) + source_section(source, fit) + ''.join(cards) + '''<footer>모든 그림이 HTML에 내장되어 인터넷 없이 열립니다. 원본 RGB와 저장된 좌표만 사용했습니다.<br>자홍색 × = 초록 수동 GT · 회색 빈 원 = 비초록 출처 미확인 기존 참조점 · 노란 선 = 출력 · 연두 점 = Replay 적용 · 하늘색 점 = N2 유지.<br>P는 모델 native 채널, G/L은 참조점 번호입니다. 모델별 승인된 전체 물체 대칭 분기 하나로 대응하며 점별 최근접 GT를 사용하지 않습니다.</footer></main></html>'''
    path.write_text(document, encoding="utf-8")


def main():
    required = [P.RAW / name for name in ("PREDICTIONS.json", "PER_FRAME_METRICS.json", "DECISIONS.json")]
    required += [P.DOC / name for name in ("RESULTS.json", "SOURCE_RESULTS.json", "FIT.json", "OUTPUTS_LOCK.json")]
    for path in required:
        assert path.exists(), f"Wait for frozen learned evaluation: {path}"
    lock = P.read(P.DOC / "OUTPUTS_LOCK.json")
    assert lock["complete"] and lock["frames"] == 222 and not lock["GT_read_for_selection"]
    for binding in lock["artifacts"]:
        P.verify_binding(binding)
    results, source, fit = (P.read(P.DOC / name) for name in ("RESULTS.json", "SOURCE_RESULTS.json", "FIT.json"))
    assert results["complete"] and source["complete"] and fit["complete"]
    assert not results["thresholds_retuned"] and not results["final_model_modified"]
    predictions = {ds: {r["id"]: r for r in rows} for ds, rows in P.read(required[0]).items()}
    metrics = {ds: {arm: {r["id"]: r for r in rows} for arm, rows in arms.items()} for ds, arms in P.read(required[1]).items()}
    decisions = {ds: {r["id"]: r for r in rows} for ds, rows in P.read(required[2]).items()}
    split_path = P.ROOT / "_docs/experiments/pallet_large_error_refiner_v1/SPLIT.json"
    records = {r["id"]: r for r in P.read(split_path)["evaluation"]}
    contract_path = P.ROOT / "_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json"
    contract = P.read(contract_path)
    groups = {o["object_type"]: o["permutations"] for o in contract["objects"]}
    P.OUT.mkdir(parents=True, exist_ok=True)
    figures = []
    for case in CASES:
        _, ds, ident, _, _ = case
        metric_ds = "GREEN150_MANUAL" if ds == "GREEN150" else ds
        scores = {arm: metrics[metric_ds][arm][ident] for arm in PLOT_ARMS}
        figures.append(figure(case, predictions[ds][ident], scores, decisions[ds][ident], records.get(ident), groups, contract["edges"]))
    page = P.OUT / "LEARNED_SELECTOR_REVIEW.html"
    html_report(results, source, fit, figures, page)
    manifest = dict(cases=figures, source_bindings=[P.bound(p) for p in [*required, split_path, contract_path]],
        renderer=P.bound(Path(__file__)), shared_renderer=P.bound(P.ROOT / "scripts/research/pallet_posefix_corner_gate_v1/visualize.py"),
        page=P.bound(page), displayed_prediction_checks=18,
        targets_read_only_after_all_output_lock=True, no_new_inference=True, no_training=True,
        fixed_case_selection="Same six prior diagnostic cases; no result-driven replacement", independent_test=False)
    (P.OUT / "VISUALIZATION_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(page=str(page), figures=[str(P.ROOT / f["output"]["path"]) for f in figures],
                         independent_displayed_checks=18), ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    main()
