"""Offline CPU overlays of the fixed corner-gate cases, after output locking.

The renderer is allowed to read evaluation targets only to display/recheck the
already frozen results.  It neither selects corrections nor changes outputs.
Raster images are original photographs with deterministic coordinate overlays,
not generated or cosmetically edited pictures.
"""

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

from . import protocol as P


ARMS = ("A_N2", "REPLAY", "LEGACY_FRAME", "CORNER_GEOMETRY_PAIR", "CORNER_FULL_PAIR")
PLOT_ARMS = ("A_N2", "REPLAY", "CORNER_FULL_PAIR")
LABELS = {
    "A_N2": "현재 최종 N2", "REPLAY": "Replay 전체 보정",
    "LEGACY_FRAME": "기존 전체사진 필터", "CORNER_GEOMETRY_PAIR": "코너쌍 기하 필터",
    "CORNER_FULL_PAIR": "코너쌍 전체 필터",
}
PLOT_LABELS = {"A_N2": "Current N2", "REPLAY": "Replay: all corrections",
               "CORNER_FULL_PAIR": "Corner-pair gate: full"}
CASES = (
    ("green_worst", "GREEN150", "capture_20260902__009323", 1, "다른 모서리로 크게 이동했던 사례"),
    ("green_truncated", "GREEN150", "capture_20260902_kimjihoon__003249", 2, "잘린 사진의 큰 보정 오류"),
    ("green_nontruncated", "GREEN150", "capture_20260902_kimjihoon__006349", 0, "잘림 없이 확률 평균이 어긋난 사례"),
    ("green_improved", "GREEN150", "capture_20260902_kimjihoon__007309", 4, "Replay의 좋은 보정이 유지되는가"),
    ("other_p4", "DEV72", "eval_pallet07:1778652168786111744", 4, "비초록: P4 개선 사례"),
    ("other_p0", "DEV72", "eval_pallet07:1778652142480077056", 0, "비초록: 비스듬한 P0 사례"),
)
PAIRS = ((0, 3), (1, 2), (4, 7), (5, 6))
BG = "#101923"
YELLOW = "#ffe45b"
MAGENTA = "#ff65ee"
GRAY = "#d4d8dc"
LIME = "#a6ff4a"
CYAN = "#56ddff"
REASONS = {
    "box_confidence": "검출 신뢰도 부족", "too_few_valid_corners": "유효 코너 6개 미만",
    "no_finite_geometry_hypothesis": "기하 가설 계산 불가",
    "invalid_or_low_corner_confidence": "무효 점 또는 코너 신뢰도 부족",
    "geometry": "코너 기하 오차 초과", "flip": "반전 일관성 실패/유효 반전 없음",
    "mode": "평균–최빈 위치 분리 초과",
}


def top(prediction):
    index = prediction["selected_index"]
    assert index is not None
    return prediction["candidates"][index]


def valid_points(points):
    return np.isfinite(points).all(-1) & ~(points == -1).all(-1)


def limits(points, hw, margin=25., minimum=110.):
    h, w = hw
    points = np.asarray(points, dtype=float)
    mask = valid_points(points)
    points = points[mask]
    points = points[(points[:, 0] >= 0) & (points[:, 0] < w)
                    & (points[:, 1] >= 0) & (points[:, 1] < h)]
    assert len(points)
    low, high = points.min(0) - margin, points.max(0) + margin
    center = (low + high) / 2
    half = np.maximum((high - low) / 2, minimum / 2)
    low, high = np.maximum(center - half, [0, 0]), np.minimum(center + half, [w - 1, h - 1])
    return [float(low[0]), float(high[0]), float(low[1]), float(high[1])]


def inside(xy, extent):
    x0, x1, y0, y1 = extent
    return bool(x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1)


def verify_metric(points, gt, valid, permutations, metric, hw):
    assert metric["matched"] and metric["evaluable"]
    means = []
    arrays = []
    for perm in permutations:
        errors = [None] * 8
        for native, canonical in enumerate(perm[:8]):
            if valid[canonical]:
                error = np.linalg.norm(points[native] - gt[canonical]) if valid_points(points)[native] else np.hypot(*hw)
                errors[canonical] = float(error)
        arrays.append(errors)
        means.append(float(np.mean([e for e in errors if e is not None])))
    branch = int(np.argmin(means))
    assert branch == metric["branch"], (branch, metric["branch"])
    assert np.isclose(means[branch], metric["frame_mean_px"], atol=1e-7, rtol=0)
    for actual, saved in zip(arrays[branch], metric["canonical_errors"]):
        assert (actual is None and saved is None) or np.isclose(actual, saved, atol=1e-7, rtol=0)


def focus_decision(diagnostic, native):
    full = diagnostic["full_pair"]
    decision = full["decision"]
    pair_index = next(i for i, pair in enumerate(PAIRS) if native in pair)
    pair = PAIRS[pair_index]
    lines = []
    if full["corner_accept"][native]:
        lines.append(f"P{native}: Replay 보정 적용")
    else:
        lines.append(f"P{native}: 현재 N2 좌표 유지")
    for corner in pair:
        raw_reasons = decision["corner_reasons"][corner]
        if raw_reasons:
            lines.append(f"P{corner}: " + ", ".join(REASONS.get(reason, reason) for reason in raw_reasons))
    if not decision["pair_accept"][pair_index] and not decision["corner_reasons"][native]:
        lines.append(f"P{native} 단독 검사는 통과했지만 연결된 P{pair[1] if pair[0] == native else pair[0]} 때문에 쌍 전체 거부")
    for step in full["recheck_history"]:
        if step["pair_rejected"][pair_index]:
            lines.append(f"혼합 결과 구조 재검사 {step['round']}회차에서 쌍 거부")
    return dict(native_corner=native, pair_index=pair_index, pair=list(pair),
                accepted=bool(full["corner_accept"][native]), explanation=lines,
                endpoint_reasons={str(k): decision["corner_reasons"][k] for k in pair})


def draw(ax, image, points, initial, gt, target_valid, edges, extent, native, canonical,
         is_green, accepted=None):
    ax.imshow(image, interpolation="nearest")
    mask = valid_points(points)
    for a, b in edges:
        if mask[a] and mask[b]:
            ax.plot(points[[a, b], 0], points[[a, b], 1], c=YELLOW, lw=1.5, alpha=.9, zorder=3)
    for index in range(8):
        if not mask[index]:
            continue
        color = YELLOW if accepted is None else (LIME if accepted[index] else CYAN)
        ax.scatter(*points[index], s=35, c=color, edgecolors="#18232b", linewidths=.7, zorder=6)
        if valid_points(initial)[index] and np.linalg.norm(points[index] - initial[index]) > .25:
            arrow = ax.annotate("", xy=points[index], xytext=initial[index],
                                arrowprops=dict(arrowstyle="->", color="white", lw=1.15), zorder=5)
            arrow.arrow_patch.set_clip_path(ax.patch)
    prefix = "G" if is_green else "L"
    for canonical_target in np.flatnonzero(target_valid):
        if inside(gt[canonical_target], extent):
            if is_green:
                ax.scatter(*gt[canonical_target], marker="x", s=75, c=MAGENTA, linewidths=2, zorder=8)
            else:
                ax.scatter(*gt[canonical_target], marker="o", s=76, facecolors="none", edgecolors=GRAY, linewidths=2, zorder=8)
            ax.annotate(f"{prefix}{canonical_target}", xy=gt[canonical_target], xytext=(5, -13),
                        textcoords="offset points", color=MAGENTA if is_green else GRAY, fontsize=9,
                        bbox=dict(facecolor=BG, alpha=.78, edgecolor="none", pad=1), zorder=10)
    if inside(points[native], extent):
        color = YELLOW if accepted is None else (LIME if accepted[native] else CYAN)
        ax.scatter(*points[native], s=155, facecolors="none", edgecolors=color, linewidths=1.6, zorder=9)
        ax.annotate(f"P{native}", xy=points[native], xytext=(8, 9), textcoords="offset points",
                    color=color, fontsize=11, weight="bold", zorder=10,
                    bbox=dict(facecolor=BG, alpha=.85, edgecolor="none", pad=1.5))
    else:
        ax.text(.5, .06, f"Focus P{native} outside shared crop", transform=ax.transAxes,
                ha="center", color="white", bbox=dict(facecolor=BG, alpha=.9, edgecolor="none"))
    x0, x1, y0, y1 = extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")


def figure_case(case, row, scores, diagnostic, record, groups, edges):
    tag, dataset, ident, canonical, description = case
    is_green = dataset == "GREEN150"
    P.verify(row["image"])
    P.verify(row["annotation"])
    image = np.asarray(Image.open(P.ROOT / row["image"]["path"]).convert("RGB"))
    assert list(image.shape[:2]) == row["raw_hw"]
    annotation = P.read(P.ROOT / row["annotation"]["path"])
    entries = annotation["objects"][0]["keypoint_annotations"]
    gt = np.asarray([entry["xy"] if entry.get("xy") is not None else [np.nan, np.nan] for entry in entries], float)
    target_valid = np.asarray(scores["A_N2"]["canonical_valid"], bool)
    if is_green:
        manual = np.array([entry.get("source") == "manual_click" and entry.get("visibility", 0) != 0 for entry in entries])
        manual &= valid_points(gt)
        manual[8] = False
        np.testing.assert_array_equal(manual[:8], target_valid)
    else:
        assert all(entry.get("source", "unknown") == "unknown" for entry in entries)
    assert target_valid[canonical]
    kind = "plastic_standard_110x110x15" if is_green else record["object_type"]
    permutations = np.asarray(groups[kind], dtype=int)
    points = {arm: np.asarray(top(row["predictions"][arm])["keypoints_xy"], float) for arm in PLOT_ARMS}
    model_info = {}
    for arm in PLOT_ARMS:
        metric = scores[arm]
        np.testing.assert_array_equal(metric["canonical_valid"], target_valid)
        verify_metric(points[arm], gt, target_valid, permutations, metric, row["raw_hw"])
        perm = permutations[metric["branch"]]
        native = int(np.flatnonzero(perm == canonical)[0])
        model_info[arm] = dict(native_corner=native, canonical_corner=canonical,
            whole_object_branch=metric["branch"], whole_object_permutation=perm.tolist(),
            predicted_xy=points[arm][native].tolist(), focus_error_px=metric["canonical_errors"][canonical],
            frame_mean_px=metric["frame_mean_px"])
    object_extent = limits(np.concatenate([gt[:8][target_valid], *[points[arm][:8] for arm in PLOT_ARMS]]), row["raw_hw"], 25)
    focus_extent = limits([gt[canonical], *[points[arm][model_info[arm]["native_corner"]] for arm in PLOT_ARMS]],
                          row["raw_hw"], 26, 135)
    prefix = "G" if is_green else "L"
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 10.0), facecolor=BG)
    for col, arm in enumerate(PLOT_ARMS):
        info = model_info[arm]
        accepted = diagnostic["full_pair"]["corner_accept"] if arm == "CORNER_FULL_PAIR" else None
        for panel_row, extent in enumerate((object_extent, focus_extent)):
            ax = axes[panel_row, col]
            draw(ax, image, points[arm], points["A_N2"], gt, target_valid, edges, extent,
                 info["native_corner"], canonical, is_green, accepted)
            if panel_row == 0:
                title = f"{PLOT_LABELS[arm]}\nframe reference error: {info['frame_mean_px']:.2f} px"
            else:
                title = (f"{prefix}{canonical} / P{info['native_corner']} / whole-object b{info['whole_object_branch']}\n"
                         f"focus error: {info['focus_error_px']:.2f} px")
            ax.set_title(title, color="white", fontsize=11, pad=8)
    fig.suptitle(f"{'GREEN manual reference' if is_green else 'NON-GREEN unknown-origin reference'}\n{ident}",
                 color="white", fontsize=14, y=.985)
    reference_legend = "Magenta x = manual GT only; no PnP-derived target is drawn." if is_green else "Gray hollow circle = stored UNKNOWN-origin reference, NOT verified manual GT."
    footer = (reference_legend + "\nYellow lines = model output; white arrows = same native point from N2.\n"
              "Full-gate panel: lime point = accepted Replay, cyan point = exact N2 fallback. Same RGB/crop within each row.\n"
              "Fixed diagnostic examples, reused evaluation. Targets are displayed/scored only after all gates were frozen.")
    fig.text(.5, .015, footer, ha="center", va="bottom", color="white", fontsize=9)
    fig.subplots_adjust(left=.015, right=.985, top=.875, bottom=.125, wspace=.035, hspace=.23)
    path = P.OUT / f"{tag}.png"
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)
    full_native = model_info["CORNER_FULL_PAIR"]["native_corner"]
    return dict(tag=tag, dataset=dataset, id=ident, description=description,
        canonical_focus=canonical, reference_kind="manual_only" if is_green else "legacy_unknown_origin",
        reference_xy=gt[canonical].tolist(), image=row["image"], annotation=row["annotation"],
        object_crop=object_extent, focus_crop=focus_extent, models=model_info,
        focus_gate=focus_decision(diagnostic, full_native),
        all_native_full_accept=diagnostic["full_pair"]["corner_accept"], output=P.bound(path))


def table(headings, rows):
    esc = lambda x: html.escape(str(x))
    return '<div class="tablewrap"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headings) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def html_report(results, figures, path):
    sections = []
    status = []
    for dataset, title in (("GREEN150_MANUAL", "초록 150장 · 수동 정답 681점"), ("DEV72", "비초록 72장 · 출처 미확인 기존 참조점 555점")):
        summaries = results["summary"][dataset]
        comparisons = results["comparisons"][dataset]
        full = comparisons["CORNER_FULL_PAIR"]
        delta = 100 * (summaries["CORNER_FULL_PAIR"]["PCK"]["10"] - summaries["A_N2"]["PCK"]["10"])
        status.append(f"<li><b>{html.escape(title)}</b>: 전체 코너 필터 정확도는 현재 N2 대비 {delta:+.3f}%p. "
                      f"Replay의 훼손 {full['raw_Replay_damages']}개 중 {full['raw_Replay_damages_prevented']}개 차단, "
                      f"좋은 보정 {full['raw_Replay_gains']}개 중 {full['raw_Replay_gains_retained']}개 유지.</li>")
        rows = []
        for arm in ARMS:
            s = summaries[arm]
            rows.append([LABELS[arm], f"{100*s['PCK']['10']:.3f}%", f"{s['correct10']}/{s['corners']}",
                         f"{s['matched_mean_px']:.3f}", f"{s['matched_pooled_corner8_median_px']:.3f}",
                         f"{s['matched_pooled_corner8_P90_px']:.3f}"])
        preservation = []
        for arm in ARMS[2:]:
            c = comparisons[arm]
            preservation.append([LABELS[arm], f"{c['raw_Replay_damages_prevented']}/{c['raw_Replay_damages']}",
                c['raw_Replay_damages_remaining'], f"{c['raw_Replay_gains_retained']}/{c['raw_Replay_gains']}",
                c['raw_Replay_gains_lost'], c['gain_over_N2'], c['damage_over_N2'], c['symmetry_branch_changes']])
        sections.append(f'<section><h2>{html.escape(title)}</h2>' + table(
            ["고정 비교군", "10px 이내 정확도", "정확점 / 전체", "매칭 평균 px", "매칭 중앙값 px", "매칭 P90 px"], rows)
            + '<p class="muted">정확도는 원래 전체 분모·검출 실패 벌점을 유지합니다. 평균·중앙값·P90은 매칭된 점 기준입니다.</p>'
            + '<h3>오류 차단과 좋은 보정 보존</h3>' + table(
                ["필터", "Replay 훼손 차단", "훼손 잔존", "Replay 복구 유지", "복구 손실", "N2 대비 복구", "N2 대비 훼손", "평가 대칭 분기 변경"], preservation)
            + '</section>')
    cards = []
    for item in figures:
        raw = (P.ROOT / item["output"]["path"]).read_bytes()
        embedded = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
        rows = []
        for arm in PLOT_ARMS:
            m = item["models"][arm]
            rows.append([LABELS[arm], f"P{m['native_corner']} ↔ {'G' if item['dataset']=='GREEN150' else 'L'}{item['canonical_focus']}",
                         f"b{m['whole_object_branch']}", f"{m['focus_error_px']:.2f}px", f"{m['frame_mean_px']:.2f}px"])
        reasons = ''.join('<li>' + html.escape(line) + '</li>' for line in item["focus_gate"]["explanation"])
        accept_list = ', '.join(f'P{i}' for i, enabled in enumerate(item['all_native_full_accept']) if enabled) or '없음'
        cards.append(f'<section id="{item["tag"]}" class="case"><h2>{html.escape(item["description"])}</h2>'
            f'<p class="mono">{html.escape(item["id"])}</p><img src="{embedded}" alt="{html.escape(item["description"])} 비교" loading="lazy">'
            + table(["모델", "동일 참조점의 native 대응", "전체 대칭 분기", "선택 코너 오차", "사진 평균 오차"], rows)
            + f'<div class="decision"><b>초점 코너쌍 판정</b><ul>{reasons}</ul><p>이 사진에서 적용된 native 코너: {accept_list}</p></div>'
            + '<p class="muted">P 번호는 모델 고유 채널, G는 수동 정답 번호, L은 출처 미확인 기존 참조점 번호입니다. '
              '대응은 저장된 전체 물체 대칭 분기 하나를 따릅니다. 정답마다 가장 가까운 예측점을 골라 붙이지 않습니다.</p></section>')
    navigation = ''.join(f'<a href="#{item["tag"]}">{html.escape(item["description"])}</a>' for item in figures)
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>팔레트 · 코너별 보정 선택 실험</title><style>
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#0a121a;color:#e7edf3;font-family:system-ui,"Noto Sans CJK KR",sans-serif;line-height:1.6}main{max-width:1500px;margin:auto;padding:28px}h1{font-size:30px;line-height:1.3}h2{margin:0 0 12px;font-size:22px}h3{font-size:17px}section{background:#121e2a;border:1px solid #2a3c4f;border-radius:14px;padding:23px;margin:24px 0}p{margin:10px 0}.muted{color:#aab8c7;font-size:14px}.warning{background:#332819;border:1px solid #795728;padding:17px;border-radius:12px}.status{border-left:4px solid #56ddff;padding-left:17px}.mono{font-family:ui-monospace,monospace;overflow-wrap:anywhere;font-size:13px}nav{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0}nav a{color:#cceaff;text-decoration:none;border:1px solid #365777;background:#172b3f;border-radius:20px;padding:7px 13px;font-size:14px}table{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}td,th{text-align:right;padding:10px 12px;border-bottom:1px solid #304153}td:first-child,th:first-child{text-align:left}th{color:#aecbdf;background:#192a3a}.tablewrap{overflow:auto;margin:12px 0}.case img{display:block;width:100%;height:auto;border-radius:8px}.decision{background:#172938;padding:15px;border-radius:8px;margin-top:16px}.decision ul,.status ul{margin:8px 0;padding-left:23px}footer{color:#93a7b8;font-size:13px;margin:28px 0}b{font-weight:650}@media(max-width:700px){main{padding:13px}section{padding:14px}h1{font-size:25px}}
</style><main><h1>어떤 코너의 보정을 적용할까?</h1>
<p>현재 N2 → Replay 전체 보정 → 코너쌍 선택 적용을 같은 사진에서 비교합니다. 선택되지 않은 코너는 현재 N2 좌표 그대로입니다.</p>
<div class="warning"><b>검사 완료 · 최종 모델/라벨 변경 없음</b><br>추론·필터 판단에 평가 GT를 넣지 않았습니다. 모든 선택을 저장한 뒤 평가와 그림에만 GT를 사용했습니다.
재학습·수도 레이블 생성·최종 모델 승격은 하지 않았습니다.<br><b>한계:</b>기존 평가 자료를 재사용한 탐색 실험이며 독립 확인이 아닙니다. 0.05 기준은 기존 필터에서 옮긴 값으로 코너별/확률분포 불확실성 기준으로 보정된 값이 아닙니다. 비초록 72장 참조점은 출처 미확인입니다.</div>
<section class="status"><h2>자동 집계 요약</h2><ul>''' + ''.join(status) + '''</ul><p class="muted">보정을 막아 오류가 줄어도 좋은 보정까지 잃을 수 있습니다. 아래 두 효과를 함께 확인하세요. 새 규칙을 최종 모델로 선정한 결과가 아닙니다.</p></section>
<nav>''' + navigation + '</nav>' + ''.join(sections) + ''.join(cards) + '''<footer>이미지와 표가 이 HTML에 내장되어 있습니다. 인터넷 없이 열 수 있습니다. 원본 RGB에 정확한 저장 좌표만 덧그렸습니다.<br>범례: 자홍색 × = 초록 수동 GT · 회색 빈 원 = 비초록 출처 미확인 참조 · 노란 선 = 출력 구조 · 연두 점 = Replay 적용 · 하늘색 점 = N2 유지.</footer></main></html>'''
    path.write_text(document, encoding="utf-8")


def main():
    required = [P.RAW / name for name in ("PREDICTIONS.json", "PER_FRAME_METRICS.json", "GATES.json")]
    required.extend([P.DOC / "RESULTS.json", P.DOC / "OUTPUTS_LOCK.json"])
    for path in required:
        assert path.exists(), f"Wait for frozen evaluation: {path}"
    output_lock = P.read(P.DOC / "OUTPUTS_LOCK.json")
    assert output_lock["complete"] and output_lock["frames"] == 222
    for binding in output_lock["artifacts"]:
        P.verify(binding)
    results = P.read(P.DOC / "RESULTS.json")
    assert results["complete"] and not results["final_model_modified"] and not results["thresholds_retuned"]
    predictions = {ds: {r["id"]: r for r in rows} for ds, rows in P.read(required[0]).items()}
    metrics = {ds: {arm: {r["id"]: r for r in rows} for arm, rows in arms.items()} for ds, arms in P.read(required[1]).items()}
    gates = {ds: {r["id"]: r for r in rows} for ds, rows in P.read(required[2]).items()}
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
        score = {arm: metrics[metric_ds][arm][ident] for arm in PLOT_ARMS}
        figures.append(figure_case(case, predictions[ds][ident], score, gates[ds][ident], records.get(ident), groups, contract["edges"]))
    page = P.OUT / "RESULTS_REVIEW.html"
    html_report(results, figures, page)
    manifest = dict(cases=figures, source_bindings=[P.bound(path) for path in [*required, split_path, contract_path]],
        renderer=P.bound(Path(__file__)), page=P.bound(page),
        displayed_prediction_checks=18, no_inference=True, no_training=True,
        targets_used_only_after_output_lock=True, independent_test=False,
        case_selection="Six fixed diagnostic cases requested before gate results; no result-dependent case replacement")
    (P.OUT / "VISUALIZATION_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(page=str(page), figures=[str(P.ROOT / item["output"]["path"]) for item in figures],
                         checked_predictions=18), ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    main()
