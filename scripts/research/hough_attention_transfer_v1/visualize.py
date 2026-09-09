#!/usr/bin/env python3
"""Render a Direct Hough A/B run's actual cross-attention and line predictions.

Usage: python scripts/research/hough_attention_transfer_v1/visualize.py --run-dir PATH

Input: PATH/visualization_manifest.json and its per-sample NPZ files. Output:
PATH/attention_gallery.html and PATH/visualizations/<sample>/{overview,roles,role_NN}.png.
The gallery requires no web server, JavaScript libraries, or network connection.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps


COLORS = {"A": "#00d4ff", "B": "#ff55dd", "GT": "#39ff73"}
CMAP = "magma"


def attention_stats(attention: np.ndarray, mask: np.ndarray | None) -> dict:
    """Report entropy against a uniform spatial distribution and optional mask mass."""
    values = np.maximum(np.nan_to_num(attention.astype(float)), 0)
    total = values.sum()
    if total <= 0:
        return {"entropy": None, "mask_mass": None, "mask_area": None}
    p = values / total
    positive = p[p > 0]
    entropy = float(-(positive * np.log(positive)).sum() / math.log(p.size))
    result = {"entropy": entropy, "mask_mass": None, "mask_area": None}
    if mask is not None:
        # INTER_AREA preserves fractional foreground coverage at grid boundaries.
        # Keep those fractions so displayed mass agrees with the training loss.
        mask = np.clip(np.asarray(mask, dtype=float), 0., 1.)
        if mask.shape != p.shape:
            raise ValueError(f"Mask shape {mask.shape} differs from attention {p.shape}")
        result.update(mask_mass=float((p * mask).sum()), mask_area=float(mask.mean()))
    return result


def metric_text(stats: dict) -> str:
    entropy = stats["entropy"]
    result = "Hnorm = n/a" if entropy is None else f"Hnorm = {entropy:.3f}"
    if stats["mask_mass"] is not None:
        result += f" | foreground mass = {100 * stats['mask_mass']:.1f}%"
        result += f" (area {100 * stats['mask_area']:.1f}%)"
    return result


def _safe_number(value: float, suffix: str) -> str:
    return f"{value:.1f}{suffix}" if np.isfinite(value) else "n/a"


def _show_image(ax, image: np.ndarray, pad_px: int = 0) -> None:
    height, width = image.shape[:2]
    ax.imshow(image)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if pad_px:
        ax.add_patch(Rectangle((pad_px - .5, pad_px - .5), width - 2 * pad_px,
                               height - 2 * pad_px, fill=False, ec="white", lw=1,
                               ls="--", alpha=.85, zorder=9))


def _draw_gt(ax, points, edges, valid, roles, supported, labels=False):
    for role in roles:
        a, b = edges[role]
        if not (valid[a] and valid[b]) or not np.isfinite(points[[a, b]]).all():
            continue
        segment = points[[a, b]]
        ax.plot(
            segment[:, 0], segment[:, 1], color=COLORS["GT"], lw=1.65,
            ls="-" if supported[role] else "--", zorder=5,
        )
        if labels:
            midpoint = segment.mean(axis=0)
            ax.text(*midpoint, str(role), fontsize=8, weight="bold", color="black",
                    ha="center", va="center", zorder=8,
                    bbox={"boxstyle": "round,pad=0.1", "fc": COLORS["GT"], "alpha": .9, "ec": "none"})


def _draw_prediction(ax, line: np.ndarray, color: str):
    """Keep infinite-line semantics: the two provided points are not endpoints."""
    if np.isfinite(line).all() and np.linalg.norm(line[1] - line[0]) > 1e-8:
        ax.axline(line[0], line[1], color=color, lw=1.6, alpha=.95, zorder=6)


def _heat(ax, image, attention, norm, pad_px=0):
    _show_image(ax, image, pad_px)
    height, width = image.shape[:2]
    return ax.imshow(
        np.maximum(np.nan_to_num(attention), 0) * attention.size,
        cmap=CMAP, norm=norm, alpha=.62, interpolation="bilinear",
        extent=(-.5, width - .5, height - .5, -.5), zorder=2,
    )


def _joint_norm(*attentions):
    values = np.concatenate([np.ravel(a) * a.size for a in attentions])
    # A shared robust color range stops one anomalous pixel hiding all other evidence.
    limit = max(1.0, float(np.quantile(values[np.isfinite(values)], .995)))
    return Normalize(0, limit)


def _save(fig, path: Path):
    fig.savefig(path, dpi=115, facecolor="white")
    plt.close(fig)


def render_sample(run_dir: Path, sample: dict, edges: np.ndarray, index: int) -> dict:
    sample_id = str(sample["id"])
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
    output = run_dir / "visualizations" / f"{index:03d}_{slug}"
    output.mkdir(parents=True, exist_ok=True)
    source = Path(sample["image"])
    if not source.is_absolute():
        source = run_dir / source
    with Image.open(source) as loaded:
        image = np.asarray(ImageOps.exif_transpose(loaded).convert("RGB"))
    with np.load(run_dir / sample["artifact"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    points = data["gt_points"]
    valid = data.get("gt_valid", np.ones(len(points), dtype=bool)).astype(bool)
    supported = data["supported"].astype(bool)
    mask = data.get("foreground_mask")
    role_count = len(edges)
    if len(supported) != role_count:
        raise ValueError(f"{sample_id}: {len(supported)} support labels for {role_count} edges")
    for name in ("A", "B"):
        if data[f"{name}_attention"].shape[0] != role_count:
            raise ValueError(f"{sample_id}: attention role count does not match edges")
        if data[f"{name}_lines"].shape != (role_count, 2, 2):
            raise ValueError(f"{sample_id}: invalid {name}_lines shape")

    population = str(sample.get("population", "unspecified"))
    pad_px = int(sample.get("pad_px", 0))
    title = f"{sample_id} | {population}"
    means = {name: data[f"{name}_attention"].mean(axis=0) for name in ("A", "B")}
    summary_stats = {name: attention_stats(means[name], mask) for name in ("A", "B")}
    norm = _joint_norm(means["A"], means["B"])
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.5))
    fig.subplots_adjust(left=.025, right=.965, top=.905, bottom=.105, hspace=.25, wspace=.12)
    fig.suptitle(title, fontsize=15, weight="bold", y=.985)
    input_caption = f" | Input includes {pad_px}px reflection padding; white border = original image" if pad_px else ""
    fig.text(.5, .953, "A: line supervision | B: line + attention supervision" + input_caption, ha="center", fontsize=10)
    _show_image(axes[0, 0], image, pad_px)
    _draw_gt(axes[0, 0], points, edges, valid, range(role_count), supported, labels=True)
    axes[0, 0].set_title("Original + ground-truth geometry (role IDs)", fontsize=11)
    last_heat = None
    for col, name in enumerate(("A", "B"), start=1):
        last_heat = _heat(axes[0, col], image, means[name], norm, pad_px)
        axes[0, col].set_title(f"{name}: attention averaged over roles and heads", fontsize=11)
        _show_image(axes[1, col], image, pad_px)
        _draw_gt(axes[1, col], points, edges, valid, range(role_count), supported)
        for line in data[f"{name}_lines"]:
            _draw_prediction(axes[1, col], line, COLORS[name])
        angles = data[f"{name}_angle"][supported]
        distances = data[f"{name}_distance"][supported]
        angles = angles[np.isfinite(angles)]
        distances = distances[np.isfinite(distances)]
        angle_label = _safe_number(float(np.median(angles)) if len(angles) else float("nan"), " deg")
        distance_label = _safe_number(float(np.median(distances)) if len(distances) else float("nan"), " px")
        axes[1, col].set_title(f"{name}: predicted lines | supported median {angle_label}, {distance_label}", fontsize=10)
    axes[1, 0].axis("off")
    notes = [
        "HOW TO READ", "",
        "Green: ground-truth edge segment",
        "Cyan: A predicted infinite line",
        "Magenta: B predicted infinite line", "",
        "GT solid: supported role",
        "GT dashed: unsupported / not scored", "",
        "Attention is cross-attention averaged over heads.",
        "Heat value = attention / uniform attention.",
        "1 means uniform spatial weight.",
        "A and B share the same color range; top 0.5% clips.", "",
        "A: " + metric_text(summary_stats["A"]),
        "B: " + metric_text(summary_stats["B"]), "",
        "Hnorm: normalized entropy (1 = uniform).",
        "Role averaging can hide role-specific differences.",
        "Use the individual-role views to inspect evidence.",
        "Attention location alone does not prove causality.",
    ]
    axes[1, 0].text(0, 1, "\n".join(notes), transform=axes[1, 0].transAxes,
                    va="top", fontsize=9, linespacing=1.32)
    color_ax = fig.add_axes([.37, .058, .45, .017])
    colorbar = fig.colorbar(last_heat, cax=color_ax, orientation="horizontal", extend="max")
    colorbar.set_label("Spatial attention relative to uniform (shared A/B scale)", fontsize=10)
    _save(fig, output / "overview.png")

    role_paths = []
    roles_metadata = []
    for role, edge in enumerate(edges):
        norm = _joint_norm(data["A_attention"][role], data["B_attention"][role])
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
        fig.subplots_adjust(left=.025, right=.918, top=.79, bottom=.075, wspace=.08)
        support_label = "supported / scored" if supported[role] else "unsupported / not scored"
        fig.suptitle(f"Role {role}: corners {edge[0]}-{edge[1]} | {support_label}", fontsize=13, weight="bold", y=.985)
        fig.text(.5, .922, "Green = GT segment | Cyan = A line | Magenta = B line | Head-averaged cross-attention", ha="center", fontsize=10)
        stats_by_model = {}
        for col, name in enumerate(("A", "B")):
            attention = data[f"{name}_attention"][role]
            heat = _heat(axes[col], image, attention, norm, pad_px)
            _draw_gt(axes[col], points, edges, valid, [role], supported)
            _draw_prediction(axes[col], data[f"{name}_lines"][role], COLORS[name])
            stats = attention_stats(attention, mask)
            stats_by_model[name] = stats
            angle = _safe_number(float(data[f"{name}_angle"][role]), " deg")
            distance = _safe_number(float(data[f"{name}_distance"][role]), " px")
            metric_prefix = "error" if supported[role] else "unscored geometric error"
            axes[col].set_title(f"{name}: {metric_prefix} {angle}, {distance}\n{metric_text(stats)}", fontsize=9)
        color_ax = fig.add_axes([.930, .20, .011, .47])
        colorbar = fig.colorbar(heat, cax=color_ax, extend="max")
        colorbar.set_label("Attention / uniform", fontsize=9)
        path = output / f"role_{role:02d}.png"
        _save(fig, path)
        role_paths.append(path)
        roles_metadata.append({"role": role, "edge": [int(x) for x in edge], "supported": bool(supported[role]), "stats": stats_by_model})

    # Reuse the exact per-role panels, so the long contact sheet and interactive view agree.
    with Image.open(role_paths[0]) as first:
        role_width, role_height = first.size
    sheet = Image.new("RGB", (role_width, role_height * len(role_paths)), "white")
    for index_in_sheet, path in enumerate(role_paths):
        with Image.open(path) as panel:
            sheet.paste(panel.convert("RGB"), (0, index_in_sheet * role_height))
    sheet.save(output / "roles.png")
    thumbnail = Image.fromarray(image)
    thumbnail.thumbnail((180, 100))
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=75)
    relative = lambda path: path.relative_to(run_dir).as_posix()
    return {
        "id": sample_id, "population": population,
        "overview": relative(output / "overview.png"),
        "sheet": relative(output / "roles.png"),
        "role_images": [relative(path) for path in role_paths],
        "artifact": str(sample["artifact"]),
        "pad_px": pad_px,
        "thumbnail": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(),
        "roles": roles_metadata, "attention_summary": summary_stats,
    }


def result_table(summary: dict) -> str:
    """Average independent run medians, never present these as pooled medians."""
    comparison = summary.get("comparison", {}) if isinstance(summary, dict) else {}
    runs = summary.get("by_run", {}) if isinstance(summary, dict) else {}
    populations = summary.get("populations", {}) if isinstance(summary, dict) else {}
    names = list(dict.fromkeys(list(comparison) + [p for run in runs.values() for p in run]))
    names.sort(key=lambda name: name != "real_dev")
    metric_keys = ("A_angle_median_deg", "B_angle_median_deg",
                   "A_distance_median_px", "B_distance_median_px")
    rows = []
    for name in names:
        per_seed = comparison.get(name, {}).get("per_seed", [])
        if not per_seed:
            for key, arm_a in runs.items():
                if not key.startswith("A_seed") or name not in arm_a:
                    continue
                arm_b = runs.get("B_seed" + key[len("A_seed"):], {}).get(name, {})
                arm_a = arm_a[name]
                per_seed.append({
                    "A_angle_median_deg": arm_a.get("angle_deg", {}).get("median"),
                    "B_angle_median_deg": arm_b.get("angle_deg", {}).get("median"),
                    "A_distance_median_px": arm_a.get("distance_px", {}).get("median"),
                    "B_distance_median_px": arm_b.get("distance_px", {}).get("median"),
                })
        complete = [row for row in per_seed if all(
            isinstance(row.get(key), (int, float)) and np.isfinite(row[key]) for key in metric_keys)]
        if not complete:
            continue
        numbers = [float(np.mean([row[key] for row in complete])) for key in metric_keys]
        count = populations.get(name)
        population_label = html.escape(name) + (f" ({int(count)}장)" if isinstance(count, (int, float)) else "")
        cells = "".join(f"<td>{number:.2f}</td>" for number in numbers)
        rows.append(f"<tr><th>{population_label}</th><td>{len(complete)}회 실행의 중앙값 평균</td>{cells}</tr>")
    if not rows:
        return '<p class="muted">집계 결과가 아직 없습니다. 원본 결과 JSON에서 평가 상태를 확인할 수 있습니다.</p>'
    return ('<div class="tablewrap"><table><thead><tr><th>평가 데이터</th><th>집계 방법</th>'
            '<th>A 각도 (°)</th><th>B 각도 (°)</th><th>A 거리 (px)</th><th>B 거리 (px)</th>'
            '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
            '<p class="muted">각 실행에서 평가 대상 role 전체의 중앙값을 구한 뒤 실행 간 산술평균을 계산했습니다. '
            '각도와 거리는 원본 이미지 좌표 기준이며, 작을수록 좋습니다.</p>')


def write_gallery(run_dir: Path, samples: list[dict], summary: dict, manifest: dict) -> Path:
    # Escape '<' so arbitrary filenames/metadata cannot close the JSON script element.
    payload = json.dumps({"samples": samples, "summary": summary}, ensure_ascii=False).replace("<", "\\u003c")
    summary_path = str(manifest.get("summary", "RESULTS.json"))
    template = """<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Direct Hough — Attention A/B</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#10131a;color:#e8edf5;font:15px/1.55 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:25px}h1{font-size:27px;margin:0 0 10px}p{max-width:1050px}
.muted{color:#aeb9ce}.toolbar{display:flex;flex-wrap:wrap;gap:14px;padding:15px;background:#1c2330;border-radius:10px;align-items:end;position:sticky;top:0;z-index:3}
label{display:flex;flex-direction:column;gap:4px}select,button{padding:9px;border:1px solid #53627a;border-radius:5px;background:#111722;color:#edf2fa;font-size:14px}button{cursor:pointer}
a{color:#88ceff}.links{display:flex;gap:22px;flex-wrap:wrap;margin:10px 0}.panel{width:100%;background:white;display:block;border-radius:7px;margin:12px 0 24px}
#thumbs{display:flex;gap:9px;overflow:auto;margin:18px 0;padding-bottom:7px}.thumb{min-width:160px;max-width:180px;font-size:12px;text-align:left;padding:7px}.thumb img{width:100%;height:86px;object-fit:cover}.thumb.active{outline:2px solid #67caff}
details{background:#1c2330;padding:14px;border-radius:8px;margin:18px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:650px;overflow:auto}
.legend{border-left:3px solid #83a7d9;padding:3px 16px}.status{font-weight:600}h2{font-size:19px}.empty{padding:25px;background:#1c2330}
.tablewrap{overflow:auto}table{width:100%;border-collapse:collapse;margin:12px 0;background:#1c2330}th,td{padding:11px 13px;border-bottom:1px solid #3b475b;text-align:right;white-space:nowrap}th:first-child,td:nth-child(2){text-align:left}thead{color:#b9d8ff}
</style><main>
<h1>Direct Hough: 합성 학습 → 실제 영상 attention 비교</h1>
<p>A는 선 정답으로, B는 선 정답과 attention 영역 지도로 학습한 모델입니다.
이 그림은 모델의 <b>실제 cross-attention 가중치를 head별로 평균</b>한 결과입니다.
이미지 위 밝은 영역은 해당 선을 예측할 때 큰 attention 가중치를 받은 위치입니다.</p>
<h2>선 정확도 비교</h2>
__RESULT_TABLE__
<p class="legend">__REAL_DEV_NOTE__실제 이미지는 학습과 체크포인트 선택에 사용하지 않았습니다.
실사 결과는 기존 개발용 데이터(DEV)에서 얻은 진단이며, 새로운 최종 테스트의 성능을 뜻하지 않습니다.</p>
<div class="links"><a href="REPORT.md">실험 보고서</a><a href="__SUMMARY_PATH__">전체 결과 JSON</a></div>
<p class="legend">색상은 균일한 attention 대비 배율입니다. 1은 균일한 가중치이며, A/B는 같은 역할에서 같은 색상 범위를 사용합니다.
상위 0.5%는 색상 최댓값으로 표시합니다. 초록색은 정답 선분, 청록색은 A 예측 직선, 분홍색은 B 예측 직선입니다.
점선 정답은 평가 대상에서 제외된 역할입니다. Attention만으로 예측의 인과적 근거를 확정할 수 없으므로 선 오차와 함께 확인하세요.</p>
<p class="muted" id="paddingNote"></p>
<div class="toolbar"><label>평가 데이터<select id="population"></select></label><label>샘플<select id="sample"></select></label>
<label>선 역할<select id="role"></select></label><button id="previous">이전 샘플</button><button id="next">다음 샘플</button></div>
<div id="thumbs"></div><div id="viewer"><div class="status" id="sampleTitle"></div>
<div class="links"><a id="overviewLink" target="_blank">개요 PNG</a><a id="sheetLink" target="_blank">12개 역할 전체 PNG</a><a id="artifactLink">원시 attention NPZ</a></div>
<h2>전체 개요</h2><p class="muted">상단: 원본과 정답 / A 평균 attention / B 평균 attention. 하단: 지표 안내 / A 직선 / B 직선.</p>
<img class="panel" id="overview" alt="Original, attention maps, and line predictions for A and B">
<h2 id="roleTitle">선별 attention</h2><p class="muted">역할 평균은 개별 선의 차이를 숨길 수 있습니다. 선 역할을 바꾸어 어느 위치에 집중하는지 확인하세요.
Hnorm은 정규화 엔트로피(1이면 균일)입니다. Foreground mass는 팔레트 마스크 안의 가중치 합이며, 괄호는 마스크 면적 비율입니다.</p>
<img class="panel" id="roleImage" alt="Per-role actual cross-attention with ground truth and predicted infinite lines">
</div><details><summary>실험 결과와 평가 조건 (원본 JSON)</summary><p><a href="__SUMMARY_PATH__">결과 JSON 열기</a></p><pre id="summary"></pre></details>
<p class="muted">HTML과 visualizations 폴더를 함께 보관하면 인터넷 연결 없이 열 수 있습니다. 전체 PNG는 고해상도 원본입니다.</p>
</main><script id="data" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const byId=id=>document.getElementById(id);let filtered=[];
const addOption=(el,value,label)=>{const option=document.createElement('option');option.value=value;option.textContent=label;el.append(option)};
addOption(byId('population'),'all','전체');for(const p of [...new Set(data.samples.map(s=>s.population))])addOption(byId('population'),p,p);
if(data.samples.some(s=>s.population==='real_dev'))byId('population').value='real_dev';
byId('summary').textContent=JSON.stringify(data.summary,null,2);
function roleChanged(){const s=filtered[Number(byId('sample').value)];if(!s)return;const r=Number(byId('role').value);const m=s.roles[r];byId('roleImage').src=s.role_images[r];byId('roleTitle').textContent=`선 역할 ${r}: corner ${m.edge.join('–')} · ${m.supported?'평가 대상':'평가 제외'}`;}
function sampleChanged(){const s=filtered[Number(byId('sample').value)];if(!s)return;const old=byId('role').value;byId('sampleTitle').textContent=`${s.population} / ${s.id}`;byId('paddingNote').textContent=s.pad_px?`입력: 반사 패딩 ${s.pad_px}px 포함. 흰 점선 안쪽이 원래 영상이며, attention은 패딩을 포함한 실제 전체 입력에 대응합니다.`:'';byId('overview').src=s.overview;byId('overviewLink').href=s.overview;byId('sheetLink').href=s.sheet;byId('artifactLink').href=s.artifact;byId('role').replaceChildren();for(const r of s.roles)addOption(byId('role'),r.role,`${r.role}: ${r.edge.join('–')}${r.supported?'':' (평가 제외)'}`);if(old&&Number(old)<s.roles.length)byId('role').value=old;for(const t of byId('thumbs').children)t.classList.toggle('active',t.dataset.index===byId('sample').value);roleChanged();}
function filterChanged(){filtered=data.samples.filter(s=>byId('population').value==='all'||s.population===byId('population').value);byId('sample').replaceChildren();byId('thumbs').replaceChildren();byId('viewer').hidden=!filtered.length;filtered.forEach((s,i)=>{addOption(byId('sample'),i,s.id);const b=document.createElement('button');b.className='thumb';b.dataset.index=i;const image=document.createElement('img');image.src=s.thumbnail;image.alt='';b.append(image,document.createElement('br'),document.createTextNode(s.id));b.onclick=()=>{byId('sample').value=i;sampleChanged()};byId('thumbs').append(b)});sampleChanged();}
byId('population').onchange=filterChanged;byId('sample').onchange=sampleChanged;byId('role').onchange=roleChanged;
function step(delta){if(!filtered.length)return;byId('sample').value=(Number(byId('sample').value)+delta+filtered.length)%filtered.length;sampleChanged()}
byId('previous').onclick=()=>step(-1);byId('next').onclick=()=>step(1);filterChanged();
</script></html>"""
    path = run_dir / "attention_gallery.html"
    real_count = summary.get("populations", {}).get("real_dev") if isinstance(summary, dict) else None
    real_note = f"실사 평가: {int(real_count)}장. " if isinstance(real_count, (int, float)) else ""
    document = template.replace("__SUMMARY_PATH__", html.escape(summary_path, quote=True))
    document = document.replace("__RESULT_TABLE__", result_table(summary)).replace("__REAL_DEV_NOTE__", real_note)
    path.write_text(document.replace("__PAYLOAD__", payload), encoding="utf-8")
    return path


def build_gallery(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "visualization_manifest.json").read_text())
    edges = np.asarray(manifest["edges"], dtype=int)
    if edges.ndim != 2 or edges.shape[1] != 2 or not len(edges):
        raise ValueError("Manifest edges must be a nonempty list of corner-index pairs")
    summary_file = run_dir / manifest.get("summary", "RESULTS.json")
    try:
        summary = json.loads(summary_file.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        summary = {"summary_unavailable": str(exc)}
    samples = []
    for index, sample in enumerate(manifest.get("samples", [])):
        samples.append(render_sample(run_dir, sample, edges, index))
        print(f"Visualized {index + 1}/{len(manifest['samples'])}: {sample['id']}", flush=True)
    gallery = write_gallery(run_dir, samples, summary, manifest)
    (run_dir / "visualizations" / "attention_statistics.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "visualizations" / "attention_statistics.json").write_text(
        json.dumps([{k: v for k, v in sample.items() if k != "thumbnail"} for sample in samples], indent=2, ensure_ascii=False), encoding="utf-8")
    return gallery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(build_gallery(args.run_dir))


if __name__ == "__main__":
    main()
