#!/usr/bin/env python3
"""Render offline side-edge DHT comparisons from a saved experiment manifest.

Usage: python scripts/research/deep_hough_side_v1/visualize.py --run-dir PATH

The probability panel is the saved model softmax, with theta in degrees on the
vertical axis and signed rho on the horizontal axis. Both gt_theta and pred_theta
are also degrees. The spatial panel is the channel-summed absolute product of
cached input VGG features and the gradient of that role's winning valid bin's
pre-softmax logit, NOT attention or a causal explanation.
No training, model imports, network access, or GPU are needed. After rendering,
open the local gallery in a browser; use --no-open to render only.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps


COLORS = {"gt": "#39ff73", "direct": "#00d4ff", "dht": "#ff55dd"}


def open_gallery(path: Path) -> dict:
    """Request a desktop tab without waiting for the browser's lifetime.

    A successful launcher handoff is recorded separately from visual inspection.
    Browser-launch failure never invalidates an already rendered experiment.
    """
    gallery = path.resolve()
    if not gallery.is_file():
        raise FileNotFoundError(gallery)
    uri = gallery.as_uri()
    log_path = gallery.parent / "GALLERY_OPEN.log"
    # Local text/html may be associated with a non-browser (Slack on this
    # workstation). Open a browser directly before trying file associations.
    commands = []
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "firefox"):
        browser = shutil.which(name)
        if browser:
            flag = "-new-window" if name == "firefox" else "--new-window"
            commands.append([browser, flag, uri])
            break
    commands.append([
        sys.executable, "-c",
        "import sys, webbrowser; sys.exit(0 if webbrowser.open_new(sys.argv[1]) else 1)",
        uri,
    ])
    desktop_opener = shutil.which("xdg-open")
    if desktop_opener:
        commands.append([desktop_opener, uri])
    attempts = []
    for command in commands:
        try:
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            try:
                returncode = process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                returncode = None
            attempt = {"launcher": command[0], "pid": process.pid,
                       "returncode": returncode}
        except OSError as exc:
            attempt = {"launcher": command[0], "error": str(exc), "returncode": -1}
        attempts.append(attempt)
        if attempt["returncode"] in (0, None):
            break
    status = ("launcher_accepted" if attempts[-1]["returncode"] == 0 else
              "launch_requested" if attempts[-1]["returncode"] is None else "launcher_failed")
    result = {"gallery": str(gallery), "uri": uri, "status": status,
              "window_visibility_confirmed": False, "attempts": attempts, "log": str(log_path)}
    (gallery.parent / "GALLERY_OPEN.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Browser {status}: {gallery} (launcher log: {log_path})", flush=True)
    return result


def resolve(run_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def number(value, suffix="") -> str:
    try:
        return f"{float(value):.2f}{suffix}" if np.isfinite(float(value)) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def show_image(ax, image, pad_px):
    height, width = image.shape[:2]
    ax.imshow(image)
    ax.set_xlim(-.5, width - .5)
    ax.set_ylim(height - .5, -.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if pad_px:
        ax.add_patch(Rectangle((pad_px - .5, pad_px - .5),
                               width - 2 * pad_px, height - 2 * pad_px,
                               fill=False, edgecolor="white", linestyle=":",
                               linewidth=1, alpha=.8, zorder=7))


def ground_truth(ax, points, edges, support, roles, labels=False):
    occupied = []
    for role in roles:
        segment = points[edges[role]]
        if not np.isfinite(segment).all():
            continue
        ax.plot(segment[:, 0], segment[:, 1], color=COLORS["gt"],
                lw=2, ls="-" if support[role] else "--", zorder=5)
        if labels:
            midpoint = segment.mean(axis=0)
            # Thin side faces can place several role midpoints within one
            # label width. Spread only the labels, retaining leader lines to
            # their exact segments and leaving all geometry untouched.
            origin = ax.transData.transform(midpoint)
            scale = ax.figure.dpi / 72
            limits = ax.get_window_extent()
            candidates = [(0., 0.)]
            for radius in (14., 28., 42., 56.):
                candidates.extend((radius * np.cos(a), radius * np.sin(a))
                                  for a in np.arange(8) * np.pi / 4)
            offset = candidates[-1]
            for candidate in candidates:
                center = origin + np.asarray(candidate) * scale
                inside = (limits.x0 + 7 * scale <= center[0] <= limits.x1 - 7 * scale
                          and limits.y0 + 8 * scale <= center[1] <= limits.y1 - 8 * scale)
                overlaps = any(abs(center[0] - old[0]) < 12 * scale
                               and abs(center[1] - old[1]) < 14 * scale for old in occupied)
                if inside and not overlaps:
                    offset = candidate
                    break
            occupied.append(origin + np.asarray(offset) * scale)
            ax.annotate(str(role), xy=midpoint, xytext=offset,
                        textcoords="offset points", color="black", fontsize=9,
                        weight="bold", ha="center", va="center", zorder=8,
                        arrowprops={"arrowstyle": "-", "color": COLORS["gt"], "lw": .8},
                        bbox={"boxstyle": "round,pad=.1", "facecolor": COLORS["gt"],
                              "edgecolor": "none", "alpha": .9})


def prediction(ax, line, model):
    # Saved points specify infinite lines; drawing them as finite segments would
    # change the model output and make the visual comparison misleading.
    if np.isfinite(line).all() and np.linalg.norm(line[1] - line[0]) > 1e-8:
        ax.axline(line[0], line[1], color=COLORS[model], lw=1.7, alpha=.95, zorder=6)


def bin_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or not len(centers) or not np.isfinite(centers).all():
        raise ValueError("Hough bin centers must be a finite nonempty vector")
    if len(centers) == 1:
        return np.array([centers[0] - .5, centers[0] + .5])
    if not np.all(np.diff(centers) > 0):
        raise ValueError("Hough bin centers must be strictly increasing")
    midpoints = (centers[:-1] + centers[1:]) / 2
    return np.r_[centers[0] - (midpoints[0] - centers[0]), midpoints,
                 centers[-1] + (centers[-1] - midpoints[-1])]


def save(fig, path):
    fig.savefig(path, dpi=115, facecolor="white")
    plt.close(fig)


def validate(data, edges, sample_id):
    roles = len(edges)
    expected = {
        "gt_points": (8, 2), "support": (roles,), "facing": (roles,),
        "direct_lines": (roles, 2, 2), "dht_lines": (roles, 2, 2),
        "direct_angle": (roles,), "dht_angle": (roles,),
        "direct_distance": (roles,), "dht_distance": (roles,),
        "gt_theta": (roles,), "gt_rho": (roles,),
        "pred_theta": (roles,), "pred_rho": (roles,),
        "feature_sensitivity": (roles, 50, 50),
    }
    for key, shape in expected.items():
        if key not in data or data[key].shape != shape:
            raise ValueError(f"{sample_id}: {key} must have shape {shape}")
    probability = data["dht_probability"]
    shape = (roles, len(data["theta_degrees"]), len(data["rho_values"]))
    if probability.shape != shape or not np.isfinite(probability).all():
        raise ValueError(f"{sample_id}: finite dht_probability with shape {shape} required")
    if np.any(probability < 0) or not np.allclose(probability.sum(axis=(1, 2)), 1, atol=1e-3):
        raise ValueError(f"{sample_id}: dht_probability must be a per-role normalized softmax")
    sensitivity = data["feature_sensitivity"]
    if not np.isfinite(sensitivity).all() or np.any(sensitivity < 0):
        raise ValueError(f"{sample_id}: feature_sensitivity must be finite and nonnegative")


def render_sample(run_dir, sample, edges, role_names, index):
    sample_id = str(sample["id"])
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
    output = run_dir / "visualizations" / f"{index:03d}_{slug}"
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(resolve(run_dir, sample["image"])) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    with np.load(resolve(run_dir, sample["artifact"]), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    validate(data, edges, sample_id)
    support = data["support"].astype(bool)
    facing = data["facing"].astype(bool)
    points = data["gt_points"]
    pad = int(sample.get("pad_px", 0))
    population = str(sample.get("population", "unspecified"))
    role_count = len(edges)
    title = f"{sample_id} | {population}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8))
    fig.subplots_adjust(left=.015, right=.985, top=.81, bottom=.12, wspace=.08)
    fig.suptitle(title, y=.98, fontsize=14, weight="bold")
    fig.text(.5, .923, "Side outlines: long side boundaries and short height lines",
             ha="center", fontsize=11)
    for col, model in enumerate((None, "direct", "dht")):
        ax = axes[col]
        show_image(ax, image, pad)
        ground_truth(ax, points, edges, support, range(role_count), labels=model is None)
        if model is None:
            ax.set_title("Input + GT geometry (side-role IDs)", fontsize=11)
        else:
            for line in data[f"{model}_lines"]:
                prediction(ax, line, model)
            angles = data[f"{model}_angle"][support]
            distances = data[f"{model}_distance"][support]
            angles = angles[np.isfinite(angles)]
            distances = distances[np.isfinite(distances)]
            angle = number(np.median(angles) if angles.size else None, " deg")
            distance = number(np.median(distances) if distances.size else None, " px")
            label = "Direct Hough" if model == "direct" else "Deep Hough Transform"
            ax.set_title(f"{label}\nScored-role median: {angle}, {distance}", fontsize=10)
    fig.text(.5, .069, "Green = GT segment | Cyan = Direct infinite line | Magenta = DHT infinite line",
             ha="center", fontsize=10)
    fig.text(.5, .031, f"GT solid = scored/support; dashed = not scored. White dotted border = original image ({pad}px padding).",
             ha="center", fontsize=9)
    save(fig, output / "overview.png")

    role_images, role_metadata = [], []
    theta_edges = bin_edges(data["theta_degrees"])
    rho_edges = bin_edges(data["rho_values"])
    for role in range(role_count):
        fig, axes = plt.subplots(2, 2, figsize=(13.6, 11.5))
        fig.subplots_adjust(left=.055, right=.94, top=.865, bottom=.13,
                            hspace=.27, wspace=.25)
        role_label = str(role_names[role])
        fig.suptitle(f"Side role {role}: {role_label} | corners {edges[role, 0]}-{edges[role, 1]}",
                     fontsize=14, weight="bold", y=.982)
        status = "scored/support" if support[role] else "not scored"
        facing_text = "yes" if facing[role] else "no"
        fig.text(.5, .947, f"{sample_id} | {population} | {status} | geometry-derived front-facing: {facing_text}",
                 fontsize=9, ha="center")
        fig.text(.5, .918, "Green = GT segment | Cyan = Direct infinite line | Magenta = DHT infinite line",
                 fontsize=10, ha="center")
        for ax, model in zip(axes[0], ("direct", "dht")):
            show_image(ax, image, pad)
            ground_truth(ax, points, edges, support, [role])
            prediction(ax, data[f"{model}_lines"][role], model)
            angle = number(data[f"{model}_angle"][role], " deg")
            distance = number(data[f"{model}_distance"][role], " px")
            model_label = "Direct Hough" if model == "direct" else "Deep Hough Transform"
            metric_label = "error" if support[role] else "unscored geometry error"
            ax.set_title(f"{model_label}\n{metric_label}: {angle}, {distance}", fontsize=10)

        ax = axes[1, 0]
        show_image(ax, image, pad)
        sensitivity = data["feature_sensitivity"][role].astype(float)
        scale = float(np.quantile(sensitivity, .995))
        # A sparse proxy can have >99.5% exact zeros. Preserve nonzero evidence.
        if scale <= 0:
            scale = float(sensitivity.max())
        relative = sensitivity / scale if scale > 0 else np.zeros_like(sensitivity)
        height, width = image.shape[:2]
        heat = ax.imshow(relative, cmap="magma", norm=Normalize(0, 1),
                         alpha=.65, interpolation="bilinear",
                         extent=(-.5, width - .5, height - .5, -.5), zorder=2)
        ground_truth(ax, points, edges, support, [role])
        prediction(ax, data["dht_lines"][role], "dht")
        ax.set_title("DHT winning-logit feature sensitivity\nChannel-summed |gradient x feature|; not attention", fontsize=10)
        cbar = fig.colorbar(heat, ax=ax, fraction=.035, pad=.025, extend="max")
        cbar.set_label("Relative sensitivity (upper 0.5% clipped)", fontsize=8)

        ax = axes[1, 1]
        probability = data["dht_probability"][role]
        # pcolormesh makes the association explicit: rows are theta, columns rho.
        heat = ax.pcolormesh(rho_edges, theta_edges, probability, shading="flat",
                             cmap="magma", vmin=0, vmax=float(probability.max()),
                             rasterized=True)
        gt_theta, gt_rho = float(data["gt_theta"][role]), float(data["gt_rho"][role])
        pred_theta, pred_rho = float(data["pred_theta"][role]), float(data["pred_rho"][role])
        if np.isfinite([gt_theta, gt_rho]).all():
            ax.scatter([gt_rho], [gt_theta], color=COLORS["gt"], marker="x", s=100,
                       linewidths=2.1, zorder=5, label="GT")
        if np.isfinite([pred_theta, pred_rho]).all():
            ax.scatter([pred_rho], [pred_theta], color=COLORS["dht"], marker="o", s=44,
                       edgecolors="white", linewidths=.8, zorder=6, label="Prediction")
        ax.set_xlim(rho_edges[0], rho_edges[-1])
        ax.set_ylim(theta_edges[0], theta_edges[-1])
        ax.set_xlabel("Signed rho (centered 50x50 feature cells)", fontsize=9)
        ax.set_ylabel("Line-normal theta (degrees)", fontsize=9)
        ax.set_title(f"DHT parameter probability (actual softmax)\nPeak probability = {float(probability.max()):.4f}", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.legend(loc="upper right", fontsize=8, facecolor="white", framealpha=.8)
        cbar = fig.colorbar(heat, ax=ax, fraction=.035, pad=.025)
        cbar.set_label("Probability per (theta, rho) bin", fontsize=8)

        fig.text(.5, .064, "Sensitivity target: the role's winning valid-bin pre-softmax logit; gradient is with respect to cached input VGG features.",
                 ha="center", fontsize=8.5)
        fig.text(.5, .043, "Sum over channels of |gradient x feature|. Brightness is scaled per role; this local proxy does not establish causal importance.",
                 ha="center", fontsize=8.5)
        fig.text(.5, .022, "Front-facing is geometry-derived, not a visibility label. Theta is the line normal: 0 deg = vertical line; 90 deg = horizontal line.",
                 ha="center", fontsize=8.5)
        path = output / f"role_{role:02d}.png"
        save(fig, path)
        role_images.append(path.relative_to(run_dir).as_posix())
        role_metadata.append({
            "role": role, "name": role_label, "support": bool(support[role]),
            "facing": bool(facing[role]), "sensitivity_scale": scale,
            **{key: float(data[key][role]) if np.isfinite(data[key][role]) else None
               for key in ("direct_angle", "direct_distance", "dht_angle", "dht_distance")},
        })
    return {"id": sample_id, "population": population,
            "overview": (output / "overview.png").relative_to(run_dir).as_posix(),
            "role_images": role_images, "roles": role_metadata,
            "artifact": str(sample["artifact"]), "pad_px": pad}


def summary_rows(value, path=()):
    """Accept nested by_run/population/group summaries without a rigid schema."""
    if not isinstance(value, dict):
        return []
    rows = []
    if isinstance(value.get("angle_deg"), dict) and isinstance(value.get("distance_px"), dict):
        angle, distance = value["angle_deg"], value["distance_px"]
        count = value.get("n_roles", value.get("count", value.get("n", value.get("n_lines", ""))))
        rows.append((" / ".join(str(part) for part in path), str(count),
                     number(angle.get("median")), number(angle.get("p90")),
                     number(distance.get("median")), number(distance.get("p90"))))
    for key, child in value.items():
        if isinstance(child, dict):
            rows.extend(summary_rows(child, path + (key,)))
    return rows


def write_gallery(run_dir, manifest, samples):
    summary_path = resolve(run_dir, manifest.get("summary", "RESULTS.json"))
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    metrics = summary_rows(summary.get("by_run", summary))
    population_order = {"real_dev": 0, "synth_test": 1, "cross_v4": 2,
                        "synth_val": 3, "synth_train": 4}

    def metric_order(row):
        parts = [part.strip() for part in row[0].split("/")]
        population = min((population_order.get(part, 99) for part in parts), default=99)
        model = 0 if parts and parts[0].startswith("Direct") else 1
        # Pair the models within each population/subset before moving to train.
        return population, len(parts), tuple(parts[2:]), model, parts[0] if parts else ""

    metrics.sort(key=metric_order)
    def metric_table(selected):
        rows = "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                       for row in selected)
        return ("<div class='tablewrap'><table><thead><tr><th>실행 / 평가군 / 부분집합</th><th>선 수</th>"
                "<th>각도 중앙값 (°)</th><th>각도 P90 (°)</th><th>위치 중앙값 (px)</th><th>위치 P90 (px)</th>"
                "</tr></thead><tbody>" + rows + "</tbody></table></div>")

    headline = [row for row in metrics if " / by_role / " not in row[0]]
    detailed = [row for row in metrics if " / by_role / " in row[0]]
    table = metric_table(headline) if headline else "<p>아래 RESULTS.json에서 저장된 평가 결과를 확인할 수 있습니다.</p>"
    if detailed:
        table += "<details><summary>역할별 전체 평가 펼치기</summary>" + metric_table(detailed) + "</details>"
    default = next((i for i, sample in enumerate(samples) if sample["population"] == "real_dev"), 0)
    # JSON inside a script element must escape '<' to keep sample IDs from
    # closing that element. The gallery uses textContent for all dynamic labels.
    serialized = json.dumps({"samples": samples, "default": default,
                             "role_names": manifest["role_names"]}, ensure_ascii=False,
                            allow_nan=False).replace("<", "\\u003c")
    summary_link = html.escape(str(manifest.get("summary", "RESULTS.json")), quote=True)
    page = """<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>팔레트 측면 외곽: Direct Hough / Deep Hough Transform</title>
<style>
:root{color-scheme:light;font-family:system-ui,-apple-system,sans-serif;color:#152036;background:#f2f5f8}
body{max-width:1500px;margin:0 auto;padding:24px}h1{font-size:25px;margin:0 0 12px}h2{font-size:19px}
p{line-height:1.6;margin:10px 0}.muted{color:#526078;font-size:14px}.card{background:white;padding:20px;border:1px solid #dce2eb;border-radius:12px;margin-bottom:18px}
.controls{display:flex;gap:16px;flex-wrap:wrap;position:sticky;top:0;z-index:10;box-shadow:0 2px 12px #1231}
label{display:flex;flex-direction:column;gap:5px;font-size:13px;font-weight:650}select{font:inherit;padding:9px;max-width:78vw;border:1px solid #bbc5d3;border-radius:6px;background:white}
img{display:block;width:100%;height:auto;background:white}a{color:#145bc0}.legend{display:flex;gap:20px;flex-wrap:wrap}.legend span::before{content:' ';display:inline-block;width:23px;height:4px;margin-right:7px;vertical-align:middle;background:var(--c)}
.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;text-align:right;border-bottom:1px solid #e4e8ee;white-space:nowrap}td:first-child,th:first-child{text-align:left}
details pre{overflow:auto;max-height:440px;font-size:12px;white-space:pre-wrap}#metadata{margin-bottom:0}noscript{color:#a22}
</style>
<body>
<div class="card"><h1>팔레트 측면 외곽선: Direct Hough와 Deep Hough Transform</h1>
<p>긴 측면 경계선 4개와 짧은 높이선 4개를 비교합니다. 이미지별 전체 예측과 선별 예측, DHT 특징 민감도, 실제 Hough 파라미터 확률을 볼 수 있습니다.</p>
<p class="legend"><span style="--c:#39c75b">초록: 정답 선분</span><span style="--c:#00b1d5">청록: Direct 예측 직선</span><span style="--c:#ef40c4">자홍: DHT 예측 직선</span></p>
<p class="muted">특징 민감도는 각 역할의 예측 후보(유효한 bin 중 최고 점수)의 softmax 이전 logit을 입력 VGG 특징에 대해 미분한 뒤, |gradient × feature|를 채널별로 더한 값입니다. Attention이 아니며, 인과적 중요도를 증명하지 않습니다. 밝기 범위는 선마다 정규화하므로 서로 다른 선의 절대 세기를 비교하지 마세요. 정답 실선은 평가 대상, 점선은 평가 제외를 뜻합니다. Front-facing은 기하 기반 분류이며 실제 가시성 정답이 아닙니다.</p>
<p class="muted">이미지는 seed 1 실행 결과이며, 기본 화면은 사전에 저장된 첫 real_dev 이미지입니다. 갤러리 일부 사례로 전체 성능을 판단하지 말고 아래 전체 평가 수치를 함께 확인하세요. 평가 표는 real_dev, synth_test 순서로 시작하며 각 실행의 수치를 표시합니다. 이 HTML과 visualizations 폴더를 함께 보관하면 인터넷 없이 열 수 있습니다.</p>
</div>
<div class="card controls"><label>이미지<select id="sample"></select></label><label>측면 선<select id="role"></select></label></div>
<div class="card"><h2 id="sample-title"></h2><a id="overview-link" target="_blank"><img id="overview" alt="원본과 정답, Direct와 DHT 전체 선 비교"></a><p class="muted" id="metadata"></p></div>
<div class="card"><h2 id="role-title"></h2><a id="role-link" target="_blank"><img id="role-image" alt="선별 예측과 특징 민감도, Hough 파라미터 확률"></a><p class="muted">파라미터 맵: 가로축 ρ는 50×50 특징 맵 중심을 원점으로 한 부호 있는 거리, 세로축 θ는 선의 법선 각도입니다. θ=0°는 수직선, θ=90°는 수평선입니다. 초록 ×는 정답, 자홍 ●는 예측입니다. 색은 저장된 softmax 확률이며 합계는 역할별로 1입니다.</p><p><a id="artifact">원본 NPZ 데이터</a></p></div>
<div class="card"><h2>저장된 전체 평가</h2>__TABLE__<p><a href="__SUMMARY__">RESULTS.json</a> · <a href="REPORT.md">실험 보고서</a> · <a href="visualization_manifest.json">시각화 표본 목록</a></p>
<details><summary>원본 평가 JSON 펼치기</summary><pre>__RAW_SUMMARY__</pre></details></div>
<noscript>드롭다운을 사용하려면 JavaScript가 필요합니다. visualizations 폴더의 overview.png와 role_00.png 등의 이미지를 직접 열어도 됩니다.</noscript>
<script type="application/json" id="data">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const sampleSelect=document.getElementById('sample'),roleSelect=document.getElementById('role');
data.samples.forEach((s,i)=>sampleSelect.add(new Option(`${s.population} / ${s.id}`,String(i))));
data.role_names.forEach((name,i)=>roleSelect.add(new Option(`${i}: ${name}`,String(i))));
sampleSelect.value=String(data.default);
function refresh(){
const s=data.samples[Number(sampleSelect.value)],r=Number(roleSelect.value),role=s.roles[r];
document.getElementById('sample-title').textContent=`${s.population} / ${s.id}`;
document.getElementById('overview').src=s.overview;document.getElementById('overview-link').href=s.overview;
document.getElementById('role-title').textContent=`측면 선 ${r}: ${data.role_names[r]}`;
document.getElementById('role-image').src=s.role_images[r];document.getElementById('role-link').href=s.role_images[r];
document.getElementById('artifact').href=s.artifact;
document.getElementById('metadata').textContent=`원본 경계는 흰 점선으로 표시됩니다. 입력 패딩 ${s.pad_px}px. 현재 선: ${role.support?'평가 대상':'평가 제외'} / 기하 기반 front-facing: ${role.facing?'예':'아니오'}.`;
}
sampleSelect.addEventListener('change',refresh);roleSelect.addEventListener('change',refresh);refresh();
</script></body></html>
"""
    page = page.replace("__TABLE__", table).replace("__SUMMARY__", summary_link)
    page = page.replace("__RAW_SUMMARY__", html.escape(json.dumps(summary, ensure_ascii=False, indent=2)))
    page = page.replace("__DATA__", serialized)
    output = run_dir / "deep_hough_gallery.html"
    output.write_text(page, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--no-open", action="store_true",
                        help="Render the gallery without opening the local browser")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "visualization_manifest.json").read_text())
    edges = np.asarray(manifest["edges"], dtype=int)
    if edges.shape != (8, 2) or np.any(edges < 0) or np.any(edges > 7):
        raise ValueError("Manifest must contain eight edges between corners 0..7")
    if len(manifest["role_names"]) != len(edges):
        raise ValueError("Manifest role_names and edges must have matching length")
    if not manifest["samples"]:
        raise ValueError("Manifest contains no samples")
    samples = []
    for index, sample in enumerate(manifest["samples"]):
        samples.append(render_sample(run_dir, sample, edges, manifest["role_names"], index))
        print(f"Rendered {index + 1}/{len(manifest['samples'])}: {sample['id']}", flush=True)
    gallery = write_gallery(run_dir, manifest, samples)
    print(gallery, flush=True)
    if not args.no_open:
        open_gallery(gallery)


if __name__ == "__main__":
    main()
