"""Postseal PRIMARY192 RGB panels, exact geometry, and parallel comparisons."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.research.pallet_boundary_action_repair_v1.display_privacy import display_image
from scripts.research.pallet_boundary_action_repair_v1.gallery import (
    project, wire, background, observations_panel, observed_marks)
from .common import read, rows, sha, write
from .figures import label, COLORS, group_name
from .registry import BASIC_GROUPS, VISUAL_GROUPS
from .visual_patches import project_queries, sample


def digest_id(fid):
    return hashlib.sha256(fid.encode()).hexdigest()[:16]


def error_text(metric):
    if not metric["pose_available"]:
        return "NO_POSE / failure"
    return (f"{metric['translation_cm']:.4g} cm / {metric['rotation_deg']:.4g} deg / "
            + ("success" if metric["success"] else "failure"))


def choose_cases(scored, new_arms):
    """Fixed E0 comparisons; never substitute a visually favorable success."""
    records = []
    for arm in new_arms:
        complete = [r for r in scored if all(r["metrics"][a][k] is not None
            for a in ("E0", arm) for k in ("translation_cm", "rotation_deg"))]
        for axis, title in (("translation_cm", "translation"), ("rotation_deg", "rotation")):
            for direction, sign in (("worsening", 1), ("improvement", -1)):
                values = [(sign * (r["metrics"][arm][axis] - r["metrics"]["E0"][axis]), r)
                          for r in complete]
                values = [(value, r) for value, r in values if value > 0]
                values.sort(key=lambda pair: (-pair[0], digest_id(pair[1]["id"])))
                records.append(dict(arm=arm, category=title + "_" + direction,
                    comparator="E0", population_count=len(values),
                    selected_id=values[0][1]["id"] if values else None))
        for category, predicate in (("binary_recovery", lambda b, a: not b and a),
                                     ("binary_damage", lambda b, a: b and not a)):
            selected = sorted([r for r in scored if predicate(r["metrics"]["E0"]["success"],
                                                              r["metrics"][arm]["success"])],
                              key=lambda r: digest_id(r["id"]))
            records.append(dict(arm=arm, category=category, comparator="E0",
                population_count=len(selected), selected_id=selected[0]["id"] if selected else None))
        selected = [r for r in scored if not r["full_bank_has_success"]
            and r["metrics"][arm]["translation_cm"] is not None
            and r["metrics"]["N0"]["translation_cm"] is not None
            and r["metrics"][arm]["translation_cm"] > r["metrics"]["N0"]["translation_cm"]]
        selected.sort(key=lambda r: (-(r["metrics"][arm]["translation_cm"]
                                      - r["metrics"]["N0"]["translation_cm"]), digest_id(r["id"])))
        records.append(dict(arm=arm, category="fixed_full_failure41_max_T_worsening", comparator="N0",
            population_count=len(selected), selected_id=selected[0]["id"] if selected else None))
    return records


def pose_panel(ax, row, arm, frame, image, reference, prediction, color):
    background(ax, frame, image)
    uv, depth = project(row["poses"][arm], frame)
    wire(ax, uv, depth, color)
    guv, gz = project(dict(available=True, R_physical=reference["R"], centroid=reference["t"]), frame)
    wire(ax, guv, gz, "#159260", "--", marker="s", alpha=.65)
    observed_marks(ax, frame["packet"]["solver_input"])
    score = prediction.get("selected_score")
    source = row["action_sources"][arm]
    text = label(arm) + "\n" + error_text(row["metrics"][arm]) + "\nsource=" + source
    if score is not None:
        text += f"; own score={score:.5g}"
    ax.set_title(text, fontsize=7, bbox=dict(facecolor="white", edgecolor="none", pad=2))


def f1_text(extra_row, candidate_id):
    action = next((r for r in extra_row["actions"] if r["candidate_id"] == candidate_id), None)
    if action is None:
        return "No corresponding actual F1 action"
    lines = ["Same R_ORDER_F1 returned physical pose", "Both hypotheses retained; owner UNVERIFIED"]
    for family in action["family_diagnostics"]:
        support = family["finite_support"]
        lines.append("\n" + family["family"])
        if support is None or not support["fixed_profile_count"]:
            lines.append("B: no original fixed observation")
        else:
            lines.append(f"B finite profiles: {support['supported_profiles']}/{support['fixed_profile_count']}")
            lines.append(str(dict(Counter(p["geometry_status"] for p in support["profiles"]))))
        for mode, value in family["original_costs"].items():
            lines.append(mode + ": " + (f"{value:.5g}" if value is not None else "missing"))
    return "\n".join(lines)


def render(row, frame, e_run, image, reference, predictions, arms, path,
           extra_row=None, *, detailed=False):
    count = 2 + len(arms) + int(detailed)
    columns = 4
    nrows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(nrows, columns, figsize=(18 if detailed else 15.5,
                                                    (3.35 if detailed else 2.65) * nrows), squeeze=False)
    fig.suptitle(row["id"] + " / " + row["partition"]
        + " / fixed representative seed20261018; one returned pose per method; all errors retained", fontsize=10)
    background(axes.flat[0], frame, image)
    axes.flat[0].set_title("Original RGB / display-only privacy edits", fontsize=8,
                          bbox=dict(facecolor="white", edgecolor="none", pad=2))
    observations_panel(axes.flat[1], frame, e_run, image)
    for ax, arm, color in zip(list(axes.flat)[2:], arms, COLORS):
        pose_panel(ax, row, arm, frame, image, reference, predictions[arm], color)
    if detailed:
        ax = axes.flat[2 + len(arms)]
        ax.axis("off")
        selected = predictions.get("R_ORDER_F1_s20261018", {}).get("selected_candidate_id")
        ax.text(0, 1, f1_text(extra_row, selected) if extra_row else "F1 unavailable",
                va="top", transform=ax.transAxes, fontsize=6.3, wrap=True)
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    fig.text(.5, .012,
        "Solid/circles: returned pose; green dashed/squares: fixed geometric reference. Hollow B/I: image observations. "
        "S: pose projection only. Model scores have different meanings and are not safety probabilities.",
        ha="center", fontsize=6)
    fig.tight_layout(rect=(0, .035, 1, .965), h_pad=4., w_pad=1.2)
    if detailed:
        fig.savefig(path, dpi=105, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(path, dpi=65, bbox_inches="tight", facecolor="white",
                    pil_kwargs=dict(quality=82, optimize=True))
    plt.close(fig)


def support_panel(row, frame, image, extra_row, selected_id, path):
    action = next(r for r in extra_row["actions"] if r["candidate_id"] == selected_id)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    records = []
    for column, family in enumerate(action["family_diagnostics"]):
        support = family["finite_support"]
        profiles = support["profiles"] if support else []
        def priority(profile):
            lam = profile.get("lambda_segment")
            return (0 if not profile["finite_support"] else 1,
                    -(max(-lam, lam - 1, 0) if lam is not None else float("inf")), profile["support_id"])
        chosen = min(profiles, key=priority) if profiles else None
        for level, ax in enumerate(axes[:, column]):
            if level == 0:
                background(ax, frame, image)
                uv, depth = project(row["poses"]["R_ORDER_F1_s20261018"], frame, family["family"])
                wire(ax, uv, depth, "#78858d", alpha=.5)
            else:
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_xlabel("Original x pixels / geometry view, including off-canvas")
                ax.set_ylabel("Original y pixels")
                ax.invert_yaxis()
                h, w = frame["trace"]["raw_hw"]
                ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], color="#aaa", lw=.7)
            if chosen:
                a, b = np.asarray(chosen["projected_endpoints_xy"])
                ax.plot([a[0], b[0]], [a[1], b[1]], "o-", color="#e47b20", lw=2)
                intersection = chosen.get("intersection_xy")
                lam = chosen.get("lambda_segment")
                if intersection is not None and np.isfinite(intersection).all():
                    q = np.asarray(intersection)
                    lo, hi = min(0., lam), max(1., lam)
                    first, last = a + lo * (b - a), a + hi * (b - a)
                    ax.plot([first[0], last[0]], [first[1], last[1]], "--", color="#8757a1")
                    ax.scatter(*q, marker="x", color="#c51b3b", s=50)
                text = chosen["geometry_status"] + "; lambda=" + (f"{lam:.6g}" if lam is not None else "undefined")
            else:
                text = "NO_FIXED_B_OBSERVATION / no confidence claim"
            ax.set_title(family["family"] + " / same returned physical pose\n" + text,
                fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=2))
        records.append(dict(family=family["family"], chosen_profile=chosen,
                            support_summary=support and {k: support[k] for k in
                            ("fixed_profile_count", "supported_profiles", "status")}))
    fig.suptitle(row["id"] + " / saved F1 diagnostics, original G13 parameterization; material owner UNVERIFIED")
    fig.text(.5, .012,
        "RGB is display-only privacy-redacted copy; measurement inputs unchanged.",
        ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .045, 1, 1), h_pad=3.5, w_pad=2)
    fig.savefig(path, dpi=125, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return records


def visual_panel(row, frame, image, geometry, prediction, path):
    uv, depth = project_queries(prediction["actual_pose"], geometry)
    # Public atlas samples only the display copy. Original unredacted patches
    # remain the actual model input and are not redistributed as a raw cache.
    patches, validity, _ = sample(np.asarray(image), uv, depth, geometry["patch_side_raw_px"])
    fig, axes = plt.subplots(4, 11, figsize=(16, 9.5))
    for region in range(20):
        row_index, column = divmod(region, 5)
        ax = axes[row_index, column]
        ax.imshow(patches[region, :3].transpose(1, 2, 0), interpolation="nearest")
        ax.set_title(f"q{region}: valid {validity[region]:.3f}", fontsize=6)
        ax.axis("off")
        null_ax = axes[row_index, column + 6]
        null_ax.imshow(np.zeros((16, 16, 3)), interpolation="nearest")
        null_ax.set_title(f"q{region}: valid {validity[region]:.3f}", fontsize=6)
        null_ax.axis("off")
    for ax in axes[:, 5]:
        ax.axis("off")
    fig.suptitle(row["id"] + " / same actual V_RGB returned pose queries\n"
        "Left: RGB display copy (privacy edits only); right: NULL color=0. Both retain identical validity/mask.", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .68), h_pad=1.8)
    location_ax = fig.add_axes((.22, .72, .56, .20))
    background(location_ax, frame, image)
    for i, (point, z) in enumerate(zip(uv, depth)):
        if np.isfinite(point).all() and np.isfinite(z) and z > 0:
            location_ax.scatter(*point, s=13, marker='o', facecolors='none', edgecolors='#e64b30')
            location_ax.annotate(str(i), point, xytext=(2, 2), textcoords='offset points',
                                 fontsize=5, color='#b52e16', clip_on=True)
    location_ax.set_title('Projected query centers on original display canvas; not measured material boundaries', fontsize=7)
    fig.text(.5, .015, "20 queries = original eight corners + twelve 3D edge midpoints. "
        "Patch enlargement is display-only; actual model uses original RGB, not this panel.", ha="center", fontsize=7)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dict(query_xy_raw=[[float(x), float(y)] if np.isfinite([x, y]).all() else None for x, y in uv],
        depth_m=[float(x) if np.isfinite(x) else None for x in depth],
        region_validity=validity.tolist(), patch_side_raw_px=geometry["patch_side_raw_px"],
        public_RGB="display-only privacy-redacted copy, not a replacement training/inference input")


def table_html(row, predictions, all_arms):
    body = []
    for arm in all_arms:
        m, p = row["metrics"][arm], predictions[arm]
        score = p.get("selected_score")
        body.append("<tr><td>" + html.escape(arm) + "</td><td>" + html.escape(error_text(m))
            + "</td><td>" + html.escape(row["action_sources"][arm]) + "</td><td>"
            + html.escape(str(p.get("selected_candidate_id", "N/A"))) + "</td><td>"
            + html.escape(str(score) if score is not None else "N/A") + "</td></tr>")
    return ("<table><thead><tr><th>방법·seed</th><th>위치 / 회전 / 공동성공</th>"
            "<th>실제 출처</th><th>선택 후보</th><th>각 모델 자체 점수</th></tr></thead><tbody>"
            + "".join(body) + "</tbody></table>")


def generate(run_dir, old_run, evaluation, reference_dir, output):
    run_dir, old_run, evaluation, reference_dir, output = map(Path,
        (run_dir, old_run, evaluation, reference_dir, output))
    started = time.monotonic()
    manifest = read(run_dir / "PREDICTION_MANIFEST.json")
    evseal = read(evaluation / "EVALUATION_SEAL.json")
    if evseal["prediction_manifest_sha256"] != sha(run_dir / "PREDICTION_MANIFEST.json"):
        raise ValueError("Gallery requires completed postseal evaluation")
    if sha(reference_dir / "FIXED_REFERENCES.json") != evseal["reference_sha256"]:
        raise ValueError("Original geometric references changed")
    refs = read(reference_dir / "FIXED_REFERENCES.json")["references"]
    scored = [r for r in rows(evaluation / "SCORED_ROWS.jsonl.gz") if r["partition"] != "CAL"]
    if len(scored) != 192:
        raise ValueError("Every original PRIMARY192 image required")
    predictions = {r["id"]: r["predictions"] for r in rows(run_dir / "PREDICTIONS.jsonl.gz")}
    upstream = {r["id"]: r for r in rows(old_run / "d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz")}
    order = {r["id"]: i for i, r in enumerate(rows(old_run / "d1_real_candidates_v1/HEAD_FEATURES.jsonl.gz"))}
    extras = {r["id"]: r for r in rows(run_dir / "f1_real_v1/F1_FEATURES.jsonl.gz")}
    privacy_path = old_run / "DISPLAY_PRIVACY_PLAN.json"
    privacy = read(privacy_path)
    if set(privacy["frames"]) != {r["id"] for r in scored}:
        raise ValueError("Original approved primary privacy mapping changed")
    new_arms = [g + "_s20261018" for g in BASIC_GROUPS + VISUAL_GROUPS
                if g + "_s20261018" in manifest["arms"]]
    simple = ["N0", "E0", "D1_H1_MLP_s20261018", "D1_H1_PAIR_ROLE_AFFINE_s20261018"] + new_arms
    detailed = ["N0", "E0", "POSEFIX_SUBPIX_PF_ANCHOR", "POSEFIX_SUBPIX_YOLO_ANCHOR",
        "D1_H1_MLP_s20261018", "D1_H1_PAIR_MLP_s20261018", "D1_H1_PAIR_ROLE_AFFINE_s20261018"] + new_arms
    selection = choose_cases(scored, new_arms)
    selected_ids = {r["selected_id"] for r in selection if r["selected_id"]}
    visual_inputs = {r["id"]: r for r in rows(run_dir / "visual_inputs_v1/VISUAL_INPUTS.jsonl.gz")}
    output.mkdir(parents=True, exist_ok=False)
    for name in ("thumbnails", "details", "frames"):
        (output / name).mkdir()
    cards, transforms = [], []
    for index, row in enumerate(scored):
        fid = row["id"]
        up = upstream[fid]
        e_run = next(rows(old_run / "d1_real_candidates_v1/frames" / f"{order[fid]:04d}" / "E_RUN.jsonl.gz"))
        if e_run["id"] != fid:
            raise ValueError("Original observation join changed")
        frame = dict(up, packet=e_run["packet"], families=e_run["families"],
            fallback=e_run["fallback"], trace=dict(raw_hw=up["raw_hw"], initial_points=up["initial_points"]))
        image = display_image(frame, privacy)
        ref = refs[row["partition"]][fid]
        name = f"{index:03d}_" + digest_id(fid)
        thumb = "thumbnails/" + name + ".jpg"
        render(row, frame, e_run, image, ref, predictions[fid], simple, output / thumb)
        paths = dict(thumbnail=thumb)
        if fid in selected_ids:
            paths["detail"] = "details/" + name + ".png"
            render(row, frame, e_run, image, ref, predictions[fid], detailed,
                   output / paths["detail"], extras[fid], detailed=True)
            if "R_ORDER_F1_s20261018" in predictions[fid]:
                paths["support"] = "details/" + name + "_support.png"
                saved = support_panel(row, frame, image, extras[fid],
                    predictions[fid]["R_ORDER_F1_s20261018"]["selected_candidate_id"], output / paths["support"])
            else:
                saved = None
            if "V_ORDER_RGB_s20261018" in predictions[fid]:
                paths["visual"] = "details/" + name + "_visual.png"
                query = visual_panel(row, frame, image, visual_inputs[fid],
                    predictions[fid]["V_ORDER_RGB_s20261018"], output / paths["visual"])
            else:
                query = None
        else:
            saved, query = None, None
        transform = dict(id=fid, raw_hw=up["raw_hw"], original_image_sha256=up["image_sha256"],
            image_canvas="exact original raw extent; no crop; proportional Matplotlib display scaling",
            display_only_redaction_boxes=privacy["frames"][fid]["redactions"],
            measurement_changed=False, paths=paths, support_geometry=saved, visual_queries=query)
        transforms.append(transform)
        table = table_html(row, predictions[fid], manifest["arms"])
        caption = " · ".join(html.escape(r["arm"] + ":" + r["category"])
            for r in selection if r["selected_id"] == fid)
        links = " ".join('<a href="../' + p + '">' + html.escape(k) + "</a>" for k, p in paths.items())
        frame_html = ("<!doctype html><meta charset=utf-8><title>" + html.escape(fid)
            + "</title><style>body{font-family:sans-serif;margin:24px}img{max-width:100%}"
              "table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccd}</style>"
            + '<a href="../index.html">전체192장</a><h1>' + html.escape(fid) + "</h1><p>" + caption
            + "</p><p>" + links + "</p>" + "".join('<img src="../' + p + '" loading="lazy">'
                for k, p in paths.items() if k != "thumbnail" or "detail" not in paths)
            + "<p>실선은 반환 자세, 초록 점선은 고정 기하 참조입니다. S는 영상 관측이 아닙니다. "
              "점수는 모델별 의미가 다르며 보정된 안전 확률이 아닙니다. 개인정보 처리는 표시 사본에만 적용했습니다.</p>"
            + table)
        (output / "frames" / (name + ".html")).write_text(frame_html)
        summary = {a: dict(t=row["metrics"][a]["translation_cm"], r=row["metrics"][a]["rotation_deg"],
                          s=row["metrics"][a]["success"]) for a in simple}
        cards.append('<article data-index="' + str(index) + '" data-metrics="'
            + html.escape(json.dumps(summary), quote=True) + '"><h3>' + str(index) + " · " + html.escape(fid)
            + '</h3><a href="frames/' + name + '.html"><img loading="lazy" src="' + thumb
            + '"></a><p>' + caption + "</p><details><summary>모든 방법·seed 수치</summary>" + table + "</details></article>")
        if (index + 1) % 24 == 0:
            print("PRIMARY_GALLERY", index + 1, "/192", flush=True)
    options = "".join('<option value="' + html.escape(a) + '">' + html.escape(label(a)) + "</option>" for a in simple)
    page = ("<!doctype html><html lang=ko><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>상태별 선택 위험·정보 비교 — 전체192장</title><style>body{font-family:sans-serif;margin:18px;background:#f3f6f8}"
        "article{background:white;margin:15px 0;padding:12px;border:1px solid #ccd}img{width:100%;height:auto}"
        "table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:5px;border:1px solid #dde}"
        "header{position:sticky;top:0;background:#fff;padding:12px;z-index:2}select,input{margin:5px;padding:5px}</style>"
        "<header><h1>전체192장 — 같은 후보에서 선택만 변경</h1><p>이미지를 누르면 원영상·관측·자세·F1 양쪽 가설·조건부RGB를 병렬로 봅니다. "
        "모든 실패와 극단값을 보존했습니다. 위치 cm / 회전 ° / 공동5cm·5° 성공입니다.</p>"
        '<label>정렬 기준 방법 <select id="arm">' + options + "</select></label>"
        '<select id="sort"><option value="index">원래 순번</option><option value="t">위치 오차 큰 순</option>'
        '<option value="r">회전 오차 큰 순</option><option value="dt">N0 대비 위치 악화 큰 순</option></select>'
        '<input id="search" placeholder="영상 ID 검색"><span id="count"></span></header><main id="cards">'
        + "".join(cards) + "</main><script>const list=document.querySelector('#cards');const cards=[...list.children];"
        "function update(){let a=document.querySelector('#arm').value,k=document.querySelector('#sort').value,q=document.querySelector('#search').value;"
        "let n=0;cards.sort((x,y)=>{let X=JSON.parse(x.dataset.metrics),Y=JSON.parse(y.dataset.metrics);"
        "let v=o=>k==='dt'?(o[a].t===null||o.N0.t===null?-Infinity:o[a].t-o.N0.t):(o[a][k]??-Infinity);"
        "return k==='index'?+x.dataset.index-y.dataset.index:v(Y)-v(X)||+x.dataset.index-y.dataset.index});"
        "for(let c of cards){c.hidden=!c.querySelector('h3').textContent.includes(q);if(!c.hidden)n++;list.append(c)}"
        "document.querySelector('#count').textContent=n+'/192'};for(let s of document.querySelectorAll('select,input'))s.oninput=update;update();</script></html>")
    (output / "index.html").write_text(page)
    write(output / "GALLERY_MANIFEST.json", dict(status="GENERATED_PIXEL_REVIEW_PENDING",
        primary_images=192, detailed_unique_images=len(selected_ids),
        detail_case_selection=selection, display_transforms=transforms,
        privacy_plan_sha256=sha(privacy_path), redaction_count=26,
        all_numeric_arms=manifest["arms"], reference_sha256=evseal["reference_sha256"],
        prediction_manifest_sha256=sha(run_dir / "PREDICTION_MANIFEST.json"),
        source_sha256=sha(__file__), wall_seconds=time.monotonic() - started,
        artifact_counts=dict(PNG=len(list(output.rglob("*.png"))), JPEG=len(list(output.rglob("*.jpg"))),
                             HTML=len(list(output.rglob("*.html")))),
        original_RGB_bulk_published=False, display_only_edits=True,
        pixel_review="PENDING; generation is not evidence of pixel inspection"))


def main():
    parser = argparse.ArgumentParser()
    for name in ("run-dir", "old-run", "evaluation", "reference-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    generate(args.run_dir, args.old_run, args.evaluation, args.reference_dir, args.output)


if __name__ == "__main__":
    main()
