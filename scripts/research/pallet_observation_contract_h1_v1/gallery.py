"""Postseal, full-canvas PRIMARY192 panels and parallel method gallery."""
from __future__ import annotations
import argparse
import hashlib
import html
import itertools
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.research.pallet_boundary_action_repair_v1.display_privacy import display_image
from scripts.research.pallet_boundary_action_repair_v1.gallery import (
    project, wire, background, observations_panel, observed_marks,
)
from scripts.research.pallet_risk_information_v1.common import read, rows, sha, write
from .registry import BASIC_GROUPS, AUGMENTATION_GROUPS

NAMES = {"N0": "N3 + SubPix (N0)", "E0": "Full statewise (E0)",
    "E0_REGENERATED": "E0 fixed replay", "D1_H1_MLP": "Original H1",
    "D1_H1_PAIR_MLP": "Two-action MLP", "D1_H1_PAIR_ROLE_AFFINE": "Two-action affine",
    "R_CONT_F0": "Continuous regression", "H1_NO_B_FEATURE": "H1 without B features",
    "H1_GUARDED_SWITCH": "Guarded H1 / active policy",
    "H1_GUARD_CALIBRATED_TRIAL": "Calibration trial / NOT ADOPTED",
    "POSEFIX_SUBPIX_PF_ANCHOR": "PoseFix + SubPix / PF anchor",
    "POSEFIX_SUBPIX_YOLO_ANCHOR": "PoseFix + SubPix / YOLO anchor",
    "AUG_RANDOM_H1": "H1 / simple data addition", "AUG_STRUCTURED_H1": "H1 / structural data addition"}
COLORS = ("#777777", "#2166ac", "#db7c20", "#7b3294", "#008d99", "#c51b3a",
          "#b2379e", "#333333", "#6b8132", "#a17a28", "#3a8f78", "#795548")


def group(arm):
    suffix = arm.rsplit("_s", 1)
    return suffix[0] if len(suffix) == 2 and suffix[1].isdigit() else arm


def label(arm):
    return NAMES.get(group(arm), arm)


def digest(fid):
    return hashlib.sha256(fid.encode()).hexdigest()[:16]


def text_metric(metric):
    if not metric["pose_available"]:
        return "NO POSE / failure"
    return f"{metric['translation_cm']:.3f} cm / {metric['rotation_deg']:.3f} deg / " + ("success" if metric["success"] else "failure")


def choose_cases(scored, new_arms):
    """Frozen deterministic H1 contrasts, including nonchange and failure harm."""
    result = []
    h1 = "D1_H1_MLP_s20261018"
    for arm in new_arms:
        for axis in ("translation_cm", "rotation_deg"):
            for name, sign in (("worsening", 1), ("improvement", -1)):
                eligible = [r for r in scored if r["metrics"][arm][axis] is not None and r["metrics"][h1][axis] is not None]
                eligible = [(sign * (r["metrics"][arm][axis] - r["metrics"][h1][axis]), r) for r in eligible]
                eligible = sorted((x for x in eligible if x[0] > 0), key=lambda x: (-x[0], digest(x[1]["id"])))
                result.append(dict(arm=arm, category=axis + "_" + name, comparator=h1,
                    population_count=len(eligible), selected_id=eligible[0][1]["id"] if eligible else None))
        for name, test in (("recovery", lambda b, a: not b and a), ("damage", lambda b, a: b and not a)):
            eligible = sorted((r for r in scored if test(r["metrics"][h1]["success"], r["metrics"][arm]["success"])), key=lambda r: digest(r["id"]))
            result.append(dict(arm=arm, category=name, comparator=h1, population_count=len(eligible), selected_id=eligible[0]["id"] if eligible else None))
        eligible = [r for r in scored if not r["metrics"][h1]["success"] and not r["metrics"][arm]["success"]
                    and r["metrics"][arm]["translation_cm"] is not None and r["metrics"][h1]["translation_cm"] is not None
                    and r["metrics"][arm]["translation_cm"] > r["metrics"][h1]["translation_cm"]]
        eligible.sort(key=lambda r: (-(r["metrics"][arm]["translation_cm"] - r["metrics"][h1]["translation_cm"]), digest(r["id"])))
        result.append(dict(arm=arm, category="both_failure_max_position_worsening", comparator=h1,
            population_count=len(eligible), selected_id=eligible[0]["id"] if eligible else None))
        unchanged = sorted((r for r in scored if r["poses"][arm] == r["poses"][h1]), key=lambda r: digest(r["id"]))
        result.append(dict(arm=arm, category="unchanged_actual_H1_pose", comparator=h1,
            population_count=len(unchanged), selected_id=unchanged[0]["id"] if unchanged else None))
    return result


def render(row, frame, e_run, image, ref, predictions, methods, path, detail=False):
    count = len(methods) + 2
    nrows = int(np.ceil(count / 4))
    fig, axes = plt.subplots(nrows, 4, figsize=(17.5 if detail else 14.5, nrows * (3.25 if detail else 2.7)), squeeze=False)
    fig.suptitle(row["id"] + " / " + row["partition"] + " / fixed representative seed20261018", fontsize=10)
    background(axes.flat[0], frame, image)
    axes.flat[0].set_title("Original image / display-only privacy edits", fontsize=8)
    observations_panel(axes.flat[1], frame, e_run, image)
    for ax, arm, color in zip(list(axes.flat)[2:], methods, itertools.cycle(COLORS)):
        background(ax, frame, image)
        uv, z = project(row["poses"][arm], frame)
        wire(ax, uv, z, color)
        uv, z = project(dict(available=True, R_physical=ref["R"], centroid=ref["t"]), frame)
        wire(ax, uv, z, "#159260", "--", marker="s", alpha=.6)
        observed_marks(ax, frame["packet"]["solver_input"])
        title = label(arm) + "\n" + text_metric(row["metrics"][arm]) + "\nsource=" + row["action_sources"][arm]
        if "guard_decision" in predictions[arm]:
            d = predictions[arm]["guard_decision"]
            title += "; SWITCH" if d["changed"] else "; H1 KEPT"
            title += "\n" + d["reason"]
        ax.set_title(title, fontsize=6.8 if detail else 7,
                     bbox=dict(facecolor="white", edgecolor="none", pad=2))
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    fig.text(.5, .012, "Solid/circles: returned model pose. Green dashed/squares: fixed geometric reference. Hollow B/I: image observations.\n"
        "S points are pose projections, not image observations. Errors describe the whole returned pose. Full original canvas; privacy edits affect display copies only.",
        ha="center", fontsize=6.2)
    fig.tight_layout(rect=(0, .05, 1, .96), h_pad=4., w_pad=1.1)
    if detail:
        fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(path, dpi=65, bbox_inches="tight", facecolor="white", pil_kwargs=dict(quality=82, optimize=True))
    plt.close(fig)


def table(row, predictions, methods):
    body = []
    for arm in methods:
        p, m = predictions[arm], row["metrics"][arm]
        detail = p.get("guard_decision", {}).get("reason", str(p.get("prediction_status", p.get("status", ""))))
        body.append("<tr><td>" + html.escape(label(arm)) + "<br><small>" + html.escape(arm) + "</small></td><td>" +
            html.escape(text_metric(m)) + "</td><td>" + html.escape(row["action_sources"][arm]) + "</td><td>" +
            html.escape(detail) + (" · 연구용 미채택 정책" if p.get("diagnostic_trial") else "") + "</td></tr>")
    return "<table><thead><tr><th>방법·seed</th><th>전체 자세 오차·성공</th><th>실제 출처</th><th>실행·복귀</th></tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def generate(run_dir, old_run, evaluation, reference_dir, output):
    run_dir, old_run, evaluation, reference_dir, output = map(Path, (run_dir, old_run, evaluation, reference_dir, output))
    start = time.monotonic()
    manifest = read(run_dir / "PREDICTION_MANIFEST.json")
    ev = read(evaluation / "EVALUATION_SEAL.json")
    assert ev["prediction_manifest_sha256"] == sha(run_dir / "PREDICTION_MANIFEST.json")
    assert sha(reference_dir / "FIXED_REFERENCES.json") == ev["reference_sha256"]
    refs = read(reference_dir / "FIXED_REFERENCES.json")["references"]
    scored = [r for r in rows(evaluation / "SCORED_ROWS.jsonl.gz") if r["partition"] != "CAL"]
    assert len(scored) == 192
    predictions = {r["id"]: r["predictions"] for r in rows(run_dir / "PREDICTIONS.jsonl.gz")}
    upstream = {r["id"]: r for r in rows(old_run / "d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz")}
    order = {r["id"]: i for i, r in enumerate(rows(old_run / "d1_real_candidates_v1/HEAD_FEATURES.jsonl.gz"))}
    privacy_path = old_run / "DISPLAY_PRIVACY_PLAN.json"
    privacy = read(privacy_path)
    assert set(privacy["frames"]) == {r["id"] for r in scored}
    new = [g + "_s20261018" for g in BASIC_GROUPS + AUGMENTATION_GROUPS if g + "_s20261018" in manifest["arms"]]
    simple = ["D1_H1_MLP_s20261018", "H1_NO_B_FEATURE_s20261018", "R_CONT_F0_s20261018",
              "H1_GUARDED_SWITCH_s20261018", "D1_H1_PAIR_ROLE_AFFINE_s20261018", "POSEFIX_SUBPIX_YOLO_ANCHOR"]
    detailed = ["N0", "E0", "D1_H1_MLP_s20261018", "H1_NO_B_FEATURE_s20261018", "R_CONT_F0_s20261018",
                "H1_GUARDED_SWITCH_s20261018", "H1_GUARD_CALIBRATED_TRIAL_s20261018",
                "D1_H1_PAIR_MLP_s20261018", "D1_H1_PAIR_ROLE_AFFINE_s20261018",
                "POSEFIX_SUBPIX_PF_ANCHOR", "POSEFIX_SUBPIX_YOLO_ANCHOR"]
    detailed += [a for a in new if a.startswith("AUG_")]
    selection = choose_cases(scored, new)
    selected_ids = {s["selected_id"] for s in selection if s["selected_id"]}
    output.mkdir(parents=True, exist_ok=False)
    for n in ("thumbnails", "details", "frames"):
        (output / n).mkdir()
    cards, transforms = [], []
    style = "body{font-family:sans-serif;margin:20px;background:#f3f6f8}article{background:white;margin:16px 0;padding:14px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:6px;border:1px solid #ccd}header{position:sticky;top:0;background:white;padding:12px;z-index:2}select,input{padding:6px;margin:4px}"
    for index, row in enumerate(scored):
        fid = row["id"]
        up = upstream[fid]
        e_run = next(rows(old_run / "d1_real_candidates_v1/frames" / f"{order[fid]:04d}" / "E_RUN.jsonl.gz"))
        assert e_run["id"] == fid
        frame = dict(up, packet=e_run["packet"], families=e_run["families"], fallback=e_run["fallback"],
                     trace=dict(raw_hw=up["raw_hw"], initial_points=up["initial_points"]))
        image = display_image(frame, privacy)
        ref = refs[row["partition"]][fid]
        name = f"{index:03d}_" + digest(fid)
        paths = dict(thumbnail="thumbnails/" + name + ".jpg")
        render(row, frame, e_run, image, ref, predictions[fid], simple, output / paths["thumbnail"])
        if fid in selected_ids:
            paths["detail"] = "details/" + name + ".png"
            render(row, frame, e_run, image, ref, predictions[fid], detailed, output / paths["detail"], True)
        caption = " · ".join(s["arm"] + ":" + s["category"] for s in selection if s["selected_id"] == fid)
        numeric = table(row, predictions[fid], manifest["arms"])
        selected_image = paths.get("detail", paths["thumbnail"])
        frame_html = "<!doctype html><html lang=ko><meta charset=utf-8><title>" + html.escape(fid) + "</title><style>" + style + "</style><a href='../index.html'>전체192장</a><h1>" + html.escape(fid) + "</h1><p>" + html.escape(caption) + "</p><img src='../" + selected_image + "'><p>초록 점선은 고정 기하 참조, 색 실선은 반환 자세입니다. 빈 B/I 표시는 영상 관측이며 S는 투영입니다. Trial은 채택되지 않은 진단용 정책입니다. 원영상 전체 범위를 보존했고 개인정보 처리는 표시 사본에만 적용했습니다.</p>" + numeric + "</html>"
        (output / "frames" / (name + ".html")).write_text(frame_html)
        metrics = {a: dict(t=row["metrics"][a]["translation_cm"], r=row["metrics"][a]["rotation_deg"], s=row["metrics"][a]["success"]) for a in simple}
        cards.append('<article data-index="' + str(index) + '" data-metrics="' + html.escape(json.dumps(metrics), quote=True) + '"><h3>' + str(index) + " · " + html.escape(fid) + '</h3><a href="frames/' + name + '.html"><img loading="lazy" src="' + paths["thumbnail"] + '"></a><p>' + html.escape(caption) + '</p><details><summary>모든 방법·seed·복귀 수치</summary>' + numeric + '</details></article>')
        transforms.append(dict(id=fid, raw_hw=up["raw_hw"], original_image_sha256=up["image_sha256"],
            original_canvas_preserved=True, no_crop=True, display_only_redactions=privacy["frames"][fid]["redactions"],
            paths=paths, inference_or_measurement_changes=False))
        if (index + 1) % 24 == 0:
            print("GALLERY", index + 1, "/192", flush=True)
    options = "".join('<option value="' + a + '">' + html.escape(label(a)) + '</option>' for a in simple)
    page = "<!doctype html><html lang=ko><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>H1 관측·제한적 변경 전체192장</title><style>" + style + "</style><header><h1>전체192장 — H1·B 특징 제거·회귀·제한적 변경 병렬 비교</h1><p>동일한 저장 후보에서 선택했습니다. 위치 cm / 회전 °, 같은 자세의 공동5cm·5° 성공입니다. 모든 실패와 극단값을 보존합니다. 썸네일을 누르면 상세 비교와 모든 seed 수치를 볼 수 있습니다.</p><select id=arm>" + options + "</select><select id=sort><option value=index>원래 순번</option><option value=t>위치 오차 큰 순</option><option value=r>회전 오차 큰 순</option><option value=dt>H1 대비 위치 악화 큰 순</option></select><input id=search placeholder='영상 ID 검색'><span id=count></span></header><main id=cards>" + "".join(cards) + "</main><script>const list=document.querySelector('#cards'),cards=[...list.children];function update(){let a=document.querySelector('#arm').value,k=document.querySelector('#sort').value,q=document.querySelector('#search').value;let n=0;cards.sort((x,y)=>{let X=JSON.parse(x.dataset.metrics),Y=JSON.parse(y.dataset.metrics);let v=o=>k==='dt'?(o[a].t===null||o.D1_H1_MLP_s20261018.t===null?-Infinity:o[a].t-o.D1_H1_MLP_s20261018.t):(o[a][k]??-Infinity);return k==='index'?+x.dataset.index-y.dataset.index:v(Y)-v(X)||+x.dataset.index-y.dataset.index});for(let c of cards){c.hidden=!c.querySelector('h3').textContent.includes(q);if(!c.hidden)n++;list.append(c)}document.querySelector('#count').textContent=n+'/192'}for(let s of document.querySelectorAll('select,input'))s.oninput=update;update();</script></html>"
    (output / "index.html").write_text(page)
    write(output / "GALLERY_MANIFEST.json", dict(status="GENERATED_PIXEL_REVIEW_PENDING", primary_images=192,
        detailed_unique_images=len(selected_ids), detail_case_selection=selection, display_transforms=transforms,
        all_numeric_arms=manifest["arms"], privacy_plan_sha256=sha(privacy_path),
        prediction_manifest_sha256=sha(run_dir / "PREDICTION_MANIFEST.json"), reference_sha256=ev["reference_sha256"],
        source_sha256=sha(__file__), wall_seconds=time.monotonic() - start,
        artifact_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
        original_RGB_bulk_published=False, pixel_review="PENDING: generation is not pixel inspection"))


def main():
    p = argparse.ArgumentParser()
    for n in ("run-dir", "old-run", "evaluation", "reference-dir", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    generate(a.run_dir, a.old_run, a.evaluation, a.reference_dir, a.output)


if __name__ == "__main__":
    main()
