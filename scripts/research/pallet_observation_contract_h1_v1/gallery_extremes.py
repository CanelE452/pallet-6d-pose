"""Descriptive all-seed maxima appendix; no method/configuration is selected."""
from __future__ import annotations
import argparse
import html
import itertools
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.research.pallet_risk_information_v1.common import read, rows, sha, write
from scripts.research.pallet_boundary_action_repair_v1.display_privacy import display_image
from scripts.research.pallet_boundary_action_repair_v1.gallery import project, wire, background, observed_marks, observations_panel
from .gallery import label, text_metric, digest, COLORS


def generate(run_dir, old_run, reference_dir, public):
    run_dir, old_run, reference_dir, public = map(Path, (run_dir, old_run, reference_dir, public))
    start = time.monotonic()
    evaluation = run_dir / "evaluation_v1"
    es = read(evaluation / "EVALUATION_SEAL.json")
    assert es["prediction_manifest_sha256"] == sha(run_dir / "PREDICTION_MANIFEST.json")
    assert es["reference_sha256"] == sha(reference_dir / "FIXED_REFERENCES.json")
    scored = [r for r in rows(evaluation / "SCORED_ROWS.jsonl.gz") if r["partition"] != "CAL"]
    manifest = read(public / "gallery/GALLERY_MANIFEST.json")
    old_artifacts = dict(manifest["artifact_sha256"])
    output = public / "gallery/extremes"
    output.mkdir(exist_ok=False)
    new = [a for a in manifest["all_numeric_arms"] if a.startswith(("H1_NO_B_FEATURE_s", "H1_GUARDED_SWITCH_s", "H1_GUARD_CALIBRATED_TRIAL_s"))]
    chosen = []
    for axis in ("translation_cm", "rotation_deg"):
        cases = [(r["metrics"][a][axis], digest(r["id"]), a, r) for r in scored for a in new if r["metrics"][a][axis] is not None]
        value, _, arm, row = sorted(cases, key=lambda x: (-x[0], x[1], x[2]))[0]
        chosen.append(dict(axis=axis, value=value, arm=arm, id=row["id"], row=row))
    plan = dict(schema="DESCRIPTIVE_ALL_SEED_EXTREMES_PRESENTATION_V1",
        status="POSTSEAL_PRESENTATION_ONLY_NO_METHOD_CHANGE", rule="For each of T/R take maximum across all9 new arms and all192 images; ties hashed frameID then arm string",
        selected=[{k:v for k,v in c.items() if k != "row"} for c in chosen],
        original192_thumbnails_and10_details_unchanged=True, automatic_model_or_threshold_selection=False)
    write(run_dir / "EXTREME_PRESENTATION_PLAN.json", plan)
    refs = read(reference_dir / "FIXED_REFERENCES.json")["references"]
    upstream = {r["id"]: r for r in rows(old_run / "d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz")}
    order = {r["id"]: i for i, r in enumerate(rows(old_run / "d1_real_candidates_v1/HEAD_FEATURES.jsonl.gz"))}
    privacy = read(old_run / "DISPLAY_PRIVACY_PLAN.json")
    records = []
    for case in chosen:
        row, fid = case["row"], case["id"]
        seed = case["arm"].rsplit("_s", 1)[1]
        suffix = "_s" + seed
        methods = ["N0", "E0", "D1_H1_MLP" + suffix, "H1_NO_B_FEATURE" + suffix,
                   "R_CONT_F0" + suffix, "H1_GUARDED_SWITCH" + suffix,
                   "H1_GUARD_CALIBRATED_TRIAL" + suffix, "POSEFIX_SUBPIX_YOLO_ANCHOR"]
        e = next(rows(old_run / "d1_real_candidates_v1/frames" / f"{order[fid]:04d}" / "E_RUN.jsonl.gz"))
        up = upstream[fid]
        frame = dict(up, packet=e["packet"], families=e["families"], fallback=e["fallback"], trace=dict(raw_hw=up["raw_hw"], initial_points=up["initial_points"]))
        image = display_image(frame, privacy)
        ref = refs[row["partition"]][fid]
        fig, axes = plt.subplots(3, 4, figsize=(18, 10.5))
        fig.suptitle(f"Descriptive all-seed maximum {case['axis']}: {case['value']:.6f} / {case['arm']}\n{fid} / same-seed comparison {seed}; unadopted trials remain diagnostic", fontsize=10)
        background(axes.flat[0], frame, image)
        axes.flat[0].set_title("Original RGB / approved display-only privacy edits", fontsize=8)
        observations_panel(axes.flat[1], frame, e, image)
        for ax, arm, color in zip(list(axes.flat)[2:], methods, itertools.cycle(COLORS)):
            background(ax, frame, image)
            uv, z = project(row["poses"][arm], frame)
            wire(ax, uv, z, color)
            uv, z = project(dict(available=True, R_physical=ref["R"], centroid=ref["t"]), frame)
            wire(ax, uv, z, "#159260", "--", marker="s", alpha=.6)
            observed_marks(ax, frame["packet"]["solver_input"])
            ax.set_title(label(arm) + (" / seed" + seed if arm.endswith(suffix) else "") + "\n" + text_metric(row["metrics"][arm]) + "\nsource=" + row["action_sources"][arm], fontsize=7)
        for ax in list(axes.flat)[10:]:
            ax.set_visible(False)
        fig.text(.5, .012, "Full original canvas. Solid: returned model pose; green dashed: fixed geometric reference; hollow B/I: image observations. S: model projection.\n"
                 "This panel selects a descriptive maximum, not a model/seed for adoption. Display-only redactions; all numerical errors remain unchanged.", ha="center", fontsize=6.5)
        fig.tight_layout(rect=(0, .05, 1, .94), h_pad=3.)
        name = case["axis"] + "_maximum_all_seeds.png"
        fig.savefig(output / name, dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        record = {k:v for k,v in case.items() if k != "row"}
        record.update(path="extremes/" + name, sha256=sha(output / name), seed=int(seed), methods=methods)
        records.append(record)
    page = "<!doctype html><html lang=ko><meta charset=utf-8><title>새 경로 전체 seed 최대 오차</title><style>body{font-family:sans-serif;margin:24px}img{width:100%;height:auto}</style><a href='../index.html'>전체192장</a><h1>새 경로·전체 seed의 최대 위치·회전 오차</h1><p>모든192장과 새9경로에서 각 축의 최대값을 결정적으로 찾은 설명용 패널입니다. 성능에 맞춰 seed나 정책을 채택하지 않았습니다. Trial은 미채택 진단입니다.</p>" + "".join("<h2>" + html.escape(str(r["axis"]) + " / " + str(r["value"]) + " / " + r["arm"]) + "</h2><img src='" + Path(r["path"]).name + "'>" for r in records) + "</html>"
    (output / "index.html").write_text(page)
    index_path = public / "gallery/index.html"
    old_index = index_path.read_bytes()
    (run_dir / "GALLERY_INDEX_BEFORE_EXTREMES.html").write_bytes(old_index)
    (run_dir / "GALLERY_MANIFEST_BEFORE_EXTREMES.json").write_text(__import__("json").dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    index_path.write_text(old_index.decode().replace("</header>", "<p><a href='extremes/index.html'>전체 seed 최대 오차 패널 — 208cm 사례 포함</a></p></header>", 1))
    unchanged = [p for p, digest_value in old_artifacts.items() if p != "index.html" and sha(public / "gallery" / p) == digest_value]
    assert len(unchanged) == len(old_artifacts) - 1
    manifest["presentation_appendix"] = dict(rule=plan["rule"], scope="Postseal descriptive all-seed maxima; original thumbnails/details unchanged", old_index_sha256=old_artifacts["index.html"], unchanged_original_artifacts=len(unchanged), selected=records, source_sha256=sha(__file__))
    manifest["artifact_sha256"] = {str(p.relative_to(public / "gallery")): sha(p) for p in sorted((public / "gallery").rglob("*")) if p.is_file() and p.name != "GALLERY_MANIFEST.json"}
    write(public / "gallery/GALLERY_MANIFEST.json", manifest, overwrite=True)
    write(run_dir / "EXTREME_PRESENTATION_RECEIPT.json", dict(**plan, artifacts=records, new_pose_solves=0, new_fits=0, source_sha256=sha(__file__), wall_seconds=time.monotonic() - start))


def main():
    p = argparse.ArgumentParser()
    for n in ("run-dir", "old-run", "reference-dir", "public"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    generate(a.run_dir, a.old_run, a.reference_dir, a.public)


if __name__ == "__main__":
    main()
