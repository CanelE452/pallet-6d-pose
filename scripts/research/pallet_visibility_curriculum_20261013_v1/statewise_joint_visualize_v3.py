"""Render sealed v3 actual RGB records, including finite feasible pose families.

This is a read-only research renderer. It performs no feature extraction, pose
estimation, or scoring. Annotation overlays require the completed postseal CAL
score. Earlier renderers and all inference sources remain unchanged.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import shutil

import cv2
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np

from . import common as C
from . import statewise_joint_visualize_v2 as V


STUDY = V.STUDY
DEFAULT_IDS = (
    "eval_night08:1779449488068910592",
    "eval_outside:1778651519235162880",
    "eval_night08:1779449488068910592|topcrop325",
    "eval_night08:1779449499143611648|topcrop334",
    "eval_night08:1779449501478488320|topcrop340",
)


def _font():
    path = Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def _non_s_raw(runtime):
    # Never use S values to choose the viewport or draw image-stage queries.
    raw = np.full((8, 2), np.nan)
    for k, state in enumerate(runtime["states"]):
        if state != "S":
            point = V._point(runtime["original_raw_points"][k])
            if point is not None:
                raw[k] = point
    return raw


def _box(ax, text, *, xy=(.02, .025), color="white", bg="#111827", fontsize=8):
    ax.text(*xy, text, transform=ax.transAxes, va="bottom", fontsize=fontsize,
            color=color, bbox=dict(facecolor=bg, edgecolor="none", alpha=.90, pad=4),
            zorder=15)


def _samples(runtime):
    joint = runtime["joint_pose"]
    ambiguity = joint.get("ambiguity") or {}
    family = bool(ambiguity.get("pose_family_from_two_BI") or
                  (runtime["accepted"] and ambiguity.get("detected")))
    values = (joint.get("pose_family_candidates") or joint.get("feasible_pose_samples", [])) if family else []
    if family and not values:
        raise AssertionError("An ambiguous family figure requires saved family samples")
    if any(sample.get("all_OT_individual_constraints_pass") is not True for sample in values):
        raise AssertionError("Only saved all-OT-feasible samples may be drawn")
    arrays = np.asarray([s["projected_native"] for s in values], float)
    if values and (arrays.shape != (len(values), 8, 2) or not np.isfinite(arrays).all()):
        raise AssertionError("Family projections must be saved finite native-8 values")
    return family, arrays


def _wireframe(ax, points, states, *, color="#d3e0ef", alpha=.4, linewidth=.8):
    for a, b in V.EDGES:
        # U remains unresolved; do not display a predicted U location as a readout.
        if "U" not in (states[a], states[b]) and np.isfinite(points[[a, b]]).all():
            ax.plot(points[[a, b], 0], points[[a, b], 1], color=color,
                    alpha=alpha, lw=linewidth, zorder=2)


def _wireframe_ids(samples, states):
    if len(samples) <= 24:
        return list(range(len(samples)))
    # Predefined, GT-independent display selection: uniform indices + extrema.
    ids = {0, *np.linspace(0, len(samples)-1, 24).astype(int).tolist()}
    for k, state in enumerate(states):
        if state in ("O", "T", "S"):
            for axis in (0, 1):
                ids.add(int(np.argmin(samples[:, k, axis])))
                ids.add(int(np.argmax(samples[:, k, axis])))
    return sorted(ids)


def _ranges(samples, states):
    return {str(k): {"min_xy": samples[:, k].min(0).tolist(),
                     "max_xy": samples[:, k].max(0).tolist()}
            for k, state in enumerate(states) if state in ("O", "T", "S")}


def _family_range_plot(samples, states, image, output, record_id, gt):
    """Show all saved sample projections on signed-log axes, including outliers."""
    ids = [k for k, s in enumerate(states) if s in ("O", "T", "S")]
    fig, axes = plt.subplots(1, len(ids), figsize=(4*len(ids), 4.8))
    h, w = image.shape[:2]
    for ax, k in zip(np.atleast_1d(axes), ids):
        state = states[k]
        points = samples[:, k]
        ax.scatter(points[:, 0], points[:, 1], s=12, alpha=.45,
                   color=V.COLORS[state], marker="s" if state == "S" else "o")
        ax.add_patch(Rectangle((0, 0), w, h, edgecolor="#168a9b", facecolor="none", lw=1.5))
        # No S GT coordinates: every displayed S coordinate is final projection.
        if state != "S" and gt is not None and V._point(gt[k]) is not None:
            ax.scatter(*gt[k], marker="+", color="#d4a800", s=55, lw=1.5, zorder=5)
        ax.set_xscale("symlog", linthresh=100)
        ax.set_yscale("symlog", linthresh=100)
        ax.xaxis.get_major_locator().set_params(numticks=7)
        ax.yaxis.get_major_locator().set_params(numticks=8)
        xlo, xhi = ax.get_xlim()
        coarse_ticks = [0., *[sign*10.**power for sign in (-1, 1) for power in range(2, 10, 2)]]
        ax.set_xticks(sorted(x for x in coarse_ticks if xlo <= x <= xhi))
        # ASCII minus signs avoid missing math glyphs in the Korean font.
        ticks = FuncFormatter(lambda value, _: "0" if value == 0 else
                              str(int(value)) if abs(value) < 1000 else f"{value:.0e}")
        ax.xaxis.set_major_formatter(ticks)
        ax.yaxis.set_major_formatter(ticks)
        ax.tick_params(labelsize=8)
        ax.invert_yaxis()
        suffix = " · pose only" if state == "S" else " · conditional T" if state == "T" else " · empirical O domain"
        ax.set_title(state+str(k)+suffix, fontsize=10)
        ax.set_xlabel("x px · signed-log scale")
        ax.set_ylabel("y px · signed-log scale")
        ax.grid(alpha=.2)
        lo, hi = points.min(0), points.max(0)
        ax.text(.02, -.27, "x [%s, %s]\ny [%s, %s] px" %
                tuple(f"{v:.6g}" for v in (lo[0], hi[0], lo[1], hi[1])),
                transform=ax.transAxes, fontsize=8)
    fig.suptitle(record_id+"\n가능한 자세 집합·단일 정밀해 아님", fontsize=12)
    fig.text(.01, .02, f"All {len(samples)} saved feasible search samples; finite sample set, not the entire feasible set or a confidence bound.\n"
             "Cyan rectangle: actual RGB canvas. S coordinates: final pose projection only. U coordinates omitted.", fontsize=9)
    fig.subplots_adjust(left=.05, right=.99, top=.80, bottom=.30, wspace=.35)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def render_record(image, runtime, *, record_id, output, gt, truth_note, actual_T):
    states = runtime["states"]
    raw = _non_s_raw(runtime)
    family, samples = _samples(runtime)
    accepted = bool(runtime["accepted"])
    measurements = runtime["frozen_visible_measurements"]
    domains = runtime["ot_domains"]["domains"]
    self_ids = [k for k, s in enumerate(states) if s == "S"]
    # V._roi receives only sanitized non-S coordinates.
    view_runtime = {"states": states, "original_raw_points": raw}
    limits = V._roi(image, view_runtime)
    h, w = image.shape[:2]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.4))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    titles = ("1  Raw non-S image predictions", "2  Frozen corrected B/I + simultaneous O/T domains",
              "3  Saved feasible pose samples" if family else
              "3  Saved joint representative" if accepted else "3  Joint PnP rejected · O/T unresolved")
    for ax, title in zip(axes, titles):
        ax.imshow(rgb)
        ax.set_xlim(limits[:2]); ax.set_ylim(limits[2:])
        ax.set_title(title, fontsize=10, loc="left", pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        if "T" in states or actual_T:
            ax.set_facecolor("#ffe8e8")
            ax.add_patch(Rectangle((-.5, -.5), w, h, facecolor="none",
                                  edgecolor=V.COLORS["T"], lw=1.4, zorder=3))
    for k, state in enumerate(states):
        if state != "S":
            V._mark(axes[0], raw[k], state, k, marker="x", label_suffix=" pred")
    initial_note = "S"+"/S".join(map(str, self_ids))+" excluded before RGB\nNo S coordinate or image query in panels 1–2"
    if actual_T:
        initial_note += "\nPostseal actual T "+str(actual_T)+" / conditional predicted T "+str([k for k, s in enumerate(states) if s == "T"])
    _box(axes[0], initial_note)
    V._domains(axes[1], domains)
    for m in measurements:
        k = m["native_corner"]
        p = V._point(raw[k]); q = V._point(m["xy"])
        if p is not None and q is not None and np.linalg.norm(p-q) > .1:
            axes[1].annotate("", q, xytext=p,
                             arrowprops=dict(arrowstyle="->", color="white", lw=1.3), zorder=4)
        V._mark(axes[1], m["xy"], m["state"], k, label_suffix=" fixed")
        V._mark(axes[2], m["xy"], m["state"], k, label_suffix=" fixed")
    _box(axes[1], "B/I image observations unchanged\nO: CAL empirical domains · coverage unproved", fontsize=7)
    clipped = {}
    wire_ids = []
    ranges = {}
    if family:
        wire_ids = _wireframe_ids(samples, states)
        for i in wire_ids:
            _wireframe(axes[2], samples[i], states, alpha=.35, linewidth=.8)
        for k, state in enumerate(states):
            if state not in ("O", "T", "S"):
                continue
            p = samples[:, k]
            axes[2].scatter(p[:, 0], p[:, 1], s=15, linewidth=.6, alpha=.60,
                            marker="s" if state == "S" else "D", facecolor="none",
                            edgecolor=V.COLORS[state], zorder=4)
            # Label one saved sample; no median/interpolated point is invented.
            V._mark(axes[2], p[0], state, k, marker="s" if state == "S" else "D", hollow=True,
                    label_suffix=" pose only" if state == "S" else " family")
            x0, x1, y1, y0 = limits
            clipped[str(k)] = int(((p[:, 0] < x0) | (p[:, 0] > x1) |
                                   (p[:, 1] < y0) | (p[:, 1] > y1)).sum())
        ranges = _ranges(samples, states)
        _box(axes[2], "가능한 자세 집합·단일 정밀해 아님\n"
             f"{len(samples)} found search samples · not all possible poses\nU stays unresolved; S = final projection only", fontsize=8)
    elif accepted:
        rigid = np.asarray(runtime["keypoint_readout"]["rigid_model_corners"], float)
        _wireframe(axes[2], rigid, states, alpha=.75)
        for k, state in enumerate(states):
            if state in ("O", "T", "S"):
                V._mark(axes[2], runtime["keypoint_readout"]["reported_points"][k], state, k,
                        marker="s" if state == "S" else "D", hollow=state == "S",
                        label_suffix=" pose only" if state == "S" else " joint")
        _box(axes[2], "Accepted saved representative · ambiguity flag false\nGlobal uniqueness has not been proved", fontsize=8)
    else:
        # Saved fallback S projection only; do not imply O/T were solved.
        for k in self_ids:
            V._mark(axes[2], runtime["keypoint_readout"]["reported_points"][k], "S", k,
                    marker="s", hollow=True, label_suffix=" fallback pose only")
        _box(axes[2], f"REJECTED: {len(measurements)} usable fixed B/I\n"
             "O/T remain unresolved; S shown from fallback pose only", color="#a52a2a", bg="white")
    # All GT is postseal CAL annotation. S annotation is deliberately omitted.
    for ax in axes:
        for k, point in enumerate(gt):
            if states[k] != "S" and V._point(point) is not None:
                ax.scatter(*point, s=28, marker="+", color="#ffff00", lw=1.0, zorder=7)
    fig.suptitle(record_id+"\nStates (native 0–7): "+" ".join(states), fontsize=11, x=.025, ha="left", y=.98)
    caption = truth_note+"\nYellow + = saved postseal CAL coordinate annotation; S GT omitted. Seen-CAL illustration, not unseen generalization."
    if family:
        caption += f"\nFinite search samples only; {len(wire_ids)} wireframes displayed, all {len(samples)} O/T/S sample points displayed."
        if any(clipped.values()):
            caption += "  Points outside this viewport: "+", ".join(states[int(k)]+k+"="+str(n) for k, n in clipped.items() if n)+"."
    fig.text(.025, .045, caption, fontsize=9, ha="left", va="bottom", linespacing=1.6)
    fig.subplots_adjust(left=.025, right=.995, bottom=.20, top=.86, wspace=.055)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    range_file = None
    if family and any(clipped.values()) and len(samples) > 24:
        range_file = output.with_name(output.name+"__all_saved_projection_ranges")
        _family_range_plot(samples, states, image, range_file, record_id, gt)
    return dict(record_id=record_id, accepted=accepted, states=states, ambiguity_family=family,
                found_finite_search_sample_count=len(samples), wireframe_sample_indices=wire_ids,
                samples_are_not_entire_feasible_set=True, sample_envelope_is_not_confidence_bound=True,
                outside_viewport_sample_count_by_native_id=clipped, saved_sample_coordinate_ranges=ranges,
                S_raw_coordinates_plotted=False, S_GT_coordinates_plotted=False,
                S_only_final_pose_projection=True, U_final_coordinates_inferred_or_plotted=False,
                U_raw_image_prediction_visible_in_panel1=any(s == "U" for s in states),
                BI_coordinates_source="exact saved frozen_visible_measurements",
                predicted_conditional_T_ids=[k for k, s in enumerate(states) if s == "T"],
                actual_postseal_source_visible_T_ids=actual_T,
                physical_truncation_success_claimed=False, truth_note=truth_note,
                png=output.with_suffix(".png").name, svg=output.with_suffix(".svg").name,
                projection_ranges_png=range_file.with_suffix(".png").name if range_file else None,
                projection_ranges_svg=range_file.with_suffix(".svg").name if range_file else None)


def render_run(run, output, select_ids=DEFAULT_IDS):
    _font()
    rows, seal = V._verified_run(run)
    score_path = Path(run)/"POSTSEAL_SCORING.json"
    scored = C.read(score_path)
    if scored["prediction_seal_sha256"] != C.sha(Path(run)/"PREDICTION_SEAL.json"):
        raise AssertionError("Postseal CAL overlays must bind to the current prediction seal")
    inputs_path = C.DOC/"nonsevere_v1/PREDICTION_INPUTS.json"
    frames = {f["id"]: f for f in C.read(inputs_path)["frames"]}
    cal_path = C.DOC/"nonsevere_v1/OOF_pose_v1/ISOLATED_CAL_SCORING.json"
    cal = C.read(cal_path)
    if cal["TEST_rows"] or cal["FINAL_rows"] or cal["severe_rows"]:
        raise AssertionError("Natural-image overlays must use isolated CAL only")
    protocol = C.read(Path(run)/"PROTOCOL.json")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output/"renderer_snapshot.py")
    shutil.copy2(Path(V.__file__), output/"renderer_v2_helper_snapshot.py")
    by_id = {r["id"]: r for r in rows}
    records = []
    for record_id in select_ids:
        row = by_id[record_id]
        if row.get("partition") != "CAL":
            raise AssertionError("This postseal annotated gallery is CAL-only")
        fid = row["source_record_id"]
        frame = frames[fid]
        source_rgb = C.ROOT/row["image"]
        expected = frame["image_sha256"] if row["image"] == frame["image"] else protocol.get("crop_source_sha256", {}).get(row["image"])
        if not expected or C.sha(source_rgb) != expected:
            raise AssertionError("Actual RGB must exactly match the sealed prediction input")
        image = cv2.imread(str(source_rgb))
        if image is None:
            raise ValueError("Could not load actual RGB: "+str(source_rgb))
        runtime = row["runtime"]
        annotation = cal["rows"][fid]
        human = annotation["metadata"]["human_states_native"][:8]
        crop = next((c for c in scored["crops"] if c["id"] == record_id), None)
        if crop:
            gt = np.asarray(crop["native_gt"], float)[:8]
            gt[~np.asarray(crop["native_gt_valid"], bool)[:8]] = np.nan
            actual_T = crop["actual_transformed_visible_T_ids"]
            predicted_T = [k for k, s in enumerate(runtime["states"]) if s == "T"]
            note = f"Actual RGB crop; physical source-visible T {actual_T}; predicted conditional T {predicted_T}. "
            if record_id.endswith("topcrop325"):
                note += "T4 missed as O4; conditional T5 feasibility does not establish physical truncation success."
            elif record_id.endswith("topcrop340"):
                note += "False T0/T3; actual T5 missed as U5. Solver feasibility is conditional on incorrect states."
            else:
                note += "Actual T5 missed as U5; joint pose rejected."
            note += "\nTranslated frozen q/K and original source geometry; no new upstream backbone crop forward."
        else:
            mapping = annotation["metadata"]["native_to_reference_corner_ids"]
            gt = np.asarray(annotation["target"]["gt"], float)[mapping]
            gt[~np.asarray(annotation["metadata"]["valid_reference_corners"], bool)[mapping]] = np.nan
            actual_T = None
            true_O = [k for k, state in enumerate(human) if state == "EXTERNAL_OCCLUDED"]
            predicted_O = [k for k, state in enumerate(runtime["states"]) if state == "O"]
            false_O = [k for k in predicted_O if k not in true_O]
            note = f"Postseal source human external O {true_O}; predicted O {predicted_O}. "
            note += f"False O {false_O}; U5 unresolved; simultaneous O domains admit a pose family." if false_O else "O4 accepted as saved joint representative; no global uniqueness proof."
        stem = record_id.replace(":", "__").replace("/", "__").replace("|", "__")
        record = render_record(image, runtime, record_id=record_id, output=output/stem,
                               gt=gt, truth_note=note, actual_T=actual_T)
        record.update(source_RGB_file=str(source_rgb.relative_to(C.ROOT)), source_RGB_sha256=C.sha(source_rgb),
                      png_sha256=C.sha(output/record["png"]), svg_sha256=C.sha(output/record["svg"]))
        records.append(record)
    index = ["<!doctype html><html lang='ko'><meta charset='utf-8'><title>Current v3 actual RGB figures</title>",
             "<style>body{font:16px sans-serif;max-width:1650px;margin:28px auto;padding:0 16px;line-height:1.55}img{width:100%}article{margin:42px 0;border-top:1px solid #bbb}code{font-size:14px}.warning{background:#fff0da;padding:12px}</style>",
             "<h1>봉인된 최신 evaluation_v3 · 실제 RGB 사례</h1>",
             "<p>CAL 사례의 사후 시각화입니다. 예측 캐시와 사후 평가의 hash 결합을 확인했으며 새 영상 추론·학습·PnP 재계산을 하지 않았습니다. S 좌표는 마지막 자세 투영에만 표시합니다. 첫 두 패널에는 S 좌표나 S 영상 query가 없습니다. 초록 B/청록 I는 저장된 보정 관측 그대로입니다.</p>",
             "<p class='warning'>가능한 자세 집합·단일 정밀해 아님: 여러 투영은 유한 검색에서 발견된 표본입니다. 전체 가능 자세를 열거했다는 뜻이나 신뢰구간이 아닙니다. Crop의 T4/T5 누락과 false T0/T3를 그대로 표시하며 물리적 절단 추론 성공이나 기존 방법보다 성능 향상을 주장하지 않습니다.</p>"]
    for r in records:
        index += ["<article><h2>"+html.escape(r["record_id"])+"</h2>",
                  "<p><code>"+html.escape(r["source_RGB_file"])+"</code><br>States: "+" ".join(r["states"])+
                  f" · accepted={r['accepted']} · pose family={r['ambiguity_family']}</p>",
                  "<p>"+html.escape(r["truth_note"]).replace("\n", "<br>")+"</p>",
                  f"<a href='{r['svg']}'>SVG</a> · <a href='{r['png']}'>PNG</a><img src='{r['png']}' alt='Actual sealed statewise pipeline'>"]
        if r["ambiguity_family"]:
            index.append(f"<p>발견된 유한 자세 표본 {r['found_finite_search_sample_count']}개. 표시 wireframe {len(r['wireframe_sample_indices'])}개; O/T/S 표본 점은 모두 표시하되 영상창 밖의 점은 잘립니다. 전체 가능 자세나 확률 경계를 뜻하지 않습니다.</p>")
        if r["projection_ranges_png"]:
            index += [f"<h3>모든 저장 표본의 투영 범위 · signed-log 좌표</h3><a href='{r['projection_ranges_svg']}'>범위 SVG</a><img src='{r['projection_ranges_png']}' alt='All saved projection sample ranges'>",
                      "<p>표시 창 밖에 있는 표본을 숨기지 않기 위해 부호를 유지하는 로그 축을 사용했습니다. 청록 직사각형은 실제 영상 경계입니다. S는 최종 투영이며 U4/U5의 위치는 추론하거나 표시하지 않습니다.</p>"]
            index.append("<table><tr><th>native ID</th><th>min x,y px</th><th>max x,y px</th></tr>")
            for k, rr in r["saved_sample_coordinate_ranges"].items():
                index.append("<tr><td>"+r["states"][int(k)]+k+"</td><td>"+html.escape(str(rr["min_xy"]))+"</td><td>"+html.escape(str(rr["max_xy"]))+"</td></tr>")
            index.append("</table>")
        index.append("</article>")
    index += ["<p><a href='FIGURE_PROVENANCE.json'>Figure provenance / exact saved sample ranges</a> · <a href='renderer_snapshot.py'>Renderer snapshot</a></p></html>"]
    (output/"index.html").write_text("\n".join(index)+"\n")
    C.write(output/"FIGURE_PROVENANCE.json", dict(schema="SEALED_ACTUAL_STATEWISE_FIGURES_V3",
        run=str(Path(run).relative_to(C.ROOT)), prediction_cache_sha256=C.sha(Path(run)/"PREDICTIONS.jsonl.gz"),
        prediction_seal_sha256=C.sha(Path(run)/"PREDICTION_SEAL.json"), postseal_score_sha256=C.sha(score_path),
        prediction_input_sha256=C.sha(inputs_path), isolated_CAL_annotation_sha256=C.sha(cal_path),
        renderer_source_sha256=C.sha(output/"renderer_snapshot.py"),
        renderer_helper_sha256=C.sha(output/"renderer_v2_helper_snapshot.py"),
        GT_overlay_postseal_CAL_only=True, poses_reestimated=False, predictions_changed=False,
        new_image_inference=False, inference_sources_modified=False, rows=records))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=STUDY/"evaluation_v3")
    parser.add_argument("--output", type=Path, default=STUDY/"current_figures_v3")
    parser.add_argument("--record-id", action="append")
    args = parser.parse_args()
    print([{k:r[k] for k in ("record_id", "accepted", "ambiguity_family", "found_finite_search_sample_count", "png")}
           for r in render_run(args.run, args.output, args.record_id or DEFAULT_IDS)])


if __name__ == "__main__":
    main()
