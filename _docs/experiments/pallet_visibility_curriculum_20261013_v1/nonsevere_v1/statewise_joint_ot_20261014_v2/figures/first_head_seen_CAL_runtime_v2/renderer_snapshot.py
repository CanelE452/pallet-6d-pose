"""Render sealed actual records: non-S RGB, frozen B/I + all O/T, joint readout.

This renderer estimates no pose and changes no model or prediction. By default
it reads no annotation file. Optional annotation overlays require a separate
post-prediction-seal scoring artifact; GT is never fabricated from a pose.
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
from pathlib import Path
import shutil

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

from . import common as C


STUDY = C.DOC / "nonsevere_v1/statewise_joint_ot_20261014_v2"
COLORS = {"B": "#25bf67", "I": "#27c0d0", "O": "#ff9800", "T": "#ee5965", "S": "#d470eb", "U": "#a8a8a8"}
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def _read_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _verified_run(run):
    run = Path(run)
    seal = C.read(run / "PREDICTION_SEAL.json")
    path = run / "PREDICTIONS.jsonl.gz"
    expected = seal.get("predictions_sha256", seal.get("sha256"))
    if not expected or C.sha(path) != expected:
        raise AssertionError("Visualization requires a correctly sealed prediction cache")
    if seal.get("before_scoring") is not True:
        raise AssertionError("Prediction cache must have been sealed before scoring")
    return _read_rows(path), seal


def _runtime(row):
    if "runtime" in row:
        return row["runtime"]
    if "result" in row:
        return row["result"]
    if "states" in row and "keypoint_readout" in row:
        return row
    raise ValueError("Unknown actual runtime row schema")


def _point(point):
    value = np.asarray(point, float)
    return value if value.shape == (2,) and np.isfinite(value).all() else None


def _mark(ax, xy, state, native_id, *, marker="o", label_suffix="", hollow=False, alpha=1.):
    point = _point(xy)
    if point is None:
        return
    color = COLORS[state]
    style=dict(color=color) if marker=="x" else dict(facecolor="none" if hollow else color,
                                                     edgecolor=color if hollow else "white")
    ax.scatter(*point, s=35 if state != "S" else 45, marker=marker,
               linewidth=1.2, alpha=alpha, zorder=5,**style)
    offsets = {0:(-24,10), 1:(5,10), 2:(5,-13), 3:(-24,-13), 4:(-18,12), 5:(5,12), 6:(8,-13), 7:(8,-16)}
    ax.annotate(state+str(native_id)+label_suffix, point, xytext=offsets.get(native_id,(5,5)),
        textcoords="offset points", fontsize=7, color=color,
        bbox=dict(facecolor="#111827", edgecolor="none", alpha=.8, pad=1.8), zorder=6)


def _domains(ax, domains):
    for domain in domains:
        native = domain.get("native_corner", domain.get("native_id"))
        state = domain["state"]
        color = COLORS[state]
        shape = domain.get("shape")
        if shape == "ellipse":
            center = domain.get("center_xy", domain.get("center"))
            axes = domain.get("half_axes_px", domain.get("axes"))
            ax.add_patch(Ellipse(center, 2*axes[0], 2*axes[1], angle=domain.get("angle_deg",0),
                edgecolor=color, facecolor=color, alpha=.17, linewidth=1.2, zorder=2))
            ax.add_patch(Ellipse(center, 2*axes[0], 2*axes[1], angle=domain.get("angle_deg",0),
                edgecolor=color, facecolor="none", linewidth=1.4, zorder=3))
            ax.text(center[0]-axes[0], center[1]-axes[1]-4, state+str(native)+" domain", color=color,
                fontsize=7, bbox=dict(facecolor="#111827", edgecolor="none", alpha=.75, pad=1))
        elif shape in ("box", "axis_aligned_box"):
            box = domain.get("xyxy", domain.get("bounds_xyxy"))
            if box is not None:
                ax.add_patch(Rectangle(box[:2], box[2]-box[0], box[3]-box[1],
                    edgecolor=color, facecolor=color, alpha=.15, linewidth=1.3))
    truncated=[d.get("native_corner",d.get("native_id")) for d in domains if d["state"]=="T"]
    if truncated:
        ax.text(.02,.12,"T%s: conditional outside-canvas union\nno finite T coordinate inferred" % "/T".join(map(str,truncated)),
            transform=ax.transAxes,color=COLORS["T"],fontsize=7,
            bbox=dict(facecolor="#111827",edgecolor="none",alpha=.8,pad=3))


def _roi(image, runtime):
    h,w = image.shape[:2]
    states = runtime["states"]
    raw = np.asarray(runtime["original_raw_points"], float)[:8]
    selected = [raw[k] for k,s in enumerate(states) if s != "S" and _point(raw[k]) is not None]
    if not selected:
        return 0,w,h,0
    q = np.asarray(selected)
    if "T" in states:
        return min(-35.,q[:,0].min()-35),max(w+35.,q[:,0].max()+35),max(h+25.,q[:,1].max()+25),min(-25.,q[:,1].min()-25)
    # This is a view transform only. S raw coordinates never enter the crop.
    x0,x1 = max(0.,q[:,0].min()-75),min(float(w),q[:,0].max()+75)
    y0,y1 = max(0.,q[:,1].min()-160),min(float(h),q[:,1].max()+70)
    if x1-x0 < 100 or y1-y0 < 100:
        return 0,w,h,0
    return x0,x1,y1,y0


def render_record(image, runtime, *, record_id, output, caption, gt_native=None):
    """Render actual saved values; preserve B/I points and mark failure explicitly."""
    states = runtime["states"]
    raw = np.asarray(runtime["original_raw_points"], float)[:8]
    measurements = runtime["frozen_visible_measurements"]
    domains = runtime["ot_domains"]["domains"]
    readout = runtime["keypoint_readout"]
    reported = readout["reported_points"]
    rigid = readout.get("rigid_model_corners")
    accepted = bool(runtime["accepted"])
    fig,axes = plt.subplots(1,3,figsize=(16,4.9),layout="constrained")
    rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    titles = ("1  Original non-S image predictions", "2  Freeze corrected B/I + all O/T domains",
              "3  Joint PnP: O/T + pose-only S" if accepted else "3  NO JOINT SOLUTION — explicit fallback")
    for ax,title in zip(axes,titles):
        ax.imshow(rgb)
        x0,x1,y1,y0 = _roi(image,runtime)
        ax.set_xlim(x0,x1)
        ax.set_ylim(y1,y0)
        ax.set_title(title,fontsize=10,loc="left",pad=7)
        ax.set_xticks([]);ax.set_yticks([])
        if "T" in states:
            ax.set_facecolor("#ffe8e8")
            ax.add_patch(Rectangle((-.5,-.5),image.shape[1],image.shape[0],facecolor="none",
                edgecolor=COLORS["T"],linewidth=1.4,zorder=3))
    for k,state in enumerate(states):
        if state != "S":
            _mark(axes[0],raw[k],state,k,marker="x")
    self_ids = [k for k,s in enumerate(states) if s == "S"]
    axes[0].text(.02,.03,"S%s excluded before RGB" % "/S".join(map(str,self_ids)),
        transform=axes[0].transAxes,fontsize=8,color="white",
        bbox=dict(facecolor="#111827",edgecolor="none",alpha=.85,pad=3))
    _domains(axes[1],domains)
    for measurement in measurements:
        k = measurement["native_corner"]
        q = _point(raw[k]);m = _point(measurement["xy"])
        if q is not None and m is not None and np.linalg.norm(q-m)>.1:
            axes[1].annotate("",m,xytext=q,arrowprops=dict(arrowstyle="->",color="white",lw=1.4),zorder=4)
        _mark(axes[1],measurement["xy"],measurement["state"],k,label_suffix=" fixed")
    axes[1].text(.02,.03,"O domains are CAL empirical uncertainty; coverage unproven",
        transform=axes[1].transAxes,fontsize=7,color="white",
        bbox=dict(facecolor="#111827",edgecolor="none",alpha=.85,pad=3))
    if accepted and rigid is not None:
        p = np.asarray(rigid,float)
        for a,b in EDGES:
            if np.isfinite(p[[a,b]]).all():
                axes[2].plot(p[[a,b],0],p[[a,b],1],color="#bedaf6",lw=.9,alpha=.65,zorder=2)
    for measurement in measurements:
        k = measurement["native_corner"]
        _mark(axes[2],measurement["xy"],measurement["state"],k)
    for k,state in enumerate(states):
        if state == "S":
            _mark(axes[2],reported[k],state,k,marker="s",hollow=True,
                  label_suffix=" pose only" if accepted else " fallback pose only")
        elif state in ("O","T") and accepted:
            _mark(axes[2],reported[k],state,k,marker="D")
    if not accepted:
        fallback = runtime.get("fallback_audit",{})
        reason = fallback.get("reason") or runtime.get("joint_pose",{}).get("fallback_reason") or "constraints not solved"
        axes[2].text(.03,.90,"O/T unresolved\n"+str(reason),transform=axes[2].transAxes,
            fontsize=8,color="#b91c1c",bbox=dict(facecolor="white",edgecolor="#b91c1c",alpha=.9,pad=5))
    dimensions=runtime.get("initial_geometry",{}).get("cf_extents")
    if dimensions:
        axes[2].text(.02,.03,"registered cuboid: "+" x ".join("%.2f" % d for d in dimensions)+" m",
            transform=axes[2].transAxes,fontsize=7,color="white",
            bbox=dict(facecolor="#111827",edgecolor="none",alpha=.85,pad=3))
    if gt_native is not None:
        for ax in axes:
            for k,point in enumerate(gt_native):
                if _point(point) is not None:
                    ax.scatter(*point,s=23,marker="+",color="#ffff00",lw=.8,zorder=4)
    fig.suptitle(record_id+"\n"+caption,fontsize=10,x=.01,ha="left")
    output=Path(output)
    fig.savefig(output.with_suffix(".png"),dpi=220,bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"),bbox_inches="tight")
    plt.close(fig)
    return dict(record_id=record_id,accepted=accepted,states=states,
        O_T_domain_ids=[d.get("native_corner",d.get("native_id")) for d in domains],S_ids=self_ids,
        plotted_S_raw_predictions=False, B_I_values_source="saved_frozen_visible_measurements",
        joint_result_is_saved_prediction=True, GT_overlay=gt_native is not None,
        image_sha256=C.sha(output.with_suffix(".png")),png=output.with_suffix(".png").name,
        svg=output.with_suffix(".svg").name,caption=caption)


def render_run(run, output, *, source_inputs=None, kind="smoke", select_ids=None, postscore_overlay=False):
    rows,seal = _verified_run(run)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),output/"renderer_snapshot.py")
    source_inputs=source_inputs or C.DOC / "nonsevere_v1/PREDICTION_INPUTS.json"
    frames={f["id"]:f for f in C.read(source_inputs)["frames"]}
    protocol=C.read(Path(run)/"PROTOCOL.json")
    scored=None
    CAL_annotations=None
    if postscore_overlay:
        scored=C.read(Path(run)/"POSTSEAL_SCORING.json")
        if scored["prediction_seal_sha256"]!=C.sha(Path(run)/"PREDICTION_SEAL.json"):
            raise AssertionError("GT overlay requires scoring bound to this prediction seal")
    records=[]
    for row in rows:
        if row.get("variant") == "S_RAW_EXTREME":
            continue
        runtime=_runtime(row)
        fid=row.get("source_record_id",row.get("record_id",row["id"].split("/ORIGINAL")[0]))
        record_id=row["id"] if kind=="primary" else fid
        if select_ids and fid not in select_ids and row.get("id") not in select_ids:
            continue
        frame=frames[fid]
        image_path=C.ROOT/row.get("image",frame["image"])
        expected_image_sha=(frame["image_sha256"] if row.get("image",frame["image"])==frame["image"] else
                            protocol.get("crop_source_sha256",{}).get(row["image"]))
        if not expected_image_sha or C.sha(image_path)!=expected_image_sha:
            raise AssertionError("Rendered actual record RGB differs from its prediction input")
        image=cv2.imread(str(image_path))
        caption=("Archived first-head seen-CAL contract demonstration; no new-model generalization claim."
                 if kind=="smoke" else "Frozen v3 "+("seen-CAL" if row.get("partition")=="CAL" else "reused TEST (exploratory)")+" actual prediction; acceptance/failure shown exactly as saved.")
        if row.get("crop_transform"):
            caption+=" Actual RGB crop; translated frozen upstream q/K, no fresh backbone crop forward."
        gt=None
        if postscore_overlay:
            crop=next((c for c in scored["crops"] if c["id"]==row["id"]),None)
            if crop is not None:
                gt=np.array(crop["native_gt"],float)[:8]
                valid=np.array(crop["native_gt_valid"],bool)[:8]
                gt[~valid]=np.nan
            elif row.get("partition")=="CAL":
                if CAL_annotations is None:
                    isolated=C.read(C.DOC/"nonsevere_v1/OOF_pose_v1/ISOLATED_CAL_SCORING.json")
                    if isolated["TEST_rows"] or isolated["FINAL_rows"] or isolated["severe_rows"]:
                        raise AssertionError("Only isolated native CAL annotations may overlay natural photos")
                    CAL_annotations=isolated["rows"]
                ann=CAL_annotations[fid]
                mapping=ann["metadata"]["native_to_reference_corner_ids"]
                gt=np.array(ann["target"]["gt"],float)[mapping]
                valid=np.array(ann["metadata"]["valid_reference_corners"],bool)[mapping]
                gt[~valid]=np.nan
            else:
                raise ValueError("Natural TEST GT overlay is not supported by this renderer")
        caption+=" Yellow +: saved native CAL annotation (postseal only)." if gt is not None else " No GT overlay."
        stem=record_id.replace(":","__").replace("/","__")
        records.append(render_record(image,runtime,record_id=record_id,output=output/stem,caption=caption,gt_native=gt))
    if not records:
        raise ValueError("No selected sealed actual records to render")
    index=["<!doctype html><meta charset='utf-8'><title>Actual statewise pipeline records</title>",
           "<style>body{font:16px sans-serif;max-width:1600px;margin:24px auto}img{width:100%}article{margin:35px 0}</style>",
           "<h1>Actual statewise pipeline records</h1>"]
    for record in records:
        index.append("<article><h2>%s</h2><p>%s</p><img src='%s'><p><a href='%s'>SVG</a></p></article>" %
            (html.escape(record["record_id"]),html.escape(record["caption"]),record["png"],record["svg"]))
    (output/"index.html").write_text("\n".join(index)+"\n")
    C.write(output/"FIGURE_PROVENANCE.json",dict(schema="SEALED_ACTUAL_STATEWISE_FIGURES_V2",kind=kind,
        prediction_cache_sha256=C.sha(Path(run)/"PREDICTIONS.jsonl.gz"),
        prediction_seal_sha256=C.sha(Path(run)/"PREDICTION_SEAL.json"),
        renderer_source_sha256=C.sha(output/"renderer_snapshot.py"),GT_reads=int(postscore_overlay),postscore_only_GT_overlay=postscore_overlay,poses_reestimated=False,
        predictions_changed=False,rows=records))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=STUDY/"runtime_smoke_v1")
    parser.add_argument("--output",type=Path,default=STUDY/"figures/first_head_seen_CAL_runtime_v1")
    parser.add_argument("--inputs",type=Path)
    parser.add_argument("--kind",choices=("smoke","primary"),default="smoke")
    parser.add_argument("--record-id",action="append")
    parser.add_argument("--postscore-overlay",action="store_true")
    args=parser.parse_args()
    print(render_run(args.run,args.output,source_inputs=args.inputs,kind=args.kind,select_ids=args.record_id,
                     postscore_overlay=args.postscore_overlay))


if __name__=="__main__":
    main()
