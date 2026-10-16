"""Standalone scientific figures from sealed predictions and reported results."""
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
bindings, figures = {}, []

def read_rows(path):
    bindings[str(path.relative_to(ROOT))] = C.sha(path)
    return list(C.rows(path))

def save(fig, name, caption):
    for ext in ("pdf", "svg", "png"):
        path = OUT / (name+"."+ext)
        if path.exists():
            raise FileExistsError("Preserve figure: "+str(path))
        fig.savefig(path, dpi=220, bbox_inches="tight")
        figures.append(dict(file=str(path.relative_to(ROOT)), sha256=C.sha(path), caption=caption))
    plt.close(fig)

groups = ("CAL_N3", "TEST_N3", "TEST_PF")
predictions, coordinates, diagnostics = {}, {}, {}
for group in groups:
    seal = C.read(HERE / group / "PREDICTION_SEAL.json")
    assert seal["before_scoring"] and seal["predictions_sha256"] == C.sha(HERE / group / "PREDICTIONS.jsonl.gz")
    predictions[group] = read_rows(HERE / group / "PREDICTIONS.jsonl.gz")
    coordinates[group] = read_rows(HERE / group / "COORDINATE_ROWS.jsonl.gz")
    diagnostics[group] = read_rows(HERE / group / "REGION_DIAGNOSTICS.jsonl.gz")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
for ax, group, label in zip(axes, ("TEST_N3", "TEST_PF"), ("N3 + SubPix", "PoseFix + SubPix")):
    rows = [r for r in coordinates[group] if r["method"] == "FINITE_REGION" and r["source"] == "IMAGE_BOUNDARY_ANCHOR_MEASUREMENT"]
    values = [np.mean([r[key] for r in rows]) for key in ("raw_error_px", "mixed_error_px", "rigid_error_px")]
    bars = ax.bar([0, 1, 2], values, color=["#9199a5", "#208770", "#4c78a8"], width=.64)
    ax.set_xticks([0, 1, 2], ["Raw KP", "Preserved RGB\nmeasurement", "Rigid model\nprojection"])
    ax.set_title(label+"; n="+str(len(rows))+" paired anchors")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+.08, f"{value:.2f}", ha="center")
    ax.set_ylim(0, 5.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Mean corner error (px)")
save(fig, "boundary_measurement_preservation", "Matched model-compatible reused TEST corners with authoritative image-boundary roles. Preserve RGB measurements separately from one-pose projections; reference is reconstructed annotation, not independent physical metrology.")

fig, ax = plt.subplots(figsize=(7, 3.6))
counts, outside, passed = [], [], []
for group in groups:
    candidates = [c for row in predictions[group] if row["method"] == "FINITE_REGION" for c in row["solution"]["region_audit"]["candidates"][1:]]
    counts.append(len(candidates))
    outside.append(sum(c["region"]["normalized_penalty"] > 0 for c in candidates))
    passed.append(sum(c.get("usable", False) and c["region"]["normalized_penalty"] > 0 for c in candidates))
positions = np.arange(3)
for offset, values, color, label in ((-.23, counts, "#9199a5", "New finite candidates"),
                                      (0., outside, "#cf8758", "Outside a coarse region"),
                                      (.23, passed, "#208770", "Outside and otherwise admissible")):
    bars = ax.bar(positions+offset, values, width=.22, color=color, label=label)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+2, str(value), ha="center", fontsize=9)
ax.set_xticks(positions, ["CAL N3 (46)", "TEST N3 (49)", "TEST PoseFix (49)"])
ax.set_ylabel("Candidate count")
ax.set_ylim(0, 175)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "region_constraint_redundancy", "All candidates with positive coarse-region penalty already failed the fixed visible/geometry/anchor checks. Soft region scoring changed zero selected poses relative to the same candidate pool without region weight. Two upstream TEST variants reuse the same49 photos.")

fig, ax = plt.subplots(figsize=(10, 3.8))
ax.set_xlim(0, 10);ax.set_ylim(0, 3.8);ax.axis("off")
boxes = [(0., 1.35, 1.65, .95, "RGB + native KP\nCamera intrinsics K\nRegistered cuboid", "#e6e9ef"),
         (2., 2.55, 2.2, .85, "Visible outline / interior\nimage measurements", "#d7eee6"),
         (2., .35, 2.2, .85, "Unobserved image KP\nempirical coarse regions", "#f4e2d7"),
         (4.7, 1.35, 2.1, .95, "Finite pose candidates\nlegacy + joint + P3P", "#dde6f4"),
         (7.25, 1.35, 2.7, .95, "Protect visible measurements\nscore coarse regions\nretain safe fallback", "#e1e8ed")]
for x,y,w,h,txt,color in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.05",facecolor=color,edgecolor="#586372"))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=9)
for start,end in (((1.7,1.9),(2.,3.)),((1.7,1.8),(2.,.8)),((4.2,3.),(4.7,2.)),((4.2,.8),(7.25,1.6)),((6.85,1.85),(7.25,1.85))):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=12,color="#586372"))
ax.text(8.6,.7,"Outputs: preserved image KP + model completion\nSeparate: one rigid model and diagnostic uncertainty",ha="center",fontsize=9)
save(fig, "statewise_method", "Operational statewise method: image evidence and coarse predicted regions have separate roles. Initial geometric flags and pose-propagated covariance remain conditional diagnostics. The mixed output is not claimed to be one exact rigid projection.")

protocol = C.read(HERE / "live_example/PROTOCOL.json")
frame_id = protocol["frame_id"]
row = next(r for r in predictions["CAL_N3"] if r["id"] == frame_id and r["method"] == "FINITE_REGION")
image = cv2.cvtColor(cv2.imread(str(ROOT / protocol["image"])), cv2.COLOR_BGR2RGB)
bindings[protocol["image"]] = C.sha(ROOT / protocol["image"])
fig, ax = plt.subplots(figsize=(8, 5))
ax.imshow(image)
raw = np.asarray(row["original_points"])[:8]
model = np.asarray(row["keypoint_readout"]["rigid_model_corners"])
measure = np.asarray(row["keypoint_readout"]["reported_points"])[:8]
ax.scatter(raw[:,0],raw[:,1],s=30,marker="x",color="lightgray",label="Raw image-predicted KP")
ax.scatter(model[:,0],model[:,1],s=28,facecolors="none",edgecolors="#65c6ff",label="Rigid model projection")
ids=row["solution"]["anchor_ids"]
ax.scatter(measure[ids,0],measure[ids,1],s=80,marker="*",color="#50e5a2",label="Preserved image anchor")
for region in row["coarse_regions"]:
    center=region["center_xy"]; radius=region["half_axes_px"][0]
    ax.add_patch(Circle(center,radius,fill=False,color="#ffae70",linewidth=1.2))
ax.plot([],[],color="#ffae70",label="Coarse predicted region")
for k in range(8):
    ax.text(model[k,0]+4,model[k,1]-4,str(k),color="white",fontsize=8)
all_points=np.vstack([raw,model])
pad=55
ax.set_xlim(max(0,all_points[:,0].min()-pad),min(image.shape[1],all_points[:,0].max()+pad))
ax.set_ylim(min(image.shape[0],all_points[:,1].max()+pad),max(0,all_points[:,1].min()-pad))
ax.set_title("Predeclared CAL example: predictions only; no GT overlay")
ax.axis("off");ax.legend(loc="upper center",bbox_to_anchor=(.5,-.03),ncol=2,fontsize=8,frameon=False)
save(fig,"statewise_CAL_example","First predeclared CAL frame wood_184309:000012. Actual RGB; native model, preserved measured anchors and image-regressed coarse circles. Circles are empirical tolerances, not physical visibility certificates or calibrated probability intervals; no GT is plotted.")
C.write(OUT / "FIGURE_MANIFEST.json",dict(source_sha256=bindings,figures=figures,script_sha256=C.sha(__file__),no_synthetic_performance_data=True,TEST_reused=True))
print("PAPER_FIGURES_SAVED",len(figures),flush=True)
