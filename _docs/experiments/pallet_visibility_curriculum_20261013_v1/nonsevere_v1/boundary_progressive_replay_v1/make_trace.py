"""Export an actual inference trajectory; illustrative frame chosen without GT."""
from pathlib import Path
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C

HERE = Path(__file__).resolve().parent
COHORT = HERE.parent


def main():
    output = HERE / "PROGRESSIVE_TRACE.png"
    if output.exists():
        raise FileExistsError("Preserve exported trace")
    obs = {r["id"]: r["observation"] for r in C.rows(HERE / "OBSERVATIONS.jsonl.gz")}
    # First in frozen input order with a corrected coordinate and >=3 fits.
    # No scores, reference labels or best-case physical accuracy are queried.
    chosen = next(r for r in C.rows(HERE / "PREDICTIONS.jsonl.gz") if r["method"] == "FUSED_ACTIVE"
        and r["solution"]["accepted"] and obs[r["id"]]["correction"]["observed_refined_ids"]
        and sum(s["accepted"] and s.get("fit_performed", True) for s in r["solution"]["stages"][1:]) >= 3)
    frame = next(f for f in C.read(COHORT / "PREDICTION_INPUTS.json")["frames"] if f["id"] == chosen["id"])
    image = cv2.cvtColor(cv2.imread(str(ROOT / frame["image"])), cv2.COLOR_BGR2RGB)
    evidence = obs[chosen["id"]]
    raw, fused, projected = map(np.asarray, (evidence["raw_points"], evidence["fused_points"], chosen["projected_points"]))
    H = np.asarray(evidence["self_hidden"], bool)
    geometry = evidence["boundary_visibility"]
    checkpoints = chosen["solution"]["final_guard"]["checkpoints"]
    seed = chosen["solution"]["stages"][0]["geometric_outline_seed"]["seed"]
    good = np.isfinite(raw[:8]).all(1) & (raw[:8, 0] >= 0) & (raw[:8, 0] < image.shape[1]) & (raw[:8, 1] >= 0) & (raw[:8, 1] < image.shape[0])
    lo, hi = raw[:8][good].min(0) - 45, raw[:8][good].max(0) + 45
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig.suptitle("Outline visibility → image measurements → progressive pose\n" + chosen["id"], fontsize=14)
    for ax in axes.flat[:3]:
        ax.imshow(image)
        ax.set_xlim(max(0, lo[0]), min(image.shape[1], hi[0]))
        ax.set_ylim(min(image.shape[0], hi[1]), max(0, lo[1]))
        ax.axis("off")
    ax = axes[0, 0]
    for k in np.flatnonzero(good):
        state = geometry["states"][k]
        color = "#af77df" if H[k] else "#48dc76" if state == "VISIBLE" else "#ff5d55" if state == "EXTERNAL_OCCLUSION_SUSPECTED" else "#ffd355"
        ax.scatter(*raw[k], c=color, s=48, edgecolors="black", linewidths=.6)
        ax.text(raw[k, 0] + 7, raw[k, 1] - 4, str(k), color="white", fontsize=9,
                bbox=dict(facecolor="black", alpha=.65, edgecolor="none", pad=1.5))
    ax.set_title("1. Original KP and conditional boundary visibility", fontsize=11)
    handles = [Line2D([], [], marker="o", color=c, linestyle="none", label=t) for c, t in
               (("#48dc76", "Visible outline"), ("#ff5d55", "External occlusion suspected"), ("#ffd355", "Other / unassessed"), ("#af77df", "Initial self-hidden"))]
    ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=.9)
    ax = axes[0, 1]
    for edge in evidence["correction"]["edge_records"]:
        if edge.get("supported"):
            measured = np.asarray(edge["observed_points"])
            ax.plot(measured[:, 0], measured[:, 1], color="#5cf1ff", linewidth=2, alpha=.9)
    ax.scatter(raw[:8][good, 0], raw[:8][good, 1], facecolors="none", edgecolors="#ffd355", s=65, label="Original KP")
    moved = [r["corner"] for r in evidence["fusion"]]
    ax.scatter(fused[moved, 0], fused[moved, 1], color="#5cf1ff", marker="x", s=45, label="Bounded blend")
    ax.set_title("2. Lines sampled from current image mask", fontsize=11)
    ax.legend(fontsize=8, loc="lower left", framealpha=.9)
    ax = axes[1, 0]
    used = chosen["solution"]["used_ids"]
    for first, second in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)):
        ax.plot(projected[[first, second], 0], projected[[first, second], 1], color="white", linewidth=1.1, alpha=.65)
    ax.scatter(fused[used, 0], fused[used, 1], c="#ffd355", s=35, label="Additional measured KP")
    ax.scatter(fused[seed, 0], fused[seed, 1], c="#48dc76", s=60, edgecolors="black", linewidths=.5, label="First seed")
    ax.scatter(projected[:, 0], projected[:, 1], c="white", marker="+", s=25, label="Pose projection (output)")
    ax.set_title("3. Initial outline seed, then one KP at a time", fontsize=11)
    ax.legend(fontsize=8, loc="lower left", framealpha=.9)
    ax = axes[1, 1]
    initial = chosen["solution"]["final_guard"]["initial"]["objective"]
    costs = [initial] + [r["full_visible"]["objective"] for r in checkpoints]
    labels = ["Initial"] + ["Seed " + str(len(r["active_ids"])) if i == 0 else "+KP " + str(len(r["active_ids"])) for i, r in enumerate(checkpoints)]
    ax.plot(range(len(costs)), costs, color="#2369a7", marker="o", linewidth=2)
    ax.axhline(initial, color="gray", linestyle="--", linewidth=1, label="Initial safeguard reference")
    ax.set_xticks(range(len(costs)), labels)
    ax.set_ylabel("Full visible robust objective + initial prior")
    ax.set_title("4. Accepted trajectory; independent q stays fixed", fontsize=11)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    ax.text(.02, .06, "Fit coordinates: image measurements\nHidden projections: completion only\nThis curve is an inference diagnostic, not GT accuracy", transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#dddddd", alpha=.9))
    fig.savefig(output, dpi=160)
    fig.savefig(HERE / "PROGRESSIVE_TRACE.pdf")
    C.write(HERE / "PROGRESSIVE_TRACE_DATA.json", dict(id=chosen["id"], selected_without_GT=True,
        selection="First frozen-input frame with a moved image coordinate and >=3 accepted fits",
        raw_points=raw, fused_points=fused, fixed_H=H, seed=seed, used_ids=used,
        checkpoints=checkpoints, objective=costs, image_sha256=frame["image_sha256"],
        predictions_sha256=C.sha(HERE / "PREDICTIONS.jsonl.gz")))
    print("PROGRESSIVE_TRACE_EXPORTED", chosen["id"], flush=True)


if __name__ == "__main__":
    main()
