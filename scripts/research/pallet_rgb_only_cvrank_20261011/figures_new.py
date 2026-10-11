"""Scientific PNGs from sealed public errors only; no images, inference or PnP."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

METHODS = ("BASE", "SUBPIX", "CVRANK_NATIVE", "CVRANK_CAP1", "CVRANK_THEN_SUBPIX",
           "OLD_P", "OLD_P_SUBPIX", "M1", "M1_SUBPIX", "RGB_ONLY", "RGB_ONLY_SUBPIX",
           "M3", "M3_SUBPIX", "POSEFIX", "POSEFIX_SUBPIX")
PRIMARY = (("M3_SUBPIX", "RGB_ONLY_SUBPIX"), ("M3_SUBPIX", "M1_SUBPIX"),
           ("M3_SUBPIX", "CVRANK_THEN_SUBPIX"))
LABELS = {"BASE": "Base", "SUBPIX": "SubPix", "CVRANK_NATIVE": "CVRank native",
          "CVRANK_CAP1": "CVRank cap1%", "CVRANK_THEN_SUBPIX": "CVRank + SP",
          "OLD_P": "OLD_P", "OLD_P_SUBPIX": "OLD_P + SP", "M1": "M1 feature",
          "M1_SUBPIX": "M1 + SP", "RGB_ONLY": "RGB only", "RGB_ONLY_SUBPIX": "RGB only + SP",
          "M3": "M3 combined", "M3_SUBPIX": "M3 + SP", "POSEFIX": "PoseFix style",
          "POSEFIX_SUBPIX": "PoseFix style + SP"}
COLORS = {"BASE": "#929292", "SUBPIX": "#4e5962", "CVRANK_NATIVE": "#8c6bb1",
          "CVRANK_CAP1": "#684995", "CVRANK_THEN_SUBPIX": "#4a2c76", "OLD_P": "#70a8a0",
          "OLD_P_SUBPIX": "#367f75", "M1": "#60ad77", "M1_SUBPIX": "#247545",
          "RGB_ONLY": "#70a6da", "RGB_ONLY_SUBPIX": "#24649c", "M3": "#eea45a",
          "M3_SUBPIX": "#bd681b", "POSEFIX": "#777b85", "POSEFIX_SUBPIX": "#171b24"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, path):
    assert not path.exists(), "Preserve completed figure"
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pipeline(path):
    fig, ax = plt.subplots(figsize=(14.7, 6.8))
    ax.set(xlim=(0, 15), ylim=(0, 7))
    ax.axis("off")
    def box(x, y, w, h, text, color, size=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.06,rounding_size=.1",
                     fc=color, ec="#46515c", lw=1.05))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size)
    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                                    color="#59636f", lw=1.2))
    box(.25, 4.65, 1.65, 1.15, "Current RGB", "#edf2f8")
    box(2.35, 4.55, 2.35, 1.35, "Frozen YOLO\nInitial corners / box\nPrediction support", "#e5eff8")
    box(5.25, 4.55, 2.55, 1.35, "Inherited OLD_P scores\n48 image slots = 0\nRegular 64 / null 61", "#ffe7e7")
    box(8.35, 4.55, 2.65, 1.35, "Joint candidate score\nBase score + RGB residual\nResidual 81: first 48 = 0", "#e6f1ea")
    box(11.55, 4.55, 3.0, 1.35, "222 candidate probabilities\n(221 positions + null)\nNative coordinate expectation", "#e6f1ea")
    arrow((1.9, 5.2), (2.35, 5.2)); arrow((4.7, 5.2), (5.25, 5.2))
    arrow((7.8, 5.2), (8.35, 5.2)); arrow((11., 5.2), (11.55, 5.2))
    box(.3, 2.3, 3.4, 1.45, "Predicted native RGB tiles\nShared tiny CNN, once per batch\n16-D samples at candidate positions\nNo GT crop or teacher output", "#eaf0fc")
    box(4.35, 2.4, 3.1, 1.2, "Role / relative coordinates\nBox shape / displacement\nCoverage / null flag", "#f2f3f5")
    box(8.15, 2.4, 3.35, 1.2, "Actual P3/P4: excluded\nAdapters / patch CNN: skipped\nAll 3 scorer entrances verified", "#fff0f0")
    box(12.05, 2.1, 2.5, 1.7, "Bare: final Base cap\n+ SP: uncapped expectation\n  -> SubPix -> Base cap once\nShared W/D selection + PnP\nK / registered dimensions", "#fff0df", 8.8)
    arrow((1.25, 4.65), (1.25, 3.75))
    arrow((3.7, 3.05), (8.35, 4.55)); arrow((6., 3.6), (6., 4.55))
    arrow((7.45, 3.15), (9.3, 4.55)); arrow((13.25, 4.55), (13.25, 3.8))
    ax.text(.25, 6.5, "RGB-only input ablation under the same warm start and replay", fontsize=14, fontweight="bold")
    ax.text(.3, 1.2, "Two seeds; OLD_P last6000 warm start; 3,000 new updates each; same M3 initialization and minibatches", fontsize=10)
    ax.text(.3, .65, "Loss: original candidate CE + fixed 0.1521498302 x native-coordinate SmoothL1; GEO = 0", fontsize=10)
    ax.text(.3, .18, "RGB-only refers to the correction stage. The frozen detector remains shared. REAL110 is an exposed development set.", fontsize=9)
    save(fig, path)


def success(path, metrics):
    packets = [metrics["groups"][m]["common_seed12"] for m in METHODS]
    rates = np.array([p["strict_success"]["rate"]*100 for p in packets])
    bounds = np.array([p["strict_success"]["CI95"] for p in packets])*100
    fig, ax = plt.subplots(figsize=(13.2, 7.5))
    y = np.arange(len(METHODS))
    ax.barh(y, rates, color=[COLORS[m] for m in METHODS], alpha=.88, height=.7)
    # Draw absolute CI endpoints, including the possible case where a bootstrap
    # interval does not bracket the original point estimate.
    ax.hlines(y, bounds[:, 0], bounds[:, 1], color="#18232f", lw=1.)
    ax.vlines(bounds[:, 0], y-.09, y+.09, color="#18232f", lw=1.)
    ax.vlines(bounds[:, 1], y-.09, y+.09, color="#18232f", lw=1.)
    ax.set_yticks(y, [LABELS[m] for m in METHODS]); ax.invert_yaxis()
    for i, p in enumerate(packets):
        n = p["strict_success"]["successes"]
        ax.text(101.5, i, f"{n}/220", va="center", fontsize=9)
    ax.set(xlim=(0, 112), xlabel="Strict success (%)", title="All 15 routes: T < 5 cm AND R < 5 deg")
    ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.grid(axis="x", alpha=.18)
    fig.text(.12, .025, "Bars: two-seed prediction counts. Intervals: 10,000 paired session-cluster draws (110 images, 9 sessions).\nRule coordinates are identical across seeds; these are alignment replicas, not independent rule trials.", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, 1)); save(fig, path)


def cdf(path, rows):
    fig, axes = plt.subplots(1, 2, figsize=(14.3, 6.5))
    for ax, key, label in zip(axes, ("translation_cm", "rotation_deg"), ("Translation error (cm)", "Rotation error (deg)")):
        largest = 0.
        for method in METHODS:
            rr = [r for r in rows if r["method"] == method]
            assert len(rr) == 220
            values = sorted(float(r["pose"][key]) for r in rr if r["available"] and r["pose"].get(key) is not None)
            if not values:
                ax.plot([], [], label=LABELS[method], color=COLORS[method]); continue
            largest = max(largest, values[-1])
            x = np.r_[0., values]
            y = np.arange(len(x))/220
            line = "-" if ("SUBPIX" in method or method == "SUBPIX") else "--"
            ax.step(x, y, where="post", color=COLORS[method], linestyle=line, lw=1.25, label=LABELS[method])
        ax.set(xlim=(0, max(1., largest*1.025)), ylim=(0, 1.015), xlabel=label, ylabel="Fraction of all 220 predictions")
        ax.grid(alpha=.2); ax.axvline(5, color="#808080", lw=.8, linestyle=":")
        ax.set_title("Full range; missing poses stay outside finite CDF")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle("T/R tails for every route; no large-error frame removed", fontsize=13)
    fig.tight_layout(rect=(0, .16, 1, .94)); save(fig, path)


def effects(path, paired, transitions):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.))
    primary = paired["primary"]
    assert {(r["after"], r["before"]) for r in primary} == set(PRIMARY)
    labels = []
    for i, pair in enumerate(PRIMARY):
        r = next(r for r in primary if (r["after"], r["before"]) == pair)
        effect = r["effect_pp"]; low, high = r["CI95"]
        axes[0].plot([low, high], [i, i], color="#374c62", lw=2)
        axes[0].plot(effect, i, "o", color="#be6f1e", ms=7)
        axes[0].text(high+.35, i, f"p={r['p_raw']:.3g}; Holm={r['p_Holm']:.3g}", va="center", fontsize=8)
        labels.append(f"M3 + SP - {LABELS[r['before']]}")
    axes[0].set_yticks(range(3), labels, fontsize=9); axes[0].invert_yaxis()
    axes[0].axvline(0, color="#737373", ls="--", lw=.9)
    axes[0].set(xlabel="Paired strict-success difference (percentage points)", title="Fixed family: 3 contrasts, session 95% CI")
    axes[0].margins(x=.38); axes[0].grid(axis="x", alpha=.18)
    y = np.arange(3)
    recovery = []; damage = []
    for after, before in PRIMARY:
        rr = [r for r in transitions if r["after"] == after and r["before"] == before]
        assert len(rr) == 220
        recovery.append(sum(r["event"] == "RECOVERY" for r in rr))
        damage.append(sum(r["event"] == "DAMAGE" for r in rr))
    axes[1].barh(y-.16, recovery, height=.3, color="#378957", label="Recovery")
    axes[1].barh(y+.16, damage, height=.3, color="#b65353", label="Damage")
    axes[1].set_yticks(y, [LABELS[b] for _, b in PRIMARY], fontsize=9); axes[1].invert_yaxis()
    for i, (a, b) in enumerate(zip(recovery, damage)):
        axes[1].text(a+.15, i-.16, str(a), va="center", fontsize=9)
        axes[1].text(b+.15, i+.16, str(b), va="center", fontsize=9)
    axes[1].set(xlabel="Image-seed transitions / 220", title="M3 + SP compared with each method")
    axes[1].legend(frameon=False); axes[1].grid(axis="x", alpha=.15)
    fig.text(.02, .02, "Previously exposed DEV; M3-M1 reuses an already observed contrast. Non-significance is not equivalence.", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, 1)); save(fig, path)


def is_true(value):
    return str(value).lower() in ("true", "1")


def numeric(value):
    return None if value in (None, "", "None", "null") else float(value)


def case_panels(path, rows, transitions):
    index = {(r["method"], r["id"], int(r["seed"])): r for r in rows}
    displayed = []
    for after, before in PRIMARY:
        for event in ("RECOVERY", "DAMAGE"):
            options = sorted((r for r in transitions if r["after"] == after and r["before"] == before and r["event"] == event),
                             key=lambda r: (r["id"], int(r["seed"])))
            displayed.append(dict(title=event, transition=options[0] if options else None,
                                  after=after, before=before, no_case=not options))
    options = sorted((r for r in transitions if is_true(r["corner_better_pose_worse"])),
                     key=lambda r: (r["id"], int(r["seed"]), r["after"], r["before"]))
    displayed.append(dict(title="Corner better, pose worse", transition=options[0] if options else None,
                          no_case=not options))
    worst = sorted((r for r in rows if r["available"]), key=lambda r: (-r["pose"]["rotation_deg"], r["id"], r["seed"], r["method"]))[0]
    displayed.append(dict(title="Largest R (all 15 routes)", transition=dict(id=worst["id"], seed=worst["seed"],
        after=worst["method"], before="BASE"), no_case=False))
    fig, axes = plt.subplots(4, 2, figsize=(15., 15.5), squeeze=False)
    selected = []
    for ax, item in zip(axes.ravel(), displayed):
        if item["no_case"]:
            ax.axis("off")
            ax.text(.5, .5, f"{item['title']}: no case\n{LABELS[item.get('after', 'M3_SUBPIX')]} versus\n{LABELS[item.get('before', 'M1_SUBPIX')]}",
                    ha="center", va="center", transform=ax.transAxes)
            selected.append(dict(title=item["title"], status="NO_CASE")); continue
        t = item["transition"]; fid, seed = t["id"], int(t["seed"])
        before, after = index[t["before"], fid, seed], index[t["after"], fid, seed]
        corners = np.arange(8)
        for row, offset, color, name in ((before, -.17, "#4f83b5", "Before"), (after, .17, "#d18a34", "After")):
            error = np.array([float(e) if e is not None else np.nan for e in row["corner_errors_px"]])
            mask = np.asarray(row["corner_valid"], bool)
            error[~mask] = np.nan
            ax.bar(corners+offset, error, width=.32, color=color, label=f"{name}: {LABELS[row['method']]}")
        text = lambda r, key: "NA" if not r["available"] else f"{r['pose'][key]:.2f}"
        subtitle = (f"T cm {text(before, 'translation_cm')} -> {text(after, 'translation_cm')}; "
                    f"R deg {text(before, 'rotation_deg')} -> {text(after, 'rotation_deg')}")
        ax.set_title(f"{item['title']} | {fid} | seed {seed}\n{subtitle}", fontsize=9)
        ax.set(xlabel="Canonical corner index (invalid reference: no bar)", ylabel="Corner error (px)")
        ax.set_xticks(corners); ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=.17)
        ax.axhline(5, color="#808080", ls=":", lw=.7)
        selected.append(dict(title=item["title"], id=fid, seed=seed, before=t["before"], after=t["after"],
            before_Tcm=before["pose"].get("translation_cm"), after_Tcm=after["pose"].get("translation_cm"),
            before_Rdeg=before["pose"].get("rotation_deg"), after_Rdeg=after["pose"].get("rotation_deg")))
    fig.suptitle("Actual evaluation cases: deterministic examples and worst rotation; all transitions remain in CSV", fontsize=13)
    fig.text(.02, .006, "No RGB or native reference/prediction coordinates. Bars use the exact whole-object evaluation alignment and valid-corner mask.", fontsize=9)
    fig.tight_layout(rect=(0, .026, 1, .965)); save(fig, path)
    return selected


def main(args):
    begin = time.monotonic()
    doc = Path(args.public_dir)
    directory = doc / "FIGURES"
    directory.mkdir(exist_ok=True)
    metrics = read(doc / "METRICS.json"); paired = read(doc / "PAIRED_RESULTS.json")
    with gzip.open(doc / "ROW_METRICS.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    with (doc / "ALL_PAIRED_TRANSITIONS.csv").open() as stream:
        transitions = list(csv.DictReader(stream))
    assert len(rows) == 3300 and len({r["id"] for r in rows}) == 110
    assert set(r["method"] for r in rows) == set(METHODS)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
    names = ("01_pipeline.png", "02_success.png", "03_full_pose_CDF.png",
             "04_paired_effects.png", "05_actual_cases.png")
    pipeline(directory / names[0]); success(directory / names[1], metrics)
    cdf(directory / names[2], rows); effects(directory / names[3], paired, transitions)
    cases = case_panels(directory / names[4], rows, transitions)
    receipt = dict(status="COMPLETE", source_sha256=sha(Path(__file__)),
        inputs=[dict(path=name, sha256=sha(doc / name)) for name in
                ("METRICS.json", "PAIRED_RESULTS.json", "ROW_METRICS.jsonl.gz", "ALL_PAIRED_TRANSITIONS.csv")],
        outputs=[dict(path="FIGURES/"+name, sha256=sha(directory / name), bytes=(directory / name).stat().st_size) for name in names],
        all_routes=15, independent_images=110, paired_prediction_rows_per_route=220, bootstrap_clusters=9,
        case_selection="First lexicographic ID/seed per primary recovery/damage; first corner-better/pose-worse; maximum observed R allroutes",
        actual_cases=cases, full_CDF_ranges=True, CDF_denominator=220, missing_poses_retained=True,
        new_GPU_forwards=0, optimizer_updates=0, production_F_calls=0, new_RGB_reads=0,
        public_native_or_reference_coordinates=False, English_labels=True,
        wall_seconds=time.monotonic()-begin)
    with (doc / "FIGURE_RECEIPT.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write("\n")
    print("FIGURES_COMPLETE", len(names), round(receipt["wall_seconds"], 3), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-dir", required=True)
    main(parser.parse_args())
