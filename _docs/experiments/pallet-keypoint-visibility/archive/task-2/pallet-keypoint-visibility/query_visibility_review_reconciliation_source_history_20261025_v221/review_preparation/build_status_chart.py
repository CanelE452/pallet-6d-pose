"""Render already-published V220 aggregate counts; no raw evaluation access."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P = Path("/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility")
OUT = P / "query_visibility_review_reconciliation_source_history_20261025_v221"
ROWS = [
    ("v61", 260, 887, 0, 2, 88, 0),
    ("legacy", 655, 319, 173, 3, 83, 4),
    ("C55 (V191)", 831, 316, 0, 53, 37, 0),
    ("C63 (V213)", 843, 304, 0, 71, 19, 0),
    ("C64 (V220)", 582, 565, 0, 55, 35, 0),
    ("C65 (V220)", 901, 246, 0, 76, 14, 0),
]


def main():
    OUT.mkdir(exist_ok=False)
    metrics = []
    for name, dd, de, du, ed, ee, eu in ROWS:
        assert dd + de + du == 1147
        assert ed + ee + eu == 90
        metrics.append(dict(model=name, D=1147, E=90, D_D=dd, D_E=de, D_U=du,
                            E_D=ed, E_E=ee, E_U=eu,
                            D_recall=dd / 1147, E_recall=ee / 90,
                            E_precision=ee / (ee + de),
                            U_rate=(du + eu) / 1237, correct=dd + ee))
    with (OUT / "ALREADY_PUBLISHED_COUNTS_AND_METRICS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0]))
        w.writeheader()
        w.writerows(metrics)
    (OUT / "STATUS_SNAPSHOT.json").write_text(json.dumps({
        "phase": "V221", "kind": "HISTORICAL_RECONCILIATION_NOT_NEW_BENCHMARK",
        "aggregate_source": "V220 canonical research_current diagnostic_results main_metrics; all counts previously published",
        "evaluation_source_commit": "59cf1e512fc8084c66f1eedc5195b4434ab9e28d",
        "same_primary_queries": 1237, "D": 1147, "E": 90,
        "new_training_updates": 0, "new_model_forwards": 0, "new_PnP": 0,
        "raw_evaluation_prediction_label_image_reads": 0,
        "selected_model": None, "adopted": False, "task_solved": False,
        "metrics": metrics,
        "C55_to_C65_D_correct_delta": 70, "C55_to_C65_E_correct_delta": -23,
        "C63_to_C65_D_correct_delta": 58, "C63_to_C65_E_correct_delta": -5,
        "coordinate_error_le_3px_D461_C65_false_E": 93,
        "notes": [
            "Repeatedly exposed diagnostic cohort, not a fresh blind test.",
            "v61 E is the historical presence/nonvisible proxy, not an external-occlusion cause output.",
            "legacy was fitted on historical private CAL and has a different training lineage.",
            "Zero predicted U in new heads comes from forced binary decoding, not calibrated certainty.",
            "Input unavailable45 and SELF136 are outside this D/E confusion denominator.",
            "Do not use these evaluation counts for new model/data/threshold/checkpoint selection."
        ]}, ensure_ascii=False, indent=2) + "\n")
    x = np.arange(len(metrics))
    colors = ["#9b9fa8", "#705783", "#56829a", "#56829a", "#b78c54", "#56829a"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), sharey=True)
    for ax, field, count, title, denom in [
        (axes[0], "D_recall", "D_D", "Preserve visible points (D recall)", 1147),
        (axes[1], "E_recall", "E_E", "Detect external occlusion (E recall)", 90),
    ]:
        values = [m[field] * 100 for m in metrics]
        ax.bar(x, values, color=colors, width=.72)
        for i, (value, m) in enumerate(zip(values, metrics)):
            ax.text(i, value + 2, f"{value:.1f}%\n{m[count]}/{denom}", ha="center", fontsize=9)
        ax.set_xticks(x, [m["model"].replace(" (", "\n(") for m in metrics], fontsize=9)
        ax.set_ylim(0, 117)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_title(title, fontsize=12, pad=12)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Recall (%)")
    fig.suptitle("Already-published results on the same 1,237 original queries", fontsize=14)
    fig.text(.5, .035, "D = 1,147; E = 90. Repeated diagnostic cohort. No new fit, inference or model selection.\n"
             "legacy predicted U: 177/1,237 (14.31%); other plotted heads: 0 (binary decoding).", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .94))
    fig.savefig(OUT / "VISIBLE_EXTERNAL_TRADEOFF.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"completed": True, "phase": "V221", "files": 3,
                      "new_training_updates": 0, "new_benchmark": False}))


if __name__ == "__main__":
    main()
