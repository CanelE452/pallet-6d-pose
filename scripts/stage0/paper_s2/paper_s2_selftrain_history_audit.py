#!/usr/bin/env python3
"""Audit existing PAPER_S2 self-training histories without opening eval images.

Only run configuration and pseudo-label training-history metadata are read.
Historical performance files are deliberately not re-aggregated because their
outside evaluation membership included sealed sessions and the training pools
shared capture sessions with filter-validation.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import csv
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data" / "pallet" / "results" / "ralph_selftrain"
OUT = (
    ROOT
    / "data"
    / "pallet"
    / "results"
    / "paper_s2_scratch_diffpnp"
    / "diagnostic_audit"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def discover() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    sources: list[dict] = []
    for directory in sorted(SOURCE.glob("h*_s2_*")):
        history_path = directory / "training_history.json"
        config_path = directory / "config.yaml"
        if not history_path.is_file() or not config_path.is_file():
            continue
        history = json.loads(history_path.read_text())
        config = yaml.safe_load(config_path.read_text())
        filter_type = config.get("geometric_filter", {}).get(
            "filter_type", "missing"
        )
        sources.append(
            {
                "run": directory.name,
                "history": str(history_path.relative_to(ROOT)),
                "history_sha256": sha256(history_path),
                "config": str(config_path.relative_to(ROOT)),
                "config_sha256": sha256(config_path),
            }
        )
        for record in history:
            pseudo = record.get("pseudo_labels", {})
            total = int(pseudo.get("total", 0))
            accepted = int(pseudo.get("accepted", 0))
            rows.append(
                {
                    "run": directory.name,
                    "round": int(record["round"]),
                    "filter_type": pseudo.get("filter_type", filter_type),
                    "total_pool_frames": total,
                    "accepted_pseudo_labels": accepted,
                    "acceptance_rate": (
                        float(pseudo.get("acceptance_rate"))
                        if pseudo.get("acceptance_rate") is not None
                        else accepted / total if total else np.nan
                    ),
                    "pnp_fail": int(pseudo.get("pnp_fail", 0)),
                    "filter_fail": int(pseudo.get("filter_fail", 0)),
                    "mean_internal_reprojection_px": pseudo.get(
                        "reproj_error_mean"
                    ),
                    "mean_training_loss": record.get("avg_loss"),
                    "training_time_seconds": record.get("time_seconds"),
                }
            )
    return rows, sources


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows: list[dict], path: Path) -> None:
    selected = [
        "h3_s2_night",
        "h4_s2_combined",
        "h8_s2_outside_looflip",
        "h8_s2_night_looflip",
        "h8_s2_noapril_looflip",
        "h8_s2_combined_looflip",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for run in selected:
        records = sorted(
            (row for row in rows if row["run"] == run),
            key=lambda row: row["round"],
        )
        if not records:
            continue
        rounds = [row["round"] for row in records]
        axes[0].plot(
            rounds,
            [100.0 * row["acceptance_rate"] for row in records],
            marker="o",
            label=run,
        )
        axes[1].plot(
            rounds,
            [row["accepted_pseudo_labels"] for row in records],
            marker="o",
            label=run,
        )
    axes[0].set(
        title="Pseudo-label acceptance rate",
        xlabel="self-training round",
        ylabel="accepted (%)",
        xticks=[1, 2],
    )
    axes[1].set(
        title="Accepted pseudo-label count",
        xlabel="self-training round",
        ylabel="frames",
        xticks=[1, 2],
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle(
        "Training metadata only — not independent validation performance",
        color="#9a2222",
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    started = dt.datetime.now(dt.timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sources = discover()
    if not rows:
        raise RuntimeError(f"no self-training histories under {SOURCE}")

    csv_path = OUT / "self_training_rounds.csv"
    plot_path = OUT / "self_training_acceptance_curve.png"
    json_path = OUT / "self_training_audit.json"
    report_path = OUT / "SELFTRAIN_AUDIT.md"
    write_csv(rows, csv_path)
    make_plot(rows, plot_path)

    completed = dt.datetime.now(dt.timezone.utc)
    payload = {
        "provenance": {
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "duration_seconds": (completed - started).total_seconds(),
            "exact_command": " ".join(sys.argv),
            "git_branch": git("branch", "--show-current"),
            "git_commit": git("rev-parse", "HEAD"),
            "python": sys.version,
            "platform": platform.platform(),
            "sources": sources,
        },
        "scope": {
            "opened_evaluation_images": False,
            "opened_final_test_images_or_annotations": False,
            "read_historical_performance_for_current_selection": False,
            "read_only": [
                "config.yaml",
                "training_history.json",
            ],
        },
        "rows": rows,
        "validity": {
            "training_metadata_usable": True,
            "historical_performance_paper_usable": False,
            "reasons": [
                "PAPER_S2 self-training pools share capture sessions with filter-validation even when exact GT frame ids were excluded",
                "historical outside evaluation scripts enumerate capturepallet02 through capturepallet09, including sealed capturepallet07 and capturepallet09",
                "the 36-frame legacy manual subset belongs to PL-pool session capturepallet11",
                "R2 was configured and run unconditionally rather than being gated on an independently valid R1 primary-metric improvement",
            ],
            "quantity_sweep_available": False,
            "quality_sweep_available": False,
            "independent_r0_r1_r2_curve_available": False,
        },
        "outputs": {
            "round_csv": str(csv_path.relative_to(ROOT)),
            "acceptance_plot": str(plot_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    indexed = {(row["run"], row["round"]): row for row in rows}
    featured = [
        ("h3_s2_night", "ransac_loo night"),
        ("h4_s2_combined", "ransac_loo combined"),
        ("h8_s2_combined_looflip", "ransac_loo + flip combined"),
    ]
    lines = [
        "# PAPER_S2 existing self-training audit",
        "",
        "This report uses only existing run configuration and pseudo-label history.",
        "No evaluation image, final-test annotation, or final-test frame was opened.",
        "",
        "## Round metadata",
        "",
        "| run | R1 accepted | R1 rate | R2 accepted | R2 rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for run, label in featured:
        first = indexed.get((run, 1))
        second = indexed.get((run, 2))
        if first and second:
            lines.append(
                f"| {label} | {first['accepted_pseudo_labels']} | "
                f"{100 * first['acceptance_rate']:.1f}% | "
                f"{second['accepted_pseudo_labels']} | "
                f"{100 * second['acceptance_rate']:.1f}% |"
            )
    lines += [
        "",
        "Each history shows fresh pseudo-label extraction in R2, but the run",
        "configuration executes two rounds unconditionally. There is no independent",
        "R1 gate satisfying the current protocol.",
        "",
        "## Validity decision",
        "",
        "- The acceptance counts/rates are usable as training-process metadata.",
        "- Historical R0/R1/R2 performance is not paper-valid: the training pool",
        "  shares sessions with filter-val, while the historical outside evaluator",
        "  included sealed `capturepallet07` and `capturepallet09`.",
        "- The old 36-frame manual aggregate is from PL-pool `capturepallet11`, not",
        "  strict filter-validation.",
        "- No nested 0/10/25/50/75/100% quantity sweep or equal-count",
        "  top/middle/bottom/random quality sweep exists.",
        "- Therefore no self-training checkpoint is selected and no performance",
        "  curve from those contaminated evaluations is used in the present audit.",
        "",
        "## BLOCKED",
        "",
        "```text",
        "BLOCKED:",
        "필요한 항목: session-independent real-unlabeled pool and strict N=87 validation, plus nested quantity/equal-count quality manifests",
        "현재 확인한 위치: data/pallet/results/ralph_selftrain/*/{config.yaml,training_history.json}",
        "시도한 명령: paper_s2_selftrain_history_audit.py (metadata-only aggregation)",
        "실패 원인: existing pool/evaluation membership is contaminated; required sweeps do not exist",
        "대체로 수행한 진단: per-round PL total/accepted/rate/PnP-fail/filter-fail/loss/runtime aggregation",
        "이 blocker가 전체 결론에 미치는 영향: existing self-training gains cannot support a paper claim or choose the final model",
        "```",
        "",
        f"Machine-readable rows: `{csv_path.relative_to(ROOT)}`",
        "",
        f"Acceptance-only figure: `{plot_path.relative_to(ROOT)}`",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "runs": len({row["run"] for row in rows}),
                "rows": len(rows),
                "csv": str(csv_path),
                "plot": str(plot_path),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
