"""Build a reproducible DEV-only comparison against the historical target models.

The primary target is FT_REFERENCE because TABLE3_META.json names it TARGET.
It is a real-supervised upper bound with known exposure to 12 DEV140 frames, so
the primary comparison removes those frames (DEV128). OLD_STAGE_A is retained
as the secondary synthetic target-specific milestone. Nothing emitted by this
script is FINAL or paper-table evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
QUEUE = (
    REPO
    / "challenge/yolo_pose_one_model/runs_camera_facing_loss/"
    "ubuntu_cf_loss_queue_20260823T0930"
)
SCRATCH = HERE.parent

TAGS = {
    "G38": "G38",
    "NEW": "G38P0TEX20K_CS60_S42_20260829",
    "OLD_STAGE_A": "OLD_STAGE_A",
    "FT_REFERENCE": "FT_REFERENCE",
}
REAL_FILES = {key: QUEUE / f"REAL_{tag}.json" for key, tag in TAGS.items()}
NEG_FILES = {key: QUEUE / f"NEGSCORE_{tag}.json" for key, tag in TAGS.items()}
NIGHT_FILES = {
    "G38": QUEUE / "NIGHT_CAND_G38.json",
    "NEW": QUEUE / "NIGHT_CAND_G38P0TEX20K_CS60_S42_20260829.json",
    "OLD_STAGE_A": QUEUE / "NIGHT_CAND_OLD_S.json",
    "FT_REFERENCE": QUEUE / "NIGHT_CAND_FT_REFERENCE.json",
}
LEAK_FILE = QUEUE / "FT_EVAL_LEAK.json"
TABLE3_META = REPO / "challenge/yolo_pose_one_model/legacy_v1v2_ft/TABLE3_META.json"
CURRENT_SUMMARY = SCRATCH / "SPATIAL_CONCAT_60K_SUMMARY.json"
CURRENT_REAL_COMPARISON = SCRATCH / "REAL_DEV_BASE_COMPARISON.json"
NEW_CHECKPOINT = (
    SCRATCH
    / "runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/last.pt"
)
OLD_CHECKPOINT = (
    REPO
    / "challenge/yolo_pose_one_model/runs/stage_a_synth_640_b32_seed42/weights/best.pt"
)
FT_CHECKPOINT = Path(
    "/home/minjae/Documents/github/25y_automatic_lifter-master/"
    "pallet_yolo26n_pose_ft.pt"
)
PAPER_BASELINE = (
    REPO
    / "challenge/yolo_pose_one_model/runs_camera_facing_loss/"
    "OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt"
)
OUTPUT_JSON = HERE / "TARGET_MODEL_COMPARISON.json"
OUTPUT_MD = HERE / "TARGET_MODEL_COMPARISON.md"


class ComparisonError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise ComparisonError(f"required artifact missing: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def frame_map(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = payload.get("per_frame")
    if not isinstance(rows, list):
        raise ComparisonError("real artifact has no per_frame rows")
    output = {str(row["frame"]): row for row in rows}
    if len(output) != len(rows):
        raise ComparisonError("duplicate frame id in real artifact")
    return output


def flattened_errors(rows: Iterable[Mapping[str, Any]]) -> np.ndarray:
    values = [np.asarray(row["err"], dtype=np.float64) for row in rows]
    if not values:
        return np.empty(0, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 8 or not np.isfinite(array).all():
        raise ComparisonError(f"invalid corner error array: {array.shape}")
    return array.reshape(-1)


def localization(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = flattened_errors(rows)
    night_rows = [row for row in rows if row["domain"] == "NIGHT"]
    night_errors = flattened_errors(night_rows)
    return {
        "n_frames": len(rows),
        "corner_median_px": float(np.median(errors)),
        "corner_p90_px": float(np.quantile(errors, 0.9)),
        "gross20_fraction": float(np.mean(errors > 20.0)),
        "night": {
            "n_frames": len(night_rows),
            "corner_median_px": float(np.median(night_errors))
            if len(night_errors)
            else None,
            "corner_p90_px": float(np.quantile(night_errors, 0.9))
            if len(night_errors)
            else None,
        },
    }


def relative_delta(candidate: float, reference: float, *, lower_is_better: bool) -> float:
    if reference == 0.0:
        return math.nan
    raw = (candidate - reference) / abs(reference)
    return -raw if not lower_is_better else raw


def population_comparison(
    candidate: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
    frame_ids: Sequence[str],
) -> dict[str, Any]:
    missing = [frame for frame in frame_ids if frame not in candidate or frame not in reference]
    if missing:
        raise ComparisonError(f"population rows missing: {missing[:3]}")
    candidate_correct = [
        frame for frame in frame_ids if candidate[frame].get("correct_box", False)
    ]
    reference_correct = [
        frame for frame in frame_ids if reference[frame].get("correct_box", False)
    ]
    both = [
        frame
        for frame in frame_ids
        if candidate[frame].get("correct_box", False)
        and reference[frame].get("correct_box", False)
    ]
    candidate_loc = localization([candidate[frame] for frame in both])
    reference_loc = localization([reference[frame] for frame in both])
    frame_delta = np.asarray(
        [
            np.median(np.asarray(candidate[frame]["err"], dtype=np.float64))
            - np.median(np.asarray(reference[frame]["err"], dtype=np.float64))
            for frame in both
        ],
        dtype=np.float64,
    )
    return {
        "population_n": len(frame_ids),
        "candidate_correct_box_n": len(candidate_correct),
        "reference_correct_box_n": len(reference_correct),
        "correct_box_gap_frames": len(candidate_correct) - len(reference_correct),
        "correct_box_gap_pp": 100.0
        * (len(candidate_correct) - len(reference_correct))
        / len(frame_ids),
        "pairwise_both_correct_box_n": len(both),
        "pairwise_frame_ids_sha256": hashlib.sha256(
            ("\n".join(both) + "\n").encode("utf-8")
        ).hexdigest(),
        "candidate_localization": candidate_loc,
        "reference_localization": reference_loc,
        "candidate_minus_reference": {
            "corner_median_px": candidate_loc["corner_median_px"]
            - reference_loc["corner_median_px"],
            "corner_median_relative_worse": relative_delta(
                candidate_loc["corner_median_px"],
                reference_loc["corner_median_px"],
                lower_is_better=True,
            ),
            "corner_p90_px": candidate_loc["corner_p90_px"]
            - reference_loc["corner_p90_px"],
            "corner_p90_relative_worse": relative_delta(
                candidate_loc["corner_p90_px"],
                reference_loc["corner_p90_px"],
                lower_is_better=True,
            ),
            "gross20_fraction": candidate_loc["gross20_fraction"]
            - reference_loc["gross20_fraction"],
            "paired_frame_median_error_delta_median_px": float(np.median(frame_delta)),
            "candidate_better_frame_fraction": float(np.mean(frame_delta < 0.0)),
        },
    }


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    ordered = np.sort(neg)
    less = np.searchsorted(ordered, pos, side="left")
    less_equal = np.searchsorted(ordered, pos, side="right")
    return float(np.mean((less + less_equal) / (2.0 * len(ordered))))


def fpr_at_recall(pos: np.ndarray, neg: np.ndarray, recall: float) -> dict[str, float]:
    if not 0.0 < recall <= 1.0:
        raise ComparisonError(f"invalid recall: {recall}")
    # Highest threshold retaining at least the requested empirical recall.
    descending = np.sort(pos)[::-1]
    rank = max(0, int(math.ceil(recall * len(descending))) - 1)
    threshold = float(descending[rank])
    return {
        "threshold": threshold,
        "actual_positive_recall": float(np.mean(pos >= threshold)),
        "negative_false_positive_rate": float(np.mean(neg >= threshold)),
    }


def negative_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ComparisonError("negative-score artifact has no rows")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    scores = np.asarray([float(row["max_conf"]) for row in rows], dtype=np.float64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) != 128 or len(neg) != 2689:
        raise ComparisonError(f"unexpected positive/negative population: {len(pos)}/{len(neg)}")
    return {
        "n_positive": len(pos),
        "n_negative": len(neg),
        "auroc": auroc(pos, neg),
        "at_confidence_0_4": {
            "positive_recall": float(np.mean(pos >= 0.4)),
            "negative_false_positive_rate": float(np.mean(neg >= 0.4)),
        },
        "positive_median_confidence": float(np.median(pos)),
        "negative_p99_confidence": float(np.quantile(neg, 0.99)),
        "fixed_recall": {
            str(value): fpr_at_recall(pos, neg, value) for value in (0.75, 0.85, 0.90)
        },
    }


def gap_closure(higher_baseline: float, candidate: float, target: float) -> float | None:
    denominator = target - higher_baseline
    if denominator == 0.0:
        return None
    return float((candidate - higher_baseline) / denominator)


def lower_gap_closure(baseline: float, candidate: float, target: float) -> float | None:
    denominator = baseline - target
    if denominator == 0.0:
        return None
    return float((baseline - candidate) / denominator)


def build() -> dict[str, Any]:
    table_meta = load_json(TABLE3_META)
    if table_meta.get("columns", {}).get("TARGET") != "FT_REFERENCE":
        raise ComparisonError("TABLE3 target binding changed")
    leak = load_json(LEAK_FILE)
    leaked = {str(value) for value in leak["leaked_frame_ids"]}
    if len(leaked) != 12 or int(leak.get("night_leak", -1)) != 0:
        raise ComparisonError("FT leakage contract changed")

    real_payloads = {key: load_json(path) for key, path in REAL_FILES.items()}
    maps = {key: frame_map(payload) for key, payload in real_payloads.items()}
    common = sorted(set.intersection(*(set(value) for value in maps.values())))
    if len(common) != 140:
        raise ComparisonError(f"expected 140 common real frames, observed {len(common)}")
    dev128 = [frame for frame in common if frame not in leaked]
    if len(dev128) != 128:
        raise ComparisonError(f"expected DEV128 after leak removal, observed {len(dev128)}")

    primary = {
        "DEV128_LEAK_EXCLUDED": population_comparison(
            maps["NEW"], maps["FT_REFERENCE"], dev128
        ),
        "DEV140_REFERENCE_ONLY": population_comparison(
            maps["NEW"], maps["FT_REFERENCE"], common
        ),
    }
    milestone = {
        "DEV128_LEAK_EXCLUDED": population_comparison(
            maps["NEW"], maps["OLD_STAGE_A"], dev128
        ),
        "DEV140_REFERENCE_ONLY": population_comparison(
            maps["NEW"], maps["OLD_STAGE_A"], common
        ),
    }

    night = {key: load_json(path) for key, path in NIGHT_FILES.items()}
    night_summary = {
        key: {
            "n": int(value["n"]),
            "top1_correct_box_n": int(round(float(value["top1_cbox"]) * int(value["n"]))),
            "top1_correct_box_fraction": float(value["top1_cbox"]),
            "any_correct_box_n": int(round(float(value["any_cbox"]) * int(value["n"]))),
            "any_correct_box_fraction": float(value["any_cbox"]),
            "candidates_per_frame": float(value["cand_per_frame"]),
            "margin_median": float(value["margin_median"]),
        }
        for key, value in night.items()
    }

    negative = {
        key: negative_metrics(load_json(path)) for key, path in NEG_FILES.items()
    }

    # Fair G38 -> NEW -> FT gap closure on leak-adjusted DEV128. Localization
    # uses only frames with a correct top-1 box for all three models.
    triple = [
        frame
        for frame in dev128
        if maps["G38"][frame].get("correct_box", False)
        and maps["NEW"][frame].get("correct_box", False)
        and maps["FT_REFERENCE"][frame].get("correct_box", False)
    ]
    triple_loc = {
        key: localization([maps[key][frame] for frame in triple])
        for key in ("G38", "NEW", "FT_REFERENCE")
    }
    cbox = {
        key: sum(bool(maps[key][frame].get("correct_box", False)) for frame in dev128)
        for key in ("G38", "NEW", "FT_REFERENCE")
    }
    closure = {
        "population": "DEV128_LEAK_EXCLUDED",
        "correct_box": {
            "values_n": cbox,
            "fraction_of_G38_to_FT_gap_closed": gap_closure(
                float(cbox["G38"]), float(cbox["NEW"]), float(cbox["FT_REFERENCE"])
            ),
        },
        "triple_correct_box_localization": {
            "n_frames": len(triple),
            "frame_ids_sha256": hashlib.sha256(
                ("\n".join(triple) + "\n").encode("utf-8")
            ).hexdigest(),
            "values": triple_loc,
            "corner_median_fraction_of_gap_closed": lower_gap_closure(
                triple_loc["G38"]["corner_median_px"],
                triple_loc["NEW"]["corner_median_px"],
                triple_loc["FT_REFERENCE"]["corner_median_px"],
            ),
            "corner_p90_fraction_of_gap_closed": lower_gap_closure(
                triple_loc["G38"]["corner_p90_px"],
                triple_loc["NEW"]["corner_p90_px"],
                triple_loc["FT_REFERENCE"]["corner_p90_px"],
            ),
        },
        "night": {
            "top1_fraction_of_gap_closed": gap_closure(
                night_summary["G38"]["top1_correct_box_n"],
                night_summary["NEW"]["top1_correct_box_n"],
                night_summary["FT_REFERENCE"]["top1_correct_box_n"],
            ),
            "any_fraction_of_gap_closed": gap_closure(
                night_summary["G38"]["any_correct_box_n"],
                night_summary["NEW"]["any_correct_box_n"],
                night_summary["FT_REFERENCE"]["any_correct_box_n"],
            ),
            "margin_fraction_of_gap_closed": gap_closure(
                night_summary["G38"]["margin_median"],
                night_summary["NEW"]["margin_median"],
                night_summary["FT_REFERENCE"]["margin_median"],
            ),
        },
    }

    paths = {
        "table3_meta": TABLE3_META,
        "leak_contract": LEAK_FILE,
        "current_spatial_summary": CURRENT_SUMMARY,
        "current_real_comparison": CURRENT_REAL_COMPARISON,
        "new_checkpoint": NEW_CHECKPOINT,
        "old_stage_a_checkpoint": OLD_CHECKPOINT,
        "ft_reference_checkpoint": FT_CHECKPOINT,
        "current_paper_baseline_checkpoint": PAPER_BASELINE,
        **{f"real_{key.lower()}": value for key, value in REAL_FILES.items()},
        **{f"night_{key.lower()}": value for key, value in NIGHT_FILES.items()},
        **{f"negative_{key.lower()}": value for key, value in NEG_FILES.items()},
    }
    for name, path in paths.items():
        if not path.is_file():
            raise ComparisonError(f"bound artifact missing ({name}): {path}")

    return {
        "schema_version": "cleanstart_target_model_comparison_v1",
        "scope": "reused development diagnostics only; not FINAL or paper-table evidence",
        "target_definition": {
            "primary": "FT_REFERENCE",
            "primary_role": "historical real-supervised target/upper bound; unfair as a zero-real peer",
            "primary_known_leakage": "12 DEV140 DAY frames occurred in FT training; primary result excludes them",
            "secondary": "OLD_STAGE_A",
            "secondary_role": "target-specific synthetic milestone for G38-to-OLD recovery",
            "paper_baseline": "G38",
            "paper_baseline_unchanged": True,
        },
        "candidate": {
            "tag": TAGS["NEW"],
            "checkpoint": str(NEW_CHECKPOINT),
            "checkpoint_sha256": sha256(NEW_CHECKPOINT),
            "base_yolo_only": True,
        },
        "primary_target_comparison": primary,
        "secondary_old_milestone_comparison": milestone,
        "night28": night_summary,
        "positive_negative_128_2689": negative,
        "g38_to_ft_gap_closure": closure,
        "spatial_s1_comparison_status": {
            "commensurate_target_comparison_available": False,
            "reason": (
                "S1 is a synthetic-DEV post-hoc parity probe; it has no real adapter, "
                "PnP evaluation, or target model evaluated on the same parity population"
            ),
            "required_next_evidence": (
                "freeze the parity winner without real access, extract real spatial features, "
                "wire parity to PnP, and evaluate on a named untouched population"
            ),
            "s1_checkpoint": load_json(CURRENT_SUMMARY)["locked_median_artifacts"]
            ["S1_SPATIAL_CONCAT"]["checkpoint"],
        },
        "negative_provenance_caveat": (
            "FT_REFERENCE negative diagnostics are not a clean target-free comparison; "
            "the historical negative provenance marks FT_HONEST_NEGATIVE_EVAL=false"
        ),
        "source_bindings": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in sorted(paths.items())
        },
        "runner_sha256": sha256(Path(__file__)),
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(result: Mapping[str, Any]) -> str:
    primary = result["primary_target_comparison"]["DEV128_LEAK_EXCLUDED"]
    old = result["secondary_old_milestone_comparison"]["DEV128_LEAK_EXCLUDED"]
    new_loc = primary["candidate_localization"]
    ft_loc = primary["reference_localization"]
    old_new_loc = old["candidate_localization"]
    old_loc = old["reference_localization"]
    night = result["night28"]
    neg = result["positive_negative_128_2689"]
    closure = result["g38_to_ft_gap_closure"]
    old_p90_relative = old["candidate_minus_reference"]["corner_p90_relative_worse"]
    old_p90_wording = (
        f"{pct(abs(old_p90_relative))} worse"
        if old_p90_relative >= 0.0
        else f"{pct(abs(old_p90_relative))} better"
    )
    lines = [
        "# New clean-start model versus historical targets",
        "",
        "> DEV diagnostic only. These values cannot populate a FINAL paper cell or replace the current G38 paper baseline.",
        "",
        "`FT_REFERENCE` is the repository-declared primary target, but it is a real-supervised upper bound. "
        "Its 12 known overlapping DAY frames are removed from the primary DEV128 comparison. "
        "`OLD_STAGE_A` is the secondary target-specific synthetic milestone.",
        "",
        "## Leak-adjusted DEV128",
        "",
        "| Comparison | Correct box | Pairwise both-correct | Corner median | Corner p90 | NIGHT top-1 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| New | {primary['candidate_correct_box_n']}/128 | "
            f"{primary['pairwise_both_correct_box_n']} | {new_loc['corner_median_px']:.3f}px | "
            f"{new_loc['corner_p90_px']:.3f}px | {night['NEW']['top1_correct_box_n']}/28 |"
        ),
        (
            f"| FT_REFERENCE target | {primary['reference_correct_box_n']}/128 | "
            f"{primary['pairwise_both_correct_box_n']} | {ft_loc['corner_median_px']:.3f}px | "
            f"{ft_loc['corner_p90_px']:.3f}px | {night['FT_REFERENCE']['top1_correct_box_n']}/28 |"
        ),
        (
            f"| OLD_STAGE_A milestone | {old['reference_correct_box_n']}/128 | "
            f"{old['pairwise_both_correct_box_n']} | {old_loc['corner_median_px']:.3f}px | "
            f"{old_loc['corner_p90_px']:.3f}px | {night['OLD_STAGE_A']['top1_correct_box_n']}/28 |"
        ),
        "",
        f"Against FT_REFERENCE, the new base is {abs(primary['correct_box_gap_frames'])} frames "
        f"({abs(primary['correct_box_gap_pp']):.3f} pp) lower in correct-box coverage. On the "
        f"{primary['pairwise_both_correct_box_n']} pairwise both-correct frames, median and p90 "
        f"corner errors are {pct(primary['candidate_minus_reference']['corner_median_relative_worse'])} "
        f"and {pct(primary['candidate_minus_reference']['corner_p90_relative_worse'])} worse.",
        "",
        f"Against OLD_STAGE_A, correct-box coverage is {abs(old['correct_box_gap_frames'])} frames lower. "
        f"On {old['pairwise_both_correct_box_n']} both-correct frames, median is effectively tied "
        f"({old_new_loc['corner_median_px']:.3f}px vs {old_loc['corner_median_px']:.3f}px), while p90 is "
        f"{old_p90_wording}.",
        "",
        "## Detection ranking on leak-adjusted positive/negative DEV",
        "",
        "| Model | AUROC | Positive recall @0.4 | Negative FPR @0.4 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (("NEW", "New"), ("OLD_STAGE_A", "OLD_STAGE_A"), ("FT_REFERENCE", "FT_REFERENCE")):
        values = neg[key]
        lines.append(
            f"| {label} | {values['auroc']:.6f} | "
            f"{pct(values['at_confidence_0_4']['positive_recall'])} | "
            f"{pct(values['at_confidence_0_4']['negative_false_positive_rate'])} |"
        )
    lines.extend(
        [
            "",
            "The FT negative result is an upper-bound diagnostic, not a clean target-free comparison; its historical provenance explicitly marks the negative evaluation as non-honest.",
            "",
            "## Fraction of the G38-to-FT gap recovered",
            "",
            f"- Correct-box coverage: {pct(closure['correct_box']['fraction_of_G38_to_FT_gap_closed'])}",
            f"- Fair common-frame corner median: {pct(closure['triple_correct_box_localization']['corner_median_fraction_of_gap_closed'])}",
            f"- Fair common-frame corner p90: {pct(closure['triple_correct_box_localization']['corner_p90_fraction_of_gap_closed'])}",
            f"- NIGHT top-1: {pct(closure['night']['top1_fraction_of_gap_closed'])}",
            "",
            "## S1 spatial head status",
            "",
            "S1 cannot yet be compared honestly with FT_REFERENCE. It is a post-hoc parity head measured on mixed synthetic DEV, whereas the target values above are real YOLO detection/keypoint diagnostics. A real spatial-feature adapter, parity-to-PnP wiring, and a named untouched evaluation population are still required.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    result = build()
    atomic_write(OUTPUT_JSON, json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    atomic_write(OUTPUT_MD, render_markdown(result))
    print(json.dumps({
        "json": str(OUTPUT_JSON.relative_to(REPO)),
        "json_sha256": sha256(OUTPUT_JSON),
        "markdown": str(OUTPUT_MD.relative_to(REPO)),
        "markdown_sha256": sha256(OUTPUT_MD),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
