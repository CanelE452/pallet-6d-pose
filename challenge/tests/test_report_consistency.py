"""세 리포트가 같은 source-of-truth 에서 계산되는지 고정한다.

DATASET_COMPOSITION.md 가 frozen DEV alias(`FINAL_EVAL_POSITIVE`, 173행 고정)를
읽는 바람에, ANNOTATION_PROGRESS 가 319 를 보이는 동안 173 을 보이고 있었다.
새로 어노테이션한 프레임이 영원히 안 보이는 상태였다.

세 리포트가 갈리지 않게 하려면 사람이 문서를 고칠 때가 아니라 여기서 막아야 한다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

from eval_workspace import (  # noqa: E402
    compute_progress,
    condition_membership,
    derive_paper_domain,
    evaluation_population_views,
    load_frames,
    load_targets,
    paper_domain_rows,
)

WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
REPORTS = WORKSPACE / "reports"


def _frames():
    return load_frames(WORKSPACE)


def _annotated_positive(frames):
    return [
        row for row in evaluation_population_views(frames)["PAPER_EVAL_POSITIVE"]
        if str(row.get("is_annotated", "")).strip().lower() == "true"
    ]


def _int_after(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    assert match, f"패턴을 못 찾음: {pattern}"
    return int(match.group(1))


def test_composition_does_not_read_the_frozen_dev_alias() -> None:
    """FINAL_EVAL 은 173행 고정이라 paper-facing 표의 population 이 될 수 없다."""
    text = (REPORTS / "DATASET_COMPOSITION.md").read_text(encoding="utf-8")
    table_rows = [
        line for line in text.splitlines()
        if line.startswith("FINAL_EVAL ") or line.startswith("FINAL_EVAL\t")
    ]
    assert not table_rows, f"표에 FINAL_EVAL 행이 남아 있다: {table_rows[:3]}"
    assert "PAPER_EVAL" in text


def test_positive_total_matches_across_progress_and_composition() -> None:
    frames = _frames()
    expected = compute_progress(frames).positive_total

    progress = (REPORTS / "ANNOTATION_PROGRESS.md").read_text(encoding="utf-8")
    composition = (REPORTS / "DATASET_COMPOSITION.md").read_text(encoding="utf-8")

    assert _int_after(progress, r"Positive total\s+(\d+)") == expected
    assert _int_after(
        composition, r"PAPER_EVAL\s+Combined positive\s+(\d+)") == expected


def test_object_counts_match_the_manifest() -> None:
    frames = _frames()
    progress = compute_progress(frames)
    composition = (REPORTS / "DATASET_COMPOSITION.md").read_text(encoding="utf-8")
    for label, value in (("Plastic", progress.plastic), ("Wood", progress.wood)):
        assert _int_after(
            composition, rf"PAPER_EVAL\s+{label}\s+(\d+)") == value


def test_paper_domain_counts_match_across_three_reports() -> None:
    frames = _frames()
    rows = {r["paper_domain"]: r for r in paper_domain_rows(frames, load_targets(WORKSPACE))}
    progress = (REPORTS / "ANNOTATION_PROGRESS.md").read_text(encoding="utf-8")
    coverage = (REPORTS / "PAPER_DOMAIN_COVERAGE.md").read_text(encoding="utf-8")
    composition = (REPORTS / "DATASET_COMPOSITION.md").read_text(encoding="utf-8")

    for domain, label in (("daytime", "Daytime"), ("nighttime", "Nighttime")):
        n = rows[domain]["frames"]
        sessions = rows[domain]["sessions"]
        for text, name in ((progress, "ANNOTATION_PROGRESS"),
                           (coverage, "PAPER_DOMAIN_COVERAGE"),
                           (composition, "DATASET_COMPOSITION")):
            found = re.search(rf"{label}\s+\S+\s+(\d+)", text)
            assert found, f"{name} 에 {label} 행이 없다"
            assert int(found.group(1)) == n, (
                f"{name} 의 {label} N={found.group(1)} != manifest {n}")
        # 세션 수는 두 리포트가 낸다
        for text, name in ((coverage, "PAPER_DOMAIN_COVERAGE"),
                           (composition, "DATASET_COMPOSITION")):
            assert str(sessions) in text, f"{name} 에 {label} 세션 수가 없다"


def test_condition_counts_agree_between_progress_and_composition() -> None:
    """조건 수치는 두 리포트가 같은 정의(condition_membership)로 세야 한다."""
    frames = _frames()
    progress_value = compute_progress(frames)
    composition = (REPORTS / "DATASET_COMPOSITION.md").read_text(encoding="utf-8")

    for label, value in (("Clean", progress_value.clean),
                         ("Occlusion", progress_value.occlusion),
                         ("Truncation", progress_value.truncation),
                         ("Far", progress_value.far),
                         ("Low angle", progress_value.low),
                         ("Mid angle", progress_value.mid),
                         ("High angle", progress_value.high)):
        found = re.search(rf"PAPER_EVAL\s+{label}\s+(\d+)", composition)
        assert found, f"DATASET_COMPOSITION 에 {label} 이 없다"
        assert int(found.group(1)) == value, (
            f"{label}: composition {found.group(1)} != progress {value}")


def test_staged_unannotated_images_do_not_inflate_any_report() -> None:
    """복사만 해둔 미어노테이션 이미지는 어느 리포트에도 세어지면 안 된다."""
    frames = _frames()
    positive = evaluation_population_views(frames)["PAPER_EVAL_POSITIVE"]
    annotated = _annotated_positive(frames)
    assert len(annotated) == compute_progress(frames).positive_total
    if len(positive) > len(annotated):
        # staging 이 실재하는 동안에만 의미 있는 검사다.
        composition = (REPORTS / "DATASET_COMPOSITION.md").read_text(encoding="utf-8")
        assert str(len(positive)) not in re.findall(
            r"Combined positive\s+(\d+)", composition)


def test_domain_counts_come_from_the_same_derivation() -> None:
    """리포트가 자기 규칙을 새로 만들지 않았는지 — 직접 세어 대조한다."""
    frames = _frames()
    rows = {r["paper_domain"]: r for r in paper_domain_rows(frames, load_targets(WORKSPACE))}
    annotated = _annotated_positive(frames)
    for domain in ("daytime", "nighttime"):
        direct = sum(1 for r in annotated if derive_paper_domain(r) == domain)
        assert rows[domain]["frames"] == direct


def test_condition_membership_is_the_only_condition_definition() -> None:
    frames = _frames()
    annotated = _annotated_positive(frames)
    progress_value = compute_progress(frames)
    direct_far = sum(1 for r in annotated if "far" in condition_membership(r))
    assert progress_value.far == direct_far
