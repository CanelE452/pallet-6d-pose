"""논문-facing domain naming 계약.

MAIN 논문 표가 쓰는 조건은 Daytime / Nighttime 두 개뿐이고, 내부 capture id
(`outside` · `noapril` · `cad`)는 결과표 헤더로 쓰지 않는다.  이 규약은 사람이
문서를 고칠 때 조용히 깨지기 쉬워서 테스트로 고정한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

from eval_workspace import (  # noqa: E402
    PAPER_DOMAIN_RULES,
    derive_paper_domain,
    load_frames,
    load_targets,
    evaluation_population_views,
    main_domains_ready,
    paper_domain_rows,
)

EXPERIMENTS = REPO_ROOT / "_docs/paper/EXPERIMENTS.md"
WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"


def _main_section() -> str:
    text = EXPERIMENTS.read_text(encoding="utf-8")
    start = text.index("# PART I")
    end = text.index("# PART II")
    body = text[start:end]
    # wording note 는 "쓰지 말라" 는 예시를 일부러 담고 있으므로 제외한다.
    if "# Wording" in body:
        head, _, tail = body.partition("# Wording")
        _, _, rest = tail.partition("\n---\n")
        body = head + rest
    return body


def test_main_section_has_no_internal_capture_ids() -> None:
    body = _main_section().lower()
    for token in ("noapril", "outside", "cad"):
        assert token not in body, f"MAIN 영역에 내부 capture id {token!r} 가 있다"


def test_main_section_does_not_claim_indoor_outdoor_factorial() -> None:
    body = _main_section().lower()
    assert "indoor/outdoor" not in body
    assert "indoor" not in body and "outdoor" not in body


def test_table_two_columns_are_daytime_and_nighttime() -> None:
    body = _main_section()
    header = next(
        line for line in body.splitlines()
        if line.startswith("Method") and "Daytime" in line
    )
    for column in ("Daytime", "Nighttime", "Mean", "Worst"):
        assert column in header, f"Table 2 에 {column} 열이 없다"


def test_paper_domain_mapping_is_exactly_three_conditions() -> None:
    assert PAPER_DOMAIN_RULES["daytime"] == {
        "acquisition_domain": "outside", "lighting": "day",
        "object_type": "plastic",
    }
    assert PAPER_DOMAIN_RULES["nighttime"] == {
        "acquisition_domain": "night", "lighting": "night",
        "object_type": "plastic",
    }


@pytest.mark.parametrize("internal", ["noapril", "cad", "forklift", "unknown"])
def test_non_main_capture_cannot_enter_a_paper_domain(internal: str) -> None:
    row = {"acquisition_domain": internal, "lighting": "day",
           "object_type": "plastic"}
    assert derive_paper_domain(row) == "none"


def test_wood_cannot_enter_a_paper_domain() -> None:
    row = {"acquisition_domain": "outside", "lighting": "day",
           "object_type": "wood"}
    assert derive_paper_domain(row) == "none"


def test_lighting_must_match_the_capture_domain() -> None:
    # outside 인데 야간으로 태깅된 프레임은 daytime 이 아니다.
    row = {"acquisition_domain": "outside", "lighting": "night",
           "object_type": "plastic"}
    assert derive_paper_domain(row) == "none"


def test_current_counts_are_recomputed_from_the_manifest() -> None:
    frames = load_frames(WORKSPACE)
    targets = load_targets(WORKSPACE)
    rows = {r["paper_domain"]: r for r in paper_domain_rows(frames, targets)}
    positive = evaluation_population_views(frames)["PAPER_EVAL_POSITIVE"]

    for domain in ("daytime", "nighttime"):
        expected = sum(1 for r in positive if derive_paper_domain(r) == domain)
        assert rows[domain]["frames"] == expected

    # 두 조건이 서로 겹치지 않는다.
    day = {r["frame_id"] for r in positive if derive_paper_domain(r) == "daytime"}
    night = {r["frame_id"] for r in positive if derive_paper_domain(r) == "nighttime"}
    assert not (day & night)


def test_dataset_is_not_ready_while_a_condition_is_below_minimum() -> None:
    frames = load_frames(WORKSPACE)
    targets = load_targets(WORKSPACE)
    rows = paper_domain_rows(frames, targets)
    short = [r for r in rows if r["frames"] < r["minimum"]]
    if short:
        assert main_domains_ready(rows) is False


def test_historical_internal_domains_are_preserved() -> None:
    """내부 provenance 는 그대로 남아 있어야 한다 — 삭제·개명 금지."""
    frames = load_frames(WORKSPACE)
    present = {str(r.get("acquisition_domain", "")) for r in frames}
    for internal in ("outside", "night", "noapril", "cad"):
        assert internal in present, f"내부 domain {internal!r} 가 사라졌다"
    assert (WORKSPACE / "ACQUISITION_DOMAIN_MAP.json").is_file()
