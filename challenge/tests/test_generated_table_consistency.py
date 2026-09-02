"""생성된 논문 표들이 서로 모순되지 않는지 강제한다.

M2 서술의 "Daytime strict keypoint 수" 가 DOPE 의 검출 수(351)를 데이터셋 속성처럼
적고 있었고, 같은 저장소의 M5 는 609 를 적고 있었다.  두 표를 사람이 눈으로
대조하지 않으면 안 잡힌다.  그래서 검사로 둔다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED = REPO_ROOT / "_docs" / "paper" / "generated"
ARM_RESULTS = (REPO_ROOT / "data" / "pallet" / "results" / "paper_eval_v1"
               / "arms" / "ARM_RESULTS.json")

pytestmark = pytest.mark.skipif(
    not ARM_RESULTS.exists() or not (GENERATED / "TABLE_M2.md").exists(),
    reason="생성 산출물이 없다 — build_experiment_tables.py 를 먼저 돌린다",
)


def arm_results() -> dict:
    return json.loads(ARM_RESULTS.read_text())


def m2_daytime_strict_count() -> int:
    text = (GENERATED / "TABLE_M2.md").read_text()
    match = re.search(r"Daytime strict keypoint 수 = (\d+)", text)
    assert match, "M2 에 Daytime strict keypoint 수 문장이 없다"
    return int(match.group(1))


def m5_daytime_row() -> list[str]:
    for line in (GENERATED / "TABLE_M5.md").read_text().splitlines():
        if line.startswith("Daytime "):
            return line.split()
    raise AssertionError("M5 에 Daytime 행이 없다")


def test_m2_and_m5_agree_on_the_daytime_keypoint_count():
    row = m5_daytime_row()
    # Condition N src n_kp ...
    assert row[2] == "strict", f"M5 Daytime 이 strict 가 아니다: {row[2]}"
    assert int(row[3]) == m2_daytime_strict_count()


def test_the_count_comes_from_the_declared_reference_model():
    """숫자가 실제로 그 모델의 값인지 — 두 표가 나란히 틀렸을 수도 있다."""

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_experiment_tables",
        REPO_ROOT / "scripts" / "paper" / "build_experiment_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.M2_REFERENCE_MODEL
    expected = (arm_results()["models"][reference]["subgroups"]["Daytime"]
                ["n_keypoints"])
    assert m2_daytime_strict_count() == expected


def test_m2_names_the_reference_model_because_the_count_is_model_dependent():
    text = (GENERATED / "TABLE_M2.md").read_text()
    assert "참조 모델" in text
    models = arm_results()["models"]
    counts = {name: block["subgroups"]["Daytime"]["n_keypoints"]
              for name, block in models.items()}
    assert len(set(counts.values())) > 1, (
        "모든 모델의 Daytime n_keypoints 가 같다면 이 경고는 불필요하다 — "
        "그때는 이 테스트를 지워라")


# ── 단위 표기 ──────────────────────────────────────────────────────────

CORNER_TABLES = ("TABLE_M1.md", "TABLE_M2.md", "TABLE_M4.md", "TABLE_M5.md",
                 "APPENDIX_TABLES.md")


@pytest.mark.parametrize("name", CORNER_TABLES)
def test_corner_columns_declare_pixel_units(name):
    """corner 는 원본 이미지 Euclidean px 다.  표에서 단위를 숨기지 않는다."""

    path = GENERATED / name
    if not path.exists():
        pytest.skip(f"{name} 없음")
    text = path.read_text()
    if "corner" not in text:
        pytest.skip(f"{name} 에 corner 열이 없다")
    headers = [line for line in text.splitlines()
               if "corner" in line and ("↓" in line or "~" in line)
               and "px" not in line.split("corner")[0]]
    bare = [line for line in headers
            if re.search(r"corner[ A-Za-z]*[↓~](?!\[px\])", line)]
    assert not bare, f"{name} 에 단위 없는 corner 헤더가 있다: {bare[:2]}"
