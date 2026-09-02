"""V5 reliability score 를 계약으로 못 박는다.

이 점수는 **학습되지 않는다**.  고정 수식 · 무감독 · 단조 · rank fusion 이다.
그리고 GT 를 읽지 않는다 — 문자열 검사는 주석에 속으므로 AST 로 강제한다
(`forbidden-token-tests-must-use-ast`).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "scripts" / "self_training_yolo" / "v5" / "reliability_score.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("v5_reliability_score", MODULE)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules["v5_reliability_score"] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def record(index: int, box=0.9, reproj=0.01, remove=0.01, flip=0.01,
           kp=0.99, r_remove=0.01, r_flip=0.01, condition="daytime") -> dict:
    return {
        "frame_id": f"f{index}", "condition": condition,
        "box_conf": box, "s_reproj": reproj, "s_remove": remove, "s_flip": flip,
        "kp_conf": [kp] * 8, "r_remove": [r_remove] * 8, "r_flip": [r_flip] * 8,
    }


# ── mid-rank 정규화 ────────────────────────────────────────────────────

def test_quality_stays_strictly_inside_zero_and_one(module):
    """0 이 나오면 기하평균이 통째로 0 이 된다.  경계를 열어 둔다."""

    values = module.mid_rank_quality([1.0, 2.0, 3.0, 4.0], higher_is_better=True)
    assert values.min() > 0.0 and values.max() < 1.0


def test_good_low_signal_is_inverted(module):
    """잔차는 작을수록 좋다 — 방향이 뒤집혀야 한다."""

    ascending = module.mid_rank_quality([0.01, 0.05, 0.5], higher_is_better=False)
    assert ascending[0] > ascending[1] > ascending[2]


def test_ties_get_the_same_quality(module):
    values = module.mid_rank_quality([1.0, 1.0, 2.0], higher_is_better=True)
    assert values[0] == pytest.approx(values[1])


def test_missing_values_are_pushed_to_the_worst_end(module):
    """신호가 없다고 좋게 봐주지 않는다."""

    good_high = module.mid_rank_quality([0.9, np.nan, 0.5], higher_is_better=True)
    assert good_high[1] == good_high.min()
    good_low = module.mid_rank_quality([0.01, np.inf, 0.5], higher_is_better=False)
    assert good_low[1] == good_low.min()


def test_the_score_is_rank_based_not_scale_based(module):
    """단조 변환에 불변이어야 한다 — 절대값을 곱하지 않는다는 뜻이다."""

    base = [record(i, box=0.80 + 0.01 * i) for i in range(6)]
    stretched = [record(i, box=(0.80 + 0.01 * i) ** 3) for i in range(6)]
    a = [r["R_total"] for r in module.score_pool(base)]
    b = [r["R_total"] for r in module.score_pool(stretched)]
    assert np.allclose(a, b)


# ── 단조성 ─────────────────────────────────────────────────────────────

def test_better_signals_give_a_higher_score(module):
    records = [record(0, box=0.99, reproj=0.001, remove=0.001, flip=0.001,
                      kp=0.999, r_remove=0.001, r_flip=0.001),
               record(1, box=0.86, reproj=0.049, remove=0.049, flip=0.049,
                      kp=0.51, r_remove=0.049, r_flip=0.049),
               record(2, box=0.92, reproj=0.02, remove=0.02, flip=0.02)]
    scored = {r["frame_id"]: r["R_total"] for r in module.score_pool(records)}
    assert scored["f0"] > scored["f2"] > scored["f1"]


# ── condition 분리 ─────────────────────────────────────────────────────

def test_day_and_night_are_normalised_separately(module):
    """야간의 raw 스케일이 달라도 야간 안에서의 순위만 반영되어야 한다."""

    day = [record(i, box=0.90 + 0.01 * i, condition="daytime") for i in range(4)]
    night = [record(10 + i, box=0.50 + 0.01 * i, condition="nighttime")
             for i in range(4)]
    scored = {r["frame_id"]: r["R_total"] for r in module.score_pool(day + night)}
    # 야간 최고가 주간 최저보다 낮을 이유가 없다 — 절대값을 쓰지 않으므로.
    assert scored["f13"] == pytest.approx(scored["f3"])


# ── 블록 가중치 ────────────────────────────────────────────────────────

def test_frame_and_keypoint_blocks_carry_equal_weight(module):
    scored = module.score_pool([record(i) for i in range(5)])
    for row in scored:
        assert row["R_total"] == pytest.approx(
            float(np.sqrt(row["R_frame_geom"] * row["R_kp_frame"])))


def test_a_corner_with_one_usable_signal_is_dropped(module):
    rows = [record(i) for i in range(4)]
    rows[0]["r_remove"] = [np.nan] * 8
    rows[0]["r_flip"] = [np.nan] * 8
    scored = module.score_pool(rows)
    assert np.isnan(scored[0]["R_corner"][0])


# ── 배분 ───────────────────────────────────────────────────────────────

def test_every_frame_gets_at_least_one_exposure(module):
    allocation = module.largest_remainder_allocation([0.9, 0.5, 0.01], 10)
    assert min(allocation) >= 1
    assert sum(allocation) == 10


def test_allocation_is_deterministic(module):
    weights = [0.9, 0.9, 0.5, 0.2, 0.05]
    first = module.largest_remainder_allocation(weights, 23)
    second = module.largest_remainder_allocation(weights, 23)
    assert first == second


def test_higher_reliability_gets_more_exposure(module):
    allocation = module.largest_remainder_allocation([0.9, 0.3], 20)
    assert allocation[0] > allocation[1]


def test_allocation_refuses_fewer_slots_than_frames(module):
    with pytest.raises(ValueError):
        module.largest_remainder_allocation([0.5, 0.5, 0.5], 2)


# ── GT 를 읽지 않는다 (AST) ────────────────────────────────────────────

FORBIDDEN_ARGUMENTS = ("gt", "ground_truth", "label", "gross", "error_px",
                       "annotation", "target", "paper_eval")


def test_no_public_function_takes_ground_truth(module):
    tree = ast.parse(MODULE.read_text())
    offending = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        for argument in node.args.args + node.args.kwonlyargs:
            if any(token in argument.arg.lower() for token in FORBIDDEN_ARGUMENTS):
                offending.append(f"{node.name}({argument.arg})")
    assert not offending, f"GT 성 인자가 있다: {offending}"


def test_the_module_never_names_the_evaluation_population(module):
    """docstring 은 제외한다 — 거기서는 "PAPER_EVAL 최적화값이 **아니다**" 처럼
    안전 문구로 등장한다.  안전 장치를 위반으로 세면 검사가 거꾸로 간다.
    검사 대상은 코드가 실제로 쓰는 문자열이다.
    """

    tree = ast.parse(MODULE.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            text = ast.get_docstring(node, clean=False)
            if text:
                docstrings.add(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in docstrings:
                continue
            lowered = node.value.lower()
            assert "paper_eval" not in lowered, node.value
            assert "pallet_eval" not in lowered, node.value


def test_the_safety_note_is_actually_present(module):
    """위 검사가 공허해지지 않게 — 안전 문구가 실제로 있는지도 본다."""

    text = MODULE.read_text()
    assert "PAPER_EVAL 최적화값이 아니" in text


def test_the_module_imports_nothing_that_reads_ground_truth(module):
    tree = ast.parse(MODULE.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= {"__future__", "numpy"}, f"예상 밖 import: {imported}"


# ── 가중치가 고정값이라는 것 ───────────────────────────────────────────

def test_all_signals_carry_equal_weight_by_construction(module):
    """0.4 / 0.3 같은 학습된 가중치가 끼어들지 않았는지."""

    source = MODULE.read_text()
    for token in ("coef_", "LogisticRegression", "RandomForest", "fit(",
                  "0.4 *", "0.3 *", "weights ="):
        assert token not in source, f"학습된 가중치의 흔적: {token}"
