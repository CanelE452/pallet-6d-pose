"""refinement 생성기가 정답을 못 보게 강제한다.  AST 로 검사한다.

    python3 -m pytest scripts/self_training_yolo/temporal_refine/ -q

문자열 검색이 아니라 AST 를 쓴다 — 주석이나 문서 문자열에 든 단어에 걸려
헛경보를 내지 않기 위해서다(이 저장소에서 반복된 실수).
"""

from __future__ import annotations

import ast
from pathlib import Path

REFINER = Path(__file__).resolve().parent / "refine_temporal_keypoints.py"
TREE = ast.parse(REFINER.read_text())


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {REFINER.name}")


def test_refine_center_takes_no_ground_truth_argument():
    node = _function("refine_center")
    names = [a.arg for a in node.args.args]
    forbidden = ("gt", "ground_truth", "manual", "annotation", "truth",
                 "label", "reference", "target")
    for name in names:
        assert not any(token in name.lower() for token in forbidden), \
            f"refine_center must not take a ground-truth argument, found {name!r}"


def test_refiner_opens_no_annotation_file():
    """어노테이션 JSON 을 여는 호출이 없어야 한다."""

    opened_literals = []
    for node in ast.walk(TREE):
        if isinstance(node, ast.Call):
            target = node.func
            name = getattr(target, "attr", getattr(target, "id", ""))
            if name in {"open", "read_text", "load", "loads", "imread", "loadtxt"}:
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        opened_literals.append(argument.value)
    forbidden = ("annotation", "manual_gt", "projected_cuboid", "GEOMETRY_RESOLVED",
                 "PAPER_EVAL", "eval_workspace")
    for literal in opened_literals:
        assert not any(token.lower() in literal.lower() for token in forbidden), \
            f"the refiner must not open {literal!r}"


def test_refiner_does_not_import_the_evaluation_workspace():
    for node in ast.walk(TREE):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert "eval_workspace" not in name, \
                "the refiner must not import the evaluation workspace"


def test_refiner_reads_no_depth():
    source = REFINER.read_text().lower()
    for token in ("/depth/", "depth_corrected", "paper_depth_selftrain"):
        assert token not in source, f"the refiner must not read depth ({token})"
