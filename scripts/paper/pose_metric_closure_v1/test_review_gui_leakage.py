"""검수 GUI 가 사람 판정을 오염시킬 정보를 읽지 못하도록 강제한다.

    python3 -m pytest scripts/paper/pose_metric_closure_v1/test_review_gui_leakage.py -q

사람이 모델의 답이나 기존 GT 배정을 보면 그 순간 독립적인 측정이 아니게 된다.
문자열 grep 이 아니라 **AST** 로 본다 — 주석·docstring 에 단어가 있다고 실패하면 안 되고,
반대로 코드에서 실제로 참조하면 반드시 잡아야 하기 때문이다.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
GUI = HERE / "review_physical_axis.py"
MANIFEST_BUILDER = HERE / "build_axis_review_manifest.py"

# 코드에서 참조되면 안 되는 이름 / 속성 / 문자열 키
FORBIDDEN_NAMES = {
    "solve_hypothesis", "solvePnP", "solvePnPRefineLM", "projectPoints",
    "select_pnp_hypotheses", "pnp_selector", "pose_metrics",
}
FORBIDDEN_STRINGS = {
    "_hidden_stored_long_axis", "axis_assignment_confirmed", "axis_assignment",
    "pose_transform", "camera_facing_pnp", "dimensions_m",
    "score_margin", "short_score", "long_score", "top_score",
    "expected_hypothesis", "selected_hypothesis",
    "rotation_error_deg", "yaw_error_deg", "translation_error_m",
    "restricted_adds_error_m", "add_error_m", "adds_error_m",
    "R0", "R5_PROPOSED", "prediction", "predictions",
}


def gui_tree() -> ast.AST:
    return ast.parse(GUI.read_text())


def referenced_names(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.FunctionDef):
            found.add(node.name)
    return found


def literal_strings(tree: ast.AST) -> set[str]:
    """docstring 은 제외하고 코드가 실제로 쓰는 문자열만 모은다."""

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                found.add(node.value)
    return found


def test_review_gui_does_not_read_existing_axis_parity():
    """기존 GT 의 미확인 parity 는 manifest 에 있지만 GUI 가 읽으면 안 된다."""

    tree = gui_tree()
    strings = literal_strings(tree)
    leaked = {s for s in strings
              if any(bad in s for bad in ("_hidden_stored", "axis_assignment",
                                          "dimensions_m", "pose_transform",
                                          "camera_facing_pnp"))}
    assert not leaked, f"the GUI references stored GT axis information: {leaked}"


def test_review_gui_does_not_render_pose_residual():
    """PnP 잔차·적합 품질이 화면에 나오면 사람이 '더 잘 맞는 쪽' 을 고르게 된다."""

    tree = gui_tree()
    names = referenced_names(tree)
    strings = literal_strings(tree)
    assert "residual" not in names, "the GUI computes a fit residual"
    assert not any("residual" in s for s in strings), \
        "the GUI renders a fit residual"
    assert not (names & {"solvePnP", "solvePnPRefineLM", "projectPoints",
                         "solve_hypothesis"}), \
        "the GUI fits a pose; fit quality must not exist on this path"


def test_review_gui_does_not_load_model_predictions():
    tree = gui_tree()
    names = referenced_names(tree)
    strings = literal_strings(tree)
    assert not (names & {"select_pnp_hypotheses", "pnp_selector"}), \
        "the GUI imports the prediction selector"
    leaked = {s for s in strings
              if any(bad in s for bad in ("score_margin", "short_score", "long_score",
                                          "expected_hypothesis", "selected_hypothesis",
                                          "R5_PROPOSED"))}
    assert not leaked, f"the GUI references model prediction fields: {leaked}"


def test_review_gui_never_opens_a_result_or_prediction_file():
    """GUI 가 여는 파일은 manifest 와 자기 라벨 sidecar 뿐이어야 한다."""

    strings = literal_strings(gui_tree())
    suspicious = {s for s in strings
                  if any(bad in s for bad in ("paper_eval_v1", "paper_selftrain",
                                              "selector_diagnostic", "weights",
                                              "REVIEWED_POSE_GT", ".pt"))}
    assert not suspicious, f"the GUI reaches for result artifacts: {suspicious}"


def test_review_gui_does_not_reopen_the_annotation():
    """annotation 을 다시 열면 저장된 pose·parity 가 이 경로로 들어올 수 있다."""

    strings = literal_strings(gui_tree())
    assert not any("annotation" == s for s in strings), \
        "the GUI reads the annotation path; the manifest already carries what it needs"


def test_manifest_keeps_the_hidden_field_but_marks_it():
    """감사용 숨김 필드는 manifest 에는 있어야 한다 — 다만 정책이 명시돼야 한다."""

    strings = literal_strings(ast.parse(MANIFEST_BUILDER.read_text()))
    assert any("_hidden_stored_long_axis" in s for s in strings)
    assert any("hidden_field_policy" in s for s in strings), \
        "the manifest must state why the hidden field exists and that it is never shown"


def test_session_propagation_is_disabled():
    import importlib.util

    spec = importlib.util.spec_from_file_location("mb", MANIFEST_BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SESSION_PROPAGATION_ALLOWED is False, \
        "camera_dynamic_0123_v4 re-derives the axes per frame; propagation can mislabel"


def test_smoke_labels_are_a_separate_file():
    source = GUI.read_text()
    tree = ast.parse(source)
    assigned = {node.targets[0].id for node in ast.walk(tree)
                if isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)}
    assert "SMOKE_LABELS" in assigned
    assert "LABELS" in assigned
    strings = literal_strings(tree)
    assert "AXIS_REVIEW_LABELS_SMOKE.json" in strings
    assert "AXIS_REVIEW_LABELS.json" in strings


def test_pose_object_contract_is_separate_from_the_historical_registry():
    contract = json.loads(
        (REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
         / "POSE_EVAL_OBJECT_CONTRACT.json").read_text())
    assert contract["relationship_to_existing_registry"]["modified"] is False
    assert contract["relationship_to_existing_registry"][
        "historical_artifact_registry_sha_rewritten"] is False
    assert contract["population"]["wood_included_in_pose_table"] is True
    for name in ("plastic_standard_110x130x11", "wood_small_80x59x14"):
        assert contract[name]["orientation_equivalence_deg"] == [0, 180]
        assert 90 in contract[name]["distinct_orientations_deg"]
        assert 270 in contract[name]["distinct_orientations_deg"]


def test_pose_object_contract_dimensions_match_the_registry():
    """별도 계약이지만 치수는 정본과 일치해야 한다 — 다른 물체를 재는 셈이 되면 안 된다."""

    contract = json.loads(
        (REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
         / "POSE_EVAL_OBJECT_CONTRACT.json").read_text())
    registry = json.loads(
        (REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json").read_text())
    for entry in registry["objects"]:
        name = entry["object_type"]
        dims = entry["physical_dimensions_m"]
        footprint = sorted((dims["x"], dims["z"]))
        spec = contract[name]["physical_dimensions_m"]
        assert spec["short"] == pytest.approx(footprint[0], abs=1e-9), name
        assert spec["long"] == pytest.approx(footprint[1], abs=1e-9), name
        assert spec["height"] == pytest.approx(dims["y"], abs=1e-9), name


def test_contract_must_be_passed_explicitly():
    import sys

    sys.path.insert(0, str(HERE))
    from pose_evaluation_paths import load_pose_object_contract

    with pytest.raises(ValueError):
        load_pose_object_contract(None)


# --------------------------------------------------- label lifecycle (headless)
#
# 실제 클릭은 사람이 하지만, 클릭이 부르는 저장/재개 로직은 여기서 검증한다.
# 창을 띄우지 않고 같은 함수를 그대로 호출한다.


def _gui_module():
    import importlib.util
    import sys

    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("review_gui", GUI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(labels, frame_id, long_axis, status):
    labels["frames"][frame_id] = {
        "object_type": "plastic_standard_110x130x11",
        "session_id": "s",
        "long_axis": long_axis,
        "short_axis": (None if long_axis is None
                       else ("CF_DEPTH" if long_axis == "CF_WIDTH" else "CF_WIDTH")),
        "status": status,
        "reviewer": "human",
        "review_timestamp": "t",
        "source": "manual_visual_review",
        "propagated_by_session": False,
    }


def test_label_lifecycle_autosave_edit_delete_and_resume(tmp_path):
    gui = _gui_module()
    labels_path = tmp_path / "L.json"
    progress_path = tmp_path / "P.json"
    frames = [{"frame_id": f"f{i}"} for i in range(5)]

    labels = gui.load_labels(labels_path)
    assert labels["frames"] == {}

    # A / B / U 세 입력
    _record(labels, "f0", "CF_WIDTH", "CONFIRMED")
    gui.save(labels, 0, 5, labels_path=labels_path, progress_path=progress_path)
    _record(labels, "f1", "CF_DEPTH", "CONFIRMED")
    gui.save(labels, 1, 5, labels_path=labels_path, progress_path=progress_path)
    _record(labels, "f2", None, "UNCLEAR")
    gui.save(labels, 2, 5, labels_path=labels_path, progress_path=progress_path)

    # autosave: 파일이 즉시 반영됐는가
    on_disk = json.loads(labels_path.read_text())
    assert on_disk["frames"]["f0"]["long_axis"] == "CF_WIDTH"
    assert on_disk["frames"]["f1"]["long_axis"] == "CF_DEPTH"
    assert on_disk["frames"]["f2"]["status"] == "UNCLEAR"
    progress = json.loads(progress_path.read_text())
    assert progress["reviewed"] == 3 and progress["confirmed"] == 2
    assert progress["unclear"] == 1

    # 수정: 같은 프레임을 다시 누르면 덮어써진다
    _record(labels, "f0", "CF_DEPTH", "CONFIRMED")
    gui.save(labels, 0, 5, labels_path=labels_path, progress_path=progress_path)
    assert json.loads(labels_path.read_text())["frames"]["f0"]["long_axis"] == "CF_DEPTH"

    # 삭제: backspace 경로
    labels["frames"].pop("f1")
    gui.save(labels, 1, 5, labels_path=labels_path, progress_path=progress_path)
    assert "f1" not in json.loads(labels_path.read_text())["frames"]

    # 재개: 첫 미검수 프레임을 찾는 규칙
    reloaded = gui.load_labels(labels_path)
    resume = next((i for i, f in enumerate(frames)
                   if f["frame_id"] not in reloaded["frames"]), 0)
    assert resume == 1, "resume must land on the frame whose label was deleted"


def test_smoke_labels_never_touch_the_real_review(tmp_path):
    gui = _gui_module()
    real, smoke = tmp_path / "real.json", tmp_path / "smoke.json"
    real_labels = gui.load_labels(real)
    _record(real_labels, "f0", "CF_WIDTH", "CONFIRMED")
    gui.save(real_labels, 0, 5, labels_path=real, progress_path=tmp_path / "rp.json")

    smoke_labels = gui.load_labels(smoke)
    _record(smoke_labels, "f0", "CF_DEPTH", "CONFIRMED")
    gui.save(smoke_labels, 0, 5, labels_path=smoke, progress_path=tmp_path / "sp.json")

    assert json.loads(real.read_text())["frames"]["f0"]["long_axis"] == "CF_WIDTH"
    assert json.loads(smoke.read_text())["frames"]["f0"]["long_axis"] == "CF_DEPTH"
    assert gui.SMOKE_LABELS != gui.LABELS
