"""The canonical evaluation set is the frames the user marked eval, nothing else.

Written after 2026-08-04, when every PAPER_S2 screen turned out to have been
judged on data/_eval_sets/* -- a 05-27 combination that predates the eval/train
toggle -- instead of the frames actually marked eval in the annotation tool.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from challenge import data_paths  # noqa: E402  (sys.path 조작 뒤여야 한다)

DOC = ROOT / "_docs/EVAL_SET_CANONICAL.md"

# 폴더 경로는 data_paths 가 유일한 출처다. 여기서 다시 문자열로 쓰면 폴더를 옮길
# 때 두 곳이 어긋난다(2026-08-14 재편에서 실제로 겪은 문제).
# 프레임 수만 이 테스트가 따로 들고 있다 — 경로가 맞아도 내용이 바뀌면 잡아야 하므로.
_EXPECTED_COUNTS = {
    "eval_outside": 22,
    "eval_noapril": 12,
    "eval_cad": 22,
    # 2026-08-07: metric_split_lock.md §1.6 의 outside final-test 세션.
    # 봉인 해제하여 정본에 편입(1회성 — data_paths.FINAL_TEST 참조).
    "eval_pallet07": 27,
    "eval_pallet09": 36,
    # 2026-08-08: lock §1.6 의 night final-test 세션. 봉인 해제하여 정본 편입.
    "eval_night08": 17,
    "eval_night09": 25,
}
EVAL_FOLDERS = {data_paths.EVAL_CANONICAL[key]: count
                for key, count in _EXPECTED_COUNTS.items()}
EXPECTED_TOTAL = data_paths.EVAL_CANONICAL_TOTAL

# lock §1.6 이 final-test 로 지정한 세션. PL 풀·threshold 캘리브에 들어가면 안 된다.
FINAL_TEST_FOLDERS = tuple(data_paths.EVAL_CANONICAL[key]
                           for key in data_paths.FINAL_TEST)
# lock §1.6 이 filter-val 로 지정한 세션에서 온 eval 프레임(=threshold 캘리브에 쓰인 세션).
# 정본에 남아 있으나 final-test 로 보고하면 안 된다.
FILTER_VAL_SESSIONS = data_paths.FILTER_VAL_SESSIONS
FORBIDDEN_EVAL_SOURCES = data_paths.FORBIDDEN_EVAL_SOURCES


def _split_of(path: pathlib.Path) -> str:
    payload = json.loads(path.read_text("utf-8"))
    objects = payload.get("objects") or [{}]
    return objects[0].get("split", "(none)")


def collect_eval_frames() -> dict[str, list[pathlib.Path]]:
    found: dict[str, list[pathlib.Path]] = {}
    for folder in EVAL_FOLDERS:
        directory = ROOT / folder
        if not directory.is_dir():
            continue
        found[folder] = sorted(
            path for path in directory.rglob("*.json") if _split_of(path) == "eval")
    return found


def test_split_lives_under_objects_not_at_the_top_level() -> None:
    """The mistake that started this: a top-level read reports 'no split'."""
    found = collect_eval_frames()
    if not found:
        pytest.skip("annotation folders not present")
    sample = next(iter(next(iter(found.values()))), None)
    if sample is None:
        pytest.skip("no eval frame")
    payload = json.loads(sample.read_text("utf-8"))
    assert "split" not in payload, "top-level split would be a different convention"
    assert payload["objects"][0]["split"] == "eval"


def test_eval_frame_count_matches_the_canonical_document() -> None:
    found = collect_eval_frames()
    if len(found) != len(EVAL_FOLDERS):
        pytest.skip("annotation folders not all present")
    for folder, expected in EVAL_FOLDERS.items():
        assert len(found[folder]) == expected, (folder, len(found[folder]), expected)
    assert sum(len(v) for v in found.values()) == EXPECTED_TOTAL


def test_every_eval_frame_has_an_image_and_a_cuboid() -> None:
    found = collect_eval_frames()
    if not found:
        pytest.skip("annotation folders not present")
    for paths in found.values():
        for path in paths:
            assert path.with_suffix(".png").is_file(), path
            objects = json.loads(path.read_text("utf-8"))["objects"][0]
            assert objects.get("projected_cuboid"), path
            assert objects.get("gt_source") == "manual", path


def test_the_canonical_document_exists_and_names_the_folders() -> None:
    assert DOC.is_file(), "the canonical eval-set document must exist"
    text = DOC.read_text("utf-8")
    for folder in EVAL_FOLDERS:
        assert folder in text
    assert "objects[0].split" in text
    for forbidden in FORBIDDEN_EVAL_SOURCES:
        assert forbidden in text, "the superseded source must be named as forbidden"


def test_new_eval_manifests_must_not_be_built_from_the_superseded_combination() -> None:
    """A manifest may only cite _eval_sets if it also records that it is stale."""
    results = ROOT / "data/pallet/results"
    if not results.is_dir():
        pytest.skip("no results tree")
    offenders = []
    for manifest in results.rglob("*manifest*.json"):
        try:
            text = manifest.read_text("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(source in text for source in FORBIDDEN_EVAL_SOURCES):
            # the historical PAPER_S2 manifest is grandfathered but must be
            # flagged in the canonical document, which the test above checks
            if manifest.name != "mechanism_val_manifest.json":
                offenders.append(str(manifest.relative_to(ROOT)))
    assert not offenders, (
        "new manifests must use objects[0].split == 'eval', see "
        f"_docs/EVAL_SET_CANONICAL.md: {offenders}")


def test_final_test_sessions_are_not_in_any_pseudo_label_pool() -> None:
    """lock §1.6: final-test 세션 프레임은 PL 풀에 있으면 안 된다 (transductive 차단)."""
    pools = sorted((ROOT / "data/pallet").glob("real_unlabeled_ralph*"))
    if not pools:
        pytest.skip("no pseudo-label pool present")
    leaked = []
    for folder in FINAL_TEST_FOLDERS:
        directory = ROOT / folder
        if not directory.is_dir():
            continue
        fids = {path.stem for path in directory.glob("*.json")}
        for pool in pools:
            for link in pool.glob("*.png"):
                # 풀 파일명은 "{session}__{fid}.png"
                fid = link.stem.split("__", 1)[-1]
                if fid in fids:
                    leaked.append(f"{pool.name}/{link.name}")
    assert not leaked, f"final-test frames leaked into the pseudo-label pool: {leaked[:10]}"


def test_train_marked_frames_are_never_counted_as_eval() -> None:
    found = collect_eval_frames()
    if not found:
        pytest.skip("annotation folders not present")
    for folder in EVAL_FOLDERS:
        directory = ROOT / folder
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.json"):
            if _split_of(path) == "train":
                assert path not in found[folder]
