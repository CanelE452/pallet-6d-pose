"""challenge/scripts/annotate/annotate.py — main entry.

Manual annotation 도구. 시퀀스 frame 에 9 keypoint 클릭으로 라벨링 + PnP 자동 풀이 +
NDDS JSON GT 저장.

분리된 모듈:
  annotate_pnp.py    PnP / 3D model / projection / MANIPULATE / TWO-LINE
  annotate_draw.py   cuboid wireframe / overlay / UI panel / render
  annotate_io.py     State / make_annotation / save / load

Keypoint 순서 (**camera-facing convention, 2026-05-22 결정**):
  0: NearTopLeft      1: NearTopRight       ★ 카메라에 보이는 가까운 면 (near = fork pocket)
  2: NearBottomRight  3: NearBottomLeft     ★
  4: FarTopLeft       5: FarTopRight        (반대편 far face, 위쪽 corner)
  6: FarBottomRight   7: FarBottomLeft
  8: Centroid         (c 키로 자동)

→ 사용자는 "보이는 가까운 면" 에 0~3 을, 가능하면 보이는 far-top (4, 5) 도 클릭.
   far-bottom (6, 7) 은 가려질 경우가 많아 g 키로 자동 채움 가능 (0~5 → 6/7/8 PnP projection).

키:
  좌클릭        활성 keypoint 위치 → 다음 idx 자동
  0~8           특정 idx 활성 (수정)
  c             centroid 자동
  z             마지막 점 undo
  d             활성 idx 삭제
  r             전체 reset
  s             저장 + 다음 frame
  v             이 frame eval/train 토글 (평가용 표시, JSON split 필드에 저장)
  b             active keypoint visibility/reason 순환 (unknown/visible/occluded/truncated)
  /             CONDITIONS 모드 진입 (/ 또는 Esc 로 CLICK 복귀)
    1/2         frame occlusion / truncation 평가 tag ON/OFF
    3/4/5       elevation LOW / MID / HIGH 직접 선택
    6           distance FAR 직접 선택
    n/m         distance NEAR / MID 직접 선택
    u           distance를 UNKNOWN으로 되돌림
    a, a        현재 변경 tag를 세션의 annotated frame에 일괄 적용
  w             W/D parity 전환 (short-face-front ↔ long-face-front)
  ;             frame 번호 입력 점프 (숫자 후 Enter, Esc 취소) — 상단 슬라이더 클릭/드래그도 가능
  ANNOT-ONLY 버튼 (우측 패널 하단 클릭) = ON 이면 n/p 가 어노된 frame 만 이동 (어노된 것만 보기)
  f             near-only 자동 저장 (0~3 만 클릭, 4~7 자동 PnP projection 채움)
  g             auto-fill 저장 (4+ 점 클릭, 미클릭 idx 자동 PnP projection 채움) ★ truncation/occlusion
  x             parallelogram 외삽 (active idx ← 같은 face 의 나머지 3 corner) ★ truncation
  m             CLICK ↔ MANIPULATE 모드 토글
  t             TWO-LINE input 토글
  n / p         다음 / 이전 frame
  , / .         -10 / +10 frame jump
  + / -         zoom in/out
  h j k l       pan (vim)
  q             종료 (MANIPULATE에서는 m으로 CLICK 복귀 후 q)

사용:
  python scripts/annotate/annotate.py --seq data/outside/capturepallet07 \
      --stride 15 --population-role DEV
  python scripts/annotate/annotate.py --seq data/outside/capturepallet09 \
      --population-role DEV \
      --out_dir challenge/data/01_real/gt_v2_canonical/manual_gt/pallet09_manual_gt
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import argparse
import copy
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Windows DPI scaling 보정 — 모니터가 100% 가 아니면 cv2 윈도우 좌표 ↔ 마우스 좌표
# mismatch 로 클릭 위치보다 점이 어긋남. import 직후 한 번만 호출.
if os.name == "nt":
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)   # Per-monitor v2 (Win10+)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()        # legacy fallback
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))


def find_repo_root(start):
    """Find the repository by its .git marker, including worktree .git files."""
    current = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError(f"cannot find repository .git marker above {start}")
        current = parent


_REPO = find_repo_root(_HERE)
sys.path.insert(0, _HERE)   # annotate_pnp / annotate_draw / annotate_io import

from annotate_pnp import (
    solve_pose, pose_from_locked, apply_manip, line_intersection,
    parallelogram_extrapolate,
    default_physical_dimensions,
)
from annotate_draw import (
    MARGIN_B,
    MARGIN_L,
    MARGIN_R,
    MARGIN_T,
    annotation_overlay_path,
    annot_button_rect,
    render,
    render_saved_annotation_overlay,
    session_button_rect,
)
from annotate_io import (
    State, make_annotation, save_frame_json, load_existing_annotation,
    _truncation_payload,
)
from object_geometry_registry import (
    DEFAULT_REGISTRY_PATH,
    PLASTIC_OBJECT_TYPE,
    WOOD_OBJECT_TYPE,
    load_object_geometry_registry,
)
from evaluation.eval_workspace import (
    FRAME_TAG_FIELDS,
    WorkspaceError,
    canonical_frame_tag_identity,
    load_frame_tag_overrides,
    load_session_metadata,
    resolve_effective_frame_tags,
    update_frame_tags_csv,
    update_frame_tags_csv_many,
)


# ─── Session pool ────────────────────────────────────────────────────────────

_ANNOTATION_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
_LEGACY_DEFAULT_K = np.array(
    [[614.18, 0.0, 329.28], [0.0, 614.31, 234.53], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def _session_image_paths(seq):
    """Return supported RGB frames without guessing anything from names."""
    rgb_dir = os.path.join(seq, "rgb")
    if not os.path.isdir(rgb_dir):
        return []
    return sorted(
        path for path in glob.glob(os.path.join(rgb_dir, "*"))
        if os.path.isfile(path)
        and os.path.splitext(path)[1].lower() in _ANNOTATION_IMAGE_SUFFIXES
    )

def discover_sessions(pools, repo):
    """pool 폴더들 아래에서 rgb/ 를 가진 촬영 세션을 모은다.

    - rgb 가 0장인 폴더는 뺀다. 목록에 뜨는데 열면 아무것도 없어 혼란만 준다.
    - 같은 어노테이션 폴더로 해석되는 세션이 둘 이상이면 하나만 남긴다. 그대로 두면
      한쪽에서 찍은 라벨을 다른 쪽에서 열어 덮어쓴다(같은 영상을 두 번 추출해 폴더가
      둘이 된 실제 사례가 있다). 이미 어노가 있는 쪽을, 없으면 이름이 긴 쪽을 남긴다.

    반환: [(name, seq_path), ...]  이름 순.
    """
    found = []
    for pool in pools:
        root = pool if os.path.isabs(pool) else os.path.join(repo, pool)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            seq = os.path.join(root, name)
            if not os.path.isdir(os.path.join(seq, "rgb")):
                continue
            if not _session_image_paths(seq):
                continue
            found.append((name, seq))

    by_out = {}
    for name, seq in found:
        od, _ = resolve_out_dir(name, repo)
        prev = by_out.get(od)
        if prev is None:
            by_out[od] = (name, seq)
            continue
        n_prev = len(_session_image_paths(prev[1]))
        n_cur = len(_session_image_paths(seq))
        keep = prev if (n_prev, len(prev[0])) >= (n_cur, len(name)) else (name, seq)
        drop = (name, seq) if keep is prev else prev
        print(f"[세션] '{drop[0]}' 은 '{keep[0]}' 과 같은 저장 폴더를 쓴다 — 목록에서 제외")
        by_out[od] = keep
    return sorted(by_out.values(), key=lambda t: t[0])


def _session_entry_parts(entry):
    """Return ``(label, source_session_dir, context_key)`` for a chooser row.

    Historical callers use ``(name, path)`` rows whose context key is the
    real session path.  Object-specific incoming views share one immutable RGB
    directory, so they add a third, unique key without changing the old API.
    """
    if len(entry) == 2:
        name, source_session_dir = entry
        return name, source_session_dir, os.path.realpath(source_session_dir)
    if len(entry) == 3:
        name, source_session_dir, context_key = entry
        return name, source_session_dir, context_key
    raise ValueError(f"invalid session chooser row: {entry!r}")


_SESSION_RUNTIME_ARG_FIELDS = (
    "object_type",
    "population_role",
    "lighting_condition",
    "intrinsics_quality",
    "intrinsics_source",
    "capture_session_id",
)

_INCOMING_FRAME_REVIEW_FIELDS = (
    "frame",
    "source_ordinal",
    "review_label",
    "exclude_reason",
)
_INCOMING_FRAME_REVIEW_LABELS = frozenset({"plastic", "wood", "exclude"})


def _incoming_reviewed_frame_partitions(
        metadata, session_dir, frame_paths, geometry_registry):
    """Load one exhaustive pixel-review manifest and return accepted views.

    ``frame_review_manifest`` in ``session.json`` is a path relative to the
    immutable incoming session.  The CSV must account for every raw RGB
    filename exactly once.  Its 1-based source ordinal is also checked against
    the sorted raw sequence, so a stale or accidentally reordered review can
    never silently select a different frame.  ``exclude`` rows remain recorded
    in the review ledger but are hidden from both PnP annotation views.
    """
    manifest_value = metadata.get("frame_review_manifest")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ValueError(
            "incoming frame_review_manifest is required before PnP annotation")
    if manifest_value != manifest_value.strip():
        raise ValueError("incoming frame_review_manifest must not contain whitespace")

    session_root = Path(session_dir).resolve()
    manifest_path = (session_root / manifest_value).resolve()
    try:
        manifest_path.relative_to(session_root)
    except ValueError as exc:
        raise ValueError(
            "incoming frame_review_manifest must stay inside the session directory"
        ) from exc
    if not manifest_path.is_file():
        raise ValueError(
            f"incoming frame_review_manifest does not exist: {manifest_value}")

    frame_count = len(frame_paths)
    declared_count = metadata.get("image_count")
    if (isinstance(declared_count, bool)
            or not isinstance(declared_count, int)):
        raise ValueError("incoming image_count must be an integer")
    if declared_count != frame_count:
        raise ValueError(
            "incoming image_count does not match rgb/: "
            f"declared={declared_count}, actual={frame_count}")

    raw_by_name = {}
    expected_ordinal_by_name = {}
    for source_ordinal, frame_path in enumerate(frame_paths, start=1):
        frame_name = Path(frame_path).name
        if frame_name in raw_by_name:
            raise ValueError(f"duplicate raw RGB filename: {frame_name}")
        raw_by_name[frame_name] = frame_path
        expected_ordinal_by_name[frame_name] = source_ordinal

    records_by_name = {}
    reason_counts = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _INCOMING_FRAME_REVIEW_FIELDS:
            raise ValueError(
                "incoming frame review manifest header must be exactly: "
                + ",".join(_INCOMING_FRAME_REVIEW_FIELDS))
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row[field] is None for field in reader.fieldnames):
                raise ValueError(
                    "review manifest row has the wrong number of columns at "
                    f"line {line_number}")
            frame_name = row["frame"]
            if not frame_name or frame_name != Path(frame_name).name:
                raise ValueError(
                    f"invalid frame filename at review manifest line {line_number}")
            if frame_name in records_by_name:
                raise ValueError(
                    f"duplicate frame in review manifest: {frame_name}")
            if frame_name not in raw_by_name:
                raise ValueError(
                    f"review manifest frame is not in raw rgb/: {frame_name}")

            ordinal_text = row["source_ordinal"]
            try:
                source_ordinal = int(ordinal_text)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "review manifest source_ordinal must be an integer at "
                    f"line {line_number}") from exc
            if ordinal_text != str(source_ordinal):
                raise ValueError(
                    "review manifest source_ordinal must use canonical integer "
                    f"text at line {line_number}")
            expected_ordinal = expected_ordinal_by_name[frame_name]
            if source_ordinal != expected_ordinal:
                raise ValueError(
                    "review manifest source_ordinal does not match sorted rgb/: "
                    f"frame={frame_name}, declared={source_ordinal}, "
                    f"expected={expected_ordinal}")

            review_label = row["review_label"]
            if review_label not in _INCOMING_FRAME_REVIEW_LABELS:
                raise ValueError(
                    f"invalid review_label at line {line_number}: {review_label!r}")
            exclude_reason = row["exclude_reason"]
            if review_label == "exclude":
                if not exclude_reason.strip():
                    raise ValueError(
                        "exclude row requires exclude_reason at review manifest "
                        f"line {line_number}")
                reason_counts[exclude_reason] = reason_counts.get(exclude_reason, 0) + 1
            elif exclude_reason:
                raise ValueError(
                    "accepted review row must have an empty exclude_reason at "
                    f"line {line_number}")

            records_by_name[frame_name] = {
                "frame": frame_name,
                "source_ordinal": source_ordinal,
                "review_label": review_label,
                "exclude_reason": exclude_reason,
            }

    missing = [name for name in raw_by_name if name not in records_by_name]
    if missing:
        preview = ",".join(missing[:5])
        raise ValueError(
            "incoming frame review manifest does not cover every raw RGB "
            f"filename; missing={preview}")

    canonical_types = {
        "plastic": geometry_registry.resolve(PLASTIC_OBJECT_TYPE).object_type,
        "wood": geometry_registry.resolve(WOOD_OBJECT_TYPE).object_type,
    }
    partitions = {object_type: [] for object_type in canonical_types.values()}
    label_counts = {label: 0 for label in sorted(_INCOMING_FRAME_REVIEW_LABELS)}
    # Always preserve the raw sequence order, independent of CSV row order.
    for frame_path in frame_paths:
        record = records_by_name[Path(frame_path).name]
        label = record["review_label"]
        label_counts[label] += 1
        if label != "exclude":
            partitions[canonical_types[label]].append(frame_path)

    if any(not paths for paths in partitions.values()):
        raise ValueError(
            "incoming frame review manifest must contain accepted plastic and wood")
    review_info = {
        "manifest_path": str(manifest_path),
        "manifest_relative_path": manifest_value,
        "label_counts": label_counts,
        "exclude_reason_counts": reason_counts,
    }
    return partitions, review_info


def _validate_incoming_staging_membership(output_dir, frame_paths):
    """Reject labels/tags whose source frame is outside this object view."""
    output = Path(output_dir)
    if not output.exists():
        return
    allowed_stems = {Path(path).stem for path in frame_paths}
    wrong_json = sorted(
        path.name for path in output.glob("*.json")
        if path.is_file() and path.stem not in allowed_stems)
    tag_rows = load_frame_tag_overrides(output)
    wrong_tags = sorted(set(tag_rows) - allowed_stems)
    if wrong_json or wrong_tags:
        details = []
        if wrong_json:
            details.append("JSON=" + ",".join(wrong_json[:5]))
        if wrong_tags:
            details.append("frame_tags=" + ",".join(wrong_tags[:5]))
        raise ValueError(
            "incoming staging output contains frames outside its object "
            "review view: " + "; ".join(details))


def _annotation_intrinsics_consensus(annotation_dir):
    """Return one K shared by canonical JSON files, or fail on disagreement."""
    matrices = []
    for path in sorted(glob.glob(os.path.join(annotation_dir, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                document = json.load(handle)
            intrinsics = document["camera_data"]["intrinsics"]
            matrix = np.array([
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            continue
        if not np.isfinite(matrix).all():
            continue
        matrices.append((path, matrix))
    if not matrices:
        return None
    reference_path, reference = matrices[0]
    for path, matrix in matrices[1:]:
        if not np.allclose(reference, matrix, rtol=0.0, atol=1e-6):
            raise ValueError(
                "evaluation annotation intrinsics disagree within session\n"
                f"first = {reference_path}\nother = {path}")
    return reference


def _resolve_session_intrinsics(session_dir, annotation_dir, evaluation=False):
    """Resolve K per session; evaluation never silently guesses a camera."""
    camera_path = os.path.join(session_dir, "cam_K.txt")
    camera_matrix = None
    if os.path.isfile(camera_path):
        try:
            camera_matrix = np.loadtxt(camera_path).reshape(3, 3)
        except Exception as exc:
            raise ValueError(f"invalid cam_K.txt: {camera_path}: {exc}") from exc
        if not np.isfinite(camera_matrix).all():
            raise ValueError(f"non-finite cam_K.txt: {camera_path}")

    annotation_matrix = (
        _annotation_intrinsics_consensus(annotation_dir) if evaluation else None)
    if camera_matrix is not None and annotation_matrix is not None:
        if not np.allclose(camera_matrix, annotation_matrix, rtol=0.0, atol=1e-6):
            raise ValueError(
                "cam_K.txt conflicts with canonical annotation intrinsics\n"
                f"session = {session_dir}\nannotations = {annotation_dir}")
        return camera_matrix, "cam_K.txt + annotation consensus"
    if camera_matrix is not None:
        return camera_matrix, "cam_K.txt"
    if annotation_matrix is not None:
        return annotation_matrix, "canonical annotation consensus"
    if evaluation:
        raise ValueError(
            "evaluation session has no trustworthy intrinsics; add cam_K.txt "
            f"before annotation: {session_dir}")
    return _LEGACY_DEFAULT_K.copy(), "legacy default"


# 촬영 폴더 이름과 어노테이션 폴더 이름이 다른 경우. 이름이 어긋나면 기존 어노가 있는데도
# 빈 폴더를 새로 만들어 "어노가 사라진" 것처럼 보인다(2026-08-15 실제로 겪음).
_OUT_ALIAS = {
    "forklift_raw_20260528": "forklift_20260528_manual_gt",
    "forklift_raw_20260528_163408": "forklift_20260528_manual_gt",
    "capturepallet11": "pallet11_gt",          # 243장이 이미 여기 있다
}

# Import compatibility for the historical audit harness.  Deliberately empty:
# session names never define DEV/FINAL membership.
_SEALED_SESSIONS = frozenset()

def _resolve_legacy_read_dir(seq_name, repo):
    """Locate an old annotation directory for read-only compatibility."""
    names = [f"{seq_name}_manual_gt"]
    if seq_name in _OUT_ALIAS:
        names.insert(0, _OUT_ALIAS[seq_name])
    for nm in names:
        for sub, eval_layout in (
                ("01_real/manual_gt", False),
                ("01_real/eval_canonical", True)):
            path = os.path.join(repo, "challenge", "data", sub, nm)
            if os.path.isdir(path):
                return path, eval_layout
    return None, False


def resolve_out_dir(seq_name, repo):
    """세션 이름 -> 비파괴 GT-v2 어노테이션 저장 폴더.

    legacy ``manual_gt`` / ``eval_canonical`` 은 이름과 배치만 조회한다. 반환 경로는
    언제나 ``01_real/gt_v2_canonical`` 아래다. 따라서 기존 라벨을 열어 새 스키마로
    저장해도 원본 JSON을 같은 위치에서 덮어쓸 수 없다.

    반환의 두 번째 값은 legacy ``eval_canonical`` 디렉터리 배치 여부뿐이다.
    DEV/FINAL 역할을 뜻하지 않는다. population 역할은 CLI로만 명시한다.
    """
    names = [f"{seq_name}_manual_gt"]
    if seq_name in _OUT_ALIAS:
        names.insert(0, _OUT_ALIAS[seq_name])
    legacy_dir, eval_layout = _resolve_legacy_read_dir(seq_name, repo)
    if legacy_dir is not None:
        layout = "eval_canonical" if eval_layout else "manual_gt"
        return os.path.join(
            repo, "challenge", "data", "01_real", "gt_v2_canonical",
            layout, os.path.basename(legacy_dir)), eval_layout
    return os.path.join(
        repo, "challenge", "data", "01_real", "gt_v2_canonical",
        "manual_gt", names[0]), False


def _path_is_within(path, root):
    """True for ``root`` itself or a descendant, resolving symlinks first."""
    try:
        return os.path.commonpath(
            [os.path.realpath(path), os.path.realpath(root)]) == os.path.realpath(root)
    except ValueError:  # Different Windows drives.
        return False


def _require_nonlegacy_output_dir(path, repo):
    """Fail closed if an annotation output could mutate source/audited data."""
    protected_roots = (
        ("legacy real GT", os.path.join(
            repo, "challenge", "data", "01_real", "manual_gt")),
        ("legacy real GT", os.path.join(
            repo, "challenge", "data", "01_real", "eval_canonical")),
        ("audited real GT-v2", os.path.join(
            repo, "challenge", "real_gt_v2")),
        ("raw pallet data", os.path.join(repo, "data", "pallet", "raw_data")),
        ("real capture source", os.path.join(
            repo, "challenge", "data", "01_real", "_live_captures")),
        ("augmented real source", os.path.join(
            repo, "challenge", "data", "01_real", "augmented")),
        ("pseudo GT source", os.path.join(
            repo, "challenge", "data", "01_real", "pseudo_gt")),
    )
    for label, root in protected_roots:
        if _path_is_within(path, root):
            raise ValueError(
                f"{label} is read-only; choose a dedicated annotation output: {path}")
    return os.path.abspath(path)


def _direct_child_name(path, parent):
    """Return a direct child's basename, resolving symlinks, else ``None``."""
    path = os.path.realpath(path)
    parent = os.path.realpath(parent)
    try:
        relative = os.path.relpath(path, parent)
    except ValueError:  # Different Windows drives.
        return None
    parts = [part for part in relative.replace("\\", "/").split("/")
             if part not in {"", "."}]
    if len(parts) != 1 or parts[0] == "..":
        return None
    return parts[0]


def _validate_evaluation_paths(eval_root, seq, out_dir, population_role):
    """Enforce role-compatible session/annotation namespaces.

    The editor reads exactly one ``sessions/<session>/rgb`` tree and writes to
    the matching ``annotations/<session>`` tree.  This prevents an accidental
    ``--out_dir .../rgb`` from making delete remove a workspace source image.
    """
    role = str(population_role).upper()
    layouts = {
        "DEV": (
            ("dev_existing/sessions", "dev_existing/annotations"),
            ("legacy_unverified/sessions", "legacy_unverified/annotations"),
        ),
        "FINAL": (
            ("final/positive/sessions", "final/positive/annotations"),
        ),
    }
    for sessions_relative, annotations_relative in layouts.get(role, ()):
        sessions_root = os.path.join(eval_root, *sessions_relative.split("/"))
        annotations_root = os.path.join(
            eval_root, *annotations_relative.split("/"))
        session_name = _direct_child_name(seq, sessions_root)
        if session_name is None:
            continue
        if not os.path.isdir(os.path.join(seq, "rgb")):
            raise ValueError(
                f"evaluation session must contain rgb/: {seq}")
        output_session = _direct_child_name(out_dir, annotations_root)
        if output_session != session_name:
            raise ValueError(
                "evaluation --out_dir must be the matching canonical namespace: "
                f"{os.path.join(annotations_root, session_name)}")
        return os.path.abspath(seq), os.path.abspath(out_dir)
    allowed = ", ".join(item[0] for item in layouts.get(role, ())) or "none"
    raise ValueError(
        f"{role} --seq must be one direct session under --eval-root/{allowed}: {seq}")


def _require_legacy_read_dir(path, repo):
    """Allow an explicit compatibility source only inside a legacy GT tree."""
    legacy_roots = (
        os.path.join(repo, "challenge", "data", "01_real", "manual_gt"),
        os.path.join(repo, "challenge", "data", "01_real", "eval_canonical"),
    )
    candidate = path if os.path.isabs(path) else os.path.join(repo, path)
    if not any(_path_is_within(candidate, root) for root in legacy_roots):
        raise ValueError(
            "--legacy-read-dir must be inside challenge/data/01_real/"
            "{manual_gt,eval_canonical}")
    if not os.path.isdir(candidate):
        raise ValueError(f"--legacy-read-dir does not exist: {candidate}")
    return os.path.abspath(candidate)


def _state_geometry_spec(s):
    """Return the immutable selected object, with plastic as API fallback."""
    spec = getattr(s, "geometry_spec", None)
    if spec is not None:
        return spec
    return None


def _state_physical_dimensions(s):
    spec = _state_geometry_spec(s)
    return (spec.physical_dimensions if spec is not None
            else default_physical_dimensions())


def _state_default_wdh(s):
    spec = _state_geometry_spec(s)
    if spec is not None:
        return tuple(float(value) for value in spec.legacy_wdh_tuple)
    physical = default_physical_dimensions()
    return (float(physical.x_m), float(physical.z_m), float(physical.y_m))


def _metadata_value(value):
    """Treat blank/unknown session values as absent, never as inferred data."""
    if value is None:
        return None
    text = str(value).strip()
    return None if not text or text.lower() == "unknown" else text


def _explicit_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _configuration_mismatch(label, cli_value, session_value):
    raise ValueError(
        f"{label} mismatch\nCLI       = {cli_value}\nsession   = {session_value}")


def _resolve_annotation_configuration(args, seq, geometry_registry, eval_root):
    """Resolve CLI/session metadata without silently choosing on conflicts.

    Evaluation sessions are explicit metadata containers.  No value here is
    derived from an image, filename, pose, or directory name.
    """
    metadata = {}
    if eval_root:
        metadata = load_session_metadata(Path(seq))
        session_id = _metadata_value(metadata.get("session_id"))
        directory_id = os.path.basename(os.path.normpath(seq))
        if session_id and session_id != directory_id:
            raise ValueError(
                "session id mismatch\n"
                f"directory = {directory_id}\n"
                f"session   = {session_id}")

    cli_object = _metadata_value(args.object_type)
    session_object = _metadata_value(metadata.get("object_type"))
    if cli_object and session_object:
        try:
            cli_spec = geometry_registry.resolve(cli_object)
            session_spec = geometry_registry.resolve(session_object)
        except ValueError as exc:
            raise ValueError(f"invalid object type in CLI/session metadata: {exc}") from exc
        if cli_spec.object_type != session_spec.object_type:
            _configuration_mismatch(
                "object type", cli_spec.object_type, session_spec.object_type)
    selected_object = cli_object or session_object or PLASTIC_OBJECT_TYPE
    geometry_spec = geometry_registry.resolve(selected_object)

    cli_role_raw = _metadata_value(args.population_role)
    cli_role = cli_role_raw.upper() if cli_role_raw else None
    session_role = _metadata_value(metadata.get("population_role"))
    if session_role:
        session_role = session_role.upper()
    if cli_role and session_role and cli_role != session_role:
        _configuration_mismatch("population role", cli_role, session_role.upper())
    selected_role = cli_role or session_role
    if selected_role not in {"DEV", "FINAL"}:
        raise ValueError(
            "population role must be DEV or FINAL; provide --population-role "
            "outside an evaluation session")

    cli_lighting = _metadata_value(args.lighting_condition)
    session_lighting = _metadata_value(metadata.get("lighting"))
    if (cli_lighting and session_lighting
            and cli_lighting.lower() != session_lighting.lower()):
        _configuration_mismatch(
            "lighting", cli_lighting.lower(), session_lighting.lower())

    cli_quality = _explicit_text(args.intrinsics_quality)
    session_quality = _explicit_text(metadata.get("intrinsics_quality"))
    if (cli_quality and session_quality
            and cli_quality.upper() != session_quality.upper()):
        _configuration_mismatch(
            "intrinsics quality", cli_quality.upper(), session_quality.upper())

    cli_source = _explicit_text(args.intrinsics_source)
    session_source = _explicit_text(metadata.get("intrinsics_source"))
    if cli_source and session_source and cli_source != session_source:
        _configuration_mismatch("intrinsics source", cli_source, session_source)

    args.object_type = geometry_spec.object_type
    args.population_role = selected_role
    args.lighting_condition = cli_lighting or session_lighting
    args.intrinsics_quality = (
        (cli_quality or session_quality).upper()
        if (cli_quality or session_quality) else None)
    args.intrinsics_source = cli_source or session_source
    if not args.capture_session_id:
        args.capture_session_id = _metadata_value(metadata.get("session_id"))
    return metadata, geometry_spec


def _discover_evaluation_session_pool(
        eval_root, initial_seq, initial_out_dir, cli_args, geometry_registry,
        repo, required_role, required_object_type):
    """Build per-session contexts for the evaluation editor.

    Writable positive sessions keep the initial population role but may use
    different registered object geometries.  Geometry, intrinsics and output
    paths are resolved independently for every row, so switching plastic ->
    wood can never reuse plastic dimensions or the previous output directory.

    ``incoming/sessions`` is intentionally different.  One continuous capture
    can contain plastic and wood, so an exhaustive pixel-review manifest assigns
    each raw frame to PLASTIC, WOOD or EXCLUDE before the capture is exposed as
    two zero-copy annotation views.  The views share RGB bytes and K but have
    independent geometry, JSON, overlay and frame-tag outputs.  They are
    FINAL-intent *staging* annotations only: saving never makes the raw capture
    an active evaluation member.
    """
    initial_seq = os.path.abspath(initial_seq)
    initial_out_dir = os.path.abspath(initial_out_dir)
    selected = []
    contexts = {}
    required_role = str(required_role).upper()

    role_layouts = {
        "DEV": ("dev_existing/sessions", "dev_existing/annotations"),
        "FINAL": ("final/positive/sessions", "final/positive/annotations"),
    }
    if required_role not in role_layouts:
        raise ValueError(
            f"evaluation session selector does not support role {required_role!r}")

    # 세션 전환은 **한 namespace 안에서만** 가능했다.  DEV 로 열면 FINAL 세션이,
    # FINAL 로 열면 DEV 세션이 목록에서 통째로 빠져 "세션이 4개뿐" 으로 보였다.
    # 이제 두 layout 을 모두 훑고, 후보마다 **자기 role/namespace** 로 검증한다.
    # 초기 세션의 role 을 다른 세션에 강요하지 않는다.
    candidate_layouts: list[tuple[str, str, str, str]] = []
    for layout_role, (sess_rel, ann_rel) in role_layouts.items():
        sess_root = os.path.join(eval_root, *sess_rel.split("/"))
        ann_root = os.path.join(eval_root, *ann_rel.split("/"))
        for cand_name, cand_path in discover_sessions([sess_root], repo):
            candidate_layouts.append((cand_name, cand_path, layout_role, ann_root))

    if all(os.path.realpath(path) != os.path.realpath(initial_seq)
           for _n, path, _r, _a in candidate_layouts):
        initial_ann = os.path.join(
            eval_root, *role_layouts[required_role][1].split("/"))
        candidate_layouts.insert(
            0, (os.path.basename(initial_seq), initial_seq, required_role,
                initial_ann))

    for name, session_path, layout_role, annotations_root in candidate_layouts:
        candidate_args = copy.copy(cli_args)
        # Explicit CLI assertions apply to the session used to open the
        # process.  Every other chooser row is governed by its own session.json
        # and must not inherit the initial object's geometry or capture ID.
        if os.path.realpath(session_path) != os.path.realpath(initial_seq):
            for field in _SESSION_RUNTIME_ARG_FIELDS:
                setattr(candidate_args, field, None)
        try:
            metadata, geometry_spec = _resolve_annotation_configuration(
                candidate_args, session_path, geometry_registry, eval_root)
            # 후보의 role 은 그 세션이 사는 namespace 가 정한다.  초기 세션만
            # CLI 가 준 role 을 그대로 쓴다 (명시 assertion 이므로).
            if os.path.realpath(session_path) == os.path.realpath(initial_seq):
                if str(candidate_args.population_role).upper() != required_role:
                    continue
            elif str(candidate_args.population_role).upper() != layout_role:
                continue
            if geometry_spec.object_type not in {
                    PLASTIC_OBJECT_TYPE, WOOD_OBJECT_TYPE}:
                continue
            if (geometry_spec.object_type == WOOD_OBJECT_TYPE
                    and candidate_args.intrinsics_quality is None):
                continue
            output_path = os.path.join(annotations_root, name)
            session_path, output_path = _validate_evaluation_paths(
                eval_root, session_path, output_path,
                candidate_args.population_role)
            camera_matrix, camera_source = _resolve_session_intrinsics(
                session_path, output_path, evaluation=True)
        except (OSError, TypeError, ValueError, WorkspaceError) as exc:
            # A broken sibling must not prevent the explicitly requested,
            # already-validated session from opening.  It is excluded rather
            # than guessed into the pool.
            print(f"[세션 제외] {name}: {exc}")
            continue
        key = os.path.realpath(session_path)
        selected.append((name, session_path))
        contexts[key] = {
            "args": candidate_args,
            "metadata": metadata,
            "geometry_spec": geometry_spec,
            "out_dir": output_path,
            "K": camera_matrix,
            "K_source": camera_source,
            "writable": True,
            "workspace_scope": required_role,
            "display_role": required_role,
            "frame_count": len(_session_image_paths(session_path)),
        }

    # Raw incoming captures remain immutable/inactive.  Validate that contract,
    # then expose two object-specific zero-copy staging rows.  A unique context
    # key is required because both chooser rows deliberately share ``seq``.
    incoming_sessions_root = os.path.join(eval_root, "incoming", "sessions")
    incoming_annotations_root = os.path.join(
        eval_root, "incoming", "annotations")
    for name, session_path in discover_sessions([incoming_sessions_root], repo):
        try:
            metadata = load_session_metadata(Path(session_path))
            session_id = _explicit_text(metadata.get("session_id"))
            if session_id != name:
                raise ValueError(
                    f"session id mismatch: directory={name}, session={session_id}")
            if metadata.get("workspace_scope") != "INCOMING_UNREVIEWED":
                raise ValueError("workspace_scope must be INCOMING_UNREVIEWED")
            if ("population_role" not in metadata
                    or metadata.get("population_role") is not None):
                raise ValueError("incoming population_role must be null")
            if metadata.get("active_evaluation_member") is not False:
                raise ValueError("incoming active_evaluation_member must be false")
            if str(metadata.get("object_type", "")).strip().lower() != "unknown":
                raise ValueError("mixed incoming object_type must be unknown")
            output_path = os.path.join(incoming_annotations_root, name)
            camera_matrix, camera_source = _resolve_session_intrinsics(
                session_path, output_path, evaluation=True)
            camera_metadata = metadata.get("camera")
            if not isinstance(camera_metadata, dict):
                raise ValueError("incoming camera metadata must be an object")
            declared_k = camera_metadata.get("K")
            if declared_k is not None:
                try:
                    declared_matrix = np.asarray(
                        declared_k, dtype=np.float64).reshape(3, 3)
                except (TypeError, ValueError) as exc:
                    raise ValueError("incoming camera.K must be a numeric 3x3") from exc
                if (not np.isfinite(declared_matrix).all()
                        or not np.allclose(
                            declared_matrix, camera_matrix,
                            rtol=0.0, atol=1e-6)):
                    raise ValueError("incoming camera.K conflicts with cam_K.txt")
            candidate_args = copy.copy(cli_args)
            for field in _SESSION_RUNTIME_ARG_FIELDS:
                setattr(candidate_args, field, None)
        except (OSError, TypeError, ValueError, WorkspaceError) as exc:
            print(f"[세션 제외] {name}: {exc}")
            continue
        frame_paths = _session_image_paths(session_path)
        source_ordinal_by_path = {
            path: ordinal for ordinal, path in enumerate(frame_paths, start=1)
        }
        try:
            object_frame_paths, frame_review = (
                _incoming_reviewed_frame_partitions(
                    metadata, session_path, frame_paths, geometry_registry))
        except (TypeError, ValueError) as exc:
            print(f"[세션 제외] {name}: {exc}")
            continue
        source_quality = (_explicit_text(
            camera_metadata.get("intrinsics_quality")) or "UNKNOWN")
        source_intrinsics = (_explicit_text(
            camera_metadata.get("intrinsics_source")) or camera_source)
        # PROVIDED_UNVERIFIED is valid capture provenance but not a GT-v2
        # intrinsics-quality enum.  Staging wood labels therefore fail closed
        # as UNKNOWN while keeping the original statement in the source text.
        gt_quality = (
            source_quality
            if source_quality in {
                "CALIBRATED", "SENSOR_PROFILE_SCALED",
                "ESTIMATED_HFOV", "UNKNOWN"}
            else "UNKNOWN")
        gt_source = (
            f"{source_intrinsics}; capture quality={source_quality}; "
            f"raw session={name}")

        for object_slug, object_type in (
                ("plastic", PLASTIC_OBJECT_TYPE),
                ("wood", WOOD_OBJECT_TYPE)):
            view_id = f"{name}__{object_slug}"
            context_key = f"incoming-annotation:{view_id}"
            geometry_spec = geometry_registry.resolve(object_type)
            view_frame_paths = object_frame_paths[geometry_spec.object_type]
            view_source_ordinals = [
                source_ordinal_by_path[path] for path in view_frame_paths
            ]
            view_metadata = copy.deepcopy(metadata)
            # Never propagate a stale ordinal contract into an object view if
            # an older session file temporarily contains both representations.
            view_metadata.pop("object_frame_partition", None)
            view_metadata.update({
                "schema_version": "pallet_eval_incoming_annotation_view_v1",
                "session_id": view_id,
                "source_session_id": name,
                "workspace_scope": "INCOMING_ANNOTATION",
                "population_role": "FINAL",
                "active_evaluation_member": False,
                "review_status": "OBJECT_SPECIFIC_ANNOTATION_STAGING",
                "object_type": geometry_spec.object_type,
                "intrinsics_quality": gt_quality,
                "intrinsics_source": gt_source,
            })
            view_args = copy.copy(candidate_args)
            view_args.object_type = geometry_spec.object_type
            view_args.population_role = "FINAL"
            view_args.lighting_condition = _metadata_value(
                metadata.get("lighting"))
            view_args.intrinsics_quality = gt_quality
            view_args.intrinsics_source = gt_source
            view_args.capture_session_id = view_id
            output_path = os.path.join(incoming_annotations_root, view_id)
            try:
                _validate_incoming_staging_membership(
                    output_path, view_frame_paths)
                view_camera_matrix, view_camera_source = (
                    _resolve_session_intrinsics(
                        session_path, output_path, evaluation=True))
            except (OSError, TypeError, ValueError) as exc:
                print(f"[세션 제외] {view_id}: {exc}")
                continue

            selected.append((
                f"{name} · {object_slug.upper()}",
                session_path,
                context_key,
            ))
            contexts[context_key] = {
                "args": view_args,
                "metadata": view_metadata,
                "geometry_spec": geometry_spec,
                "out_dir": output_path,
                "tag_session_dir": output_path,
                "source_session_dir": session_path,
                "frame_paths": view_frame_paths,
                "source_ordinals": view_source_ordinals,
                "source_frame_count": len(frame_paths),
                "frame_review_manifest_path": frame_review["manifest_path"],
                "frame_review_label": object_slug,
                "frame_review_label_counts": dict(frame_review["label_counts"]),
                "frame_review_exclude_reason_counts": dict(
                    frame_review["exclude_reason_counts"]),
                "K": view_camera_matrix,
                "K_source": view_camera_source,
                "writable": True,
                "workspace_scope": "INCOMING_ANNOTATION",
                "display_role": "STAGING",
                "frame_count": len(view_frame_paths),
                "intrinsics_quality": gt_quality,
                "intrinsics_source": gt_source,
                "active_evaluation_member": False,
                "refresh_evaluation": False,
                "force_explicit_object_type": True,
            }

    initial_key = os.path.realpath(initial_seq)
    if initial_key not in contexts:
        raise ValueError(
            "initial evaluation session was excluded from its own safe pool: "
            f"{initial_seq}")
    initial_context = contexts[initial_key]
    if os.path.realpath(initial_context["out_dir"]) != os.path.realpath(initial_out_dir):
        raise ValueError(
            "initial evaluation output changed during session discovery: "
            f"requested={initial_out_dir}, resolved={initial_context['out_dir']}")
    expected_initial_object = geometry_registry.resolve(
        required_object_type).object_type
    if initial_context["geometry_spec"].object_type != expected_initial_object:
        raise ValueError(
            "initial evaluation geometry changed during session discovery: "
            f"requested={expected_initial_object}, "
            f"resolved={initial_context['geometry_spec'].object_type}")
    return selected, contexts


def _mark_annotation_dirty(s):
    s.dirty = True
    s.annotation_dirty = True
    s.discard_armed = None


def _has_unsaved_changes(s):
    return bool(
        getattr(s, "dirty", False)
        or getattr(s, "annotation_dirty", False)
        or getattr(s, "frame_tags_dirty", False))


def _confirm_discard(s, action, message):
    """Require the same navigation action twice without losing retry state."""
    if not _has_unsaved_changes(s):
        s.discard_armed = None
        return True
    if getattr(s, "discard_armed", None) == action:
        s.dirty = False
        s.annotation_dirty = False
        s.frame_tags_dirty = False
        s.discard_armed = None
        return True
    s.discard_armed = action
    print(message)
    return False


def session_summary(sessions, repo, population_role="DEV", output_dirs=None,
                    contexts=None):
    """세션 목록에 frame/GT 수와 세션별 역할·object·상태를 붙인다.

    ``contexts``를 생략한 호출에는 과거 4-tuple API를 유지한다.  Evaluation
    chooser는 context-aware dictionaries를 받아 mixed geometry와 REVIEW ONLY를
    각 행에 정확히 표시한다.
    """
    rows = []
    output_dirs = output_dirs or {}
    for entry in sessions:
        name, seq, context_key = _session_entry_parts(entry)
        context = (contexts or {}).get(context_key)
        n = (int(context["frame_count"])
             if context is not None and context.get("frame_count") is not None
             else len(_session_image_paths(seq)))
        od = output_dirs.get(context_key)
        if context is not None:
            od = context.get("out_dir")
        if od is None:
            od, _legacy_eval_layout = resolve_out_dir(name, repo)
        writable = bool(context.get("writable", True)) if context else True
        done = (len(glob.glob(os.path.join(od, "*.json")))
                if writable else None)
        if context is None:
            rows.append(
                (name, n, done, str(population_role).upper() == "FINAL"))
            continue
        spec = context.get("geometry_spec")
        metadata = context.get("metadata") or {}
        active_evaluation_member = bool(
            context.get("active_evaluation_member", True))
        rows.append({
            "name": name,
            "frames": n,
            "done": done,
            "role": context.get("display_role") or "-",
            "object": (spec.object_type if spec is not None else "MIXED"),
            "lighting": str(metadata.get("lighting") or "unknown").upper(),
            "writable": writable,
            "status": (
                "STAGING EDIT"
                if writable and not active_evaluation_member
                else "EDIT" if writable else "REVIEW ONLY"),
        })
    return rows


def pick_session_dialog(rows, current):
    """진짜 드롭다운으로 세션을 고른다. 선택 인덱스 또는 None(취소).

    OpenCV 창에는 위젯을 붙이는 API 가 없어서 목록을 이미지에 그려야 했는데, tkinter 는
    표준 라이브러리라 콤보박스를 그냥 띄울 수 있다. 별도 프로세스가 아니라 별도 Tk 루트로
    잠깐 열었다 닫으므로 cv2 이벤트 루프와 섞이지 않는다.
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as e:
        print(f"[WARN] tkinter 사용 불가({e}) — 패널 목록으로 대체")
        return None

    labels = []
    for i, row in enumerate(rows):
        if isinstance(row, dict):
            done = row.get("done")
            done_text = "-" if done is None else str(done)
            labels.append(
                f"{i+1:>2}. {row['name']}   ({row['frames']}f, GT {done_text})  "
                f"[{row['role']} · {row['object']} · {row['lighting']} · "
                f"{row['status']}]"
            )
        else:
            nm, nfr, done, sealed = row
            labels.append(
                f"{i+1:>2}. {nm}   ({nfr}f, done {done})"
                f"{'  [FINAL ROLE]' if sealed else ''}")
    picked = {"i": None}

    root = tk.Tk()
    root.title("세션 선택")
    root.attributes("-topmost", True)
    frm = ttk.Frame(root, padding=12)
    frm.grid()
    ttk.Label(frm, text="촬영 세션을 고르세요").grid(column=0, row=0, sticky="w", pady=(0, 6))
    var = tk.StringVar(value=labels[current] if 0 <= current < len(labels) else labels[0])
    box = ttk.Combobox(frm, textvariable=var, values=labels, state="readonly", width=96)
    box.grid(column=0, row=1, pady=(0, 10))
    box.focus()

    def ok(*_):
        try:
            picked["i"] = labels.index(var.get())
        except ValueError:
            picked["i"] = None
        root.destroy()

    ttk.Button(frm, text="열기", command=ok).grid(column=0, row=2, sticky="e")
    box.bind("<Return>", ok)
    box.bind("<<ComboboxSelected>>", lambda e: None)
    root.bind("<Escape>", lambda e: root.destroy())
    root.update_idletasks()
    root.geometry(f"+{root.winfo_screenwidth()//2 - 220}+{root.winfo_screenheight()//2 - 80}")
    root.mainloop()
    return picked["i"]


# ─── Mouse callback ──────────────────────────────────────────────────────────

WIN = "Annotate"

_VISIBILITY_REASON_CYCLE = (
    (0, "unknown"),
    (2, "visible"),
    (1, "occluded"),
    (1, "truncated"),
)

_CONDITION_TAG_ON_VALUES = {
    # Evaluation membership only distinguishes none from any positive severity.
    # Keep the existing canonical generic-positive values in storage while the
    # per-frame UI exposes the simpler ON/OFF decision.
    "occlusion": "medium",
    "truncation": "mild",
}
_CONDITION_TAG_POSITIVE_VALUES = frozenset({"mild", "medium", "heavy"})
_CONDITION_MODE_TAG_KEYS = {
    ord("1"): "occlusion",
    ord("2"): "truncation",
}
_CONDITION_MODE_ELEVATION_KEYS = {
    ord("3"): "low",
    ord("4"): "mid",
    ord("5"): "high",
}
_CONDITION_MODE_DISTANCE_KEYS = {
    ord("n"): "near",
    ord("m"): "mid",
    ord("6"): "far",
}
_ACTIVE_EVALUATION_TAG_FIELDS = (
    "occlusion",
    "truncation",
    "elevation_bin",
    "distance_bin",
)
# ``/`` is deliberately unused by CLICK, MANIPULATE, goto, line input, and
# session navigation.  A punctuation key also avoids the old C/c ambiguity
# where Shift+C entered CONDITIONS but lowercase c ran centroid auto-fill.
_CONDITION_MODE_KEY = ord("/")

# Qt/OpenCV cannot reliably change a trackbar maximum after creation, so the
# frame slider uses one fixed-resolution position range for every session.
_FRAME_SLIDER_TICKS = 500


def _frame_cur_to_tick(current, total):
    return (0 if total <= 1 else
            int(round(current / (total - 1) * _FRAME_SLIDER_TICKS)))


def _frame_tick_to_cur(tick, total):
    return (0 if total <= 1 else
            int(round(tick / _FRAME_SLIDER_TICKS * (total - 1))))


def _frame_trackbar_target(current, total, tick):
    """Return a user-requested frame, never a lossy programmatic round-trip.

    Large incoming sessions contain far more frames than the fixed 500 slider
    ticks, so adjacent frames often map to the same tick. Converting that
    unchanged tick back to a frame would roll sequential navigation back to a
    representative frame. Only a raw tick change can be a slider request.
    """
    if total <= 1 or int(tick) == _frame_cur_to_tick(current, total):
        return None
    target = _frame_tick_to_cur(int(tick), total)
    return None if target == current else target
# Qt5 HighGUI folds Shift+letter back to lowercase in this environment.  Use
# one unmodified punctuation key for goto so trying the documented command can
# never fall through to lowercase ``g`` (auto-fill save).
_GOTO_MODE_KEY = ord(";")


def _state_annotation_tag_evidence(s):
    """Build only explicit annotation evidence for the shared tag resolver."""
    document = (
        copy.deepcopy(s.legacy_document)
        if isinstance(getattr(s, "legacy_document", None), dict)
        else {"objects": [{}]})
    objects = document.get("objects")
    if not isinstance(objects, list) or not objects or not isinstance(objects[0], dict):
        document["objects"] = [{}]
    obj = document["objects"][0]
    obj["keypoint_annotations"] = copy.deepcopy(
        _ensure_keypoint_annotations(s))
    obj["occlusion_level"] = str(
        getattr(s, "occlusion_level", "unknown") or "unknown")
    reasons = {
        str(entry.get("reason", "unknown")).strip().lower()
        for entry in obj["keypoint_annotations"] if isinstance(entry, dict)
    }
    pose = getattr(s, "pose", None)
    projected = pose.get("projected_all") if isinstance(pose, dict) else None
    if (projected is not None and len(projected) > 0
            and getattr(s, "img_shape", None) is not None):
        height, width = s.img_shape[:2]
        obj["truncation"] = _truncation_payload(
            list(projected)[:8], width, height)
    elif "truncated" in reasons:
        obj["truncation"] = {"is_truncated": True}
    return document


def _refresh_effective_frame_tags(s, *, use_state_evidence=False):
    metadata = (
        s.session_metadata if isinstance(getattr(s, "session_metadata", None), dict)
        else {})
    overrides = (
        s.frame_tag_overrides
        if isinstance(getattr(s, "frame_tag_overrides", None), dict) else {})
    kwargs = {}
    if use_state_evidence:
        kwargs["annotation_document"] = _state_annotation_tag_evidence(s)
    else:
        annotation_path = getattr(s, "current_annotation_path", None)
        if annotation_path:
            kwargs["annotation_path"] = Path(annotation_path)
    tags, sources = resolve_effective_frame_tags(metadata, overrides, **kwargs)
    s.frame_tags = tags
    s.frame_tag_sources = sources
    return tags, sources


def _load_frame_tag_state(s, frame_identity, annotation_path):
    """Load one frame's explicit overrides and resolve effective UI values."""
    s.current_frame_identity = os.path.basename(str(frame_identity))
    s.current_annotation_path = os.path.abspath(str(annotation_path))
    s.frame_tag_pending_updates = {}
    s.condition_batch_armed = None
    s.frame_tag_overrides = {}
    session_dir = getattr(s, "eval_session_dir", None)
    if session_dir:
        all_overrides = load_frame_tag_overrides(Path(session_dir))
        canonical = canonical_frame_tag_identity(
            s.current_frame_identity, session_id=Path(session_dir).name)
        row = all_overrides.get(canonical, {})
        s.frame_tag_overrides = {
            field: str(row.get(field, "")).strip()
            for field in FRAME_TAG_FIELDS
            if str(row.get(field, "")).strip().lower() not in {"", "unknown"}
        }
    _refresh_effective_frame_tags(s)
    # Retain the last explicit value for diagnostics/backward-compatible State
    # snapshots.  Toggle decisions themselves use the live effective value.
    s.frame_tag_cycle_values = {
        field: s.frame_tag_overrides.get(field, "unknown")
        for field in FRAME_TAG_FIELDS
    }
    s.frame_tags_dirty = False


def _set_frame_tag_override(s, field, value, *, use_state_evidence=False):
    """Set one explicit FRAME override and refresh the panel immediately."""
    if field not in FRAME_TAG_FIELDS:
        raise ValueError(f"condition mode does not edit {field!r}")
    value = str(value).strip().lower()
    cursors = (
        s.frame_tag_cycle_values
        if isinstance(getattr(s, "frame_tag_cycle_values", None), dict) else {})
    cursors[field] = value
    s.frame_tag_cycle_values = cursors
    pending = (
        s.frame_tag_pending_updates
        if isinstance(getattr(s, "frame_tag_pending_updates", None), dict) else {})
    pending[field] = value
    s.frame_tag_pending_updates = pending
    overrides = (
        s.frame_tag_overrides
        if isinstance(getattr(s, "frame_tag_overrides", None), dict) else {})
    overrides[field] = value
    s.frame_tag_overrides = overrides
    s.frame_tags_dirty = True
    s.dirty = True
    s.discard_armed = None
    s.condition_batch_armed = None
    _refresh_effective_frame_tags(
        s, use_state_evidence=use_state_evidence)
    effective = s.frame_tags[field]
    source = s.frame_tag_sources[field]
    if field in _CONDITION_TAG_ON_VALUES:
        display = (
            "ON" if effective in _CONDITION_TAG_POSITIVE_VALUES
            else "OFF" if effective == "none" else "UNKNOWN"
        )
    else:
        display = str(effective or "unknown").upper()
    print(f"[Frame tag] {field} -> {display} [{source}]")
    return value


def _toggle_frame_condition(s, field):
    """Toggle one binary condition and always create an explicit FRAME value."""
    if field not in _CONDITION_TAG_ON_VALUES:
        raise ValueError(f"condition mode does not edit {field!r}")

    tags = s.frame_tags if isinstance(getattr(s, "frame_tags", None), dict) else {}
    current = str(tags.get(field, "unknown")).strip().lower()
    value = (
        "none"
        if current in _CONDITION_TAG_POSITIVE_VALUES
        else _CONDITION_TAG_ON_VALUES[field]
    )
    return _set_frame_tag_override(
        s, field, value, use_state_evidence=True)


def _set_frame_elevation(s, value):
    """Select exactly one paper elevation bin without a nested chooser."""
    if value not in {"low", "mid", "high"}:
        raise ValueError(f"invalid elevation_bin: {value!r}")
    return _set_frame_tag_override(s, "elevation_bin", value)


def _set_frame_distance(s, value):
    """Select one distance bin used by frame or session batch annotation."""
    if value not in {"near", "mid", "far"}:
        raise ValueError(f"invalid distance_bin: {value!r}")
    return _set_frame_tag_override(s, "distance_bin", value)


def _clear_frame_distance_tag(s):
    """Clear the manual distance label back to UNKNOWN/default.

    ``unknown`` retains the dataset contract's existing meaning: remove the
    explicit FRAME override and fall through to an explicit session default
    when one exists.  The active evaluation sessions have no distance default,
    so their effective value becomes UNKNOWN immediately.
    """
    _set_frame_tag_override(s, "distance_bin", "unknown")


def _annotated_session_frames(s, out_json):
    """Return RGB filenames whose matching canonical annotation exists."""
    tag_session_dir = getattr(s, "eval_session_dir", None)
    source_session_dir = (
        getattr(s, "source_session_dir", None) or tag_session_dir)
    eval_root = getattr(s, "eval_root", None)
    if not tag_session_dir or not source_session_dir or not eval_root:
        raise WorkspaceError(
            "session batch tags require an evaluation session (--eval-root)")
    annotation_dir = os.path.abspath(os.path.dirname(out_json))
    if getattr(s, "refresh_evaluation_workspace", True):
        try:
            _validate_evaluation_paths(
                eval_root,
                source_session_dir,
                annotation_dir,
                getattr(s, "population_role", ""),
            )
        except ValueError as exc:
            raise WorkspaceError(str(exc)) from exc
    else:
        expected_dir = os.path.abspath(
            getattr(s, "annotation_output_dir", "") or "__missing__")
        staging_root = os.path.join(eval_root, "incoming", "annotations")
        if (os.path.realpath(annotation_dir) != os.path.realpath(expected_dir)
                or not _path_is_within(annotation_dir, staging_root)
                or os.path.realpath(tag_session_dir)
                != os.path.realpath(annotation_dir)):
            raise WorkspaceError(
                "incoming staging annotation/tag path mismatch: "
                f"source={source_session_dir}, target={annotation_dir}")

    eligible_paths = (
        getattr(s, "session_frame_paths", None)
        or _session_image_paths(source_session_dir))
    source_by_stem = {
        Path(path).stem: Path(path).name
        for path in eligible_paths
    }
    annotated = []
    for annotation_path in sorted(Path(annotation_dir).glob("*.json")):
        if not annotation_path.is_file():
            continue
        frame_name = source_by_stem.get(annotation_path.stem)
        if frame_name is not None:
            annotated.append(frame_name)
    return annotated


def _apply_pending_tags_to_annotated_session(s, out_json, src_png):
    """Double-confirm and atomically apply current edits to annotated frames."""
    if not getattr(s, "session_writable", True):
        s.condition_batch_armed = None
        _toast(
            s,
            "REVIEW ONLY: batch disabled",
            (40, 40, 230),
            log="[REVIEW ONLY] incoming 세션에서는 조건 일괄 적용을 저장하지 않습니다.",
        )
        return False
    if getattr(s, "annotation_dirty", False):
        s.condition_batch_armed = None
        _toast(
            s,
            "BATCH BLOCKED: save keypoints first",
            (40, 40, 230),
            log="[BATCH BLOCKED] unsaved annotation/keypoint edits exist",
        )
        return False
    updates = dict(getattr(s, "frame_tag_pending_updates", None) or {})
    if not updates:
        s.condition_batch_armed = None
        _toast(
            s,
            "BATCH: change a condition first",
            (0, 180, 255),
            log="[Batch tags] no current frame condition edits to apply",
        )
        return False
    try:
        frames = _annotated_session_frames(s, out_json)
    except (OSError, TypeError, ValueError, WorkspaceError) as exc:
        s.condition_batch_armed = None
        _toast(s, "BATCH BLOCKED", (40, 40, 230), log=f"[Batch tags] {exc}")
        return False
    if not frames:
        s.condition_batch_armed = None
        _toast(
            s,
            "BATCH: no annotated frames",
            (0, 180, 255),
            log="[Batch tags] no matching annotation JSON files in this session",
        )
        return False

    signature = (
        os.path.realpath(getattr(s, "eval_session_dir", "")),
        tuple(sorted(updates.items())),
        tuple(frames),
    )
    if getattr(s, "condition_batch_armed", None) != signature:
        s.condition_batch_armed = signature
        fields = ",".join(sorted(updates))
        _toast(
            s,
            f"PRESS a AGAIN: {len(frames)} annotated",
            (0, 180, 255),
            log=f"[Batch tags] confirm: {fields} -> {len(frames)} annotated frames",
            seconds=3.0,
        )
        return False

    session_dir = Path(s.eval_session_dir)
    try:
        update_frame_tags_csv_many(
            session_dir,
            {frame_name: updates for frame_name in frames},
        )
    except (OSError, TypeError, ValueError, WorkspaceError) as exc:
        s.condition_batch_armed = None
        _toast(
            s,
            "BATCH FAILED: CSV unchanged",
            (40, 40, 230),
            log=f"[Batch tags] atomic update failed: {exc}",
        )
        return False

    _load_frame_tag_state(s, s.current_frame_identity, out_json)
    s.dirty = False
    s.annotation_dirty = False
    s.frame_tags_dirty = False
    s.discard_armed = None
    _refresh_evaluation_workspace(
        s, out_json, image_path=src_png, deleted=False)
    fields = ",".join(sorted(updates))
    _toast(
        s,
        f"BATCH SAVED: {len(frames)} annotated",
        (0, 220, 0),
        log=f"[Batch tags] saved {fields} to {len(frames)} annotated frames",
        seconds=2.5,
    )
    return True


def _set_condition_mode(s, enabled):
    """Enter/leave the key-isolated evaluation-condition editor."""

    enabled = bool(enabled)
    if enabled and not getattr(s, "session_writable", True):
        s.condition_mode = False
        _toast(
            s,
            "REVIEW ONLY: conditions disabled",
            (40, 40, 230),
            log="[REVIEW ONLY] incoming 세션은 조건 tag를 저장하지 않습니다.",
        )
        return
    s.condition_mode = enabled
    if enabled:
        # Do not leave a half-entered line intersection consuming mouse
        # clicks while the keyboard-only condition editor is on screen.
        s.line_mode = False
        s.line_pts = []
        _toast(
            s,
            "CONDITIONS: n/m/6=DIST u=UNKNOWN a,a=BATCH",
            (0, 220, 255),
            seconds=2.0,
        )
        print("[Mode] CLICK -> CONDITIONS  "
              "(1/2=ON/OFF, 3/4/5=elevation, n/m/6=distance, "
              "u=distance unknown, "
              "a,a=batch, s=save, /=back)")
    else:
        s.condition_batch_armed = None
        _toast(s, "CONDITION MODE -> CLICK", (0, 220, 255), seconds=1.2)
        print("[Mode] CONDITIONS -> CLICK")


def _handle_condition_key(key, s, out_json, out_png, src_png, K):
    """Handle condition editing without colliding with keypoint shortcuts."""

    if not getattr(s, "session_writable", True):
        s.condition_mode = False
        _toast(
            s,
            "REVIEW ONLY: conditions disabled",
            (40, 40, 230),
            log="[REVIEW ONLY] incoming 세션은 조건 tag를 저장하지 않습니다.",
        )
        return None
    if key in (_CONDITION_MODE_KEY, 27):
        _set_condition_mode(s, False)
        return None
    field = _CONDITION_MODE_TAG_KEYS.get(key)
    if field is not None:
        value = _toggle_frame_condition(s, field)
        number = key - ord("0")
        display = "ON" if value in _CONDITION_TAG_POSITIVE_VALUES else "OFF"
        _toast(s, f"{number} {field}: {display}", (0, 220, 255), seconds=1.2)
        return None
    elevation = _CONDITION_MODE_ELEVATION_KEYS.get(key)
    if elevation is not None:
        _set_frame_elevation(s, elevation)
        _toast(
            s,
            f"ELEVATION: {elevation.upper()}",
            (0, 220, 255),
            seconds=1.2,
        )
        return None
    distance = _CONDITION_MODE_DISTANCE_KEYS.get(key)
    if distance is not None:
        _set_frame_distance(s, distance)
        _toast(
            s,
            f"DISTANCE: {distance.upper()}",
            (0, 220, 255),
            seconds=1.2,
        )
        return None
    if key == ord("u"):
        _clear_frame_distance_tag(s)
        _toast(
            s,
            "DISTANCE: UNKNOWN",
            (0, 220, 255),
            log="[Frame tag] distance_bin -> UNKNOWN",
            seconds=1.8,
        )
        return None
    if key == ord("a"):
        _apply_pending_tags_to_annotated_session(s, out_json, src_png)
        return None
    if key == ord("s"):
        # CLICK-mode ``s`` intentionally deletes an annotation when every
        # point is empty.  That destructive shortcut must not leak into the
        # metadata editor: a user entering CONDITIONS on a blank frame expects
        # to edit tags, never to delete GT.
        if not any(point is not None for point in (s.kps_2d or [])):
            _toast(
                s,
                "SAVE BLOCKED: annotate keypoints first",
                (40, 40, 230),
                log="[SAVE BLOCKED] CONDITIONS mode cannot delete annotation; "
                    "return to CLICK and annotate keypoints first.",
            )
            return None
        return _handle_click_key(key, s, out_json, out_png, src_png, K)
    # CONDITIONS is fully modal: every non-owned key, including 0/8/9 and all
    # CLICK/MANIPULATE/navigation shortcuts, is a true no-op.
    return None


def _write_state_frame_tags(s):
    if not getattr(s, "session_writable", True):
        raise WorkspaceError(
            "REVIEW ONLY incoming session cannot write frame tags")
    if not getattr(s, "frame_tags_dirty", False):
        return
    session_dir = getattr(s, "eval_session_dir", None)
    if not session_dir:
        raise WorkspaceError(
            "frame tag edits require an evaluation session (--eval-root)")
    updates = dict(getattr(s, "frame_tag_pending_updates", None) or {})
    update_frame_tags_csv(
        Path(session_dir), s.current_frame_identity, updates)
    # Reload the committed row instead of assuming the merge result.
    all_overrides = load_frame_tag_overrides(Path(session_dir))
    canonical = canonical_frame_tag_identity(
        s.current_frame_identity, session_id=Path(session_dir).name)
    row = all_overrides.get(canonical, {})
    s.frame_tag_overrides = {
        field: str(row.get(field, "")).strip()
        for field in FRAME_TAG_FIELDS
        if str(row.get(field, "")).strip().lower() not in {"", "unknown"}
    }
    s.frame_tag_pending_updates = {}
    s.frame_tags_dirty = False
    _refresh_effective_frame_tags(s)


def _missing_evaluation_tags(s):
    tags = getattr(s, "frame_tags", None) or {}
    return [
        field for field in _ACTIVE_EVALUATION_TAG_FIELDS
        if str(tags.get(field, "unknown")).strip().lower() == "unknown"
    ]


def _point_in_image(s: State, point):
    if point is None or s.img_shape is None:
        return False
    h, w = s.img_shape[:2]
    return bool(0.0 <= float(point[0]) < float(w)
                and 0.0 <= float(point[1]) < float(h))


def _ensure_keypoint_annotations(s: State):
    """Keep nine explicit states; never infer visibility from old coordinates."""
    old = s.keypoint_annotations if isinstance(s.keypoint_annotations, list) else []
    entries = []
    for i in range(9):
        point = s.kps_2d[i] if s.kps_2d is not None and i < len(s.kps_2d) else None
        base = dict(old[i]) if i < len(old) and isinstance(old[i], dict) else {}
        base["xy"] = list(point) if point is not None else base.get("xy")
        base.setdefault("visibility", 0)
        base.setdefault("source", "unknown")
        base.setdefault("reason", "unknown")
        base["in_frame"] = _point_in_image(s, base.get("xy"))
        entries.append(base)
    s.keypoint_annotations = entries
    return entries


def _set_keypoint_state(s: State, index, point, *, source, visibility, reason):
    entries = _ensure_keypoint_annotations(s)
    xy = None if point is None else [float(point[0]), float(point[1])]
    entries[index] = {
        "xy": xy,
        "visibility": int(visibility),
        "in_frame": _point_in_image(s, xy),
        "source": source,
        "reason": reason,
    }


def _clear_keypoint_state(s: State, index):
    _set_keypoint_state(
        s, index, None, source="unknown", visibility=0, reason="unknown")


def _cycle_visibility_reason(s: State, index):
    entries = _ensure_keypoint_annotations(s)
    if entries[index].get("xy") is None:
        print(f"[Visibility] kp{index}: 좌표가 없어 visibility를 확정할 수 없습니다.")
        return
    current = (int(entries[index].get("visibility", 0)),
               str(entries[index].get("reason", "unknown")))
    if current == (1, "unknown"):
        next_state = (1, "occluded")
    else:
        try:
            position = _VISIBILITY_REASON_CYCLE.index(current)
        except ValueError:
            position = 0
        next_state = _VISIBILITY_REASON_CYCLE[
            (position + 1) % len(_VISIBILITY_REASON_CYCLE)]
    entries[index]["visibility"], entries[index]["reason"] = next_state
    entries[index]["in_frame"] = _point_in_image(s, entries[index].get("xy"))
    _mark_annotation_dirty(s)
    _refresh_effective_frame_tags(s, use_state_evidence=True)
    print(f"[Visibility] kp{index}: visibility={next_state[0]} "
          f"reason={next_state[1]} source={entries[index].get('source', 'unknown')}")


def _mark_projected_fallbacks(s: State):
    """Record auto-filled provenance without replacing legacy manual_kps."""
    if s.pose is None:
        return
    entries = _ensure_keypoint_annotations(s)
    changed = False
    for i, point in enumerate(s.pose.get("projected_all", [])[:9]):
        if s.kps_2d[i] is not None or (point[0] == -1 and point[1] == -1):
            continue
        source = "centroid_auto" if i == 8 else "pnp_projected"
        entries[i] = {
            "xy": [float(point[0]), float(point[1])],
            "visibility": 1,
            "in_frame": _point_in_image(s, point),
            "source": source,
            "reason": "unknown",
        }
        changed = True
    if changed:
        _mark_annotation_dirty(s)


def _sync_axis_candidates(s: State):
    """Keep the unobservable signed pair without promoting either candidate.

    Camera-facing PnP can determine W/D parity, but a pallet image cannot
    distinguish the two signs inside that parity (0/180 or 90/270).  Normal
    annotation therefore always preserves the pair as unresolved GT.
    """
    if s.pose is None:
        return
    candidates = list(s.pose.get("_axis_assignment_candidates") or [])
    if not candidates:
        s.axis_assignment_candidates = []
        s.axis_assignment = None
        s.axis_assignment_confirmed = False
        return
    s.axis_assignment_candidates = candidates
    s.axis_assignment = None
    s.axis_assignment_confirmed = False


def _cycle_wd_parity(s: State):
    """Switch the human-reviewed camera-facing W/D parity hypothesis."""
    if s.pose is None:
        print("[W/D] PnP 후보가 아직 없습니다.")
        return
    available = []
    for item in s.pose.get("_wd_candidates", []) or []:
        value = item.get("camera_facing_hypothesis")
        if value and value not in available:
            available.append(value)
    if len(available) < 2:
        print(f"[W/D] 전환 가능한 두 parity가 없습니다: {available or 'none'}")
        return
    current = s.pose.get("_camera_facing_hypothesis")
    index = available.index(current) if current in available else -1
    target = available[(index + 1) % len(available)]
    s.camera_facing_hypothesis_override = target
    # A W/D switch changes the unresolved signed pair.
    s.axis_assignment = None
    s.axis_assignment_candidates = []
    s.axis_assignment_confirmed = False
    s._pose_key = None
    _mark_annotation_dirty(s)
    print(f"[W/D] manual parity correction: {current} -> {target}; "
          "signed pair stays unresolved")


def _save_contract_error(s: State):
    entries = _ensure_keypoint_annotations(s)
    if str(s.population_role).upper() == "FINAL":
        if not any(point is not None for point in (s.kps_2d or [])):
            return "FINAL save blocked: deleting all points cannot bypass review gates"
        unknown = [i for i in range(8)
                   if int(entries[i].get("visibility", 0)) == 0]
        if unknown:
            return ("FINAL save blocked: kp0~7 visibility unknown at "
                    + ",".join(map(str, unknown)))
    return None


def _save_contract_screen_text(error):
    """Return a short, actionable ASCII toast for a FINAL save failure."""
    marker = "visibility unknown at "
    if marker in error:
        indices = error.split(marker, 1)[1]
        return f"SAVE BLOCKED: set visibility kp{indices} with b"
    if "deleting all points" in error:
        return "SAVE BLOCKED: FINAL cannot delete all keypoints"
    return "SAVE BLOCKED: FINAL review incomplete"


def _make_state_annotation(s: State, K):
    error = _save_contract_error(s)
    if error:
        _toast(s, _save_contract_screen_text(error), (40, 40, 230), log=error)
        return None
    return make_annotation(
        s.kps_2d, s.pose, s.img_shape, K,
        dims=tuple(s.pose.get("dims") or _state_default_wdh(s)),
        split=s.split,
        extrap_mask=s.extrap_mask,
        keypoint_annotations=s.keypoint_annotations,
        axis_assignment=None,
        axis_assignment_candidates=s.axis_assignment_candidates,
        axis_assignment_confirmed=False,
        legacy_object=s.legacy_object,
        legacy_document=s.legacy_document,
        population_role=s.population_role,
        metadata=s.capture_metadata,
        occlusion_level=s.occlusion_level,
        geometry_spec=_state_geometry_spec(s),
        intrinsics_quality=getattr(s, "intrinsics_quality", None),
        intrinsics_source=getattr(s, "intrinsics_source", None),
        force_explicit_object_type=getattr(
            s, "force_explicit_object_type", False),
    )


def _save_state_annotation(s: State, K, out_json, out_png, src_png):
    if not getattr(s, "session_writable", True):
        _toast(
            s,
            "REVIEW ONLY: save disabled",
            (40, 40, 230),
            log="[REVIEW ONLY] incoming 세션은 GT를 저장하거나 평가에 편입하지 않습니다.",
        )
        return False
    try:
        _require_nonlegacy_output_dir(os.path.dirname(out_json), _REPO)
    except ValueError as exc:
        _toast(s, "[SAVE BLOCKED] legacy GT is read-only", (40, 40, 230),
               log=f"GT v2 save path rejected: {exc}: {out_json}")
        return False
    metadata_only = bool(
        os.path.isfile(out_json)
        and getattr(s, "frame_tags_dirty", False)
        and not getattr(s, "annotation_dirty", False)
        and os.path.abspath(out_json) == os.path.abspath(
            getattr(s, "loaded_annotation_path", "") or "__not_loaded__"))
    if metadata_only:
        # The canonical JSON was already parsed into this state.  Keeping its
        # bytes untouched is the strongest guarantee that a tag-only edit
        # cannot perturb keypoints, PnP pose, or any compatibility payload.
        error = _save_contract_error(s)
        if error:
            _toast(s, _save_contract_screen_text(error), (40, 40, 230),
                   log=error)
            return False
        print(f"[Metadata-only] GT unchanged: {out_json}")
    else:
        ann = _make_state_annotation(s, K)
        if ann is None:
            return False
        try:
            save_frame_json(out_json, out_png, src_png, ann,
                            registry_path=getattr(s, "geometry_registry_path", None))
        except (TypeError, ValueError) as exc:
            _toast(s, "[SAVE BLOCKED] v2 schema error", (40, 40, 230),
                   log=f"GT v2 schema validation failed: {exc}")
            return False

    # A CSV failure happens after a valid GT commit (or validation of an
    # existing canonical GT).  Do not roll back GT and do not auto-advance.
    s.annotation_dirty = False
    s.loaded_annotation_path = os.path.abspath(out_json)
    s.current_annotation_path = os.path.abspath(out_json)
    try:
        _write_state_frame_tags(s)
    except (OSError, TypeError, ValueError, WorkspaceError) as exc:
        s.dirty = True
        s.frame_tags_dirty = True
        _toast(
            s, "[SAVE PARTIAL] FRAME TAGS NOT SAVED", (40, 40, 230),
            log="[SAVE PARTIAL]\nGT saved\nFRAME TAGS NOT SAVED\n"
                f"reason: {exc}")
        return False
    if getattr(s, "eval_session_dir", None):
        _refresh_effective_frame_tags(s)
    # Read the committed JSON back for the review cache.  Overlay/report
    # failures are intentionally non-fatal: a valid GT save must never be
    # rolled back because a derived artifact could not be refreshed.
    try:
        overlay_path = render_saved_annotation_overlay(src_png, out_json)
        print(f"[Overlay] {overlay_path}")
    except Exception as exc:
        print(f"[WARN] overlay failed for {out_json}: {exc}")
    _refresh_evaluation_workspace(
        s, out_json, image_path=src_png, deleted=False)
    if (getattr(s, "eval_session_dir", None)
            and getattr(s, "active_evaluation_member", True)):
        missing = _missing_evaluation_tags(s)
        if missing:
            print("[WARN] evaluation metadata incomplete: " + ",".join(missing))
    s.dirty = False
    s.annotation_dirty = False
    s.frame_tags_dirty = False
    s.discard_armed = None
    return True


def _refresh_evaluation_workspace(s, annotation_path, *, image_path=None,
                                  deleted=False):
    """Refresh one manifest row and lightweight progress when configured.

    ``scripts/evaluation/eval_dataset_status.py`` owns dataset semantics.  The
    editor only calls its small public hook; it never invokes the import or a
    full audit after every keypress.
    """
    eval_root = getattr(s, "eval_root", None)
    if not eval_root:
        return
    if not getattr(s, "refresh_evaluation_workspace", True):
        return
    if not getattr(s, "session_writable", True):
        return
    if not _path_is_within(annotation_path, eval_root):
        print("[WARN] evaluation refresh skipped: annotation is outside "
              f"--eval-root: {annotation_path}")
        return
    try:
        from evaluation.eval_dataset_status import refresh_after_annotation
    except (ImportError, AttributeError) as exc:
        print("[WARN] evaluation refresh unavailable; expected "
              "evaluation.eval_dataset_status.refresh_after_annotation: "
              f"{exc}")
        return
    try:
        summary = refresh_after_annotation(
            root=eval_root,
            annotation_path=annotation_path,
            image_path=image_path,
            deleted=bool(deleted),
        )
    except Exception as exc:
        print(f"[WARN] evaluation manifest/progress refresh failed: {exc}")
        return
    if summary:
        print(str(summary))


def _display_to_canvas(x, y, s):
    """마우스 좌표를 그대로 쓴다 — 스케일 보정을 하면 안 된다.

    WINDOW_NORMAL 로 창을 리사이즈하면 마우스 콜백이 "화면에 표시된 픽셀" 좌표를 줄
    것 같지만, 실제로는 OpenCV 가 이미 이미지(캔버스) 좌표로 변환해서 준다.
    2026-08-15 실측:

        raw=(863,890)  win=(1131x857)  canvas=(1320x1000)
                 ^^^ 창 높이 857 을 넘는 값이 온다 = 이미 캔버스 좌표

    여기에 canvas/win 비를 한 번 더 곱했다가 클릭이 오른쪽·아래로 밀렸다(1039 > 1000).
    보정은 필요 없다. 이 함수는 그 사실을 남겨 두려고 남긴다.
    """
    if os.environ.get("ANNOT_DEBUG_XY"):
        shp = getattr(s, "disp_shape", None)
        try:
            _, _, rw, rh = cv2.getWindowImageRect(WIN)
        except cv2.error:
            rw = rh = -1
        print(f"[xy] ({x},{y})  win=({rw}x{rh}) canvas={shp}")
    return x, y


def on_mouse(event, x, y, flags, s: State):
    """L click = keypoint set + active advance.  R click = delete.
    TWO-LINE 모드 시 4 클릭으로 교점 계산해서 active kp 위치 결정."""
    x, y = _display_to_canvas(x, y, s)
    s.last_mouse = (x, y)

    # CONDITIONS is a fully keyboard-modal editor.  Stop before both the image
    # canvas and panel hit-tests so neither keypoints nor navigation widgets can
    # change while the condition/tag editor is active.
    if getattr(s, "condition_mode", False):
        return

    # panel 영역 (확장 캔버스 우측) 클릭은 무시.
    # render() 가 [확장캔버스 | panel] 을 hstack — 확장캔버스 폭 = image_w + MARGIN_L + MARGIN_R.
    # zoom 후에도 확장캔버스는 원래 폭으로 resize 되므로 panel 경계는 항상 canvas_w.
    if s.img is not None:
        canvas_w = s.img.shape[1] + MARGIN_L + MARGIN_R
        if x >= canvas_w:
            # 패널 영역 — ANNOT-ONLY 버튼 클릭만 처리, 나머지는 무시.
            if event == cv2.EVENT_LBUTTONDOWN:
                canvas_h = s.img.shape[0] + MARGIN_T + MARGIN_B
                px, py = x - canvas_w, y
                bx0, by0, bx1, by1 = annot_button_rect(canvas_h)
                if bx0 <= px <= bx1 and by0 <= py <= by1:
                    if not getattr(s, "session_writable", True):
                        _toast(
                            s,
                            "REVIEW ONLY: no annotations",
                            (40, 40, 230),
                            log="[REVIEW ONLY] incoming 세션에는 ANNOT-ONLY 대상 GT가 없습니다.",
                            seconds=1.2,
                        )
                        return
                    s.annot_only = not s.annot_only
                    print(f"[Annot-only] {'ON' if s.annot_only else 'OFF'}"
                          f"  (n/p 로 어노된 frame 만 이동)")
                    return
                sx0, sy0, sx1, sy1 = session_button_rect(canvas_h)
                if sx0 <= px <= sx1 and sy0 <= py <= sy1:
                    s.sess_open = True      # 메인 루프가 목록을 채워 연다
            return
    if not getattr(s, "session_writable", True):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            _toast(
                s,
                "REVIEW ONLY: raw browse",
                (40, 40, 230),
                log="[REVIEW ONLY] incoming mixed frame에는 keypoint를 기록하지 않습니다.",
                seconds=1.2,
            )
        return
    # MANIPULATE 모드에서는 마우스 클릭으로 점 안 찍음.
    if s.mode != "click":
        return

    # screen → 확장캔버스 좌표 (zoom/pan 역변환) → image 좌표 (margin offset 제거).
    # image 밖 (u<0, v>480 등) 도 정상 — 여백에서 클릭한 코너의 실제 픽셀 좌표.
    cu = (x / s.zoom) + s.pan[0]
    cv = (y / s.zoom) + s.pan[1]
    u = cu - MARGIN_L
    v = cv - MARGIN_T

    # TWO-LINE sub-mode
    if s.line_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            if s.line_pts is None:
                s.line_pts = []
            s.line_pts.append([float(u), float(v)])
            if len(s.line_pts) == 4:
                pt = line_intersection(s.line_pts[0], s.line_pts[1],
                                       s.line_pts[2], s.line_pts[3])
                if pt is not None:
                    target = s.active
                    s.kps_2d[target] = pt
                    if s.extrap_mask is not None:
                        s.extrap_mask[target] = True   # v7: t 외삽 표시
                    _set_keypoint_state(
                        s, target, pt, source="extrapolated",
                        visibility=1, reason="unknown")
                    _mark_annotation_dirty(s)
                    if s.active < 8:
                        s.active += 1
                    print(f"[Line] intersection → kp{s.active-1 if s.active>0 else 0}: "
                          f"({pt[0]:.1f}, {pt[1]:.1f})")
                else:
                    print("[Line] 평행선 — 교점 없음. 다시 시도하세요.")
                s.line_mode = False
                s.line_pts = []
        elif event == cv2.EVENT_RBUTTONDOWN:
            if s.line_pts:
                s.line_pts.pop()
        return

    # 일반 CLICK 모드
    if event == cv2.EVENT_LBUTTONDOWN:
        target = s.active
        s.kps_2d[target] = [float(u), float(v)]
        if s.extrap_mask is not None:
            s.extrap_mask[target] = False    # v7: 직접 클릭 = 외삽 아님
        _set_keypoint_state(
            s, target, s.kps_2d[target], source="manual_click",
            visibility=2, reason="visible")
        _mark_annotation_dirty(s)
        if s.active < 8:
            s.active += 1
    elif event == cv2.EVENT_RBUTTONDOWN:
        if s.kps_2d[s.active] is not None:
            s.kps_2d[s.active] = None
            if s.extrap_mask is not None:
                s.extrap_mask[s.active] = False
            _clear_keypoint_state(s, s.active)
            s.pose = None
            _mark_annotation_dirty(s)


def _pose_inputs_key(s: State, K):
    """pose 를 결정하는 입력들의 지문. 바뀌지 않았으면 다시 풀 필요가 없다."""
    kps = tuple(None if p is None else (float(p[0]), float(p[1])) for p in (s.kps_2d or ()))
    ex = tuple(bool(b) for b in (s.extrap_mask or ()))
    lp = s.locked_pose
    lk = None if lp is None else (np.asarray(lp["R"]).tobytes(),
                                  np.asarray(lp["t"]).tobytes(),
                                  tuple(lp.get("dims") or ()))
    physical = _state_physical_dimensions(s)
    physical_key = (
        float(physical.x_m), float(physical.y_m), float(physical.z_m))
    return (kps, ex, s.mode, lk, s.img_shape, K.tobytes(), physical_key,
            getattr(s, "camera_facing_hypothesis_override", None))


def _empty_session_screen(seq_name, args):
    """빈 세션일 때 띄우는 안내 화면. 검은 창만 보이면 멈춘 것처럼 보인다."""
    img = np.zeros((240, 720, 3), dtype=np.uint8)
    for i, line in enumerate([
            f"'{seq_name[:40]}' : no frames",
            f"stride={args.stride}  start={args.start}",
            "session slider / [ ] / TAB = another session",
            "q = quit"]):
        cv2.putText(img, line, (20, 50 + i * 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def update_pose(s: State, K, force=False):
    """현재 mode 에 따라 pose 재계산. MANIPULATE 모드면 locked_pose 직접 사용.

    렌더 루프가 매 프레임(20ms 주기) 호출하는데 solve_pose 는 실측 87~186ms 라
    UI 가 5 FPS 로 떨어졌다. 입력이 그대로면 결과도 그대로이므로 건너뛴다.
    """
    if not getattr(s, "session_writable", True):
        s.pose = None
        s.locked_pose = None
        s._pose_key = None
        return
    if not force:
        key = _pose_inputs_key(s, K)
        if key == getattr(s, "_pose_key", object()):
            return
        s._pose_key = key
    if s.mode == "manip" and s.locked_pose is not None:
        s.pose = pose_from_locked(s, K)
    else:
        # v7: t/x 외삽 점 weight 0.3 + degenerate cuboid reject (img_shape 기반)
        s.pose = solve_pose(s.kps_2d, K,
                            extrapolated_mask=s.extrap_mask,
                            img_shape=s.img_shape,
                            weight_extrapolated_in_refine=True,
                            physical_dimensions=_state_physical_dimensions(s),
                            camera_facing_hypothesis_override=getattr(
                                s, "camera_facing_hypothesis_override", None))
    _sync_axis_candidates(s)
    if getattr(s, "eval_session_dir", None):
        _refresh_effective_frame_tags(s, use_state_evidence=True)


# ─── Key dispatchers ──────────────────────────────────────────────────────────

def _handle_manip_key(key, s, out_json, out_png, src_png, K):
    """MANIPULATE 모드 키 처리. Returns: 'next' | 'quit' | None."""
    if not getattr(s, "session_writable", True):
        s.mode = "click"
        s.locked_pose = None
        _toast(
            s,
            "REVIEW ONLY: manipulate disabled",
            (40, 40, 230),
            log="[REVIEW ONLY] incoming mixed frame에서는 pose를 편집하지 않습니다.",
        )
        return None
    ts = s.trans_step
    rs = s.rot_step_deg
    if key == ord('b'):
        _cycle_visibility_reason(s, s.active)
    elif key == ord('a'): apply_manip(s, dx=-ts)
    elif key == ord('d'): apply_manip(s, dx=+ts)
    elif key == ord('w'): apply_manip(s, dy=-ts)
    elif key == ord('x'): apply_manip(s, dy=+ts)
    elif key == ord('q'): apply_manip(s, dz=-ts)
    elif key == ord('e'): apply_manip(s, dz=+ts)
    elif key == ord('j'): apply_manip(s, dyaw=-rs)
    elif key == ord('l'): apply_manip(s, dyaw=+rs)
    elif key == ord('i'): apply_manip(s, dpitch=-rs)
    elif key == ord('k'): apply_manip(s, dpitch=+rs)
    elif key == ord('u'): apply_manip(s, droll=-rs)
    elif key == ord('o'): apply_manip(s, droll=+rs)
    elif key == ord('1'):
        s.trans_step = max(0.001, s.trans_step / 2.0)
        print(f"[step] trans={s.trans_step*100:.2f}cm")
    elif key == ord('2'):
        s.trans_step = min(0.5, s.trans_step * 2.0)
        print(f"[step] trans={s.trans_step*100:.2f}cm")
    elif key == ord('3'):
        s.rot_step_deg = max(0.5, s.rot_step_deg / 2.0)
        print(f"[step] rot={s.rot_step_deg:.2f}\xb0")
    elif key == ord('4'):
        s.rot_step_deg = min(45, s.rot_step_deg * 2.0)
        print(f"[step] rot={s.rot_step_deg:.2f}\xb0")
    elif key == ord('s'):
        if s.pose is None:
            return None
        # locked_pose 의 projected_cuboid 를 그대로 manual_kps 로 덮어쓰기
        proj = s.pose["projected_all"]
        s.kps_2d = [None if (p[0] == -1 and p[1] == -1) else list(p) for p in proj]
        for i, point in enumerate(s.kps_2d):
            if point is None:
                _clear_keypoint_state(s, i)
            else:
                _set_keypoint_state(
                    s, i, point,
                    source="centroid_auto" if i == 8 else "pnp_projected",
                    visibility=1, reason="unknown")
        _mark_annotation_dirty(s)
        if not _save_state_annotation(s, K, out_json, out_png, src_png):
            return None
        print(f"[Saved manip] {out_json}  reproj={s.pose['reproj_error_px']:.2f}px")
        s.dirty = False
        s.mode = "click"
        s.locked_pose = None
        return 'save-next'
    return None


def _toast(s, screen_text, color=(60, 200, 60), log=None, seconds=2.5):
    """화면 위에 잠깐 뜨는 알림. 터미널 print 만으로는 쓰는 사람이 못 본다.

    이 툴의 안내가 전부 stdout 으로만 나가서, 실제로는 동작했는데도 "아무 일도
    안 일어난다" 로 보였다(2026-08-16 삭제 기능에서 실제로 겪음).

    ★ screen_text 는 **반드시 ASCII** 여야 한다. cv2.putText 의 Hershey 폰트에는
      한글 glyph 가 없어서 한글을 넘기면 통째로 '?????' 로 그려진다(실제로 겪음).
      화면은 영문, 터미널(log)은 한글 — 그래서 둘을 나눠 받는다.
    """
    s.toast = (screen_text, color, time.time() + float(seconds))
    print(log if log is not None else screen_text)


def _delete_annotation(s, out_json, out_png):
    """점을 다 지운 뒤 's' — 이 프레임의 저장된 어노를 없앤다.

    JSON 은 완전히 지우지 않고 `.json.deleted` 로 옮긴다. `_has_annot`/done 집계는
    `.json` 만 보므로 "없어진" 것으로 정확히 취급되고, 잘못 눌렀으면 확장자만 떼면
    되돌아온다. 그 하나로 충분해서 확인 절차는 두지 않는다 — 처음엔 "한 번 더 s" 를
    요구하게 만들었는데, 쓰는 사람 입장에선 그게 그냥 "안 지워진다" 였다.
    PNG 사본은 지운다(원본이 촬영 폴더에 그대로 있어 언제든 다시 만들어진다).

    일반 FINAL population은 삭제를 허용하지 않는다. 단, ``--eval-root``로 지정한
    workspace의 ``final/positive/annotations`` 아래 label은 progress를
    되돌릴 수 있도록 복구 가능한 ``.deleted`` 이동을 허용한다.
    """
    if not getattr(s, "session_writable", True):
        _toast(
            s,
            "REVIEW ONLY: delete disabled",
            (40, 40, 230),
            log="[REVIEW ONLY] incoming 세션에는 삭제할 canonical GT가 없습니다.",
        )
        return None
    is_final = str(getattr(s, "population_role", "DEV")).upper() == "FINAL"
    eval_root = getattr(s, "eval_root", None)
    workspace_final_root = (
        os.path.join(eval_root, "final", "positive", "annotations")
        if eval_root else None)
    is_workspace_final = bool(
        is_final and workspace_final_root
        and _path_is_within(out_json, workspace_final_root))
    if is_final and not is_workspace_final:
        _toast(s, "[DELETE BLOCKED] FINAL population", (40, 40, 230),
               log="FINAL 삭제 차단: visibility 검토를 통과한 라벨은 "
                   "--eval-root의 final/positive workspace 밖에서는 "
                   "annotation UI로 제거할 수 없습니다.")
        return None
    try:
        _require_nonlegacy_output_dir(os.path.dirname(out_json), _REPO)
    except ValueError as exc:
        _toast(s, "[DELETE BLOCKED] legacy GT is read-only", (40, 40, 230),
               log=f"legacy GT 삭제 차단: {exc}: {out_json}")
        return None
    if not os.path.exists(out_json):
        _toast(s, "[DELETE] no saved annotation on this frame", (180, 180, 180),
               log="[삭제] 이 프레임엔 저장된 어노가 없다")
        s.dirty = False
        s.annotation_dirty = False
        s.frame_tags_dirty = False
        return None

    bak = out_json + ".deleted"
    try:
        os.replace(out_json, bak)
    except OSError as e:
        _toast(s, f"[ERROR] delete failed: {e}", (60, 60, 220),
               log=f"[ERROR] 삭제 실패: {e}")
        return None
    if os.path.exists(out_png):
        try:
            os.remove(out_png)
        except OSError as e:
            print(f"[WARN] PNG 사본은 못 지웠다: {e}")
    overlay_path = annotation_overlay_path(out_json)
    if os.path.exists(overlay_path):
        try:
            os.remove(overlay_path)
        except OSError as e:
            print(f"[WARN] overlay delete failed for {overlay_path}: {e}")

    _refresh_evaluation_workspace(
        s, out_json, image_path=None, deleted=True)

    name = os.path.basename(out_json)
    _toast(s, f"[DELETED] {name}  (restore: drop .deleted)",
           log=f"[삭제됨] {name}  (.deleted 로 복구 가능)")
    s.dirty = False
    s.annotation_dirty = False
    s.frame_tags_dirty = False
    return 'save-next'


def _handle_click_key(key, s, out_json, out_png, src_png, K):
    """CLICK 모드 키 처리. Returns: 'next' | 'prev' | 'quit' | None."""
    if key == ord('q'):
        # 저장 안 한 클릭이 있으면 한 번 막는다. 'n' 에만 이 보호가 있어서 q/p 로는
        # 찍던 걸 그냥 잃었다(2026-08-15). 같은 규칙으로 통일 — 다시 누르면 진행.
        if not _confirm_discard(
                s, "quit",
                "[WARN] 미저장 변경 있음. 저장하려면 's', 버리려면 'q' 를 한 번 더."):
            return None
        return 'quit'

    if not getattr(s, "session_writable", True):
        # REVIEW ONLY keeps the raw-browser controls, while every mutation is
        # a visible no-op.  Goto and session switching are handled by the outer
        # loop before this dispatcher.
        if key == ord('n'):
            return 'next'
        if key == ord('p'):
            return 'prev'
        if key == ord(','):
            return 'jump-10'
        if key == ord('.'):
            return 'jump+10'
        if key in (ord('+'), ord('=')):
            s.zoom = min(4.0, s.zoom * 1.5)
            return None
        if key in (ord('-'), ord('_')):
            s.zoom = max(1.0, s.zoom / 1.5)
            if s.zoom <= 1.001:
                s.pan = [0, 0]
            return None
        if key == ord('h'):
            s.pan[0] -= 20
            return None
        if key == ord('l'):
            s.pan[0] += 20
            return None
        if key == ord('k'):
            s.pan[1] -= 20
            return None
        if key == ord('j'):
            s.pan[1] += 20
            return None
        _toast(
            s,
            "REVIEW ONLY: navigation only",
            (40, 40, 230),
            log="[REVIEW ONLY] incoming 세션에서는 탐색 키만 사용할 수 있습니다.",
            seconds=1.2,
        )
        return None

    if key == _CONDITION_MODE_KEY:
        _set_condition_mode(s, True)
        return None

    if key == ord('v'):
        s.split = "train" if s.split == "eval" else "eval"
        _mark_annotation_dirty(s)
        print(f"[Split] this frame -> {s.split.upper()}  (저장 시 JSON 에 반영)")
        return None

    if key == ord('b'):
        _cycle_visibility_reason(s, s.active)
        return None

    if key == ord('w'):
        _cycle_wd_parity(s)
        return None

    if key == ord('s'):
        # 점을 다 지운 뒤 's' = "이 프레임 어노를 없앤다". 예전엔 pose 가 None 이라
        # 저장 가드에 막혀서, 잘못 찍은 프레임의 GT 를 툴 안에서 지울 방법이 없었다.
        if not any(k is not None for k in s.kps_2d):
            return _delete_annotation(s, out_json, out_png)
        if s.pose is None:
            print(f"[WARN] PnP 실패 — 최소 4점 필요 (현재 "
                  f"{sum(1 for k in s.kps_2d if k is not None)}점). 저장 안 됨. "
                  f"어노를 없애려면 'r' 로 전부 지운 뒤 's'.")
            return None
        # manual_kps 는 사용자 클릭 그대로 저장 (위치 안 옮김).
        # swap 보정은 라벨링 후 fix_manual_swap.py 후처리.
        if not _save_state_annotation(s, K, out_json, out_png, src_png):
            return None
        print(f"[Saved] {out_json}  reproj={s.pose['reproj_error_px']:.2f}px")
        s.dirty = False
        return 'save-next'

    if key == ord('f'):
        # Front-only 자동 저장: 0~3 만 클릭한 상태에서 PnP projection 으로 4~7 채움.
        # cargo 가 rear face 가린 시퀀스용 단축키.
        if s.pose is None:
            print("[WARN] PnP 실패 — 0~3 4점 모두 필요")
            return None
        # kps_2d 를 projection 으로 덮지 않는다. make_annotation 이 미클릭 idx 를
        # 이미 projection 으로 채우므로 결과 projected_cuboid 는 똑같고, manual_kps 에는
        # "사람이 찍은 점" 만 남아 자동채움과 구분된다(2026-08-15).
        n_auto = sum(1 for k in s.kps_2d[:8] if k is None)
        _mark_projected_fallbacks(s)
        if not _save_state_annotation(s, K, out_json, out_png, src_png):
            return None
        print(f"[Saved front-only] {out_json}  reproj={s.pose['reproj_error_px']:.2f}px "
              f"(click-only 기준 — 자동채움 {n_auto}점의 오차는 포함 안 됨)")
        s.dirty = False
        return 'save-next'

    if key == ord('g'):
        # Auto-fill 저장: 사용자가 클릭한 점은 그대로 두고, 미클릭 0~7 점은 PnP
        # projection 으로 채워서 저장. 8 (centroid) 도 PnP projection 으로 채움.
        # 저장 후 frame 은 안 넘김 — 사용자가 시각적 확인 후 'n' 직접 누름.
        # ★ Truncation 시 (예: 0,3 image 밖 → 012456 만 클릭) 도 동작:
        #    4+점 클릭 + PnP 풀이 가능 → 미클릭 idx 자동 채움.
        n_clicked_07 = sum(1 for k in s.kps_2d[:8] if k is not None)
        if n_clicked_07 < 4:
            print(f"[WARN] g: 0~7 중 4 점 이상 필요. 현재 {n_clicked_07}/8 점.")
            return None
        if s.pose is None:
            print("[WARN] g: PnP 실패 — 위치 확인 (face-flip strict reject 가능). "
                  "위치 미세조정 후 재시도.")
            return None
        # 0~7 + 8(centroid) 중 미클릭만 PnP projection 으로 채움. 사용자 클릭은 그대로.
        proj = s.pose["projected_all"]
        n_auto = 0
        for i in range(9):
            if s.kps_2d[i] is None and not (proj[i][0] == -1 and proj[i][1] == -1):
                s.kps_2d[i] = list(proj[i])
                # 자동채움은 관측이 아니다. 표시하지 않으면 오차 0 인 "클릭" 으로 섞여
                # 표시 reproj 를 끌어내리고(1.56 -> 0.87px) 재풀이 때 full weight 로
                # 들어간다. extrap_mask 를 세우면 속 빈 원으로 그려지고 report 에서 빠진다.
                if s.extrap_mask is not None:
                    s.extrap_mask[i] = True
                _set_keypoint_state(
                    s, i, s.kps_2d[i],
                    source="centroid_auto" if i == 8 else "pnp_projected",
                    visibility=1, reason="unknown")
                n_auto += 1
        if n_auto:
            _mark_annotation_dirty(s)
        if not _save_state_annotation(s, K, out_json, out_png, src_png):
            return None
        print(f"[Saved auto-fill] {out_json}  reproj={s.pose['reproj_error_px']:.2f}px "
              f"({n_clicked_07} manual + {n_auto} auto-fill) — 시각 확인 후 'n' 으로 다음 frame")
        s.dirty = False
        return None   # ★ frame 안 넘김. 사용자 확인 후 'n' 직접 누름.

    if key == ord('x'):
        # Parallelogram 외삽: 활성 idx (0~7) 의 위치를 그 idx 가 속한 face 의 나머지 3
        # corner 로부터 외삽. truncation 시 (image 밖이라 클릭 불가) 단축키.
        # 예: 012 클릭 + active=3 + 'x' → 3 = 0 + (2 - 1) 자동.
        # 후보 face 여러 개면 평균. centroid (idx=8) 는 'c' 로 처리, 'x' 는 무효.
        if s.active >= 8:
            print("[Parallelogram] active=8 (centroid) — 'c' 키 사용")
            return None
        pt, fname, finds = parallelogram_extrapolate(s.kps_2d, s.active)
        if pt is None:
            print(f"[Parallelogram] kp{s.active} 외삽 실패 — 같은 face 의 다른 3 corner "
                  f"중 미클릭 있음. (face 후보: FRONT/BACK/TOP/BOTTOM/LEFT/RIGHT 중 "
                  f"kp{s.active} 포함 face 의 나머지 3 점 모두 필요)")
            return None
        s.kps_2d[s.active] = pt
        if s.extrap_mask is not None:
            s.extrap_mask[s.active] = True   # v7: x parallelogram 외삽 표시
        _set_keypoint_state(
            s, s.active, pt, source="extrapolated",
            visibility=1, reason="unknown")
        _mark_annotation_dirty(s)
        print(f"[Parallelogram] kp{s.active} ← face={fname} {finds} → "
              f"({pt[0]:.1f}, {pt[1]:.1f})")
        if s.active < 8:
            s.active += 1
        return None

    if key == ord('n'):
        if not _confirm_discard(
                s, "next",
                "[WARN] 미저장 변경 있음. 다시 'n' 누르면 무시하고 다음."):
            return None
        return 'next'
    if key == ord('p'):
        if not _confirm_discard(
                s, "prev",
                "[WARN] 미저장 변경 있음. 다시 'p' 누르면 무시하고 이전."):
            return None
        return 'prev'
    if key == ord(','):
        return 'jump-10'
    if key == ord('.'):
        return 'jump+10'

    if key in (ord('0'), ord('1'), ord('2'), ord('3'), ord('4'),
               ord('5'), ord('6'), ord('7'), ord('8')):
        s.active = key - ord('0')
        return None

    if key == ord('c'):
        p8 = None if s.pose is None else s.pose["projected_all"][8]
        if p8 is not None and not (p8[0] == -1 and p8[1] == -1):
            s.kps_2d[8] = list(p8)
            # PnP 자신의 centroid 투영이다. 관측으로 취급하면 오차 0 인 점이 다음 solve 에
            # 들어가 표시 reproj 를 낮추고 이후 코너 수정의 효과를 둔하게 만든다.
            if s.extrap_mask is not None:
                s.extrap_mask[8] = True
            _set_keypoint_state(
                s, 8, s.kps_2d[8], source="centroid_auto",
                visibility=1, reason="unknown")
            _mark_annotation_dirty(s)
            print(f"[Centroid] PnP projection: ({s.kps_2d[8][0]:.1f}, {s.kps_2d[8][1]:.1f})")
        else:
            pts = [k for k in s.kps_2d[:8] if k is not None]
            if len(pts) >= 4:
                s.kps_2d[8] = [float(np.mean([p[0] for p in pts])),
                               float(np.mean([p[1] for p in pts]))]
                _set_keypoint_state(
                    s, 8, s.kps_2d[8], source="centroid_auto",
                    visibility=1, reason="unknown")
                _mark_annotation_dirty(s)
                print("[Centroid] fallback (image corner mean) — PnP 풀린 후 c 권장")
        return None

    if key == ord('z'):
        if s.line_mode and s.line_pts:
            s.line_pts.pop()
        else:
            # "active-1 을 지운다" 로는 숫자키로 idx 를 옮긴 뒤 엉뚱한 점이 지워진다
            # (예: 0~5 찍고 '2' 로 이동 후 z -> kp1 이 사라짐). undo 는 "마지막으로 찍힌
            # 점" 을 되돌리는 것이므로 채워진 것 중 가장 큰 idx 를 지운다.
            last = max((i for i, k in enumerate(s.kps_2d) if k is not None), default=None)
            if last is None:
                return None                     # 지울 게 없으면 dirty 도 세우지 않는다
            s.kps_2d[last] = None
            if s.extrap_mask is not None:
                s.extrap_mask[last] = False
            _clear_keypoint_state(s, last)
            s.active = last
            _mark_annotation_dirty(s)
        return None

    if key == ord('d'):
        if s.kps_2d[s.active] is not None:
            s.kps_2d[s.active] = None
            if s.extrap_mask is not None:
                s.extrap_mask[s.active] = False
            _clear_keypoint_state(s, s.active)
            s.pose = None
            _mark_annotation_dirty(s)
            print(f"[Delete] kp{s.active} 삭제")
        return None

    if key == ord('t'):
        if s.line_mode:
            s.line_mode = False
            s.line_pts = []
            print("[Line] 취소")
        else:
            s.line_mode = True
            s.line_pts = []
            print(f"[Line] kp{s.active} 위치 추정 — 4 번 클릭: line1-A, B, line2-A, B")
        return None

    if key == ord('r'):
        # 이미 빈 프레임이면 바꾼 게 없다. 여기서 dirty 를 세우면 다음 이동이 한 번
        # 막혀서 "n 이 안 먹는다" 로 보인다.
        had = any(k is not None for k in s.kps_2d)
        s.kps_2d = [None] * 9
        s.extrap_mask = [False] * 9
        s.keypoint_annotations = None
        _ensure_keypoint_annotations(s)
        s.active = 0
        s.line_mode = False
        s.line_pts = []
        if had:
            _mark_annotation_dirty(s)
        return None

    if key in (ord('+'), ord('=')):
        old_z = s.zoom
        s.zoom = min(4.0, s.zoom * 1.5)
        if s.last_mouse and s.img is not None:
            # pan/zoom 은 확장 캔버스 기준 (render 의 crop 도 확장 캔버스 = vis 에 작동).
            h = s.img.shape[0] + MARGIN_T + MARGIN_B
            w = s.img.shape[1] + MARGIN_L + MARGIN_R
            cx = s.pan[0] + (s.last_mouse[0] / old_z)
            cy = s.pan[1] + (s.last_mouse[1] / old_z)
            s.pan[0] = int(cx - (w / s.zoom) / 2)
            s.pan[1] = int(cy - (h / s.zoom) / 2)
        return None
    if key in (ord('-'), ord('_')):
        s.zoom = max(1.0, s.zoom / 1.5)
        if s.zoom <= 1.001:
            s.pan = [0, 0]
        return None

    if key == ord('h'): s.pan[0] -= 20
    elif key == ord('l'): s.pan[0] += 20
    elif key == ord('k'): s.pan[1] -= 20
    elif key == ord('j'): s.pan[1] += 20
    return None


# ─── Main loop ────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq",     default="data/outside/capturepallet02")
    ap.add_argument("--out_dir", default=None,
                    help="기본: challenge/data/01_real/gt_v2_canonical/... . "
                         "legacy manual_gt/eval_canonical 경로는 명시해도 거부됨")
    ap.add_argument("--out-root", "--out_root", default=None,
                    help="세션마다 <root>/<세션명>_manual_gt 로 저장할 상위 폴더. "
                         "--out_dir 과 달리 세션 목록을 한 개로 좁히지 않는다.")
    ap.add_argument("--stride",  type=int, default=30, help="N frame 마다 1개 annotate")
    ap.add_argument("--start",   type=int, default=0, help="시작 frame idx")
    ap.add_argument("--pool", nargs="+",
                    default=["data/pallet/raw_data/outside", "data/pallet/raw_data/night"],
                    help="세션을 찾을 폴더들. 툴 안에서 [ ] 키로 세션 전환, TAB 으로 목록 선택.")
    ap.add_argument("--fixed-window", action="store_true",
                    help="창 크기를 고정(예전 동작). 마우스 좌표가 어긋나면 이걸로 되돌린다.")
    ap.add_argument("--win-w", type=int, default=1180, help="창 초기 가로 (조절 가능)")
    ap.add_argument("--win-h", type=int, default=880, help="창 초기 세로 (조절 가능)")
    ap.add_argument("--default_split", choices=["eval", "train"], default="train",
                    help="새 frame 의 기본 split (v 키로 토글). 기본 train — eval 로 쓸 것만 v "
                         "로 표시. eval GT 대량 어노 시 --default_split eval.")
    ap.add_argument("--population-role", "--population_role", default=None,
                    choices=["DEV", "FINAL"],
                    help="GT population 역할. eval은 session.json에서 자동, 일반 실행은 필수.")
    ap.add_argument("--object-type", default=None,
                    help="OBJECT_GEOMETRY_REGISTRY type/alias. eval은 session.json, "
                         "일반 실행은 미지정 시 plastic")
    ap.add_argument("--geometry-registry", default=str(DEFAULT_REGISTRY_PATH),
                    help="object geometry registry JSON (기본: repository contract)")
    ap.add_argument("--legacy-read-dir", default=None,
                    help="기존 JSON을 읽기만 할 legacy manual_gt/eval_canonical 폴더")
    ap.add_argument(
        "--intrinsics-quality", default=None,
        choices=["CALIBRATED", "SENSOR_PROFILE_SCALED", "ESTIMATED_HFOV", "UNKNOWN"],
        help="wood에는 필수; plastic 기존 출력은 생략 가능")
    ap.add_argument("--intrinsics-source", default=None)
    ap.add_argument("--capture-session-id", "--capture_session_id", default=None)
    ap.add_argument("--camera-serial", "--camera_serial", default=None)
    ap.add_argument("--capture-timestamp", "--capture_timestamp", default=None)
    ap.add_argument("--lighting-condition", "--lighting_condition", default=None)
    ap.add_argument(
        "--eval-root", default=None,
        help="evaluation workspace root; enables per-save manifest/progress refresh")
    args = ap.parse_args(argv)
    # Session metadata is filled into ``args`` below.  Preserve the actual CLI
    # values so a sibling session is resolved from its own metadata instead of
    # inheriting lighting/session id from whichever session opened first.
    cli_session_args = copy.copy(args)

    registry_path = (args.geometry_registry
                     if os.path.isabs(args.geometry_registry)
                     else os.path.join(_REPO, args.geometry_registry))
    try:
        geometry_registry = load_object_geometry_registry(registry_path)
    except (OSError, TypeError, ValueError) as exc:
        ap.error(f"invalid object geometry registry: {exc}")
    legacy_read_dir = None
    if args.legacy_read_dir:
        try:
            legacy_read_dir = _require_legacy_read_dir(
                args.legacy_read_dir, _REPO)
        except ValueError as exc:
            ap.error(str(exc))
    eval_root = None
    if args.eval_root:
        eval_root = (args.eval_root if os.path.isabs(args.eval_root)
                     else os.path.join(_REPO, args.eval_root))
        eval_root = os.path.abspath(eval_root)
        if not os.path.isdir(eval_root):
            ap.error(f"--eval-root does not exist or is not a directory: {eval_root}")

    requested_out = None
    if args.out_dir:
        requested_out = (args.out_dir if os.path.isabs(args.out_dir)
                         else os.path.join(_REPO, args.out_dir))
        try:
            _require_nonlegacy_output_dir(requested_out, _REPO)
        except ValueError as exc:
            ap.error(str(exc))

    seq = args.seq if os.path.isabs(args.seq) else os.path.join(_REPO, args.seq)
    seq = os.path.abspath(seq)
    try:
        session_metadata, geometry_spec = _resolve_annotation_configuration(
            args, seq, geometry_registry, eval_root)
    except (OSError, TypeError, ValueError, WorkspaceError) as exc:
        ap.error(str(exc))
    if (geometry_spec.object_type == WOOD_OBJECT_TYPE
            and args.intrinsics_quality is None):
        ap.error("wood annotation requires --intrinsics-quality in CLI or session.json")
    if eval_root:
        if requested_out is None:
            ap.error("--eval-root requires an explicit canonical --out_dir")
        try:
            seq, requested_out = _validate_evaluation_paths(
                eval_root, seq, requested_out, args.population_role)
        except ValueError as exc:
            ap.error(str(exc))
    seq_name = os.path.basename(seq.rstrip("/\\"))

    # 세션 풀 — 일반 --out_dir은 여전히 단일 세션으로 잠근다. Evaluation
    # workspace는 각 세션의 role/object/K/output을 독립 검증하므로 같은 role의
    # plastic/wood를 함께 편집할 수 있다. INCOMING_UNREVIEWED는 원본을 유지한
    # 채 객체별 zero-copy STAGING context 두 개로 연다.
    evaluation_contexts = {}
    session_output_dirs = {}
    if eval_root:
        try:
            sessions, evaluation_contexts = _discover_evaluation_session_pool(
                eval_root, seq, requested_out, cli_session_args,
                geometry_registry, _REPO, args.population_role,
                geometry_spec.object_type)
        except (OSError, TypeError, ValueError, WorkspaceError) as exc:
            ap.error(str(exc))
        session_output_dirs = {
            key: context["out_dir"]
            for key, context in evaluation_contexts.items()
        }
    else:
        sessions = discover_sessions(args.pool, _REPO)
        if args.out_dir or not sessions:
            sessions = [(seq_name, seq)]
        elif all(_session_entry_parts(entry)[1] != seq for entry in sessions):
            sessions.insert(0, (seq_name, seq))
        # 세션 목록(TAB)의 GT 개수는 이 map 을 본다. 비워 두면 resolve_out_dir 이
        # 정본 경로를 돌려줘 실제로 저장한 라벨이 done 0 으로 보인다.
        if args.out_root:
            _root = (args.out_root if os.path.isabs(args.out_root)
                     else os.path.join(_REPO, args.out_root))
            session_output_dirs = {
                _session_entry_parts(entry)[2]:
                    os.path.join(_root, f"{_session_entry_parts(entry)[0]}_manual_gt")
                for entry in sessions
            }
        elif args.out_dir:
            _od = (args.out_dir if os.path.isabs(args.out_dir)
                   else os.path.join(_REPO, args.out_dir))
            session_output_dirs = {
                _session_entry_parts(entry)[2]: _od for entry in sessions
            }
    sess_i = next((
        i for i, entry in enumerate(sessions)
        if _session_entry_parts(entry)[1] == seq
        and len(entry) == 2
    ), 0)
    active_context = None

    def load_session(i):
        """세션 i 로 전환. FINAL 표시는 명시적 population role만 따른다."""
        nonlocal session_metadata, geometry_spec, active_context
        nm, sq, context_key = _session_entry_parts(sessions[i])
        if eval_root:
            context = evaluation_contexts[context_key]
            active_context = context
            session_args = context["args"]
            for field in _SESSION_RUNTIME_ARG_FIELDS:
                setattr(args, field, getattr(session_args, field))
            session_metadata = dict(context["metadata"])
            geometry_spec = context["geometry_spec"]
            od = context["out_dir"]
        elif args.out_dir:
            active_context = {
                "writable": True,
                "workspace_scope": None,
                "display_role": args.population_role,
            }
            od = args.out_dir if os.path.isabs(args.out_dir) \
                else os.path.join(_REPO, args.out_dir)
        elif args.out_root:
            active_context = {
                "writable": True,
                "workspace_scope": None,
                "display_role": args.population_role,
            }
            root = (args.out_root if os.path.isabs(args.out_root)
                    else os.path.join(_REPO, args.out_root))
            od = os.path.join(root, f"{nm}_manual_gt")
        else:
            active_context = {
                "writable": True,
                "workspace_scope": None,
                "display_role": args.population_role,
            }
            od, _legacy_eval_layout = resolve_out_dir(nm, _REPO)
        od = _require_nonlegacy_output_dir(od, _REPO)
        sealed = args.population_role == "FINAL"
        # 여기서 makedirs 하면 세션 목록을 둘러보기만 해도 빈 GT 폴더가 생겨
        # done 집계와 discover_sessions 의 중복 판정이 오염된다. 저장할 때 만든다.
        if eval_root:
            k = np.asarray(context["K"], dtype=np.float64).copy()
            k_source = context["K_source"]
        else:
            k, k_source = _resolve_session_intrinsics(
                sq, od, evaluation=False)
        rp = list(active_context.get("frame_paths") or _session_image_paths(sq))
        sel = list(range(args.start, len(rp), args.stride))
        writable = bool(active_context.get("writable", True))
        staging = bool(
            writable and not active_context.get("active_evaluation_member", True))
        print(f"\n[Session {i+1}/{len(sessions)}] {nm}"
              f"{'   ★FINAL population role — save gates active' if sealed else ''}"
              f"{'   [REVIEW ONLY · mixed raw capture]' if not writable else ''}"
              f"{'   [STAGING · evaluation 미편입]' if staging else ''}")
        print(f"           {len(sel)} frames (stride={args.stride}) of {len(rp)}")
        print(f"           Output: {od if writable else 'NONE (review only)'}")
        if staging:
            print("           Save scope: object-specific staging only; "
                  "evaluation manifest/progress unchanged")
        if geometry_spec is None:
            print("           Object: MIXED / UNKNOWN (PnP disabled)")
        else:
            print(f"           Object: {geometry_spec.object_type} "
                  f"XYZ={geometry_spec.physical_dimensions_m}")
        print(f"           K = fx={k[0,0]:.1f} cx={k[0,2]:.1f} cy={k[1,2]:.1f}"
              f"  [{k_source}]")
        return sq, nm, od, sealed, k, rp, sel

    seq, seq_name, out_dir, sealed, K, rgb_paths, selected = load_session(sess_i)
    if not rgb_paths:
        print(f"[ERROR] no rgb frames in {seq}")
        return

    win = WIN
    if args.fixed_window:
        cv2.namedWindow(win)                       # 예전 동작: 크기 고정
    else:
        # 캔버스는 image(640x480) + 여백(200/200/200/320) + 패널(280) = 약 1320x1000 이라
        # 1080p 화면을 거의 채운다. 창이 화면보다 커지면 OpenCV 가 pan 모드로 들어가
        # 커서가 손 모양이 되고 클릭이 안 먹는다. 크기 조절 가능한 창으로 열고 초기값을
        # 화면보다 작게 잡아 그 상태를 피한다. (2026-08-15 사용자 요청)
        # WINDOW_GUI_NORMAL 이 핵심이다. 기본값인 WINDOW_GUI_EXPANDED 는 Qt 백엔드의 자체
        # 확대/이동 기능을 켜는데, 그게 켜져 있으면 휠 확대 후 드래그가 우리 pan 이 아니라
        # Qt 의 pan 으로 먹어서 (1) 커서가 손 모양이 되고 (2) 화면이 아니라 내용이 끌려가고
        # (3) 클릭 좌표가 확대 전 기준으로 들어와 엉뚱한 곳에 점이 찍힌다.
        # 확대는 툴 자체의 +/- 키(zoom/pan)로만 하도록 Qt 쪽을 꺼 둔다.
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(win, args.win_w, args.win_h)
    s = State()
    s.geometry_registry = geometry_registry
    s.geometry_spec = geometry_spec
    s.object_type = (
        geometry_spec.object_type if geometry_spec is not None else "unknown")
    s.intrinsics_quality = args.intrinsics_quality
    s.intrinsics_source = args.intrinsics_source
    # 저장 시 GT v2 스키마가 object_type 을 조회할 레지스트리. 지정하지 않으면
    # argparse 기본값이라 논문 정본이 그대로 쓰인다.
    s.geometry_registry_path = args.geometry_registry
    s.eval_root = eval_root
    s.eval_session_dir = (
        active_context.get("tag_session_dir", seq) if eval_root else None)
    s.source_session_dir = (
        active_context.get("source_session_dir", seq) if eval_root else seq)
    s.session_frame_paths = list(active_context.get("frame_paths") or rgb_paths)
    s.annotation_output_dir = out_dir
    s.active_evaluation_member = bool(
        active_context.get("active_evaluation_member", True))
    s.refresh_evaluation_workspace = bool(
        active_context.get("refresh_evaluation", True))
    s.force_explicit_object_type = bool(
        active_context.get("force_explicit_object_type", False))
    s.session_metadata = dict(session_metadata)
    s.session_writable = bool(active_context.get("writable", True))
    s.workspace_scope = active_context.get("workspace_scope")
    s.object_type_source = (
        "SESSION" if _metadata_value(session_metadata.get("object_type")) else "UNSET")
    s.lighting = args.lighting_condition or "unknown"
    s.lighting_source = (
        "SESSION" if _metadata_value(session_metadata.get("lighting")) else "UNSET")
    cv2.setMouseCallback(win, on_mouse, s)
    # frame 점프 슬라이더 — 눈금을 프레임 수가 아니라 **퍼센트(0~100)** 로 둔다.
    #
    # cv2.setTrackbarMax 는 Qt 백엔드에서 표시를 갱신하지 않는다(2026-08-15 재현:
    # max 60 -> 12 로 바꿔도 "(00/60)" 그대로). 그래서 세션마다 max 를 맞출 수가 없다.
    # 전 세션 최대치로 고정하고 읽은 값을 클램프해 봤더니 더 나빴다 — 61장짜리 세션에서
    # 슬라이더를 60 너머로 끌면 클램프된 값이 다시 setTrackbarPos 로 슬라이더를 되돌려,
    # 슬라이더가 튕기고 화면이 60 에 멈춘 채로 있었다.
    # 퍼센트면 세션 길이와 무관하게 항상 전 구간을 쓸 수 있고 max 를 바꿀 일도 없다.
    # 실제 프레임 번호는 우측 STATUS 의 "frame N/M" 이 보여준다.
    cv2.createTrackbar(
        "frame%", win, 0, _FRAME_SLIDER_TICKS, lambda v: None)
    # 세션 슬라이더 — 드래그하면 촬영 세션이 바뀐다. 키(TAB)가 창 포커스에 따라 안 먹는
    # 경우가 있어서, 이미 동작이 검증된 트랙바/마우스 방식을 주 수단으로 둔다.
    if len(sessions) > 1:
        cv2.createTrackbar("session", win, sess_i, len(sessions) - 1, lambda v: None)

    def _has_annot(ci):
        st = os.path.splitext(os.path.basename(rgb_paths[selected[ci]]))[0]
        return os.path.exists(os.path.join(out_dir, st + ".json"))

    def _step_annot(ci, d):
        """방향 d(+1/-1)로 어노된 다음 frame 인덱스. 없으면 제자리."""
        j = ci + d
        while 0 <= j < len(selected):
            if _has_annot(j):
                return j
            j += d
        return ci

    cur = 0
    # 루프 조건을 "cur 이 범위 안" 으로 두면, 마지막 프레임에서 저장(s)해 cur 이 하나 넘는
    # 순간 창이 조용히 닫힌다. 세션을 갈아탈 수 있게 된 뒤로는 그게 "이미지가 갑자기 안 보인다"
    # 로 나타난다(2026-08-15). 이제는 범위를 벗어나면 끝 프레임에 머무르고 이유를 알린다.
    # 종료는 q(quit) 로만 한다.
    while True:
        if cur >= len(selected):
            # 마지막에서 더 가려 하면 다음 세션으로 넘어간다. 예전엔 창이 그냥 닫혔고,
            # 그 다음엔 끝에 머물기만 해서 "n 이 안 먹는다" 로 보였다(2026-08-15).
            # 세션을 순서대로 훑는 게 이 툴의 실제 사용 방식이라 자동으로 넘긴다.
            if len(sessions) > 1:
                nxt = (sess_i + 1) % len(sessions)
                print(f"[끝] {seq_name} {len(selected)}장 끝 — 다음 세션 "
                      f"'{sessions[nxt][0]}' 으로 이동합니다. (되돌리려면 session 슬라이더)")
                sess_i = nxt
                seq, seq_name, out_dir, sealed, K, rgb_paths, selected = load_session(sess_i)
                cv2.setTrackbarPos("session", win, sess_i)
                cur = 0
            else:
                print(f"[끝] {seq_name} 마지막 프레임입니다 (총 {len(selected)}장).")
                cur = len(selected) - 1
        elif cur < 0:
            cur = 0
        if not selected:
            # 빈 세션에서 그냥 continue 하면 키를 하나도 안 읽어 q 도 안 먹는 영구 정지가
            # 된다(단일 세션일 때 실제로 그랬다). 여기서도 종료·세션이동은 되게 한다.
            print(f"[WARN] {seq_name}: stride={args.stride} start={args.start} 로 뽑히는 "
                  f"프레임이 없다. session 슬라이더로 다른 세션을 고르거나 q 로 종료.")
            while True:
                cv2.imshow(win, _empty_session_screen(seq_name, args))
                k = cv2.waitKey(50) & 0xFF
                if k in (ord('q'), 27):
                    cv2.destroyAllWindows()
                    return
                if len(sessions) > 1:
                    tb = cv2.getTrackbarPos("session", win)
                    if tb != sess_i:
                        sess_i = tb
                    elif k == ord(']'):
                        sess_i = (sess_i + 1) % len(sessions)
                    elif k in (ord('['), 9):
                        sess_i = (sess_i - 1) % len(sessions)
                    else:
                        continue
                    (seq, seq_name, out_dir, sealed, K,
                     rgb_paths, selected) = load_session(sess_i)
                    cv2.setTrackbarPos("session", win, sess_i)
                    cur = 0
                    break
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    return
            continue
        frame_idx = selected[cur]
        path = rgb_paths[frame_idx]
        stem = os.path.splitext(os.path.basename(path))[0]
        out_json = os.path.join(out_dir, f"{stem}.json")
        out_png  = os.path.join(out_dir, f"{stem}.png")

        # 프레임 reset + 기존 라벨 로드
        img = cv2.imread(path)
        if img is None:
            # 잘린 PNG·권한·삭제된 파일. 예전엔 여기서 AttributeError 로 창이 죽었다.
            print(f"[WARN] 이미지를 읽지 못했다: {path} — 다음 프레임으로")
            cur += 1
            continue
        s.img = img
        s.img_shape = s.img.shape
        s.kps_2d = [None] * 9
        s.extrap_mask = [False] * 9    # v7: 외삽 점 표시 (t/x 입력 시 True)
        s.keypoint_annotations = None
        _ensure_keypoint_annotations(s)
        s.axis_assignment = None
        s.axis_assignment_candidates = []
        s.axis_assignment_confirmed = False
        s.session_writable = bool(active_context.get("writable", True))
        s.workspace_scope = active_context.get("workspace_scope")
        s.population_role = (
            args.population_role
            or active_context.get("display_role")
            or "UNSET")
        s.geometry_registry = geometry_registry
        s.geometry_spec = geometry_spec
        s.object_type = (
            geometry_spec.object_type if geometry_spec is not None else "unknown")
        s.object_type_source = (
            "SESSION"
            if "object_type" in session_metadata else "UNSET")
        s.lighting = args.lighting_condition or "unknown"
        s.lighting_source = (
            "SESSION"
            if _metadata_value(session_metadata.get("lighting")) else "UNSET")
        s.loaded_object_type = None
        s.loaded_annotation_path = None
        s.current_annotation_path = os.path.abspath(out_json)
        s.current_frame_identity = os.path.basename(path)
        s.intrinsics_quality = (
            args.intrinsics_quality
            or active_context.get("intrinsics_quality"))
        s.intrinsics_source = (
            args.intrinsics_source
            or active_context.get("intrinsics_source"))
        s.eval_session_dir = (
            active_context.get("tag_session_dir", seq) if eval_root else None)
        s.source_session_dir = (
            active_context.get("source_session_dir", seq) if eval_root else seq)
        s.session_frame_paths = list(
            active_context.get("frame_paths") or rgb_paths)
        source_ordinals = active_context.get("source_ordinals")
        s.source_frame_ordinal = (
            int(source_ordinals[frame_idx])
            if source_ordinals is not None else None)
        s.source_frame_count = (
            int(active_context["source_frame_count"])
            if active_context.get("source_frame_count") is not None else None)
        s.annotation_output_dir = out_dir
        s.active_evaluation_member = bool(
            active_context.get("active_evaluation_member", True))
        s.refresh_evaluation_workspace = bool(
            active_context.get("refresh_evaluation", True))
        s.force_explicit_object_type = bool(
            active_context.get("force_explicit_object_type", False))
        s.session_metadata = dict(session_metadata)
        s.capture_metadata = {
            "capture_session_id": args.capture_session_id or seq_name,
            "camera_serial": args.camera_serial,
            "capture_timestamp": args.capture_timestamp,
            "lighting_condition": args.lighting_condition,
        }
        s.legacy_document = None
        s.legacy_object = None
        s.camera_facing_hypothesis_override = None
        s.occlusion_level = "unknown"
        s.active = 0
        s.pose = None
        s.zoom = 1.0
        s.pan = [0, 0]
        s.dirty = False
        s.annotation_dirty = False
        s.frame_tags_dirty = False
        s.frame_tags = {}
        s.frame_tag_sources = {}
        s.frame_tag_overrides = {}
        s.frame_tag_pending_updates = {}
        s.frame_tag_cycle_values = {}
        s.discard_armed = None
        # ★ 모드/입력 중 상태도 반드시 초기화한다. 예전엔 mode/locked_pose 가 남아,
        #   MANIPULATE 로 잠근 이전 프레임의 R,t 를 새 프레임에 그대로 투영해 manual GT 로
        #   저장할 수 있었다. 새 프레임은 kps_2d 가 비어 있어 reproj 가 0.00px 로 찍히는
        #   바람에 완벽한 라벨처럼 보였다. line/goto 상태도 남으면 클릭·키가 먹통이 된다.
        s.mode = "click"
        s.condition_mode = False
        s.locked_pose = None
        s.line_mode = False
        s.line_pts = None
        s.goto_mode = False
        s.goto_buf = ""
        s.goto = None
        s._pose_key = None             # 새 프레임 — pose 캐시 무효화
        s.toast = None                 # 알림이 다음 프레임으로 새지 않게
        s.split = args.default_split   # 기본 split; 기존 JSON 있으면 load 가 override
        if not s.session_writable:
            s.annot_only = False
        load_json = out_json
        load_is_read_only_legacy = False
        if s.session_writable and not os.path.exists(load_json):
            automatic_legacy_dir, _ = _resolve_legacy_read_dir(seq_name, _REPO)
            source_dir = legacy_read_dir or automatic_legacy_dir
            legacy_json = (os.path.join(source_dir, f"{stem}.json")
                           if source_dir else None)
            if legacy_json and os.path.exists(legacy_json):
                load_json = legacy_json
                load_is_read_only_legacy = True
                print(f"[Read-only legacy source] {legacy_json}\n"
                      f"                          save target: {out_json}")
        if (s.session_writable and load_existing_annotation(
                s, load_json, read_only=load_is_read_only_legacy)):
            loaded_role = str(s.population_role).strip().upper()
            if loaded_role != args.population_role:
                raise RuntimeError(
                    "annotation population role mismatch\n"
                    f"annotation = {loaded_role}\n"
                    f"session    = {args.population_role}\n"
                    f"path       = {load_json}")
            loaded_metadata = dict(s.capture_metadata or {})
            loaded_metadata.update({
                key: value for key, value in {
                    "capture_session_id": args.capture_session_id or seq_name,
                    "camera_serial": args.camera_serial,
                    "capture_timestamp": args.capture_timestamp,
                    "lighting_condition": args.lighting_condition,
                }.items() if value is not None
            })
            s.capture_metadata = loaded_metadata
            s.population_role = args.population_role
            _ensure_keypoint_annotations(s)
            update_pose(s, K)
        try:
            _load_frame_tag_state(s, path, out_json)
        except (OSError, TypeError, ValueError, WorkspaceError) as exc:
            raise RuntimeError(
                f"frame tag metadata load failed for {path}: {exc}") from exc
        # Loading/re-solving an existing label is not an editor modification.
        s.dirty = False
        s.annotation_dirty = False
        s.discard_armed = None
        if len(selected) > 1:
            cv2.setTrackbarPos(
                "frame%", win, _frame_cur_to_tick(cur, len(selected)))

        # 메인 루프 (한 프레임)
        s.sess_name, s.sess_sealed = seq_name, sealed   # 헤더 표시용
        next_action = None
        prev_ao = s.annot_only

        def _guard_dirty(what):
            """저장 안 한 클릭이 있으면 이번 한 번은 막는다. True 면 진행해도 된다.

            반드시 이 안쪽 루프에서 호출해야 한다. 바깥 루프로 나가면 프레임 reset 이
            돌아 지키려던 클릭이 지워진다. 프레임을 떠나는 모든 경로가 이걸 통과한다 —
            예전엔 n/p/q 만 검사해서 슬라이더·goto·jump·세션전환으로는 그냥 사라졌다.
            """
            return _confirm_discard(
                s, f"guard:{what}",
                f"[WARN] 미저장 변경 있음. 저장은 's', 버리고 {what} 하려면 한 번 더.")

        while next_action is None:
            update_pose(s, K)
            vis = render(s, cur, len(selected), stem)
            s.disp_shape = vis.shape[:2]   # _display_to_canvas 가 쓰는 실제 렌더 크기
            cv2.imshow(win, vis)
            key = cv2.waitKey(20) & 0xFF

            # 위젯(ANNOT-ONLY 버튼 / frame 슬라이더)은 키 입력이 없을 때(255)만 폴링한다.
            # 키 처리 前에 폴링하면 실제 키입력(예: 's' 저장)을 삼킬 수 있어 분리.
            if key == 255:
                # ANNOT-ONLY 버튼 토글 감지 (마우스 콜백이 s.annot_only 변경)
                if s.annot_only != prev_ao:
                    prev_ao = s.annot_only
                    if s.annot_only and not _has_annot(cur):   # 켜면 가까운 어노 frame 으로
                        nj = _step_annot(cur, +1)
                        if nj == cur:
                            nj = _step_annot(cur, -1)
                        if nj != cur and _guard_dirty("프레임 이동"):
                            s.goto = nj
                            next_action = 'goto'
                # SESSION 버튼 → 드롭다운 다이얼로그 (마우스 콜백은 플래그만 세움)
                if getattr(s, "sess_open", False):
                    s.sess_open = False
                    if len(sessions) > 1:
                        j = pick_session_dialog(
                            session_summary(
                                sessions, _REPO, args.population_role,
                                session_output_dirs,
                                evaluation_contexts if eval_root else None),
                            sess_i)
                        if j is not None and j != sess_i and _guard_dirty("세션 이동"):
                            s.sess_pick = j
                            next_action = 'sess-pick'
                    continue
                # 세션 슬라이더 (frame 보다 먼저 — 세션이 바뀌면 frame 은 무의미)
                if next_action is None and len(sessions) > 1:
                    tb_s = cv2.getTrackbarPos("session", win)
                    if tb_s != sess_i:
                        if not _guard_dirty("세션 이동"):
                            cv2.setTrackbarPos("session", win, sess_i)   # 핸들 되돌리기
                            continue
                        s.sess_pick = tb_s
                        next_action = 'sess-pick'
                        continue
                # 목록에서 항목을 클릭했으면 그 세션으로
                if next_action is None and getattr(s, "sess_pick", None) is not None:
                    next_action = 'sess-pick'
                    continue
                # frame 슬라이더 드래그/클릭 점프.
                # 슬라이더 max 는 전 세션 최대라, 현재 세션 길이로 잘라 쓴다.
                # 슬라이더는 퍼센트다. 현재 위치와 다른 프레임을 가리킬 때만 이동한다.
                # 퍼센트 -> 프레임 변환이 반올림이라, 같은 프레임을 가리키는 눈금
                # 범위에서는 아무 일도 일어나지 않아야 슬라이더가 튕기지 않는다.
                if next_action is None and len(selected) > 1:
                    want = _frame_trackbar_target(
                        cur, len(selected),
                        cv2.getTrackbarPos("frame%", win))
                    if want is not None:
                        if not _guard_dirty("프레임 이동"):
                            cv2.setTrackbarPos(
                                "frame%", win,
                                _frame_cur_to_tick(cur, len(selected)))
                            continue
                        s.goto = want
                        next_action = 'goto'
                continue

            # / opens a keyboard-isolated condition editor.  In this sub-mode
            # The modal owns condition selectors plus double-confirmed session
            # batch apply; every other editor key is a no-op.
            if getattr(s, "condition_mode", False):
                next_action = _handle_condition_key(
                    key, s, out_json, out_png, path, K)
                continue
            # ── Goto 번호 입력 모드 (; 로 진입, 숫자 타이핑 후 Enter) ──
            if s.goto_mode:
                if ord('0') <= key <= ord('9'):
                    s.goto_buf += chr(key)
                elif key in (13, 10):                       # Enter = 점프
                    if s.goto_buf and not _guard_dirty("프레임 이동"):
                        s.goto_mode = False
                        continue
                    if s.goto_buf:
                        s.goto = max(0, min(len(selected) - 1, int(s.goto_buf) - 1))
                        s.goto_mode = False
                        next_action = 'goto'
                        continue
                    s.goto_mode = False
                elif key == 27:                             # Esc = 취소
                    s.goto_mode = False
                elif key in (8, 127):                       # Backspace
                    s.goto_buf = s.goto_buf[:-1]
                continue
            if key == _GOTO_MODE_KEY:
                s.goto_mode = True
                s.goto_buf = ""
                continue

            # TAB = 세션 드롭다운.
            if key == 9 and len(sessions) > 1:
                j = pick_session_dialog(
                    session_summary(
                        sessions, _REPO, args.population_role,
                        session_output_dirs,
                        evaluation_contexts if eval_root else None),
                    sess_i)
                if j is not None and j != sess_i and _guard_dirty("세션 이동"):
                    s.sess_pick = j
                    next_action = 'sess-pick'
                continue
            if key == ord('[') and len(sessions) > 1:      # 이전 세션
                if _guard_dirty("세션 이동"):
                    s.sess_pick = (sess_i - 1) % len(sessions)
                    next_action = 'sess-pick'
                continue
            if key == ord(']') and len(sessions) > 1:      # 다음 세션
                if _guard_dirty("세션 이동"):
                    s.sess_pick = (sess_i + 1) % len(sessions)
                    next_action = 'sess-pick'
                continue

            # ── Mode toggle ──
            if key == ord('m'):
                if not s.session_writable:
                    _toast(
                        s,
                        "REVIEW ONLY: manipulate disabled",
                        (40, 40, 230),
                        log="[REVIEW ONLY] incoming mixed frame에서는 pose를 편집하지 않습니다.",
                    )
                    continue
                if s.mode == "click":
                    if s.pose is None:
                        print("[WARN] PnP 가 아직 안 풀려서 manipulate 진입 불가. 4점 이상 필요.")
                        continue
                    s.mode = "manip"
                    s.locked_pose = {
                        "R": s.pose["R"].copy(),
                        "t": s.pose["t"].copy(),
                        "dims": tuple(
                            s.pose.get("dims") or _state_default_wdh(s)),
                    }
                    for pose_key in (
                        "_wd_hypothesis", "_camera_facing_hypothesis",
                        "_axis_assignment", "_axis_assignment_candidates",
                        "_physical_dimensions_m", "_wd_selection_reason",
                        "_wd_candidates", "_wd_ambiguous",
                    ):
                        if pose_key in s.pose:
                            s.locked_pose[pose_key] = s.pose[pose_key]
                    print("[Mode] CLICK → MANIPULATE")
                else:
                    s.mode = "click"
                    s.locked_pose = None
                    print("[Mode] MANIPULATE → CLICK")
                continue

            # ── Mode-specific dispatcher ──
            if s.mode == "manip":
                next_action = _handle_manip_key(key, s, out_json, out_png, path, K)
            else:
                next_action = _handle_click_key(key, s, out_json, out_png, path, K)

        # 다음 frame 결정
        if next_action == 'quit':
            break
        elif next_action == 'save-next':
            cur += 1                                    # 저장 후엔 항상 다음 sequential frame
        elif next_action == 'next':
            cur = _step_annot(cur, +1) if s.annot_only else cur + 1
        elif next_action == 'prev':
            cur = _step_annot(cur, -1) if s.annot_only else max(0, cur - 1)
        elif next_action == 'jump-10':
            cur = max(0, cur - 10)
        elif next_action == 'jump+10':
            cur = min(len(selected) - 1, cur + 10)
        elif next_action == 'goto':
            if s.goto is not None:
                cur = max(0, min(len(selected) - 1, s.goto))
            s.goto = None
        elif next_action == 'sess-pick':
            # 세션 교체: 경로/K/프레임 목록을 갈아끼우고 첫 프레임으로. 창은 그대로 둔다.
            # 미저장 가드는 여기가 아니라 키/폴링 쪽(_guard_dirty)에 있다. 여기서 막으면
            # continue 가 바깥 루프로 가 프레임 reset 이 돌면서, 지키려던 클릭이 오히려
            # 지워진다(2026-08-15).
            sess_i = s.sess_pick
            seq, seq_name, out_dir, sealed, K, rgb_paths, selected = load_session(sess_i)
            s.sess_pick = None
            if len(sessions) > 1:
                cv2.setTrackbarPos("session", win, sess_i)
            if not rgb_paths:
                print(f"[WARN] {seq_name}: rgb 프레임이 없어 건너뛴다")
                continue
            if len(selected) > 1:
                cv2.setTrackbarPos("frame%", win, 0)
            cur = 0

    cv2.destroyAllWindows()
    saved = len(glob.glob(os.path.join(out_dir, "*.json")))
    print(f"\n[Done] quit. saved={saved} JSON files in {out_dir}")


if __name__ == "__main__":
    main()
