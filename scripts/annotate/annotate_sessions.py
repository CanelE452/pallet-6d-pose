"""Session discovery, validation, and chooser metadata for annotation."""

from __future__ import annotations

import copy
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

from object_geometry_registry import PLASTIC_OBJECT_TYPE, WOOD_OBJECT_TYPE
from evaluation.eval_workspace import (
    WorkspaceError,
    load_frame_tag_overrides,
    load_session_metadata,
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
    FINAL-intent *staging* annotations only: the raw capture never becomes an
    evaluation member; each saved/reviewed frame is independently synchronized
    to the combined evaluation collection by the editor hook.
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
            "workspace_scope": layout_role,
            "display_role": candidate_args.population_role,
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
                "refresh_evaluation": True,
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
