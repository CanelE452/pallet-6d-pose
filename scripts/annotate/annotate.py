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
  w             W/D parity 전환 (short-face-front ↔ long-face-front)
  y             현재 W/D parity 안에서 signed canonical axis 확인/순환
  G 또는 :      frame 번호 입력 점프 (숫자 후 Enter, Esc 취소) — 상단 슬라이더 클릭/드래그도 가능
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
  q / Q         종료

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
import glob
import os
import sys
import time

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
)
from object_geometry_registry import (
    DEFAULT_REGISTRY_PATH,
    PLASTIC_OBJECT_TYPE,
    WOOD_OBJECT_TYPE,
    load_object_geometry_registry,
)


# ─── Session pool ────────────────────────────────────────────────────────────

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
            if not glob.glob(os.path.join(seq, "rgb", "*.png")):
                continue
            found.append((name, seq))

    by_out = {}
    for name, seq in found:
        od, _ = resolve_out_dir(name, repo)
        prev = by_out.get(od)
        if prev is None:
            by_out[od] = (name, seq)
            continue
        n_prev = len(glob.glob(os.path.join(prev[1], "rgb", "*.png")))
        n_cur = len(glob.glob(os.path.join(seq, "rgb", "*.png")))
        keep = prev if (n_prev, len(prev[0])) >= (n_cur, len(name)) else (name, seq)
        drop = (name, seq) if keep is prev else prev
        print(f"[세션] '{drop[0]}' 은 '{keep[0]}' 과 같은 저장 폴더를 쓴다 — 목록에서 제외")
        by_out[od] = keep
    return sorted(by_out.values(), key=lambda t: t[0])


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


def session_summary(sessions, repo, population_role="DEV"):
    """세션 목록에 프레임 수와 이미 어노된 JSON 수를 붙인다."""
    rows = []
    for name, seq in sessions:
        n = len(glob.glob(os.path.join(seq, "rgb", "*.png")))
        od, _legacy_eval_layout = resolve_out_dir(name, repo)
        done = len(glob.glob(os.path.join(od, "*.json")))
        rows.append((name, n, done, str(population_role).upper() == "FINAL"))
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

    labels = [f"{i+1:>2}. {nm}   ({nfr}f, done {done}){'  [FINAL ROLE]' if sealed else ''}"
              for i, (nm, nfr, done, sealed) in enumerate(rows)]
    picked = {"i": None}

    root = tk.Tk()
    root.title("세션 선택")
    root.attributes("-topmost", True)
    frm = ttk.Frame(root, padding=12)
    frm.grid()
    ttk.Label(frm, text="촬영 세션을 고르세요").grid(column=0, row=0, sticky="w", pady=(0, 6))
    var = tk.StringVar(value=labels[current] if 0 <= current < len(labels) else labels[0])
    box = ttk.Combobox(frm, textvariable=var, values=labels, state="readonly", width=52)
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
    s.dirty = True
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
        s.dirty = True


def _sync_axis_hypothesis(s: State):
    if s.pose is None:
        return
    candidates = list(s.pose.get("_axis_assignment_candidates") or [])
    if not candidates:
        return
    old = list(s.axis_assignment_candidates or [])
    if old != candidates:
        s.axis_assignment_candidates = candidates
        s.axis_assignment = candidates[0]
        s.axis_assignment_confirmed = False
    elif s.axis_assignment not in candidates:
        s.axis_assignment = candidates[0]
        s.axis_assignment_confirmed = False


def _cycle_axis_assignment(s: State):
    candidates = list(s.axis_assignment_candidates or [])
    if not candidates:
        print("[Axis] PnP W/D hypothesis가 아직 없습니다.")
        return
    if not s.axis_assignment_confirmed:
        s.axis_assignment = (s.axis_assignment if s.axis_assignment in candidates
                             else candidates[0])
        s.axis_assignment_confirmed = True
    else:
        current = (candidates.index(s.axis_assignment)
                   if s.axis_assignment in candidates else -1)
        s.axis_assignment = candidates[(current + 1) % len(candidates)]
    s.dirty = True
    print(f"[Axis] confirmed {s.axis_assignment}; candidates={candidates}")


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
    # A W/D switch changes the allowed signed pair.  Confirmation from the old
    # parity must never carry across it.
    s.axis_assignment = None
    s.axis_assignment_candidates = []
    s.axis_assignment_confirmed = False
    s._pose_key = None
    s.dirty = True
    print(f"[W/D] manual parity correction: {current} -> {target}; "
          "signed axis reset (press y after checking the projection)")


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
        if not s.axis_assignment_confirmed:
            return "FINAL save blocked: signed axis assignment is not confirmed (press y)"
    return None


def _make_state_annotation(s: State, K):
    error = _save_contract_error(s)
    if error:
        _toast(s, "[SAVE BLOCKED] check visibility/axis", (40, 40, 230), log=error)
        return None
    return make_annotation(
        s.kps_2d, s.pose, s.img_shape, K,
        dims=tuple(s.pose.get("dims") or _state_default_wdh(s)),
        split=s.split,
        extrap_mask=s.extrap_mask,
        keypoint_annotations=s.keypoint_annotations,
        axis_assignment=s.axis_assignment,
        axis_assignment_candidates=s.axis_assignment_candidates,
        axis_assignment_confirmed=s.axis_assignment_confirmed,
        legacy_object=s.legacy_object,
        legacy_document=s.legacy_document,
        population_role=s.population_role,
        metadata=s.capture_metadata,
        occlusion_level=s.occlusion_level,
        geometry_spec=_state_geometry_spec(s),
        intrinsics_quality=getattr(s, "intrinsics_quality", None),
        intrinsics_source=getattr(s, "intrinsics_source", None),
    )


def _save_state_annotation(s: State, K, out_json, out_png, src_png):
    try:
        _require_nonlegacy_output_dir(os.path.dirname(out_json), _REPO)
    except ValueError as exc:
        _toast(s, "[SAVE BLOCKED] legacy GT is read-only", (40, 40, 230),
               log=f"GT v2 save path rejected: {exc}: {out_json}")
        return False
    ann = _make_state_annotation(s, K)
    if ann is None:
        return False
    try:
        save_frame_json(out_json, out_png, src_png, ann)
    except (TypeError, ValueError) as exc:
        _toast(s, "[SAVE BLOCKED] v2 schema error", (40, 40, 230),
               log=f"GT v2 schema validation failed: {exc}")
        return False
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
                    s.annot_only = not s.annot_only
                    print(f"[Annot-only] {'ON' if s.annot_only else 'OFF'}"
                          f"  (n/p 로 어노된 frame 만 이동)")
                    return
                sx0, sy0, sx1, sy1 = session_button_rect(canvas_h)
                if sx0 <= px <= sx1 and sy0 <= py <= sy1:
                    s.sess_open = True      # 메인 루프가 목록을 채워 연다
            return
    # MANIPULATE 모드에서는 마우스 클릭으로 점 안 찍음
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
                    s.dirty = True
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
        s.dirty = True
        if s.active < 8:
            s.active += 1
    elif event == cv2.EVENT_RBUTTONDOWN:
        if s.kps_2d[s.active] is not None:
            s.kps_2d[s.active] = None
            if s.extrap_mask is not None:
                s.extrap_mask[s.active] = False
            _clear_keypoint_state(s, s.active)
            s.pose = None
            s.dirty = True


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
    _sync_axis_hypothesis(s)


# ─── Key dispatchers ──────────────────────────────────────────────────────────

def _handle_manip_key(key, s, out_json, out_png, src_png, K):
    """MANIPULATE 모드 키 처리. Returns: 'next' | 'quit' | None."""
    ts = s.trans_step
    rs = s.rot_step_deg
    if key == ord('b'):
        _cycle_visibility_reason(s, s.active)
    elif key == ord('y'):
        _cycle_axis_assignment(s)
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
    elif key == ord('S'):
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
        s.dirty = True
        if not _save_state_annotation(s, K, out_json, out_png, src_png):
            return None
        print(f"[Saved manip] {out_json}  reproj={s.pose['reproj_error_px']:.2f}px")
        s.dirty = False
        s.mode = "click"
        s.locked_pose = None
        return 'save-next'
    elif key == ord('s'):
        # 소문자 's' 는 manip 에서 아무 일도 안 해서 "저장이 안 된다" 로 보였다.
        print("[manip] 저장은 대문자 'S'. 어노를 없애려면 'm' 으로 CLICK 모드로 나간 뒤 "
              "'r' 로 전부 지우고 's'.")
    elif key == ord('Q'):
        return 'quit'
    return None


def _toast(s, screen_text, color=(60, 200, 60), log=None):
    """화면 위에 잠깐 뜨는 알림. 터미널 print 만으로는 쓰는 사람이 못 본다.

    이 툴의 안내가 전부 stdout 으로만 나가서, 실제로는 동작했는데도 "아무 일도
    안 일어난다" 로 보였다(2026-08-16 삭제 기능에서 실제로 겪음).

    ★ screen_text 는 **반드시 ASCII** 여야 한다. cv2.putText 의 Hershey 폰트에는
      한글 glyph 가 없어서 한글을 넘기면 통째로 '?????' 로 그려진다(실제로 겪음).
      화면은 영문, 터미널(log)은 한글 — 그래서 둘을 나눠 받는다.
    """
    s.toast = (screen_text, color, time.time() + 2.5)
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
               log="FINAL 삭제 차단: visibility/axis gate를 통과한 라벨은 "
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
    return 'save-next'


def _handle_click_key(key, s, out_json, out_png, src_png, K):
    """CLICK 모드 키 처리. Returns: 'next' | 'prev' | 'quit' | None."""
    if key == ord('q'):
        # 저장 안 한 클릭이 있으면 한 번 막는다. 'n' 에만 이 보호가 있어서 q/p 로는
        # 찍던 걸 그냥 잃었다(2026-08-15). 같은 규칙으로 통일 — 다시 누르면 진행.
        if s.dirty:
            print("[WARN] 미저장 변경 있음. 저장하려면 's', 버리려면 'q' 를 한 번 더.")
            s.dirty = False
            return None
        return 'quit'

    if key == ord('v'):
        s.split = "train" if s.split == "eval" else "eval"
        s.dirty = True
        print(f"[Split] this frame -> {s.split.upper()}  (저장 시 JSON 에 반영)")
        return None

    if key == ord('b'):
        _cycle_visibility_reason(s, s.active)
        return None

    if key == ord('w'):
        _cycle_wd_parity(s)
        return None

    if key == ord('y'):
        _cycle_axis_assignment(s)
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
            s.dirty = True
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
        s.dirty = True
        print(f"[Parallelogram] kp{s.active} ← face={fname} {finds} → "
              f"({pt[0]:.1f}, {pt[1]:.1f})")
        if s.active < 8:
            s.active += 1
        return None

    if key == ord('n'):
        if s.dirty:
            print("[WARN] 미저장 변경 있음. 다시 'n' 누르면 무시하고 다음.")
            s.dirty = False
            return None
        return 'next'
    if key == ord('p'):
        if s.dirty:
            print("[WARN] 미저장 변경 있음. 다시 'p' 누르면 무시하고 이전.")
            s.dirty = False
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
            s.dirty = True
            print(f"[Centroid] PnP projection: ({s.kps_2d[8][0]:.1f}, {s.kps_2d[8][1]:.1f})")
        else:
            pts = [k for k in s.kps_2d[:8] if k is not None]
            if len(pts) >= 4:
                s.kps_2d[8] = [float(np.mean([p[0] for p in pts])),
                               float(np.mean([p[1] for p in pts]))]
                _set_keypoint_state(
                    s, 8, s.kps_2d[8], source="centroid_auto",
                    visibility=1, reason="unknown")
                s.dirty = True
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
            s.dirty = True
        return None

    if key == ord('d'):
        if s.kps_2d[s.active] is not None:
            s.kps_2d[s.active] = None
            if s.extrap_mask is not None:
                s.extrap_mask[s.active] = False
            _clear_keypoint_state(s, s.active)
            s.pose = None
            s.dirty = True
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
            s.dirty = True
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
    ap.add_argument("--population-role", "--population_role", required=True,
                    choices=["DEV", "FINAL"],
                    help="명시적 GT population 역할. FINAL은 kp0~7 unknown 저장을 차단.")
    ap.add_argument("--object-type", default=PLASTIC_OBJECT_TYPE,
                    help="OBJECT_GEOMETRY_REGISTRY의 canonical type 또는 alias")
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

    registry_path = (args.geometry_registry
                     if os.path.isabs(args.geometry_registry)
                     else os.path.join(_REPO, args.geometry_registry))
    try:
        geometry_registry = load_object_geometry_registry(registry_path)
        geometry_spec = geometry_registry.resolve(args.object_type)
    except (OSError, TypeError, ValueError) as exc:
        ap.error(f"invalid object geometry registry/selection: {exc}")
    if (geometry_spec.object_type == WOOD_OBJECT_TYPE
            and args.intrinsics_quality is None):
        ap.error("wood annotation requires --intrinsics-quality")
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
    if eval_root:
        if requested_out is None:
            ap.error("--eval-root requires an explicit canonical --out_dir")
        try:
            seq, requested_out = _validate_evaluation_paths(
                eval_root, seq, requested_out, args.population_role)
        except ValueError as exc:
            ap.error(str(exc))
    seq_name = os.path.basename(seq.rstrip("/\\"))

    # 세션 풀 — 툴 안에서 [ ] / TAB 으로 갈아탄다. --out_dir 을 직접 준 경우엔 그
    # 세션만 다루도록 풀을 잠근다(지정한 출력 폴더가 다른 세션에 새어들지 않게).
    sessions = discover_sessions(args.pool, _REPO)
    if args.out_dir or not sessions:
        sessions = [(seq_name, seq)]
    elif all(p != seq for _, p in sessions):
        sessions.insert(0, (seq_name, seq))
    sess_i = next((i for i, (_, p) in enumerate(sessions) if p == seq), 0)

    def load_session(i):
        """세션 i 로 전환. FINAL 표시는 명시적 population role만 따른다."""
        nm, sq = sessions[i]
        if args.out_dir:
            od = args.out_dir if os.path.isabs(args.out_dir) \
                else os.path.join(_REPO, args.out_dir)
        else:
            od, _legacy_eval_layout = resolve_out_dir(nm, _REPO)
        od = _require_nonlegacy_output_dir(od, _REPO)
        sealed = args.population_role == "FINAL"
        # 여기서 makedirs 하면 세션 목록을 둘러보기만 해도 빈 GT 폴더가 생겨
        # done 집계와 discover_sessions 의 중복 판정이 오염된다. 저장할 때 만든다.
        _DEFAULT_K = np.array([[614.18, 0, 329.28], [0, 614.31, 234.53], [0, 0, 1]],
                              dtype=np.float64)
        kp = os.path.join(sq, "cam_K.txt")
        k = _DEFAULT_K
        if os.path.isfile(kp):
            try:
                k = np.loadtxt(kp).reshape(3, 3)
            except Exception as e:
                # 깨진 cam_K 하나에 창이 죽으면 다른 세션도 못 본다. 기본 K 로 계속하되
                # 이 세션 라벨은 잘못된 intrinsic 으로 풀린다는 걸 크게 알린다.
                print(f"[ERROR] cam_K.txt 를 읽지 못했다 ({e}) — 기본 K 로 진행. "
                      f"이 세션의 pose 는 신뢰하지 말 것: {kp}")
        rp = sorted(glob.glob(os.path.join(sq, "rgb", "*.png")))
        sel = list(range(args.start, len(rp), args.stride))
        print(f"\n[Session {i+1}/{len(sessions)}] {nm}"
              f"{'   ★FINAL population role — save gates active' if sealed else ''}")
        print(f"           {len(sel)} frames (stride={args.stride}) of {len(rp)}")
        print(f"           Output: {od}")
        print(f"           Object: {geometry_spec.object_type} "
              f"XYZ={geometry_spec.physical_dimensions_m}")
        print(f"           K = fx={k[0,0]:.1f} cx={k[0,2]:.1f} cy={k[1,2]:.1f}")
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
    s.object_type = geometry_spec.object_type
    s.intrinsics_quality = args.intrinsics_quality
    s.intrinsics_source = args.intrinsics_source
    s.eval_root = eval_root
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
    SLIDER_TICKS = 500          # 최대 세션(185장)도 전 프레임 도달 (100 이면 절반만 닿는다)

    def _cur_to_tick(c, n):
        return 0 if n <= 1 else int(round(c / (n - 1) * SLIDER_TICKS))

    def _tick_to_cur(t, n):
        return 0 if n <= 1 else int(round(t / SLIDER_TICKS * (n - 1)))

    cv2.createTrackbar("frame%", win, 0, SLIDER_TICKS, lambda v: None)
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
                if k in (ord('q'), ord('Q'), 27):
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
        s.population_role = args.population_role
        s.geometry_registry = geometry_registry
        s.geometry_spec = geometry_spec
        s.object_type = geometry_spec.object_type
        s.loaded_object_type = None
        s.intrinsics_quality = args.intrinsics_quality
        s.intrinsics_source = args.intrinsics_source
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
        # ★ 모드/입력 중 상태도 반드시 초기화한다. 예전엔 mode/locked_pose 가 남아,
        #   MANIPULATE 로 잠근 이전 프레임의 R,t 를 새 프레임에 그대로 투영해 manual GT 로
        #   저장할 수 있었다. 새 프레임은 kps_2d 가 비어 있어 reproj 가 0.00px 로 찍히는
        #   바람에 완벽한 라벨처럼 보였다. line/goto 상태도 남으면 클릭·키가 먹통이 된다.
        s.mode = "click"
        s.locked_pose = None
        s.line_mode = False
        s.line_pts = None
        s.goto_mode = False
        s.goto_buf = ""
        s.goto = None
        s._pose_key = None             # 새 프레임 — pose 캐시 무효화
        s.toast = None                 # 알림이 다음 프레임으로 새지 않게
        s.split = args.default_split   # 기본 split; 기존 JSON 있으면 load 가 override
        load_json = out_json
        load_is_read_only_legacy = False
        if not os.path.exists(load_json):
            automatic_legacy_dir, _ = _resolve_legacy_read_dir(seq_name, _REPO)
            source_dir = legacy_read_dir or automatic_legacy_dir
            legacy_json = (os.path.join(source_dir, f"{stem}.json")
                           if source_dir else None)
            if legacy_json and os.path.exists(legacy_json):
                load_json = legacy_json
                load_is_read_only_legacy = True
                print(f"[Read-only legacy source] {legacy_json}\n"
                      f"                          save target: {out_json}")
        if load_existing_annotation(
                s, load_json, read_only=load_is_read_only_legacy):
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
        if len(selected) > 1:
            cv2.setTrackbarPos("frame%", win, _cur_to_tick(cur, len(selected)))

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
            if s.dirty:
                print(f"[WARN] 미저장 변경 있음. 저장은 's', 버리고 {what} 하려면 한 번 더.")
                s.dirty = False
                return False
            return True

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
                            session_summary(sessions, _REPO, args.population_role), sess_i)
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
                    want = _tick_to_cur(cv2.getTrackbarPos("frame%", win), len(selected))
                    if want != cur:
                        if not _guard_dirty("프레임 이동"):
                            cv2.setTrackbarPos("frame%", win, _cur_to_tick(cur, len(selected)))
                            continue
                        s.goto = want
                        next_action = 'goto'
                continue

            # ── Goto 번호 입력 모드 (G/: 로 진입, 숫자 타이핑 후 Enter) ──
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
            if key in (ord('G'), ord(':')):
                s.goto_mode = True
                s.goto_buf = ""
                continue

            # TAB = 세션 드롭다운. 'S' 는 MANIPULATE 의 save+next 라 쓰지 않는다.
            if key == 9 and len(sessions) > 1:
                j = pick_session_dialog(
                    session_summary(sessions, _REPO, args.population_role), sess_i)
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
