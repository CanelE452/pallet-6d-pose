"""real 손 어노테이션 라벨의 규약 감사 — 프레임 단위 판정.  읽기 전용.

목적 : 다음 실험(real supervision 의 앙각 구성 ablation)에 쓸 real 라벨이 신뢰
       가능한지 착수 전에 확인한다.  직전 시도 `paper_real_ft_v1` 은 학습 전에
       중단됐고 사유가 라벨 결함이었다(당시 402장 중 106장 LR 순서 위반,
       187장 90도 stale).  그 뒤 live_capture_gt 가 851장으로 늘었고 추가분
       감사 기록이 저장소에 없다.

지표 : 프레임마다 LABEL_OK / LR_ORDER_VIOLATION / YAW90_STALE / OTHER_DEFECT /
       AMBIGUOUS 판정 + 카메라 앙각(도).  앙각 층별 LABEL_OK 수가 다음 실험의
       arm 크기를 정한다.

## 판정을 어떻게 세웠나 — 그리고 왜 순진한 검사가 무의미한가

저장된 `pose_transform` 은 독립 계측이 아니라 사람이 찍은 점으로 푼 PnP 해다
(GT_TRUST_AUDIT.md).  그래서 "pose 를 투영해 `manual_kps` 와 맞는지" 만 보면
거의 모든 프레임이 통과한다 — 같은 것을 두 번 재는 셈이다(실측: 1,202 프레임
identity 최대오차 p99 = 10.2 px, 중앙값 1.3 px).  독립 신호가 필요하다.

이 도구는 서로 독립인 세 축을 본다.

  (1) 교차필드   `keypoint_annotations`(사람 클릭 순서)  대
                 `pose_transform`/`manual_kps`/`projected_cuboid`(솔버 축배정).
                 두 값이 같은 파일 안에 있고 서로 다른 계보를 갖는다.
                 어긋남이 yaw 순열로 설명되면 YAW90_STALE,
                 거울(mirror) 순열이면 LR_ORDER_VIOLATION.

  (2) 규약 불변식 camera-facing 0123 은 "0~3 = 카메라를 향한 면" 을 요구한다.
                 0123 면의 법선(local -Z)과 시선의 사잇각(= 면 기울기)이
                 90도 근처면 그 면은 카메라를 향한 게 아니라 옆으로 서 있다.
                 이 검사는 점을 안 쓰므로 PnP 순환에 걸리지 않는다.

  (3) 시간 일관성 같은 세션에서 카메라가 몇 cm 움직였는데 라벨 회전이 ~90도
                 튀면 물리적으로 불가능하다 = 축 phase 가 프레임마다 흔들린다.
                 정사각 팔레트(W=D)의 C4 모호성을 실측으로 잡는다.

앙각은 상판 법선 n = R @ (0,-1,0) 과 팔레트→카메라 u = -t/|t| 로 asin(|n·u|).
n 은 Y 축 회전에 불변이므로 **위 90도 phase 문제의 영향을 받지 않는다**.

permutation 은 scripts/paper/diagnose_axis_failures.py 의 PERMUTATIONS 정본과
같은 정의를 쓴다 (gt[i] <- pred[perm[i]]).

기존 GT JSON 은 읽기만 한다.  수정·이동·삭제 0건.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "annotate"))

import annotate_pnp as APNP  # noqa: E402
from challenge.data_paths import EVAL_CANONICAL, REAL_MANUAL_GT  # noqa: E402

# ── permutation 정본 (scripts/paper/diagnose_axis_failures.py 와 같은 정의) ──
FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6, 8]
YAW90 = [1, 5, 6, 2, 0, 4, 7, 3, 8]


def compose(outer, inner):
    return [inner[index] for index in outer]


YAW180 = compose(YAW90, YAW90)
YAW270 = compose(YAW180, YAW90)
PERMUTATIONS = {
    "identity": list(range(9)),
    "yaw90": YAW90,
    "yaw180": YAW180,
    "yaw270": YAW270,
    "mirror": FLIP_IDX,
    "mirror_yaw90": compose(FLIP_IDX, YAW90),
    "mirror_yaw180": compose(FLIP_IDX, YAW180),
    "mirror_yaw270": compose(FLIP_IDX, YAW270),
}
ROT_PERMS = ("yaw90", "yaw180", "yaw270")
MIRROR_PERMS = ("mirror", "mirror_yaw90", "mirror_yaw180", "mirror_yaw270")
# 각 순열이 대응하는 물체 Y축 회전 (정사각 W=D 일 때만 성립).  검증: probe 로
# kp_ann[i] == proj(R @ Ry(+90) @ X_i) 임을 실측 (잔차 중앙값 0.00 px).
PERM_TO_RY = {"identity": 0.0, "yaw90": -90.0, "yaw180": 180.0, "yaw270": 90.0}

# ── 임계 — 전부 [추정][미검증].  분포를 먼저 보고 정했다. ────────────────────
CLICK_MATCH_PX = 5.0     # 교차필드 일치로 볼 최대 오차
OK_PX = 15.0             # 클릭 잔차가 이 이상이면 순열과 무관하게 의심
FACE_EDGEON_DEG = 60.0   # 0123 면 기울기가 이 이상이면 "카메라를 향하지 않음"
NEIGHBOUR_TRANS_M = 0.15  # 이 이하로 움직였는데 회전이 튀면 phase flip
NEIGHBOUR_FLIP_DEG = 45.0
NEIGHBOUR_C4_DEG = 15.0
MIN_CLICKS = 4
SENTINEL_TOL = 1e-6
ROT_ORTHO_TOL = 1e-4
ELEV_BINS = ("<8", "8-15", ">=15")


def elev_bin(e):
    return "<8" if e < 8 else "8-15" if e < 15 else ">=15"


def load_K(lab):
    k = ((lab.get("camera_data") or {}).get("intrinsics") or {})
    if not all(isinstance(k.get(x), (int, float)) for x in ("fx", "fy", "cx", "cy")):
        return None
    return np.array([[k["fx"], 0.0, k["cx"]],
                     [0.0, k["fy"], k["cy"]],
                     [0.0, 0.0, 1.0]], float)


def model_points(dims):
    return np.asarray(APNP.make_pallet_keypoints_3d_diagram(
        width=dims["width"], depth=dims["depth"], height=dims["height"]), float)


def ry(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def geodesic_deg(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1.0) / 2.0, -1.0, 1.0))))


def as_xy(p):
    if isinstance(p, (list, tuple)) and len(p) == 2 \
            and all(isinstance(c, (int, float)) for c in p) \
            and all(math.isfinite(float(c)) for c in p):
        return [float(p[0]), float(p[1])]
    return None


def is_int_coord(p):
    return abs(p[0] - round(p[0])) < 1e-9 and abs(p[1] - round(p[1])) < 1e-9


def is_sentinel(p):
    return abs(p[0] + 1.0) < SENTINEL_TOL and abs(p[1] + 1.0) < SENTINEL_TOL


def read_points(o):
    """(A, srcA, M) — A = keypoint_annotations, M = manual_kps(없으면 cuboid+centroid)."""
    A = np.full((9, 2), np.nan)
    srcA = [None] * 9
    ann = o.get("keypoint_annotations")
    has_A = isinstance(ann, list) and len(ann) == 9
    if has_A:
        for i, a in enumerate(ann):
            if isinstance(a, dict):
                srcA[i] = a.get("source")
                xy = as_xy(a.get("xy"))
                if xy:
                    A[i] = xy
    M = np.full((9, 2), np.nan)
    mk = o.get("manual_kps")
    if isinstance(mk, list) and len(mk) == 9:
        for i, p in enumerate(mk):
            xy = as_xy(p)
            if xy:
                M[i] = xy
    else:
        pc, ct = o.get("projected_cuboid"), o.get("projected_cuboid_centroid")
        if isinstance(pc, list) and len(pc) == 8:
            for i, p in enumerate(pc):
                xy = as_xy(p)
                if xy:
                    M[i] = xy
        xy = as_xy(ct)
        if xy:
            M[8] = xy
    return (A if has_A else None), srcA, M


def click_indices(A, srcA, M):
    """클릭 인덱스와 그 좌표계.  source 태그가 있으면 그것을, 없으면 정수좌표를 쓴다."""
    known = {"manual_click", "pnp_projected", "centroid_auto"}
    if A is not None and any(s in known for s in srcA):
        idx = [i for i in range(9) if srcA[i] == "manual_click"
               and np.isfinite(A[i]).all() and not is_sentinel(A[i])]
        return idx, A, "keypoint_annotations"
    src = A if A is not None else M
    idx = [i for i in range(9) if np.isfinite(src[i]).all()
           and not is_sentinel(src[i]) and is_int_coord(src[i])]
    return idx, src, ("keypoint_annotations(int)" if A is not None else "manual_kps(int)")


# 이미지는 GT 폴더 옆에 있기도 하고(초기 세션) 촬영본 폴더에만 있기도 하다
# (`_live_captures/*/sessions/<세션>/rgb/`).  둘 다 본다.
_IMAGE_INDEX = None


def image_index():
    global _IMAGE_INDEX
    if _IMAGE_INDEX is None:
        idx = collections.defaultdict(list)
        for base in (ROOT / "challenge/data/01_real", ROOT / "data/pallet/raw_data"):
            if base.exists():
                for p in base.rglob("*.png"):
                    idx[p.stem].append(p)
        _IMAGE_INDEX = idx
    return _IMAGE_INDEX


def find_image(path, folder):
    if path.with_suffix(".png").exists():
        return path.with_suffix(".png")
    sess = folder[:-len("_manual_gt")] if folder.endswith("_manual_gt") else folder
    for cand in image_index().get(path.stem, []):
        parts = cand.parts
        if sess in parts or f"{sess}_manual_gt" in parts:
            return cand
    return None


def audit_frame(path, group, folder):
    r = {c: "" for c in COLS}
    img = find_image(path, folder)
    r.update(folder=folder, frame=path.stem, group=group,
             image_exists=str(img is not None),
             image_path=str(img.relative_to(ROOT)) if img else "",
             n_click=0, n_sentinel=0, n_none=0, n_evidence=0)
    try:
        lab = json.loads(path.read_text("utf-8"))
    except Exception as exc:
        r.update(verdict="OTHER_DEFECT", reason=f"JSON_PARSE_FAIL:{type(exc).__name__}")
        return r, None
    objs = lab.get("objects") or []
    if not objs:
        r.update(verdict="OTHER_DEFECT", reason="NO_OBJECT")
        return r, None
    o = objs[0]
    r["object_type"] = o.get("object_type") or lab.get("object_type") or ""
    r["split"] = o.get("split") or ""
    r["pose_status"] = o.get("pose_status") or ""
    r["session"] = lab.get("capture_session_id") or folder
    r["gt_source"] = o.get("gt_source") or ""
    if isinstance(o.get("reproj_error_px"), (int, float)):
        r["stored_reproj_px"] = round(float(o["reproj_error_px"]), 3)

    hard = []
    K = load_K(lab)
    if K is None:
        hard.append("INTRINSICS_MISSING")
    dims = o.get("dimensions_m") or {}
    if not all(isinstance(dims.get(k), (int, float)) and dims[k] > 0
               for k in ("width", "height", "depth")):
        hard.append("BAD_DIMS")
    else:
        r["dims_wd"] = f"{dims['width']:.3f}x{dims['depth']:.3f}"
        r["is_square"] = str(abs(dims["width"] - dims["depth"]) < 1e-6)

    T = o.get("pose_transform")
    R = t = None
    if not (isinstance(T, list) and len(T) == 4 and all(isinstance(x, list) and len(x) == 4 for x in T)):
        hard.append("BAD_POSE_SHAPE")
    else:
        Mx = np.asarray(T, float)
        if not np.isfinite(Mx).all():
            hard.append("NONFINITE_POSE")
        else:
            R, t = Mx[:3, :3], Mx[:3, 3]
            ortho = float(np.abs(R.T @ R - np.eye(3)).max())
            det = float(np.linalg.det(R))
            if ortho > ROT_ORTHO_TOL:
                hard.append(f"R_NOT_ORTHONORMAL({ortho:.1e})")
            if abs(det - 1.0) > 1e-3:
                hard.append(f"DET_NOT_1({det:.4f})")
            if t[2] <= 0:
                hard.append(f"CAM_BEHIND(tz={t[2]:.3f})")
            r["dist_m"] = round(float(np.linalg.norm(t)), 3)
            if float(np.linalg.norm(t)) > 1e-9 and ortho <= ROT_ORTHO_TOL:
                n = R @ np.array([0.0, -1.0, 0.0])
                u = -t / float(np.linalg.norm(t))
                r["elevation_deg"] = round(float(np.degrees(
                    np.arcsin(np.clip(abs(float(n @ u)), 0.0, 1.0)))), 2)

    A, srcA, M = read_points(o)
    prim = A if A is not None else M
    n_none_M = int(np.sum(~np.isfinite(M).all(axis=1)))
    n_none_A = int(np.sum(~np.isfinite(A).all(axis=1))) if A is not None else 0
    r["n_none"] = max(n_none_M, n_none_A)
    r["n_sentinel"] = int(sum(1 for i in range(9)
                              if np.isfinite(M[i]).all() and is_sentinel(M[i])))
    ev, S, evsrc = click_indices(A, srcA, M)
    r["n_click"] = len(ev)
    r["evidence_source"] = evsrc
    r["has_kpann"] = str(A is not None)

    if hard:
        r.update(verdict="OTHER_DEFECT", reason="|".join(hard))
        return r, None
    if not np.isfinite(prim).all() or not np.isfinite(M).all():
        r["defect_note"] = f"NULL_KEYPOINT(n={r['n_none']})"
    if r["n_sentinel"]:
        r["defect_note"] = (r["defect_note"] + "|" if r["defect_note"] else "") \
            + f"SENTINEL(n={r['n_sentinel']})"

    X = model_points(dims)
    cam = (R @ X.T).T + t
    if (cam[:, 2] <= 0).any():
        r.update(verdict="OTHER_DEFECT",
                 reason=f"PROJ_BEHIND_CAM({int(np.sum(cam[:, 2] <= 0))})")
        return r, None
    P = np.stack([K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2],
                  K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]], axis=1)
    r["proj_diag_px"] = round(float(np.hypot(P[:8, 0].ptp(), P[:8, 1].ptp())), 1)

    # (2) 규약 불변식 — 점을 쓰지 않는다
    view = -t / float(np.linalg.norm(t))
    r["face_obliquity_deg"] = round(float(np.degrees(np.arccos(np.clip(
        float((R @ np.array([0.0, 0.0, -1.0])) @ view), -1.0, 1.0)))), 1)
    r["near_ok"] = str(bool(cam[:4, 2].mean() < cam[4:8, 2].mean()))
    r["top_ok"] = str(bool(cam[[0, 1, 4, 5], 1].mean() < cam[[2, 3, 6, 7], 1].mean()))
    u = P[:, 0]
    r["lr_screen_flip"] = str(not (u[0] < u[1] and u[3] < u[2]))

    r["n_evidence"] = len(ev)
    if len(ev) < MIN_CLICKS:
        why = "NO_MANUAL_CLICKS_APRILTAG_GT" if r["gt_source"] == "apriltag" \
            else "INSUFFICIENT_CLICK_EVIDENCE"
        r.update(verdict="AMBIGUOUS", reason=f"{why}(n={len(ev)})")
        return r, (R, t, "identity")

    # (1) 교차필드 / 내부일관 순열 검사
    errs = {k: float(np.max(np.linalg.norm(
        S[ev] - P[[PERMUTATIONS[k][i] for i in ev]], axis=1))) for k in PERMUTATIONS}
    e_id = errs["identity"]
    best = min(errs, key=errs.get)
    r["identity_max_px"] = round(e_id, 2)
    r["best_perm"] = best
    r["best_perm_max_px"] = round(errs[best], 2)
    cross = evsrc == "keypoint_annotations"
    r["test_power"] = "CROSS_FIELD" if cross else "INTERNAL_ONLY"

    face_bad = float(r["face_obliquity_deg"]) > FACE_EDGEON_DEG
    perm_bad = best != "identity" and errs[best] < CLICK_MATCH_PX and errs[best] < 0.5 * e_id

    if perm_bad and best in MIRROR_PERMS:
        r.update(verdict="LR_ORDER_VIOLATION",
                 reason=f"{best} fits {errs[best]:.1f}px vs identity {e_id:.1f}px")
    elif perm_bad and best in ROT_PERMS:
        r.update(verdict="YAW90_STALE",
                 reason=f"keypoint_annotations 는 {best} — pose/manual_kps 의 0123 면 "
                        f"기울기 {r['face_obliquity_deg']}deg"
                        f"{' (edge-on)' if face_bad else ''}")
    elif face_bad:
        r.update(verdict="YAW90_STALE",
                 reason=f"0123 면이 카메라를 향하지 않음(기울기 {r['face_obliquity_deg']}deg) "
                        f"— 교차 증거 없음")
    elif e_id > OK_PX:
        r.update(verdict="OTHER_DEFECT",
                 reason=f"HIGH_CLICK_RESIDUAL identity={e_id:.1f}px best={best}:{errs[best]:.1f}px")
    elif r["near_ok"] != "True" or r["top_ok"] != "True":
        r.update(verdict="OTHER_DEFECT",
                 reason=f"INVARIANT near_ok={r['near_ok']} top_ok={r['top_ok']}")
    elif r["defect_note"]:
        r.update(verdict="OTHER_DEFECT", reason=r["defect_note"])
    else:
        r.update(verdict="LABEL_OK", reason=f"identity={e_id:.1f}px, face={r['face_obliquity_deg']}deg")

    # keypoint_annotations 를 정본으로 삼았을 때 학습 라벨로 쓸 수 있는가.
    # (LABEL_OK 이거나, 어긋남이 yaw 순열 하나로 완전히 설명되는 YAW90_STALE)
    r["usable_if_kpann_canonical"] = str(
        (not r["defect_note"])
        and r["verdict"] in ("LABEL_OK", "YAW90_STALE")
        and (r["verdict"] == "LABEL_OK" or (cross and perm_bad)))
    Reff = R @ ry(PERM_TO_RY.get(best, 0.0)) if perm_bad else R
    return r, (Reff, t, best, R)


COLS = ["folder", "frame", "verdict", "elevation_deg", "n_click", "n_sentinel",
        "dims_wd", "reason", "image_exists",
        "group", "session", "object_type", "split", "gt_source", "pose_status",
        "n_none", "n_evidence", "evidence_source", "test_power", "has_kpann",
        "identity_max_px", "best_perm", "best_perm_max_px",
        "face_obliquity_deg", "near_ok", "top_ok", "lr_screen_flip", "is_square",
        "proj_diag_px", "dist_m", "stored_reproj_px", "defect_note",
        "neighbour_phase_flip", "usable_if_kpann_canonical", "image_path"]


def collect_targets():
    out, seen = [], set()

    def add(group, name, d):
        if d.exists() and d.resolve() not in seen:
            seen.add(d.resolve())
            out.append((group, name, d))

    lc = ROOT / "challenge/data/01_real/live_capture_gt"
    for d in sorted(p for p in lc.iterdir() if p.is_dir() and not p.name.startswith("_")):
        add("live_capture_gt", d.name, d)
    for rel in REAL_MANUAL_GT.values():
        add("REAL_MANUAL_GT", Path(rel).name, ROOT / rel)
    for rel in EVAL_CANONICAL.values():
        add("EVAL_CANONICAL", Path(rel).name, ROOT / rel)
    for sub in ("migrated_gt", "migrated_gt_wood"):
        base = ROOT / "challenge/real_gt_v2" / sub
        if base.exists():
            for d in sorted({p.parent for p in base.rglob("*.json")}):
                add(f"real_gt_v2/{sub}", str(d.relative_to(base)), d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pallet/results/accuracy_root_cause_v1")
    a = ap.parse_args()
    outdir = ROOT / a.out
    outdir.mkdir(parents=True, exist_ok=True)

    rows, poses = [], collections.defaultdict(list)
    for group, folder, d in collect_targets():
        for p in sorted(d.glob("*.json")):
            r, pose = audit_frame(p, group, folder)
            rows.append(r)
            if pose is not None:
                poses[(group, folder)].append(
                    (p.stem, pose[0], pose[1], r, pose[3] if len(pose) > 3 else pose[0]))

    # (3) 시간 일관성 — 같은 세션 인접 프레임의 C4 phase 튐
    c4 = [ry(90.0 * k) for k in range(4)]
    flips = {"raw_pose": 0, "kpann_phase": 0}
    pairs = 0
    for key, seq in poses.items():
        seq.sort(key=lambda x: x[0])
        for a, b in zip(seq, seq[1:]):
            if float(np.linalg.norm(a[2] - b[2])) > NEIGHBOUR_TRANS_M:
                continue
            pairs += 1
            for tag, ia in (("kpann_phase", 1), ("raw_pose", 4)):
                R1, R2 = a[ia], b[ia]
                d_id = geodesic_deg(R1, R2)
                d_c4 = min(geodesic_deg(R1, R2 @ C) for C in c4)
                if d_id > NEIGHBOUR_FLIP_DEG and d_c4 < NEIGHBOUR_C4_DEG:
                    flips[tag] += 1
                    if tag == "kpann_phase":
                        a[3]["neighbour_phase_flip"] = "True"
                        b[3]["neighbour_phase_flip"] = "True"
    print(f"neighbour pairs (<= {NEIGHBOUR_TRANS_M} m apart) = {pairs}  "
          f"phase flips: raw pose_transform = {flips['raw_pose']}, "
          f"keypoint_annotations phase = {flips['kpann_phase']}")
    for r in rows:
        if not r["neighbour_phase_flip"]:
            r["neighbour_phase_flip"] = "False"

    csv_path = outdir / "REAL_LABEL_AUDIT.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}  rows={len(rows)}")
    return rows


if __name__ == "__main__":
    main()
