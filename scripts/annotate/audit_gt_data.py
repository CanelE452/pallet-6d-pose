"""저장된 GT 를 검사한다 — 어노테이션 **결과물** 감사.  모델 예측 미사용.

`_audit_annotate.py` 는 어노 **코드**를 검사하고, 이 도구는 **저장된 JSON** 을 검사한다.
둘은 다른 것을 잡는다: 코드가 옳아도 저장 시점에 PnP 가 실패했거나 옛 값이 남아 있으면
파일은 깨진 채로 남는다.

## 왜 새로 만들었나 — 기존 도구가 못 잡는 것

`REAL_GT_QA_.../qa_risk.py`(repo 밖, 일회성)는 R1 스키마 + 클릭점에서 PnP 재풀이 +
LOO + robust-z 를 한다.  좋은 도구지만 **저장된 `pose_transform` 자체를 검사하지 않는다** —
점에서 다시 풀 뿐이다.  그래서 "PnP 가 제대로 안 된 채로 저장된" 경우를 놓친다.

이 도구의 핵심은 T2 다:

    저장된 pose_transform + dims + K 로 3D 모델을 투영한 결과가
    저장된 projected_cuboid 와 일치하는가?

불일치하면 **파일 안의 두 값이 서로 다른 것을 말하고 있다**는 뜻이고, 어느 쪽을 믿고
평가했느냐에 따라 결과가 갈린다.

## 판정 계층

    T1 SCHEMA          모양·유한성·sentinel·중복·화면밖·회전행렬 유효성
    T2 STORED POSE ★   저장 pose <-> 저장 projection 일치 / 깊이 양수(cheirality)
    T3 RESOLVE         점에서 PnP 재풀이 -> 잔차, 저장 pose 와의 R/t 차이, LOO
    T4 GEOMETRY        dims W/D 스왑 적합성, centroid 일치, 앞/뒤 면 깊이 순서
    T5 OUTLIER         정규화 잔차의 robust z (셋 안에서 튀는 프레임)

severity 는 RED / AMBER / GREEN 이고, **왜 그렇게 판정했는지 사유를 함께 남긴다.**
자동 수정은 하지 않는다 — memory `annotate-tool-audit-and-sentinel-gt-damage` 의 교훈대로
복구 스크립트는 "이미 맞는 값" 에 같은 계산을 돌려 검증한 뒤에만 적용한다.

## 사용

    python scripts/annotate/audit_gt_data.py --glob "challenge/data/01_real/**/*_manual_gt"
    python scripts/annotate/audit_gt_data.py --dir challenge/data/01_real/manual_gt/capturenight09_manual_gt
    ... --overlays          RED 프레임 오버레이 PNG 도 저장 (눈으로 확인용)
"""
from __future__ import annotations

import argparse
import csv
import glob as globmod
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts/annotate"))

import cv2                        # noqa: E402
import annotate_pnp as APNP       # noqa: E402

SENTINEL_TOL = 1e-6
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


# ---------------------------------------------------------------- 도움 함수

def load_K(label):
    k = label["camera_data"]["intrinsics"]
    return np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]],
                    float)


def model_points(dims):
    """9점 (8 corner + centroid) — 어노 툴과 같은 함수를 쓴다."""
    return APNP.make_pallet_keypoints_3d_diagram(
        width=dims["width"], depth=dims["depth"], height=dims["height"])


def project(R, t, X, K):
    cam = (R @ np.asarray(X, float).T).T + np.asarray(t, float)
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (K @ cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def solve_pnp(obj, img, K):
    if len(obj) < 4:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj.astype(np.float64), img.astype(np.float64),
                                  K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj.astype(np.float64), img.astype(np.float64),
                                      K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def geodesic_deg(A, B):
    c = (np.trace(np.asarray(A).T @ np.asarray(B)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def is_click(p):
    """정수 좌표 = 사람 클릭, 소수 = 외삽.  표식이지 증거는 아니다(qa_risk 와 동일 정의)."""
    return bool(abs(p[0] - round(p[0])) < 1e-9 and abs(p[1] - round(p[1])) < 1e-9)


# ---------------------------------------------------------------- 프레임 감사

def audit_frame(path, tol_stored=1.0, tol_gross=10.0,
                tol_pose_deg=0.5, tol_pose_m=0.01):
    row = {"file": os.path.relpath(path, ROOT),
           "folder": os.path.basename(os.path.dirname(path)),
           "frame": os.path.splitext(os.path.basename(path))[0]}
    flags = []

    try:
        lab = json.load(open(path))
    except Exception as e:
        row.update(severity="RED", reasons=f"JSON_PARSE_FAIL:{type(e).__name__}")
        return row, None
    objs = lab.get("objects") or []
    if not objs:
        row.update(severity="RED", reasons="NO_OBJECT")
        return row, None
    o = objs[0]

    # ---------- T1 SCHEMA ----------
    proj = np.asarray(o.get("projected_cuboid") or [], float)
    dims = o.get("dimensions_m") or {}
    row["stored_reproj_error_px"] = o.get("reproj_error_px")
    row["gt_source"] = o.get("gt_source")
    row["split"] = o.get("split")

    if proj.shape != (8, 2):
        flags.append(f"BAD_PROJ_SHAPE_{proj.shape}")
    if proj.size and not np.isfinite(proj).all():
        flags.append("NONFINITE_PROJ")
    if not dims or min(dims.values()) <= 0:
        flags.append("BAD_DIMS")
    if o.get("reproj_error_px") is None:
        flags.append("NO_REPROJ_FIELD")
    elif float(o["reproj_error_px"]) < 0:
        flags.append("REPROJ_NEGATIVE")      # 툴이 계산 못 한 채 저장된 표식
    mk = o.get("manual_kps")
    if mk is None:
        flags.append("NO_MANUAL_KPS")
    elif len(mk) != 9:
        flags.append(f"MANUAL_KPS_LEN_{len(mk)}")
    elif any(e is None for e in mk):
        flags.append(f"MANUAL_KPS_NONE_x{sum(1 for e in mk if e is None)}")

    sent = []
    if proj.shape == (8, 2):
        sent = [i for i in range(8)
                if abs(proj[i][0] + 1) < SENTINEL_TOL and abs(proj[i][1] + 1) < SENTINEL_TOL]
        if sent:
            flags.append(f"SENTINEL_x{len(sent)}")
        w = lab["camera_data"]["width"]
        h = lab["camera_data"]["height"]
        far = [i for i in range(8) if i not in sent and
               (proj[i][0] < -3 * w or proj[i][0] > 4 * w
                or proj[i][1] < -3 * h or proj[i][1] > 4 * h)]
        if far:
            flags.append(f"FAR_OUTSIDE_x{len(far)}")
        dup = [(a, b) for a in range(8) for b in range(a + 1, 8)
               if a not in sent and b not in sent
               and np.linalg.norm(proj[a] - proj[b]) < 1e-6]
        if dup:
            flags.append(f"DUPLICATE_POINT_x{len(dup)}")
    row["n_sentinel"] = len(sent)

    pt = np.asarray(o.get("pose_transform") or [], float)
    R_st = t_st = None
    if pt.shape != (4, 4) or not np.isfinite(pt).all():
        flags.append("BAD_POSE_SHAPE")
    else:
        R_st, t_st = pt[:3, :3], pt[:3, 3]
        orth = float(np.abs(R_st.T @ R_st - np.eye(3)).max())
        det = float(np.linalg.det(R_st))
        row["R_orthonormality_err"] = round(orth, 6)
        row["R_det"] = round(det, 6)
        if orth > 1e-3:
            flags.append(f"R_NOT_ORTHONORMAL_{orth:.2e}")
        if abs(det - 1.0) > 1e-3:
            flags.append(f"R_DET_{det:.4f}")

    hard = bool(flags)
    use = [i for i in range(8) if i not in sent] if proj.shape == (8, 2) else []
    X9 = model_points(dims) if dims and min(dims.values()) > 0 else None
    K = load_K(lab)

    # ---------- T2 STORED POSE (★ 핵심) ----------
    if R_st is not None and X9 is not None and use:
        uv, z = project(R_st, t_st, X9[:8], K)
        d = np.linalg.norm(uv[use] - proj[use], axis=1)
        row["stored_pose_reproj_med"] = round(float(np.median(d)), 3)
        row["stored_pose_reproj_max"] = round(float(np.max(d)), 3)
        # ★절대 잔차로 판정하지 않는다.  클릭이 2~5px 어긋나는 건 흔하고, 그건
        #   "저장이 잘못됐다" 가 아니라 어노테이션 정밀도다.  T3 에서 같은 점으로
        #   다시 푼 해와 비교해 **저장 pose 가 점이 허용하는 최선보다 나쁜가**를 본다.
        #   (첫 판에서 이걸 절대 2px 로 잡아 정상 프레임 15/27 을 RED 로 오탐했다.)
        nb = int((z[use] <= 0).sum())
        row["n_behind_camera"] = nb
        if nb:
            flags.append(f"BEHIND_CAMERA_x{nb}")

    # ---------- T3 RESOLVE ----------
    diag = np.nan
    if use:
        g = proj[use]
        diag = float(np.hypot(g[:, 0].ptp(), g[:, 1].ptp()))
    row["bbox_diag_px"] = round(diag, 2) if np.isfinite(diag) else ""

    if X9 is not None and len(use) >= 4:
        pose = solve_pnp(X9[use], proj[use], K)
        if pose is None:
            flags.append("PNP_RESOLVE_FAIL")
        else:
            R_re, t_re = pose
            uv, _ = project(R_re, t_re, X9[use], K)
            res = np.linalg.norm(uv - proj[use], axis=1)
            row["resolve_reproj_med"] = round(float(np.median(res)), 3)
            row["resolve_reproj_p90"] = round(float(np.percentile(res, 90)), 3)
            # 클릭 부정확은 정상 범위가 넓다 — gross 수준에서만 hard 로 본다.
            # 중간 구간은 T5 의 robust z 가 셋 안에서 상대적으로 잡는다.
            if np.median(res) > tol_gross:
                flags.append(f"GROSS_RESIDUAL_{np.median(res):.1f}px")
            if R_st is not None:
                dR = geodesic_deg(R_st, R_re)
                dt = float(np.linalg.norm(np.asarray(t_st) - t_re))
                row["stored_vs_resolved_R_deg"] = round(dR, 3)
                row["stored_vs_resolved_t_m"] = round(dt, 4)
                # ★진짜 "PnP 가 제대로 안 된 채 저장" 의 서명:
                #   저장 pose 가 점에서 푼 해와 다르다.
                excess = row["stored_pose_reproj_med"] - row["resolve_reproj_med"] \
                    if "stored_pose_reproj_med" in row else 0.0
                row["stored_pose_excess_px"] = round(float(excess), 3)
                if dR > tol_pose_deg or dt > tol_pose_m or excess > tol_stored:
                    flags.append(
                        f"STORED_POSE_MISMATCH_dR{dR:.2f}deg_dt{dt:.3f}m_excess{excess:.1f}px")
            loo = []
            for j in use:
                rest = [k for k in use if k != j]
                p2 = solve_pnp(X9[rest], proj[rest], K) if len(rest) >= 4 else None
                if p2 is None:
                    loo.append(np.nan)
                    continue
                uvj, _ = project(p2[0], p2[1], X9[[j]], K)
                loo.append(float(np.linalg.norm(uvj[0] - proj[j])))
            loo = np.array(loo, float)
            if np.isfinite(loo).any():
                row["max_loo_px"] = round(float(np.nanmax(loo)), 3)
                row["max_loo_kp"] = int(use[int(np.nanargmax(loo))])

    # ---------- T4 GEOMETRY ----------
    if X9 is not None and len(use) >= 4 and dims:
        sw = {"width": dims["depth"], "depth": dims["width"], "height": dims["height"]}
        if abs(sw["width"] - dims["width"]) > 1e-9:
            Xs = model_points(sw)
            ps = solve_pnp(Xs[use], proj[use], K)
            if ps is not None:
                uvs, _ = project(ps[0], ps[1], Xs[use], K)
                e_sw = float(np.median(np.linalg.norm(uvs - proj[use], axis=1)))
                e_now = row.get("resolve_reproj_med", np.inf)
                row["swapped_dims_reproj_med"] = round(e_sw, 3)
                if np.isfinite(e_now) and e_sw < e_now * 0.5:
                    flags.append(f"DIMS_SWAP_SUSPECT_{e_sw:.1f}vs{e_now:.1f}")
    cen = o.get("projected_cuboid_centroid")
    if cen and R_st is not None and X9 is not None:
        uvc, _ = project(R_st, t_st, X9[8:9], K)
        row["centroid_err_px"] = round(float(np.linalg.norm(uvc[0] - np.asarray(cen, float))), 3)
        if row["centroid_err_px"] > 5.0:
            flags.append(f"CENTROID_MISMATCH_{row['centroid_err_px']:.1f}px")
    if R_st is not None and X9 is not None:
        _, z = project(R_st, t_st, X9[:8], K)
        if np.isfinite(z).all() and z[:4].mean() > z[4:].mean():
            flags.append("NEAR_FACE_IS_FARTHER")   # 0~3 이 near 여야 한다

    if mk and isinstance(mk, list):
        row["n_click"] = int(sum(1 for e in mk[:8]
                                 if e is not None and is_click(e)))

    row["hard_flags"] = ";".join(flags)
    row["_hard"] = hard or any(f.startswith(("STORED_POSE_MISMATCH", "GROSS_RESIDUAL",
                                             "BEHIND_CAMERA", "PNP_RESOLVE_FAIL",
                                             "DIMS_SWAP_SUSPECT", "NEAR_FACE_IS_FARTHER"))
                               for f in flags)
    return row, (proj, R_st, t_st, X9, K, use)


# ---------------------------------------------------------------- 집계·출력

def robust_z(rows, key):
    v = np.array([r.get(key, np.nan) if isinstance(r.get(key), (int, float))
                  else np.nan for r in rows], float)
    m = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - m))
    s = 1.4826 * mad if mad > 0 else np.nanstd(v)
    return (v - m) / s if s and np.isfinite(s) and s > 0 else np.zeros_like(v)


def draw_overlay(path, payload, out_png):
    proj, R_st, t_st, X9, K, use = payload
    img = cv2.imread(path.replace(".json", ".png"))
    if img is None:
        return False
    if R_st is not None and X9 is not None:
        uv, _ = project(R_st, t_st, X9[:8], K)
        for a, b in EDGES:
            if np.isfinite(uv[a]).all() and np.isfinite(uv[b]).all():
                cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)),
                         (0, 0, 255), 2, cv2.LINE_AA)          # red = 저장 pose
    for i in use:
        cv2.circle(img, tuple(proj[i].astype(int)), 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(img, str(i), tuple((proj[i] + 5).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_png, img)                                   # green = 저장 projection
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", default=[],
                    help='감사할 폴더 glob (예: "challenge/data/01_real/**/*_manual_gt")')
    ap.add_argument("--dir", action="append", default=[], help="감사할 폴더 (여러 번 가능)")
    ap.add_argument("--out", default="_docs/audits/gt_data_audit")
    ap.add_argument("--tol-stored", type=float, default=1.0,
                    help="저장 pose 가 재풀이 해보다 얼마나 더 나빠도 되는가 (px, excess)")
    ap.add_argument("--tol-gross", type=float, default=10.0,
                    help="점에서 재풀이한 PnP 잔차의 gross 임계 (px)")
    ap.add_argument("--tol-pose-deg", type=float, default=0.5,
                    help="저장 pose vs 재풀이 pose 회전 차이 허용 (deg)")
    ap.add_argument("--tol-pose-m", type=float, default=0.01,
                    help="저장 pose vs 재풀이 pose 평행이동 차이 허용 (m)")
    ap.add_argument("--overlays", action="store_true", help="RED 프레임 오버레이 저장")
    a = ap.parse_args()

    dirs = list(a.dir)
    for g in a.glob:
        dirs += [d for d in globmod.glob(os.path.join(ROOT, g), recursive=True)
                 if os.path.isdir(d)]
    if not dirs:
        raise SystemExit("감사할 폴더가 없다 — --glob 또는 --dir 을 줄 것")
    files = sorted({p for d in dirs for p in globmod.glob(os.path.join(d, "*.json"))})
    print(f"폴더 {len(set(dirs))}개 · JSON {len(files)}개", flush=True)

    out_dir = os.path.join(ROOT, a.out)
    os.makedirs(out_dir, exist_ok=True)
    rows, payloads = [], {}
    for i, p in enumerate(files):
        r, pl = audit_frame(p, a.tol_stored, a.tol_gross,
                            a.tol_pose_deg, a.tol_pose_m)
        rows.append(r)
        if pl is not None:
            payloads[r["file"]] = (p, pl)
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    # ---------- T5 OUTLIER ----------
    for key, zname in (("resolve_reproj_med", "z_resolve"),
                       ("stored_pose_reproj_med", "z_stored")):
        z = robust_z(rows, key)
        for r, v in zip(rows, z):
            r[zname] = round(float(v), 2) if np.isfinite(v) else ""

    for r in rows:
        zz = max([abs(r[k]) for k in ("z_resolve", "z_stored")
                  if isinstance(r.get(k), float)] or [0.0])
        if r.get("severity") == "RED":
            continue
        if r.pop("_hard", False):
            r["severity"] = "RED"
        elif zz > 5:
            r["severity"] = "RED"
        elif zz > 3 or r.get("hard_flags"):
            r["severity"] = "AMBER"
        else:
            r["severity"] = "GREEN"
        if not r.get("reasons"):
            r["reasons"] = r.get("hard_flags") or (f"robust_z={zz:.1f}" if zz > 3 else "")
    for r in rows:
        r.pop("_hard", None)

    fields = sorted({k for r in rows for k in r},
                    key=lambda k: (k not in ("file", "folder", "frame", "severity",
                                             "reasons"), k))
    with open(os.path.join(out_dir, "GT_DATA_AUDIT.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: ({"RED": 0, "AMBER": 1, "GREEN": 2}
                                                [r["severity"]], r["file"])))

    import collections
    sev = collections.Counter(r["severity"] for r in rows)
    reason = collections.Counter()
    for r in rows:
        for fl in (r.get("hard_flags") or "").split(";"):
            if fl:
                reason[fl.split("_x")[0].split("_", 2)[0] if False else fl.split("_")[0]] += 1
    summary = {"n_files": len(files), "severity": dict(sev),
               "flag_families": dict(reason.most_common()),
               "thresholds": {"tol_stored_px": a.tol_stored,
                              "tol_gross_px": a.tol_gross,
                              "tol_pose_deg": a.tol_pose_deg,
                              "tol_pose_m": a.tol_pose_m,
                              "robust_z_red": 5, "robust_z_amber": 3},
               "checks": {
                   "T1": "schema · sentinel · duplicate · far-outside · R 유효성",
                   "T2": "★저장 pose_transform 을 투영해 저장 projected_cuboid 와 대조 + cheirality",
                   "T3": "점에서 PnP 재풀이 · 저장 pose 와의 R/t 차이 · LOO",
                   "T4": "dims W/D 스왑 적합성 · centroid 일치 · near/far 면 깊이 순서",
                   "T5": "정규화 잔차 robust z"},
               "note": "자동 수정하지 않는다.  복구는 '이미 맞는 값' 검증 후에만."}
    json.dump(summary, open(os.path.join(out_dir, "GT_DATA_AUDIT_SUMMARY.json"), "w"),
              indent=1, ensure_ascii=False)

    if a.overlays:
        od = os.path.join(out_dir, "overlays_red")
        os.makedirs(od, exist_ok=True)
        n = 0
        for r in rows:
            if r["severity"] != "RED" or r["file"] not in payloads:
                continue
            p, pl = payloads[r["file"]]
            if draw_overlay(p, pl, os.path.join(
                    od, f"{r['folder']}__{r['frame']}.png")):
                n += 1
        print(f"  overlays(RED) {n} -> {od}", flush=True)

    print(f"\nseverity: {dict(sev)}")
    print("상위 flag:")
    for k, v in reason.most_common(12):
        print(f"  {k:26} {v}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
