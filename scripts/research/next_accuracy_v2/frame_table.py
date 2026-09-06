"""live_capture_gt 851장의 프레임 단위 표를 만든다 (§8 · §10).

한 프레임당: 앙각 · 거리 · 투영 크기 · **측정 밝기와 주야** · 물체 · 클릭 증거 수 ·
가림/잘림 · 재투영 오차 · pose 상태.  이 표 하나로 §8 의 GT 분류와 §10 의 앙각 집계를 낸다.

주야는 세션명의 시각으로 **추정하지 않는다** — 이미지의 실제 평균 휘도를 재서
정하고, 촬영 시각은 교차확인으로만 쓴다.  시계가 밤이어도 조명이 켜져 있을 수 있다.

앙각 정의는 `scripts/research/accuracy_root_cause_v1/elevation_check.py` 와 같다:
  상판 법선 n = R @ (0,-1,0),  팔레트->카메라 u = -t/|t|,  elev = asin(|n.u|)
  0도 = 완전 edge-on.
읽기 전용.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
GT_ROOT = REPO / "challenge/data/01_real/live_capture_gt"
CAPTURE_ROOT = REPO / "challenge/data/01_real/_live_captures"
BINS = [(0, 3), (3, 8), (8, 15), (15, 30), (30, 91)]


def bin_of(e):
    for lo, hi in BINS:
        if lo <= e < hi:
            return f"{lo}-{hi}" if hi < 91 else ">=30"
    return "?"


NIGHT_V = 60.0   # [추정][미검증] 사전등록된 값이 아니다.  실측 분포는 **단봉**이고
                 # 최솟값이 64.6 이라 60 이하 어떤 값을 골라도 NIGHT 은 0 이다.


def mean_luminance(img_path: Path):
    """이미지의 평균 휘도(0~255).  1/4 축소로 읽어 속도를 아낀다."""
    import cv2
    im = cv2.imread(str(img_path), cv2.IMREAD_REDUCED_COLOR_4)
    if im is None:
        return None
    return float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean())


def session_clock(name: str):
    """세션명에서 HHMMSS 를 뽑는다.  교차확인용이지 판정 근거가 아니다."""
    import re
    m = re.search(r"_(\d{8})_(\d{6})$", name) or re.search(r"_(\d{6})$", name)
    if not m:
        return None
    hhmmss = m.groups()[-1]
    return f"{hhmmss[:2]}:{hhmmss[2:4]}"


def session_group(name: str):
    hits = list(CAPTURE_ROOT.glob(f"*/sessions/{name}/rgb"))
    return hits[0].parents[2].name if len(hits) == 1 else None


def main() -> int:
    rows = []
    for gt_dir in sorted(GT_ROOT.glob("*_manual_gt")):
        sess = gt_dir.name[: -len("_manual_gt")]
        grp = session_group(sess)
        clock = session_clock(sess)
        hits = list(CAPTURE_ROOT.glob(f"*/sessions/{sess}/rgb"))
        rgb_root = hits[0] if len(hits) == 1 else None
        for jp in sorted(gt_dir.glob("*.json")):
            if not jp.stem.isdigit():
                continue
            doc = json.loads(jp.read_text(encoding="utf-8"))
            ob = doc["objects"][0]
            ann = ob.get("keypoint_annotations") or []
            src = Counter(e.get("source") for e in ann[:9])
            xy = [e.get("xy") for e in ann[:9] if e.get("xy") is not None]

            elev = dist = None
            T = ob.get("pose_transform")
            if T:
                T = np.asarray(T, float)
                R, t = T[:3, :3], T[:3, 3]
                if np.linalg.norm(t) > 1e-9:
                    n = R @ np.array([0.0, -1.0, 0.0])
                    u = -t / np.linalg.norm(t)
                    elev = float(np.degrees(np.arcsin(np.clip(abs(float(n @ u)), 0, 1))))
                    dist = float(np.linalg.norm(t))

            diag = None
            if len(xy) >= 4:
                a = np.asarray(xy, float)
                diag = float(np.hypot(a[:, 0].ptp(), a[:, 1].ptp()))

            rgb = rgb_root / f"{jp.stem}.png" if rgb_root else None
            lum = mean_luminance(rgb) if (rgb and rgb.is_file()) else None
            dims = ob.get("physical_dimensions_m") or ob.get("dimensions_m") or {}
            rows.append({
                "session": sess, "group": grp, "frame": jp.stem,
                "path": str(jp.relative_to(REPO)),
                "elev_deg": None if elev is None else round(elev, 2),
                "elev_bin": "?" if elev is None else bin_of(elev),
                "dist_m": None if dist is None else round(dist, 3),
                "proj_diag_px": None if diag is None else round(diag, 1),
                "mean_luminance": None if lum is None else round(lum, 1),
                "day_night": (None if lum is None
                              else ("NIGHT" if lum < NIGHT_V else "DAY")),
                "session_clock": clock,
                "object_type": ob.get("object_type") or doc.get("object_type"),
                "dims": dims,
                "n_manual_click": src.get("manual_click", 0),
                "n_pnp_projected": src.get("pnp_projected", 0),
                "n_centroid_auto": src.get("centroid_auto", 0),
                "n_unknown": src.get("unknown", 0),
                "n_xy_none": sum(1 for e in ann[:9] if e.get("xy") is None),
                "occlusion_level": ob.get("occlusion_level"),
                "truncation": ob.get("truncation"),
                "pose_status": ob.get("pose_status"),
                "reproj_error_px": ob.get("reproj_error_px"),
                "n_pose_candidates": len(ob.get("canonical_pose_candidates") or []),
                "extrapolated": sum(ob.get("extrapolated_mask") or []),
                "population_role": doc.get("population_role"),
            })

    dst = REPO / "data/pallet/results/next_accuracy_v2/LIVE_GT_FRAME_TABLE.json"
    dst.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"frames {len(rows)}   sessions {len({r['session'] for r in rows})}   "
          f"groups {len({r['group'] for r in rows})}")
    print("\n--- 앙각 구간 ---")
    c = Counter(r["elev_bin"] for r in rows)
    for k in ["0-3", "3-8", "8-15", "15-30", ">=30", "?"]:
        if c.get(k):
            s = len({r["session"] for r in rows if r["elev_bin"] == k})
            print(f"  {k:<7}{c[k]:>5} 프레임   {s:>3} 세션")
    print("\n--- 클릭 증거 수 분포 (manual_click 개수) ---")
    for k, v in sorted(Counter(r["n_manual_click"] for r in rows).items()):
        print(f"  {k} 클릭 {v:>5}")
    print("\n--- 물체 치수 ---")
    for k, v in Counter(
            f"{r['dims'].get('x')}x{r['dims'].get('y')}x{r['dims'].get('z')}"
            if isinstance(r["dims"], dict) else str(r["dims"]) for r in rows).items():
        print(f"  {k:<26}{v}")
    for f in ("occlusion_level", "truncation", "pose_status", "population_role"):
        print(f"\n--- {f} ---")
        for k, v in Counter(str(r[f]) for r in rows).most_common(6):
            print(f"  {k:<26}{v}")
    print(f"\nwrote {dst.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
