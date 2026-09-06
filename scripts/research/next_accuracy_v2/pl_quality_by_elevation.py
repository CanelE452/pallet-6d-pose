"""저앙각 pseudo-label 품질 감사 (§17 · §18) — 새 추론 0회.

`data/pallet/results/paper_selftrain_v1/M4_FRAME_RECORDS.json` 은 teacher(R0, sha
970a0913…) 의 프레임별 기록을 이미 담고 있다 — 코너 오차, box_conf, s_reproj /
s_remove / s_flip, 필터별 통과 판정.  거기에 **앙각만 붙여서** 층화한다.

앙각 정의는 `scripts/research/accuracy_root_cause_v1/elevation_check.py` 와 같다.
threshold 는 새로 만들지 않는다 — `PSEUDOLABEL_FILTER_LOCK.json` 과
`metric_split_lock.md` §2.2 (gross 20px / catastrophic 40px) 가 이미 동결한 값이다.

★모집단 주의: 이것은 **논문 트랙**(PAPER_EVAL plastic 194장, 직사각 물체) 이다.
§11 의 과제 트랙(정사각 물체 live_capture_gt)과 섞지 말 것.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
REC = REPO / "data/pallet/results/paper_selftrain_v1/M4_FRAME_RECORDS.json"
WS = REPO / "data/evaluation/pallet_eval_v1"
GROSS, CATA = 20.0, 40.0
BANDS = [("<8", 0, 8), ("8-15", 8, 15), (">=15", 15, 91)]


def elevation_index():
    idx = {}
    for p in WS.rglob("annotations/*/*.json"):
        try:
            ob = json.loads(p.read_text(encoding="utf-8"))["objects"][0]
        except Exception:
            continue
        T = ob.get("pose_transform")
        if not T:
            continue
        T = np.asarray(T, float)
        R, t = T[:3, :3], T[:3, 3]
        if np.linalg.norm(t) < 1e-9:
            continue
        n = R @ np.array([0.0, -1.0, 0.0])
        u = -t / np.linalg.norm(t)
        idx[f"{p.parent.name}:{p.stem}"] = float(
            np.degrees(np.arcsin(np.clip(abs(float(n @ u)), 0, 1))))
    return idx


def band_of(e):
    for name, lo, hi in BANDS:
        if lo <= e < hi:
            return name
    return "?"


def stats(rows, key="errors_px"):
    e = [v for r in rows for v in (r.get(key) or []) if v is not None]
    if not e:
        return None
    a = np.array(e, float)
    return {"n_frames": len(rows), "n_kp": len(a),
            "median_px": round(float(np.median(a)), 2),
            "p90_px": round(float(np.percentile(a, 90)), 2),
            "gross20_pct": round(float((a > GROSS).mean() * 100), 1),
            "gross40_pct": round(float((a > CATA).mean() * 100), 1)}


def main() -> int:
    doc = json.loads(REC.read_text(encoding="utf-8"))
    frames, arms = doc["frames"], doc["arms"]
    elev = elevation_index()

    missing = 0
    for f in frames:
        e = elev.get(f["frame_id"])
        if e is None:
            missing += 1
        f["elev_deg"] = e
        f["band"] = "?" if e is None else band_of(e)

    out = {"schema_version": "next_accuracy_v2_pl_quality_by_elevation_v1",
           "population": doc["population"],
           "★population_note": "논문 트랙(직사각 물체). §11 의 정사각 과제 트랙과 다른 모집단이다.",
           "teacher_sha256": doc["teacher_sha256"],
           "filter_lock_sha256": doc["filter_lock_sha256"],
           "new_inference": 0, "frames_without_elevation": missing,
           "thresholds": {"gross_px": GROSS, "catastrophic_px": CATA,
                          "source": "metric_split_lock.md §2.2 [LOCKED]"},
           "by_band": {}, "filters": {}}

    print(f"프레임 {len(frames)}   앙각 못 구함 {missing}")
    print("\n=== RAW teacher 품질 (필터 전) ===")
    print(f"{'층':<8}{'프레임':>6}{'검출':>6}{'kp':>6}{'med':>8}{'p90':>8}"
          f"{'>20px':>8}{'>40px':>8}")
    for name, _, _ in BANDS:
        rows = [f for f in frames if f["band"] == name and f["detected"]]
        allrows = [f for f in frames if f["band"] == name]
        s = stats(rows)
        out["by_band"][name] = {"n_frames": len(allrows),
                                "n_detected": len(rows), "raw": s}
        if s:
            print(f"{name:<8}{len(allrows):>6}{len(rows):>6}{s['n_kp']:>6}"
                  f"{s['median_px']:>8.2f}{s['p90_px']:>8.2f}"
                  f"{s['gross20_pct']:>7.1f}%{s['gross40_pct']:>7.1f}%")

    print("\n=== 필터별 · 층별 (통과분만) ===")
    hdr = f"{'filter':<18}{'층':<7}{'통과':>6}{'coverage':>10}{'med':>8}{'p90':>8}{'>20px':>8}{'>40px':>8}"
    print(hdr)
    for arm in arms:
        out["filters"][arm] = {}
        for name, _, _ in BANDS:
            pool = [f for f in frames if f["band"] == name and f["detected"]]
            kept = [f for f in pool if f["verdict"].get(arm)]
            rej = [f for f in pool if not f["verdict"].get(arm)]
            sk, sr = stats(kept), stats(rej)
            cov = len(kept) / len(pool) if pool else 0.0
            out["filters"][arm][name] = {
                "n_pool": len(pool), "n_kept": len(kept),
                "coverage": round(cov, 3), "pass": sk, "reject": sr}
            if sk:
                print(f"{arm:<18}{name:<7}{len(kept):>6}{cov*100:>9.1f}%"
                      f"{sk['median_px']:>8.2f}{sk['p90_px']:>8.2f}"
                      f"{sk['gross20_pct']:>7.1f}%{sk['gross40_pct']:>7.1f}%")

    print("\n=== §18 핵심 질문 ===")
    q = {}
    for arm in arms:
        for name, _, _ in BANDS:
            d = out["filters"][arm][name]
            if not d["pass"] or not d["reject"]:
                continue
            q[f"{arm}|{name}"] = {
                "gross20_before": out["by_band"][name]["raw"]["gross20_pct"],
                "gross20_after": d["pass"]["gross20_pct"],
                "separation_px": round(d["reject"]["median_px"] - d["pass"]["median_px"], 2),
                "coverage": d["coverage"]}
    out["key_questions"] = q
    for name, _, _ in BANDS:
        k = f"F4_PROPOSED|{name}"
        if k in q:
            v = q[k]
            print(f"  {name:<7} gross20 {v['gross20_before']:5.1f}% -> "
                  f"{v['gross20_after']:5.1f}%   통과/기각 분리 "
                  f"{v['separation_px']:+.2f} px   coverage {v['coverage']*100:.1f}%")

    dst = REPO / "data/pallet/results/next_accuracy_v2/PL_QUALITY_BY_ELEVATION.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {dst.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
