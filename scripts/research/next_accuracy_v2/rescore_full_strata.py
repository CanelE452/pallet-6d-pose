"""§11 이 요구한 층 전부 + §9 의 4개 수 + METHOD_LOCK 의 secondary metric 을 낸다.

새 학습 0.  기존 체크포인트를 다시 채점만 한다.

빠져 있던 것 (판별자 지적):
  §11  ALL / >=15 / DAY / NIGHT 층 없음, 사유도 없음
  §9   N_total / N_metric_eligible / N_excluded_ambiguous / N_excluded_suspect 병기 없음
  §11  contract − legacy 짝지은 CI 가 산출물에 없다 (문서에만 있었다)
  LOCK secondary 중 NME · pnp_projected 층 · keypoint 단위 gross 미측정

주야는 세션명 시각이 아니라 **측정 휘도**로 정한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DRV = REPO / "challenge/yolo_pose_one_model/next_accuracy_v2"
sys.path.insert(0, str(DRV))
import run_corrected_real_ft as S1  # noqa: E402

RESULTS = REPO / "data/pallet/results/next_accuracy_v2"
ARMS = {"R0": S1.BASE,
        "legacy": DRV / "legacy_s0/weights/best.pt",
        "contract": DRV / "contract_s0/weights/best.pt"}


def build_index():
    """held-out 프레임에 층 정보를 붙인다 — 적격/비적격 keypoint 를 모두 들고 온다."""
    table = {f"{r['session']}_manual_gt/{r['frame']}": r
             for r in json.loads((RESULTS / "LIVE_GT_FRAME_TABLE.json")
                                 .read_text(encoding="utf-8"))}
    part = json.loads((RESULTS / "GT_PARTITION.json").read_text(encoding="utf-8"))
    held = [r for r in part if r["split_role"] == "HELD_OUT"]
    lums = sorted(table[r["frame_id"]]["mean_luminance"] for r in held
                  if table[r["frame_id"]]["mean_luminance"] is not None)
    q1, q2 = np.percentile(lums, [33.3, 66.7])

    gt_root = REPO / "challenge/data/01_real/live_capture_gt"
    idx = {}
    for r in held:
        folder = f"{r['session']}_manual_gt"
        stem = f"{folder}__{r['frame']}"
        doc = json.loads((gt_root / folder / f"{r['frame']}.json").read_text(encoding="utf-8"))
        ann = doc["objects"][0]["keypoint_annotations"][:9]
        xy = np.full((9, 2), np.nan)
        elig = np.zeros(9, bool)
        derived = np.zeros(9, bool)
        for i, e in enumerate(ann):
            if e.get("xy") is not None:
                xy[i] = e["xy"]
                elig[i] = (e.get("source") == "manual_click")
                derived[i] = e.get("source") in ("pnp_projected", "centroid_auto")
        t = table[r["frame_id"]]
        lum = t["mean_luminance"]
        idx[stem] = {
            "xy": xy, "eligible": elig, "derived": derived,
            "band": ("<8" if r["elev_bin"] in ("0-3", "3-8")
                     else ("8-15" if r["elev_bin"] == "8-15" else ">=15")),
            "day_night": t["day_night"], "mean_luminance": lum,
            "illum": ("LOW" if lum <= q1 else ("MID" if lum <= q2 else "HIGH")),
            "session": r["session"], "eligibility_2d": r["eligibility_2d"],
        }
    return idx, (float(q1), float(q2))


def evaluate(weights, idx):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    stems = sorted(idx)
    rows = []
    for i in range(0, len(stems), 32):
        cs = stems[i:i + 32]
        paths = [str(REPO / "challenge/yolo_pose_one_model/datasets/live_gt_contract_v2"
                     / "images/val" / f"{s}.png") for s in cs]
        for s, res in zip(cs, model.predict(paths, imgsz=640, verbose=False, device=0)):
            g = idx[s]
            rec = {"stem": s, **{k: g[k] for k in
                                 ("band", "day_night", "illum", "session",
                                  "mean_luminance", "eligibility_2d")},
                   "detected": False, "elig_err": [], "derived_err": [], "nme": None}
            if res.keypoints is not None and len(res.boxes):
                k = int(np.argmax(res.boxes.conf.cpu().numpy()))
                kp = res.keypoints.xy.cpu().numpy()[k] - S1.PAD
                d = np.linalg.norm(kp - g["xy"], axis=1)
                b = res.boxes.xyxy.cpu().numpy()[k]
                diag = float(np.hypot(b[2] - b[0], b[3] - b[1])) or 1.0
                rec["detected"] = True
                rec["elig_err"] = [float(d[j]) for j in range(9)
                                   if g["eligible"][j] and np.isfinite(d[j])]
                rec["derived_err"] = [float(d[j]) for j in range(9)
                                      if g["derived"][j] and np.isfinite(d[j])]
                if rec["elig_err"]:
                    rec["nme"] = float(np.median(rec["elig_err"]) / diag)
            rows.append(rec)
    return rows


def summarise(rows, key, value, idx):
    sel = [r for r in rows if r[key] == value] if key else rows
    e = [v for r in sel for v in r["elig_err"]]
    dv = [v for r in sel for v in r["derived_err"]]
    nme = [r["nme"] for r in sel if r["nme"] is not None]
    n_amb = sum(len(r["derived_err"]) for r in sel)
    return {
        # §9 가 요구한 네 수
        "N_total": len(sel),
        "N_metric_eligible": len(e),
        "N_excluded_ambiguous": n_amb,
        "N_excluded_suspect": sum(1 for r in sel if r["eligibility_2d"] == "GT_SUSPECT"),
        "detection": f"{sum(r['detected'] for r in sel)}/{len(sel)}",
        "median_px": round(float(np.median(e)), 3) if e else None,
        "p90_px": round(float(np.percentile(e, 90)), 3) if e else None,
        "gross25_kp_pct": round(float((np.array(e) > 25).mean() * 100), 2) if e else None,
        "nme_median": round(float(np.median(nme)), 5) if nme else None,
        "derived_median_px": round(float(np.median(dv)), 3) if dv else None,
    }


def main() -> int:
    idx, (q1, q2) = build_index()
    print(f"held-out {len(idx)}   조도 삼분위 경계 {q1:.1f} / {q2:.1f}")
    dn = {}
    for v in idx.values():
        dn[v["day_night"]] = dn.get(v["day_night"], 0) + 1
    print(f"주야(측정 휘도 기준): {dn}")

    raw = {name: evaluate(w, idx) for name, w in ARMS.items()}
    per = {n: {r["stem"]: (float(np.median(r["elig_err"])) if r["elig_err"] else np.nan)
               for r in rows} for n, rows in raw.items()}

    strata = ([(None, "ALL")] + [("band", b) for b in ("<8", "8-15", ">=15")]
              + [("day_night", d) for d in ("DAY", "NIGHT")]
              + [("illum", i) for i in ("LOW", "MID", "HIGH")])
    table = {}
    for arm, rows in raw.items():
        table[arm] = {label: summarise(rows, key, label if key else None, idx)
                      for key, label in strata}

    # 짝지은 차이 — 세 쌍 전부 (contract−legacy 가 그동안 산출물에 없었다)
    sess = {r["stem"]: r["session"] for r in raw["R0"]}
    band = {r["stem"]: r for r in raw["R0"]}
    deltas = {}
    for a, b in (("contract", "R0"), ("legacy", "R0"), ("contract", "legacy")):
        deltas[f"{a}_minus_{b}"] = {}
        for key, label in strata:
            pairs = []
            for stem in per["R0"]:
                if key and band[stem][key] != label:
                    continue
                x, y = per[a][stem], per[b][stem]
                if np.isfinite(x) and np.isfinite(y):
                    pairs.append((sess[stem], y - x))   # 양수 = a 우세
            ci = S1.session_cluster_ci(pairs) if len(pairs) >= 3 else None
            deltas[f"{a}_minus_{b}"][label] = ci

    out = {"schema_version": "next_accuracy_v2_full_strata_v1",
           "new_training": 0, "checkpoints": {k: str(v) for k, v in ARMS.items()},
           "day_night_basis": "측정 평균 휘도 (세션명 시각이 아님). 경계 60.0",
           "illum_tercile_bounds": [round(q1, 1), round(q2, 1)],
           "day_night_counts": dn,
           "per_arm_by_stratum": table,
           "paired_deltas_positive_means_first_better": deltas}
    dst = RESULTS / "FULL_STRATA_RESCORE.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'층':<8}{'N':>5}{'적격kp':>7}{'제외모호':>9}"
          + "".join(f"{a:>12}" for a in ("R0", "legacy", "contract")))
    for key, label in strata:
        s0 = table["R0"][label]
        if not s0["N_total"]:
            print(f"{label:<8}{0:>5}   (해당 프레임 없음)")
            continue
        print(f"{label:<8}{s0['N_total']:>5}{s0['N_metric_eligible']:>7}"
              f"{s0['N_excluded_ambiguous']:>9}"
              + "".join(f"{(table[a][label]['median_px'] or float('nan')):>12.2f}"
                        for a in ("R0", "legacy", "contract")))
    print("\n짝지은 차이 (양수 = 앞쪽 우세)")
    for k, v in deltas.items():
        for label in ("ALL", "<8", "8-15", ">=15", "DAY", "LOW", "MID", "HIGH"):
            c = v.get(label)
            if c:
                print(f"  {k:<22}{label:<6} {c['median']:+7.2f} "
                      f"[{c['lo']:+.2f}, {c['hi']:+.2f}]  세션 {c['n_sessions']}")
    print(f"\nwrote {dst.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
