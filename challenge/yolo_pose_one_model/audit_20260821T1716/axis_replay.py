"""PHASE B3~B5 — object-frame 계약 replay.

재추론 0.  `analysis_pre_v2/_cc_raw_dump.json`(conf=0.001 전수 후보) 을 쓰고,
정본 audit(audit_20260821T1449)의 규칙을 그대로 적용한다:
    candidate floor tau* = 0.0094,  native top-1 = box confidence 최대,
    rerank 없음, Hough 없음, threshold 재튜닝 없음.
tau* 는 진단용 floor 이며 **deployment threshold 가 아니다**.

조건
  E0  CURRENT_REPLAY          per-frame GT label dimensions (현재 evaluator)
  E1  DEPLOYABLE_FIXED_DIMS   annotate_pnp.PALLET_DIMS = (1.1, 1.3, 0.11) 고정
  E1b DEPLOY_PROBE_REPROJ     두 배정을 다 풀어 **재투영 오차**로 선택 (GT 미사용)
                              ★ 방법 제안이 아니라 "정보가 복원 가능한가" 탐침
  E2  ORACLE_DIM_CHOICE       두 배정 중 pose 오차가 좋은 쪽 (GT 사용)
                              ★ ORACLE — NOT DEPLOYABLE
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0/real_eval", "scripts/annotate"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import cv2                       # noqa: E402
import re_metrics as RM          # noqa: E402
import annotate_pnp as APNP      # noqa: E402

A2 = os.path.join(ROOT, "challenge/yolo_pose_one_model/analysis_pre_v2")
PIPE = os.path.join(ROOT, "challenge/yolo_pose_one_model/paper_generic_pipeline")
CANON = os.path.join(ROOT, "challenge/yolo_pose_one_model/audit_20260821T1449")
OUT = os.path.dirname(os.path.abspath(__file__))
TAU = 0.0094                     # 정본 audit 의 tau* — deployment threshold 아님
NOMINAL = APNP.PALLET_DIMS       # (1.1, 1.3, 0.11) — 사전 고정 상수
GROSS_R = 10.0
B = 10000


def model_of(w, d, h):
    return APNP.make_pallet_keypoints_3d_diagram(width=w, depth=d, height=h)[:8]


def solve(px, model, K):
    ok, rv, tv = cv2.solvePnP(model, px.reshape(-1, 1, 2), K, None,
                              flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(model, px.reshape(-1, 1, 2), K, None, rv, tv)
    return cv2.Rodrigues(rv)[0], tv.reshape(3)


def score(pose, px, model, K, Rg, tg):
    if pose is None:
        return {"R": np.nan, "t": np.nan, "s5": 0, "reproj": np.inf}
    R, t = pose
    Re, te = RM.pose_error(R, t, Rg, tg)
    cam = (R @ model.T).T + t
    z = np.clip(cam[:, 2], 1e-6, None)
    proj = (K @ (cam / z[:, None]).T).T[:, :2]
    return {"R": Re, "t": te,
            "s5": int(RM.success_5cm5deg(R, t, Rg, tg)),
            "reproj": float(np.median(np.linalg.norm(proj - px, axis=1)))}


def main():
    raw = json.load(open(os.path.join(A2, "_cc_raw_dump.json")))
    man = {i["frame_id"]: i for i in
           json.load(open(os.path.join(PIPE, "eval_manifest.json")))["items"]}
    rows = []
    for e in raw["positive"]:
        m = man[e["fid"]]
        K = np.asarray(m["K"], float)
        Rg, tg = np.asarray(m["R_gt"], float), np.asarray(m["t_gt"], float)
        dm = m["dimensions_m"]
        surv = [b for b in e["boxes"] if b["conf"] >= TAU]
        r = {"fid": e["fid"], "set": m["set"], "population": m["population"],
             "label_w": dm["width"], "label_d": dm["depth"],
             "label_variant": f"{dm['width']}x{dm['depth']}",
             "n_surv": len(surv)}
        if not surv:
            rows.append({**r, **{f"{c}_{k}": np.nan for c in
                                 ("E0", "E1", "E1b", "E2")
                                 for k in ("R", "t")},
                         **{f"{c}_s5": 0 for c in ("E0", "E1", "E1b", "E2")},
                         "E1b_pick": None, "E2_pick": None})
            continue
        px = np.asarray(max(surv, key=lambda b: b["conf"])["kps"], float)[:8]
        # E0 — 현재 evaluator: per-frame GT label dims
        m0 = model_of(dm["width"], dm["depth"], dm["height"])
        s0 = score(solve(px, m0, K), px, m0, K, Rg, tg)
        # E1 — 고정 nominal
        m1 = model_of(NOMINAL[0], NOMINAL[1], NOMINAL[2])
        s1 = score(solve(px, m1, K), px, m1, K, Rg, tg)
        # 두 배정 (1.1,1.3) / (1.3,1.1)
        mA = model_of(1.1, 1.3, dm["height"])
        mB = model_of(1.3, 1.1, dm["height"])
        sA = score(solve(px, mA, K), px, mA, K, Rg, tg)
        sB = score(solve(px, mB, K), px, mB, K, Rg, tg)
        # E1b — 재투영으로 선택 (GT 미사용)
        pickb = "A" if sA["reproj"] <= sB["reproj"] else "B"
        s1b = sA if pickb == "A" else sB
        # E2 — GT 로 좋은 쪽 (ORACLE)
        rA = sA["R"] if np.isfinite(sA["R"]) else 1e9
        rB = sB["R"] if np.isfinite(sB["R"]) else 1e9
        pick2 = "A" if rA <= rB else "B"
        s2 = sA if pick2 == "A" else sB
        rows.append({**r,
                     **{f"E0_{k}": s0[k] for k in ("R", "t", "s5")},
                     **{f"E1_{k}": s1[k] for k in ("R", "t", "s5")},
                     **{f"E1b_{k}": s1b[k] for k in ("R", "t", "s5")},
                     **{f"E2_{k}": s2[k] for k in ("R", "t", "s5")},
                     "E1b_pick": pickb, "E2_pick": pick2,
                     "E1b_correct": int((pickb == "A") ==
                                        (abs(dm["width"] - 1.1) < 1e-6))})
    with open(os.path.join(OUT, "AXIS_CONTRACT_REPLAY.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(rows)

    def agg(sub, c):
        R = np.array([r[f"{c}_R"] for r in sub], float)
        t = np.array([r[f"{c}_t"] for r in sub], float)
        g = np.isfinite(R)
        n = len(sub)
        return {"n": n, "n_solved": int(g.sum()),
                "uncond_5cm5": float(np.sum([r[f"{c}_s5"] for r in sub]) / n),
                "s5_hits": int(np.sum([r[f"{c}_s5"] for r in sub])),
                "R_median": float(np.median(R[g])) if g.any() else None,
                "R_p90": float(np.percentile(R[g], 90)) if g.any() else None,
                "t_median": float(np.nanmedian(t)) if g.any() else None,
                "t_p90": float(np.nanpercentile(t, 90)) if g.any() else None,
                "gross_R_rate": float(np.mean(
                    [(not np.isfinite(r[c + "_R"])) or r[c + "_R"] > GROSS_R
                     for r in sub]))}

    # population membership — 실제 frame ID 를 파일에서 읽는다
    nf = list(csv.DictReader(open(os.path.join(CANON, "NEAR_FAR_AUDIT.csv"))))
    gross24 = {x["fid"] for x in nf}
    nearfar11 = {x["fid"] for x in nf if x["improved_by_near_far"] == "True"}
    det = json.load(open(os.path.join(A2, "_rr_detail.json")))
    b59 = {d["fid"] for d in det if d["cls"] == "B_CORRECT_BOX_BAD_KP"}
    pops = {"ALL_161": [r for r in rows],
            "GROSS_R24": [r for r in rows if r["fid"] in gross24],
            "NEARFAR_IMPROVED_11": [r for r in rows if r["fid"] in nearfar11],
            "B_CORRECT_BOX_BAD_KP_59": [r for r in rows if r["fid"] in b59],
            "REAL_DEV_OPEN_56": [r for r in rows
                                 if r["population"] == "REAL_DEV_OPEN_56"],
            "REAL_CHALLENGE_DEV_105": [r for r in rows if r["population"]
                                       == "REAL_CHALLENGE_DEV_105"]}
    for s in sorted({r["set"] for r in rows}):
        pops[f"session::{s}"] = [r for r in rows if r["set"] == s]

    summary = {k: {c: agg(v, c) for c in ("E0", "E1", "E1b", "E2")}
               for k, v in pops.items() if v}

    # ---- session-cluster paired bootstrap ----
    sessions = sorted({r["set"] for r in rows})
    by = {s: [r for r in rows if r["set"] == s] for s in sessions}
    rng = np.random.default_rng(20260821)

    def boot(metric):
        d = []
        for _ in range(B):
            pick = rng.integers(0, len(sessions), len(sessions))
            samp = [r for i in pick for r in by[sessions[i]]]
            d.append(metric(samp))
        d = np.array(d)
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    def s5_delta(sub):
        return (np.mean([r["E1_s5"] for r in sub])
                - np.mean([r["E0_s5"] for r in sub]))

    def gross_delta(sub):
        def gr(c):
            return np.mean([(not np.isfinite(r[f"{c}_R"])) or
                            r[f"{c}_R"] > GROSS_R for r in sub])
        return gr("E1") - gr("E0")

    lo5, hi5 = boot(s5_delta)
    log, hig = boot(gross_delta)
    obs5 = s5_delta(rows); obsg = gross_delta(rows)
    # frame iid (보조)
    idx = np.arange(len(rows))
    d5 = [s5_delta([rows[i] for i in rng.integers(0, len(rows), len(rows))])
          for _ in range(2000)]

    stats = {"metric": "E1(deployable fixed dims) - E0(current GT-label dims)",
             "uncond_5cm5_delta": {"observed": obs5,
                                   "cluster_CI95": [lo5, hi5],
                                   "excludes_zero": bool(lo5 > 0 or hi5 < 0)},
             "gross_R_rate_delta": {"observed": obsg,
                                    "cluster_CI95": [log, hig],
                                    "excludes_zero": bool(log > 0 or hig < 0)},
             "frame_iid_secondary_CI95": [float(np.percentile(d5, 2.5)),
                                          float(np.percentile(d5, 97.5))],
             "B": B, "cluster_unit": "capture session (7개)"}

    out = {"rule": {"candidate_floor_tau": TAU,
                    "tau_source": "audit_20260821T1449/FAILURE_LOCUS_SUMMARY.json"
                                  " phase_C.tau_star",
                    "★tau_note": "진단용 floor 이며 deployment threshold 가 아니다",
                    "top1": "native box confidence 최대", "rerank": False,
                    "hough": False, "inference": "재추론 0 — _cc_raw_dump.json"},
           "conditions": {
               "E0": "per-frame GT label dimensions_m (현재 evaluator)",
               "E1": f"고정 nominal {tuple(NOMINAL)} = annotate_pnp.PALLET_DIMS",
               "E1b": "두 배정을 다 풀어 재투영 오차로 선택 (GT 미사용). "
                      "★탐침이지 제안이 아니다",
               "E2": "두 배정 중 pose 가 좋은 쪽 (GT 사용). "
                     "★ORACLE — NOT DEPLOYABLE"},
           "membership_provenance": {
               "ALL_161": "paper_generic_pipeline/eval_manifest.json",
               "GROSS_R24": "audit_20260821T1449/NEAR_FAR_AUDIT.csv (24행 전체)",
               "NEARFAR_IMPROVED_11": "같은 파일 improved_by_near_far==True",
               "B_CORRECT_BOX_BAD_KP_59":
                   "analysis_pre_v2/_rr_detail.json cls==B_CORRECT_BOX_BAD_KP",
               "★규칙차이": "B59 는 conf=0.001 top-5 기준으로 분류됐고 이 replay 는 "
                          "tau*=0.0094 native top-1 이다. 두 규칙이 다르므로 "
                          "B59 부분집합 수치는 그 차이를 안고 읽어야 한다."},
           "summary": summary, "bootstrap": stats,
           "E1b_dim_recoverability": {
               "correct_pick_rate": float(np.mean(
                   [r["E1b_correct"] for r in rows if r.get("E1b_correct")
                    is not None])),
               "note": "재투영만으로 어느 배정인지 맞힐 수 있는 비율. "
                       "이건 배포 가능성 탐침이며 방법 제안이 아니다."}}
    json.dump(out, open(os.path.join(OUT, "AXIS_CONTRACT_REPLAY.json"), "w"),
              indent=1, ensure_ascii=False)

    print(f"{'population':28}{'cond':>5}{'n':>5}{'5cm5':>8}{'hits':>6}"
          f"{'R med':>8}{'R p90':>8}{'t med':>8}{'grossR':>8}")
    print("─" * 84)
    for k in ("ALL_161", "REAL_DEV_OPEN_56", "REAL_CHALLENGE_DEV_105",
              "GROSS_R24", "NEARFAR_IMPROVED_11", "B_CORRECT_BOX_BAD_KP_59"):
        for c in ("E0", "E1", "E1b", "E2"):
            s = summary[k][c]
            print(f"{k if c=='E0' else '':28}{c:>5}{s['n']:>5}"
                  f"{s['uncond_5cm5']:>8.3f}{s['s5_hits']:>6}"
                  f"{(s['R_median'] or 0):>8.2f}{(s['R_p90'] or 0):>8.2f}"
                  f"{(s['t_median'] or 0):>8.4f}{s['gross_R_rate']:>8.3f}")
        print()
    print(f"E1-E0 5cm5 delta {obs5:+.4f}  cluster CI95 [{lo5:+.4f}, {hi5:+.4f}]"
          f"  0배제={stats['uncond_5cm5_delta']['excludes_zero']}")
    print(f"E1-E0 grossR delta {obsg:+.4f}  cluster CI95 [{log:+.4f}, {hig:+.4f}]"
          f"  0배제={stats['gross_R_rate_delta']['excludes_zero']}")
    print(f"E1b 가 올바른 배정을 고른 비율 "
          f"{out['E1b_dim_recoverability']['correct_pick_rate']:.3f}")


if __name__ == "__main__":
    main()
