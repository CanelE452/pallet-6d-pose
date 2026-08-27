"""PHASE 2~5 — 예측 가능한 gate + session LOSO CV + 사전등록 판정.

gate feature 는 `pnp_reproj` **하나만** 쓴다.  D7 에서 pose 오차와 연관이 확인됐고
추론 시 바로 계산되며, GT 를 쓰지 않는다.  여러 feature 를 결과를 보고 조합하지
않는다.

★ 같은 DEV 에서 tau 를 고르고 같은 DEV 로 평가하면 낙관 편향이다.
그래서 **session Leave-One-Out** — held-out session 하나를 빼고 나머지 6 개에서
tau 를 고른 뒤 held-out 에서만 평가한다.  7 회 반복.
"""
from __future__ import annotations

import csv, json, os
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
OUT = os.path.join(ROOT, "challenge/yolo_pose_one_model/analysis_pre_v2")

# 결과 보기 전에 고정
TAU_GRID = [50, 60, 70, 80, 90]        # pnp_reproj 백분위
T_DEGRADE_MAX = 0.05                   # translation median 악화 허용
S5_DEGRADE_MAX = 0.0                   # 5cm5 악화 허용 (0pp)
VERDICT_GATE = {
    "challenge_R_improve_min": 0.10, "challenge_5cm5_gain_min": 0.03,
    "challenge_t_degrade_max": 0.05,
    "open_5cm5_degrade_max": 0.02, "open_R_degrade_max": 0.05,
    "cv_folds_non_worse_min": 5,
    "fail_challenge_R_gain_below": 0.05, "fail_open_damage_above": 0.05,
    "fail_t_damage_above": 0.10,
}


def load():
    rows = []
    with open(os.path.join(OUT, "CONDITIONAL_HOUGH_PER_FRAME.csv")) as fh:
        for r in csv.DictReader(fh):
            d = {"fid": r["fid"], "set": r["set"], "population": r["population"]}
            for k, v in r.items():
                if k in d:
                    continue
                try:
                    d[k] = float(v) if v not in ("", "None") else np.nan
                except ValueError:
                    d[k] = np.nan
            rows.append(d)
    return rows


def apply_gate(rows, tau):
    """reproj <= tau 면 point-only, 아니면 Hough/F3."""
    out = []
    for r in rows:
        use_h = np.isfinite(r["reproj"]) and r["reproj"] > tau
        pre = "YH" if use_h else "Y0"
        out.append({**r, "activated": int(use_h),
                    "YG_R": r[f"{pre}_R"], "YG_t": r[f"{pre}_t"],
                    "YG_adds": r[f"{pre}_adds"], "YG_iou": r[f"{pre}_iou"],
                    "YG_s5": r[f"{pre}_s5"]})
    return out


def summ(rows, pre):
    R = np.array([r[f"{pre}_R"] for r in rows], float)
    t = np.array([r[f"{pre}_t"] for r in rows], float)
    g = np.isfinite(R)
    return {"n": len(rows),
            "R_median": float(np.median(R[g])) if g.any() else np.nan,
            "R_p90": float(np.percentile(R[g], 90)) if g.any() else np.nan,
            "t_median": float(np.nanmedian(t)),
            "ADD_S": float(np.nanmedian([r[f"{pre}_adds"] for r in rows])),
            "IoU": float(np.nanmedian([r[f"{pre}_iou"] for r in rows])),
            "success_5cm5": float(np.mean([r[f"{pre}_s5"] for r in rows]))}


def select_tau(train):
    """R median 최소화, 단 t 악화 <=5% 이고 5cm5 악화 0pp."""
    base = summ(train, "Y0")
    reproj = np.array([r["reproj"] for r in train], float)
    reproj = reproj[np.isfinite(reproj)]
    best = None
    for q in TAU_GRID:
        tau = float(np.percentile(reproj, q))
        g = summ(apply_gate(train, tau), "YG")
        t_deg = (g["t_median"] - base["t_median"]) / max(base["t_median"], 1e-9)
        s5_deg = base["success_5cm5"] - g["success_5cm5"]
        if t_deg > T_DEGRADE_MAX or s5_deg > S5_DEGRADE_MAX:
            continue
        if best is None or g["R_median"] < best[1]["R_median"]:
            best = (tau, g, q)
    if best is None:                     # 어떤 tau 도 제약을 못 지키면 항상 point-only
        return float("inf"), None, None
    return best[0], best[1], best[2]


def main():
    rows = load()
    sessions = sorted({r["set"] for r in rows})
    folds = []
    for held in sessions:
        train = [r for r in rows if r["set"] != held]
        test = [r for r in rows if r["set"] == held]
        tau, _, q = select_tau(train)
        gated = apply_gate(test, tau)
        y0, yg = summ(test, "Y0"), summ(gated, "YG")
        folds.append({"held_out": held, "n_test": len(test),
                      "tau": None if not np.isfinite(tau) else round(tau, 3),
                      "tau_percentile": q,
                      "activation_rate": round(float(np.mean(
                          [g["activated"] for g in gated])), 4),
                      "Y0_R_median": round(y0["R_median"], 3),
                      "YG_R_median": round(yg["R_median"], 3),
                      "R_non_worse": bool(yg["R_median"] <= y0["R_median"] + 1e-9),
                      "Y0_5cm5": round(y0["success_5cm5"], 4),
                      "YG_5cm5": round(yg["success_5cm5"], 4),
                      "Y0_t": round(y0["t_median"], 4),
                      "YG_t": round(yg["t_median"], 4)})

    # 전체 CV 예측 (각 프레임은 자기 session 이 held-out 일 때의 tau 로)
    tau_by_session = {f["held_out"]: (f["tau"] if f["tau"] is not None
                                      else float("inf")) for f in folds}
    cv_rows = []
    for r in rows:
        tau = tau_by_session[r["set"]]
        cv_rows.append(apply_gate([r], tau)[0])

    report = {"note": "gate feature 는 pnp_reproj 하나. GT 미사용. "
                      "tau 는 session LOSO 로 고른다.",
              "tau_grid_percentiles": TAU_GRID,
              "selection_objective": "R median 최소화 s.t. t 악화<=5%, 5cm5 악화<=0pp",
              "verdict_gate": VERDICT_GATE,
              "folds": folds, "populations": {}}
    for pop in ("REAL_DEV_OPEN_56", "REAL_CHALLENGE_DEV_105"):
        sub = [r for r in cv_rows if r["population"] == pop]
        b = {a: summ(sub, a) for a in ("Y0", "YH", "YG")}
        b["activation_rate"] = round(float(np.mean(
            [r["activated"] for r in sub])), 4)
        b["R_win_fraction_vs_Y0"] = round(float(np.mean(
            [1 if (np.isfinite(r["YG_R"]) and np.isfinite(r["Y0_R"])
                   and r["YG_R"] < r["Y0_R"]) else 0 for r in sub])), 4)
        b["t_non_worse_fraction"] = round(float(np.mean(
            [1 if (np.isfinite(r["YG_t"]) and np.isfinite(r["Y0_t"])
                   and r["YG_t"] <= r["Y0_t"] + 1e-9) else 0 for r in sub])), 4)
        report["populations"][pop] = b
    # domain 별
    report["by_domain"] = {}
    for s in sessions:
        sub = [r for r in cv_rows if r["set"] == s]
        report["by_domain"][s] = {
            "n": len(sub),
            "Y0_R": round(summ(sub, "Y0")["R_median"], 3),
            "YG_R": round(summ(sub, "YG")["R_median"], 3),
            "Y0_5cm5": round(summ(sub, "Y0")["success_5cm5"], 4),
            "YG_5cm5": round(summ(sub, "YG")["success_5cm5"], 4),
            "activation": round(float(np.mean([r["activated"] for r in sub])), 4)}

    ch = report["populations"]["REAL_CHALLENGE_DEV_105"]
    op = report["populations"]["REAL_DEV_OPEN_56"]
    def rel(a, b):
        return (a - b) / max(abs(a), 1e-9)
    ch_R_gain = rel(ch["Y0"]["R_median"], ch["YG"]["R_median"])
    ch_5_gain = ch["YG"]["success_5cm5"] - ch["Y0"]["success_5cm5"]
    ch_t_deg = rel(ch["YG"]["t_median"], ch["Y0"]["t_median"]) * -1 \
        if ch["Y0"]["t_median"] else 0
    ch_t_deg = (ch["YG"]["t_median"] - ch["Y0"]["t_median"]) / max(ch["Y0"]["t_median"], 1e-9)
    op_5_deg = op["Y0"]["success_5cm5"] - op["YG"]["success_5cm5"]
    op_R_deg = (op["YG"]["R_median"] - op["Y0"]["R_median"]) / max(op["Y0"]["R_median"], 1e-9)
    folds_ok = sum(1 for f in folds if f["R_non_worse"])

    supported = (ch_R_gain >= VERDICT_GATE["challenge_R_improve_min"]
                 and ch_5_gain >= VERDICT_GATE["challenge_5cm5_gain_min"]
                 and ch_t_deg <= VERDICT_GATE["challenge_t_degrade_max"]
                 and op_5_deg <= VERDICT_GATE["open_5cm5_degrade_max"]
                 and op_R_deg <= VERDICT_GATE["open_R_degrade_max"]
                 and folds_ok >= VERDICT_GATE["cv_folds_non_worse_min"])
    failed = (ch_R_gain < VERDICT_GATE["fail_challenge_R_gain_below"]
              or op_R_deg > VERDICT_GATE["fail_open_damage_above"]
              or ch_t_deg > VERDICT_GATE["fail_t_damage_above"])
    verdict = ("CONDITIONAL_HOUGH_SUPPORTED" if supported else
               "HOUGH_TRACK_CLOSED" if failed else
               "PROMISING_NOT_ESTABLISHED")
    report["verdict_inputs"] = {
        "challenge_R_gain": round(ch_R_gain, 4),
        "challenge_5cm5_gain": round(ch_5_gain, 4),
        "challenge_t_degrade": round(ch_t_deg, 4),
        "open_5cm5_degrade": round(op_5_deg, 4),
        "open_R_degrade": round(op_R_deg, 4),
        "cv_folds_R_non_worse": f"{folds_ok}/{len(folds)}"}
    report["VERDICT"] = verdict
    json.dump(report, open(os.path.join(OUT, "CONDITIONAL_HOUGH_CV.json"), "w"),
              indent=1, ensure_ascii=False)

    print("=== session LOSO ===")
    print(f"{'held out':12}{'n':>4}{'tau':>8}{'act':>7}{'Y0 R':>8}{'YG R':>8}"
          f"{'비악화':>8}{'Y0 5cm5':>9}{'YG 5cm5':>9}")
    for f in folds:
        print(f"{f['held_out'].replace('eval_',''):12}{f['n_test']:>4}"
              f"{(f['tau'] if f['tau'] is not None else -1):>8.2f}"
              f"{f['activation_rate']:>7.2f}{f['Y0_R_median']:>8.2f}"
              f"{f['YG_R_median']:>8.2f}{str(f['R_non_worse']):>8}"
              f"{f['Y0_5cm5']:>9.3f}{f['YG_5cm5']:>9.3f}")
    for pop, b in report["populations"].items():
        print(f"\n=== {pop} (n={b['Y0']['n']}) ===")
        print(f"{'arm':6}{'R med':>9}{'R p90':>9}{'t med':>9}{'5cm5':>8}")
        for a in ("Y0", "YH", "YG"):
            s = b[a]
            print(f"{a:6}{s['R_median']:>9.2f}{s['R_p90']:>9.2f}"
                  f"{s['t_median']:>9.4f}{s['success_5cm5']:>8.3f}")
        print(f"  activation {b['activation_rate']:.3f}  "
              f"R win {b['R_win_fraction_vs_Y0']:.3f}  "
              f"t 비악화 {b['t_non_worse_fraction']:.3f}")
    print(f"\n판정 입력: {report['verdict_inputs']}")
    print(f"VERDICT = {verdict}")


if __name__ == "__main__":
    main()
