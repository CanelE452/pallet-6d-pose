"""G38(원래) -> LV1V2FT(개선) 이 목표까지의 격차를 얼마나 메웠나.

closure = (A_gap - C_gap) / A_gap,  gap = |x - target|
  A = G38(원래) · C = 후보 · target = 목표

★ 기존 REAL_GAP_CLOSURE.json 의 수치는 어느 블록으로 냈는지 재현되지 않아
  (FT/DATA_C 가 available/correct_box/paired 어디와도 불일치) 그 값을 따라가지
  않는다. 여기서는 모집단을 명시한다.

  coverage 계열 (cbox recall, night top1/any) = 프레임 수 지표라 모집단 보정 불필요
  localization 계열 (corner median/p90)       = 비교 대상 전 모델이 correct_box 인
                                                교집합에서만 잰다. 안 그러면 검출이
                                                좋은 모델이 어려운 프레임을 떠안아
                                                손해를 본다(Simpson).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
Q = HERE.parents[0] / "runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"

BASE = "G38"
NIGHT_TAG = {"OLD_STAGE_A": "OLD_S"}


def real(t):
    return json.loads((Q / f"REAL_{t}.json").read_text())


def night(t):
    return json.loads((Q / f"NIGHT_CAND_{NIGHT_TAG.get(t, t)}.json").read_text())


def pf(t):
    return {f["frame"]: f for f in real(t)["per_frame"]}


def closure(a, c, tgt, lower_is_better):
    """A_gap 대비 얼마나 메웠나. 목표를 넘어서면 >1."""
    ag, cg = abs(a - tgt), abs(c - tgt)
    if ag == 0:
        return None
    return (ag - cg) / ag


def run(cand, targets):
    tags = [BASE, cand] + targets
    P = {t: pf(t) for t in tags}
    common = sorted(set.intersection(*(set(P[t]) for t in tags)))
    inter = [f for f in common if all(P[t][f].get("correct_box") for t in tags)]

    def errs(t, frames):
        e = [x for f in frames for x in P[t][f]["err"]]
        a = np.asarray(e, dtype=float)
        return float(np.median(a)), float(np.percentile(a, 90))

    out = {"candidate": cand, "base": BASE, "targets": targets,
           "n_common_frames": len(common),
           "n_matched_correct_box_all_models": len(inter),
           "coverage": {}, "localization": {}, "night": {}}

    R = {t: real(t) for t in tags}
    N = {t: night(t) for t in tags}

    out["coverage"]["correct_box_recall"] = {
        t: R[t]["correct_box_recall"] for t in tags}
    out["coverage"]["detection_recall"] = {t: R[t]["detection_recall"] for t in tags}
    out["night"]["top1_frames"] = {t: round(N[t]["top1_cbox"] * N[t]["n"]) for t in tags}
    out["night"]["any_frames"] = {t: round(N[t]["any_cbox"] * N[t]["n"]) for t in tags}
    out["night"]["margin_median"] = {t: N[t]["margin_median"] for t in tags}

    med, p90 = {}, {}
    for t in tags:
        med[t], p90[t] = errs(t, inter)
    out["localization"]["corner_median"] = med
    out["localization"]["corner_p90"] = p90

    night_inter = [f for f in inter if P[BASE][f]["domain"] == "NIGHT"]
    out["night"]["n_matched"] = len(night_inter)
    if night_inter:
        nm, np90 = {}, {}
        for t in tags:
            nm[t], np90[t] = errs(t, night_inter)
        out["night"]["corner_median_matched"] = nm
        out["night"]["corner_p90_matched"] = np90

    spec = [
        ("correct_box_recall", out["coverage"]["correct_box_recall"], False),
        ("night_top1_frames", out["night"]["top1_frames"], False),
        ("night_any_frames", out["night"]["any_frames"], False),
        ("night_margin_median", out["night"]["margin_median"], False),
        ("corner_median (matched)", med, True),
        ("corner_p90 (matched)", p90, True),
    ]
    if night_inter:
        spec += [("night_corner_median (matched)", out["night"]["corner_median_matched"], True),
                 ("night_corner_p90 (matched)", out["night"]["corner_p90_matched"], True)]

    out["closure"] = {}
    for name, d, lower in spec:
        out["closure"][name] = {
            tgt: closure(d[BASE], d[cand], d[tgt], lower) for tgt in targets}
        out["closure"][name]["_values"] = {t: d[t] for t in tags}
    return out


def fmt(o):
    cand, tgts = o["candidate"], o["targets"]
    L = [f"목표까지의 격차를 얼마나 메웠나  (base={o['base']} · 후보={cand})",
         f"localization 은 {o['n_matched_correct_box_all_models']}장 "
         f"(비교 전 모델이 모두 correct_box 인 교집합) 에서 계산",
         f"night matched = {o['night']['n_matched']}장",
         ""]
    w = 30
    hdr = f"{'지표':{w}s} {o['base']:>10s} {cand:>10s}" + "".join(f" {t:>14s}" for t in tgts)
    L += [hdr, "─" * len(hdr)]
    for name, blk in o["closure"].items():
        v = blk["_values"]
        row = f"{name:{w}s} {v[o['base']]:>10.3f} {v[cand]:>10.3f}"
        for t in tgts:
            row += f" {v[t]:>14.3f}"
        L.append(row)
        cl = f"{'  └ closure':{w}s} {'':>10s} {'':>10s}"
        for t in tgts:
            c = blk[t]
            cl += f" {(f'{c*100:+.0f}%' if c is not None else 'n/a'):>14s}"
        L.append(cl)
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    cand = sys.argv[1] if len(sys.argv) > 1 else "LV1V2FT"
    o = run(cand, ["OLD_STAGE_A", "FT_REFERENCE"])
    (HERE / f"GAP_CLOSURE_{cand}.json").write_text(
        json.dumps(o, ensure_ascii=False, indent=2), encoding="utf-8")
    print(fmt(o))
