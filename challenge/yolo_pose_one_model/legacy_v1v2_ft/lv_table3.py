"""목표 / 이전 / 개선 3열 평가표 — 같은 하네스로 잰 값만 한 표에 놓는다.

★ localization(코너 오차)은 **세 모델이 모두 correct_box 인 교집합**에서만 잰다.
  각자 자기가 맞춘 프레임 위에서 재면 검출이 좋은 모델이 어려운 프레임을 떠안아
  손해를 본다(Simpson). 그래서 모집단 크기를 표에 같이 적는다.
★ 오탐은 **같은 recall** 에서 비교한다. FT 가 신뢰도 분포를 통째로 올려서
  같은 conf 가 같은 동작점이 아니다.
"""
from __future__ import annotations

import json
import os
import unicodedata

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
Q = os.path.join(ROOT, "challenge/yolo_pose_one_model/runs_camera_facing_loss/"
                       "ubuntu_cf_loss_queue_20260823T0930")
HERE = os.path.dirname(os.path.abspath(__file__))
COLS = [("TARGET", "FT_REFERENCE"), ("BEFORE", "G38"), ("AFTER", "LV1V2FT")]


def J(p):
    return json.load(open(p))


def dw(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s, w, right=True):
    f = " " * max(0, w - dw(s))
    return f + s if right else s + f


def main():
    R = {k: J(f"{Q}/REAL_{t}.json") for k, t in COLS}
    N = {k: J(f"{Q}/NIGHT_CAND_{t}.json") for k, t in COLS}
    G = {}
    for k, t in COLS:
        p = f"{Q}/NEGSCORE_{t}.json"
        G[k] = J(p)["rows"] if os.path.exists(p) else None

    PF = {k: {f["frame"]: f for f in R[k]["per_frame"]} for k, _ in COLS}
    keys = [k for k, _ in COLS]
    common = sorted(set.intersection(*(set(PF[k]) for k in keys)))
    inter = [f for f in common if all(PF[k][f].get("correct_box") for k in keys)]
    inter_n = [f for f in inter if PF[keys[0]][f]["domain"] == "NIGHT"]

    def mp(k, frames):
        a = np.array([e for f in frames for e in PF[k][f]["err"]], float)
        return float(np.median(a)), float(np.percentile(a, 90))

    def fp_at(k, target):
        rows = G[k]
        if rows is None:
            return None
        pos = [r for r in rows if r["label"] == 1]
        neg = [r for r in rows if r["label"] == 0]
        best = None
        for c in sorted({round(r["max_conf"], 4) for r in pos}):
            rec = sum(1 for r in pos if r["max_conf"] >= c) / len(pos)
            if rec >= target:
                best = sum(1 for r in neg if r["max_conf"] >= c) / len(neg)
        return best

    def auroc(k):
        rows = G[k]
        if rows is None:
            return None
        s = np.array([r["max_conf"] for r in rows if r["label"] == 1])
        t = np.array([r["max_conf"] for r in rows if r["label"] == 0])
        rk = np.concatenate([s, t]).argsort().argsort() + 1
        return float((rk[:len(s)].sum() - len(s) * (len(s) + 1) / 2) / (len(s) * len(t)))

    n28 = N[keys[0]]["n"]
    rows = [
        ("검출", None, None),
        ("  correct_box (140장 중)", "{:.0f}", [R[k]["correct_box_recall"] * R[k]["n_total"] for k in keys]),
        ("  detection_recall", "{:.3f}", [R[k]["detection_recall"] for k in keys]),
        ("야간 (n=%d)" % n28, None, None),
        ("  top1 맞춘 프레임", "{:.0f}", [N[k]["top1_cbox"] * n28 for k in keys]),
        ("  any 맞춘 프레임", "{:.0f}", [N[k]["any_cbox"] * n28 for k in keys]),
        ("  margin 중앙값", "{:.3f}", [N[k]["margin_median"] for k in keys]),
        ("  후보/프레임", "{:.2f}", [N[k]["cand_per_frame"] for k in keys]),
        ("코너오차 (공통 %d장)" % len(inter), None, None),
        ("  median (px)", "{:.2f}", [mp(k, inter)[0] for k in keys]),
        ("  p90 (px)", "{:.2f}", [mp(k, inter)[1] for k in keys]),
        ("코너오차 야간 (공통 %d장)" % len(inter_n), None, None),
        ("  median (px)", "{:.2f}", [mp(k, inter_n)[0] for k in keys]),
        ("  p90 (px)", "{:.2f}", [mp(k, inter_n)[1] for k in keys]),
        ("오탐 (neg 2,689장)", None, None),
        ("  AUROC", "{:.4f}", [auroc(k) for k in keys]),
        ("  FP율 @recall 0.75", "{:.2%}", [fp_at(k, 0.75) for k in keys]),
        ("  FP율 @recall 0.85", "{:.2%}", [fp_at(k, 0.85) for k in keys]),
        ("  FP율 @recall 0.90", "{:.2%}", [fp_at(k, 0.90) for k in keys]),
    ]

    name_w = max(dw(r[0]) for r in rows) + 2
    head = ["TARGET (목표)", "BEFORE (이전)", "AFTER (개선)"]
    cw = 15
    out = [pad("지표", name_w, False) + "".join(pad(h, cw) for h in head)]
    out.append("─" * (name_w + cw * 3))
    for name, fmt, vals in rows:
        if fmt is None:
            out.append(pad(name, name_w, False))
            continue
        cells = "".join(pad("n/a" if v is None else fmt.format(v), cw) for v in vals)
        out.append(pad(name, name_w, False) + cells)
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(HERE, "TABLE3.txt"), "w", encoding="utf-8").write(txt + "\n")
    json.dump({"columns": dict(COLS), "n_common_correct_box": len(inter),
               "n_common_night": len(inter_n),
               "weights": {k: R[k]["weights"] for k in keys}},
              open(os.path.join(HERE, "TABLE3_META.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
