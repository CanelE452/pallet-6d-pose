"""사후 감사 — gate 수치를 그대로 믿지 않고 모집단을 맞춰 다시 잰다.

cf_real_eval 의 correct_box 블록은 "그 모델이 박스를 맞춘 프레임" 위에서 median 을
낸다. FT 로 correct_box 가 119 -> 138 로 늘면 **비교 대상 프레임 집합이 바뀌므로**
median 을 직접 비교할 수 없다(Simpson). 여기서는

  ① MATCHED  : 양쪽 다 correct_box 인 프레임만 → localization 순수 비교
  ② COVERAGE : correct_box 프레임 수 변화 → 검출 측 이득
  ③ FP@recall: negative 오탐을 같은 recall 에서 비교 (고정 conf 비교는 무효 —
               FT 가 신뢰도 분포를 통째로 올려서 같은 conf 가 같은 동작점이 아니다)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
Q = HERE.parents[0] / "runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
A_TAG, B_TAG = "G38", "LV1V2FT"


def per_frame(tag):
    return {f["frame"]: f for f in json.loads((Q / f"REAL_{tag}.json").read_text())["per_frame"]}


def neg_rows(tag):
    r = json.loads((Q / f"NEGSCORE_{tag}.json").read_text())["rows"]
    return [x for x in r if x["label"] == 1], [x for x in r if x["label"] == 0]


def med_p90(errs):
    a = np.asarray(errs, dtype=float)
    return float(np.median(a)), float(np.percentile(a, 90))


def fp_at_recall(pos, neg, target):
    best = None
    for c in sorted({round(r["max_conf"], 4) for r in pos}):
        rec = sum(1 for r in pos if r["max_conf"] >= c) / len(pos)
        if rec >= target:
            best = {"conf": c, "recall": rec,
                    "fp_frame_rate": sum(1 for r in neg if r["max_conf"] >= c) / len(neg)}
    return best


def auroc(pos, neg):
    s = np.array([r["max_conf"] for r in pos])
    t = np.array([r["max_conf"] for r in neg])
    allv = np.concatenate([s, t])
    rk = allv.argsort().argsort() + 1
    return float((rk[:len(s)].sum() - len(s) * (len(s) + 1) / 2) / (len(s) * len(t)))


def main():
    A, B = per_frame(A_TAG), per_frame(B_TAG)
    common = sorted(set(A) & set(B))
    both = [f for f in common if A[f].get("correct_box") and B[f].get("correct_box")]

    out = {"n_common": len(common), "matched": {}, "coverage": {}, "negative": {}}

    for scope, frames in (("ALL", both),
                          ("DAY", [f for f in both if A[f]["domain"] == "DAY"]),
                          ("NIGHT", [f for f in both if A[f]["domain"] == "NIGHT"])):
        if not frames:
            continue
        ma, pa = med_p90([e for f in frames for e in A[f]["err"]])
        mb, pb = med_p90([e for f in frames for e in B[f]["err"]])
        out["matched"][scope] = {
            "n_frames": len(frames),
            "corner_median": {A_TAG: ma, B_TAG: mb, "rel_gain": (ma - mb) / ma},
            "corner_p90": {A_TAG: pa, B_TAG: pb, "rel_gain": (pa - pb) / pa},
        }

    for scope, frames in (("ALL", common),
                          ("DAY", [f for f in common if A[f]["domain"] == "DAY"]),
                          ("NIGHT", [f for f in common if A[f]["domain"] == "NIGHT"])):
        ca = sum(1 for f in frames if A[f].get("correct_box"))
        cb = sum(1 for f in frames if B[f].get("correct_box"))
        out["coverage"][scope] = {"n": len(frames), A_TAG: ca, B_TAG: cb, "delta": cb - ca}

    gained = [f for f in common if B[f].get("correct_box") and not A[f].get("correct_box")]
    lost = [f for f in common if A[f].get("correct_box") and not B[f].get("correct_box")]
    out["coverage"]["gained_frames"] = len(gained)
    out["coverage"]["lost_frames"] = lost
    if gained:
        gm, gp = med_p90([e for f in gained for e in B[f]["err"]])
        out["coverage"]["gained_corner_median"] = gm
        out["coverage"]["gained_corner_p90"] = gp

    pa_, na_ = neg_rows(A_TAG)
    pb_, nb_ = neg_rows(B_TAG)
    out["negative"] = {
        "n_pos": len(pa_), "n_neg": len(na_),
        "auroc": {A_TAG: auroc(pa_, na_), B_TAG: auroc(pb_, nb_)},
        "fp_at_recall": {f"{t:.2f}": {A_TAG: fp_at_recall(pa_, na_, t),
                                      B_TAG: fp_at_recall(pb_, nb_, t)}
                         for t in (0.60, 0.70, 0.75, 0.85, 0.90)},
    }

    (HERE / "MATCHED_AUDIT.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
