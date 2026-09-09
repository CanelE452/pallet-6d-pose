"""목표 / 이전 / 개선 3열 오버레이.

그리기 규약(green=GT, blue=예측 kp, red=예측 pose 재투영)과 패널 렌더는
`mc_overlay_pair` 의 함수를 그대로 import 해서 쓴다. 따로 그리면 두 그림이
다른 것을 말한다.

★ 프레임은 고르지 않는다. 이번 실험의 미해결 질문(= 야간에서 코너 p90 이 왜
  나빠졌나)에 답하도록 층을 정하고, 각 층에서 규칙으로 자동 선정한다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts/stage0/model_compare"))
import cv2                       # noqa: E402
import mc_frames as MF           # noqa: E402
import mc_overlay_pair as MO     # noqa: E402

TARGET, BEFORE, AFTER = "LV3_TARGET", "LV3_BEFORE", "LV3_AFTER"
LABEL = {
    TARGET: "TARGET  real-FT  (upper bound)",
    BEFORE: "BEFORE  G38  (target-free)",
    AFTER:  "AFTER   G38 + LEGACY_V1V2 FT",
}
OUT = os.path.join(ROOT, "challenge/yolo_pose_one_model/legacy_v1v2_ft/overlay3")

# 야간 세션 — 파일명 규약으로 판별한다(정본 폴더명에 night 가 들어간다).
def is_night(key: str) -> bool:
    return "night" in key.lower()


def main():
    os.makedirs(OUT, exist_ok=True)
    dumps = {n: MO.load(n) for n in (TARGET, BEFORE, AFTER)}

    rec = []
    for key, sealed, jp, ip, label in MF.frames():
        fid = os.path.splitext(os.path.basename(jp))[0]
        es = {n: dumps[n].get((key, fid)) for n in (TARGET, BEFORE, AFTER)}
        if any(e is None for e in es.values()):
            continue
        m = {n: MO.measure(es[n], label, n) for n in (TARGET, BEFORE, AFTER)}
        rec.append({"key": key, "fid": fid, "ip": ip, "night": is_night(key), "m": m})

    def cm(r, n):
        v = r["m"][n][0]["corner_med"]
        return v if np.isfinite(v) else None

    def solved(r, n):
        return r["m"][n][0]["pnp_ok"] and cm(r, n) is not None

    both = [r for r in rec if solved(r, BEFORE) and solved(r, AFTER)]
    night_both = [r for r in both if r["night"]]
    day_both = [r for r in both if not r["night"]]
    gained = [r for r in rec if solved(r, AFTER) and not solved(r, BEFORE)]
    night_gained = [r for r in gained if r["night"]]

    def worse(r):   # AFTER 가 BEFORE 보다 얼마나 나쁜가 (양수 = 악화)
        return cm(r, AFTER) - cm(r, BEFORE)

    def mid(v, k):
        return sorted(v, key=k)[len(v) // 2] if v else None

    strata = [
        ("N1_night_worst",
         "NIGHT: AFTER degrades most vs BEFORE  (the open question: p90 tail)",
         max(night_both, key=worse) if night_both else None),
        ("N2_night_typical",
         "NIGHT: median-change frame  (is the tail typical or exceptional?)",
         mid(night_both, worse)),
        ("N3_night_gained",
         "NIGHT: BEFORE fails, AFTER solves  (where the detection gain comes from)",
         mid(night_gained, lambda r: cm(r, AFTER)) if night_gained else None),
        ("D1_day_typical",
         "DAY: median-change frame",
         mid(day_both, worse)),
        ("D2_day_gained",
         "DAY: BEFORE fails, AFTER solves",
         mid([r for r in gained if not r["night"]], lambda r: cm(r, AFTER))),
        ("X_after_best",
         "COUNTER-EXAMPLE: AFTER improves most over BEFORE",
         min(both, key=worse) if both else None),
    ]

    index = []
    for tag, why, r in strata:
        if r is None:
            print(f"  {tag}: 해당 프레임 없음 -> 건너뜀", flush=True)
            continue
        panels = []
        for n in (TARGET, BEFORE, AFTER):
            panels.append(MO.panel(r["ip"], *r["m"][n], LABEL[n], why))
        sep = np.full((panels[0].shape[0], 4, 3), 255, np.uint8)
        row = np.hstack([panels[0], sep, panels[1], sep, panels[2]])
        name = f"{tag}__{r['key']}__{r['fid']}.png"
        cv2.imwrite(os.path.join(OUT, name), row)
        index.append({
            "file": name, "stratum": tag, "why": why, "set": r["key"],
            "fid": r["fid"], "night": r["night"],
            **{n: {k: r["m"][n][0].get(k) for k in
                   ("pnp_ok", "corner_med", "R", "yaw", "t")}
               for n in (TARGET, BEFORE, AFTER)},
        })
        print(f"  {tag:18} {r['key']}/{r['fid']} -> {name}", flush=True)

    # 컨택트 시트 — 낱장을 하나로. 축소 후에도 읽히도록 요약 한 줄을 다시 얹는다.
    if index:
        tiles, W = [], 1500
        for it in index:
            img = cv2.imread(os.path.join(OUT, it["file"]))
            sc = W / img.shape[1]
            img = cv2.resize(img, (W, int(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)

            def brief(d):
                if not d["pnp_ok"]:
                    return "PnP FAILED"
                return f"{d['corner_med']:.1f}px"
            bar = np.zeros((26, W, 3), np.uint8)
            cv2.putText(bar, f"{it['stratum']}  |  TARGET {brief(it[TARGET])}"
                             f"   BEFORE {brief(it[BEFORE])}"
                             f"   AFTER {brief(it[AFTER])}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.vstack([bar, img]))
        h = max(t.shape[0] for t in tiles)
        tiles = [np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)])
                 for t in tiles]
        sheet = np.vstack(tiles)
        cap = np.zeros((30, sheet.shape[1], 3), np.uint8)
        cv2.putText(cap, "each tile: TARGET(real-FT) | BEFORE(G38) | AFTER(G38+LV1V2 FT)"
                         "   green=GT  blue=pred kp  red=pred pose reproj",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        sheet = np.vstack([cap, sheet])
        p = os.path.join(OUT, "CONTACT_SHEET.png")
        cv2.imwrite(p, sheet)
        print(f"  contact sheet {sheet.shape[1]}x{sheet.shape[0]} -> {p}", flush=True)

    json.dump({"models": LABEL,
               "selection": "층화 자동 선정 — 사람이 고르지 않았다",
               "frame_set": "mc_frames.frames() = 정본 161 (수치 표의 DEV140 과 다름)",
               "legend": "green=GT  blue=pred keypoint  red=pred pose 재투영",
               "panels": index},
              open(os.path.join(OUT, "INDEX.json"), "w"), indent=1, default=str)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
