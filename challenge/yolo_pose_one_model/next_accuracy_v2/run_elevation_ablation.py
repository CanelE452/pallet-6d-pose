#!/usr/bin/env python3
"""§13 · §14 앙각 구성 ablation — 2x2 transfer matrix. 학습부터 판정·알림까지 한 파일에서.

    STEP 1  arm 목록 6개(L/M x draw 0,1,2)로 데이터셋을 빌드
    STEP 2  R0 -> 각 arm 137장으로 FT 6 run
    STEP 3  같은 held-out 297장을 <8 / 8-15 층으로 나눠 재채점
    STEP 4  2x2 판정 후 알림

사전등록: data/pallet/results/next_accuracy_v2/METHOD_LOCK.json stage_2
복제는 seed 가 아니라 **membership 이 다른 draw** 로 만든다 — ultralytics seed 는
dataloader 에 도달하지 않아 텐서가 비트 동일해진다(stage_1 에서 실측 확인).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[2]
ONE = REPO / "challenge/yolo_pose_one_model"
ARMS = REPO / "data/pallet/results/next_accuracy_v2/arms"
BUILDER = REPO / "scripts/research/next_accuracy_v2/build_contract_dataset.py"

import run_corrected_real_ft as S1  # noqa: E402  전제·평가·부트스트랩을 그대로 쓴다

DRAWS = [0, 1, 2]
BANDS = ["L", "M"]


def main() -> int:
    t0 = time.time()
    assert S1.sha256(S1.BASE) == S1.BASE_SHA, "base checkpoint SHA 불일치"
    lock = json.loads((REPO / "data/pallet/results/next_accuracy_v2/METHOD_LOCK.json")
                      .read_text(encoding="utf-8"))
    assert lock["status"] == "FROZEN"
    print(f"사전등록 arm_size = {lock['stage_2']['arm_size']}", flush=True)

    from ultralytics import YOLO
    datasets, runs = {}, {}
    for band in BANDS:
        for d in DRAWS:
            tag = f"{band}_d{d}"
            ds = ONE / f"datasets/arm_{tag}"
            datasets[tag] = ds
            if not (ds / "data.yaml").is_file():
                print(f"\n=== STEP 1 build {tag} ===", flush=True)
                subprocess.run(
                    [sys.executable, str(BUILDER), "--out", f"datasets/arm_{tag}",
                     "--train-subset", str(ARMS / f"arm_{band}_d{d}.txt")],
                    check=True, cwd=str(REPO))
            n = len(list((ds / "labels/train").glob("*.txt")))
            assert n == lock["stage_2"]["arm_size"], f"{tag}: train {n} != 137"

            out = HERE / f"arm_{tag}/weights/best.pt"
            runs[tag] = out
            if not out.is_file():
                S1.drop_label_cache(ds)
                print(f"\n=== STEP 2 train {tag} ===", flush=True)
                YOLO(str(S1.BASE)).train(data=str(ds / "data.yaml"), project=str(HERE),
                                         name=f"arm_{tag}", seed=42, **S1.RECIPE)
            assert out.is_file(), f"{tag}: best.pt 없음"

    print("\n=== STEP 3 재채점 ===", flush=True)
    idx = S1.load_eval_index()
    images = ONE / "datasets/live_gt_contract_v2/images/val"
    per, raw = {}, {}
    for tag, w in [("R0", S1.BASE)] + sorted(runs.items()):
        rows = S1.evaluate(Path(w), idx, images)
        raw[tag] = rows
        per[tag] = S1.per_frame_median(rows)
        for b in ("<8", "8-15"):
            e = [v for r in rows if r["band"] == b for v in r["errs"]]
            print(f"   {tag:<8}{b:<6} med {np.median(e):6.2f} px  kp {len(e)}", flush=True)

    band_of = {r["stem"]: r["band"] for r in raw["R0"]}
    sess_of = {r["stem"]: r["session"] for r in raw["R0"]}

    print("\n=== STEP 4 2x2 ===", flush=True)
    cell = {}
    for band in BANDS:
        for b in ("<8", "8-15"):
            vals = []
            for stem in per["R0"]:
                if band_of[stem] != b:
                    continue
                v = [per[f"{band}_d{d}"][stem] for d in DRAWS]
                v = [x for x in v if np.isfinite(x)]
                if v:
                    vals.append(float(np.mean(v)))
            cell[f"train{band}_eval{b}"] = round(float(np.median(vals)), 3)
    print(f"{'':<12}{'eval <8':>10}{'eval 8-15':>12}")
    for band in BANDS:
        print(f"train {band:<7}{cell[f'train{band}_eval<8']:>10.2f}"
              f"{cell[f'train{band}_eval8-15']:>12.2f}")

    diff = {}
    for b in ("<8", "8-15"):
        pairs = []
        for stem in per["R0"]:
            if band_of[stem] != b:
                continue
            l = [per[f"L_d{d}"][stem] for d in DRAWS]
            m = [per[f"M_d{d}"][stem] for d in DRAWS]
            l = [x for x in l if np.isfinite(x)]
            m = [x for x in m if np.isfinite(x)]
            if l and m:
                pairs.append((sess_of[stem], float(np.mean(m)) - float(np.mean(l))))
        diff[b] = S1.session_cluster_ci(pairs)   # 양수 = L 우세

    a, c = cell["trainL_eval<8"], cell["trainM_eval<8"]
    dd, bb = cell["trainM_eval8-15"], cell["trainL_eval8-15"]
    diag = (a < c) and (dd < bb)
    l_wins_low = bool(diff["<8"] and diff["<8"]["lo"] > 0)
    verdict = ("REGIME_SPECIFIC_SUPERVISION_SUPPORTED" if (diag and l_wins_low)
               else ("GENERAL_REAL_APPEARANCE_EFFECT"
                     if (diff["<8"] and diff["<8"]["lo"] <= 0 <= diff["<8"]["hi"])
                     else "INCONCLUSIVE"))

    res = {"schema_version": "next_accuracy_v2_stage2_result_v1",
           "method_lock_stage": "stage_2", "arm_size": lock["stage_2"]["arm_size"],
           "replication": "membership draw x3 (seed 아님 — seed 는 텐서 비트 동일)",
           "transfer_matrix_median_px": cell,
           "M_minus_L_positive_means_L_better": diff,
           "diagonal_advantage": diag, "L_wins_low_band": l_wins_low,
           "verdict": verdict,
           "raw": {t: [{k: v for k, v in r.items() if k != "errs"} |
                       {"median_err_px": (float(np.median(r["errs"])) if r["errs"] else None)}
                       for r in rows] for t, rows in raw.items()},
           "elapsed_min": round((time.time() - t0) / 60, 1), "FINAL": "STEP4_DONE"}
    (HERE / "RESULT_STAGE2.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"[next_accuracy_v2 §13] {verdict}",
             f"2x2 중앙값 px  trainL: <8 {a:.2f} / 8-15 {bb:.2f}   "
             f"trainM: <8 {c:.2f} / 8-15 {dd:.2f}"]
    for b in ("<8", "8-15"):
        d = diff[b]
        if d:
            lines.append(f"  M-L {b:<6} Δ{d['median']:+.2f} px "
                         f"[{d['lo']:+.2f}, {d['hi']:+.2f}] 세션 {d['n_sessions']}")
    lines.append(f"  {res['elapsed_min']} 분")
    S1.notify("\n".join(lines))
    print("\n" + "\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
