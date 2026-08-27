"""3-arm 평가 — 정본 evaluator 두 개를 **그대로** 재사용한다.

새로 짜지 않는다.  자를 새로 만들면 기존 수치와 비교가 안 되고, 그 자가 맞는지를
다시 증명해야 한다.

    cf_real_eval.py   positive DEV.  pad=100 reflect, imgsz 640, top-1 by box conf
                      -> detection recall / correct-box recall / corner err (DAY·NIGHT)
    neg_eval_one.py   POS + NEG 점수 덤프.  threshold-free(conf 0.001)
                      -> AUPRC / AUROC / FPR@TPR / matched-recall FP

## 두 evaluator 의 모집단이 다르다 — 섞지 않는다

    cf_real_eval   REVIEWED_CLEAN_REALDEV_V2  **140장** (DAY 112 / NIGHT 28)
    neg_eval_one   위 140 에서 FT_EVAL_LEAK 12장 제외 = **128장** (DAY 100 / NIGHT 28)
                   + real negative 2,689

leak 12장은 FT 모델이 학습에서 본 프레임이다.  이번 arm 들은 **real 을 한 장도
학습하지 않으므로** 우리에게는 누수가 아니다.  그래도 카드(real128) 수치와 맞추려면
neg 쪽 모집단을 그대로 두는 편이 낫다.  지표마다 어느 모집단인지 결과에 박는다.

## S1 정의를 여기서 못박는다 (결과 보기 전)

`detection_recall` 은 conf 0.001 기준이라 거의 항상 1.0 이고 신호가 없다.
**S1 = detection_recall_deploy (conf >= 0.40)** 를 쓴다.  운영에서 의미 있는 쪽이다.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

ROOT = "/home/minjae/Documents/github/pallet-pose"
HN = os.path.join(ROOT, "challenge/yolo_pose_one_model/hard_negative_v1")
Q = os.path.join(ROOT, "challenge/yolo_pose_one_model/runs_camera_facing_loss/"
                       "ubuntu_cf_loss_queue_20260823T0930")
EVAL = os.path.join(HN, "evaluation")
ARMS = {"HC": "HC_POSREPEAT1900",
        "HM": "HM_HARDNEG1900_STOCK",
        "HF": "HF_HARDNEG1900_FOCALNEG"}


def run(cmd):
    print("  $ " + " ".join(os.path.basename(c) if c.endswith(".py") else c
                            for c in cmd[-6:]), flush=True)
    p = subprocess.run(cmd, cwd=Q, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        raise SystemExit(f"FAILED: {cmd}")
    print("    " + p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "",
          flush=True)


def main():
    os.makedirs(EVAL, exist_ok=True)
    py = sys.executable
    for arm, run_name in ARMS.items():
        w = os.path.join(HN, "runs", run_name, "weights", "best.pt")
        if not os.path.exists(w):
            raise SystemExit(f"weights 없음: {w}")
        tag = f"HN_{arm}"
        run([py, os.path.join(Q, "cf_real_eval.py"), "--weights", w, "--tag", tag])
        run([py, os.path.join(Q, "neg_eval_one.py"), "--weights", w, "--tag", tag])
        for src, dst in ((f"REAL_{tag}.json", f"POSITIVE_DEV__{arm}.json"),
                         (f"NEGSCORE_{tag}.json", f"NEGSCORE__{arm}.json")):
            shutil.copy(os.path.join(Q, src), os.path.join(EVAL, dst))
    json.dump({"populations": {
        "positive_dev": "REVIEWED_CLEAN_REALDEV_V2 140장 (DAY 112 / NIGHT 28)",
        "negscore_pos": "위 140 - FT_EVAL_LEAK 12 = 128장 (DAY 100 / NIGHT 28)",
        "negscore_neg": "real negative 2,689"},
        "S1_definition": "detection_recall_deploy (conf >= 0.40)",
        "note": "arm 들은 real 을 학습하지 않았으므로 leak 12장은 우리에겐 누수가 아니다",
        "evaluators": ["cf_real_eval.py", "neg_eval_one.py"]},
        open(os.path.join(EVAL, "EVAL_CONTRACT.json"), "w"),
        indent=1, ensure_ascii=False)
    print(f"-> {EVAL}")


if __name__ == "__main__":
    sys.exit(main())
