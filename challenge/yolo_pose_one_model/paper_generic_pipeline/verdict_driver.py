"""PHASE 6+13+14 — 판정 + 다음 행동 + 풍부한 알림.

이미 도는 `run_paper_generic.sh` 가 train→dump→기본판정 까지 한다.  이 파일은
그 뒤를 잇는다 — 편집하면 실행 중인 bash 가 깨지므로 확장은 별도 파일로 둔다.

게이트는 결과를 보기 전에 여기 박혀 있다.
"""
from __future__ import annotations

import json, os, subprocess, sys

ROOT = "/home/minjae/Documents/github/pallet-pose"
YR = os.path.join(ROOT, "challenge/yolo_pose_one_model")
PIPE = os.path.join(YR, "paper_generic_pipeline")
EVAL = os.path.join(YR, "evaluation")
ANA = os.path.join(YR, "analysis")
REF_5EP = 0.057      # y_BROAD40K 5epoch challenge 5cm5

NEXT = {
    "STRONG_PASS": ("RUN_SEED43_AND_SEED44",
                    "seed 재현이 먼저. medium / negative FT / 새 데이터는 아직 금지."),
    "WEAK_PASS": ("BROAD_FAMILY_V2_SPEC_REVIEW",
                  "seed 만 늘리지 않는다. 새 모델도 돌리지 않는다."),
    "FAIL": ("FAILURE_AUDIT_REVIEW",
             "자동 추가 학습 금지. leakage/변환/padding/keypoint order/box 검출/"
             "도메인 붕괴를 먼저 본다."),
}


def main():
    v = json.load(open(os.path.join(EVAL, "PAPER_YOLO_VERDICT.json")))
    r = json.load(open(os.path.join(EVAL, "PAPER_YOLO_REAL_DEV_RESULT.json")))
    verdict = v["verdict"]
    me = r["models"]["yolo26n_paper_generic_v1"]
    o, c = me["OPEN_56"], me["REAL_CHALLENGE_DEV_105"]

    dom = {}
    path = os.path.join(ANA, "domain_breakdown.csv")
    if os.path.exists(path):
        import csv
        for row in csv.DictReader(open(path)):
            dom[row["set"]] = row

    generic = [s for s in ("eval_outside", "eval_noapril", "eval_cad") if s in dom]
    target = [s for s in ("eval_pallet07", "eval_pallet09") if s in dom]
    night = [s for s in ("eval_night08", "eval_night09") if s in dom]

    def mean5(keys):
        vals = [float(dom[k]["success_5cm5"]) for k in keys if k in dom]
        return round(sum(vals) / len(vals), 3) if vals else None

    seed_ready = all(os.path.exists(os.path.join(PIPE, "seed_replicates", f))
                     for f in ("run_seed43.sh", "run_seed44.sh",
                               "expected_args_seed43.json",
                               "expected_args_seed44.json"))
    action, why = NEXT[verdict]
    lines = [f"VERDICT = {verdict}", "",
             f"OPEN_56       R med {o['R_deg']['median']:.2f}deg  "
             f"t med {o['t_m']['median']:.3f}m  5cm5 {o['success_5cm5deg']:.3f}  "
             f"corner {o['corner_px']['median']:.1f}px",
             f"CHALLENGE_105 R med {c['R_deg']['median']:.2f}deg  "
             f"t med {c['t_m']['median']:.3f}m  5cm5 {c['success_5cm5deg']:.3f}  "
             f"corner {c['corner_px']['median']:.1f}px",
             f"native availability (challenge) {v['inputs']['challenge_availability']:.3f}",
             "",
             f"generic(outside/noapril/cad) 5cm5 {mean5(generic)}",
             f"target (pallet07/09)         5cm5 {mean5(target)}",
             f"night  (night08/09)          5cm5 {mean5(night)}",
             f"참조: 5epoch 진단본 challenge 5cm5 = {REF_5EP}",
             "",
             f"NEXT: {action}",
             f"  {why}",
             f"seed43/44 package ready: {seed_ready}  (AUTORUN_NEXT = False)"]
    text = "\n".join(lines)
    print(text)
    json.dump({"verdict": verdict, "next_action": action, "why": why,
               "seed_package_ready": seed_ready, "AUTORUN_NEXT": False,
               "domain": {"generic": mean5(generic), "target": mean5(target),
                          "night": mean5(night)}},
              open(os.path.join(EVAL, "PAPER_YOLO_NEXT.json"), "w"), indent=1)
    notify = os.path.expanduser("~/.claude/hooks/discord-notify.sh")
    if os.path.exists(notify):
        subprocess.run([notify, f"**PAPER_GENERIC_V1 판정**\n\n{text}"],
                       capture_output=True)


if __name__ == "__main__":
    main()
