"""FINAL_TRAIN_RESULT.md -- written from the artefacts, not from memory."""
from __future__ import annotations

import json, pathlib, sys

ROOT = pathlib.Path("/home/minjae/Documents/github/pallet-pose")
OUT = ROOT / "data/pallet/results/paper_s2_multihead"
FINAL = OUT / "final_train"


def main():
    package = json.loads((FINAL / "FINAL_MODEL_PACKAGE.json").read_text())
    smoke = json.loads((FINAL / "FINAL_TRAIN_SMOKE.json").read_text())
    config = json.loads((FINAL / "FINAL_TRAIN_CONFIG.json").read_text())
    unchanged = config["unchanged_from_e3"]

    lines = ["# FINAL TRAIN RESULT", "",
             "architecture search 가 아니다. 확정된 것을 40,000 으로 한 번 다시 학습했다.",
             "", "## 완료 판정 (산출물로만. exit code 안 씀)", "", "```"]
    for seed, entry in package["seeds"].items():
        lines.append(f"{seed}  checkpoint {entry['checkpoint_exists']}  "
                     f"pool {entry['pool']}  eval {entry['evaluated_every_mark']}  "
                     f"COMPLETE {entry['COMPLETE']}")
    lines += [f"manifest sha256 일치  {package['training_data']['sha256_match']}",
              f"ALL_COMPLETE          {package['ALL_COMPLETE']}", "```", "",
              "## 구조 스모크 (성능 아님)", "",
              "모든 프레임이 in-train 이라 수치 비교는 하지 않는다. 물어본 건 하나 —",
              "체크포인트가 구조적으로 멀쩡한가.", "", "```"]
    for seed, entry in smoke["seeds"].items():
        for population, block in entry["populations"].items():
            if not block.get("cache"):
                lines.append(f"{seed} {population:<14} cache 없음")
                continue
            solved = block["solved"]
            lines.append(
                f"{seed} {population:<14} n={block['n']:3d}  "
                f"corner/theta/score4kp finite {block['corner_finite']}/"
                f"{block['theta_finite']}/{block['score_4kp_finite']}  "
                f"F0 {solved.get('F0')}/{block['n']}  F3 {solved.get('F3')}/{block['n']}")
    lines += [f"STRUCTURAL_OK_ALL  {smoke['STRUCTURAL_OK_ALL']}", "```", "",
              "## config — E3 에서 바뀐 것은 pool 하나", "", "```",
              f"architecture   {unchanged['architecture']}",
              f"arm            {unchanged['arm']}",
              f"frozen early   {unchanged['frozen_early_boundary']}",
              f"optimizer      {unchanged['optimizer']}  lr {unchanged['lr']}  "
              f"wd {unchanged['weight_decay']}  batch {unchanged['batch']}",
              f"ramp           {unchanged['ramp_steps']}",
              f"marks          {unchanged['marks']}",
              f"lambda_corner  {unchanged['lambda_corner']}",
              f"pool           {config['changed']['pool']['before']} -> "
              f"{config['changed']['pool']['after']}   <- 유일한 변경", "```", "",
              "## 학습하지 않은 것", "", "```",
              "L_neg 0 / negative batch 0 / dense zero suppression 없음 /",
              "detached linear presence 없음",
              "EDGE_HARD sampling weight 0   CORNER_LA sampling weight 0", "```", "",
              "## threshold", "",
              f"`PRESENCE_THRESHOLD = {package['PRESENCE_THRESHOLD']}`", "",
              "합성 dev 에서 얻은 값은 initial range/reference artifact 다.",
              "최종 threshold 는 REAL_DEV 에서 정한다 "
              "(`_delivery/README_DELIVERY_20260820.txt` 계약).", "",
              "## 이 checkpoint 로 하면 안 되는 말", "",
              "historical MH_DEV 6,242 가 학습 pool 안에 있다.",
              "**MH_DEV 를 unseen / held-out 이라고 부를 수 없다.**",
              "synthetic 지표로 checkpoint 를 고르지도 않았다 — step 25,000 사전 고정.", "",
              "## 다음", "",
              "REAL IN-HOUSE DEV / TEST 구축 및 annotation.",
              "`real_eval/` 의 프로토콜·스키마·manifest 템플릿과 "
              "`scripts/stage0/real_eval/re_metrics.py` 평가기가 준비돼 있다.", ""]
    (FINAL / "FINAL_TRAIN_RESULT.md").write_text("\n".join(lines))
    print("-> FINAL_TRAIN_RESULT.md")


if __name__ == "__main__":
    main()
