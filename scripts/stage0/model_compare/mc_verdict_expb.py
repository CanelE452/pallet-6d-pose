"""실험 B 판정 — 기준을 결과 보기 전에 코드에 박는다."""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_geom as MG  # noqa: E402
ROOT = "/home/minjae/Documents/github/pallet-pose"
OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
# 사전등록 기준: dev105 5cm5 를 FINAL40K(0.009) 와 y_synth(0.248) 중 어디에 붙는가
LOW, HIGH = 0.05, 0.15


def main():
    MG.main(models=("yolo26n_broad40k_5ep", "FINAL40K_seed1",
                    "yolo26n_synth", "yolo26n_ft"))
    r = json.load(open(os.path.join(OUT, "MODEL_COMPARE_GEOM.json")))
    os.replace(os.path.join(OUT, "MODEL_COMPARE_GEOM.json"),
               os.path.join(OUT, "EXP_B_GEOM.json"))
    b = r["models"]["yolo26n_broad40k_5ep"]
    f = r["models"]["FINAL40K_seed1"]
    dev = b["REAL_CHALLENGE_DEV_105"]["success_5cm5deg"]
    lines = ["[실험 B] YOLO26n · BROAD 40K · 5 epoch (FINAL40K 와 동일 데이터·노출)", ""]
    for name, blk in (("yolo26n_broad40k_5ep", b), ("FINAL40K_seed1", f)):
        o = blk["OPEN_56"]; d = blk["REAL_CHALLENGE_DEV_105"]
        lines.append(f"  {name:24} open 5cm5 {o['success_5cm5deg']:.3f} "
                     f"corner {o['corner_px']['median']:6.2f}px | "
                     f"dev105 5cm5 {d['success_5cm5deg']:.3f} "
                     f"corner {d['corner_px']['median']:7.2f}px")
    lines.append("")
    lines.append(f"  사전등록 기준: dev105 5cm5 <= {LOW} -> 데이터 탓 / "
                 f">= {HIGH} -> 아키텍처·해상도 탓")
    if dev <= LOW:
        verdict = ("DATA_IS_THE_CAUSE — 같은 데이터를 주면 YOLO 도 무너진다. "
                   "아키텍처·해상도는 무죄. corner 해상도 개조는 근거 없음")
    elif dev >= HIGH:
        verdict = ("ARCHITECTURE_IS_THE_CAUSE — 같은 데이터로도 YOLO 는 버틴다. "
                   "corner 해상도/표현이 병목")
    else:
        verdict = f"INCONCLUSIVE — dev105 5cm5 {dev:.3f} 가 두 기준 사이"
    lines.append(f"  판정: {verdict}")
    print("\n".join(lines))
    json.dump({"verdict": verdict, "dev105_5cm5": dev,
               "gate": {"low": LOW, "high": HIGH}},
              open(os.path.join(OUT, "EXP_B_VERDICT.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
