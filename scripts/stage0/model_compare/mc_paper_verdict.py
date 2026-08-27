"""PHASE 10 사전등록 판정 — 기준은 결과 보기 전에 박혀 있다."""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_geom as MG  # noqa: E402
ROOT = "/home/minjae/Documents/github/pallet-pose"
OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
EVAL = os.path.join(ROOT, "challenge/yolo_pose_one_model/evaluation")
NEW = "yolo26n_paper_generic_v1"
GATE = {"STRONG": {"challenge_availability": 0.75, "challenge_R_median": 5.0,
                   "challenge_5cm5": 0.20, "open_5cm5": 0.45},
        "WEAK": {"challenge_5cm5": 0.10, "challenge_R_median": 8.0},
        "FAIL": {"challenge_5cm5": 0.10, "challenge_availability": 0.50}}
REF_5EP_CHALLENGE_5CM5 = 0.057   # y_BROAD40K 5epoch 진단본


def main():
    os.makedirs(EVAL, exist_ok=True)
    MG.main(models=(NEW, "yolo26n_broad40k_5ep", "yolo26n_synth",
                    "yolo26n_ft", "FINAL40K_seed1"))
    r = json.load(open(os.path.join(OUT, "MODEL_COMPARE_GEOM.json")))
    os.replace(os.path.join(OUT, "MODEL_COMPARE_GEOM.json"),
               os.path.join(EVAL, "PAPER_YOLO_REAL_DEV_RESULT.json"))
    b = r["models"][NEW]
    o, c = b["OPEN_56"], b["REAL_CHALLENGE_DEV_105"]
    dump = json.load(open(os.path.join(OUT, f"kps_{NEW}.json")))
    ch = [f for f in dump["frames"] if f["sealed"]]
    avail = sum(1 for f in ch if f["kps"] is not None) / max(len(ch), 1)

    strong = (avail >= GATE["STRONG"]["challenge_availability"]
              and c["R_deg"]["median"] <= GATE["STRONG"]["challenge_R_median"]
              and c["success_5cm5deg"] >= GATE["STRONG"]["challenge_5cm5"]
              and o["success_5cm5deg"] >= GATE["STRONG"]["open_5cm5"])
    fail = (c["success_5cm5deg"] < GATE["FAIL"]["challenge_5cm5"]
            or avail < GATE["FAIL"]["challenge_availability"])
    weak = (not strong and not fail
            and c["success_5cm5deg"] >= GATE["WEAK"]["challenge_5cm5"]
            and c["R_deg"]["median"] <= GATE["WEAK"]["challenge_R_median"]
            and c["success_5cm5deg"] > REF_5EP_CHALLENGE_5CM5)
    verdict = ("STRONG_PASS" if strong else "FAIL" if fail
               else "WEAK_PASS" if weak else "FAIL")
    lines = [f"[PAPER_GENERIC_V1]  target-free BROAD 40K, 60 epoch, point-only", ""]
    for name, blk in r["models"].items():
        oo, cc = blk["OPEN_56"], blk["REAL_CHALLENGE_DEV_105"]
        lines.append(f"  {name:26} open R {oo['R_deg']['median']:>6.2f} "
                     f"5cm5 {oo['success_5cm5deg']:.3f} | "
                     f"challenge R {cc['R_deg']['median']:>6.2f} "
                     f"5cm5 {cc['success_5cm5deg']:.3f}")
    lines += ["",
              f"  challenge native availability  {avail:.3f}  (gate >= 0.75 / fail < 0.50)",
              f"  challenge canonical R median   {c['R_deg']['median']:.2f} deg",
              f"  challenge canonical 5cm5       {c['success_5cm5deg']:.3f}",
              f"  open56 5cm5                    {o['success_5cm5deg']:.3f}",
              f"  참조: 5epoch 진단본 challenge 5cm5 = {REF_5EP_CHALLENGE_5CM5}",
              "", f"  VERDICT = {verdict}"]
    print("\n".join(lines))
    json.dump({"verdict": verdict, "gate": GATE,
               "inputs": {"challenge_availability": round(avail, 4),
                          "challenge_R_median": c["R_deg"]["median"],
                          "challenge_5cm5": c["success_5cm5deg"],
                          "open_5cm5": o["success_5cm5deg"]},
               "reference_5ep": REF_5EP_CHALLENGE_5CM5},
              open(os.path.join(EVAL, "PAPER_YOLO_VERDICT.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
